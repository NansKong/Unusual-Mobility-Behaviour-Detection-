"""
Stage 2 — DeepSORT Tracking
Maintains stable track IDs across frames by feeding YOLOv8n detections into DeepSORT.

Output: track_history dict  {track_id: [(cx, cy, frame_idx, [x1,y1,x2,y2]), ...]}
  - cx, cy = centre of bounding box (float, pixels)
  - frame_idx = sampled frame index (not original video frame number)
  - bbox = [x1, y1, x2, y2] absolute coords

Usage:
    from step2_track import Tracker
    tracker = Tracker()
    track_history = tracker.track(frames, all_detections)

CLI:
    python step2_track.py --video data/ucf_crime/Fighting001.mp4
"""

import numpy as np
import argparse
from collections import defaultdict
from pathlib import Path

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None  # type: ignore

from step0_sampling import sample_frames
from step1_detect import Detector, detections_to_deepsort_input

# ── Constants (from PRD §7.2) ─────────────────────────────────────────────────

MAX_AGE   = 30    # frames — keep lost track alive for up to 30 frames
N_INIT    = 3     # frames — confirm track after 3 consistent detections
NN_BUDGET = 100   # appearance descriptor memory size

# Type alias for readability
TrackPoint  = tuple[float, float, int, list[float]]           # (cx, cy, frame_idx, bbox)
TrackHistory = dict[int, list[TrackPoint]]                    # track_id → [TrackPoint, ...]


class Tracker:
    """
    Wraps deep-sort-realtime for person tracking.

    Args:
        max_age:   Frames to keep a lost track alive (default 30).
        n_init:    Detections needed before a track is confirmed (default 3).
        nn_budget: Appearance feature memory budget (default 100).
    """

    def __init__(
        self,
        max_age: int = MAX_AGE,
        n_init: int = N_INIT,
        nn_budget: int = NN_BUDGET,
    ):
        if DeepSort is None:
            raise ImportError(
                "deep-sort-realtime is not installed.\n"
                "Run: pip install deep-sort-realtime==1.3.2\n"
                "Do NOT install the original 'deep_sort' package — it has TF1/TF2 conflicts."
            )

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
        )
        print(f"[step2] DeepSORT initialised (max_age={max_age}, n_init={n_init}, nn_budget={nn_budget})")

    def track(
        self,
        frames: list[np.ndarray],
        all_detections: list[list[list[float]]],
        verbose: bool = True,
    ) -> TrackHistory:
        """
        Run DeepSORT over all frames and accumulate track histories.

        Args:
            frames:          BGR frames from step0 (used by DeepSORT for appearance features).
            all_detections:  Per-frame detections from step1 — list of [x1,y1,x2,y2,conf].

        Returns:
            track_history dict: {track_id: [(cx, cy, frame_idx, [x1,y1,x2,y2]), ...]}
        """
        assert len(frames) == len(all_detections), (
            f"Frame count ({len(frames)}) != detection count ({len(all_detections)}). "
            "Ensure step0 and step1 were run on the same video."
        )

        track_history: TrackHistory = defaultdict(list)
        active_ids_seen = set()

        for frame_idx, (frame, dets) in enumerate(zip(frames, all_detections)):
            # Convert to deep-sort-realtime format: ([l,t,w,h], conf, class_id)
            ds_input = detections_to_deepsort_input(dets)

            # Update tracker — DeepSORT returns confirmed Track objects
            tracks = self.tracker.update_tracks(ds_input, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    # Skip tentative tracks (fewer than n_init detections)
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = track.to_ltrb()          # left-top-right-bottom
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bbox = [x1, y1, x2, y2]

                track_history[track_id].append((cx, cy, frame_idx, bbox))
                active_ids_seen.add(track_id)

            if verbose and (frame_idx % 50 == 0 or frame_idx == len(frames) - 1):
                n_confirmed = sum(1 for t in tracks if t.is_confirmed())
                print(f"[step2] Frame {frame_idx+1}/{len(frames)} — {n_confirmed} confirmed track(s)")

        track_history = dict(track_history)

        if verbose:
            lengths = [len(v) for v in track_history.values()]
            print(f"[step2] Tracking complete. "
                  f"{len(track_history)} unique tracks found. "
                  f"Avg length: {np.mean(lengths):.1f} frames, "
                  f"Max length: {max(lengths) if lengths else 0} frames.")

        return track_history


def print_track_summary(track_history: TrackHistory) -> None:
    """Print a human-readable summary of all tracks."""
    print(f"\n{'─'*55}")
    print(f"{'Track ID':>10} | {'Frames':>8} | {'Start':>6} | {'End':>6}")
    print(f"{'─'*55}")
    for tid, points in sorted(track_history.items()):
        frame_indices = [p[2] for p in points]
        print(f"{tid:>10} | {len(points):>8} | {min(frame_indices):>6} | {max(frame_indices):>6}")
    print(f"{'─'*55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: DeepSORT tracking")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--max_age", type=int, default=MAX_AGE)
    parser.add_argument("--n_init", type=int, default=N_INIT)
    parser.add_argument("--device", default="", help="YOLO device: 'cuda', 'cpu', or ''")
    args = parser.parse_args()

    print("[step2] Sampling frames ...")
    frames = sample_frames(args.video)

    print("[step2] Running detection ...")
    detector = Detector(device=args.device)
    all_detections = detector.detect_frames(frames, verbose=False)

    tracker = Tracker(max_age=args.max_age, n_init=args.n_init)
    track_history = tracker.track(frames, all_detections)

    print_track_summary(track_history)
