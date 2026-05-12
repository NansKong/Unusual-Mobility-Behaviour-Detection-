"""
Stage 3 — Trajectory Cleaning
Filters out short tracks and interpolates small positional gaps caused by occlusion.

Rules (PRD §7.3):
  - Discard tracks shorter than MIN_LEN frames
  - Interpolate gaps up to MAX_GAP frames using linear interpolation (scipy)
  - Output cleaned track history in same format as step2

Usage:
    from step3_clean import clean_tracks
    cleaned = clean_tracks(track_history)

CLI:
    python step3_clean.py --video data/ucf_crime/Fighting001.mp4 --out_csv outputs/tracks.csv
"""

import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
from scipy.interpolate import interp1d

from step0_sampling import sample_frames
from step1_detect import Detector
from step2_track import Tracker, TrackHistory, print_track_summary

# ── Constants (PRD §7.3) ──────────────────────────────────────────────────────

MIN_LEN = 20   # minimum track length in frames; shorter tracks are discarded
MAX_GAP = 5    # maximum gap (frames) to interpolate over

# Type alias — same structure as step2 output
TrackPoint = tuple[float, float, int, list[float]]


def clean_tracks(
    track_history: TrackHistory,
    min_len: int = MIN_LEN,
    max_gap: int = MAX_GAP,
    verbose: bool = True,
) -> TrackHistory:
    """
    Filter and interpolate raw track histories from DeepSORT.

    Steps per track:
      1. Sort observations by frame index.
      2. Identify gaps between consecutive observed frames.
      3. Interpolate cx, cy (and bbox) linearly across gaps <= max_gap.
      4. Discard the track if final length < min_len.

    Args:
        track_history: Raw track dict from step2 — {track_id: [(cx, cy, frame_idx, bbox), ...]}
        min_len:       Minimum number of frames required (default 20).
        max_gap:       Maximum gap (frames) to interpolate (default 5).
        verbose:       Print cleaning statistics.

    Returns:
        Cleaned track_history in the same format.
    """
    cleaned: TrackHistory = {}
    n_discarded = 0
    n_interpolated_total = 0

    for track_id, points in track_history.items():
        # Sort by frame index
        points = sorted(points, key=lambda p: p[2])

        # Extract arrays for interpolation
        frame_indices = np.array([p[2] for p in points], dtype=float)
        cx_arr       = np.array([p[0] for p in points], dtype=float)
        cy_arr       = np.array([p[1] for p in points], dtype=float)
        # bbox: x1, y1, x2, y2
        bboxes       = np.array([p[3] for p in points], dtype=float)  # (N, 4)

        # Detect gaps > max_gap; only interpolate within gaps <= max_gap
        dense_frame_indices, n_interp = _interpolate_gaps(
            frame_indices, cx_arr, cy_arr, bboxes, max_gap
        )
        n_interpolated_total += n_interp

        # Rebuild points list after interpolation
        new_cx, new_cy, new_bboxes = dense_frame_indices[1], dense_frame_indices[2], dense_frame_indices[3]
        new_frames = dense_frame_indices[0]

        rebuilt = [
            (float(new_cx[i]), float(new_cy[i]), int(new_frames[i]), new_bboxes[i].tolist())
            for i in range(len(new_frames))
        ]

        # Length filter
        if len(rebuilt) < min_len:
            n_discarded += 1
            continue

        cleaned[track_id] = rebuilt

    if verbose:
        total_raw = len(track_history)
        total_kept = len(cleaned)
        print(f"[step3] Cleaning complete.")
        print(f"  Raw tracks:          {total_raw}")
        print(f"  Discarded (< {min_len} frames): {n_discarded}")
        print(f"  Kept:                {total_kept}")
        print(f"  Frames interpolated: {n_interpolated_total}")

    return cleaned


def _interpolate_gaps(
    frame_indices: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    bboxes: np.ndarray,
    max_gap: int,
) -> tuple[tuple, int]:
    """
    Interpolate positional gaps of <= max_gap frames using linear interpolation.
    Gaps larger than max_gap are left as-is (the track will still have discontinuities,
    but we keep both segments — the length filter handles very short combined tracks).

    Args:
        frame_indices: Sorted array of observed frame indices.
        cx, cy:        Centre coords per observed frame.
        bboxes:        (N, 4) array of bbox coords per observed frame.
        max_gap:       Maximum gap to fill.

    Returns:
        Tuple of (new_frames, new_cx, new_cy, new_bboxes), n_interpolated_frames
    """
    if len(frame_indices) < 2:
        return (frame_indices, cx, cy, bboxes), 0

    all_frames: list[float] = []
    all_cx:     list[float] = []
    all_cy:     list[float] = []
    all_bboxes: list[np.ndarray] = []
    n_interp = 0

    for i in range(len(frame_indices)):
        all_frames.append(frame_indices[i])
        all_cx.append(cx[i])
        all_cy.append(cy[i])
        all_bboxes.append(bboxes[i])

        if i < len(frame_indices) - 1:
            gap = int(frame_indices[i + 1]) - int(frame_indices[i]) - 1
            if 0 < gap <= max_gap:
                # Interpolate missing frames
                t0, t1 = frame_indices[i], frame_indices[i + 1]
                for missing_t in range(int(t0) + 1, int(t1)):
                    alpha = (missing_t - t0) / (t1 - t0)
                    interp_cx = cx[i] + alpha * (cx[i + 1] - cx[i])
                    interp_cy = cy[i] + alpha * (cy[i + 1] - cy[i])
                    interp_bbox = bboxes[i] + alpha * (bboxes[i + 1] - bboxes[i])

                    all_frames.append(float(missing_t))
                    all_cx.append(interp_cx)
                    all_cy.append(interp_cy)
                    all_bboxes.append(interp_bbox)
                    n_interp += 1

    return (
        np.array(all_frames),
        np.array(all_cx),
        np.array(all_cy),
        np.array(all_bboxes),
    ), n_interp


def save_tracks_csv(track_history: TrackHistory, out_path: str) -> None:
    """
    Save cleaned track history to a CSV for Dev B hand-off.

    Columns: track_id, frame_idx, cx, cy, x1, y1, x2, y2
    (bbox_crops_path will be added by Dev B's step4B)

    Args:
        track_history: Cleaned track dict.
        out_path:      Output CSV file path.
    """
    rows = []
    for track_id, points in track_history.items():
        for cx, cy, frame_idx, bbox in points:
            rows.append({
                "track_id":  track_id,
                "frame_idx": frame_idx,
                "cx":        round(cx, 2),
                "cy":        round(cy, 2),
                "x1":        round(bbox[0], 2),
                "y1":        round(bbox[1], 2),
                "x2":        round(bbox[2], 2),
                "y2":        round(bbox[3], 2),
            })

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[step3] Track CSV saved → {out_path}  ({len(df)} rows, {df['track_id'].nunique()} tracks)")


def load_tracks_csv(csv_path: str) -> TrackHistory:
    """
    Load a cleaned track CSV back into the TrackHistory dict format.
    Useful for Dev B to consume Dev A's hand-off file.

    Args:
        csv_path: Path to CSV produced by save_tracks_csv().

    Returns:
        TrackHistory dict.
    """
    df = pd.read_csv(csv_path)
    track_history: TrackHistory = {}
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("frame_idx")
        points = [
            (
                float(row["cx"]),
                float(row["cy"]),
                int(row["frame_idx"]),
                [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
            )
            for _, row in group.iterrows()
        ]
        track_history[int(track_id)] = points
    return track_history


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Trajectory cleaning")
    parser.add_argument("--video",    required=True,  help="Path to input video")
    parser.add_argument("--out_csv",  default=None,   help="Save cleaned tracks to CSV (for Dev B hand-off)")
    parser.add_argument("--min_len",  type=int, default=MIN_LEN)
    parser.add_argument("--max_gap",  type=int, default=MAX_GAP)
    parser.add_argument("--device",   default="",     help="YOLO device: 'cuda', 'cpu', or ''")
    args = parser.parse_args()

    print("[step3] Sampling frames ...")
    frames = sample_frames(args.video)

    print("[step3] Detecting persons ...")
    detector = Detector(device=args.device)
    all_detections = detector.detect_frames(frames, verbose=False)

    print("[step3] Tracking ...")
    tracker = Tracker()
    raw_tracks = tracker.track(frames, all_detections, verbose=False)
    print_track_summary(raw_tracks)

    cleaned = clean_tracks(raw_tracks, min_len=args.min_len, max_gap=args.max_gap)

    if args.out_csv:
        save_tracks_csv(cleaned, args.out_csv)
    else:
        print_track_summary(cleaned)
