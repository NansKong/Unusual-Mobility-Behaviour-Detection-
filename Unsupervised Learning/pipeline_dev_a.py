"""
pipeline_dev_a.py — Dev A End-to-End Runner (Stages 0–3 + Branch A)

Runs the full Dev A pipeline on a single video and produces two hand-off files
for Dev B:
  1. outputs/tracks/<video_stem>_tracks.csv    — cleaned track history
  2. outputs/features/<video_stem>_branch_a.csv — 11-dim trajectory features

Usage:
    python pipeline_dev_a.py --video data/ucf_crime/Fighting001.mp4

    # With explicit options:
    python pipeline_dev_a.py \
        --video    data/ucf_crime/Fighting001.mp4 \
        --device   cuda \
        --out_dir  outputs/ \
        --min_len  20 \
        --max_gap  5

The hand-off CSVs conform to the interface spec in PRD §16:
  tracks CSV:    track_id, frame_idx, cx, cy, x1, y1, x2, y2
  features CSV:  track_id, mean_speed, speed_std, ..., avg_turn_angle  (11 cols)
"""

import argparse
import os
from pathlib import Path

from step0_sampling import sample_frames
from step1_detect    import Detector
from step2_track     import Tracker
from step3_clean     import clean_tracks, save_tracks_csv
from step4a_traj     import TrajectoryFeatureExtractor, save_features_csv, validate_features


def run_dev_a_pipeline(
    video_path:  str,
    out_dir:     str  = "outputs",
    device:      str  = "",
    min_len:     int  = 20,
    max_gap:     int  = 5,
) -> tuple[str, str]:
    """
    Run the complete Dev A pipeline and save hand-off files.

    Args:
        video_path: Path to .mp4 or .avi input video.
        out_dir:    Root output directory.
        device:     YOLO/torch device string ('cuda', 'cpu', or '' for auto).
        min_len:    Minimum track length in frames (step3).
        max_gap:    Maximum interpolation gap in frames (step3).

    Returns:
        (tracks_csv_path, features_csv_path) — paths to the two hand-off files.
    """
    stem = Path(video_path).stem
    tracks_csv_path   = os.path.join(out_dir, "tracks",   f"{stem}_tracks.csv")
    features_csv_path = os.path.join(out_dir, "features", f"{stem}_branch_a.csv")

    print(f"\n{'='*60}")
    print(f"Dev A Pipeline — {stem}")
    print(f"{'='*60}\n")

    # ── Stage 0: Frame sampling ──────────────────────────────────────────────
    print("── Stage 0: Frame sampling ──────────────────────")
    frames = sample_frames(video_path)
    print(f"  → {len(frames)} frames @ 5 FPS, width=640\n")

    # ── Stage 1: Person detection ────────────────────────────────────────────
    print("── Stage 1: YOLOv8n person detection ────────────")
    detector = Detector(device=device)
    all_detections = detector.detect_frames(frames, verbose=True)
    n_with_persons = sum(1 for d in all_detections if d)
    print(f"  → {n_with_persons}/{len(frames)} frames had detections\n")

    # ── Stage 2: DeepSORT tracking ───────────────────────────────────────────
    print("── Stage 2: DeepSORT tracking ───────────────────")
    tracker    = Tracker()
    raw_tracks = tracker.track(frames, all_detections, verbose=True)
    print(f"  → {len(raw_tracks)} raw tracks\n")

    # ── Stage 3: Trajectory cleaning ─────────────────────────────────────────
    print("── Stage 3: Trajectory cleaning ─────────────────")
    cleaned = clean_tracks(raw_tracks, min_len=min_len, max_gap=max_gap, verbose=True)
    save_tracks_csv(cleaned, tracks_csv_path)
    print(f"  → {len(cleaned)} tracks kept\n")

    # ── Stage 4A: Branch A features ──────────────────────────────────────────
    print("── Stage 4A: Branch A trajectory features ───────")
    extractor   = TrajectoryFeatureExtractor()
    features_df = extractor.extract_all(cleaned, verbose=True)
    validate_features(features_df)         # AC-4 check: zero NaN values
    save_features_csv(features_df, features_csv_path)

    print(f"\n{'='*60}")
    print(f"Dev A Pipeline COMPLETE")
    print(f"  Tracks CSV   → {tracks_csv_path}")
    print(f"  Features CSV → {features_csv_path}")
    print(f"{'='*60}\n")

    return tracks_csv_path, features_csv_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dev A end-to-end pipeline (Stages 0–3 + Branch A)"
    )
    parser.add_argument("--video",   required=True,  help="Path to input video (.mp4/.avi)")
    parser.add_argument("--out_dir", default="outputs", help="Root output directory")
    parser.add_argument("--device",  default="",     help="'cuda', 'cpu', or '' for auto")
    parser.add_argument("--min_len", type=int, default=20, help="Min track length (frames)")
    parser.add_argument("--max_gap", type=int, default=5,  help="Max interpolation gap (frames)")
    args = parser.parse_args()

    run_dev_a_pipeline(
        video_path = args.video,
        out_dir    = args.out_dir,
        device     = args.device,
        min_len    = args.min_len,
        max_gap    = args.max_gap,
    )
