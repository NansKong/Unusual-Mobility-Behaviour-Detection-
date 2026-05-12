"""
pipeline.py — Master End-to-End Pipeline Runner

Executes the entire Unsupervised Mobility Behaviour Detection pipeline
(Stages 0 through 7) in a single command.

Usage:
    python pipeline.py --video data/ucf_crime/videos/Test/Fighting/Fighting002_x264.mp4
    python pipeline.py --video data/ucf_crime/videos/Test/Fighting/Fighting002_x264.mp4 --device cpu
    python pipeline.py --video data/ucf_crime/videos/Test/Fighting/Fighting002_x264.mp4 --skip_to 4b
"""

import argparse
import os
import sys
from pathlib import Path

# ── Ensure src/ is on the path so all step modules can be imported ────────────
sys.path.insert(0, os.path.dirname(__file__))


def run_full_pipeline(
    video_path: str,
    out_dir: str = "outputs",
    device: str = "cuda",
    min_len: int = 20,
    max_gap: int = 5,
    skip_to: str = "",
    annotation_path: str = "",
):
    """
    Run the complete pipeline (Stages 0–7) on a single video.

    Args:
        video_path:      Path to the input .mp4/.avi video file.
        out_dir:         Root output directory for all artifacts.
        device:          Compute device ('cuda', 'cpu', or '' for auto).
        min_len:         Minimum track length in frames (Stage 3).
        max_gap:         Maximum interpolation gap in frames (Stage 3).
        skip_to:         Skip stages up to this point. Options:
                         '' (run all), '4b', '5', '6', '7'.
                         Requires outputs from prior stages to exist.
        annotation_path: Path to annotation file for AUC evaluation (Stage 7).
    """
    video_path = str(Path(video_path).resolve())
    stem = Path(video_path).stem

    # ── Output paths ──────────────────────────────────────────────────────────
    tracks_csv   = os.path.join(out_dir, "tracks",   f"{stem}_tracks.csv")
    branch_a_csv = os.path.join(out_dir, "features", f"{stem}_branch_a.csv")
    branch_b_csv = os.path.join(out_dir, "features", f"{stem}_branch_b.csv")
    fused_csv    = os.path.join(out_dir, "features", f"{stem}_fused.csv")
    scores_csv   = os.path.join(out_dir, "scores",   f"{stem}_scores.csv")
    out_video    = os.path.join(out_dir, "videos",   f"{stem}_annotated.mp4")
    out_plot     = os.path.join(out_dir, "plots",    f"{stem}_score_dist.png")

    if not annotation_path:
        annotation_path = os.path.join("data", "ucf_crime", "annotations",
                                       "Temporal_Anomaly_Annotation.txt")

    print(f"\n{'#'*70}")
    print(f"  Unsupervised Mobility Pipeline — {stem}")
    print(f"  Video  : {video_path}")
    print(f"  Device : {device}")
    print(f"  Output : {out_dir}")
    if skip_to:
        print(f"  Skip to: Stage {skip_to}")
    print(f"{'#'*70}\n")

    stages_to_run = _get_stages(skip_to)

    # ======================================================================
    # STAGES 0–4A: Detection, Tracking, Cleaning, Trajectory Features
    # ======================================================================
    if "dev_a" in stages_to_run:
        print(f"\n{'='*60}")
        print("  Stages 0–4A: Sampling → Detection → Tracking → Features")
        print(f"{'='*60}\n")

        from step0_sampling import sample_frames
        from step1_detect   import Detector
        from step2_track    import Tracker
        from step3_clean    import clean_tracks, save_tracks_csv
        from step4a_traj    import TrajectoryFeatureExtractor, save_features_csv, validate_features

        # Stage 0
        print("── Stage 0: Frame Sampling ──────────────────────")
        frames = sample_frames(video_path)
        print(f"  → {len(frames)} frames sampled\n")

        # Stage 1
        print("── Stage 1: YOLOv8n Person Detection ───────────")
        detector = Detector(device=device)
        all_detections = detector.detect_frames(frames, verbose=True)
        n_with = sum(1 for d in all_detections if d)
        print(f"  → {n_with}/{len(frames)} frames had detections\n")

        # Stage 2
        print("── Stage 2: DeepSORT Tracking ──────────────────")
        tracker    = Tracker()
        raw_tracks = tracker.track(frames, all_detections, verbose=True)
        print(f"  → {len(raw_tracks)} raw tracks\n")

        # Stage 3
        print("── Stage 3: Trajectory Cleaning ────────────────")
        cleaned = clean_tracks(raw_tracks, min_len=min_len, max_gap=max_gap, verbose=True)
        save_tracks_csv(cleaned, tracks_csv)
        print(f"  → {len(cleaned)} tracks kept\n")

        if len(cleaned) == 0:
            print("Pipeline stopped: No tracks survived cleaning.")
            return

        # Stage 4A
        print("── Stage 4A: Branch A Trajectory Features ──────")
        extractor   = TrajectoryFeatureExtractor()
        features_df = extractor.extract_all(cleaned, verbose=True)
        validate_features(features_df)
        save_features_csv(features_df, branch_a_csv)
        print()

    # ======================================================================
    # STAGE 4B: VideoMAE Embeddings (Branch B)
    # ======================================================================
    if "4b" in stages_to_run:
        print(f"\n{'='*60}")
        print("  Stage 4B: VideoMAE Embeddings (Branch B)")
        print(f"{'='*60}\n")

        from step4b_vmae import get_videomae_embeddings

        _check_file(tracks_csv, "tracks CSV (run stages 0–4A first)")
        df_b = get_videomae_embeddings(video_path, tracks_csv, branch_b_csv, device)

        if len(df_b) == 0:
            print("Pipeline stopped: No VideoMAE embeddings generated.")
            return

    # ======================================================================
    # STAGE 5: Feature Fusion
    # ======================================================================
    if "5" in stages_to_run:
        print(f"\n{'='*60}")
        print("  Stage 5: Feature Fusion (Branch A + Branch B)")
        print(f"{'='*60}\n")

        from step5_fuse import fuse_features

        _check_file(branch_a_csv, "Branch A features (run stages 0–4A first)")
        _check_file(branch_b_csv, "Branch B features (run stage 4B first)")
        df_fused = fuse_features(branch_a_csv, branch_b_csv, fused_csv)

        if len(df_fused) == 0:
            print("Pipeline stopped: Feature fusion produced no output.")
            return

    # ======================================================================
    # STAGE 6: HDBSCAN Clustering & Anomaly Scoring
    # ======================================================================
    if "6" in stages_to_run:
        print(f"\n{'='*60}")
        print("  Stage 6: HDBSCAN Clustering & Anomaly Scoring")
        print(f"{'='*60}\n")

        from step6_cluster import cluster_and_score

        _check_file(fused_csv, "fused features (run stage 5 first)")
        df_scores = cluster_and_score(fused_csv, scores_csv)

        if len(df_scores) == 0:
            print("Pipeline stopped: Clustering produced no output.")
            return

    # ======================================================================
    # STAGE 7: Evaluation & Visualisation
    # ======================================================================
    if "7" in stages_to_run:
        print(f"\n{'='*60}")
        print("  Stage 7: Evaluation & Visualisation")
        print(f"{'='*60}\n")

        from step7_eval import evaluate_and_visualize

        _check_file(tracks_csv,  "tracks CSV")
        _check_file(scores_csv,  "scores CSV (run stage 6 first)")
        evaluate_and_visualize(
            video_path, tracks_csv, scores_csv,
            annotation_path, out_video, out_plot
        )

    # ======================================================================
    print(f"\n{'#'*70}")
    print("  Pipeline Completed Successfully!")
    print(f"  Tracks     : {tracks_csv}")
    print(f"  Branch A   : {branch_a_csv}")
    print(f"  Branch B   : {branch_b_csv}")
    print(f"  Fused      : {fused_csv}")
    print(f"  Scores     : {scores_csv}")
    print(f"  Video      : {out_video}")
    print(f"  Plot       : {out_plot}")
    print(f"{'#'*70}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_stages(skip_to: str) -> list[str]:
    """Return list of stage keys to run based on skip_to."""
    all_stages = ["dev_a", "4b", "5", "6", "7"]
    if not skip_to:
        return all_stages

    skip_map = {
        "4b": 1,   # start from stage 4b
        "5":  2,
        "6":  3,
        "7":  4,
    }
    idx = skip_map.get(skip_to.lower())
    if idx is None:
        print(f"WARNING: Unknown skip_to='{skip_to}'. Running all stages.")
        return all_stages
    return all_stages[idx:]


def _check_file(path: str, description: str):
    """Verify a required intermediate file exists."""
    if not os.path.isfile(path):
        print(f"\nERROR: Required file not found: {path}")
        print(f"       ({description})")
        print("       Run the earlier stages first, or remove --skip_to.")
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(
        description="Full End-to-End Unsupervised Mobility Pipeline (Stages 0–7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything from scratch:
  python pipeline.py --video data/ucf_crime/videos/Test/Fighting/Fighting002_x264.mp4

  # Resume from Stage 4B (if stages 0-4A outputs already exist):
  python pipeline.py --video data/ucf_crime/videos/Test/Fighting/Fighting002_x264.mp4 --skip_to 4b

  # Resume from Stage 5 (if branch_a and branch_b CSVs already exist):
  python pipeline.py --video data/ucf_crime/videos/Test/Fighting/Fighting002_x264.mp4 --skip_to 5
""",
    )
    parser.add_argument("--video",      required=True,  help="Path to input video (.mp4/.avi)")
    parser.add_argument("--out_dir",    default="outputs", help="Root output directory (default: outputs)")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device: 'cuda' or 'cpu' (default: auto)")
    parser.add_argument("--min_len",    type=int, default=20, help="Min track length in frames (default: 20)")
    parser.add_argument("--max_gap",    type=int, default=5,  help="Max interpolation gap in frames (default: 5)")
    parser.add_argument("--skip_to",    default="", choices=["", "4b", "5", "6", "7"],
                        help="Skip to this stage (requires prior outputs to exist)")
    parser.add_argument("--annotation", default="", help="Path to annotation file for AUC eval")
    args = parser.parse_args()

    run_full_pipeline(
        video_path=args.video,
        out_dir=args.out_dir,
        device=args.device,
        min_len=args.min_len,
        max_gap=args.max_gap,
        skip_to=args.skip_to,
        annotation_path=args.annotation,
    )
