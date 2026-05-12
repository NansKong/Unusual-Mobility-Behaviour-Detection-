"""
Stage 7 — Evaluation and Visualisation
Computes frame-level AUC if annotations exist, generates annotated output video,
and plots anomaly score distributions.
"""

import os
import argparse
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

SAMPLE_FPS = 5   # must match step0_sampling.py TARGET_FPS


def load_annotations(annotation_path, video_name):
    """
    Parses Temporal_Anomaly_Annotation.txt for the specific video.
    Returns a list of (start_frame, end_frame) indicating anomalous segments.
    """
    if not os.path.exists(annotation_path):
        return []

    anomalies = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Format depends on UCF-Crime annotation file, commonly: video_name start end
            if len(parts) >= 3 and video_name in parts[0]:
                anomalies.append((int(parts[1]), int(parts[2])))
    return anomalies


def evaluate_and_visualize(video_path, tracks_csv, scores_csv, annotation_path, out_video_path, out_plot_path):
    video_name = Path(video_path).stem
    print(f"[step7] Processing video: {video_name}")

    # Load data
    tracks_df = pd.read_csv(tracks_csv)
    scores_df = pd.read_csv(scores_csv)

    # Merge scores into tracks
    df = pd.merge(tracks_df, scores_df, on="track_id", how="left")
    # Fill NaN scores (for tracks that might have been filtered out) with 0
    df["anomaly_score"] = df["anomaly_score"].fillna(0.0)
    if "cluster" not in df.columns:
        df["cluster"] = -1
    df["cluster"] = df["cluster"].fillna(-1)

    # Video setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0:
        native_fps = 30.0

    # Step interval used by step0_sampling
    frame_interval = max(1, round(native_fps / SAMPLE_FPS))

    print(f"[step7] Video: {width}x{height}, {native_fps:.1f} FPS, "
          f"{total_frames} frames, sample_interval={frame_interval}")

    Path(out_video_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_video_path, fourcc, native_fps, (width, height))

    # Build a lookup: sampled_frame_idx → list of (track_id, x1, y1, x2, y2, score, cluster)
    # Also convert sampled_frame_idx to native frame number for matching
    frame_track_data = {}
    for _, row in df.iterrows():
        sampled_idx = int(row['frame_idx'])
        native_frame = sampled_idx * frame_interval

        if native_frame not in frame_track_data:
            frame_track_data[native_frame] = []
        frame_track_data[native_frame].append({
            'track_id': int(row['track_id']),
            'x1': int(row['x1']),
            'y1': int(row['y1']),
            'x2': int(row['x2']),
            'y2': int(row['y2']),
            'anomaly_score': float(row['anomaly_score']),
            'cluster': int(row['cluster']),
        })

    # Pre-compute frame-level max anomaly score for ALL native frames
    frame_scores = np.zeros(total_frames)

    # Propagate scores to nearby native frames (within the sampling interval)
    # so annotations drawn continuously instead of only every Nth frame
    for native_frame, track_list in frame_track_data.items():
        max_score = max(t['anomaly_score'] for t in track_list)
        # Spread to surrounding frames within the interval
        for offset in range(frame_interval):
            idx = native_frame + offset
            if idx < total_frames:
                frame_scores[idx] = max(frame_scores[idx], max_score)

    # Frame loop — read every native frame for smooth output video
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if this native frame has track data (or is near one that does)
        # Draw bboxes on exact sampled frames
        if current_frame in frame_track_data:
            for track in frame_track_data[current_frame]:
                score = track['anomaly_score']
                x1, y1, x2, y2 = track['x1'], track['y1'], track['x2'], track['y2']

                # Clamp to frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                # Color: green (safe) → red (anomaly)
                color = (0, int(255 * (1 - score)), int(255 * score))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track['track_id']} S:{score:.2f}",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add frame-level score text
        score_here = frame_scores[current_frame] if current_frame < len(frame_scores) else 0.0
        score_color = (0, 0, 255) if score_here > 0.75 else (0, 255, 0)
        cv2.putText(frame, f"Score: {score_here:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)

        out_vid.write(frame)
        current_frame += 1

    cap.release()
    out_vid.release()
    print(f"[step7] Annotated video saved -> {out_video_path}")

    # AUC Evaluation
    anomalies = load_annotations(annotation_path, video_name)
    if anomalies:
        y_true = np.zeros(total_frames)
        for start, end in anomalies:
            y_true[start:end+1] = 1

        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            auc = roc_auc_score(y_true, frame_scores)
            print(f"[step7] Frame-level AUC: {auc:.4f}")
        else:
            print("[step7] Cannot compute AUC: all frames have same label.")
    else:
        print("[step7] No annotations found. Skipping AUC calculation.")

    # Plotting
    Path(out_plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Score distribution
    axes[0].hist(scores_df["anomaly_score"], bins=20, color='steelblue', alpha=0.8, edgecolor='white')
    axes[0].set_title(f"Anomaly Score Distribution — {video_name}")
    axes[0].set_xlabel("Anomaly Score")
    axes[0].set_ylabel("Track Count")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Frame-level score timeline
    axes[1].plot(frame_scores, color='crimson', linewidth=0.5, alpha=0.8)
    axes[1].set_title(f"Frame-level Anomaly Score — {video_name}")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Max Anomaly Score")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    # Overlay annotation regions if available
    if anomalies:
        for start, end in anomalies:
            axes[1].axvspan(start, end, alpha=0.2, color='red', label='Anomaly GT')

    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=150)
    plt.close()
    print(f"[step7] Score distribution plot saved -> {out_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks_csv", required=True)
    parser.add_argument("--scores_csv", required=True)
    parser.add_argument("--annotation_path", default="data/ucf_crime/annotations/Temporal_Anomaly_Annotation.txt")
    parser.add_argument("--out_video", default="outputs/videos/annotated_output.mp4")
    parser.add_argument("--out_plot", default="outputs/plots/score_dist.png")
    args = parser.parse_args()

    evaluate_and_visualize(
        args.video, args.tracks_csv, args.scores_csv,
        args.annotation_path, args.out_video, args.out_plot
    )
