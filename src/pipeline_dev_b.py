"""
Dev B End-to-End Pipeline
Executes Stages 4B -> 5 -> 6 -> 7 sequentially, taking Dev A's CSVs as input.
"""

import os
import argparse
from pathlib import Path

from step4b_vmae import get_videomae_embeddings
from step5_fuse import fuse_features
from step6_cluster import cluster_and_score
from step7_eval import evaluate_and_visualize

def run_pipeline(video_path, tracks_csv, branch_a_csv, out_dir, device="cuda"):
    video_name = Path(video_path).stem
    print(f"\n{'='*50}\nStarting Dev B Pipeline for {video_name}\n{'='*50}")
    
    branch_b_csv = os.path.join(out_dir, "features", f"{video_name}_branch_b.csv")
    fused_csv = os.path.join(out_dir, "features", f"{video_name}_fused.csv")
    scores_csv = os.path.join(out_dir, "scores", f"{video_name}_scores.csv")
    out_video = os.path.join(out_dir, "videos", f"{video_name}_annotated.mp4")
    out_plot = os.path.join(out_dir, "plots", f"{video_name}_score_dist.png")
    annotation_path = "data/ucf_crime/annotations/Temporal_Anomaly_Annotation.txt"
    
    # Stage 4B: VideoMAE
    print("\n--- Stage 4B: VideoMAE Embeddings ---")
    df_b = get_videomae_embeddings(video_path, tracks_csv, branch_b_csv, device)
    if len(df_b) == 0:
        print("Pipeline stopped: No VideoMAE embeddings generated.")
        return
        
    # Stage 5: Feature Fusion
    print("\n--- Stage 5: Feature Fusion ---")
    df_fused = fuse_features(branch_a_csv, branch_b_csv, fused_csv)
    if len(df_fused) == 0:
        print("Pipeline stopped: Feature fusion failed.")
        return
        
    # Stage 6: Clustering & Scoring
    print("\n--- Stage 6: HDBSCAN Clustering ---")
    df_scores = cluster_and_score(fused_csv, scores_csv)
    if len(df_scores) == 0:
        print("Pipeline stopped: Clustering failed.")
        return
        
    # Stage 7: Evaluation & Visualisation
    print("\n--- Stage 7: Evaluation & Visualisation ---")
    evaluate_and_visualize(video_path, tracks_csv, scores_csv, annotation_path, out_video, out_plot)
    
    print(f"\n{'='*50}\nPipeline Completed! Outputs saved to {out_dir}\n{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dev B pipeline (Stages 4B to 7)")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--tracks_csv", required=True, help="Dev A tracks CSV")
    parser.add_argument("--branch_a_csv", required=True, help="Dev A branch A features CSV")
    parser.add_argument("--out_dir", default="outputs", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Compute device (cuda/cpu)")
    
    args = parser.parse_args()
    
    run_pipeline(args.video, args.tracks_csv, args.branch_a_csv, args.out_dir, args.device)
