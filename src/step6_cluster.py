"""
Stage 6 — HDBSCAN Clustering
Clusters fused features and assigns a continuous anomaly score [0,1]
based on the distance to the nearest normal cluster centroid.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import hdbscan

def compute_cluster_centroids(features, labels):
    centroids = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue
        mask = (labels == label)
        centroids[label] = np.mean(features[mask], axis=0)
    return centroids

def cluster_and_score(fused_csv, out_csv):
    print(f"[step6] Loading fused features from {fused_csv}")
    df = pd.read_csv(fused_csv)
    
    if len(df) == 0:
        print("[step6] WARNING: Empty feature dataframe.")
        return pd.DataFrame()
        
    feature_cols = [c for c in df.columns if c != "track_id"]
    features = df[feature_cols].values
    
    # Run HDBSCAN
    # min_cluster_size=5 (reduced to 3 in fallback if needed, but PRD says start with 5)
    print("[step6] Running HDBSCAN...")
    n_samples = len(features)
    
    # HDBSCAN needs at least 2 points; with very few tracks, skip clustering
    if n_samples <= 2:
        print(f"[step6] WARNING: Only {n_samples} track(s). Too few for clustering — marking all as anomalous.")
        labels = np.full(n_samples, -1)
    else:
        min_cluster_size = min(5, max(2, n_samples - 1))
        # min_samples must be <= n_samples for the internal k-NN query
        min_samp = min(3, n_samples)
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samp, metric='euclidean')
        labels = clusterer.fit_predict(features)
    
    centroids = compute_cluster_centroids(features, labels)
    
    scores = np.zeros(len(features))
    
    if len(centroids) == 0:
        print("[step6] WARNING: All tracks labelled as noise (-1). Falling back to uniform anomaly score.")
        # If everything is noise, we might assign a high score to all
        scores = np.ones(len(features))
    else:
        # Compute distance to nearest centroid
        for i, feat in enumerate(features):
            min_dist = float('inf')
            for c_label, centroid in centroids.items():
                dist = np.linalg.norm(feat - centroid)
                if dist < min_dist:
                    min_dist = dist
            scores[i] = min_dist
            
        # Normalize scores to [0,1]
        max_score = np.max(scores)
        if max_score > 0:
            scores = scores / max_score
            
    df["cluster"] = labels
    df["anomaly_score"] = scores
    
    out_df = df[["track_id", "cluster", "anomaly_score"]]
    
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    
    print(f"[step6] Scores saved -> {out_csv} ({len(out_df)} tracks)")
    print(f"[step6] Found {len(centroids)} valid clusters and {np.sum(labels == -1)} noise points.")
    return out_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused_csv", required=True)
    parser.add_argument("--out_csv", default="outputs/scores/scores.csv")
    args = parser.parse_args()
    
    cluster_and_score(args.fused_csv, args.out_csv)
