"""
Stage 5 — Feature Fusion
Independently scales Branch A (trajectory) and Branch B (VideoMAE),
applies PCA to Branch B (1024 -> 50), and fuses them using weighted concatenation.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def fuse_features(branch_a_csv, branch_b_csv, out_csv):
    print(f"[step5] Loading Branch A features from {branch_a_csv}")
    df_a = pd.read_csv(branch_a_csv)
    print(f"[step5] Loading Branch B features from {branch_b_csv}")
    df_b = pd.read_csv(branch_b_csv)
    
    # Merge on track_id (inner join ensures we only keep tracks present in both)
    df_merged = pd.merge(df_a, df_b, on="track_id", how="inner")
    
    if len(df_merged) == 0:
        print("[step5] WARNING: No tracks found in common between Branch A and Branch B.")
        return pd.DataFrame()
        
    # Isolate feature columns
    cols_a = [c for c in df_a.columns if c != "track_id"]
    cols_b = [c for c in df_b.columns if c != "track_id"]
    
    features_a = df_merged[cols_a].values
    features_b = df_merged[cols_b].values
    
    print("[step5] Scaling features independently...")
    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    
    features_a_scaled = scaler_a.fit_transform(features_a)
    features_b_scaled = scaler_b.fit_transform(features_b)
    
    print("[step5] Applying PCA to Branch B (1024 -> 50 components)...")
    # Note: PRD specifies fitting on normal set only to avoid data leakage.
    # Since this pipeline runs on a single video, we fit on the available tracks.
    # For a real deployment, a pre-fit PCA model should be loaded here.
    n_components = min(50, len(features_b_scaled)) # Handle case where n_tracks < 50
    pca = PCA(n_components=n_components)
    features_b_pca = pca.fit_transform(features_b_scaled)
    
    # Pad to 50 if n_tracks < 50 to maintain the 61-dim structure
    if n_components < 50:
        pad_width = 50 - n_components
        features_b_pca = np.pad(features_b_pca, ((0, 0), (0, pad_width)), mode='constant')
        
    print("[step5] Concatenating: 0.3 * Branch_A + 0.7 * Branch_B_PCA...")
    # PRD weights
    fused_features = np.concatenate([
        0.3 * features_a_scaled,
        0.7 * features_b_pca
    ], axis=1)
    
    # Build output DataFrame
    out_rows = []
    for idx, row in df_merged.iterrows():
        out_row = {"track_id": row["track_id"]}
        for i, val in enumerate(fused_features[idx]):
            out_row[f"fused_{i}"] = val
        out_rows.append(out_row)
        
    out_df = pd.DataFrame(out_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    
    print(f"[step5] Fused features saved -> {out_csv} ({len(out_df)} tracks, {fused_features.shape[1]} features)")
    return out_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch_a_csv", required=True)
    parser.add_argument("--branch_b_csv", required=True)
    parser.add_argument("--out_csv", default="outputs/features/fused.csv")
    args = parser.parse_args()
    
    fuse_features(args.branch_a_csv, args.branch_b_csv, args.out_csv)
