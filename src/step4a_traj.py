"""
Stage 4A — Branch A: Trajectory Feature Extraction (11-dimensional)
Computes the 11 kinematic features per cleaned track defined in PRD §6.2.

Features:
  1.  mean_speed         — mean Euclidean displacement per frame
  2.  speed_std          — std of frame-to-frame speed
  3.  max_speed          — peak instantaneous speed
  4.  accel_mean         — mean absolute acceleration
  5.  accel_std          — std of acceleration
  6.  stop_ratio         — fraction of frames where speed < 1.0 px/frame
  7.  total_distance     — cumulative path length (pixels)
  8.  displacement       — straight-line start-to-end distance
  9.  path_efficiency    — displacement / total_distance  (clamped [0,1])
  10. direction_variance — variance of movement angles (radians)
  11. avg_turn_angle     — mean absolute angular change (radians)

No NaN values are allowed in the output (guaranteed by _safe_* helpers).

Usage:
    from step4a_traj import TrajectoryFeatureExtractor
    extractor = TrajectoryFeatureExtractor()
    features_df = extractor.extract_all(cleaned_track_history)

CLI:
    python step4a_traj.py --video data/ucf_crime/Fighting001.mp4 --out_csv outputs/branch_a_features.csv
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from step0_sampling import sample_frames
from step1_detect import Detector
from step2_track import Tracker, TrackHistory
from step3_clean import clean_tracks, save_tracks_csv

# ── Constants ─────────────────────────────────────────────────────────────────

STOP_SPEED_THRESHOLD = 1.0   # px/frame — below this counts as "stopped"
FEATURE_NAMES = [
    "mean_speed",
    "speed_std",
    "max_speed",
    "accel_mean",
    "accel_std",
    "stop_ratio",
    "total_distance",
    "displacement",
    "path_efficiency",
    "direction_variance",
    "avg_turn_angle",
]


class TrajectoryFeatureExtractor:
    """
    Computes 11 kinematic trajectory features per track.

    Methods:
        extract_one(points)  → np.ndarray shape (11,)
        extract_all(tracks)  → pd.DataFrame with columns FEATURE_NAMES + 'track_id'
    """

    def extract_one(self, points: list[tuple]) -> np.ndarray:
        """
        Compute 11-dim feature vector for a single track.

        Args:
            points: List of (cx, cy, frame_idx, bbox) from step3.
                    Must have >= 2 points; ideally >= 20 (enforced by step3).

        Returns:
            np.ndarray of shape (11,) — all values are finite floats, no NaN.
        """
        # Sort by frame index (should already be sorted by step3, but be safe)
        points = sorted(points, key=lambda p: p[2])

        cx = np.array([p[0] for p in points], dtype=np.float64)
        cy = np.array([p[1] for p in points], dtype=np.float64)

        if len(cx) < 2:
            # Edge case: single-point track — return zeros
            return np.zeros(11, dtype=np.float64)

        # ── Frame-to-frame displacements (speeds) ────────────────────────────
        dx = np.diff(cx)
        dy = np.diff(cy)
        speeds = np.sqrt(dx**2 + dy**2)   # px/frame, length = N-1

        # ── Accelerations ────────────────────────────────────────────────────
        if len(speeds) >= 2:
            accels = np.abs(np.diff(speeds))   # length = N-2
        else:
            accels = np.array([0.0])

        # ── Direction angles ─────────────────────────────────────────────────
        angles = np.arctan2(dy, dx)            # length = N-1, radians in (-π, π]

        if len(angles) >= 2:
            # Angular change between consecutive steps (wrapped to [-π, π])
            raw_diffs = np.diff(angles)
            turn_angles = np.abs((raw_diffs + np.pi) % (2 * np.pi) - np.pi)
        else:
            turn_angles = np.array([0.0])

        # ── Aggregate features ───────────────────────────────────────────────
        mean_speed      = _safe_mean(speeds)
        speed_std       = _safe_std(speeds)
        max_speed       = float(np.max(speeds)) if len(speeds) > 0 else 0.0
        accel_mean      = _safe_mean(accels)
        accel_std       = _safe_std(accels)
        stop_ratio      = float(np.mean(speeds < STOP_SPEED_THRESHOLD))
        total_distance  = float(np.sum(speeds))
        displacement    = float(np.sqrt((cx[-1] - cx[0])**2 + (cy[-1] - cy[0])**2))
        path_efficiency = _safe_divide(displacement, total_distance)  # clamped [0,1]
        dir_variance    = _safe_var(angles)
        avg_turn_angle  = _safe_mean(turn_angles)

        features = np.array([
            mean_speed,
            speed_std,
            max_speed,
            accel_mean,
            accel_std,
            stop_ratio,
            total_distance,
            displacement,
            path_efficiency,
            dir_variance,
            avg_turn_angle,
        ], dtype=np.float64)

        # Final NaN/Inf guard — should never trigger with _safe_ helpers,
        # but acts as an explicit last-resort check
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        assert features.shape == (11,), f"Feature vector has wrong shape: {features.shape}"
        assert not np.any(np.isnan(features)), "NaN values found in feature vector — this is a bug."

        return features

    def extract_all(
        self,
        track_history: TrackHistory,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Extract features for all tracks and return as a DataFrame.

        Args:
            track_history: Cleaned track dict from step3.
            verbose:       Print progress.

        Returns:
            pd.DataFrame with columns: ['track_id'] + FEATURE_NAMES
            One row per track.  No NaN values in any feature column.
        """
        rows = []
        for track_id, points in track_history.items():
            vec = self.extract_one(points)
            row = {"track_id": track_id}
            row.update(dict(zip(FEATURE_NAMES, vec.tolist())))
            rows.append(row)

        if not rows:
            print("[step4a] WARNING: No tracks to extract features from.")
            return pd.DataFrame(columns=["track_id"] + FEATURE_NAMES)

        df = pd.DataFrame(rows)

        # Verify no NaN in feature columns (acceptance criterion AC-4)
        feature_cols = [c for c in df.columns if c != "track_id"]
        nan_counts = df[feature_cols].isna().sum()
        if nan_counts.any():
            bad = nan_counts[nan_counts > 0].to_dict()
            raise ValueError(f"[step4a] NaN values found in features — AC-4 VIOLATED: {bad}")

        if verbose:
            print(f"[step4a] Features extracted for {len(df)} tracks.")
            print(df[FEATURE_NAMES].describe().T[["mean", "std", "min", "max"]].to_string())

        return df


# ── Safe arithmetic helpers ───────────────────────────────────────────────────

def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if len(arr) > 0 else 0.0

def _safe_std(arr: np.ndarray) -> float:
    return float(np.std(arr)) if len(arr) > 1 else 0.0

def _safe_var(arr: np.ndarray) -> float:
    return float(np.var(arr)) if len(arr) > 1 else 0.0

def _safe_divide(num: float, denom: float) -> float:
    """path_efficiency = displacement / total_distance, clamped to [0,1]."""
    if denom <= 0.0:
        return 0.0
    return float(np.clip(num / denom, 0.0, 1.0))


def save_features_csv(df: pd.DataFrame, out_path: str) -> None:
    """Save Branch A feature DataFrame to CSV (part of Dev A → Dev B hand-off)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[step4a] Branch A features saved → {out_path}  ({len(df)} tracks, {len(FEATURE_NAMES)} features)")


def validate_features(df: pd.DataFrame) -> bool:
    """
    Run acceptance criterion AC-4: all 11-dim Branch A vectors contain zero NaN values.

    Returns True if valid, raises AssertionError if not.
    """
    feature_cols = FEATURE_NAMES
    missing = [c for c in feature_cols if c not in df.columns]
    assert not missing, f"Missing feature columns: {missing}"
    assert len(df.columns.intersection(feature_cols)) == 11, "Expected exactly 11 feature columns"

    nan_mask = df[feature_cols].isna()
    assert not nan_mask.any().any(), (
        f"AC-4 FAILED: NaN values found.\n{nan_mask.sum()[nan_mask.sum() > 0]}"
    )

    shape_ok = df[feature_cols].shape[1] == 11
    assert shape_ok, f"Expected 11 feature columns, got {df[feature_cols].shape[1]}"

    print("[step4a] AC-4 PASSED: 11-dim Branch A vectors — zero NaN values confirmed.")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4A: Branch A trajectory features")
    parser.add_argument("--video",          required=True, help="Path to input video")
    parser.add_argument("--out_csv",        default="outputs/branch_a_features.csv",
                        help="Output CSV for Branch A features")
    parser.add_argument("--tracks_csv",     default=None,
                        help="Optional: load pre-computed tracks CSV instead of re-running steps 0-3")
    parser.add_argument("--device",         default="", help="YOLO device: 'cuda', 'cpu', or ''")
    args = parser.parse_args()

    if args.tracks_csv:
        from step3_clean import load_tracks_csv
        print(f"[step4a] Loading tracks from {args.tracks_csv} ...")
        cleaned = load_tracks_csv(args.tracks_csv)
    else:
        print("[step4a] Running full pipeline (steps 0–3) ...")
        frames         = sample_frames(args.video)
        detector       = Detector(device=args.device)
        all_detections = detector.detect_frames(frames, verbose=False)
        tracker        = Tracker()
        raw_tracks     = tracker.track(frames, all_detections, verbose=False)
        cleaned        = clean_tracks(raw_tracks)

    extractor  = TrajectoryFeatureExtractor()
    features_df = extractor.extract_all(cleaned)

    validate_features(features_df)
    save_features_csv(features_df, args.out_csv)
