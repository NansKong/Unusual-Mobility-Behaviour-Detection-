# Dev A — Stages 0–3 + Branch A

> **Unusual Mobility Behaviour Detection** | Dev A workload per PRD §16

---

## What's in this folder

| File | Stage | Description |
|---|---|---|
| `step0_sampling.py` | 0 | Frame extraction @ 5 FPS, resize to 640px |
| `step1_detect.py` | 1 | YOLOv8n person detection (conf ≥ 0.4, min 64×64 px) |
| `step2_track.py` | 2 | DeepSORT tracking (max_age=30, n_init=3) |
| `step3_clean.py` | 3 | Trajectory cleaning (min 20 frames, interpolate gaps ≤ 5) |
| `step4a_traj.py` | 4A | 11-dim trajectory feature extraction (Branch A) |
| `pipeline_dev_a.py` | 0–4A | Full Dev A end-to-end runner |

---

## Quick start

### 1. Environment setup (Day 1)

```bash
conda create -n mobility python=3.10.14
conda activate mobility

# CUDA torch first (critical order)
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# All other deps
pip install ultralytics==8.1.47 deep-sort-realtime==1.3.2 filterpy==1.4.5 \
    scipy==1.13.0 pandas==2.2.2 opencv-python==4.9.0.80 scikit-learn==1.4.2 \
    tqdm==4.66.4 Pillow==10.3.0
```

> ⚠️ **numpy must be 1.26.4** — numpy 2.x breaks `filterpy` (DeepSORT Kalman filter).
> Verify with `python -c "import numpy; print(numpy.__version__)"` → should print `1.26.4`.

### 2. Smoke test (Day 1)

```bash
# Test YOLO on one UCF-Crime video
python step1_detect.py --video data/ucf_crime/Fighting001.mp4 --device cuda
```

Expected output: bounding boxes printed per frame, summary at the end.

### 3. Run the full Dev A pipeline

```bash
python pipeline_dev_a.py \
    --video   data/ucf_crime/Fighting001.mp4 \
    --out_dir outputs/ \
    --device  cuda
```

This produces two hand-off files for Dev B:
- `outputs/tracks/Fighting001_tracks.csv`
- `outputs/features/Fighting001_branch_a.csv`

---

## Hand-off format for Dev B (PRD §16)

### `*_tracks.csv`
```
track_id, frame_idx, cx, cy, x1, y1, x2, y2
1, 0, 312.4, 205.1, 280.0, 150.0, 344.8, 260.2
1, 1, 314.2, 207.8, 282.1, 152.3, 346.3, 263.3
...
```

### `*_branch_a.csv`
```
track_id, mean_speed, speed_std, max_speed, accel_mean, accel_std,
          stop_ratio, total_distance, displacement, path_efficiency,
          direction_variance, avg_turn_angle
1, 3.21, 1.45, 12.3, 0.82, 0.61, 0.04, 487.3, 210.5, 0.43, 1.02, 0.31
...
```

---

## Branch A — 11 trajectory features (PRD §6.2)

| # | Feature | Description |
|---|---|---|
| 1 | `mean_speed` | Mean Euclidean displacement per frame |
| 2 | `speed_std` | Std of frame-to-frame speed |
| 3 | `max_speed` | Peak instantaneous speed |
| 4 | `accel_mean` | Mean absolute acceleration |
| 5 | `accel_std` | Std of acceleration |
| 6 | `stop_ratio` | Fraction of frames where speed < 1.0 px/frame |
| 7 | `total_distance` | Cumulative path length (pixels) |
| 8 | `displacement` | Straight-line start-to-end distance |
| 9 | `path_efficiency` | Displacement / total_distance, clamped [0,1] |
| 10 | `direction_variance` | Variance of movement angles (radians) |
| 11 | `avg_turn_angle` | Mean absolute angular change (radians) |

All features are guaranteed NaN-free (validated by `validate_features()` — acceptance criterion **AC-4**).

---

## Acceptance criteria (Dev A responsibilities)

| ID | Criterion | How verified |
|---|---|---|
| AC-1 | Environment installs without conflict | `pip install` exits 0 |
| AC-2 | YOLOv8n detects ≥ 90% of visible persons on 10 test frames | `step1_detect.py` output |
| AC-3 | Track IDs stable for ≥ 20 consecutive frames on 3 test clips | `step2_track.py` summary table |
| AC-4 | 11-dim Branch A vectors contain zero NaN values | `validate_features()` in `step4a_traj.py` |

---

## Known failure modes (Dev A)

| Issue | Cause | Fix |
|---|---|---|
| `ImportError: filterpy` | numpy 2.x installed | Pin `numpy==1.26.4` |
| All tracks discarded | Clips < 20 frames | Reduce `--min_len` or use longer clips |
| YOLO detects zero persons | Threshold too high | Lower `--conf` to 0.3 |
| OOM during YOLO | Large batch | YOLO runs one frame at a time — should not OOM |
| `ValueError: NaN in features` | Bug in trajectory math | `_safe_*` helpers should prevent this; check step3 output |

---

## Day-by-day schedule

| Day | Task | Done when |
|---|---|---|
| 1 | Conda env + YOLO smoke test | YOLO prints detections on 1 video |
| 2 | DeepSORT integration + visualise track IDs | Track IDs stable across 20+ frames |
| 3 | Trajectory cleaner + Branch A features | `validate_features()` passes with zero NaN |
| 4 | Hand off `*_tracks.csv` + `*_branch_a.csv` to Dev B | Files saved, Dev B confirms format |
