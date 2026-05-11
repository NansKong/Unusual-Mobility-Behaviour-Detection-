# Unusual Mobility Behaviour Detection for Security Surveillance

**Version:** 1.0
**Project Type:** Unsupervised ML Research Project
**Dataset:** UCF-Crime (1,900 surveillance videos, 13 anomaly classes)

## 1. Executive Summary
This repository contains an unsupervised machine learning pipeline that detects unusual mobility behaviour in security surveillance video. The system requires **no labelled training data** at inference time.

The pipeline fuses two complementary representations of human movement:
1. **Geometric trajectory features** (Branch A)
2. **Deep spatiotemporal video embeddings** (Branch B)

It applies density-based clustering to separate normal from anomalous behaviour. Anomalies are ranked by a continuous score derived from cluster geometry, enabling both binary alerting and severity ranking.

## 2. Core Architecture
The system fuses the outputs of two parallel branches into a 61-dimensional feature vector, clustered using HDBSCAN.

### Primary Models & Tools
* **Detection:** YOLOv8n (Person class only, `conf >= 0.4`, min `64x64px`)
* **Tracking:** DeepSORT (`max_age=30`, `n_init=3`)
* **Spatiotemporal Embedding (Branch B):** VideoMAE (`OPear/videomae-large-finetuned-UCF-Crime`)
* **Clustering:** HDBSCAN
* **Evaluation Metric:** Frame-level AUC (ROC)

### Pipeline Stages
* **Stage 0:** Frame sampling & resizing (5 FPS, width=640px)
* **Stage 1:** YOLOv8n Person Detection
* **Stage 2:** DeepSORT Tracking
* **Stage 3:** Trajectory Cleaning (Min 20 frames, scipy interpolation)
* **Stage 4A (Branch A):** Trajectory Feature Extraction (11-dim vector per track: speed, accel, etc.)
* **Stage 4B (Branch B):** VideoMAE Embedding (1024-dim CLS token extraction)
* **Stage 5:** Feature Fusion (Scaling + PCA to 50-dim + Weighted Concatenation to 61-dim)
* **Stage 6:** HDBSCAN Clustering & Continuous Anomaly Score [0,1]
* **Stage 7:** Evaluation (Frame-level AUC) & Annotated Video Export

## 3. Project Structure
```text
unusual_mobility_detection/
├── data/
│   └── ucf_crime/
│       ├── annotations/       # Temporal_Anomaly_Annotation.txt
│       └── videos/            # Raw .mp4/.avi video files
├── models/
│   ├── yolov8n.pt             # Auto-downloaded by ultralytics
│   └── videomae_ucf/          # Cached HuggingFace VideoMAE model
├── src/
│   ├── download_model.py      # Script to locally cache VideoMAE
│   ├── step0_sampling.py      # Frame extraction
│   ├── step1_detect.py        # YOLOv8n person detection
│   ├── step2_track.py         # DeepSORT tracking
│   ├── step3_clean.py         # Trajectory cleaning
│   ├── step4a_traj.py         # Branch A feature extraction
│   ├── step4b_vmae.py         # Branch B VideoMAE embedding
│   ├── step5_fuse.py          # Scaling + PCA + fusion
│   ├── step6_cluster.py       # HDBSCAN + anomaly scoring
│   ├── step7_eval.py          # AUC + visualisation
│   └── pipeline.py            # Full end-to-end runner
├── outputs/
│   ├── scores/                # Per-track anomaly score CSVs
│   ├── videos/                # Annotated output videos
│   └── plots/                 # Score distribution charts
├── requirements.txt
└── README.md
```

## 4. Setup & Installation
### Prerequisites
* Python `3.10.14`
* CUDA `11.8`

> [!WARNING]
> You **must** strictly adhere to the pinned dependency versions, specifically `numpy==1.26.4` and `torch==2.1.2+cu118`. Newer versions (like Numpy 2.x) will break internal Kalman filters used by DeepSORT.

### Installation Instructions
1. **Create and activate a virtual environment (Python 3.10.14 required):**
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # Linux/Mac: source venv/bin/activate
   ```
2. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Pre-download the VideoMAE Model Cache (Dev B / Shared Task):**
   ```bash
   python src/download_model.py
   ```
   *This caches the model locally to `models/videomae_ucf/` to prevent HuggingFace timeout failures.*

## 5. Developer Workflow (Split Execution)

### Dev A (Tracking & Trajectories)
* **Day 1:** Environment setup + YOLO smoke test on 1 UCF video.
* **Day 2:** DeepSORT integration; track IDs visualization.
* **Day 3:** Trajectory cleaning & extracting Branch A 11-dim features. Hand-off CSV to Dev B.

### Dev B (Embeddings & Clustering)
* **Day 1:** Pre-download VideoMAE model & verify CLS token shape. (Done via `download_model.py` and `step4b_vmae.py`)
* **Day 4:** Receive Dev A's track CSV. Run Branch B VideoMAE on track bounding boxes.
* **Day 5:** StandardScaler + PCA (fit on normal set only) + Fusion.
* **Day 6:** HDBSCAN + continuous score [0,1] & cluster visualization.
* **Day 7:** End-to-end evaluation (AUC $\ge$ 0.80) & annotated video export.

## 6. Known Failure Modes & Mitigations
* **Numpy ImportError:** Ensure `numpy==1.26.4` is strictly pinned.
* **OOM (Out of Memory):** Call `torch.cuda.empty_cache()` per track in Stage 4B.
* **All tracks labeled -1 (Noise):** Decrease `min_cluster_size` to 3, or fallback to DBSCAN.
* **VideoMAE tiny crops error:** Ensure YOLO output explicitly filters detections `< 64x64px`.
* **Data Leakage:** PCA must be fitted **only** on the normal training trajectories, not testing anomaly data.
