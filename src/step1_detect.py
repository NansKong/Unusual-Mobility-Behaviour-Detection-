"""
Stage 1 — YOLOv8n Person Detection
Runs YOLOv8n on each sampled frame, returns bounding boxes for person class only.

Output per frame: list of [x1, y1, x2, y2, conf] — float, absolute pixel coords.
Boxes smaller than MIN_BBOX_PX x MIN_BBOX_PX are filtered out (unusable for VideoMAE).

Usage:
    from step1_detect import Detector
    detector = Detector()
    detections = detector.detect_frames(frames)   # frames from step0

CLI:
    python step1_detect.py --video data/ucf_crime/Fighting001.mp4
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Optional

# Defer heavy import so the module is importable even without ultralytics installed,
# allowing unit tests to mock the class.
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore

from step0_sampling import sample_frames

# ── Constants ─────────────────────────────────────────────────────────────────

YOLO_WEIGHTS   = "yolov8n.pt"   # auto-downloaded by ultralytics on first run
PERSON_CLASS   = 0              # COCO class index for 'person'
CONF_THRESHOLD = 0.4
MIN_BBOX_PX    = 64             # minimum side length in pixels


class Detector:
    """
    Wraps YOLOv8n for per-frame person detection.

    Args:
        weights:   Path to .pt file or model name (auto-downloaded).
        conf:      Confidence threshold (default 0.4).
        min_bbox:  Minimum bbox width AND height in pixels (default 64).
        device:    'cuda', 'cpu', or '' for auto-select.
    """

    def __init__(
        self,
        weights: str = YOLO_WEIGHTS,
        conf: float = CONF_THRESHOLD,
        min_bbox: int = MIN_BBOX_PX,
        device: str = "",
    ):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics==8.1.47")

        self.conf = conf
        self.min_bbox = min_bbox
        print(f"[step1] Loading YOLO weights: {weights}")
        self.model = YOLO(weights)
        self.device = device
        print(f"[step1] YOLO model loaded. Person class={PERSON_CLASS}, conf={conf}, min_bbox={min_bbox}px")

    def detect_frame(self, frame: np.ndarray) -> list[list[float]]:
        """
        Detect persons in a single BGR frame.

        Args:
            frame: BGR numpy array (H, W, 3).

        Returns:
            List of [x1, y1, x2, y2, conf] for each valid person detection.
            Empty list if no persons found.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            classes=[PERSON_CLASS],
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # Filter tiny bboxes — too small for VideoMAE crops
                w = x2 - x1
                h = y2 - y1
                if w < self.min_bbox or h < self.min_bbox:
                    continue

                detections.append([x1, y1, x2, y2, conf])

        return detections

    def detect_frames(
        self,
        frames: list[np.ndarray],
        verbose: bool = True,
    ) -> list[list[list[float]]]:
        """
        Detect persons across all sampled frames.

        Args:
            frames:  List of BGR numpy arrays from step0.
            verbose: Print detection summary.

        Returns:
            List (one entry per frame) of lists of [x1,y1,x2,y2,conf].
            Index aligns 1-to-1 with input `frames`.
        """
        all_detections: list[list[list[float]]] = []
        total_persons = 0

        for i, frame in enumerate(frames):
            dets = self.detect_frame(frame)
            all_detections.append(dets)
            total_persons += len(dets)

            if verbose and (i % 50 == 0 or i == len(frames) - 1):
                print(f"[step1] Frame {i+1}/{len(frames)} — {len(dets)} person(s) detected")

        if verbose:
            print(f"[step1] Done. {total_persons} total detections across {len(frames)} frames.")

        return all_detections


def detections_to_deepsort_input(
    detections: list[list[float]],
) -> list[tuple[list[float], float, int]]:
    """
    Convert our detection format to deep-sort-realtime's expected input format.

    deep-sort-realtime expects:
        list of ([left, top, w, h], confidence, class_id)

    Args:
        detections: List of [x1, y1, x2, y2, conf] from detect_frame().

    Returns:
        List of ([l, t, w, h], conf, class_id) tuples.
    """
    ds_input = []
    for x1, y1, x2, y2, conf in detections:
        l, t = x1, y1
        w, h = x2 - x1, y2 - y1
        ds_input.append(([l, t, w, h], conf, PERSON_CLASS))
    return ds_input


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: YOLOv8n person detection")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--min_bbox", type=int, default=MIN_BBOX_PX)
    parser.add_argument("--device", default="", help="'cuda', 'cpu', or '' for auto")
    args = parser.parse_args()

    print(f"[step1] Sampling frames from {args.video} ...")
    frames = sample_frames(args.video)

    detector = Detector(conf=args.conf, min_bbox=args.min_bbox, device=args.device)
    detections = detector.detect_frames(frames)

    # Quick summary
    non_empty = sum(1 for d in detections if d)
    print(f"\nSummary: {non_empty}/{len(detections)} frames had at least one person detected.")
    if detections[0]:
        print(f"Example detection (frame 0): {detections[0]}")
