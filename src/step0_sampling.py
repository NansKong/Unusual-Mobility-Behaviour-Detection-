"""
Stage 0 — Frame Sampling + Resize
Extracts frames from a video at 5 FPS, resizes to width=640 (aspect-preserving).
Output: list of BGR numpy arrays (in-memory) OR saved as .jpg files to disk.

Usage:
    from step0_sampling import sample_frames
    frames = sample_frames("path/to/video.mp4")

CLI:
    python step0_sampling.py --video data/ucf_crime/Fighting001.mp4 --out_dir outputs/frames/
"""

import cv2
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


TARGET_FPS = 5
TARGET_WIDTH = 640


def sample_frames(
    video_path: str,
    target_fps: int = TARGET_FPS,
    target_width: int = TARGET_WIDTH,
    save_dir: str | None = None,
) -> list[np.ndarray]:
    """
    Sample frames from a video at `target_fps`, resize to `target_width` (aspect-preserving).

    Args:
        video_path:   Path to .mp4 or .avi file.
        target_fps:   Desired output frame rate (default 5).
        target_width: Output frame width in pixels (default 640).
        save_dir:     If provided, saves frames as JPEG files here (for debugging/inspection).

    Returns:
        List of BGR numpy arrays, shape (H, W, 3).

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError:      If OpenCV cannot open the file.
    """
    video_path = str(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0:
        # Fallback: assume 30 FPS if metadata is missing
        native_fps = 30.0
        print(f"[WARNING] Could not read FPS from {video_path}. Assuming 30 FPS.")

    # How many native frames to skip between each sampled frame
    frame_interval = max(1, round(native_fps / target_fps))

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    frame_idx = 0
    saved_idx = 0

    pbar = tqdm(total=total_frames, desc=f"Sampling {Path(video_path).name}", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            resized = _resize_frame(frame, target_width)
            frames.append(resized)

            if save_dir:
                out_path = os.path.join(save_dir, f"frame_{saved_idx:05d}.jpg")
                cv2.imwrite(out_path, resized)
                saved_idx += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"[step0] Sampled {len(frames)} frames from {Path(video_path).name} "
          f"(native {native_fps:.1f} FPS → {target_fps} FPS, interval={frame_interval})")
    return frames


def _resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Resize frame to target_width, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR)


def sample_frames_batch(
    video_paths: list[str],
    target_fps: int = TARGET_FPS,
    target_width: int = TARGET_WIDTH,
    save_root: str | None = None,
) -> dict[str, list[np.ndarray]]:
    """
    Convenience wrapper: sample frames from multiple videos.

    Args:
        video_paths: List of video file paths.
        target_fps:  Desired FPS.
        target_width: Desired width.
        save_root:   Root dir; each video gets its own sub-folder if provided.

    Returns:
        Dict mapping video filename (stem) → list of BGR frames.
    """
    results = {}
    for vp in video_paths:
        stem = Path(vp).stem
        save_dir = os.path.join(save_root, stem) if save_root else None
        results[stem] = sample_frames(vp, target_fps, target_width, save_dir)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0: Frame sampling and resize")
    parser.add_argument("--video", required=True, help="Path to input video (.mp4/.avi)")
    parser.add_argument("--out_dir", default=None, help="Directory to save frames as JPEG")
    parser.add_argument("--fps", type=int, default=TARGET_FPS, help="Target FPS (default 5)")
    parser.add_argument("--width", type=int, default=TARGET_WIDTH, help="Target width px (default 640)")
    args = parser.parse_args()

    frames = sample_frames(args.video, args.fps, args.width, args.out_dir)
    print(f"Done. {len(frames)} frames extracted. Shape of first frame: {frames[0].shape}")
