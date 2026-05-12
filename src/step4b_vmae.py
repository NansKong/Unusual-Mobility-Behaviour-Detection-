"""
Stage 4B — Branch B: VideoMAE Embeddings (1024-dimensional)
Extracts spatiotemporal features from bounding box crops of tracked persons.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from transformers import VideoMAEImageProcessor, VideoMAEModel
from tqdm import tqdm

SAMPLE_FPS = 5   # must match step0_sampling.py TARGET_FPS


def extract_crops_for_track(track_group, video_path):
    """
    Given a track's rows (frame_idx, x1, y1, x2, y2) and a video path,
    extract the crops and return them along with their original frame indices.

    frame_idx values in the CSV are indices into the *sampled* sequence
    (at SAMPLE_FPS).  We convert them back to native video frame numbers
    using the video's actual FPS so that cap.set() seeks to the right place.
    """
    crops = []
    frame_indices = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[step4b] ERROR: Could not open video: {video_path}")
        return crops, frame_indices

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0:
        native_fps = 30.0
        print(f"[step4b] WARNING: Could not read FPS, assuming 30 FPS")

    # Step interval used by step0_sampling (how many native frames per sampled frame)
    frame_interval = max(1, round(native_fps / SAMPLE_FPS))

    print(f"[step4b]   Video: {Path(video_path).name}, "
          f"native_fps={native_fps:.1f}, total_frames={total_frames}, "
          f"frame_interval={frame_interval}")

    track_group = track_group.sort_values("frame_idx")

    for _, row in track_group.iterrows():
        sampled_idx = int(row['frame_idx'])
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])

        # Convert sampled index → native frame number
        native_frame_no = sampled_idx * frame_interval

        if native_frame_no >= total_frames:
            print(f"[step4b]   Skipping sampled_idx={sampled_idx} "
                  f"(native={native_frame_no} >= total={total_frames})")
            continue

        # Direct seek — avoids sequential fast-forward error and stale `ret` bug
        cap.set(cv2.CAP_PROP_POS_FRAMES, native_frame_no)
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"[step4b]   WARNING: Could not read native frame {native_frame_no} "
                  f"(sampled_idx={sampled_idx})")
            continue

        # Clamp bbox to frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 - x1) >= 32 and (y2 - y1) >= 32:   # lowered from 64 to 32
            crop = frame[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop)
            frame_indices.append(sampled_idx)
        else:
            print(f"[step4b]   Crop too small at frame {sampled_idx}: "
                  f"{x2-x1}x{y2-y1}")

    cap.release()
    print(f"[step4b]   Track crops collected: {len(crops)}")
    return crops, frame_indices


def get_videomae_embeddings(video_path, tracks_csv, out_csv, device="cuda"):
    print(f"[step4b] Loading tracks from {tracks_csv}")
    tracks_df = pd.read_csv(tracks_csv)

    # Verify video exists before loading model
    if not os.path.isfile(video_path):
        print(f"[step4b] ERROR: Video file not found: {video_path}")
        print(f"[step4b] Hint: Make sure --video points to the actual .mp4 file, not a directory.")
        return pd.DataFrame()

    local_dir = os.path.join(os.path.dirname(__file__), "..", "models", "videomae_ucf")
    if not os.path.exists(local_dir):
        # Fallback to huggingface hub if local cache doesn't exist
        local_dir = "OPear/videomae-large-finetuned-UCF-Crime"

    print(f"[step4b] Loading VideoMAE model from {local_dir}")
    processor = VideoMAEImageProcessor.from_pretrained(local_dir)
    model = VideoMAEModel.from_pretrained(local_dir)
    model.to(device)
    model.eval()

    out_rows = []
    track_groups = tracks_df.groupby("track_id")

    for track_id, group in tqdm(track_groups, desc="Extracting VideoMAE embeddings"):
        crops, _ = extract_crops_for_track(group, video_path)

        if len(crops) == 0:
            print(f"[step4b]   Track {track_id}: 0 crops — skipped")
            continue

        clip_length = 16
        stride = 4

        if len(crops) < clip_length:
            # Pad by repeating last frame so short tracks still produce an embedding
            pad_needed = clip_length - len(crops)
            print(f"[step4b]   Track {track_id}: only {len(crops)} crops, "
                  f"padding with {pad_needed} repeated frames")
            crops = crops + [crops[-1]] * pad_needed

        clips = []
        for i in range(0, len(crops) - clip_length + 1, stride):
            clips.append(crops[i:i + clip_length])

        if not clips:
            # Fallback: single clip from whatever we have
            clips = [crops[:clip_length]]

        track_embeddings = []
        for clip in clips:
            inputs = processor(clip, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # CLS token is at index 0
                cls_token = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                track_embeddings.append(cls_token)

        if track_embeddings:
            # Mean pool over all clips
            track_emb = np.mean(np.concatenate(track_embeddings, axis=0), axis=0)
            row = {"track_id": track_id}
            for i, val in enumerate(track_emb):
                row[f"vmae_{i}"] = val
            out_rows.append(row)

        torch.cuda.empty_cache()

    if not out_rows:
        print("[step4b] WARNING: No embeddings extracted. All tracks might be too short or crops too small.")
        return pd.DataFrame()

    out_df = pd.DataFrame(out_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[step4b] Branch B features saved -> {out_csv} ({len(out_df)} tracks, 1024 features)")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video (.mp4/.avi)")
    parser.add_argument("--tracks_csv", required=True, help="Input tracks CSV")
    parser.add_argument("--out_csv", default="outputs/features/branch_b.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    get_videomae_embeddings(args.video, args.tracks_csv, args.out_csv, args.device)
