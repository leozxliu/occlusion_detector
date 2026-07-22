"""Extract per-channel cropped frames and temporal labels from the video.

Each of the two vertical channels is treated as an independent dataset.
Labeling (from configs/data.yaml):
    time < flow_before_s  -> label 0 (flowing)
    time > occ_after_s    -> label 1 (occluded)
    otherwise             -> label -1 (transition, excluded from training)

Usage:
    python -m clotting.io.extract --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from clotting.config import load_yaml, project_root


def label_for_time(t_s: float, flow_before_s: float, occ_after_s: float) -> int:
    if t_s < flow_before_s:
        return 0
    if t_s > occ_after_s:
        return 1
    return -1  # transition / excluded


def extract(config_path: str) -> None:
    root = project_root()
    cfg = load_yaml(config_path)
    fps = float(cfg["fps"])
    video_path = root / cfg["video"]
    frames_dir = root / cfg["paths"]["frames_dir"]
    labels_dir = root / cfg["paths"]["labels_dir"]
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    flow_before_s = float(cfg["labels"]["flow_before_s"])
    occ_after_s = float(cfg["labels"]["occ_after_s"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writers = {}
    files = {}
    for name in cfg["channels"]:
        (frames_dir / name).mkdir(parents=True, exist_ok=True)
        f = open(labels_dir / f"{name}.csv", "w", newline="")
        w = csv.writer(f)
        w.writerow(["frame_idx", "time_s", "label", "path"])
        writers[name] = w
        files[name] = f

    counts = {name: {0: 0, 1: 0, -1: 0} for name in cfg["channels"]}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = idx / fps
        label = label_for_time(t_s, flow_before_s, occ_after_s)
        for name, ch in cfg["channels"].items():
            x0, y0, x1, y1 = ch["roi"]
            crop = frame[y0:y1, x0:x1]
            rel = f"{name}/{idx:05d}.png"
            cv2.imwrite(str(frames_dir / rel), crop)
            writers[name].writerow([idx, f"{t_s:.4f}", label, rel])
            counts[name][label] += 1
        idx += 1

    cap.release()
    for f in files.values():
        f.close()

    print(f"Read {idx} frames (reported {n_frames}) at {fps} fps.")
    for name in cfg["channels"]:
        c = counts[name]
        print(f"  {name}: flowing={c[0]} occluded={c[1]} transition/excluded={c[-1]}")
    print(f"Frames -> {frames_dir}")
    print(f"Labels -> {labels_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    extract(args.config)


if __name__ == "__main__":
    main()
