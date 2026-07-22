"""Run trained Stage 1 model(s) on a NEW video to detect occlusion onset.

- Auto-detects channel ROIs (orientation-agnostic).
- Ensembles the leave-one-channel-out fold models for a robust prediction.
- Emits per-channel probability signals, detected onset times, and a demo figure
  (probability curve + filmstrip of real frames annotated with the prediction).

Usage:
    python -m clotting.infer.predict --video "Example RBC.avi" --out runs/rbc
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from clotting.config import load_yaml, project_root
from clotting.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from clotting.eval.onset import detect_onset
from clotting.infer.roi import detect_channels
from clotting.models.encoder import FrameClassifier


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(run_dir: Path, backbone: str, device: str) -> list[FrameClassifier]:
    models = []
    for ckpt in sorted(run_dir.glob("model_test_*.pt")):
        m = FrameClassifier(backbone, pretrained=False)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.to(device).eval()
        models.append(m)
    if not models:
        raise SystemExit(f"No model_test_*.pt found in {run_dir}")
    return models


def orient_crop(crop: np.ndarray, rotate: bool) -> np.ndarray:
    """Rotate horizontal channels to vertical so flow reads top->bottom.

    The training video is vertical with flow top->bottom. Horizontal videos here
    flow right->left originally; a 90 deg counter-clockwise rotation maps the
    right (inlet) edge to the top, so flow becomes top->bottom after rotation.
    The SAME rotation is used for the model input and for plotting.
    """
    if rotate:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop


def preprocess(crop: np.ndarray, img_size: int, rotate: bool = False) -> np.ndarray:
    crop = orient_crop(crop, rotate)
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)


@torch.no_grad()
def infer_channel(models, frames, device) -> np.ndarray:
    x = torch.from_numpy(np.stack(frames)).float().to(device)
    probs = []
    for m in models:
        probs.append(torch.softmax(m(x), dim=1)[:, 1].cpu().numpy())
    return np.mean(probs, axis=0)  # ensemble average


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="runs/infer")
    ap.add_argument("--run-dir", default="runs/stage1")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    root = project_root()
    cfg = load_yaml(args.config)
    try:
        sp = float(load_yaml(args.data_config).get("speedup", 1.0))
    except Exception:
        sp = 1.0
    img_size = int(cfg["model"]["img_size"])
    thr = float(cfg["eval"]["onset_threshold"])
    sustain = int(cfg["eval"]["onset_sustain_frames"])
    device = pick_device()
    print(f"device={device} speedup={sp}")

    video_path = str((root / args.video) if not Path(args.video).is_absolute() else Path(args.video))
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    rois, orientation = detect_channels(video_path, fps)
    rotate = orientation == "horizontal"
    print(f"detected {len(rois)} channels ({orientation}): {rois} (rotate={rotate})")

    models = load_models(root / args.run_dir, cfg["model"]["backbone"], device)
    print(f"ensembling {len(models)} fold model(s)")

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # One pass over the video collecting crops per channel.
    cap = cv2.VideoCapture(video_path)
    per_ch_frames = {i: [] for i in range(len(rois))}
    times = []
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        times.append(idx / fps)
        for ci, (x0, y0, x1, y1) in enumerate(rois):
            per_ch_frames[ci].append(preprocess(fr[y0:y1, x0:x1], img_size, rotate))
        idx += 1
    cap.release()
    times = np.array(times)

    results = {"video": args.video, "fps": fps, "orientation": orientation, "channels": []}
    signals = {}
    for ci in range(len(rois)):
        probs = np.zeros(len(times), dtype=np.float32)
        frames = per_ch_frames[ci]
        for s in range(0, len(frames), args.batch):
            probs[s:s + args.batch] = infer_channel(models, frames[s:s + args.batch], device)
        onset = detect_onset(times, probs, thr, sustain)
        onset_real = onset * sp if onset is not None else None
        signals[ci] = probs
        results["channels"].append({
            "roi": rois[ci],
            "pred_onset_video_s": onset,
            "pred_onset_real_s": onset_real,
        })
        with open(out_dir / f"signal_ch{ci}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame_idx", "time_video_s", "time_real_s", "prob_occ"])
            for k in range(len(times)):
                wr.writerow([k, f"{times[k]:.4f}", f"{times[k]*sp:.4f}", f"{probs[k]:.6f}"])
        rr = f"{onset_real:.1f}" if onset_real is not None else "None"
        print(f"  channel {ci}: detected onset = {rr} s (real)")

    results["speedup"] = sp
    results["rotated_for_display"] = rotate
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot(video_path, fps, rois, orientation, times, signals, results, thr, out_dir, sp, rotate)
    print(f"results -> {out_dir}")


def _grab(cap, fps, t, roi, rotate=False):
    x0, y0, x1, y1 = roi
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, fr = cap.read()
    crop = orient_crop(fr[y0:y1, x0:x1], rotate)  # same rotation as model input
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _add_flow_arrow(ax_first):
    """Draw a single downward 'flow' arrow to the left of the filmstrip.

    All channels are displayed with flow top->bottom (horizontal videos are
    rotated to match), so one downward arrow describes every panel.
    """
    ax_first.annotate(
        "", xy=(-0.55, 0.1), xytext=(-0.55, 0.9), xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#1f6fb2", lw=2.5),
    )
    ax_first.text(-0.78, 0.5, "flow", rotation=90, va="center", ha="center",
                  color="#1f6fb2", fontsize=9, fontweight="bold",
                  transform=ax_first.transAxes)


def _plot(video_path, fps, rois, orientation, times, signals, results, thr, out_dir, sp=1.0, rotate=False):
    cap = cv2.VideoCapture(video_path)
    tmax = float(times.max())
    rot_note = "  [rotated 90 deg CCW for display: flow top->bottom]" if rotate else ""
    for ci, roi in enumerate(rois):
        probs = signals[ci]
        onset = results["channels"][ci]["pred_onset_video_s"]  # video-time
        strip_ts = [1.0, tmax * 0.3]
        if onset is not None:
            strip_ts += [max(0, onset - 1.5), onset, onset + 1.5]
        strip_ts += [tmax * 0.8, tmax - 0.5]
        strip_ts = sorted(t for t in strip_ts if 0 <= t <= tmax)

        fig = plt.figure(figsize=(13, 6.8))
        gs = fig.add_gridspec(2, len(strip_ts), height_ratios=[2.2, 1.5], hspace=0.62, wspace=0.1)
        ax = fig.add_subplot(gs[0, :])
        ax.plot(times * sp, probs, lw=1.3, color="#1f77b4", label="P(occluded)")
        ax.axhline(thr, color="grey", ls=":", lw=1, label=f"threshold {thr:g}")
        if onset is not None:
            ax.axvline(onset * sp, color="#ff7f0e", lw=2, label=f"DETECTED onset {onset*sp:.0f}s")
        for t in strip_ts:
            ax.plot([t * sp], [probs[int(np.argmin(np.abs(times - t)))]], "o", color="#333", ms=4)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P(occluded)")
        ax.set_xlabel("real time (s)")
        ax.set_title(f"NEW VIDEO — channel {ci} ({orientation}): occlusion-onset detection{rot_note}")
        ax.legend(loc="center left", fontsize=8, ncol=2)

        strip_axes = []
        for j, t in enumerate(strip_ts):
            crop = _grab(cap, fps, t, roi, rotate)
            p = probs[int(np.argmin(np.abs(times - t)))]
            axi = fig.add_subplot(gs[1, j])
            axi.imshow(crop)
            axi.set_xticks([]); axi.set_yticks([])
            pred = "OCC" if p > thr else "flow"
            color = "#d62728" if p > thr else "#2ca02c"
            axi.set_title(f"{t*sp:.0f}s\nP={p:.2f} {pred}", fontsize=8, color=color)
            for s in axi.spines.values():
                s.set_color(color); s.set_linewidth(2.5)
            strip_axes.append(axi)
        _add_flow_arrow(strip_axes[0])

        out = out_dir / f"demo_ch{ci}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")
    cap.release()


if __name__ == "__main__":
    main()
