"""Visual proof of occlusion-onset detection.

For each held-out channel produces a figure with:
  - top: per-frame P(occluded) curve, label bands, true vs detected onset
  - bottom: a filmstrip of the actual channel frames at key timestamps,
            each annotated with time, P(occluded) and predicted class.

Usage:
    python -m clotting.eval.demo --config configs/baseline.yaml \
        --data-config configs/data.yaml
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

from clotting.config import load_yaml, project_root


def read_signal(path: Path):
    ts, pr, idx = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(float(row["time_s"]))
            pr.append(float(row["prob_occ"]))
            idx.append(int(row["frame_idx"]))
    return np.array(ts), np.array(pr), np.array(idx)


def nearest(ts: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(ts - t)))


def load_crop(frames_dir: Path, ch: str, frame_idx: int) -> np.ndarray:
    img = cv2.imread(str(frames_dir / ch / f"{frame_idx:05d}.png"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _add_flow_arrow(ax_first):
    """Single downward 'flow' arrow to the left of the filmstrip (flow top->bottom)."""
    ax_first.annotate(
        "", xy=(-0.55, 0.1), xytext=(-0.55, 0.9), xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#1f6fb2", lw=2.5),
    )
    ax_first.text(-0.78, 0.5, "flow", rotation=90, va="center", ha="center",
                  color="#1f6fb2", fontsize=9, fontweight="bold",
                  transform=ax_first.transAxes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args()

    root = project_root()
    cfg = load_yaml(args.config)
    dcfg = load_yaml(args.data_config)
    out_dir = root / cfg["output_dir"]
    frames_dir = root / dcfg["paths"]["frames_dir"]
    summary = json.loads((out_dir / "summary.json").read_text())

    sp = float(dcfg.get("speedup", 1.0))
    flow_before = float(dcfg["labels"]["flow_before_s"]) * sp
    occ_after = float(dcfg["labels"]["occ_after_s"]) * sp
    thr = float(cfg["eval"]["onset_threshold"])

    for fold in summary["folds"]:
        ch = fold["test_channel"]
        ts, pr, idx = read_signal(out_dir / f"signal_test_{ch}.csv")  # ts in video-time
        ts_r = ts * sp  # real-time axis for plotting
        po = fold["pred_onset_s"]  # video-time

        # Filmstrip timestamps (video-time for frame selection).
        strip_ts = [3.0, 8.0]
        if po is not None:
            strip_ts += [po - 1.0, po, po + 1.0]
        strip_ts += [22.0, 33.0]
        strip_ts = sorted(t for t in strip_ts if 0 <= t <= ts.max())

        fig = plt.figure(figsize=(13, 7.4))
        gs = fig.add_gridspec(2, len(strip_ts), height_ratios=[2.2, 1.6], hspace=0.62, wspace=0.1)

        ax = fig.add_subplot(gs[0, :])
        ax.plot(ts_r, pr, lw=1.3, color="#1f77b4", label="P(occluded)")
        ax.axhline(thr, color="grey", ls=":", lw=1, label=f"threshold {thr:g}")
        ax.axvspan(0, flow_before, color="#2ca02c", alpha=0.08)
        ax.axvspan(occ_after, ts_r.max(), color="#d62728", alpha=0.08)
        ax.axvspan(flow_before, occ_after, color="grey", alpha=0.15, label="unlabeled (excluded)")
        if po is not None:
            ax.axvline(po * sp, color="#ff7f0e", lw=2,
                       label=f"DETECTED onset {po*sp:.0f}s")
        for t in strip_ts:
            ax.plot([t * sp], [pr[nearest(ts, t)]], "o", color="#333", ms=4)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P(occluded)")
        ax.set_xlabel("real time (s)")
        ax.set_title(f"Held-out {ch}: occlusion-onset detection  |  frame F1={fold['metrics']['f1']:.3f}")
        ax.legend(loc="center left", fontsize=8, ncol=2)

        strip_axes = []
        for j, t in enumerate(strip_ts):
            k = nearest(ts, t)
            crop = load_crop(frames_dir, ch, int(idx[k]))
            axi = fig.add_subplot(gs[1, j])
            axi.imshow(crop)
            axi.set_xticks([]); axi.set_yticks([])
            p = pr[k]
            pred = "OCC" if p > thr else "flow"
            color = "#d62728" if p > thr else "#2ca02c"
            axi.set_title(f"{t*sp:.0f}s\nP={p:.2f} {pred}", fontsize=8, color=color)
            for s in axi.spines.values():
                s.set_color(color); s.set_linewidth(2.5)
            strip_axes.append(axi)
        _add_flow_arrow(strip_axes[0])

        out = out_dir / f"demo_{ch}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
