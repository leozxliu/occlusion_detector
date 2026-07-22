"""Plot per-frame occlusion-probability signals with labels and detected onset.

Usage:
    python -m clotting.eval.plot_signal --config configs/baseline.yaml \
        --data-config configs/data.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clotting.config import load_yaml, project_root


def read_signal(path: Path):
    ts, pr = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(float(row["time_s"]))
            pr.append(float(row["prob_occ"]))
    return ts, pr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args()

    root = project_root()
    cfg = load_yaml(args.config)
    dcfg = load_yaml(args.data_config)
    out_dir = root / cfg["output_dir"]
    summary = json.loads((out_dir / "summary.json").read_text())

    sp = float(dcfg.get("speedup", 1.0))
    flow_before = float(dcfg["labels"]["flow_before_s"]) * sp
    occ_after = float(dcfg["labels"]["occ_after_s"]) * sp

    folds = summary["folds"]
    fig, axes = plt.subplots(len(folds), 1, figsize=(10, 3.2 * len(folds)), sharex=True)
    if len(folds) == 1:
        axes = [axes]

    for ax, fold in zip(axes, folds):
        ch = fold["test_channel"]
        ts, pr = read_signal(out_dir / f"signal_test_{ch}.csv")
        ts = [t * sp for t in ts]
        ax.plot(ts, pr, lw=1.2, color="#1f77b4", label="P(occluded)")
        ax.axvspan(0, flow_before, color="#2ca02c", alpha=0.08, label="labeled flowing")
        ax.axvspan(occ_after, max(ts), color="#d62728", alpha=0.08, label="labeled occluded")
        ax.axvspan(flow_before, occ_after, color="grey", alpha=0.15, label="unlabeled (excluded)")
        po = fold["pred_onset_s"]
        if po is not None:
            po_r = po * sp
            ax.axvline(po_r, color="#ff7f0e", ls="-", lw=1.6,
                       label=f"detected onset {po_r:.0f}s")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P(occluded)")
        ax.set_title(f"held-out {ch}  |  frame F1={fold['metrics']['f1']:.3f}")
        ax.legend(loc="center left", fontsize=7, ncol=2)

    axes[-1].set_xlabel("real time (s)")
    fig.suptitle("Stage 1 — per-frame occlusion probability (leave-one-channel-out)")
    fig.tight_layout()
    out = out_dir / "stage1_signals.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
