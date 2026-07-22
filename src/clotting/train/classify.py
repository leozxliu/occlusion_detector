"""Stage 1 training with leave-one-channel-out cross-validation.

For each channel held out as the test set, we train on the remaining
channel(s), evaluate frame-level metrics on the held-out labeled frames,
then run inference over ALL frames of the held-out channel (including the
excluded transition window) to produce a probability signal and detect the
occlusion onset time.

Usage:
    python -m clotting.train.classify \
        --data-config configs/data.yaml \
        --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clotting.config import load_yaml, project_root
from clotting.data.dataset import FrameDataset, class_weights, read_labels
from clotting.eval.metrics import binary_metrics
from clotting.eval.onset import detect_onset
from clotting.models.encoder import FrameClassifier


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_fold(
    test_channel: str,
    channels: list[str],
    frames_dir: Path,
    labels_dir: Path,
    cfg: dict,
    device: str,
) -> dict:
    m = cfg["model"]
    t = cfg["train"]
    img_size = int(m["img_size"])

    train_channels = [c for c in channels if c != test_channel]
    train_records: list = []
    for c in train_channels:
        train_records += read_labels(labels_dir / f"{c}.csv")
    test_records_all = read_labels(labels_dir / f"{test_channel}.csv")

    train_ds = FrameDataset(frames_dir, train_records, img_size, train=True, labeled_only=True)
    test_ds = FrameDataset(frames_dir, test_records_all, img_size, train=False, labeled_only=True)
    all_ds = FrameDataset(frames_dir, test_records_all, img_size, train=False, labeled_only=False)

    nw = int(t["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=int(t["batch_size"]), shuffle=True,
                              num_workers=nw, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=int(t["batch_size"]), shuffle=False, num_workers=nw)
    all_loader = DataLoader(all_ds, batch_size=int(t["batch_size"]), shuffle=False, num_workers=nw)

    model = FrameClassifier(m["backbone"], bool(m["pretrained"]), float(m["dropout"])).to(device)
    if bool(t["freeze_backbone"]):
        for p in model.backbone.parameters():
            p.requires_grad = False

    w = class_weights(train_records).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))

    print(f"[fold test={test_channel}] train frames={len(train_ds)} test frames={len(test_ds)}")
    for epoch in range(int(t["epochs"])):
        model.train()
        running = 0.0
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.long().to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
        print(f"  epoch {epoch+1:02d}/{t['epochs']} loss={running/max(1,len(train_ds)):.4f}")

    # Frame-level metrics on held-out labeled frames.
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y, *_ in test_loader:
            x = x.to(device)
            p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            y_pred += (p > 0.5).astype(int).tolist()
            y_true += y.int().tolist()
    metrics = binary_metrics(np.array(y_true), np.array(y_pred))

    # Full-signal inference over ALL frames (incl. transition) for onset detection.
    times, probs, idxs = [], [], []
    with torch.no_grad():
        for x, _y, frame_idx, time_s in all_loader:
            x = x.to(device)
            p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            probs += p.tolist()
            times += [float(v) for v in time_s]
            idxs += [int(v) for v in frame_idx]
    times = np.array(times)
    probs = np.array(probs)

    ecfg = cfg["eval"]
    pred_onset = detect_onset(times, probs, float(ecfg["onset_threshold"]),
                              int(ecfg["onset_sustain_frames"]))

    return {
        "test_channel": test_channel,
        "metrics": metrics,
        "pred_onset_s": pred_onset,
        "signal": {"frame_idx": idxs, "time_s": times.tolist(), "prob_occ": probs.tolist()},
        "model_state": model.state_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()

    root = project_root()
    dcfg = load_yaml(args.data_config)
    cfg = load_yaml(args.config)
    set_seed(int(cfg["train"]["seed"]))

    device = pick_device()
    print(f"device={device}")

    frames_dir = root / dcfg["paths"]["frames_dir"]
    labels_dir = root / dcfg["paths"]["labels_dir"]
    channels = list(dcfg["channels"].keys())
    speedup = float(dcfg.get("speedup", 1.0))

    out_dir = root / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"speedup": speedup, "folds": []}
    for test_channel in channels:
        res = train_one_fold(test_channel, channels, frames_dir, labels_dir, cfg, device)
        torch.save(res.pop("model_state"), out_dir / f"model_test_{test_channel}.pt")

        signal = res.pop("signal")
        with open(out_dir / f"signal_test_{test_channel}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame_idx", "time_s", "prob_occ"])
            for fi, ts, pr in zip(signal["frame_idx"], signal["time_s"], signal["prob_occ"]):
                wr.writerow([fi, f"{ts:.4f}", f"{pr:.6f}"])

        po = res["pred_onset_s"]
        res["pred_onset_real_s"] = po * speedup if po is not None else None
        summary["folds"].append(res)

        me = res["metrics"]
        rr = f"{res['pred_onset_real_s']:.1f}" if po is not None else "None"
        print(f"[fold test={test_channel}] acc={me['accuracy']:.3f} f1={me['f1']:.3f} "
              f"detected_onset={rr}s (real)")

    summary["mean_f1"] = float(np.mean([f["metrics"]["f1"] for f in summary["folds"]]))
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Stage 1 leave-one-channel-out summary ===")
    print(f"mean frame F1: {summary['mean_f1']:.3f}")
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
