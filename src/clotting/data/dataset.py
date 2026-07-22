"""Per-frame dataset for Stage 1 (flowing vs occluded classification)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class FrameRecord:
    frame_idx: int
    time_s: float
    label: int
    path: str


def read_labels(csv_path: str | Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                FrameRecord(
                    frame_idx=int(row["frame_idx"]),
                    time_s=float(row["time_s"]),
                    label=int(row["label"]),
                    path=row["path"],
                )
            )
    return records


class FrameDataset(Dataset):
    """Loads cropped channel frames. Optionally filter to labeled frames only."""

    def __init__(
        self,
        frames_dir: str | Path,
        records: list[FrameRecord],
        img_size: int = 224,
        train: bool = False,
        labeled_only: bool = True,
    ):
        self.frames_dir = Path(frames_dir)
        self.img_size = img_size
        self.train = train
        self.records = [r for r in records if (r.label in (0, 1))] if labeled_only else records

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # Light augmentation; flow/occlusion is texture+brightness, so keep it gentle.
        if np.random.rand() < 0.5:
            img = img[:, ::-1]  # horizontal flip
        if np.random.rand() < 0.5:
            img = img[::-1, :]  # vertical flip (channel is roughly symmetric)
        if np.random.rand() < 0.5:
            factor = 0.85 + 0.3 * np.random.rand()
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img)

    def __getitem__(self, i: int):
        r = self.records[i]
        img = cv2.imread(str(self.frames_dir / r.path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        if self.train:
            img = self._augment(img)
        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).float()
        y = torch.tensor(float(r.label))
        return x, y, r.frame_idx, float(r.time_s)


def class_weights(records: list[FrameRecord]) -> torch.Tensor:
    labs = [r.label for r in records if r.label in (0, 1)]
    n0 = max(1, labs.count(0))
    n1 = max(1, labs.count(1))
    total = n0 + n1
    # inverse-frequency weights, normalized
    return torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float32)
