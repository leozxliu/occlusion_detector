"""Auto-detect microfluidic channel ROIs in a video.

Channels are the regions that darken as thrombus forms (bright translucent flow
-> dark packed occlusion). We compare an early frame to a late frame and find
contiguous bands of strong brightness drop, trying both orientations
(vertical bands via column profile, horizontal bands via row profile) and
picking whichever gives the cleaner set of bands.
"""
from __future__ import annotations

import cv2
import numpy as np


def _gray_at(cap, fps: float, t_s: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_s * fps))
    ok, fr = cap.read()
    if not ok:  # fall back to a valid frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, fr = cap.read()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _runs_from_profile(drop: np.ndarray, frac: float, min_len: int):
    k = max(3, (len(drop) // 20) | 1)  # odd kernel
    drop = cv2.GaussianBlur(drop.reshape(-1, 1), (1, k), 0).ravel()
    thr = drop.max() * frac
    mask = drop > thr
    runs, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        if not m and s is not None:
            runs.append((s, i - 1))
            s = None
    if s is not None:
        runs.append((s, len(mask) - 1))
    runs = [(a, b) for a, b in runs if (b - a) > min_len]
    contrast = float(drop.max() / (np.median(drop) + 1e-6))
    return runs, contrast


def detect_channels(
    video_path: str,
    fps: float,
    t_early: float = 1.0,
    t_late_frac: float = 0.9,
    frac: float = 0.35,
) -> tuple[list[tuple[int, int, int, int]], str]:
    """Return (rois, orientation). rois are [x0,y0,x1,y1]; orientation in {vertical,horizontal}."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    early = _gray_at(cap, fps, t_early)
    late = _gray_at(cap, fps, (n / fps) * t_late_frac)
    cap.release()
    diff = early - late

    col_runs, col_contrast = _runs_from_profile(diff.mean(axis=0), frac, min_len=w // 30)
    row_runs, row_contrast = _runs_from_profile(diff.mean(axis=1), frac, min_len=h // 30)

    # Prefer the axis with a clean, small number of bands and higher contrast.
    def score(runs, contrast):
        if not runs:
            return -1
        return contrast - 2.0 * max(0, len(runs) - 2)

    if score(col_runs, col_contrast) >= score(row_runs, row_contrast):
        rois = [(a, 0, b, h) for a, b in col_runs]
        return rois, "vertical"
    rois = [(0, a, w, b) for a, b in row_runs]
    return rois, "horizontal"
