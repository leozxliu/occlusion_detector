"""Turn a per-frame occlusion-probability signal into a single onset time.

Stage 1 uses a simple sustained-threshold (hysteresis) rule: the detector
triggers at the first frame whose probability exceeds `threshold` and stays
above it for `sustain` consecutive frames. This avoids firing on transient
single-frame spikes.
"""
from __future__ import annotations

import numpy as np


def detect_onset(
    times_s: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    sustain: int = 5,
) -> float | None:
    order = np.argsort(times_s)
    times_s = np.asarray(times_s)[order]
    probs = np.asarray(probs)[order]
    above = probs > threshold
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= sustain:
            return float(times_s[i - sustain + 1])
    return None
