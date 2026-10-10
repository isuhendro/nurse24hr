"""Perception interface: frame in, :class:`Features` out.

Step 2 ships only the interface and a placeholder. Step 3 adds the MediaPipe
hands + face implementation. The placeholder reports zero confidence on every
frame, so with no perception attached the pipeline raises FAULT after
``t_fault`` seconds. That is the intended fail-loud behaviour, not a bug.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from nurse24hr.features import Features


class Perceiver(Protocol):
    name: str

    def process(self, image: np.ndarray, t: float) -> Features:
        """Compute features for one BGR frame taken at time ``t``."""
        ...

    def draw(self, image: np.ndarray, f: Features) -> None:
        """Optionally draw landmarks / boxes onto ``image`` in place for the debug overlay."""
        ...


class NullPerceiver:
    """No perception. Every frame is low confidence."""

    name = "none (step 3)"

    def process(self, image: np.ndarray, t: float) -> Features:
        return Features(t=t, d=None, conf=0.0, n_persons=0)

    def draw(self, image: np.ndarray, f: Features) -> None:
        return None


def crop_roi(image: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    """Return the bed ROI view of ``image`` (no copy), or the whole image."""
    if roi is None:
        return image
    x0, y0, x1, y1 = roi
    h, w = image.shape[:2]
    x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
    return image[y0:y1, x0:x1]
