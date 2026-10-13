"""Seed labels for the tube detector without a trained model (TS-001 §7.4 steps 2-3).

The insertion point of a nasogastric tube is at the nostril, and the tube is
taped from there across the cheek toward the ear. Given the nostril position
and face width from face landmarks (computed elsewhere, passed in as numbers),
we can produce:

* a tight **insertion-point box** around the nostril,
* a **prior polyline** for the tube path on the configured side,
* **YOLO-format label lines** for classes ``0 insertion_point``, ``1 tube``.

The prior is only a seed. A :class:`Segmenter` refines it into a pixel mask;
``mask_to_box`` turns that mask into the tube box. Two implementations:

* :class:`NullSegmenter` draws a thin line along the prior. The pipeline runs
  end to end with it and the resulting labels are reviewed by a human.
* **Plugging in SAM**: implement ``segment(image, points)`` by calling a
  promptable segmenter with ``points`` as positive point prompts (they lie on
  the prior) and return the best mask as ``uint8`` HxW with 1 = tube. Keep the
  import inside the class so this module never depends on it.
* **VLM check**: after segmentation, optionally ask a vision-language model
  "is the feeding tube visible and does this box cover it?" on the cropped
  frame and route rejects to manual review. That belongs in a separate review
  script; this module only produces candidates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import cv2
import numpy as np

CLASS_NAMES: dict[int, str] = {0: "insertion_point", 1: "tube"}
Side = Literal["left", "right"]
Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


@dataclass(frozen=True, slots=True)
class Seed:
    insertion_box: Box
    tube_prior: list[tuple[float, float]]
    """Points along the expected tube path, starting at the nostril."""


def insertion_box(nostril_xy: tuple[float, float], face_w: float, *, scale: float = 0.15) -> Box:
    """Square box of side ``scale * face_w`` centred on the nostril."""
    half = 0.5 * scale * face_w
    x, y = nostril_xy
    return (x - half, y - half, x + half, y + half)


def tube_prior(nostril_xy: tuple[float, float], face_w: float, side: Side, *, n_points: int = 6,
               length_scale: float = 0.9, droop: float = 0.15) -> list[tuple[float, float]]:
    """Polyline from the nostril toward the ear on ``side`` (image coordinates, +x = right).

    The tape usually runs slightly downward along the cheek before rising to
    the ear, so the path gets a small quadratic droop.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    direction = 1.0 if side == "right" else -1.0
    x0, y0 = nostril_xy
    length = length_scale * face_w
    pts = []
    for i in range(n_points):
        a = i / (n_points - 1)
        x = x0 + direction * a * length
        y = y0 + droop * face_w * 4 * a * (1 - a)  # 0 at ends, max droop mid-way
        pts.append((x, y))
    return pts


def make_seed(nostril_xy: tuple[float, float], face_w: float, side: Side) -> Seed:
    return Seed(insertion_box(nostril_xy, face_w), tube_prior(nostril_xy, face_w, side))


class Segmenter(Protocol):
    def segment(self, image: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
        """Return a uint8 HxW mask (1 = tube) given positive point prompts on the tube."""
        ...


class NullSegmenter:
    """Draws the prior polyline as a thin mask. Lets the pipeline run without SAM."""

    def __init__(self, thickness_frac: float = 0.03) -> None:
        self.thickness_frac = thickness_frac

    def segment(self, image: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        if len(points) < 2:
            return mask
        span = max(1.0, math.dist(points[0], points[-1]))
        thickness = max(1, int(round(self.thickness_frac * span)))
        pts = np.array([[int(round(x)), int(round(y))] for x, y in points], np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=thickness, lineType=cv2.LINE_8)
        return mask


def mask_to_box(mask: np.ndarray) -> Box | None:
    """Tight box around all non-zero pixels, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0)


def clip_box(box: Box, w: int, h: int) -> Box:
    x0, y0, x1, y1 = box
    return (min(max(x0, 0.0), w), min(max(y0, 0.0), h), min(max(x1, 0.0), w), min(max(y1, 0.0), h))


def yolo_line(class_id: int, box: Box, w: int, h: int) -> str:
    """``class cx cy bw bh`` normalised to [0, 1]; the box is clipped to the image first."""
    if class_id not in CLASS_NAMES:
        raise ValueError(f"unknown class id {class_id}")
    x0, y0, x1, y1 = clip_box(box, w, h)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        raise ValueError("box has no area inside the image")
    cx, cy = x0 + bw / 2, y0 + bh / 2
    return f"{class_id} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def parse_yolo_line(line: str, w: int, h: int) -> tuple[int, Box]:
    cid, cx, cy, bw, bh = line.split()
    cx_, cy_, bw_, bh_ = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
    return int(cid), (cx_ - bw_ / 2, cy_ - bh_ / 2, cx_ + bw_ / 2, cy_ + bh_ / 2)


def seed_labels(image: np.ndarray, nostril_xy: tuple[float, float], face_w: float, side: Side,
                segmenter: Segmenter | None = None) -> list[str]:
    """Full seed step for one frame: YOLO lines for insertion point and tube."""
    h, w = image.shape[:2]
    seed = make_seed(nostril_xy, face_w, side)
    seg = segmenter or NullSegmenter()
    lines = [yolo_line(0, seed.insertion_box, w, h)]
    tube_box = mask_to_box(seg.segment(image, seed.tube_prior))
    if tube_box is not None:
        lines.append(yolo_line(1, tube_box, w, h))
    return lines
