"""Tube detector perceiver (TS-001 §5.2 v2): wraps a base perceiver and adds ``tube_disp``.

The base perceiver (MediaPipe hands + face) produces ``Features`` whose
``tube_pt`` is the nostril proxy and whose ``face_w`` is the normalisation
scale. This wrapper runs a trained YOLO detector and:

* replaces ``tube_pt`` with the detected insertion point (class 0), or the
  tube box centre (class 1) if the insertion point is not seen;
* computes ``tube_disp``: distance of that anchor from a rolling baseline,
  divided by ``face_w``. The baseline is an EMA that learns quickly over the
  first ``warmup_s`` seconds of confident detections and then drifts slowly,
  so a pulled tube registers as displacement rather than becoming the new
  normal within a few frames.

If the weights file is missing or ``ultralytics`` is not installed, the
wrapper is a transparent pass-through (logged once), so v1 keeps working with
the nostril proxy and ``tube_disp = None``.

The ultralytics call sits behind the :class:`Detector` protocol so tests use
:class:`FakeDetector`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from nurse24hr.features import Features, Point
from nurse24hr.perception import Perceiver

log = logging.getLogger(__name__)

CLASS_INSERTION = 0
CLASS_TUBE = 1


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    conf: float
    box: tuple[float, float, float, float]
    """x0, y0, x1, y1 in pixels."""

    @property
    def centre(self) -> Point:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2, (y0 + y1) / 2)


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class FakeDetector:
    """Returns a scripted list of detections per call. For tests."""

    def __init__(self, script: list[list[Detection]]) -> None:
        self.script = script
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        out = self.script[min(self.calls, len(self.script) - 1)] if self.script else []
        self.calls += 1
        return out


class UltralyticsDetector:
    """Thin adapter over ``ultralytics.YOLO``. Imported lazily."""

    def __init__(self, weights: str | Path, conf: float = 0.4, imgsz: int = 640, device: str | None = None) -> None:
        from ultralytics import YOLO  # type: ignore[import-not-found]

        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
        out: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for cid, cf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                out.append(Detection(int(cid), float(cf), (xyxy[0], xyxy[1], xyxy[2], xyxy[3])))
        return out


def load_detector(weights_path: str | Path | None) -> Detector | None:
    """Return a real detector if weights exist and ultralytics imports; otherwise None."""
    if not weights_path or not Path(weights_path).exists():
        log.info("tube detector weights not found (%s); using insertion-point proxy", weights_path)
        return None
    try:
        return UltralyticsDetector(weights_path)
    except ImportError:
        log.warning("ultralytics not installed; tube detector disabled (uv add ultralytics)")
        return None


class TubeDetectorPerceiver:
    """Perceiver that augments a base perceiver's Features with a detected tube anchor and tube_disp."""

    def __init__(self, weights_path: str | Path | None, base: Perceiver, *, detector: Detector | None = None,
                 min_conf: float = 0.4, warmup_s: float = 2.0, fast_alpha: float = 0.3,
                 slow_alpha: float = 0.005, every_n: int = 1) -> None:
        self.base = base
        self.detector = detector if detector is not None else load_detector(weights_path)
        self.min_conf = min_conf
        self.warmup_s = warmup_s
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.every_n = max(1, every_n)
        self.name = f"{base.name} + tube detector" if self.detector else f"{base.name} (no tube detector)"
        self._baseline: Point | None = None
        self._first_t: float | None = None
        self._frames = 0
        self._last_anchor: Point | None = None
        self._last_dets: list[Detection] = []

    # ------------------------------------------------------------ perceiver

    def process(self, image: np.ndarray, t: float) -> Features:
        f = self.base.process(image, t)
        if self.detector is None:
            return f
        self._frames += 1
        if (self._frames - 1) % self.every_n == 0:
            self._last_dets = self.detector.detect(image)
        anchor = self._pick_anchor(self._last_dets)
        self._last_anchor = anchor
        if anchor is None:
            return f
        disp = self._update_baseline(anchor, t)
        tube_disp = None
        if disp is not None and f.face_w:
            tube_disp = disp / f.face_w
        d = f.d
        if f.face_w and (f.wrist_l is not None or f.wrist_r is not None):
            dists = [math.dist(w, anchor) for w in (f.wrist_l, f.wrist_r) if w is not None]
            d = min(dists) / f.face_w
        return replace(f, tube_pt=anchor, tube_disp=tube_disp, d=d)

    def draw(self, image: np.ndarray, f: Features) -> None:
        self.base.draw(image, f)
        for det in self._last_dets:
            x0, y0, x1, y1 = (int(round(v)) for v in det.box)
            colour = (0, 255, 255) if det.class_id == CLASS_INSERTION else (255, 128, 0)
            cv2.rectangle(image, (x0, y0), (x1, y1), colour, 1)
        if self._baseline is not None and self._last_anchor is not None:
            bx, by = (int(round(v)) for v in self._baseline)
            ax, ay = (int(round(v)) for v in self._last_anchor)
            cv2.circle(image, (bx, by), 4, (255, 255, 255), 1)
            cv2.line(image, (bx, by), (ax, ay), (0, 0, 255), 1)

    # -------------------------------------------------------------- helpers

    def _pick_anchor(self, dets: list[Detection]) -> Point | None:
        good = [d for d in dets if d.conf >= self.min_conf]
        ins = [d for d in good if d.class_id == CLASS_INSERTION]
        if ins:
            return max(ins, key=lambda d: d.conf).centre
        tube = [d for d in good if d.class_id == CLASS_TUBE]
        if tube:
            return max(tube, key=lambda d: d.conf).centre
        return None

    def _update_baseline(self, anchor: Point, t: float) -> float | None:
        """EMA baseline; returns pixel displacement of ``anchor`` from it (None on the first frame)."""
        if self._baseline is None:
            self._baseline = anchor
            self._first_t = t
            return None
        assert self._first_t is not None
        alpha = self.fast_alpha if (t - self._first_t) < self.warmup_s else self.slow_alpha
        disp = math.dist(anchor, self._baseline)
        bx, by = self._baseline
        self._baseline = (bx + alpha * (anchor[0] - bx), by + alpha * (anchor[1] - by))
        return disp

    def reset_baseline(self) -> None:
        """Call after a nurse re-seats the tube and acknowledges, so the new position is normal."""
        self._baseline = None
        self._first_t = None
