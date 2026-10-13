"""Sample still frames from a staged session video for detector labeling (TS-001 §7.4 step 1).

Frames are taken at a low base rate everywhere and at a higher rate inside
labelled time ranges where the tube is handled (grasp, pull, caregiver): that
is where the tube is occluded or displaced, which is exactly where the
detector has to work. Labels come from the session script CSV
(``t_start,t_end,label``), so no manual annotation is needed to decide where
to sample densely.

Timestamps use ``frame_index / fps`` so sampling is deterministic and matches
the file clock used by :mod:`nurse24hr.capture`.

Usage::

    python -m nurse24hr.data.sample_frames session.mp4 session_labels.csv out/frames
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True, slots=True)
class LabelRange:
    t_start: float
    t_end: float
    label: str

    def contains(self, t: float) -> bool:
        return self.t_start <= t < self.t_end


@dataclass(frozen=True, slots=True)
class SampledFrame:
    path: Path
    t: float
    label: str
    """Label of the range containing ``t``, or ``"unlabeled"``."""


def read_labels(csv_path: str | Path) -> list[LabelRange]:
    out: list[LabelRange] = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(LabelRange(float(row["t_start"]), float(row["t_end"]), row["label"].strip()))
    return out


def label_at(t: float, ranges: Sequence[LabelRange]) -> str:
    for r in ranges:
        if r.contains(t):
            return r.label
    return "unlabeled"


def sample_times(duration: float, ranges: Sequence[LabelRange], *, base_rate_hz: float,
                 dense_labels: Iterable[str], dense_rate_hz: float) -> list[float]:
    """Return the sorted, de-duplicated sample times for a clip of ``duration`` seconds.

    Pure function so the density logic is testable without video.
    """
    dense = set(dense_labels)
    times: set[float] = set()
    if base_rate_hz > 0:
        step = 1.0 / base_rate_hz
        n = int(duration / step) + 1
        times.update(round(i * step, 6) for i in range(n) if i * step < duration)
    if dense_rate_hz > 0:
        step = 1.0 / dense_rate_hz
        for r in ranges:
            if r.label not in dense:
                continue
            t = r.t_start
            while t < min(r.t_end, duration):
                times.add(round(t, 6))
                t += step
    return sorted(times)


def sample_frames(video_path: str | Path, labels_csv: str | Path, out_dir: str | Path, *,
                  base_rate_hz: float = 1.0, dense_labels: Iterable[str] = ("grasp", "pull", "caregiver"),
                  dense_rate_hz: float = 5.0, jpeg_quality: int = 92) -> list[SampledFrame]:
    """Write sampled JPEGs and ``manifest.csv`` to ``out_dir``; return what was written."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranges = read_labels(labels_csv)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if n_frames > 0 else 0.0

    wanted = sample_times(duration, ranges, base_rate_hz=base_rate_hz, dense_labels=dense_labels,
                          dense_rate_hz=dense_rate_hz)
    # Map each wanted time to a frame index; several times may collapse onto one frame.
    wanted_idx = sorted({int(round(t * fps)) for t in wanted if int(round(t * fps)) < max(n_frames, 1)})

    out: list[SampledFrame] = []
    idx = 0
    next_pos = 0
    try:
        while next_pos < len(wanted_idx):
            ok, img = cap.read()
            if not ok:
                break
            if idx == wanted_idx[next_pos]:
                t = idx / fps
                ms = int(round(t * 1000))
                path = out_dir / f"{video_path.stem}_{ms:08d}.jpg"
                cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                out.append(SampledFrame(path, t, label_at(t, ranges)))
                next_pos += 1
            idx += 1
    finally:
        cap.release()

    with open(out_dir / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "t", "label"])
        for s in out:
            w.writerow([s.path.name, f"{s.t:.3f}", s.label])
    return out


def _main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Sample frames from a staged session for labeling.")
    ap.add_argument("video")
    ap.add_argument("labels_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--base-rate", type=float, default=1.0, help="samples per second everywhere")
    ap.add_argument("--dense-rate", type=float, default=5.0, help="samples per second inside dense labels")
    ap.add_argument("--dense-labels", default="grasp,pull,caregiver")
    args = ap.parse_args()
    frames = sample_frames(args.video, args.labels_csv, args.out_dir, base_rate_hz=args.base_rate,
                           dense_labels=args.dense_labels.split(","), dense_rate_hz=args.dense_rate)
    by_label: dict[str, int] = {}
    for f in frames:
        by_label[f.label] = by_label.get(f.label, 0) + 1
    print(f"wrote {len(frames)} frames to {args.out_dir}: {by_label}")


if __name__ == "__main__":  # pragma: no cover
    _main()
