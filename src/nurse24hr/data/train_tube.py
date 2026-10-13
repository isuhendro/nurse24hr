"""Train and export the YOLO-nano tube detector (TS-001 §5.2 v2, §7).

Expected dataset layout (YOLO)::

    <data>/
      train/images/*.jpg   train/labels/*.txt
      val/images/*.jpg     val/labels/*.txt      # real staged frames only, never synthetic

Classes: ``0 insertion_point``, ``1 tube`` (see :mod:`nurse24hr.data.seed_labels`).

``ultralytics`` is imported lazily so this module (and the rest of the data
package) imports on machines without it. Install with ``uv add ultralytics``.

Usage::

    python -m nurse24hr.data.train_tube --data datasets/tube --epochs 50 --model yolo11n.pt --device mps
    python -m nurse24hr.data.train_tube --export runs/tube/weights/best.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from nurse24hr.data.seed_labels import CLASS_NAMES

INSTALL_HINT = "ultralytics is not installed. Run `uv add ultralytics` (needs PyTorch; use --device mps on Apple Silicon)."


@dataclass(frozen=True, slots=True)
class TrainConfig:
    data_dir: Path
    epochs: int = 50
    imgsz: int = 640
    model: str = "yolo11n.pt"
    device: str = "mps"
    batch: int = 16
    project: Path = Path("runs")
    name: str = "tube"


def write_dataset_yaml(data_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """Write the ultralytics dataset YAML next to the data (or at ``out_path``)."""
    data_dir = Path(data_dir).resolve()
    out_path = Path(out_path) if out_path else data_dir / "tube.yaml"
    doc = {
        "path": str(data_dir),
        "train": "train/images",
        "val": "val/images",
        "names": {int(k): v for k, v in CLASS_NAMES.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out_path


def _import_ultralytics():
    try:
        import ultralytics  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return ultralytics


def train(cfg: TrainConfig) -> Path:
    """Fine-tune the detector. Returns the path to ``best.pt``."""
    ultralytics = _import_ultralytics()
    data_yaml = write_dataset_yaml(cfg.data_dir)
    model = ultralytics.YOLO(cfg.model)
    model.train(data=str(data_yaml), epochs=cfg.epochs, imgsz=cfg.imgsz, device=cfg.device, batch=cfg.batch,
                project=str(cfg.project), name=cfg.name, exist_ok=True)
    return Path(cfg.project) / cfg.name / "weights" / "best.pt"


def export_coreml(weights: str | Path, imgsz: int = 640) -> Path:
    """Export trained weights to CoreML for the iPhone path. Returns the exported path."""
    ultralytics = _import_ultralytics()
    model = ultralytics.YOLO(str(weights))
    out = model.export(format="coreml", imgsz=imgsz, nms=True)
    return Path(out)


def _main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Train / export the tube detector.")
    ap.add_argument("--data", help="dataset root with train/ and val/")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--export", metavar="WEIGHTS", help="export these weights to CoreML instead of training")
    args = ap.parse_args()
    if args.export:
        print(export_coreml(args.export, args.imgsz))
        return
    if not args.data:
        ap.error("--data is required for training")
    best = train(TrainConfig(Path(args.data), args.epochs, args.imgsz, args.model, args.device, args.batch))
    print(f"best weights: {best}")


if __name__ == "__main__":  # pragma: no cover
    _main()
