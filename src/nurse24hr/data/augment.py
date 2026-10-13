"""Synthetic augmentation for the tube detector (TS-001 §7.5). numpy + OpenCV only.

Three ideas, in order of value:

* ``simulate_ir``: approximate the night IR camera before we own one.
  Grayscale, boosted contrast, sensor noise, slight blur, darkened background.
* ``copy_paste_tube``: paste a tube cutout from a labelled frame onto another
  face image, anchored at that face's nostril and scaled by its face width.
  Labels come for free. This is how we widen coverage beyond our volunteers.
* ``random_occlusion``: grey / dark rectangles standing in for a blanket or a
  hand, never covering more than a fraction of any box, so the detector learns
  partial views without the labels becoming lies.

``augment_dataset`` is a thin driver over a YOLO-layout folder
(``images/*.jpg`` + ``labels/*.txt``). Validation data must stay real; run
this on the training split only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from nurse24hr.data.seed_labels import Box, clip_box, parse_yolo_line, yolo_line

YoloBox = tuple[int, Box]


def simulate_ir(image: np.ndarray, rng: np.random.Generator, *, contrast: float = 1.4,
                noise_sigma: float = 8.0, blur_sigma: float = 0.8, background_gain: float = 0.6) -> np.ndarray:
    """Return a 3-channel grayscale image that looks like an IR night camera frame."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = float(gray.mean())
    gray = (gray - mean) * contrast + mean
    # IR illuminators fall off with distance: darken the darker half of the histogram further.
    dark = gray < mean
    gray[dark] *= background_gain
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    gray += rng.normal(0.0, noise_sigma, gray.shape).astype(np.float32)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _rotate_patch(patch: np.ndarray, mask: np.ndarray, angle_deg: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw = int(math_ceil(h * sin + w * cos))
    nh = int(math_ceil(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    p = cv2.warpAffine(patch, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    k = cv2.warpAffine(mask, m, (nw, nh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return p, k


def math_ceil(x: float) -> int:
    return int(-(-x // 1))


def copy_paste_tube(dst_image: np.ndarray, dst_nostril_xy: tuple[float, float], dst_face_w: float,
                    tube_patch: np.ndarray, tube_mask: np.ndarray, rng: np.random.Generator, *,
                    src_face_w: float | None = None, insertion_scale: float = 0.15,
                    scale_jitter: tuple[float, float] = (0.85, 1.15), max_rotation_deg: float = 20.0,
                    colour_jitter: float = 0.15) -> tuple[np.ndarray, list[YoloBox]]:
    """Paste ``tube_patch`` (BGR) where ``tube_mask`` (uint8, 1 = tube) is set.

    The patch is placed so that its left-most masked pixel column (the tube
    end nearest the insertion point, by convention) lands on the nostril. If
    ``src_face_w`` is given the patch is first rescaled by
    ``dst_face_w / src_face_w`` so tube thickness stays plausible.

    Returns the composited image and YOLO boxes (class 0 insertion point,
    class 1 tube) in destination pixel coordinates.
    """
    out = dst_image.copy()
    h, w = out.shape[:2]
    mask = (tube_mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        raise ValueError("tube_mask is empty")

    scale = float(rng.uniform(*scale_jitter))
    if src_face_w:
        scale *= dst_face_w / src_face_w
    angle = float(rng.uniform(-max_rotation_deg, max_rotation_deg))
    patch, mask = _rotate_patch(tube_patch, mask, angle, scale)
    if rng.random() < 0.5:
        patch, mask = cv2.flip(patch, 1), cv2.flip(mask, 1)

    # colour jitter on the patch only (tube colour / tape varies)
    gain = 1.0 + rng.uniform(-colour_jitter, colour_jitter, size=3)
    patch = np.clip(patch.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    ys, xs = np.nonzero(mask)
    anchor_x = xs.min()
    anchor_y = int(round(ys[xs == anchor_x].mean()))
    ox = int(round(dst_nostril_xy[0])) - anchor_x
    oy = int(round(dst_nostril_xy[1])) - anchor_y

    ph, pw = mask.shape
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + pw), min(h, oy + ph)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("patch lands entirely outside the destination image")
    sub_mask = mask[y0 - oy:y1 - oy, x0 - ox:x1 - ox].astype(bool)
    region = out[y0:y1, x0:x1]
    region[sub_mask] = patch[y0 - oy:y1 - oy, x0 - ox:x1 - ox][sub_mask]

    sy, sx = np.nonzero(sub_mask)
    tube_box: Box = (x0 + sx.min(), y0 + sy.min(), x0 + sx.max() + 1.0, y0 + sy.max() + 1.0)
    half = 0.5 * insertion_scale * dst_face_w
    nx, ny = dst_nostril_xy
    ins_box = clip_box((nx - half, ny - half, nx + half, ny + half), w, h)
    return out, [(0, ins_box), (1, clip_box(tube_box, w, h))]


def _overlap_frac(box: Box, rect: Box) -> float:
    bx0, by0, bx1, by1 = box
    rx0, ry0, rx1, ry1 = rect
    iw = max(0.0, min(bx1, rx1) - max(bx0, rx0))
    ih = max(0.0, min(by1, ry1) - max(by0, ry0))
    area = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return iw * ih / area


def random_occlusion(image: np.ndarray, boxes: list[YoloBox], rng: np.random.Generator, *,
                     max_rects: int = 2, max_cover: float = 0.6, attempts: int = 20) -> np.ndarray:
    """Add 0..max_rects grey/dark rectangles that cover at most ``max_cover`` of any box."""
    out = image.copy()
    h, w = out.shape[:2]
    n = int(rng.integers(0, max_rects + 1))
    for _ in range(n):
        for _ in range(attempts):
            rw, rh = rng.uniform(0.1, 0.5) * w, rng.uniform(0.1, 0.5) * h
            rx, ry = rng.uniform(0, w - rw), rng.uniform(0, h - rh)
            rect: Box = (rx, ry, rx + rw, ry + rh)
            if all(_overlap_frac(b, rect) <= max_cover for _, b in boxes):
                shade = int(rng.integers(20, 140))
                colour = (shade, shade, shade)
                cv2.rectangle(out, (int(rx), int(ry)), (int(rx + rw), int(ry + rh)), colour, -1)
                break
    return out


def _flip_boxes(boxes: list[YoloBox], w: int) -> list[YoloBox]:
    return [(c, (w - x1, y0, w - x0, y1)) for c, (x0, y0, x1, y1) in boxes]


def augment_image(image: np.ndarray, boxes: list[YoloBox], rng: np.random.Generator) -> tuple[np.ndarray, list[YoloBox]]:
    """One random augmentation of a labelled image: optional flip, IR sim, brightness, occlusion."""
    out, bxs = image, list(boxes)
    h, w = out.shape[:2]
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
        bxs = _flip_boxes(bxs, w)
    if rng.random() < 0.5:
        out = simulate_ir(out, rng)
    else:
        gain = float(rng.uniform(0.6, 1.3))
        out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.5, 1.5)))
    out = random_occlusion(out, bxs, rng)
    return out, bxs


def augment_dataset(in_dir: str | Path, out_dir: str | Path, n_per_image: int, rng: np.random.Generator) -> int:
    """Read ``in_dir/images/*.jpg`` + ``in_dir/labels/*.txt``; write ``n_per_image`` variants each.

    Returns the number of images written. Output uses the same YOLO layout.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    written = 0
    for img_path in sorted((in_dir / "images").glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = in_dir / "labels" / (img_path.stem + ".txt")
        boxes: list[YoloBox] = []
        if label_path.exists():
            boxes = [parse_yolo_line(ln, w, h) for ln in label_path.read_text().splitlines() if ln.strip()]
        for k in range(n_per_image):
            aug, bxs = augment_image(img, boxes, rng)
            stem = f"{img_path.stem}_aug{k:02d}"
            cv2.imwrite(str(out_dir / "images" / f"{stem}.jpg"), aug)
            (out_dir / "labels" / f"{stem}.txt").write_text("\n".join(yolo_line(c, b, w, h) for c, b in bxs) + ("\n" if bxs else ""))
            written += 1
    return written


def _main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Augment a YOLO-layout training folder.")
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=4, help="augmented copies per image")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n = augment_dataset(args.in_dir, args.out_dir, args.n, np.random.default_rng(args.seed))
    print(f"wrote {n} augmented images to {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover
    _main()
