"""Data pipeline tests (TS-001 §7): sampling, seed labels, augmentation, training stub, tube detector."""

from __future__ import annotations

import builtins
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from nurse24hr.data import augment, sample_frames as sf, seed_labels as sl, train_tube
from nurse24hr.features import Features
from nurse24hr.perception import NullPerceiver
from nurse24hr.tube_detector import Detection, FakeDetector, TubeDetectorPerceiver

W, H, FPS, N = 160, 120, 15, 150  # 10 s clip


@pytest.fixture(scope="module")
def session(tmp_path_factory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("session")
    video = d / "s01.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    assert w.isOpened()
    for i in range(N):
        img = np.zeros((H, W, 3), np.uint8)
        img[:, :, 2] = int(255 * i / N)
        w.write(img)
    w.release()
    labels = d / "s01_labels.csv"
    with open(labels, "w", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(["t_start", "t_end", "label"])
        cw.writerow([0.0, 4.0, "idle"])
        cw.writerow([4.0, 6.0, "grasp"])
        cw.writerow([6.0, 7.0, "pull"])
        cw.writerow([7.0, 10.0, "scratch"])
    return video, labels


# ------------------------------------------------------------- sampling

def test_sample_times_density():
    ranges = [sl_range(0, 4, "idle"), sl_range(4, 6, "grasp"), sl_range(6, 7, "pull"), sl_range(7, 10, "scratch")]
    times = sf.sample_times(10.0, ranges, base_rate_hz=1.0, dense_labels=("grasp", "pull"), dense_rate_hz=5.0)
    assert times == sorted(set(times))
    in_dense = [t for t in times if 4.0 <= t < 7.0]
    outside = [t for t in times if not (4.0 <= t < 7.0)]
    assert len(in_dense) == 15  # 3 s * 5 Hz (base samples coincide with dense ones)
    assert len(outside) == 7  # 0,1,2,3,7,8,9


def sl_range(a, b, label):
    return sf.LabelRange(float(a), float(b), label)


def test_sample_frames_writes_jpegs_and_manifest(session, tmp_path):
    video, labels = session
    out = sf.sample_frames(video, labels, tmp_path / "frames")
    assert len(out) == 22
    assert all(f.path.exists() and f.path.suffix == ".jpg" for f in out)
    assert out[0].path.name == "s01_00000000.jpg"
    by_label = {}
    for f in out:
        by_label[f.label] = by_label.get(f.label, 0) + 1
    assert by_label["grasp"] == 10 and by_label["pull"] == 5
    assert by_label["idle"] == 4 and by_label["scratch"] == 3
    rows = list(csv.DictReader(open(tmp_path / "frames" / "manifest.csv")))
    assert len(rows) == 22 and set(rows[0]) == {"path", "t", "label"}
    img = cv2.imread(str(out[-1].path))
    assert img.shape == (H, W, 3)


def test_sample_frames_missing_video(tmp_path, session):
    with pytest.raises(RuntimeError):
        sf.sample_frames(tmp_path / "nope.avi", session[1], tmp_path / "x")


# ---------------------------------------------------------- seed labels

def test_insertion_box_geometry():
    box = sl.insertion_box((100.0, 60.0), 80.0)
    assert box == (94.0, 54.0, 106.0, 66.0)


def test_tube_prior_direction_and_length():
    right = sl.tube_prior((100.0, 60.0), 80.0, "right")
    left = sl.tube_prior((100.0, 60.0), 80.0, "left")
    assert right[0] == (100.0, 60.0) and left[0] == (100.0, 60.0)
    assert right[-1][0] > 100 and left[-1][0] < 100
    assert abs(right[-1][0] - 100.0) == pytest.approx(0.9 * 80.0)
    assert right[-1][1] == pytest.approx(60.0)  # droop returns to zero at the end
    assert max(y for _, y in right) > 60.0  # droops downward (image coords) mid-way
    with pytest.raises(ValueError):
        sl.tube_prior((0, 0), 10, "up")  # type: ignore[arg-type]


def test_null_segmenter_mask_to_box_covers_prior():
    img = np.zeros((H, W, 3), np.uint8)
    prior = sl.tube_prior((40.0, 50.0), 60.0, "right")
    mask = sl.NullSegmenter().segment(img, prior)
    assert mask.dtype == np.uint8 and mask.shape == (H, W) and mask.max() == 1
    box = sl.mask_to_box(mask)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 <= 40 <= x1 and x0 <= prior[-1][0] <= x1
    assert sl.mask_to_box(np.zeros((H, W), np.uint8)) is None


def test_yolo_line_format_and_bounds():
    line = sl.yolo_line(1, (-10.0, 10.0, 90.0, 70.0), W, H)  # clipped at x=0
    cid, cx, cy, bw, bh = line.split()
    assert cid == "1"
    vals = [float(v) for v in (cx, cy, bw, bh)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert vals[0] == pytest.approx(45 / W) and vals[2] == pytest.approx(90 / W)
    _, box = sl.parse_yolo_line(line, W, H)
    assert box == pytest.approx((0.0, 10.0, 90.0, 70.0), abs=1e-3)
    with pytest.raises(ValueError):
        sl.yolo_line(2, (0, 0, 10, 10), W, H)
    with pytest.raises(ValueError):
        sl.yolo_line(0, (-20, -20, -10, -10), W, H)


def test_seed_labels_end_to_end():
    img = np.zeros((H, W, 3), np.uint8)
    lines = sl.seed_labels(img, (60.0, 50.0), 50.0, "right")
    assert len(lines) == 2
    assert lines[0].startswith("0 ") and lines[1].startswith("1 ")


# ---------------------------------------------------------- augmentation

def _face_image() -> np.ndarray:
    img = np.full((H, W, 3), (60, 80, 120), np.uint8)
    cv2.circle(img, (80, 60), 35, (150, 170, 210), -1)
    return img


def test_simulate_ir_shape_dtype_determinism():
    img = _face_image()
    a = augment.simulate_ir(img, np.random.default_rng(1))
    b = augment.simulate_ir(img, np.random.default_rng(1))
    c = augment.simulate_ir(img, np.random.default_rng(2))
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a[:, :, 0], a[:, :, 1]) and np.array_equal(a[:, :, 1], a[:, :, 2])
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def _tube_patch() -> tuple[np.ndarray, np.ndarray]:
    patch = np.zeros((20, 60, 3), np.uint8)
    mask = np.zeros((20, 60), np.uint8)
    cv2.line(patch, (0, 10), (59, 10), (230, 230, 230), 3)
    cv2.line(mask, (0, 10), (59, 10), 1, 3)
    return patch, mask


def test_copy_paste_anchors_at_nostril_and_boxes_inside():
    rng = np.random.default_rng(0)
    nostril = (75.0, 65.0)
    for _ in range(10):
        out, boxes = augment.copy_paste_tube(_face_image(), nostril, 70.0, *_tube_patch(), rng)
        assert out.shape == (H, W, 3)
        assert [c for c, _ in boxes] == [0, 1]
        for _, (x0, y0, x1, y1) in boxes:
            assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
        _, (tx0, ty0, tx1, ty1) = boxes[1]
        # the tube box must touch the nostril neighbourhood
        assert tx0 - 3 <= nostril[0] <= tx1 + 3 and ty0 - 3 <= nostril[1] <= ty1 + 3
        assert not np.array_equal(out, _face_image())
    with pytest.raises(ValueError):
        augment.copy_paste_tube(_face_image(), nostril, 70.0, _tube_patch()[0], np.zeros((20, 60), np.uint8), rng)


def test_random_occlusion_respects_cover_limit():
    img = _face_image()
    boxes = [(0, (70.0, 55.0, 90.0, 75.0)), (1, (75.0, 60.0, 140.0, 70.0))]
    for seed in range(30):
        rng = np.random.default_rng(seed)
        out = augment.random_occlusion(img, boxes, rng, max_rects=2, max_cover=0.6)
        assert out.shape == img.shape
        for _, (x0, y0, x1, y1) in boxes:
            region = out[int(y0):int(y1), int(x0):int(x1)]
            orig = img[int(y0):int(y1), int(x0):int(x1)]
            changed = np.any(region != orig, axis=2).mean()
            assert changed <= 0.6 + 0.05


def test_augment_dataset_writes_n_per_image(tmp_path):
    src = tmp_path / "train"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir()
    for i in range(3):
        cv2.imwrite(str(src / "images" / f"f{i}.jpg"), _face_image())
        (src / "labels" / f"f{i}.txt").write_text(sl.yolo_line(0, (70, 55, 90, 75), W, H) + "\n" + sl.yolo_line(1, (75, 60, 140, 70), W, H) + "\n")
    n = augment.augment_dataset(src, tmp_path / "aug", 4, np.random.default_rng(0))
    assert n == 12
    imgs = sorted((tmp_path / "aug" / "images").glob("*.jpg"))
    labels = sorted((tmp_path / "aug" / "labels").glob("*.txt"))
    assert len(imgs) == 12 and len(labels) == 12
    for lb in labels:
        lines = lb.read_text().splitlines()
        assert len(lines) == 2
        for ln in lines:
            cid, *vals = ln.split()
            assert cid in ("0", "1") and all(0 <= float(v) <= 1 for v in vals)


# -------------------------------------------------------------- training

def test_write_dataset_yaml(tmp_path):
    p = train_tube.write_dataset_yaml(tmp_path / "ds")
    text = p.read_text()
    assert "train/images" in text and "val/images" in text and "insertion_point" in text


def test_train_raises_clear_import_error_without_ultralytics(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ImportError("no module")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="uv add ultralytics"):
        train_tube.train(train_tube.TrainConfig(tmp_path))
    with pytest.raises(ImportError, match="uv add ultralytics"):
        train_tube.export_coreml(tmp_path / "w.pt")


# --------------------------------------------------------- tube detector

class StubBase:
    """Base perceiver that returns a fixed wrist / nostril / face width."""

    name = "stub"

    def __init__(self, wrist=(130.0, 60.0), nostril=(80.0, 60.0), face_w=50.0):
        self.wrist, self.nostril, self.face_w = wrist, nostril, face_w

    def process(self, image, t):
        d = np.hypot(self.wrist[0] - self.nostril[0], self.wrist[1] - self.nostril[1]) / self.face_w
        return Features(t=t, d=d, conf=0.9, wrist_r=self.wrist, tube_pt=self.nostril, face_w=self.face_w)

    def draw(self, image, f):
        return None


def test_pass_through_without_weights(tmp_path):
    base = StubBase()
    p = TubeDetectorPerceiver(tmp_path / "missing.pt", base)
    assert p.detector is None and "no tube detector" in p.name
    f = p.process(np.zeros((H, W, 3), np.uint8), 0.0)
    assert f == base.process(None, 0.0)
    assert f.tube_disp is None
    p.draw(np.zeros((H, W, 3), np.uint8), f)  # must not raise


def test_pass_through_with_null_base(tmp_path):
    p = TubeDetectorPerceiver(None, NullPerceiver())
    f = p.process(np.zeros((H, W, 3), np.uint8), 1.0)
    assert f.conf == 0.0 and f.d is None


def test_tube_disp_rises_when_anchor_moves():
    img = np.zeros((H, W, 3), np.uint8)
    steady = [[Detection(0, 0.9, (78, 58, 82, 62))]] * 45  # 3 s at 15 fps, anchor at (80, 60)
    moved = [[Detection(0, 0.9, (78 + 25, 58, 82 + 25, 62))]] * 15  # anchor jumps 25 px (0.5 face widths)
    none = [[]] * 3
    det = FakeDetector(steady + moved + none)
    p = TubeDetectorPerceiver(None, StubBase(), detector=det, warmup_s=2.0)
    assert "tube detector" in p.name and "no" not in p.name
    disps = []
    feats = []
    for i in range(45 + 15 + 3):
        f = p.process(img, i / 15)
        feats.append(f)
        disps.append(f.tube_disp)
    assert disps[0] is None  # first frame establishes the baseline
    assert all(d is not None and d < 0.02 for d in disps[1:45])
    assert feats[44].tube_pt == (80.0, 60.0)
    # right after the jump, displacement is ~0.5 face widths and decays only slowly (slow alpha)
    assert disps[45] == pytest.approx(0.5, abs=0.02)
    assert disps[59] > 0.45
    assert feats[45].tube_pt == (105.0, 60.0)
    # d is recomputed against the detected anchor: wrist (130,60) is now 25 px = 0.5 fw away
    assert feats[45].d == pytest.approx(0.5)
    # with no detection the base features pass through untouched
    assert feats[-1].tube_disp is None and feats[-1].tube_pt == (80.0, 60.0)


def test_tube_box_used_when_no_insertion_point_and_low_conf_ignored():
    det = FakeDetector([[Detection(1, 0.8, (90, 55, 150, 65)), Detection(0, 0.1, (0, 0, 4, 4))]])
    p = TubeDetectorPerceiver(None, StubBase(), detector=det, min_conf=0.4)
    f = p.process(np.zeros((H, W, 3), np.uint8), 0.0)
    assert f.tube_pt == (120.0, 60.0)


def test_reset_baseline_after_reseat():
    det = FakeDetector([[Detection(0, 0.9, (78, 58, 82, 62))]] * 10 + [[Detection(0, 0.9, (108, 58, 112, 62))]] * 10)
    p = TubeDetectorPerceiver(None, StubBase(), detector=det)
    img = np.zeros((H, W, 3), np.uint8)
    for i in range(10):
        p.process(img, i / 15)
    f = p.process(img, 10 / 15)
    assert f.tube_disp == pytest.approx(0.6, abs=0.05)
    p.reset_baseline()
    f = p.process(img, 11 / 15)
    assert f.tube_disp is None
    f = p.process(img, 12 / 15)
    assert f.tube_disp == pytest.approx(0.0, abs=0.01)
