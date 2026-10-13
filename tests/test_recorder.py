"""Clip recorder tests. Headless, synthetic 64x48 frames with a file clock."""

import json
import time

import cv2
import numpy as np
import pytest

from nurse24hr import recorder as rec_mod
from nurse24hr.capture import Frame
from nurse24hr.events import Event, State
from nurse24hr.recorder import Recorder

FPS = 15
W, H = 64, 48


def frame(seq: int) -> Frame:
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :, 2] = seq % 256
    return Frame(seq=seq, t=seq / FPS, image=img)


def feed(rec: Recorder, start: int, n: int, features: bool = False) -> int:
    for seq in range(start, start + n):
        feats = {"d": 0.2, "conf": 0.9, "state": "WATCH", "ignored": 1} if features else None
        rec.push(frame(seq), feats)
    return start + n


def ev(t: float, state: State = State.ALARM, prev: State = State.WARN, reason: str = "test") -> Event:
    return Event(t, "transition", prev, state, reason)


def wait_idle(rec: Recorder, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while rec.pending and time.monotonic() < deadline:
        time.sleep(0.02)
    assert rec.pending == 0, "writer did not drain"


def count_frames(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


# ------------------------------------------------------------ pre-buffer

def test_prebuffer_is_time_based_and_bounded(tmp_path):
    rec = Recorder(tmp_path, pre_s=2.0, post_s=1.0, fps=FPS)
    feed(rec, 0, 10 * FPS)  # 10 s of frames
    assert rec.buffered <= int(2.0 * FPS * 1.5)
    oldest = rec._ring[0][0].t
    newest = rec._ring[-1][0].t
    assert newest - oldest <= 2.0 + 1e-9
    assert newest - oldest >= 2.0 - 2 / FPS
    rec.close()


def test_prebuffer_stores_references_not_copies(tmp_path):
    rec = Recorder(tmp_path, pre_s=1.0, post_s=1.0, fps=FPS)
    f = frame(0)
    rec.push(f)
    assert rec._ring[0][0].image is f.image
    rec.close()


# ------------------------------------------------------------ writing

def test_trigger_writes_clip_with_pre_and_post_frames(tmp_path):
    rec = Recorder(tmp_path, pre_s=2.0, post_s=1.0, fps=FPS)
    nxt = feed(rec, 0, 5 * FPS)                     # 5 s, buffer holds last 2 s
    clip_id = rec.trigger(ev(nxt / FPS))
    assert clip_id.startswith(f"{int(nxt / FPS * 1000):012d}_ALARM_")
    assert rec.active
    feed(rec, nxt, 3 * FPS)                          # 3 s more, post-roll is 1 s
    assert not rec.active
    rec.close()
    wait_idle(rec)

    assert rec.errors == []
    assert len(rec.written) == 1
    path = rec.written[0]
    assert path.exists() and path.stat().st_size > 0
    side = json.loads((tmp_path / f"{clip_id}.json").read_text())
    expected = int(2.0 * FPS) + int(1.0 * FPS)
    assert abs(side["n_frames"] - expected) <= 3
    assert abs(count_frames(path) - side["n_frames"]) <= 2
    assert side["clip_id"] == clip_id
    assert side["event"]["state"] == "ALARM" and side["event"]["prev"] == "WARN"
    assert side["event"]["reason"] == "test"
    assert side["path"] == str(path)
    assert side["fps"] == FPS
    assert side["t_end"] - side["t_start"] == pytest.approx(3.0, abs=3 / FPS)
    assert side["t_end"] >= nxt / FPS + 1.0 - 1 / FPS
    assert len(side["events"]) == 1
    assert side["features"] == []
    assert side["extra"] == {}


def test_sidecar_carries_extra_and_features(tmp_path):
    rec = Recorder(tmp_path, pre_s=1.0, post_s=0.5, fps=FPS)
    nxt = feed(rec, 0, FPS, features=True)
    rec.trigger(ev(nxt / FPS, State.WARN, State.WATCH, "grasp"), extra={"bed": "bed-01"})
    feed(rec, nxt, FPS, features=True)
    rec.close()
    wait_idle(rec)
    side = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert side["extra"] == {"bed": "bed-01"}
    assert side["n_frames"] == len(side["features"])
    assert set(side["features"][0]) == {"t", "d", "conf", "state"}
    assert side["features"][0]["t"] == side["t_start"]
    assert side["features"][-1]["t"] == side["t_end"]


def test_second_trigger_during_postroll_extends_one_incident(tmp_path):
    rec = Recorder(tmp_path, pre_s=1.0, post_s=1.0, fps=FPS)
    nxt = feed(rec, 0, 2 * FPS)
    a = rec.trigger(ev(nxt / FPS, State.WARN, State.WATCH, "grasp"))
    nxt = feed(rec, nxt, int(0.5 * FPS))
    b = rec.trigger(ev(nxt / FPS, State.ALARM, State.WARN, "pull"), extra={"x": 1})
    assert a == b
    nxt = feed(rec, nxt, int(0.8 * FPS))
    assert rec.active, "post-roll should have been extended past the original end"
    feed(rec, nxt, FPS)
    assert not rec.active
    rec.close()
    wait_idle(rec)

    assert len(rec.written) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1
    side = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert [e["state"] for e in side["events"]] == ["WARN", "ALARM"]
    assert side["event"]["state"] == "WARN"
    assert side["extra"] == {"x": 1}
    # 1 s pre + 0.5 s + 1 s post from the second trigger
    assert abs(side["n_frames"] - int(2.5 * FPS)) <= 3


def test_separate_incidents_make_separate_clips(tmp_path):
    rec = Recorder(tmp_path, pre_s=0.5, post_s=0.5, fps=FPS)
    nxt = feed(rec, 0, FPS)
    rec.trigger(ev(nxt / FPS))
    nxt = feed(rec, nxt, 2 * FPS)
    rec.trigger(ev(nxt / FPS))
    feed(rec, nxt, 2 * FPS)
    rec.close()
    wait_idle(rec)
    assert len(rec.written) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_close_flushes_active_incident(tmp_path):
    rec = Recorder(tmp_path, pre_s=1.0, post_s=10.0, fps=FPS)
    nxt = feed(rec, 0, FPS)
    rec.trigger(ev(nxt / FPS))
    feed(rec, nxt, 3)  # post-roll far from complete
    rec.close()
    assert rec.pending == 0
    assert len(rec.written) == 1
    side = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert side["n_frames"] == FPS + 3
    assert rec.errors == []


def test_push_and_trigger_after_close_are_ignored(tmp_path):
    rec = Recorder(tmp_path, pre_s=1.0, post_s=1.0, fps=FPS)
    rec.close()
    rec.push(frame(0))
    assert rec.trigger(ev(0.0)) == ""
    assert rec.buffered == 0


def test_disabled_recorder_is_noop(tmp_path):
    rec = Recorder(tmp_path, enabled=False)
    feed(rec, 0, 30)
    assert rec.buffered == 0
    assert rec.trigger(ev(2.0)) == ""
    assert not rec.active
    rec.close()
    assert list(tmp_path.iterdir()) == []
    assert rec.errors == []


def test_clip_ids_are_unique_for_same_timestamp(tmp_path):
    rec = Recorder(tmp_path, pre_s=0.5, post_s=0.2, fps=FPS)
    ids = set()
    nxt = 0
    for _ in range(3):
        nxt = feed(rec, nxt, 5)
        ids.add(rec.trigger(ev(1.0)))
        nxt = feed(rec, nxt, FPS)
    rec.close()
    wait_idle(rec)
    assert len(ids) == 3


# ------------------------------------------------------------ fallback / errors

def test_writer_falls_back_when_first_fourccs_fail(tmp_path, monkeypatch):
    real_writer = cv2.VideoWriter
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    tried = []

    class Flaky:
        def __init__(self, path, fourcc, fps, size):
            tried.append(fourcc)
            self._w = real_writer(path, fourcc, fps, size) if fourcc == mjpg else None

        def isOpened(self):
            return self._w is not None and self._w.isOpened()

        def write(self, img):
            self._w.write(img)

        def release(self):
            if self._w is not None:
                self._w.release()

    monkeypatch.setattr(rec_mod.cv2, "VideoWriter", Flaky)
    rec = Recorder(tmp_path, pre_s=0.5, post_s=0.5, fps=FPS)
    nxt = feed(rec, 0, FPS)
    rec.trigger(ev(nxt / FPS))
    feed(rec, nxt, FPS)
    rec.close()
    wait_idle(rec)

    assert len(tried) == 3
    assert rec.errors == []
    assert len(rec.written) == 1
    assert rec.written[0].suffix == ".avi"
    assert not list(tmp_path.glob("*.mp4")), "failed attempts must not leave empty files"
    side = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert side["path"].endswith(".avi")


def test_write_failure_is_captured_not_raised(tmp_path, monkeypatch):
    class Broken:
        def __init__(self, *a, **k):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(rec_mod.cv2, "VideoWriter", Broken)
    rec = Recorder(tmp_path, pre_s=0.5, post_s=0.2, fps=FPS)
    nxt = feed(rec, 0, FPS)
    rec.trigger(ev(nxt / FPS))
    feed(rec, nxt, FPS)
    rec.close()
    wait_idle(rec)
    assert rec.written == []
    assert len(rec.errors) == 1 and "no video writer" in rec.errors[0]
    assert not list(tmp_path.glob("*.json"))


def test_push_errors_are_captured_not_raised(tmp_path):
    rec = Recorder(tmp_path, pre_s=1.0, post_s=1.0, fps=FPS)
    rec.push("not a frame")  # type: ignore[arg-type]
    assert len(rec.errors) == 1 and rec.errors[0].startswith("push:")
    # recorder keeps working afterwards
    feed(rec, 0, 5)
    assert rec.buffered == 5
    rec.close()


def test_trigger_errors_are_captured_not_raised(tmp_path):
    rec = Recorder(tmp_path, pre_s=1.0, post_s=1.0, fps=FPS)
    feed(rec, 0, 5)
    assert rec.trigger("not an event") == ""  # type: ignore[arg-type]
    assert len(rec.errors) == 1 and rec.errors[0].startswith("trigger:")
    assert not rec.active
    rec.close()
