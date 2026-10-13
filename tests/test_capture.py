"""Capture thread and frame slot tests. Headless: a synthetic video file, no webcam."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from nurse24hr.capture import CaptureThread, Frame, LatestFrame, parse_source
from nurse24hr.config import Thresholds
from nurse24hr.events import State
from nurse24hr.perception import NullPerceiver, crop_roi
from nurse24hr.app import Pipeline, build_pipeline, run
from nurse24hr.alarm import AlarmConfig, AlarmManager, NullSounder
from nurse24hr.notify import RecordingNotifier

W, H, FPS, N = 160, 120, 15, 45  # 3 s clip


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("video") / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    assert writer.isOpened(), "MJPG writer unavailable in this OpenCV build"
    for i in range(N):
        img = np.zeros((H, W, 3), np.uint8)
        img[:, :, 1] = int(255 * i / N)  # frame index encoded in green channel
        writer.write(img)
    writer.release()
    return path


def read_all(path: Path, **kw) -> list[Frame]:
    """Read every frame. Unpaced file replay applies back-pressure, so nothing is dropped."""
    slot = LatestFrame()
    cap = CaptureThread(str(path), slot, fps=FPS, realtime=False, **kw)
    cap.start()
    frames = []
    while True:
        f = slot.get(timeout=0.2)
        if f is not None:
            frames.append(f)
        elif cap.ended.is_set():
            break
    cap.join(timeout=2)
    assert cap.error is None
    return frames


def test_parse_source():
    assert parse_source("webcam") == 0
    assert parse_source("2") == 2
    assert parse_source(1) == 1
    assert parse_source("rtsp://x/y") == "rtsp://x/y"
    assert parse_source("a.mp4") == "a.mp4"


def test_file_source_reads_all_frames_with_file_clock(clip):
    frames = read_all(clip)
    assert len(frames) == N
    assert [f.seq for f in frames] == list(range(N))
    ts = [f.t for f in frames]
    assert ts[0] == 0.0
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert abs(ts[-1] - (N - 1) / FPS) < 1e-6
    assert frames[-1].image.shape == (H, W, 3)


def test_file_timestamps_are_deterministic_across_runs(clip):
    a = [f.t for f in read_all(clip)]
    b = [f.t for f in read_all(clip)]
    assert a == b


def test_unpaced_file_applies_backpressure_so_slow_consumer_sees_every_frame(clip):
    slot = LatestFrame()
    cap = CaptureThread(str(clip), slot, fps=FPS, realtime=False)
    cap.start()
    seqs = []
    while True:
        f = slot.get(timeout=0.5)
        if f is None:
            if cap.ended.is_set():
                break
            continue
        seqs.append(f.seq)
        time.sleep(0.01)  # slower than the decoder
    assert seqs == list(range(N))
    assert slot.dropped == 0


def test_paced_file_drops_stale_frames_for_slow_consumer(clip):
    slot = LatestFrame()
    cap = CaptureThread(str(clip), slot, fps=FPS, realtime=True)
    cap.start()
    cap.ended.wait(timeout=10)
    f = slot.get(timeout=0.5)
    assert f is not None and f.seq == N - 1
    assert slot.dropped == N - 1
    assert slot.get(timeout=0.05) is None  # nothing newer


def test_stop_unblocks_backpressured_producer(clip):
    slot = LatestFrame()
    cap = CaptureThread(str(clip), slot, fps=FPS, realtime=False)
    cap.start()
    assert slot.get(timeout=2) is not None  # take one, leave the producer blocked on the next
    time.sleep(0.1)
    cap.stop()
    cap.join(timeout=2)
    assert not cap.is_alive()


def test_realtime_pacing_roughly_matches_fps(clip):
    slot = LatestFrame()
    cap = CaptureThread(str(clip), slot, fps=FPS, realtime=True)
    t0 = time.monotonic()
    cap.start()
    cap.ended.wait(timeout=10)
    elapsed = time.monotonic() - t0
    expected = (N - 1) / FPS
    assert expected * 0.8 <= elapsed <= expected * 1.6 + 0.3


def test_stop_ends_thread_promptly(clip):
    slot = LatestFrame()
    cap = CaptureThread(str(clip), slot, fps=FPS, realtime=True, loop=True)
    cap.start()
    assert slot.get(timeout=2) is not None
    cap.stop()
    cap.join(timeout=2)
    assert not cap.is_alive()


def test_loop_restarts_file(clip):
    slot = LatestFrame()
    cap = CaptureThread(str(clip), slot, fps=FPS, realtime=False, loop=True)
    cap.start()
    deadline = time.monotonic() + 5
    while cap.frames_read < N + 5 and time.monotonic() < deadline:
        slot.get(timeout=0.1)  # keep consuming; unpaced files apply back-pressure
    cap.stop()
    cap.join(timeout=2)
    assert cap.frames_read >= N + 5


def test_missing_file_reports_error():
    slot = LatestFrame()
    cap = CaptureThread("/nonexistent/video.mp4", slot, realtime=False)
    cap.start()
    cap.ended.wait(timeout=5)
    assert cap.error and "cannot open" in cap.error


def test_crop_roi_clamps_to_image():
    img = np.zeros((H, W, 3), np.uint8)
    assert crop_roi(img, None) is img
    assert crop_roi(img, (10, 20, 60, 70)).shape == (50, 50, 3)
    assert crop_roi(img, (-10, -10, 10_000, 10_000)).shape == (H, W, 3)


def _pipeline(th: Thresholds, notifier: RecordingNotifier) -> Pipeline:
    alarm = AlarmManager(notifier=notifier, sounder=NullSounder(), cfg=AlarmConfig(audit_log=None))
    return Pipeline(thresholds=th, perceiver=NullPerceiver(), notifier=notifier, alarm=alarm, fps=FPS)


def test_headless_run_with_null_perceiver_faults_after_t_fault(clip):
    """No perception = no confident view = FAULT after t_fault. Fail loud by design."""
    events = []
    notes = RecordingNotifier()
    p = _pipeline(Thresholds(t_fault=1.0), notes)
    m = run(str(clip), p, realtime=False, headless=True, on_event=events.append)
    assert m.state is State.FAULT
    assert len(events) == 1 and events[0].state is State.FAULT
    assert abs(events[0].t - 1.0) < 3 / FPS
    # the alarm manager saw it and the state topic was published about once per second
    assert any(n["level"] == "fault" for n in notes.of("event"))
    assert 2 <= len(notes.of("state")) <= 4


def test_headless_run_max_frames(clip):
    p = _pipeline(Thresholds(), RecordingNotifier())
    m = run(str(clip), p, realtime=False, headless=True, max_frames=5, on_event=lambda e: None)
    assert m.state is State.IDLE


def test_build_pipeline_from_config_dict(tmp_path):
    cfg = {
        "thresholds": {"t_fault": 2.0},
        "alarm": {"audit_log": str(tmp_path / "audit.jsonl")},
        "recorder": {"enabled": True, "out_dir": str(tmp_path / "clips"), "pre_s": 1, "post_s": 1},
        "logging": {"features": str(tmp_path / "f.parquet")},
        "watchdog": {"heartbeat_s": 1.0},
    }
    p = build_pipeline(cfg, headless=True, perceiver=NullPerceiver())
    assert p.thresholds.t_fault == 2.0
    assert p.recorder is not None and p.logger is not None
    assert p.watchdog_cfg.heartbeat_s == 1.0
    p.close()
