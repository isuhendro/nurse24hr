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
from nurse24hr.app import run

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


def test_slow_consumer_sees_newest_frame_and_drops_stale(clip):
    slot = LatestFrame()
    cap = CaptureThread(str(clip), slot, fps=FPS, realtime=False)
    cap.start()
    cap.ended.wait(timeout=5)
    f = slot.get(timeout=0.5)
    assert f is not None and f.seq == N - 1
    assert slot.dropped == N - 1
    assert slot.get(timeout=0.05) is None  # nothing newer


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
        time.sleep(0.02)
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


def test_headless_run_with_null_perceiver_faults_after_t_fault(clip):
    """No perception = no confident view = FAULT after t_fault. Fail loud by design."""
    events = []
    th = Thresholds(t_fault=1.0)
    m = run(str(clip), thresholds=th, perceiver=NullPerceiver(), roi=None, fps=FPS, realtime=False,
            headless=True, loop=False, on_event=events.append)
    assert m.state is State.FAULT
    assert len(events) == 1 and events[0].state is State.FAULT
    assert abs(events[0].t - 1.0) < 3 / FPS


def test_headless_run_max_frames(clip):
    m = run(str(clip), thresholds=Thresholds(), perceiver=NullPerceiver(), roi=None, fps=FPS, realtime=False,
            headless=True, loop=False, max_frames=5, on_event=lambda e: None)
    assert m.state is State.IDLE
