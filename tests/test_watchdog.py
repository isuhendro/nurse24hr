"""Watchdog tests. Pure core with explicit clocks; the thread test uses a fake clock and tiny poll."""

import time

import numpy as np

from nurse24hr.notify import RecordingNotifier
from nurse24hr.watchdog import Fault, ProbeSample, Watchdog, WatchdogConfig, WatchdogThread, frame_signature


def healthy(now, sig=None):
    """Inputs for a pipeline that read a frame and produced features at ``now``."""
    return dict(capture_last_read_t=now, capture_error=None, capture_ended=False,
                last_features_t=now, frame_signature=sig)


def test_healthy_pipeline_has_no_faults():
    wd = Watchdog(notifier=RecordingNotifier())
    for i in range(50):
        now = i * 0.5
        assert wd.check(now, **healthy(now, sig=i)) == []
    assert wd.active == {} and wd.notifier.of("fault") == []


def test_camera_loss_is_edge_triggered_and_recovers():
    n = RecordingNotifier()
    wd = Watchdog(notifier=n)
    wd.check(0.0, **healthy(0.0, sig=1))
    assert wd.check(1.9, capture_last_read_t=0.0, capture_error=None, capture_ended=False,
                    last_features_t=0.0, frame_signature=1) == []
    new = wd.check(2.1, capture_last_read_t=0.0, capture_error=None, capture_ended=False,
                   last_features_t=0.0, frame_signature=1)
    assert [f.kind for f in new] == ["camera_loss"]
    assert new[0].since == 0.0 and "2.1s" in new[0].reason
    # still faulty: nothing new reported
    assert wd.check(3.0, capture_last_read_t=0.0, capture_error=None, capture_ended=False,
                    last_features_t=0.0, frame_signature=1) == []
    assert len(n.of("fault")) == 1
    # frames return -> recovered once
    assert wd.check(4.0, **healthy(4.0, sig=2)) == []
    assert [f.kind for f in wd.recovered] == ["camera_loss"]
    faults = n.of("fault")
    assert len(faults) == 2 and faults[1]["status"] == "recovered" and faults[1]["kind"] == "camera_loss"
    assert wd.active == {}


def test_camera_loss_before_first_frame_counts_from_start():
    wd = Watchdog()
    assert wd.check(10.0, None, None, False, None, None) == []
    assert wd.check(11.9, None, None, False, None, None) == []
    kinds = {f.kind for f in wd.check(12.1, None, None, False, None, None)}
    assert kinds == {"camera_loss"}


def test_frozen_frames():
    n = RecordingNotifier()
    wd = Watchdog(notifier=n)
    for i in range(12):  # frames keep arriving but content never changes
        now = i * 0.5
        new = wd.check(now, **healthy(now, sig=12345))
        if now <= 5.0:
            assert new == [], now
    assert "frozen_frames" in wd.active
    assert wd.active["frozen_frames"].since == 0.0
    assert len(n.of("fault")) == 1
    # content changes -> recovered
    wd.check(6.0, **healthy(6.0, sig=999))
    assert "frozen_frames" not in wd.active
    assert n.of("fault")[-1]["status"] == "recovered"


def test_frozen_not_reported_when_no_frames_arrive():
    wd = Watchdog()
    wd.check(0.0, **healthy(0.0, sig=7))
    new = wd.check(8.0, capture_last_read_t=0.0, capture_error=None, capture_ended=False,
                   last_features_t=0.0, frame_signature=7)
    assert [f.kind for f in new] == ["camera_loss"]


def test_inference_stall():
    wd = Watchdog()
    wd.check(0.0, **healthy(0.0, sig=1))
    assert wd.check(1.5, capture_last_read_t=1.5, capture_error=None, capture_ended=False,
                    last_features_t=0.0, frame_signature=2) == []
    new = wd.check(2.5, capture_last_read_t=2.5, capture_error=None, capture_ended=False,
                   last_features_t=0.0, frame_signature=3)
    assert [f.kind for f in new] == ["inference_stall"]
    assert new[0].since == 0.0
    wd.check(3.0, **healthy(3.0, sig=4))
    assert wd.active == {}


def test_stall_not_reported_during_camera_loss():
    wd = Watchdog()
    wd.check(0.0, **healthy(0.0))
    new = wd.check(5.0, capture_last_read_t=0.0, capture_error=None, capture_ended=False,
                   last_features_t=0.0, frame_signature=None)
    assert [f.kind for f in new] == ["camera_loss"]


def test_capture_error_and_source_ended():
    wd = Watchdog()
    wd.check(0.0, **healthy(0.0))
    new = wd.check(0.1, capture_last_read_t=0.1, capture_error="cannot open", capture_ended=True,
                   last_features_t=0.1, frame_signature=None)
    assert [f.kind for f in new] == ["capture_error"]
    assert "cannot open" in new[0].reason
    wd2 = Watchdog()
    wd2.check(0.0, **healthy(0.0))
    new = wd2.check(0.1, capture_last_read_t=0.1, capture_error=None, capture_ended=True,
                    last_features_t=0.1, frame_signature=None)
    assert [f.kind for f in new] == ["source_ended"]


def test_multiple_faults_each_reported_once():
    n = RecordingNotifier()
    wd = Watchdog(notifier=n)
    wd.check(0.0, **healthy(0.0, sig=1))
    new = wd.check(10.0, capture_last_read_t=0.0, capture_error="read failed", capture_ended=True,
                   last_features_t=0.0, frame_signature=1)
    assert sorted(f.kind for f in new) == ["camera_loss", "capture_error"]
    assert wd.check(11.0, capture_last_read_t=0.0, capture_error="read failed", capture_ended=True,
                    last_features_t=0.0, frame_signature=1) == []
    assert len(n.of("fault")) == 2


def test_heartbeat_cadence():
    n = RecordingNotifier()
    wd = Watchdog(WatchdogConfig(heartbeat_s=10.0), notifier=n)
    assert wd.maybe_heartbeat(0.0, fps=14.9, frames_read=3) is True
    assert wd.maybe_heartbeat(5.0) is False
    assert wd.maybe_heartbeat(9.99) is False
    assert wd.maybe_heartbeat(10.0, fps=15.0, frames_read=150) is True
    hb = n.of("heartbeat")
    assert len(hb) == 2
    assert hb[0] == {"t": 0.0, "fps": 14.9, "frames_read": 3, "faults": []}
    wd.check(10.0, **healthy(0.0))  # camera lost for 10 s
    wd.maybe_heartbeat(20.0)
    assert n.of("heartbeat")[-1]["faults"] == ["camera_loss"]


def test_frame_signature_stable_and_sensitive():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert frame_signature(a) == frame_signature(a.copy())
    b = a.copy()
    b[0, 0, 0] ^= 0xFF  # (0, 0) is always a sampled pixel
    assert frame_signature(a) != frame_signature(b)
    assert frame_signature(np.zeros((480, 640), np.uint8)) != frame_signature(np.ones((480, 640), np.uint8))
    # cheap: independent of resolution
    big = np.zeros((2160, 3840, 3), np.uint8)
    t0 = time.perf_counter()
    for _ in range(20):
        frame_signature(big)
    assert (time.perf_counter() - t0) / 20 < 0.01


def test_config_from_dict():
    cfg = WatchdogConfig.from_dict({"camera_loss_s": 3, "heartbeat_s": 5})
    assert cfg.camera_loss_s == 3.0 and cfg.heartbeat_s == 5.0 and cfg.frozen_s == 5.0
    import pytest
    with pytest.raises(KeyError):
        WatchdogConfig.from_dict({"nope": 1})


def test_thread_polls_routes_faults_and_stops():
    n = RecordingNotifier()
    wd = Watchdog(WatchdogConfig(heartbeat_s=1000.0), notifier=n)
    clock = {"t": 0.0}
    state = {"last_read": 0.0}

    def probe():
        return ProbeSample(capture_last_read_t=state["last_read"], capture_error=None, capture_ended=False,
                           last_features_t=state["last_read"], frame_signature=None, fps=15.0, frames_read=10)

    got: list[Fault] = []
    th = WatchdogThread(wd, probe, poll_s=0.005, on_fault=got.append, clock=lambda: clock["t"])
    th.start()
    deadline = time.monotonic() + 2
    while th.ticks < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert got == []
    clock["t"] = 5.0  # frames stopped 5 s ago
    while not got and time.monotonic() < deadline:
        time.sleep(0.005)
    assert [f.kind for f in got] == ["camera_loss"]
    th.stop()
    assert not th.is_alive() and th.error is None
    assert len(n.of("heartbeat")) == 1  # first tick only, cadence 1000 s
    assert len([f for f in n.of("fault") if f["status"] == "active"]) == 1


def test_thread_crash_is_reported_not_silent():
    n = RecordingNotifier()
    wd = Watchdog(notifier=n)

    def probe():
        raise RuntimeError("boom")

    th = WatchdogThread(wd, probe, poll_s=0.005)
    th.start()
    th.join(timeout=2)
    assert not th.is_alive()
    assert th.error and "boom" in th.error
    assert n.of("fault")[0]["kind"] == "watchdog_crashed"
