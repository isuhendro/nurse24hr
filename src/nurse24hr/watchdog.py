"""Watchdog: detects a silently failing pipeline and says so loudly.

The worst failure mode of a safety monitor is quiet death: the camera cable
pulled, an RTSP stream that repeats its last frame, an inference thread that
hung. None of those produce a state-machine transition, so the state machine
cannot catch them. This module watches the plumbing instead (TS-001 §5.6).

The core, :meth:`Watchdog.check`, is pure: it takes ``now`` and a snapshot of
pipeline health and returns the faults that *started* this tick. Faults are
edge-triggered: one ``fault`` notification when a fault begins, one with
``status: recovered`` when it clears, nothing in between. A heartbeat goes to
the station every ``heartbeat_s`` so the station can alarm if *this process*
dies.

The watchdog never changes :class:`~nurse24hr.events.EventMachine` state. The
run loop routes new faults to :meth:`~nurse24hr.alarm.AlarmManager.raise_fault`.

Clock: every time passed in must come from ``time.monotonic()`` (the wall
clock of this process), including ``last_features_t``, because the question
is "how long since this process last did X", not frame time.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nurse24hr.notify import Notifier


@dataclass(frozen=True, slots=True)
class WatchdogConfig:
    camera_loss_s: float = 2.0
    frozen_s: float = 5.0
    stall_s: float = 2.0
    heartbeat_s: float = 10.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> WatchdogConfig:
        data = dict(data or {})
        known = {"camera_loss_s", "frozen_s", "stall_s", "heartbeat_s"}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"unknown watchdog config keys: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in data.items()})


@dataclass(frozen=True, slots=True)
class Fault:
    kind: str
    """camera_loss | frozen_frames | inference_stall | capture_error | source_ended"""
    reason: str
    since: float
    """When the condition started (monotonic clock)."""


@dataclass(frozen=True, slots=True)
class ProbeSample:
    """Snapshot of pipeline health, produced by the run loop's probe callable."""

    capture_last_read_t: float | None
    capture_error: str | None
    capture_ended: bool
    last_features_t: float | None
    frame_signature: int | None
    fps: float = 0.0
    frames_read: int = 0


def frame_signature(image: np.ndarray, grid: int = 16) -> int:
    """Cheap content hash for frozen-frame detection.

    Downsamples to roughly ``grid x grid`` samples per channel by striding, so
    the cost is independent of resolution. Identical frames hash equal; any
    real sensor noise changes the hash, so a live camera is never "frozen".
    """
    if image.ndim < 2:
        raise ValueError("expected an image array")
    h, w = image.shape[:2]
    sy, sx = max(1, h // grid), max(1, w // grid)
    sample = np.ascontiguousarray(image[::sy, ::sx])
    digest = hashlib.blake2b(sample.tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class Watchdog:
    """Pure fault detector with edge-triggered reporting."""

    def __init__(self, cfg: WatchdogConfig | None = None, notifier: Notifier | None = None) -> None:
        self.cfg = cfg or WatchdogConfig()
        self.notifier = notifier
        self.active: dict[str, Fault] = {}
        self.recovered: list[Fault] = []
        """Faults that cleared on the most recent :meth:`check`."""
        self._started: float | None = None
        self._last_sig: int | None = None
        self._sig_since: float | None = None
        self._last_heartbeat: float | None = None

    # ---------------------------------------------------------------- core

    def check(self, now: float, capture_last_read_t: float | None, capture_error: str | None,
              capture_ended: bool, last_features_t: float | None, frame_signature: int | None) -> list[Fault]:
        """Evaluate health at ``now``. Returns faults that began on this call."""
        if self._started is None:
            self._started = now
        conditions: dict[str, tuple[str, float]] = {}

        if capture_error:
            conditions["capture_error"] = (f"capture error: {capture_error}", now)
        elif capture_ended:
            conditions["source_ended"] = ("video source ended", now)

        last_read = capture_last_read_t if capture_last_read_t is not None else self._started
        frames_arriving = True
        if now - last_read > self.cfg.camera_loss_s:
            frames_arriving = False
            conditions["camera_loss"] = (f"no frame for {now - last_read:.1f}s", last_read)

        if frame_signature is not None:
            if frame_signature != self._last_sig:
                self._last_sig = frame_signature
                self._sig_since = now
            elif frames_arriving and self._sig_since is not None and now - self._sig_since > self.cfg.frozen_s:
                conditions["frozen_frames"] = (f"identical frames for {now - self._sig_since:.1f}s", self._sig_since)

        if frames_arriving:
            last_feat = last_features_t if last_features_t is not None else self._started
            if now - last_feat > self.cfg.stall_s:
                conditions["inference_stall"] = (f"no features for {now - last_feat:.1f}s", last_feat)

        new: list[Fault] = []
        for kind, (reason, since) in conditions.items():
            if kind not in self.active:
                f = Fault(kind, reason, since)
                self.active[kind] = f
                new.append(f)
                self._notify_fault(f, now, "active")

        self.recovered = [f for kind, f in self.active.items() if kind not in conditions]
        for f in self.recovered:
            del self.active[f.kind]
            self._notify_fault(f, now, "recovered")
        return new

    def check_sample(self, now: float, s: ProbeSample) -> list[Fault]:
        return self.check(now, s.capture_last_read_t, s.capture_error, s.capture_ended,
                          s.last_features_t, s.frame_signature)

    def maybe_heartbeat(self, now: float, fps: float = 0.0, frames_read: int = 0) -> bool:
        """Send a heartbeat if one is due. Returns True when sent."""
        if self._last_heartbeat is not None and now - self._last_heartbeat < self.cfg.heartbeat_s:
            return False
        self._last_heartbeat = now
        if self.notifier is not None:
            self.notifier.notify("heartbeat", {
                "t": now,
                "fps": round(fps, 2),
                "frames_read": frames_read,
                "faults": [f.kind for f in self.active.values()],
            })
        return True

    def _notify_fault(self, f: Fault, now: float, status: str) -> None:
        if self.notifier is not None:
            self.notifier.notify("fault", {"kind": f.kind, "reason": f.reason, "since": f.since, "t": now, "status": status})


class WatchdogThread(threading.Thread):
    """Polls ``probe`` every ``poll_s``, runs the watchdog and routes new faults to ``on_fault``."""

    def __init__(self, watchdog: Watchdog, probe: Callable[[], ProbeSample], *, poll_s: float = 0.5,
                 on_fault: Callable[[Fault], None] | None = None, clock: Callable[[], float] = time.monotonic,
                 name: str = "watchdog") -> None:
        super().__init__(name=name, daemon=True)
        self.watchdog = watchdog
        self.probe = probe
        self.poll_s = poll_s
        self.on_fault = on_fault
        self.clock = clock
        self._stop_evt = threading.Event()
        self.ticks = 0
        self.error: str | None = None

    def stop(self, join: bool = True, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if join and self.is_alive():
            self.join(timeout)

    def run(self) -> None:
        try:
            while not self._stop_evt.is_set():
                now = self.clock()
                sample = self.probe()
                for f in self.watchdog.check_sample(now, sample):
                    if self.on_fault is not None:
                        self.on_fault(f)
                self.watchdog.maybe_heartbeat(now, sample.fps, sample.frames_read)
                self.ticks += 1
                self._stop_evt.wait(self.poll_s)
        except Exception as exc:  # noqa: BLE001 - surfaced via .error, never silently dies
            self.error = f"{type(exc).__name__}: {exc}"
            if self.watchdog.notifier is not None:
                self.watchdog.notifier.notify("fault", {"kind": "watchdog_crashed", "reason": self.error,
                                                        "since": self.clock(), "t": self.clock(), "status": "active"})
