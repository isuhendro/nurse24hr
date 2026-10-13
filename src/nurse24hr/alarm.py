"""Alarm manager: turns state-machine events into sound, notifications, escalation and an audit trail.

Why a separate module from ``events.py``: the state machine is pure and knows
nothing about time-of-day, speakers or nurses. This module owns the side
effects, so it can be swapped (afplay on a Mac, GPIO buzzer on a Jetson)
without touching detection logic, and tested with a recording notifier and a
silent sounder.

Levels (TS-001 §5.4)
--------------------
WARN   station notification only. No bedside sound in v1 unless ``warn_sound`` is set.
ALARM  loud looping local sound + remote notification. Latched.
FAULT  distinct looping fault tone + remote notification. Latched.
Latched alarms repeat until :meth:`AlarmManager.ack`. If nobody acks within
``escalate_after_s`` an ``escalation`` notification goes out, and again every
``escalate_after_s`` after that.

Clock
-----
Nothing here reads the wall clock. ``handle`` uses ``Event.t`` and the run loop
calls :meth:`AlarmManager.tick` with the *same* clock (frame time), so a
replayed recording escalates at the same instants as the live run did and the
tests are deterministic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nurse24hr.events import Event, State
from nurse24hr.notify import Notifier


# ------------------------------------------------------------------ sound

class Sounder(Protocol):
    def play(self, path: str, loop: bool) -> None: ...
    def stop(self) -> None: ...
    def tick(self) -> None:
        """Called periodically; implementations use it to keep a looping sound alive."""
        ...


class NullSounder:
    """Silent sounder that records what it was asked to do. For tests and ``--headless``."""

    def __init__(self) -> None:
        self.played: list[tuple[str, bool]] = []
        self.stops = 0
        self.current: tuple[str, bool] | None = None

    def play(self, path: str, loop: bool) -> None:
        self.played.append((path, loop))
        self.current = (path, loop)

    def stop(self) -> None:
        self.stops += 1
        self.current = None

    def tick(self) -> None:
        return None


class AfplaySounder:
    """macOS ``afplay`` driven through a subprocess. Loops by respawning on exit.

    Never raises: a missing binary or a failed spawn is recorded in ``error`` so
    the alarm still goes out over the network.
    """

    def __init__(self, binary: str = "afplay") -> None:
        self.binary = binary
        self.proc: subprocess.Popen | None = None
        self.current: tuple[str, bool] | None = None
        self.error: str | None = None

    def _spawn(self, path: str) -> None:
        if shutil.which(self.binary) is None:
            self.error = f"{self.binary} not found"
            return
        try:
            self.proc = subprocess.Popen(
                [self.binary, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.proc = None

    def play(self, path: str, loop: bool) -> None:
        self.stop()
        self.current = (path, loop)
        self._spawn(path)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except OSError:
                pass
        self.proc = None
        self.current = None

    def tick(self) -> None:
        if self.current is None or not self.current[1]:
            return
        if self.proc is None or self.proc.poll() is not None:
            self._spawn(self.current[0])


# ----------------------------------------------------------------- config

@dataclass(frozen=True, slots=True)
class AlarmConfig:
    """Mirrors the ``alarm:`` section of ``config.yaml``."""

    warn_sound: str | None = None
    alarm_sound: str | None = None
    fault_sound: str | None = None
    escalate_after_s: float = 30.0
    audit_log: str | None = "logs/audit.jsonl"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AlarmConfig:
        data = dict(data or {})
        known = {"warn_sound", "alarm_sound", "fault_sound", "escalate_after_s", "audit_log"}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"unknown alarm config keys: {sorted(unknown)}")
        if "escalate_after_s" in data:
            data["escalate_after_s"] = float(data["escalate_after_s"])
        return cls(**data)


LEVEL_BY_STATE: dict[State, str] = {
    State.IDLE: "info",
    State.WATCH: "info",
    State.WARN: "warn",
    State.ALARM: "alarm",
    State.FAULT: "fault",
}


@dataclass
class Latched:
    event_id: str
    level: str
    since: float
    next_escalation: float
    escalations: int = 0


@dataclass
class AlarmManager:
    """Side-effect owner for alarms. See module docstring.

    Call :meth:`handle` with every :class:`Event` from the state machine,
    :meth:`raise_fault` for watchdog faults, :meth:`tick` once per frame with
    the frame clock, and :meth:`ack` when a nurse acknowledges.
    """

    notifier: Notifier
    sounder: Sounder = field(default_factory=NullSounder)
    cfg: AlarmConfig = field(default_factory=AlarmConfig)
    latched: Latched | None = None
    _missing_warned: set[str] = field(default_factory=set)
    _audit_fh: Any = None

    def __post_init__(self) -> None:
        if self.cfg.audit_log:
            p = Path(self.cfg.audit_log)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._audit_fh = open(p, "a", encoding="utf-8")

    # ---------------------------------------------------------------- api

    @property
    def is_latched(self) -> bool:
        return self.latched is not None

    def handle(self, event: Event) -> str:
        """Process one state-machine event. Returns the notification's event_id."""
        if event.kind == "ack":
            self.ack(event.t, user="machine")
            return self.latched.event_id if self.latched else ""

        event_id = str(uuid.uuid4())
        level = "suppressed" if event.kind == "suppressed" else LEVEL_BY_STATE[event.state]
        payload = {
            "event_id": event_id,
            "level": level,
            "state": event.state.value,
            "prev": event.prev.value,
            "reason": event.reason,
            "t": event.t,
        }

        if event.kind == "transition":
            if event.state is State.WARN:
                self._play(self.cfg.warn_sound, loop=False, t=event.t)
            elif event.state is State.ALARM:
                self._latch(event_id, "alarm", event.t)
                self._play(self.cfg.alarm_sound, loop=True, t=event.t)
            elif event.state is State.FAULT:
                self._latch(event_id, "fault", event.t)
                self._play(self.cfg.fault_sound, loop=True, t=event.t)
            elif event.state is State.IDLE and self.latched is None:
                self.sounder.stop()

        self.notifier.notify("event", payload)
        self._audit("event", payload)
        return event_id

    def raise_fault(self, kind: str, reason: str, t: float) -> str:
        """Watchdog fault. Latches like FAULT; cleared by :meth:`ack`.

        An already latched ALARM keeps its (louder) sound; the fault is still
        notified and audited.
        """
        event_id = str(uuid.uuid4())
        payload = {
            "event_id": event_id,
            "level": "fault",
            "state": State.FAULT.value,
            "prev": None,
            "kind": kind,
            "reason": reason,
            "t": t,
        }
        if self.latched is None:
            self._latch(event_id, "fault", t)
            self._play(self.cfg.fault_sound, loop=True, t=t)
        self.notifier.notify("event", payload)
        self._audit("fault", payload)
        return event_id

    def ack(self, t: float, user: str = "local") -> bool:
        """Clear the latched alarm. Returns False if nothing was latched."""
        if self.latched is None:
            return False
        payload = {"event_id": self.latched.event_id, "user": user, "t": t,
                   "latched_for_s": round(t - self.latched.since, 3)}
        self.latched = None
        self.sounder.stop()
        self.notifier.notify("ack", payload)
        self._audit("ack", payload)
        return True

    def tick(self, now: float) -> None:
        """Advance escalation and keep a looping sound alive. ``now`` is frame time."""
        self.sounder.tick()
        if self.latched is None:
            return
        while now >= self.latched.next_escalation:
            self.latched.escalations += 1
            payload = {
                "event_id": self.latched.event_id,
                "level": self.latched.level,
                "n": self.latched.escalations,
                "unacked_for_s": round(now - self.latched.since, 3),
                "t": now,
            }
            self.notifier.notify("escalation", payload)
            self._audit("escalation", payload)
            self.latched.next_escalation += self.cfg.escalate_after_s

    def close(self) -> None:
        self.sounder.stop()
        if self._audit_fh is not None:
            self._audit_fh.close()
            self._audit_fh = None

    # ------------------------------------------------------------ helpers

    def _latch(self, event_id: str, level: str, t: float) -> None:
        self.latched = Latched(event_id, level, t, t + self.cfg.escalate_after_s)

    def _play(self, path: str | None, *, loop: bool, t: float) -> None:
        if not path:
            return
        if not os.path.exists(path):
            if path not in self._missing_warned:
                self._missing_warned.add(path)
                payload = {"event_id": str(uuid.uuid4()), "level": "warning", "state": None, "prev": None,
                           "reason": f"sound file not found: {path}", "t": t}
                self.notifier.notify("event", payload)
                self._audit("warning", payload)
            return
        self.sounder.play(path, loop)

    def _audit(self, kind: str, payload: dict[str, Any]) -> None:
        """One JSON line: ``entry`` says what kind of record it is; the payload is copied verbatim."""
        if self._audit_fh is None:
            return
        self._audit_fh.write(json.dumps({"entry": kind, **payload}, default=str) + "\n")
        self._audit_fh.flush()
