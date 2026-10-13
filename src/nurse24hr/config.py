"""Threshold configuration for the event state machine."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class Thresholds:
    """All tunable parameters of :class:`nurse24hr.events.EventMachine`.

    Defaults mirror ``config.yaml``. Distances in face-widths, times in seconds.
    """

    d_watch: float = 0.6
    d_grasp: float = 0.3
    d_release: float = 0.8

    t_grasp: float = 1.5
    t_alarm: float = 3.0
    t_release: float = 2.0

    v_pull: float = 0.8
    disp_max: float = 0.3
    osc_max: float = 0.05
    pinch_max: float = 0.12

    conf_min: float = 0.5
    t_fault: float = 5.0

    max_dt: float = 0.5
    missing_evidence_counts: bool = True
    """If osc and pinch are both unavailable, dwell alone may raise WARN (favours recall)."""
    missing_evidence_alarms: bool = False
    """If true the WARN-dwell ALARM also fires with no pinch / osc data ever seen. Off by default:
    coarse body-model hands drift onto the face when the real hands are hidden, and dwell alone
    then alarms on nothing (bedside false alarms, 2026-10-11)."""
    t_single_person: float = 1.5
    """After a second person was seen, the count must read one for this long before WARN / ALARM
    may fire. The per-frame count flickers while a caregiver leans over the bed. 0 disables."""
    t_warn_reset: float = 1.0
    """In WARN, the hand outside the watch band for this long resets the ALARM dwell clock."""
    tape_warn: bool = True
    """A still or pinched hand dwelling at the cheek tape raises WARN (never ALARM by itself)."""
    pull_arm_s: float = 0.5
    """Pull motion only counts if the hand was at the tube within this many seconds."""
    reach_count_warn: int = 3
    """Entries into WATCH within ``reach_window_s`` that raise WARN ("repeated reaching"). 0 disables."""
    reach_window_s: float = 60.0

    def __post_init__(self) -> None:
        if not (self.d_grasp < self.d_watch < self.d_release):
            raise ValueError("require d_grasp < d_watch < d_release for hysteresis")
        for name in ("t_grasp", "t_alarm", "t_release", "t_fault", "max_dt", "t_warn_reset"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.t_single_person < 0:
            raise ValueError("t_single_person must be >= 0")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Thresholds:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"unknown threshold keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Thresholds:
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        return cls.from_dict(doc.get("thresholds", {}))
