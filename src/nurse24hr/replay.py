"""Offline replay, metrics and threshold sweeps over logged Features.

This is the tuning loop from the spec (§8): run ``events.py`` over a Parquet
log in milliseconds, score it against the session's script-derived labels,
and sweep thresholds without touching video.

Labels CSV: ``t_start,t_end,label`` with a header. Labels follow the staged
protocol (§7.3): idle, scratch, rub, other, touch, grasp, pull, caregiver,
occlusion.

Metrics
-------
* recall: fraction of ``pull`` segments with an ALARM inside
  ``[t_start, t_end + grace]``.
* time_to_alarm: ALARM time minus pull segment start, per detected pull.
* false alarms: ALARM transitions outside every pull window, attributed to the
  label active at that time (``none`` when unlabeled), plus a rate per hour.
* warn counts per label, for tuning the station-notification load.

Usage::

    nurse24hr-replay session.parquet --labels session_labels.csv
    nurse24hr-replay session.parquet --labels l.csv --sweep t_grasp=1.0,1.5,2.0 --sweep osc_max=0.04,0.05,0.06
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path

from nurse24hr.config import Thresholds
from nurse24hr.events import Event, State, run
from nurse24hr.features import Features
from nurse24hr.features_log import read_features

Segment = tuple[float, float, str]


def read_labels(path: str | Path) -> list[Segment]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [(float(r["t_start"]), float(r["t_end"]), r["label"].strip()) for r in rows]


def label_at(segments: list[Segment], t: float) -> str:
    for t0, t1, lab in segments:
        if t0 <= t <= t1:
            return lab
    return "none"


@dataclass
class Metrics:
    n_pull: int = 0
    n_pull_detected: int = 0
    time_to_alarm: list[float] = field(default_factory=list)
    false_alarms_by_label: dict[str, int] = field(default_factory=dict)
    warns_by_label: dict[str, int] = field(default_factory=dict)
    faults: int = 0
    duration_s: float = 0.0

    @property
    def recall(self) -> float | None:
        return None if self.n_pull == 0 else self.n_pull_detected / self.n_pull

    @property
    def false_alarms(self) -> int:
        return sum(self.false_alarms_by_label.values())

    @property
    def false_alarms_per_hour(self) -> float | None:
        return None if self.duration_s <= 0 else self.false_alarms / (self.duration_s / 3600)

    @property
    def median_tta(self) -> float | None:
        return statistics.median(self.time_to_alarm) if self.time_to_alarm else None

    def summary(self) -> str:
        rec = "n/a" if self.recall is None else f"{self.recall:.2f} ({self.n_pull_detected}/{self.n_pull})"
        tta = "n/a" if self.median_tta is None else f"{self.median_tta:.2f}s (max {max(self.time_to_alarm):.2f}s)"
        fah = "n/a" if self.false_alarms_per_hour is None else f"{self.false_alarms_per_hour:.2f}/h"
        return (f"recall {rec}  time-to-alarm {tta}  false alarms {self.false_alarms} ({fah}) "
                f"by label {dict(sorted(self.false_alarms_by_label.items()))}  warns {dict(sorted(self.warns_by_label.items()))}  "
                f"faults {self.faults}")


def score(events: list[Event], segments: list[Segment], duration_s: float, *, grace_s: float = 3.0,
          ack_after_s: float = 1.0) -> Metrics:
    """Score a replay. Latched alarms are auto-acked ``ack_after_s`` later in :func:`replay_with_ack`."""
    m = Metrics(duration_s=duration_s)
    alarms = [e for e in events if e.kind == "transition" and e.state is State.ALARM]
    warns = [e for e in events if e.kind == "transition" and e.state is State.WARN]
    m.faults = sum(1 for e in events if e.kind == "transition" and e.state is State.FAULT)
    pulls = [(t0, t1) for t0, t1, lab in segments if lab == "pull"]
    m.n_pull = len(pulls)
    used: set[int] = set()
    for t0, t1 in pulls:
        hit = next((i for i, e in enumerate(alarms) if i not in used and t0 <= e.t <= t1 + grace_s), None)
        if hit is not None:
            used.add(hit)
            m.n_pull_detected += 1
            m.time_to_alarm.append(alarms[hit].t - t0)
    for i, e in enumerate(alarms):
        if i in used:
            continue
        lab = label_at(segments, e.t)
        m.false_alarms_by_label[lab] = m.false_alarms_by_label.get(lab, 0) + 1
    for e in warns:
        lab = label_at(segments, e.t)
        m.warns_by_label[lab] = m.warns_by_label.get(lab, 0) + 1
    return m


def replay_with_ack(features: list[Features], th: Thresholds, *, ack_after_s: float = 1.0) -> list[Event]:
    """Replay, acknowledging any latched state after ``ack_after_s`` so later events are still observed.

    An ack also rebases ``tube_disp``: in the live system the nurse re-seats the
    tube and the detector baseline resets, so a tube that stays displaced after
    the ack must not re-alarm forever. Here the displacement at ack time is
    subtracted from all later values.
    """
    from nurse24hr.events import EventMachine

    m = EventMachine(th)
    events: list[Event] = []
    latched_at: float | None = None
    disp_offset = 0.0
    for f in features:
        if disp_offset and f.tube_disp is not None:
            f = replace(f, tube_disp=max(0.0, f.tube_disp - disp_offset))
        events.extend(m.step(f))
        if m.state.latched:
            if latched_at is None:
                latched_at = f.t
            elif f.t - latched_at >= ack_after_s:
                events.extend(m.ack(f.t))
                latched_at = None
                if f.tube_disp:
                    disp_offset += f.tube_disp
        else:
            latched_at = None
    return events


def parse_sweep(items: list[str]) -> dict[str, list]:
    grid: dict[str, list] = {}
    for item in items:
        key, _, vals = item.partition("=")
        parsed = []
        for v in vals.split(","):
            v = v.strip()
            parsed.append(v.lower() == "true" if v.lower() in ("true", "false") else float(v))
        grid[key.strip()] = parsed
    return grid


def sweep(features: list[Features], segments: list[Segment], base: Thresholds, grid: dict[str, list],
          duration_s: float) -> list[tuple[dict, Metrics]]:
    keys = list(grid)
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        overrides = dict(zip(keys, combo))
        th = replace(base, **overrides)
        evs = replay_with_ack(features, th)
        results.append((overrides, score(evs, segments, duration_s)))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nurse24hr-replay", description="Replay logged features through the state machine.")
    ap.add_argument("parquet")
    ap.add_argument("--labels", help="labels CSV (t_start,t_end,label)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sweep", action="append", default=[], help="param=v1,v2,... (repeatable)")
    ap.add_argument("--quiet", action="store_true", help="do not print the event timeline")
    args = ap.parse_args(argv)

    th = Thresholds.from_yaml(args.config) if Path(args.config).exists() else Thresholds()
    feats = read_features(args.parquet)
    if not feats:
        print("no features in file")
        return 1
    duration = feats[-1].t - feats[0].t
    segments = read_labels(args.labels) if args.labels else []

    if args.sweep:
        grid = parse_sweep(args.sweep)
        rows = sweep(feats, segments, th, grid, duration)
        rows.sort(key=lambda r: (-(r[1].recall or 0), r[1].false_alarms, r[1].median_tta or 0))
        for overrides, m in rows:
            print(f"{overrides}  ->  {m.summary()}")
        return 0

    evs = replay_with_ack(feats, th)
    if not args.quiet:
        for e in evs:
            print(str(e))
    print(f"frames {len(feats)}  duration {duration:.1f}s")
    if segments:
        print(score(evs, segments, duration).summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
