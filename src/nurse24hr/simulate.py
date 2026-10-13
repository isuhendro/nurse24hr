"""Synthetic feature-sequence generator for the state machine.

No camera, no video. Each scenario builds a trajectory of ``d`` (hand-to-tube
distance), ``pinch``, ``tube_disp``, ``conf`` and ``n_persons`` from
piecewise segments, then derives ``v_radial`` and ``osc`` the same way
perception will: finite difference and trailing 1 s standard deviation.
That keeps the simulated features honest about one key fact: a scratch
produces large radial velocities too, so velocity alone can never be the
pull trigger.

Scenarios map to the table in the spec (TS-001 §7.2). Each returns a
:class:`Scenario` with the sequence and the states it must / must not reach.

Run ``python -m nurse24hr.simulate --help``.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from nurse24hr.config import Thresholds
from nurse24hr.events import SEVERITY, Event, EventMachine, State
from nurse24hr.features import Features

FPS = 15
OPEN_HAND = 0.40  # pinch value for extended fingers
PINCH_GRIP = 0.05  # pinch value for a grip on the tube
FAR = 2.0  # resting hand distance


@dataclass
class Scenario:
    name: str
    features: list[Features]
    must_reach: State | None = None
    """A state the run must enter at some point (e.g. ALARM for a pull)."""
    max_state: State = State.ALARM
    """Highest severity the run is allowed to reach."""
    expect_suppressed: bool = False
    expect_final: State | None = None
    labels: list[tuple[float, float, str]] = field(default_factory=list)


class Trajectory:
    """Piecewise builder. Rows are (d, pinch, disp, conf, n_persons, d_tape).

    ``tape`` is the hand's distance to the cheek tape anchor; ``None`` means
    unknown (the default, so the original scenarios are unaffected).
    """

    def __init__(self, rng: np.random.Generator, fps: int = FPS) -> None:
        self.rng = rng
        self.fps = fps
        self.rows: list[tuple[float, float, float, float, int, float | None]] = []
        self.labels: list[tuple[float, float, str]] = []

    @property
    def t(self) -> float:
        return len(self.rows) / self.fps

    def _n(self, dur: float) -> int:
        return max(1, round(dur * self.fps))

    def hold(self, dur: float, d: float, *, pinch: float = OPEN_HAND, disp: float = 0.0,
             conf: float = 0.95, n: int = 1, noise: float = 0.01, label: str | None = None,
             tape: float | None = None) -> Trajectory:
        """Hand stays put with small jitter."""
        t0 = self.t
        for _ in range(self._n(dur)):
            tp = None if tape is None else max(0.0, tape + self.rng.normal(0, noise))
            self.rows.append((max(0.0, d + self.rng.normal(0, noise)), pinch, disp, conf, n, tp))
        if label:
            self.labels.append((t0, self.t, label))
        return self

    def oscillate(self, dur: float, center: float, amp: float, freq: float, *, pinch: float = OPEN_HAND,
                  disp: float = 0.0, conf: float = 0.95, n: int = 1, noise: float = 0.01,
                  label: str | None = None, tape: float | None = None, tape_amp: float = 0.0) -> Trajectory:
        """Scratch / rub: sinusoidal hand motion around the contact point."""
        t0 = self.t
        phase = self.rng.uniform(0, 2 * math.pi)
        for i in range(self._n(dur)):
            osc = math.sin(2 * math.pi * freq * i / self.fps + phase)
            d = center + amp * osc + self.rng.normal(0, noise)
            tp = None if tape is None else max(0.0, tape + tape_amp * osc + self.rng.normal(0, noise))
            self.rows.append((max(0.0, d), pinch, disp, conf, n, tp))
        if label:
            self.labels.append((t0, self.t, label))
        return self

    def ramp(self, dur: float, d0: float, d1: float, *, pinch: float = OPEN_HAND, disp0: float = 0.0,
             disp1: float = 0.0, conf: float = 0.95, n: int = 1, noise: float = 0.01,
             label: str | None = None, tape0: float | None = None, tape1: float | None = None) -> Trajectory:
        """Linear move of the hand (and optionally the tube)."""
        t0 = self.t
        steps = self._n(dur)
        for i in range(steps):
            a = (i + 1) / steps
            d = d0 + (d1 - d0) * a + self.rng.normal(0, noise)
            tp = None if tape0 is None or tape1 is None else max(0.0, tape0 + (tape1 - tape0) * a + self.rng.normal(0, noise))
            self.rows.append((max(0.0, d), pinch, disp0 + (disp1 - disp0) * a, conf, n, tp))
        if label:
            self.labels.append((t0, self.t, label))
        return self

    def flicker(self, dur: float, d: float, *, pinch: float, period: float = 0.5, low_conf: float = 0.2,
                high_conf: float = 0.9, noise: float = 0.01, label: str | None = None) -> Trajectory:
        """Hold with confidence alternating low/high every ``period/2`` seconds."""
        t0 = self.t
        half = max(1, round(period * self.fps / 2))
        for i in range(self._n(dur)):
            conf = low_conf if (i // half) % 2 else high_conf
            self.rows.append((max(0.0, d + self.rng.normal(0, noise)), pinch, 0.0, conf, 1, None))
        if label:
            self.labels.append((t0, self.t, label))
        return self

    def build(self, *, with_tube_disp: bool = True, with_evidence: bool = True) -> list[Features]:
        """Derive v_radial and osc, emit Features.

        ``with_tube_disp=False`` emulates v1 (no tube detector).
        ``with_evidence=False`` drops osc and pinch, emulating a hand model failure.
        """
        d = np.array([r[0] for r in self.rows])
        n = len(d)
        t = np.arange(n) / self.fps
        v = np.zeros(n)
        v[1:] = (d[1:] - d[:-1]) * self.fps
        # light smoothing over 2 frames, as perception would do
        v[1:] = 0.5 * (v[1:] + v[:-1])
        win = self.fps
        # osc is the hand's own motion; with only anchor distances available, the
        # larger scatter of the two distances is the honest proxy.
        tape = np.array([np.nan if r[5] is None else r[5] for r in self.rows])
        osc = np.array([
            max(d[max(0, i - win + 1): i + 1].std(),
                0.0 if np.isnan(tape[i]) else np.nanstd(tape[max(0, i - win + 1): i + 1]))
            for i in range(n)
        ])
        out = []
        for i, (di, pinch, disp, conf, np_, tp) in enumerate(self.rows):
            out.append(Features(
                t=float(t[i]),
                d=float(di),
                conf=float(conf),
                n_persons=np_,
                v_radial=float(v[i]),
                pinch=float(pinch) if with_evidence else None,
                osc=float(osc[i]) if with_evidence else None,
                tube_disp=float(disp) if with_tube_disp else None,
                d_tape=None if tp is None else float(tp),
            ))
        return out


# ---------------------------------------------------------------- scenarios

def idle(rng: np.random.Generator) -> Scenario:
    tr = Trajectory(rng)
    for _ in range(int(rng.integers(6, 12))):
        tr.hold(rng.uniform(3, 6), rng.uniform(1.5, 3.0), noise=0.03, label="idle")
    return Scenario("idle", tr.build(), max_state=State.IDLE, expect_final=State.IDLE, labels=tr.labels)


def nose_scratch_short(rng: np.random.Generator) -> Scenario:
    tr = Trajectory(rng)
    tr.hold(2.0, FAR, label="idle")
    tr.ramp(0.4, FAR, 0.2)
    tr.oscillate(rng.uniform(0.3, 1.2), center=0.2, amp=rng.uniform(0.08, 0.12), freq=rng.uniform(3, 5), label="scratch")
    tr.ramp(0.4, 0.2, FAR)
    tr.hold(3.0, FAR, label="idle")
    return Scenario("nose_scratch_short", tr.build(), max_state=State.WATCH, expect_final=State.IDLE, labels=tr.labels)


def long_face_rub(rng: np.random.Generator) -> Scenario:
    tr = Trajectory(rng)
    tr.hold(2.0, FAR, label="idle")
    tr.ramp(0.5, FAR, 0.3)
    tr.oscillate(rng.uniform(2, 6), center=0.3, amp=rng.uniform(0.08, 0.12), freq=rng.uniform(1.5, 3), pinch=0.35, label="rub")
    tr.ramp(0.5, 0.3, FAR)
    tr.hold(3.0, FAR, label="idle")
    return Scenario("long_face_rub", tr.build(), max_state=State.WARN, expect_final=State.IDLE, labels=tr.labels)


def wipe_mouth(rng: np.random.Generator) -> Scenario:
    tr = Trajectory(rng)
    tr.hold(2.0, FAR, label="idle")
    tr.ramp(0.5, FAR, 0.35)
    tr.oscillate(rng.uniform(0.5, 2.0), center=0.35, amp=0.06, freq=rng.uniform(0.8, 1.5), pinch=0.2, label="other")
    tr.ramp(0.5, 0.35, FAR)
    tr.hold(3.0, FAR, label="idle")
    return Scenario("wipe_mouth", tr.build(), max_state=State.WARN, expect_final=State.IDLE, labels=tr.labels)


def grasp_hold_release(rng: np.random.Generator) -> Scenario:
    tr = Trajectory(rng)
    tr.hold(2.0, FAR, label="idle")
    tr.ramp(0.5, FAR, 0.15)
    tr.hold(rng.uniform(2.0, 4.0), 0.15, pinch=PINCH_GRIP, label="grasp")
    # fingers open first, then the hand leaves: a release, not a pull
    tr.ramp(0.5, 0.15, FAR, pinch=OPEN_HAND)
    tr.hold(3.0, FAR, label="idle")
    return Scenario("grasp_hold_release", tr.build(), must_reach=State.WARN, max_state=State.WARN,
                    expect_final=State.IDLE, labels=tr.labels)


def grasp_and_pull(rng: np.random.Generator) -> Scenario:
    tr = Trajectory(rng)
    tr.hold(2.0, FAR, label="idle")
    tr.ramp(0.5, FAR, 0.15)
    tr.hold(rng.uniform(1.7, 3.0), 0.15, pinch=PINCH_GRIP, label="grasp")
    # pinch held while the hand moves away and the tube comes with it
    tr.ramp(rng.uniform(0.5, 1.0), 0.15, 1.0, pinch=PINCH_GRIP, disp0=0.0, disp1=0.5, label="pull")
    tr.hold(2.0, 1.0, pinch=PINCH_GRIP, disp=0.5)
    return Scenario("grasp_and_pull", tr.build(), must_reach=State.ALARM, expect_final=State.ALARM, labels=tr.labels)


def fast_yank(rng: np.random.Generator) -> Scenario:
    tr = Trajectory(rng)
    tr.hold(2.0, FAR, label="idle")
    tr.ramp(0.4, FAR, 0.1)
    tr.hold(0.5, 0.1, pinch=PINCH_GRIP, label="grasp")
    tr.ramp(0.3, 0.1, 1.2, pinch=PINCH_GRIP, disp0=0.0, disp1=0.6, label="pull")
    tr.hold(2.0, 1.2, pinch=PINCH_GRIP, disp=0.6)
    return Scenario("fast_yank", tr.build(), must_reach=State.ALARM, expect_final=State.ALARM, labels=tr.labels)


def caregiver_adjust(rng: np.random.Generator) -> Scenario:
    tr = Trajectory(rng)
    tr.hold(2.0, FAR, n=2, label="idle")
    tr.ramp(0.5, FAR, 0.15, n=2)
    tr.hold(rng.uniform(5, 20), 0.15, pinch=PINCH_GRIP, disp=0.1, n=2, noise=0.02, label="caregiver")
    tr.ramp(0.5, 0.15, FAR, n=2, disp0=0.1, disp1=0.0)
    tr.hold(3.0, FAR, n=2, label="idle")
    return Scenario("caregiver_adjust", tr.build(), max_state=State.WATCH, expect_suppressed=True,
                    expect_final=State.IDLE, labels=tr.labels)


def blanket_over_face(rng: np.random.Generator) -> Scenario:
    tr = Trajectory(rng)
    tr.hold(2.0, FAR, label="idle")
    tr.hold(rng.uniform(6, 10), FAR, conf=0.1, label="occlusion")
    return Scenario("blanket_over_face", tr.build(), must_reach=State.FAULT, max_state=State.FAULT,
                    expect_final=State.FAULT, labels=tr.labels)


def landmark_dropout_during_grasp(rng: np.random.Generator) -> Scenario:
    tr = Trajectory(rng)
    tr.hold(2.0, FAR, label="idle")
    tr.ramp(0.5, FAR, 0.15)
    tr.flicker(rng.uniform(4.0, 5.0), 0.15, pinch=PINCH_GRIP, period=0.5, label="grasp")
    tr.ramp(0.5, 0.15, FAR, pinch=OPEN_HAND)
    tr.hold(3.0, FAR, label="idle")
    return Scenario("landmark_dropout_during_grasp", tr.build(), must_reach=State.WARN, max_state=State.WARN,
                    expect_final=State.IDLE, labels=tr.labels)


def tape_pick(rng: np.random.Generator) -> Scenario:
    """Hand picks at the cheek tape: still, pinched, 2-5 s. Tube itself stays ~0.7 away. WARN, never ALARM."""
    tr = Trajectory(rng)
    tr.hold(2.0, FAR, tape=FAR, label="idle")
    tr.ramp(0.5, FAR, 0.7, tape0=FAR, tape1=0.1)
    tr.hold(rng.uniform(2.0, 5.0), 0.7, pinch=PINCH_GRIP, tape=0.1, label="tape")
    tr.ramp(0.4, 0.7, FAR, pinch=PINCH_GRIP, tape0=0.1, tape1=FAR)  # leaves still pinched: not a pull
    tr.hold(3.0, FAR, tape=FAR, label="idle")
    return Scenario("tape_pick", tr.build(), must_reach=State.WARN, max_state=State.WARN, expect_final=State.IDLE, labels=tr.labels)


def cheek_scratch(rng: np.random.Generator) -> Scenario:
    """Scratch at the cheek tape: oscillating, open fingers. WATCH only."""
    tr = Trajectory(rng)
    tr.hold(2.0, FAR, tape=FAR, label="idle")
    tr.ramp(0.4, FAR, 0.7, tape0=FAR, tape1=0.15)
    tr.oscillate(rng.uniform(0.5, 3.0), center=0.7, amp=0.05, freq=rng.uniform(3, 5), tape=0.15, tape_amp=rng.uniform(0.08, 0.12), label="scratch")
    tr.ramp(0.4, 0.7, FAR, tape0=0.15, tape1=FAR)
    tr.hold(3.0, FAR, tape=FAR, label="idle")
    return Scenario("cheek_scratch", tr.build(), max_state=State.WATCH, expect_final=State.IDLE, labels=tr.labels)


def tape_then_tube_pull(rng: np.random.Generator) -> Scenario:
    """Peel the tape, then grasp and pull the tube. WARN early (tape), ALARM on the pull."""
    tr = Trajectory(rng)
    tr.hold(2.0, FAR, tape=FAR, label="idle")
    tr.ramp(0.5, FAR, 0.7, tape0=FAR, tape1=0.1)
    tr.hold(rng.uniform(2.0, 4.0), 0.7, pinch=PINCH_GRIP, tape=0.1, label="tape")
    tr.ramp(0.4, 0.7, 0.15, pinch=OPEN_HAND, tape0=0.1, tape1=0.5)
    tr.hold(rng.uniform(1.7, 2.5), 0.15, pinch=PINCH_GRIP, tape=0.5, label="grasp")
    tr.ramp(rng.uniform(0.5, 1.0), 0.15, 1.0, pinch=PINCH_GRIP, disp0=0.0, disp1=0.5, tape0=0.5, tape1=1.2, label="pull")
    tr.hold(2.0, 1.0, pinch=PINCH_GRIP, disp=0.5, tape=1.2)
    return Scenario("tape_then_tube_pull", tr.build(), must_reach=State.ALARM, expect_final=State.ALARM, labels=tr.labels)


SCENARIOS: dict[str, Callable[[np.random.Generator], Scenario]] = {
    "tape_pick": tape_pick,
    "cheek_scratch": cheek_scratch,
    "tape_then_tube_pull": tape_then_tube_pull,
    "idle": idle,
    "nose_scratch_short": nose_scratch_short,
    "long_face_rub": long_face_rub,
    "wipe_mouth": wipe_mouth,
    "grasp_hold_release": grasp_hold_release,
    "grasp_and_pull": grasp_and_pull,
    "fast_yank": fast_yank,
    "caregiver_adjust": caregiver_adjust,
    "blanket_over_face": blanket_over_face,
    "landmark_dropout_during_grasp": landmark_dropout_during_grasp,
}


def generate(name: str, seed: int) -> Scenario:
    return SCENARIOS[name](np.random.default_rng(seed))


# ------------------------------------------------------------------- runner

@dataclass
class Outcome:
    final: State
    reached: set[State]
    events: list[Event]
    suppressed: bool

    @property
    def max_state(self) -> State:
        return max(self.reached, key=lambda s: SEVERITY[s])


def run_scenario(sc: Scenario, th: Thresholds | None = None) -> Outcome:
    m = EventMachine(th)
    reached = {State.IDLE}
    events: list[Event] = []
    for f in sc.features:
        evs = m.step(f)
        events.extend(evs)
        reached.add(m.state)
    return Outcome(m.state, reached, events, any(e.kind == "suppressed" for e in events))


def check(sc: Scenario, out: Outcome) -> list[str]:
    """Return a list of violated expectations (empty = pass)."""
    problems = []
    if SEVERITY[out.max_state] > SEVERITY[sc.max_state] or (
        out.max_state is State.FAULT and sc.max_state is not State.FAULT
    ):
        problems.append(f"reached {out.max_state.value}, max allowed {sc.max_state.value}")
    if sc.must_reach is not None and sc.must_reach not in out.reached:
        problems.append(f"never reached {sc.must_reach.value}")
    if sc.expect_final is not None and out.final is not sc.expect_final:
        problems.append(f"final state {out.final.value}, expected {sc.expect_final.value}")
    if sc.expect_suppressed and not out.suppressed:
        problems.append("expected a suppressed event")
    return problems


def _main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Run synthetic scenarios through the state machine.")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), help="single scenario to print in detail")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=50, help="seeds per scenario in summary mode")
    ap.add_argument("--config", help="config.yaml with a thresholds: section")
    args = ap.parse_args()
    th = Thresholds.from_yaml(args.config) if args.config else Thresholds()

    if args.scenario:
        sc = generate(args.scenario, args.seed)
        out = run_scenario(sc, th)
        print(f"# {sc.name} seed={args.seed}  frames={len(sc.features)}  duration={sc.features[-1].t:.1f}s")
        for t0, t1, lab in sc.labels:
            print(f"  label {t0:6.2f}-{t1:6.2f}s {lab}")
        print("events:")
        for e in out.events:
            print("  " + str(e))
        print(f"final={out.final.value} max={out.max_state.value}")
        problems = check(sc, out)
        print("PASS" if not problems else "FAIL: " + "; ".join(problems))
        return

    print(f"{'scenario':32} {'runs':>4} {'pass':>4}  final-state histogram")
    for name in SCENARIOS:
        hist: dict[str, int] = {}
        passed = 0
        for seed in range(args.n):
            sc = generate(name, seed)
            out = run_scenario(sc, th)
            hist[out.final.value] = hist.get(out.final.value, 0) + 1
            passed += not check(sc, out)
        print(f"{name:32} {args.n:4d} {passed:4d}  {hist}")


if __name__ == "__main__":  # pragma: no cover
    _main()
