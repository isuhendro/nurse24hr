"""Pure event state machine. No I/O, no clocks, no threads.

Feed it one :class:`Features` per frame via :meth:`EventMachine.step` and it
returns the events that frame produced. Time comes only from ``Features.t``,
so the same feature sequence always yields the same events, live or replayed.

States
------
IDLE   nothing near the tube
WATCH  a hand is within ``d_watch`` of the tube / insertion point
WARN   a hand has been at the tube (``d < d_grasp``) for ``t_grasp`` with
       grasp evidence (still hand or pinch grip). Station notification.
ALARM  pull-out in progress or long grasp. Latched until ack.
FAULT  cannot see the patient for ``t_fault``. Latched until ack.

Scratch vs pull
---------------
A face scratch puts the hand at the insertion point too, so proximity and
dwell alone never alarm. WARN additionally needs grasp evidence (low
oscillation or a pinch grip); ALARM additionally needs pull motion while
pinched, tube displacement, or a long WARN dwell. Scratching oscillates with
extended fingers and never moves the tube, so it stops at WATCH.

Tape on the cheek
-----------------
An NG tube is secured with tape on the cheek, and patients usually pick at
that tape before pulling. ``d_tape`` is the hand's distance to the tape
anchor. Dwelling there with grasp evidence raises WARN (reason "hand at
tape"); it never reaches ALARM on its own, because WARN dwell and pull motion
are both tied to the tube anchor. ``d_tape`` of ``None`` means "unknown", and
is treated as far.

Repeated reaching
-----------------
A patient who keeps bringing a hand to the face is agitated and at risk even
if no single reach lasts long enough to be a grasp. ``reach_count_warn``
entries into WATCH within ``reach_window_s`` raise WARN ("repeated reaching").

Two pull signatures are handled because perception can supply ``tube_pt`` in
two ways:

* v1, insertion-point proxy (nostril from face landmarks): during a pull the
  hand moves away from the nostril, so ``d`` rises fast while ``pinch`` is
  held. Caught by the pull-motion rule.
* v2, tube detector: ``tube_pt`` follows the tube, so ``d`` stays small while
  ``tube_disp`` grows. Caught by the displacement rule, from any state.

Missing evidence
----------------
Coarse hands from the body model carry no ``osc`` / ``pinch``. With
``missing_evidence_counts`` dwell alone may raise WARN, but the WARN-dwell
ALARM is held until real grasp evidence has been seen at least once during
the WARN (``missing_evidence_alarms`` lifts this). Body-model wrists drift
onto the face when the real hands are hidden, so dwell on them is weak.

Second person
-------------
``n_persons > 1`` suppresses WARN / ALARM. The per-frame count flickers while
a caregiver leans over the bed, so after a second person was seen the count
must read one for ``t_single_person`` before escalation resumes, and the
WARN dwell clock does not advance while suppressed.

Low confidence
--------------
A low-confidence frame (``conf < conf_min`` or ``d is None``) holds state:
no timer advances or resets. Sustained low confidence for ``t_fault`` raises
FAULT. FAULT is an alarm, never a silence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nurse24hr.config import Thresholds
from nurse24hr.features import Features


class State(str, Enum):
    IDLE = "IDLE"
    WATCH = "WATCH"
    WARN = "WARN"
    ALARM = "ALARM"
    FAULT = "FAULT"

    @property
    def latched(self) -> bool:
        return self in (State.ALARM, State.FAULT)


SEVERITY: dict[State, int] = {
    State.IDLE: 0,
    State.WATCH: 1,
    State.WARN: 2,
    State.ALARM: 3,
    State.FAULT: 3,
}


@dataclass(frozen=True, slots=True)
class Event:
    t: float
    kind: str
    """``transition`` | ``suppressed`` | ``ack``"""
    prev: State
    state: State
    reason: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if self.kind == "transition":
            return f"{self.t:8.2f}s  {self.prev.value:>5} -> {self.state.value:<5}  {self.reason}"
        return f"{self.t:8.2f}s  [{self.kind}] in {self.state.value}: {self.reason}"


class EventMachine:
    """Deterministic grasp / pull detector over a stream of :class:`Features`."""

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self.th = thresholds or Thresholds()
        self.state: State = State.IDLE
        self._prev_t: float | None = None
        self._prev_disp: float | None = None
        self.grasp_dwell = 0.0
        """Continuous time with ``d < d_grasp`` (good frames only)."""
        self.warn_dwell = 0.0
        """Time in WARN with the hand at the tube (``d < d_grasp``, good frames only)."""
        self.release_dwell = 0.0
        """Continuous time with ``d > d_release`` (good frames only)."""
        self.lowconf_dwell = 0.0
        """Continuous low-confidence time."""
        self.tape_dwell = 0.0
        """Continuous time with ``d_tape < d_grasp`` (good frames only)."""
        self.off_dwell = 0.0
        """Continuous time with the hand outside the watch band (good frames only)."""
        self.single_dwell = self.th.t_single_person
        """Continuous time with ``n_persons <= 1``. Starts settled: nobody extra seen yet."""
        self._last_on_tube_t: float | None = None
        self._suppressed_notified = False
        self._held_notified = False
        self._warn_evidence = False
        """Real grasp evidence (osc / pinch) seen since entering WARN."""
        self._watch_entries: list[float] = []

    # ------------------------------------------------------------------ public

    def step(self, f: Features) -> list[Event]:
        """Consume one frame. Returns the events it produced (usually none)."""
        th = self.th
        events: list[Event] = []

        dt = 0.0 if self._prev_t is None else min(max(f.t - self._prev_t, 0.0), th.max_dt)
        self._prev_t = f.t

        if self.state.latched:
            return events

        good = f.conf >= th.conf_min and f.d is not None
        if not good:
            self.lowconf_dwell += dt
            if self.lowconf_dwell >= th.t_fault:
                events.append(
                    self._transition(f.t, State.FAULT, f"no confident view of patient for {self.lowconf_dwell:.1f}s")
                )
            return events
        self.lowconf_dwell = 0.0

        d = f.d
        assert d is not None
        if f.n_persons > 1:
            self.single_dwell = 0.0
        else:
            self.single_dwell += dt
        suppressed = f.n_persons > 1 or self.single_dwell < th.t_single_person
        if not suppressed:
            self._suppressed_notified = False

        d_tape = f.d_tape if f.d_tape is not None else float("inf")

        # Dwell bookkeeping. Grasp/tape dwell accumulate inside the grasp band, hold while the
        # hand hovers in the watch band (jitter at the boundary must not wipe them), and reset
        # only once the hand is outside the watch band.
        away = d > th.d_release and d_tape > th.d_release
        self.release_dwell = self.release_dwell + dt if away else 0.0
        if d < th.d_grasp:
            self.grasp_dwell += dt
        elif d >= th.d_watch:
            self.grasp_dwell = 0.0
        if d_tape < th.d_grasp:
            self.tape_dwell += dt
        elif d_tape >= th.d_watch:
            self.tape_dwell = 0.0
        if d < th.d_grasp:
            self._last_on_tube_t = f.t
        self.off_dwell = self.off_dwell + dt if (d >= th.d_watch and d_tape >= th.d_watch) else 0.0
        if self.state is State.WARN:
            if d < th.d_grasp and not suppressed:
                # WARN dwell counts time *on the tube* with nobody else at the bed. A hand
                # hovering between the grasp and release bands neither advances nor resets it.
                self.warn_dwell += dt
            elif self.off_dwell >= th.t_warn_reset:
                # Hand left the tube for a while: a later return starts the ALARM clock afresh.
                self.warn_dwell = 0.0
            if self._real_evidence(f):
                self._warn_evidence = True

        # 1. Hysteresis reset: hand clearly away for long enough.
        if self.state in (State.WATCH, State.WARN) and self.release_dwell >= th.t_release:
            events.append(self._transition(f.t, State.IDLE, f"hand away for {self.release_dwell:.1f}s"))
            self._reset_dwell()
            self._prev_disp = f.tube_disp
            return events

        # 2. Tube displacement is strong evidence on its own, from any state.
        if f.tube_disp is not None and f.tube_disp > th.disp_max:
            self._escalate(f, State.ALARM, f"tube displaced {f.tube_disp:.2f} from baseline", suppressed, events)
            self._prev_disp = f.tube_disp
            return events

        if self.state is State.IDLE:
            if d < th.d_watch:
                events.append(self._transition(f.t, State.WATCH, f"hand within {d:.2f} of tube"))
            elif th.tape_warn and d_tape < th.d_watch:
                events.append(self._transition(f.t, State.WATCH, f"hand within {d_tape:.2f} of tape"))
            if self.state is State.WATCH:
                self._watch_entries = [t0 for t0 in self._watch_entries if f.t - t0 <= th.reach_window_s] + [f.t]
                if th.reach_count_warn and len(self._watch_entries) >= th.reach_count_warn:
                    n = len(self._watch_entries)
                    self._watch_entries = []
                    self._escalate(f, State.WARN, f"repeated reaching: {n} times in {th.reach_window_s:.0f}s", suppressed, events)

        elif self.state is State.WATCH:
            if self.grasp_dwell >= th.t_grasp and self._grasp_evidence(f):
                self._escalate(f, State.WARN, f"grasp held {self.grasp_dwell:.1f}s ({self._evidence_label(f)})", suppressed, events)
            elif th.tape_warn and self.tape_dwell >= th.t_grasp and self._grasp_evidence(f):
                self._escalate(f, State.WARN, f"hand at tape {self.tape_dwell:.1f}s ({self._evidence_label(f)})", suppressed, events)

        elif self.state is State.WARN:
            if self.warn_dwell >= th.t_alarm:
                if self._warn_evidence or th.missing_evidence_alarms:
                    self._escalate(f, State.ALARM, f"grasp dwell {self.warn_dwell:.1f}s", suppressed, events)
                elif not self._held_notified:
                    events.append(Event(f.t, "suppressed", self.state, self.state,
                                        f"ALARM held, no grasp evidence (pinch/osc) seen: grasp dwell {self.warn_dwell:.1f}s"))
                    self._held_notified = True
            elif self._pull_motion(f):
                self._escalate(f, State.ALARM, f"pull motion v_radial={f.v_radial:.2f}", suppressed, events)

        self._prev_disp = f.tube_disp
        return events

    def ack(self, t: float) -> list[Event]:
        """Nurse acknowledgement. Clears a latched ALARM / FAULT back to IDLE."""
        if not self.state.latched:
            return []
        prev = self.state
        self.state = State.IDLE
        self._reset_dwell()
        self.lowconf_dwell = 0.0
        return [Event(t, "ack", prev, State.IDLE, "acknowledged")]

    # ----------------------------------------------------------------- helpers

    def _grasp_evidence(self, f: Features) -> bool:
        th = self.th
        if f.osc is None and f.pinch is None:
            return th.missing_evidence_counts
        still = f.osc is not None and f.osc < th.osc_max
        pinched = f.pinch is not None and f.pinch < th.pinch_max
        return still or pinched

    def _real_evidence(self, f: Features) -> bool:
        """Grasp evidence from measured osc / pinch only; missing data never counts."""
        th = self.th
        still = f.osc is not None and f.osc < th.osc_max
        pinched = f.pinch is not None and f.pinch < th.pinch_max
        return still or pinched

    def _evidence_label(self, f: Features) -> str:
        parts = []
        if f.osc is not None and f.osc < self.th.osc_max:
            parts.append(f"still osc={f.osc:.3f}")
        if f.pinch is not None and f.pinch < self.th.pinch_max:
            parts.append(f"pinch={f.pinch:.2f}")
        return ", ".join(parts) or "no osc/pinch data"

    def _pull_motion(self, f: Features) -> bool:
        """Hand moving away from the tube while still gripping it (or while the tube follows).

        Only armed while the hand was at the tube within ``pull_arm_s``: a pinched
        hand leaving the cheek tape is not a pull.
        """
        th = self.th
        if f.v_radial is None or f.v_radial < th.v_pull:
            return False
        if self._last_on_tube_t is None or f.t - self._last_on_tube_t > th.pull_arm_s:
            return False
        pinched = f.pinch is not None and f.pinch < th.pinch_max
        disp_rising = (
            f.tube_disp is not None
            and self._prev_disp is not None
            and f.tube_disp > self._prev_disp
            and f.tube_disp > th.disp_max / 2
        )
        return pinched or disp_rising

    def _escalate(self, f: Features, target: State, reason: str, suppressed: bool, events: list[Event]) -> None:
        if suppressed:
            if not self._suppressed_notified:
                why = (f"{f.n_persons} persons in ROI" if f.n_persons > 1
                       else f"second person seen {self.single_dwell:.1f}s ago")
                events.append(Event(f.t, "suppressed", self.state, self.state, f"{target.value} suppressed, {why}: {reason}"))
                self._suppressed_notified = True
            return
        events.append(self._transition(f.t, target, reason))
        if target is State.WARN:
            self.warn_dwell = 0.0
            self._warn_evidence = self._real_evidence(f)
            self._held_notified = False

    def _transition(self, t: float, target: State, reason: str) -> Event:
        ev = Event(t, "transition", self.state, target, reason)
        self.state = target
        return ev

    def _reset_dwell(self) -> None:
        self.grasp_dwell = 0.0
        self.warn_dwell = 0.0
        self.release_dwell = 0.0
        self.tape_dwell = 0.0
        self.off_dwell = 0.0
        self._suppressed_notified = False
        self._held_notified = False
        self._warn_evidence = False


def run(features: list[Features], thresholds: Thresholds | None = None) -> tuple[EventMachine, list[Event]]:
    """Replay a whole feature sequence. Returns the machine and all events."""
    m = EventMachine(thresholds)
    out: list[Event] = []
    for f in features:
        out.extend(m.step(f))
    return m, out
