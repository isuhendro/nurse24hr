"""Unit tests for the pure state machine, built from hand-written frames."""

from pathlib import Path

import pytest

from nurse24hr import EventMachine, Features, State, Thresholds
from nurse24hr.events import run

FPS = 15
DT = 1 / FPS
TH = Thresholds()


def frames(spec: list[tuple[float, dict]], t0: float = 0.0) -> list[Features]:
    """Build frames from (duration, field overrides) segments at 15 fps."""
    out: list[Features] = []
    t = t0
    for dur, kw in spec:
        base = {"d": 2.0, "conf": 0.95, "n_persons": 1, "v_radial": 0.0, "pinch": 0.4, "osc": 0.005}
        base.update(kw)
        for _ in range(max(1, round(dur * FPS))):
            out.append(Features(t=t, **base))
            t += DT
    return out


def states(fs: list[Features], th: Thresholds = TH) -> list[State]:
    m = EventMachine(th)
    seq = []
    for f in fs:
        m.step(f)
        seq.append(m.state)
    return seq


# ------------------------------------------------------------ basic flow

def test_idle_stays_idle():
    m, evs = run(frames([(10, {"d": 1.8})]))
    assert m.state is State.IDLE
    assert evs == []


def test_hand_near_enters_watch():
    seq = states(frames([(1, {"d": 1.8}), (1, {"d": 0.5})]))
    assert seq[-1] is State.WATCH


def test_watch_to_warn_needs_dwell_and_evidence():
    # still hand at the tube for 2 s -> WARN
    seq = states(frames([(1, {"d": 1.8}), (2, {"d": 0.2, "osc": 0.01, "pinch": 0.4})]))
    assert seq[-1] is State.WARN
    # same dwell but oscillating with open fingers (a scratch) -> stays WATCH
    seq = states(frames([(1, {"d": 1.8}), (2, {"d": 0.2, "osc": 0.09, "pinch": 0.4})]))
    assert seq[-1] is State.WATCH
    # oscillating but pinched -> WARN (pinch is grasp evidence)
    seq = states(frames([(1, {"d": 1.8}), (2, {"d": 0.2, "osc": 0.09, "pinch": 0.05})]))
    assert seq[-1] is State.WARN


def test_warn_before_t_grasp_does_not_fire():
    seq = states(frames([(1, {"d": 1.8}), (1.2, {"d": 0.2})]))
    assert State.WARN not in seq


def test_grasp_dwell_holds_in_watch_band_and_resets_outside_it():
    # 1 s at tube, 0.5 s hovering just outside the grasp band, 1 s at tube: dwell held -> WARN
    seq = states(frames([(1, {"d": 1.8}), (1, {"d": 0.2}), (0.5, {"d": 0.45}), (1, {"d": 0.2})]))
    assert seq[-1] is State.WARN
    # same but the hand goes beyond the watch band in between: dwell reset -> no WARN
    seq = states(frames([(1, {"d": 1.8}), (1, {"d": 0.2}), (0.5, {"d": 0.7}), (1, {"d": 0.2})]))
    assert State.WARN not in seq


def test_repeated_reaching_raises_warn():
    reach = [(1.0, {"d": 0.5}), (3.0, {"d": 1.8})]  # in, then away long enough to return to IDLE
    seq = states(frames([(1, {"d": 1.8})] + reach * 3))
    assert State.WARN in seq
    idx = seq.index(State.WARN)
    # fires on the third entry, not earlier
    assert idx * DT >= 1 + 2 * 4.0 - DT
    th = Thresholds(reach_count_warn=0)
    assert State.WARN not in states(frames([(1, {"d": 1.8})] + reach * 5), th)


def test_repeated_reaching_window_expires():
    th = Thresholds(reach_window_s=5.0)
    reach = [(1.0, {"d": 0.5}), (5.0, {"d": 1.8})]  # 6 s per cycle > window
    assert State.WARN not in states(frames([(1, {"d": 1.8})] + reach * 4), th)


def test_warn_to_alarm_by_dwell():
    seq = states(frames([(1, {"d": 1.8}), (5, {"d": 0.2})]))
    assert seq[-1] is State.ALARM
    # not before t_grasp + t_alarm
    idx = seq.index(State.ALARM)
    assert idx * DT >= 1 + TH.t_grasp + TH.t_alarm - 2 * DT


def test_warn_to_alarm_by_pull_with_pinch():
    seq = states(frames([
        (1, {"d": 1.8}),
        (2, {"d": 0.2, "pinch": 0.05}),              # WARN
        (0.3, {"d": 0.6, "pinch": 0.05, "v_radial": 1.5}),  # moving away, still pinched
    ]))
    assert seq[-1] is State.ALARM


def test_fast_hand_away_without_pinch_is_a_release_not_a_pull():
    seq = states(frames([
        (1, {"d": 1.8}),
        (2, {"d": 0.2, "pinch": 0.05}),              # WARN
        (0.3, {"d": 0.6, "pinch": 0.4, "v_radial": 1.5}),  # fingers open, hand leaves
        (3, {"d": 1.8}),
    ]))
    assert State.ALARM not in seq
    assert seq[-1] is State.IDLE


def test_tube_displacement_alarms_from_watch_without_warn():
    seq = states(frames([(1, {"d": 1.8}), (0.3, {"d": 0.2}), (0.2, {"d": 0.2, "tube_disp": 0.5})]))
    assert State.WARN not in seq
    assert seq[-1] is State.ALARM


def test_tube_displacement_alarms_even_from_idle():
    # hand not seen near the tube, but the tube moved: still alarm (recall first)
    seq = states(frames([(1, {"d": 1.8}), (0.2, {"d": 1.8, "tube_disp": 0.4})]))
    assert seq[-1] is State.ALARM


# ------------------------------------------------------------ hysteresis

def test_release_returns_to_idle_after_t_release():
    seq = states(frames([(1, {"d": 1.8}), (1, {"d": 0.5}), (3, {"d": 1.5})]))
    assert seq[-1] is State.IDLE
    first_idle_after = next(i for i, s in enumerate(seq[30:], start=30) if s is State.IDLE)
    assert (first_idle_after - 30) * DT >= TH.t_release - DT


def test_hand_between_watch_and_release_bands_holds_watch():
    seq = states(frames([(1, {"d": 1.8}), (1, {"d": 0.5}), (5, {"d": 0.7})]))
    assert seq[-1] is State.WATCH


# ------------------------------------------------------------ latching

def test_alarm_latches_until_ack():
    fs = frames([(1, {"d": 1.8}), (5, {"d": 0.2}), (10, {"d": 2.5})])
    m, _ = run(fs)
    assert m.state is State.ALARM
    evs = m.ack(fs[-1].t + 1)
    assert m.state is State.IDLE
    assert evs[0].kind == "ack" and evs[0].prev is State.ALARM


def test_ack_when_not_latched_is_noop():
    m = EventMachine()
    assert m.ack(0.0) == []
    assert m.state is State.IDLE


# ------------------------------------------------------------ confidence

def test_low_confidence_holds_state_and_timers():
    fs = frames([
        (1, {"d": 1.8}),
        (1, {"d": 0.2}),                  # 1.0 s grasp dwell
        (3, {"d": 0.2, "conf": 0.1}),     # low conf: hold, do not advance or reset
        (0.3, {"d": 0.2}),                # 1.3 s total: still not WARN
    ])
    seq = states(fs)
    assert seq[-1] is State.WATCH
    assert State.FAULT not in seq
    # a bit more good time pushes it over
    m, _ = run(fs + frames([(0.4, {"d": 0.2})], t0=fs[-1].t + DT))
    assert m.state is State.WARN


def test_sustained_low_confidence_raises_fault():
    seq = states(frames([(1, {"d": 1.8}), (6, {"d": 1.8, "conf": 0.1})]))
    assert seq[-1] is State.FAULT
    idx = seq.index(State.FAULT)
    assert abs(idx * DT - (1 + TH.t_fault)) < 3 * DT


def test_missing_d_counts_as_low_confidence():
    seq = states(frames([(6, {"d": None, "conf": 0.9})]))
    assert seq[-1] is State.FAULT


def test_fault_latches_and_acks():
    fs = frames([(6, {"conf": 0.1}), (5, {"d": 1.8})])
    m, _ = run(fs)
    assert m.state is State.FAULT
    m.ack(fs[-1].t)
    assert m.state is State.IDLE


def test_alarm_takes_precedence_over_later_occlusion():
    m, _ = run(frames([(1, {"d": 1.8}), (5, {"d": 0.2}), (10, {"conf": 0.1})]))
    assert m.state is State.ALARM


def test_missing_evidence_defaults_to_counting():
    seq = states(frames([(1, {"d": 1.8}), (2, {"d": 0.2, "osc": None, "pinch": None})]))
    assert seq[-1] is State.WARN
    th = Thresholds(missing_evidence_counts=False)
    seq = states(frames([(1, {"d": 1.8}), (2, {"d": 0.2, "osc": None, "pinch": None})]), th)
    assert seq[-1] is State.WATCH


# ------------------------------------------------------------ tape on the cheek

def test_hand_dwelling_at_tape_raises_warn_but_never_alarm():
    seq = states(frames([(1, {"d": 1.8, "d_tape": 1.8}), (8, {"d": 0.7, "d_tape": 0.1, "pinch": 0.05})]))
    assert State.WARN in seq
    assert State.ALARM not in seq
    assert seq[-1] is State.WARN


def test_tape_warn_needs_grasp_evidence():
    # scratching at the cheek: oscillating, open fingers -> WATCH only
    seq = states(frames([(1, {"d": 1.8, "d_tape": 1.8}), (4, {"d": 0.7, "d_tape": 0.1, "osc": 0.09, "pinch": 0.4})]))
    assert seq[-1] is State.WATCH and State.WARN not in seq


def test_pinched_hand_leaving_tape_is_not_a_pull():
    seq = states(frames([
        (1, {"d": 1.8, "d_tape": 1.8}),
        (3, {"d": 0.7, "d_tape": 0.1, "pinch": 0.05}),                       # WARN from tape
        (0.3, {"d": 1.2, "d_tape": 0.8, "pinch": 0.05, "v_radial": 2.0}),   # fast move away, still pinched
        (3, {"d": 1.8, "d_tape": 1.8}),
    ]))
    assert State.WARN in seq and State.ALARM not in seq
    assert seq[-1] is State.IDLE


def test_pull_is_still_caught_right_after_leaving_the_tube():
    seq = states(frames([
        (1, {"d": 1.8}),
        (2, {"d": 0.2, "pinch": 0.05}),
        (0.2, {"d": 0.5, "pinch": 0.05, "v_radial": 2.0}),  # within pull_arm_s of being at the tube
    ]))
    assert seq[-1] is State.ALARM


def test_release_requires_hand_away_from_tape_too():
    seq = states(frames([(1, {"d": 1.8, "d_tape": 1.8}), (1, {"d": 0.5, "d_tape": 0.5}), (4, {"d": 1.5, "d_tape": 0.5})]))
    assert seq[-1] is State.WATCH  # still at the tape, so no hysteresis reset


def test_tape_logic_can_be_disabled():
    th = Thresholds(tape_warn=False)
    seq = states(frames([(1, {"d": 1.8, "d_tape": 1.8}), (8, {"d": 1.8, "d_tape": 0.1, "pinch": 0.05})]), th)
    assert seq[-1] is State.IDLE


# ------------------------------------------------------------ suppression

def test_second_person_suppresses_warn_and_alarm_but_logs_it():
    fs = frames([(1, {"d": 1.8, "n_persons": 2}), (6, {"d": 0.2, "pinch": 0.05, "n_persons": 2})])
    m, evs = run(fs)
    assert m.state is State.WATCH
    sup = [e for e in evs if e.kind == "suppressed"]
    assert len(sup) == 1, "suppression should be reported once, not every frame"
    assert "WARN" in sup[0].reason


def test_suppression_lifts_when_caregiver_leaves():
    seq = states(frames([
        (1, {"d": 1.8, "n_persons": 2}),
        (3, {"d": 0.2, "pinch": 0.05, "n_persons": 2}),
        (0.5, {"d": 0.2, "pinch": 0.05, "n_persons": 1}),
    ]))
    assert seq[-1] is State.WATCH, "count must read one for t_single_person first"
    seq = states(frames([
        (1, {"d": 1.8, "n_persons": 2}),
        (3, {"d": 0.2, "pinch": 0.05, "n_persons": 2}),
        (TH.t_single_person + 0.2, {"d": 0.2, "pinch": 0.05, "n_persons": 1}),
    ]))
    assert seq[-1] is State.WARN


def test_flickering_person_count_keeps_suppressing():
    # A caregiver leaning over the bed: the per-frame count alternates 2, 1, 2, 1 ...
    fs = frames([(1, {"d": 1.8})])
    t = fs[-1].t
    for i in range(15 * 10):
        t += DT
        fs.append(Features(t=t, d=0.2, conf=0.95, n_persons=2 if i % 2 else 1, pinch=0.05, osc=0.0))
    m, evs = run(fs)
    assert m.state is State.WATCH
    assert all(e.kind == "suppressed" for e in evs if e.state in (State.WARN, State.ALARM))
    assert "second person seen" in next(e.reason for e in evs if e.kind == "suppressed")


def test_warn_dwell_does_not_advance_while_suppressed():
    th = Thresholds(t_single_person=0.0)  # isolate the clock from the debounce
    seq = states(frames([
        (1, {"d": 1.8}),
        (2, {"d": 0.2, "pinch": 0.05}),              # -> WARN
        (5, {"d": 0.2, "pinch": 0.05, "n_persons": 2}),  # caregiver at the tube: clock frozen
        (2, {"d": 0.2, "pinch": 0.05}),              # alone again, 2 s < t_alarm
    ]), th)
    assert seq[-1] is State.WARN
    seq2 = states(frames([
        (1, {"d": 1.8}),
        (2, {"d": 0.2, "pinch": 0.05}),
        (5, {"d": 0.2, "pinch": 0.05, "n_persons": 2}),
        (TH.t_alarm + 0.3, {"d": 0.2, "pinch": 0.05}),
    ]), th)
    assert seq2[-1] is State.ALARM


def test_person_debounce_can_be_disabled():
    th = Thresholds(t_single_person=0.0)
    seq = states(frames([
        (1, {"d": 1.8, "n_persons": 2}),
        (3, {"d": 0.2, "pinch": 0.05, "n_persons": 2}),
        (0.2, {"d": 0.2, "pinch": 0.05, "n_persons": 1}),
    ]), th)
    assert seq[-1] is State.WARN


# ------------------------------------------------------------ missing evidence and the ALARM clock

def test_dwell_alarm_needs_real_evidence_at_least_once():
    none = {"d": 0.2, "osc": None, "pinch": None}
    m, evs = run(frames([(1, {"d": 1.8}), (8, none)]))
    assert m.state is State.WARN, "WARN by dwell alone, but no ALARM on coarse hands"
    held = [e for e in evs if e.kind == "suppressed"]
    assert len(held) == 1 and "ALARM held" in held[0].reason
    # one real pinch reading during the WARN is enough
    seq = states(frames([(1, {"d": 1.8}), (3, none), (0.2, {"d": 0.2, "pinch": 0.05, "osc": None}), (4, none)]))
    assert seq[-1] is State.ALARM
    # the old behaviour is still available
    seq = states(frames([(1, {"d": 1.8}), (8, none)]), Thresholds(missing_evidence_alarms=True))
    assert seq[-1] is State.ALARM


def test_warn_dwell_resets_after_hand_leaves_watch_band():
    seq = states(frames([
        (1, {"d": 1.8}),
        (2, {"d": 0.2, "pinch": 0.05}),      # WARN, ~0.5 s of dwell
        (TH.t_warn_reset + 0.3, {"d": 0.7}),  # out of the watch band, not far enough for release
        (2.3, {"d": 0.2, "pinch": 0.05}),    # back: fresh clock, 2.3 s < t_alarm
    ]))
    assert seq[-1] is State.WARN
    seq = states(frames([
        (1, {"d": 1.8}),
        (2, {"d": 0.2, "pinch": 0.05}),      # ~0.5 s of WARN dwell
        (0.5, {"d": 0.7}),                   # brief excursion keeps the clock
        (2.8, {"d": 0.2, "pinch": 0.05}),    # 0.5 + 2.8 > t_alarm
    ]))
    assert seq[-1] is State.ALARM


# ------------------------------------------------------------ robustness

def test_timestamp_gap_is_capped():
    # a 10 s stall between two frames must not count as 10 s of dwell
    fs = frames([(1, {"d": 1.8}), (0.5, {"d": 0.2})])
    late = Features(t=fs[-1].t + 10.0, d=0.2, conf=0.95, pinch=0.05, osc=0.005)
    m, _ = run(fs + [late])
    assert m.state is State.WATCH
    assert m.grasp_dwell <= 0.5 + TH.max_dt + DT


def test_deterministic_replay():
    fs = frames([(1, {"d": 1.8}), (2, {"d": 0.2, "pinch": 0.05}), (1, {"d": 0.7, "v_radial": 1.2, "pinch": 0.05})])
    _, a = run(fs)
    _, b = run(fs)
    assert a == b


# ------------------------------------------------------------ config

def test_config_yaml_matches_defaults():
    th = Thresholds.from_yaml(Path(__file__).resolve().parents[1] / "config.yaml")
    assert th == Thresholds()


def test_thresholds_validate_hysteresis_order():
    with pytest.raises(ValueError):
        Thresholds(d_watch=0.2, d_grasp=0.3)


def test_thresholds_reject_unknown_keys():
    with pytest.raises(KeyError):
        Thresholds.from_dict({"d_watchh": 0.6})
