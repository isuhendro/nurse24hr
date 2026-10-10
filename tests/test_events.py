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


def test_grasp_dwell_resets_when_hand_leaves_grasp_band():
    # 1 s at tube, 0.5 s just outside, 1 s at tube: never 1.5 s continuous
    seq = states(frames([(1, {"d": 1.8}), (1, {"d": 0.2}), (0.5, {"d": 0.45}), (1, {"d": 0.2})]))
    assert State.WARN not in seq


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
    assert seq[-1] is State.WARN


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
