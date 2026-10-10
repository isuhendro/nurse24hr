"""Scenario suite: every synthetic scenario across many seeds (TS-001 §7.2).

The one assertion that matters most: no scratch or rub scenario ever reaches
ALARM. The one recall assertion that matters most: every pull reaches ALARM.
"""

import pytest

from nurse24hr import State, Thresholds
from nurse24hr.simulate import SCENARIOS, Trajectory, check, generate, run_scenario, FAR, PINCH_GRIP
import numpy as np

SEEDS = range(40)
NEGATIVE = ["idle", "nose_scratch_short", "long_face_rub", "wipe_mouth", "caregiver_adjust"]
POSITIVE = ["grasp_and_pull", "fast_yank"]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_meets_expectations(name, seed):
    sc = generate(name, seed)
    out = run_scenario(sc)
    problems = check(sc, out)
    assert not problems, f"{name} seed={seed}: " + "; ".join(problems) + "\n" + "\n".join(map(str, out.events))


@pytest.mark.parametrize("name", NEGATIVE)
def test_negatives_never_alarm(name):
    for seed in range(200):
        out = run_scenario(generate(name, seed))
        assert State.ALARM not in out.reached, f"{name} seed={seed} alarmed:\n" + "\n".join(map(str, out.events))


@pytest.mark.parametrize("name", POSITIVE)
def test_positives_always_alarm(name):
    for seed in range(200):
        out = run_scenario(generate(name, seed))
        assert State.ALARM in out.reached, f"{name} seed={seed} missed:\n" + "\n".join(map(str, out.events))


def test_time_to_alarm_within_budget():
    """From the start of the pull segment to ALARM must be well under 3 s."""
    delays = []
    for seed in range(100):
        sc = generate("grasp_and_pull", seed)
        out = run_scenario(sc)
        pull_start = next(t0 for t0, _, lab in sc.labels if lab == "pull")
        alarm_t = next(e.t for e in out.events if e.state is State.ALARM)
        delays.append(alarm_t - pull_start)
    assert float(np.median(delays)) <= 1.0
    assert max(delays) <= 3.0


def test_scratch_without_evidence_features_still_never_alarms():
    """If the hand model fails (no osc / pinch), a scratch may reach WARN but not ALARM."""
    for seed in range(100):
        rng = np.random.default_rng(seed)
        tr = Trajectory(rng)
        tr.hold(2.0, FAR)
        tr.ramp(0.4, FAR, 0.2)
        tr.oscillate(rng.uniform(1, 4), center=0.2, amp=0.1, freq=4)
        tr.ramp(0.4, 0.2, FAR)
        tr.hold(3.0, FAR)
        from nurse24hr.simulate import Scenario
        sc = Scenario("scratch_no_evidence", tr.build(with_evidence=False), max_state=State.WARN)
        out = run_scenario(sc)
        assert State.ALARM not in out.reached


def test_pull_in_v1_without_tube_detector_still_alarms():
    """v1 has no tube_disp. The pinch + radial-motion rule must carry the pull on its own."""
    for seed in range(100):
        rng = np.random.default_rng(seed)
        tr = Trajectory(rng)
        tr.hold(2.0, FAR)
        tr.ramp(0.5, FAR, 0.15)
        tr.hold(rng.uniform(1.7, 3.0), 0.15, pinch=PINCH_GRIP)
        tr.ramp(rng.uniform(0.5, 1.0), 0.15, 1.0, pinch=PINCH_GRIP)
        tr.hold(2.0, 1.0, pinch=PINCH_GRIP)
        out = run_scenario(type(generate("idle", 0))("pull_v1", tr.build(with_tube_disp=False)))
        assert State.ALARM in out.reached, f"seed={seed}\n" + "\n".join(map(str, out.events))


@pytest.mark.xfail(strict=True, reason="Known v1 gap: a sub-1.5 s yank without a tube detector cannot be distinguished "
                                       "from a brief touch. Needs tube_disp (v2 detector) or a learned action classifier.")
def test_fast_yank_in_v1_without_tube_detector():
    for seed in range(20):
        sc = generate("fast_yank", seed)
        rng = np.random.default_rng(seed)
        tr = Trajectory(rng)
        tr.hold(2.0, FAR)
        tr.ramp(0.4, FAR, 0.1)
        tr.hold(0.5, 0.1, pinch=PINCH_GRIP)
        tr.ramp(0.3, 0.1, 1.2, pinch=PINCH_GRIP)
        tr.hold(2.0, 1.2, pinch=PINCH_GRIP)
        out = run_scenario(type(sc)("fast_yank_v1", tr.build(with_tube_disp=False)))
        assert State.ALARM in out.reached


def test_thresholds_are_not_at_a_cliff():
    """Small threshold perturbations must not flip the core outcomes."""
    for scale in (0.8, 1.25):
        th = Thresholds(
            d_grasp=0.3 * scale, d_watch=0.6 * scale, d_release=0.8 * scale,
            t_grasp=1.5 * scale, osc_max=0.05 * scale, pinch_max=0.12 * scale, v_pull=0.8 * scale,
        )
        for seed in range(30):
            assert State.ALARM not in run_scenario(generate("nose_scratch_short", seed), th).reached
            assert State.ALARM in run_scenario(generate("grasp_and_pull", seed), th).reached
