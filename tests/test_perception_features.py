"""FeatureDeriver tests: observation -> Features arithmetic, no MediaPipe, no camera."""

import math
from pathlib import Path

import numpy as np
import pytest

from nurse24hr.mp_perceiver import FAR, FaceGate, FeatureDeriver, HandObs, RawObservation, pose_near_rejected

FPS = 15
FACE_W = 200.0
NOSE = (320.0, 240.0)


CHEEK = (NOSE[0] + 0.45 * FACE_W, NOSE[1] + 0.1 * FACE_W)


def hand(grip, pinch_px=60.0, score=0.9, label="Right"):
    return HandObs(grip=grip, wrist=(grip[0], grip[1] + 80), pinch_px=pinch_px, score=score, label=label)


def obs(hands=(), face=True, n_faces=None):
    return RawObservation(tube_pt=NOSE if face else None, face_w=FACE_W if face else None,
                          hands=tuple(hands), n_faces=(1 if face else 0) if n_faces is None else n_faces,
                          tape_pts=(CHEEK,) if face else ())


def test_no_face_is_zero_confidence_and_no_distance():
    f = FeatureDeriver().update(obs(face=False), 0.0)
    assert f.conf == 0.0 and f.d is None


def test_face_without_hands_is_far_and_confident():
    f = FeatureDeriver().update(obs(), 0.0)
    assert f.d == FAR and f.conf >= 0.5 and f.pinch is None


def test_distance_and_pinch_are_normalised_by_face_width():
    dv = FeatureDeriver(median_frames=1)
    g = (NOSE[0] + 0.5 * FACE_W, NOSE[1])
    f = dv.update(obs([hand(g, pinch_px=0.1 * FACE_W)]), 0.0)
    assert f.d == pytest.approx(0.5)
    assert f.pinch == pytest.approx(0.1)


def test_closest_hand_is_used():
    dv = FeatureDeriver(median_frames=1)
    near = hand((NOSE[0] + 20, NOSE[1]), label="Left")
    far = hand((NOSE[0] + 300, NOSE[1]), label="Right")
    f = dv.update(obs([far, near]), 0.0)
    assert f.d == pytest.approx(0.1)
    assert f.wrist_l is not None and f.wrist_r is not None


def test_median_smoothing_kills_single_frame_spike():
    dv = FeatureDeriver(median_frames=5)
    ds = []
    for i in range(6):
        g = (NOSE[0] + (0.9 if i == 3 else 0.2) * FACE_W, NOSE[1])
        ds.append(dv.update(obs([hand(g)]), i / FPS).d)
    assert max(ds) < 0.5


def test_radial_velocity_sign_and_magnitude():
    dv = FeatureDeriver(median_frames=1, velocity_alpha=1.0)
    # hand moving away from the nose at 1 face-width per second
    for i in range(5):
        f = dv.update(obs([hand((NOSE[0] + (0.1 + i / FPS) * FACE_W, NOSE[1]))]), i / FPS)
    assert f.v_radial == pytest.approx(1.0, rel=0.05)
    # moving toward -> negative
    for j in range(1, 5):
        f = dv.update(obs([hand((NOSE[0] + (0.1 + (4 - j) / FPS) * FACE_W, NOSE[1]))]), (4 + j) / FPS)
    assert f.v_radial < -0.5


def test_oscillation_distinguishes_scratch_from_hold():
    still = FeatureDeriver(median_frames=1)
    scratch = FeatureDeriver(median_frames=1)
    for i in range(FPS * 2):
        t = i / FPS
        still_f = still.update(obs([hand((NOSE[0] + 0.2 * FACE_W + np.random.default_rng(i).normal(0, 0.5), NOSE[1]))]), t)
        amp = 0.1 * FACE_W * math.sin(2 * math.pi * 4 * t)
        scratch_f = scratch.update(obs([hand((NOSE[0] + 0.2 * FACE_W + amp, NOSE[1]))]), t)
    assert still_f.osc < 0.02
    assert scratch_f.osc > 0.05


def test_tape_distance_uses_nearest_tape_anchor_and_survives_face_memory():
    dv = FeatureDeriver(median_frames=1, tube_memory_s=2.0)
    f = dv.update(obs([hand((CHEEK[0] + 0.1 * FACE_W, CHEEK[1]))]), 0.0)
    assert f.d_tape == pytest.approx(0.1) and f.tape_pt == CHEEK
    assert f.d > 0.3  # the tube itself is not close
    f = dv.update(obs([hand((CHEEK[0] + 0.1 * FACE_W, CHEEK[1]))], face=False), 0.5)
    assert f.d_tape == pytest.approx(0.1) and f.conf == pytest.approx(0.6)
    assert dv.update(obs(), 3.5).d_tape == pytest.approx(3.0)  # no hands for > hand_memory_s: far
    assert dv.update(RawObservation(NOSE, FACE_W, (), 1), 6.0).d_tape is None  # face without tape anchors


def test_vanished_hand_is_held_then_released():
    dv = FeatureDeriver(median_frames=1, hand_memory_s=2.0)
    dv.update(obs([hand((NOSE[0] + 0.2 * FACE_W, NOSE[1]))]), 0.0)
    f = dv.update(obs(), 1.0)  # hand not seen: hold, reduced confidence, evidence unknown
    assert f.d == pytest.approx(0.2) and f.conf == pytest.approx(0.6)
    assert f.pinch is None and f.osc is None
    f = dv.update(obs(), 3.0)  # beyond memory: away
    assert f.d == FAR and f.conf == pytest.approx(0.9) and f.osc == 0.0


def test_coarse_pose_hand_reports_no_pinch_or_oscillation():
    dv = FeatureDeriver(median_frames=1)
    h = HandObs(grip=(NOSE[0] + 0.1 * FACE_W, NOSE[1]), wrist=(0, 0), pinch_px=5.0, score=0.9, label="Right", coarse=True)
    f = dv.update(obs([h]), 0.0)
    assert f.d == pytest.approx(0.1) and f.pinch is None and f.osc is None


def test_pose_hands_need_visibility():
    from nurse24hr.mp_perceiver import pose_observation

    lms = _pose_lms({16: _LM(0.30, 0.80, 0.35), 20: _LM(0.28, 0.85, 0.35)})  # subject's right wrist/index
    p = pose_observation(lms, 1, lambda lm: (lm.x * 1000, lm.y * 1000))
    assert p is not None and all(h.label != "Right" for h in p.hands), "0.35 is a hidden hand at the default cutoff"
    p = pose_observation(lms, 1, lambda lm: (lm.x * 1000, lm.y * 1000), hand_min_vis=0.3)
    assert any(h.label == "Right" and h.coarse for h in p.hands)


def test_persons_heuristic_counts_extra_hands():
    dv = FeatureDeriver()
    hands = [hand((100, 100), label="Left"), hand((200, 100), label="Right"), hand((300, 100), label="Left")]
    f = dv.update(obs(hands), 0.0)
    assert f.n_persons == 2
    assert dv.update(obs(hands[:2], n_faces=2), 0.1).n_persons == 2
    assert dv.update(obs(hands[:1]), 0.2).n_persons == 1


def test_hand_covering_face_keeps_last_anchor_then_expires():
    dv = FeatureDeriver(median_frames=1, tube_memory_s=2.0, memory_conf=0.6)
    dv.update(obs([hand((NOSE[0] + 1.0 * FACE_W, NOSE[1]))]), 0.0)
    # face lost, hand right at the remembered nose: still measurable, reduced confidence
    f = dv.update(obs([hand((NOSE[0] + 0.1 * FACE_W, NOSE[1]), pinch_px=10)], face=False), 1.0)
    assert f.conf == pytest.approx(0.6) and f.d == pytest.approx(0.1) and f.tube_pt == NOSE
    assert f.pinch == pytest.approx(10 / FACE_W)
    # beyond the memory window: cannot see the patient
    f = dv.update(obs([hand((NOSE[0] + 0.1 * FACE_W, NOSE[1]))], face=False), 3.5)
    assert f.conf == 0.0 and f.d is None


def test_face_lost_without_hands_is_not_rescued_by_memory():
    dv = FeatureDeriver(median_frames=1)
    dv.update(obs([hand((NOSE[0] + 1.0 * FACE_W, NOSE[1]))]), 0.0)
    f = dv.update(obs(face=False), 0.5)
    assert f.conf == 0.0 and f.d is None


def test_memory_feeds_dwell_through_the_state_machine():
    """Hand covers the nose for 3 s after a confident approach: WARN, not FAULT."""
    from nurse24hr import EventMachine, State

    dv = FeatureDeriver(median_frames=1)
    m = EventMachine()
    t = 0.0
    for i in range(FPS):  # approach with face visible
        m.step(dv.update(obs([hand((NOSE[0] + 0.5 * FACE_W, NOSE[1]))]), t)); t += 1 / FPS
    for i in range(FPS * 3):  # hand on nose, face occluded, still, pinched
        m.step(dv.update(obs([hand((NOSE[0] + 0.05 * FACE_W, NOSE[1]), pinch_px=8)], face=False), t)); t += 1 / FPS
    assert m.state is State.WARN


def test_dropout_does_not_create_velocity_spike():
    dv = FeatureDeriver(median_frames=1, velocity_alpha=1.0)
    dv.update(obs([hand((NOSE[0] + 0.2 * FACE_W, NOSE[1]))]), 0.0)
    dv.update(obs(face=False), 1.0)
    f = dv.update(obs([hand((NOSE[0] + 0.2 * FACE_W, NOSE[1]))]), 1.1)
    assert f.v_radial == 0.0


def test_detection_window_zooms_around_tracked_face_and_falls_back():
    from nurse24hr.mp_perceiver import detection_window

    base = (0, 0, 1280, 720)
    assert detection_window(None, 0, base) == base
    win = detection_window((640.0, 360.0, 100.0), 0, base, scale=3.0, min_px=256)
    assert win == (490, 210, 790, 510)  # 3 face-widths, centred
    win = detection_window((640.0, 360.0, 40.0), 0, base, scale=3.0, min_px=256)
    assert win == (512, 232, 768, 488)  # never smaller than min_px
    assert detection_window((20.0, 20.0, 100.0), 0, base) == (0, 0, 170, 170)  # clipped to the frame
    assert detection_window((640.0, 360.0, 100.0), 15, base, lost_frames=15) == base  # lost too long
    roi = (300, 100, 900, 600)
    assert detection_window((640.0, 360.0, 100.0), 3, roi) == (490, 210, 790, 510)
    assert detection_window(None, 0, roi) == roi


class _LM:
    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.visibility = x, y, visibility


def _pose_lms(over=None):
    """33 pose landmarks in normalised coords; nose at (0.5, 0.3), ears 0.1 apart, hands low."""
    lms = [_LM(0.0, 0.0, 0.0) for _ in range(33)]
    base = {0: (0.50, 0.30), 2: (0.52, 0.28), 5: (0.48, 0.28), 7: (0.55, 0.30), 8: (0.45, 0.30),
            15: (0.70, 0.80), 16: (0.30, 0.80), 19: (0.72, 0.85), 20: (0.28, 0.85), 21: (0.71, 0.84), 22: (0.29, 0.84)}
    for i, (x, y) in base.items():
        lms[i] = _LM(x, y, 0.95)
    for i, v in (over or {}).items():
        lms[i] = v
    return lms


def test_pose_observation_reduces_body_landmarks_to_anchors():
    from nurse24hr.mp_perceiver import pose_observation

    to_px = lambda lm: (lm.x * 1000, lm.y * 1000)
    p = pose_observation(_pose_lms(), 1, to_px)
    assert p is not None
    assert p.nose == (500.0, 300.0)
    assert p.face_w == pytest.approx(100.0)  # ear to ear
    assert len(p.ears) == 2 and all(e is not None for e in p.ears)
    assert len(p.tape_pts) == 2 and p.tape_pts[0] == pytest.approx((475.0, 300.0))
    assert {h.label for h in p.hands} == {"Right", "Left"}
    right = next(h for h in p.hands if h.label == "Right")
    assert right.wrist == (300.0, 800.0) and right.pinch_px == pytest.approx(_dist_px((280, 850), (290, 840)))


def _dist_px(a, b):
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_pose_observation_handles_hidden_parts():
    from nurse24hr.mp_perceiver import pose_observation

    to_px = lambda lm: (lm.x * 1000, lm.y * 1000)
    # nose hidden -> no observation
    assert pose_observation(_pose_lms({0: _LM(0.5, 0.3, 0.1)}), 1, to_px) is None
    # one ear hidden -> face width from the eyes
    p = pose_observation(_pose_lms({7: _LM(0.55, 0.30, 0.1)}), 1, to_px)
    assert p is not None and p.ears[1] is None and p.face_w == pytest.approx(2.5 * 40.0)
    assert len(p.tape_pts) == 1
    # hands under the blanket -> no hands, but the patient is still seen
    p = pose_observation(_pose_lms({15: _LM(0, 0, 0.1), 16: _LM(0, 0, 0.1)}), 2, to_px)
    assert p is not None and p.hands == () and p.n_poses == 2


def test_pose_anchors_feed_the_deriver_at_reduced_confidence():
    obs_pose = RawObservation(tube_pt=NOSE, face_w=FACE_W, hands=(), n_faces=1, tape_pts=(CHEEK,), quality=0.7, source="pose")
    f = FeatureDeriver().update(obs_pose, 0.0)
    assert f.conf == pytest.approx(0.7) and f.d == FAR and f.d_tape == FAR


MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.mark.skipif(not (MODELS / "hand_landmarker.task").exists(), reason="model files not downloaded")
def test_mediapipe_perceiver_runs_on_blank_frame():
    from nurse24hr.mp_perceiver import MediaPipeConfig, MediaPipePerceiver

    p = MediaPipePerceiver(MediaPipeConfig(hand_model=MODELS / "hand_landmarker.task", face_model=MODELS / "face_landmarker.task"))
    try:
        img = np.zeros((240, 320, 3), np.uint8)
        f = p.process(img, 0.0)
        assert f.conf == 0.0 and f.d is None and f.n_persons == 0
        f2 = p.process(img, 0.0)  # equal timestamp must not crash (monotonic guard)
        assert f2.conf == 0.0
        p.draw(img, f2)  # no landmarks, must be a no-op
        assert p.last_window == (0, 0, 320, 240)
        p.cfg.roi = (10, 20, 200, 150)
        p.process(img, 1.0)
        assert p.last_window == (10, 20, 200, 150)
    finally:
        p.close()


# ------------------------------------------------------------ face gate

def test_face_gate_rejects_a_much_larger_face():
    g = FaceGate(scale_max=1.6)
    assert g.check("mesh", 100.0, 0.0), "nothing learned yet: accept"
    for i in range(20):
        g.accept("mesh", 100.0 + (i % 3), float(i))
    assert g.check("mesh", 115.0, 20.0)        # patient turned a little
    assert not g.check("mesh", 300.0, 20.0)    # visitor leaning in
    assert not g.check("mesh", 40.0, 20.0)     # someone far behind the bed


def test_pose_inside_a_rejected_face_is_that_person():
    rejected = [(10.0, (400.0, 300.0), 250.0)]
    assert pose_near_rejected((450.0, 350.0), rejected, 10.5, memory_s=1.0)
    assert not pose_near_rejected((900.0, 300.0), rejected, 10.5, memory_s=1.0), "a different head"
    assert not pose_near_rejected((450.0, 350.0), rejected, 12.0, memory_s=1.0), "memory expired"


def test_face_gate_relearns_after_a_long_gap():
    g = FaceGate(scale_max=1.6, relearn_s=30.0)
    g.accept("mesh", 100.0, 0.0)
    assert not g.check("mesh", 300.0, 10.0)
    assert g.check("mesh", 300.0, 31.0), "camera moved: nothing accepted for 30 s, start over"


def test_face_gate_can_be_disabled():
    g = FaceGate(scale_max=0)
    g.accept("mesh", 100.0, 0.0)
    assert g.check("mesh", 1000.0, 1.0)


def test_rejected_face_blocks_anchor_memory():
    dv = FeatureDeriver(median_frames=1)
    dv.update(obs([hand((NOSE[0] + 0.1 * FACE_W, NOSE[1]))]), 0.0)
    # patient's face gone, a visitor's (rejected) face and hand in view: not the patient's hand
    blocked = RawObservation(tube_pt=None, face_w=None, hands=(hand((NOSE[0] + 0.1 * FACE_W, NOSE[1])),),
                             n_faces=1, source="none", n_rejected=1)
    f = dv.update(blocked, 1.0)
    assert f.d is None and f.conf == 0.0
    # the same frame without a rejected face is the hand-over-face case and keeps the anchor
    f = dv.update(RawObservation(tube_pt=None, face_w=None, hands=(hand((NOSE[0] + 0.1 * FACE_W, NOSE[1])),),
                                 n_faces=0, source="none"), 2.0)
    assert f.d == pytest.approx(0.1)
