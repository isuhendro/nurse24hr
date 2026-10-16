"""FeatureDeriver tests: observation -> Features arithmetic, no MediaPipe, no camera."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from nurse24hr.mp_perceiver import (FAR, Candidate, FaceGate, corroborated, FeatureDeriver, HandObs, RawObservation, lock_admits,
                                    merge_candidates, plausible_hand, plausible_head, pose_near_rejected, remember_candidates, roll_rot, rotate_image,
                                    sane_width, scan_tiles, unrotate_point)

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


# ------------------------------------------------------------ patient lock

def test_lock_admits_only_heads_near_the_patient_when_locked():
    track = (400.0, 300.0, 100.0)
    assert lock_admits((900.0, 300.0), track, locked=False, radius=2.0), "unlocked: anyone"
    assert lock_admits((450.0, 320.0), track, locked=True, radius=2.0), "patient shifted a little"
    assert lock_admits((590.0, 300.0), track, locked=True, radius=2.0), "just inside two face widths"
    assert not lock_admits((900.0, 300.0), track, locked=True, radius=2.0), "a different head across the room"
    assert lock_admits((900.0, 300.0), track, locked=True, radius=0), "radius disabled"
    assert lock_admits((900.0, 300.0), None, locked=True, radius=2.0), "nothing known yet"


def test_merge_candidates_orders_left_to_right_and_dedups_the_pose_head():
    a = Candidate((600.0, 200.0), 100.0, "mesh")
    b = Candidate((200.0, 220.0), 90.0, "mesh")
    assert merge_candidates([a, b], None) == [b, a]
    on_a = Candidate((620.0, 210.0), 110.0, "pose")
    assert merge_candidates([a, b], on_a) == [b, a], "body model saw the same head as mesh face a"
    third = Candidate((400.0, 200.0), 80.0, "pose")
    assert merge_candidates([a, b], third) == [b, third, a], "a head only the body model found is offered too"
    assert merge_candidates([], third) == [third]


@pytest.mark.skipif(not (MODELS / "hand_landmarker.task").exists(), reason="model files not downloaded")
def test_select_locks_and_sticks_until_cleared():
    from nurse24hr.mp_perceiver import MediaPipeConfig, MediaPipePerceiver

    p = MediaPipePerceiver(MediaPipeConfig(hand_model=MODELS / "hand_landmarker.task", face_model=MODELS / "face_landmarker.task",
                                           pose_model=None))
    try:
        assert p.select(1) is None and not p.locked, "nothing in view: nothing to select"
        left, right = Candidate((100.0, 120.0), 60.0, "mesh"), Candidate((500.0, 130.0), 70.0, "mesh")
        p.candidates = [left, right]
        p.survey = True
        assert p.select(2) == right
        assert p.locked and not p.survey, "choosing ends survey mode"
        assert p._track == (500.0, 130.0, 70.0)
        assert p.face_gate.baseline("mesh") == 70.0, "size baseline restarts on the chosen head"
        assert p.selection == {"people": 2, "patient": None, "locked": True, "survey": False}
        assert "locked" in p.selection_label

        # Frames with no face: the lock does not move or expire.
        blank = np.zeros((240, 320, 3), np.uint8)
        for i in range(40):
            f = p.process(blank, 0.1 * i)
        assert f.conf == 0.0 and p.locked and p._track[:2] == (500.0, 130.0)
        assert p.candidates == [] and p.patient_index is None

        # Click on the other head: the lock follows the click; click on nothing: unchanged.
        p.candidates = [left, right]
        assert p.select_at(105, 118) == left and p._track[:2] == (100.0, 120.0)
        assert p.select_at(300, 10) is None and p._track[:2] == (100.0, 120.0)
        assert p.select(5) is None, "no fifth head"

        p.select(0)
        assert not p.locked and p.selection["locked"] is False
    finally:
        p.close()


def test_remember_candidates_keeps_a_head_through_dropped_frames_without_duplicating_it():
    a = Candidate((600.0, 200.0), 100.0, "mesh")
    hist = remember_candidates([], [a], 0.0, memory_s=1.0)
    hist = remember_candidates(hist, [], 0.5, memory_s=1.0)
    assert [h.cand for h in hist] == [a], "unseen for half a second: still listed"
    moved = Candidate((680.0, 205.0), 30.0, "pose")  # same head moved a little, body model with a collapsed width
    hist = remember_candidates(hist, [moved], 0.6, memory_s=1.0)
    assert [(h.cand, h.hits) for h in hist] == [(moved, 2)], "replaced, not added beside the old one; hits carry over"
    other = Candidate((200.0, 200.0), 90.0, "mesh")
    hist = remember_candidates(hist, [other], 0.7, memory_s=1.0)
    assert len(hist) == 2
    hist = remember_candidates(hist, [other], 1.65, memory_s=1.0)
    assert [(h.cand, h.hits) for h in hist] == [(other, 2)], "the first head expired"


def test_phantom_heads_that_jump_about_never_build_hits():
    hist = []
    for i in range(10):  # body model hallucinating along the frame edge: a new spot every frame
        hist = remember_candidates(hist, [Candidate((40.0 + 150.0 * (i % 4), 200.0 + 120.0 * (i % 3)), 30.0, "pose")], i / 15, memory_s=1.0)
    assert max(h.hits for h in hist) < 3
    steady = []
    for i in range(3):
        steady = remember_candidates(steady, [Candidate((600.0 + i, 300.0), 90.0, "mesh")], i / 15, memory_s=1.0)
    assert [h.hits for h in steady] == [3]


def test_plausible_head_rejects_offscreen_and_tiny_heads():
    roi = (0, 0, 1280, 720)
    assert plausible_head(Candidate((600.0, 300.0), 90.0, "pose"), roi, 24)
    assert not plausible_head(Candidate((-45.0, 210.0), 55.0, "pose"), roi, 24), "off the left edge"
    assert not plausible_head(Candidate((989.0, 203.0), 3.0, "pose"), roi, 24), "three pixels wide"
    assert not plausible_head(Candidate((600.0, 300.0), 90.0, "pose"), (700, 0, 1280, 720), 24), "outside the bed ROI"


# ------------------------------------------------------------ rotation

@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_unrotate_point_inverts_rotate_image(rot):
    img = np.zeros((40, 60), np.uint8)  # h=40, w=60
    img[7, 23] = 255
    r = rotate_image(img, rot)
    ys, xs = np.nonzero(r)
    x, y = unrotate_point(xs[0] + 0.5, ys[0] + 0.5, rot, 60, 40)  # pixel centres
    assert (round(x - 0.5), round(y - 0.5)) == (23, 7)


def _first_frame(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        for _ in range(30):
            ok, img = cap.read()
            if not ok:
                return None
        return img
    finally:
        cap.release()


@pytest.mark.skipif(not (MODELS / "hand_landmarker.task").exists(), reason="model files not downloaded")
def test_scan_finds_a_sideways_face_and_selecting_it_tracks_at_that_rotation():
    """A frame with an upright face, turned on its side: the per-frame models (upright only) lose
    it, the scan finds it at the matching rotation, and after selection the mesh tracks it."""
    import cv2
    from nurse24hr.mp_perceiver import MediaPipeConfig, MediaPipePerceiver

    clips = sorted((MODELS.parent / "clips").glob("*.mp4"))
    if not clips:
        pytest.skip("no recorded clip to take a face from")
    cfg = MediaPipeConfig(hand_model=MODELS / "hand_landmarker.task", face_model=MODELS / "face_landmarker.task", pose_model=None)
    upright = None
    p = MediaPipePerceiver(cfg)
    try:
        for clip in clips:
            img = _first_frame(clip)
            if img is None:
                continue
            f = p.process(img, 0.0)
            if f.tube_pt is not None and p.last_obs.source == "mesh":
                upright = (img, f.tube_pt, f.face_w)
                break
    finally:
        p.close()
    if upright is None:
        pytest.skip("no clip frame with a mesh face")
    img, nose, face_w = upright
    h, w = img.shape[:2]
    turned = rotate_image(img, 270)  # face now lies on its side; turning the crop 90 cw rights it
    exp_x, exp_y = (nose[1], w - nose[0])  # where the nose lands after a 270 turn

    p = MediaPipePerceiver(MediaPipeConfig(**{**cfg.__dict__, "survey_every_s": 0.5, "survey_rotations": (0, 90, 270)}))
    try:
        cands = p.scan(turned)
        found = [c for c in cands if math.dist(c.center, (exp_x, exp_y)) < face_w]
        assert found, f"scan did not list the sideways face near ({exp_x:.0f}, {exp_y:.0f}): {cands}"
        assert found[0].rot == 90

        # Per-frame loop on the turned frame: the upright models fail, the background scan then
        # points the tracker at the head with its rotation, and the mesh locks on.
        seen_mesh = False
        for i in range(8):
            f = p.process(turned, 0.1 * i)
            seen_mesh = seen_mesh or (p.last_obs.source == "mesh" and f.tube_pt is not None
                                      and math.dist(f.tube_pt, (exp_x, exp_y)) < 0.5 * face_w)
        assert seen_mesh, "tracking never switched to the rotation the scan found"
        assert p._rot == 90 and p.patient_index is not None
        assert any("lying" in s for s in [p.selection_label]) or p.candidates[p.patient_index].rot == 90
    finally:
        p.close()


def test_scan_tiles_cover_the_roi_with_half_overlap():
    base = (0, 0, 1280, 720)
    assert scan_tiles(base, 0) == [base]
    assert scan_tiles(base, 2000) == [base], "tile larger than the ROI: one window"
    tiles = scan_tiles(base, 640)
    assert all(x1 - x0 == 640 and y1 - y0 == 640 for x0, y0, x1, y1 in tiles)
    assert all(0 <= x0 and x1 <= 1280 and 0 <= y0 and y1 <= 720 for x0, y0, x1, y1 in tiles)
    assert {t[0] for t in tiles} == {0, 320, 640} and {t[1] for t in tiles} == {0, 80}, "steps of half a tile, last one flush with the edge"
    # every pixel is covered
    cover = np.zeros((720, 1280), bool)
    for x0, y0, x1, y1 in tiles:
        cover[y0:y1, x0:x1] = True
    assert cover.all()
    off = scan_tiles((100, 50, 740, 530), 640)
    assert off == [(100, 50, 740, 530)], "ROI exactly one tile"


def test_roll_rot_reads_head_tilt_from_the_face_line():
    assert roll_rot((100.0, 200.0), (200.0, 200.0)) == 0, "upright"
    assert roll_rot((100.0, 200.0), (190.0, 230.0)) == 0, "a little tilted is still upright"
    assert roll_rot((100.0, 300.0), (100.0, 200.0)) == 90, "lying, crown to the left: turn the image clockwise"
    assert roll_rot((100.0, 200.0), (100.0, 300.0)) == 270, "lying, crown to the right"
    assert roll_rot((200.0, 200.0), (100.0, 200.0)) == 180, "upside down"


@pytest.mark.skipif(not (MODELS / "hand_landmarker.task").exists(), reason="model files not downloaded")
def test_lock_state_round_trips_through_a_new_perceiver():
    from nurse24hr.mp_perceiver import MediaPipeConfig, MediaPipePerceiver

    cfg = MediaPipeConfig(hand_model=MODELS / "hand_landmarker.task", face_model=MODELS / "face_landmarker.task", pose_model=None)
    a = MediaPipePerceiver(cfg)
    b = MediaPipePerceiver(cfg)
    try:
        assert a.lock_state == {"locked": False}
        assert not b.restore_lock(a.lock_state) and not b.locked
        a.candidates = [Candidate((900.0, 400.0), 150.0, "mesh", 270)]
        a.select(1)
        state = a.lock_state
        assert state == {"locked": True, "cx": 900.0, "cy": 400.0, "face_w": 150.0, "rot": 270}
        assert b.restore_lock(json.loads(json.dumps(state)))
        assert b.locked and b._track == (900.0, 400.0, 150.0) and b._rot == 270
        assert b.face_gate.baseline("mesh") == 150.0
        assert not b._admits((200.0, 400.0)), "the other person's seat is outside the restored lock"
        assert b._admits((950.0, 420.0))
    finally:
        a.close()
        b.close()


def test_plausible_hand_drops_phantoms_but_not_small_real_hands():
    assert plausible_hand(20.0, 83.0, 0.45) is False, "a quarter of a face wide: the taped tube, not a hand"
    assert plausible_hand(45.0, 83.0, 0.45) is True, "foreshortened real hand"
    assert plausible_hand(0.0, 83.0, 0.45) is True, "body-model hand: length unknown"
    assert plausible_hand(20.0, None, 0.45) is True, "no face to compare with"
    assert plausible_hand(20.0, 83.0, 0) is True, "disabled"


def test_corroborated_needs_a_recent_body_model_hand_near_the_hand():
    h = hand((450.0, 320.0))  # grip at (450, 320), wrist 80 px below
    seen = [(10.0, (430.0, 410.0), (300.0, 300.0))]  # body model: wrist near this hand's wrist, grip elsewhere
    assert corroborated(h, 100.0, seen, 10.5, memory_s=1.5), "wrists agree, half a second ago"
    assert corroborated(h, 100.0, [(10.0, (900.0, 900.0), (470.0, 330.0))], 10.5, memory_s=1.5), "grips agree"
    assert not corroborated(h, 100.0, [(10.0, (700.0, 600.0), (700.0, 500.0))], 10.5, memory_s=1.5), "body model's hand is elsewhere"
    assert not corroborated(h, 100.0, seen, 12.0, memory_s=1.5), "sighting expired"
    assert not corroborated(h, 100.0, [], 10.5, memory_s=1.5), "body model saw no hands at all"
    assert corroborated(h, 100.0, [], 10.5, memory_s=0), "disabled"


def test_sane_width_never_lets_the_lock_radius_collapse():
    assert sane_width(4.0, 73.0, 24) == 73.0, "mesh baseline known: use it"
    assert sane_width(4.0, None, 24) == 24.0, "no baseline, collapsed pose width: floor"
    assert sane_width(80.0, None, 24) == 80.0
    assert sane_width(None, None, 24) == 24.0


@pytest.mark.skipif(not (MODELS / "hand_landmarker.task").exists(), reason="model files not downloaded")
def test_locked_patient_survives_a_collapsed_pose_width():
    from nurse24hr.mp_perceiver import MediaPipeConfig, MediaPipePerceiver

    p = MediaPipePerceiver(MediaPipeConfig(hand_model=MODELS / "hand_landmarker.task", face_model=MODELS / "face_landmarker.task", pose_model=None))
    try:
        p.candidates = [Candidate((1143.0, 383.0), 73.0, "mesh")]
        p.select(1)
        p._track = (1135.0, 389.0, 4.0)  # what a profile-view body model wrote into the anchor on 2026-10-16
        assert p._admits((1153.0, 398.0)), "the patient's own face 20 px away must still be admitted"
        assert not p._admits((550.0, 280.0)), "the visitor across the room is still not"
        assert p.lock_state["face_w"] == 73.0, "the saved lock carries the real width, not the collapsed one"
        p.face_gate.reset()
        assert p._admits((1153.0, 398.0)), "even with no mesh baseline the floor keeps a usable radius"
        pose_cand = Candidate((900.0, 400.0), 3.0, "pose")
        p._lock_on(pose_cand)
        assert p._track[2] >= p.cfg.candidate_min_px
    finally:
        p.close()


@pytest.mark.skipif(not (MODELS / "hand_landmarker.task").exists(), reason="model files not downloaded")
def test_lost_patient_is_readmitted_within_relock_radius_only_if_lying_the_same_way():
    from nurse24hr.mp_perceiver import MediaPipeConfig, MediaPipePerceiver

    p = MediaPipePerceiver(MediaPipeConfig(hand_model=MODELS / "hand_landmarker.task", face_model=MODELS / "face_landmarker.task", pose_model=None))
    try:
        p.candidates = [Candidate((983.0, 506.0), 67.0, "mesh", 90)]
        p.select(1)
        moved = Candidate((1063.0, 392.0), 95.0, "mesh", 90)   # camera nudged: 139 px away, 2 x 67 = 134 px radius
        upright = Candidate((1063.0, 392.0), 95.0, "mesh", 0)  # same spot, but an upright head
        assert not p._admit(moved), "just outside the lock radius while still tracked: not admitted"
        p._lost = p.cfg.track_lost_frames
        assert p._admit(moved), "lost for a while, same way of lying, within the relock radius"
        assert not p._admit(upright), "an upright head there is someone else"
        far = Candidate((583.0, 275.0), 111.0, "mesh", 90)
        assert not p._admit(far), "seven face widths away: beyond the relock radius"
        p.clear_selection()
        assert p._admit(far), "unlocked: anyone"
    finally:
        p.close()
