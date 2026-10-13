"""MediaPipe-based perception: hands + face landmarks -> :class:`Features`.

Split in two so the arithmetic is testable without a camera or a model:

* :class:`MediaPipePerceiver` runs the MediaPipe Tasks hand and face
  landmarkers and reduces each frame to a :class:`RawObservation` in pixels.
* :class:`FeatureDeriver` turns a stream of observations into ``Features``:
  normalisation by face width, median smoothing of ``d``, radial velocity,
  pinch, the person count heuristic, and ``osc``: the grip point's own
  positional scatter over the trailing second, in face widths. It is measured
  on the hand, not on a distance to one anchor, so a scratch at the cheek
  tape looks just as "busy" as a scratch at the nose.

v1 insertion-point proxy
------------------------
Without a tube detector the tube anchor is the subnasale (face landmark 2,
between the nostrils), or a specific nostril (98 = subject's right, 327 =
subject's left) when ``tube_side`` is configured. Face width is the distance
between the face-oval extremes 234 and 454.

The "hand point" used for ``d`` is the grip point, the midpoint of thumb tip
and index tip, rather than the wrist named in the spec. The grip point is what
touches the tube and it makes ``pinch`` and ``d`` refer to the same location.
Wrist positions are still recorded for logging.

Confidence
----------
* No face and no hands: ``conf = 0``. The patient cannot be seen; the state
  machine holds and eventually raises FAULT.
* No face but a hand is visible, within ``tube_memory_s`` of the last good
  face: the hand is probably covering the nose, which is exactly the grasp we
  care about. ``d`` is measured against the remembered tube point at
  ``memory_conf`` (above ``conf_min``), so dwell keeps accumulating instead of
  turning a grasp into a FAULT. The anchor goes stale if the head moves while
  covered, which is why the window is short.
* Face but no hands: hands are away or under the blanket, which is the normal
  sleeping case. ``d`` is set far (``FAR``) with face confidence.
* Face and hands: confidence blends in the closer hand's handedness score.

Pose first, face mesh second
----------------------------
The face mesh needs a large, fairly frontal face. A real patient in a bed,
seen from the side under ward lighting with tube and tape on the face, often
defeats it (observed 2026-10-11). The pose landmarker is full-range and finds
the body reliably, and it carries nose, ears, wrists, index fingers and
thumbs. So:

* pose runs on the bed ROI and answers "is the patient here" and "where is
  the head";
* the head position seeds the zoom crop for the face mesh and hands;
* if the face mesh fails, pose supplies coarse anchors (nose as tube point,
  nose-ear midpoints as tape points, ear-to-ear distance as face width) at
  ``pose_conf``; if the hand model fails, pose index/thumb/wrist stand in;
* FAULT therefore only happens when nobody is visible at all.

Range and the tracking crop
---------------------------
MediaPipe's face detector is short-range: a face that is a small patch of a
wide frame is not found. Two remedies, both in full-frame coordinates:

* a static bed ROI (``roi``), the spec's answer for a fixed camera;
* a tracking crop: once a face has been seen, detection runs on a window of
  ``track_scale`` face-widths around it (a digital zoom). If nothing is found
  there, the same frame is retried on the full ROI. After
  ``track_lost_frames`` misses the crop is dropped. Hands that enter the crop
  are seen; a second person's hands outside it are not, which the person
  count tolerates because the caregiver's face is usually in the wide retry.

Persons
-------
``n_persons = max(faces, ceil(hands / 2))``. A caregiver's hand reaching into
the bed ROI while their face is out of frame still counts as a second person.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nurse24hr.features import Features, Point

FAR = 3.0
"""``d`` reported when the face is visible but no hand is."""

# Face mesh landmark indices.
LM_SUBNASALE = 2
LM_NOSTRIL_RIGHT = 98  # subject's right
LM_NOSTRIL_LEFT = 327  # subject's left
LM_FACE_RIGHT = 234
LM_FACE_LEFT = 454
LM_CHEEK_RIGHT = 205  # subject's right cheek, where NG tube tape usually sits
LM_CHEEK_LEFT = 425
# Hand landmark indices.
LM_WRIST = 0
LM_THUMB_TIP = 4
LM_INDEX_TIP = 8


@dataclass(frozen=True, slots=True)
class HandObs:
    grip: Point
    wrist: Point
    pinch_px: float
    score: float
    label: str  # "Left" / "Right" as reported by the model
    coarse: bool = False
    """True when the hand comes from the body model: position is usable, pinch and oscillation are not."""


@dataclass(frozen=True, slots=True)
class RawObservation:
    """One frame reduced to pixels. ``face_w`` and ``tube_pt`` are None without a face."""

    tube_pt: Point | None
    face_w: float | None
    hands: tuple[HandObs, ...]
    n_faces: int
    tape_pts: tuple[Point, ...] = ()
    """Tape anchors on the cheek(s). Empty when the face is not seen."""
    quality: float | None = None
    """Base confidence override for this frame (e.g. coarse pose anchors). None = face mesh default."""
    source: str = "mesh"
    """Which model produced the anchors: ``mesh``, ``pose`` or ``none``."""
    n_rejected: int = 0
    """Faces seen but rejected as not the patient (wrong size for the tracked face: a visitor leaning in)."""


# Pose landmark indices (subject's side).
P_NOSE, P_EYE_L, P_EYE_R, P_EAR_L, P_EAR_R = 0, 2, 5, 7, 8
P_WRIST_L, P_WRIST_R, P_INDEX_L, P_INDEX_R, P_THUMB_L, P_THUMB_R = 15, 16, 19, 20, 21, 22


@dataclass(frozen=True, slots=True)
class PoseObs:
    nose: Point
    face_w: float
    """Ear-to-ear distance, or 2.5× inter-ocular when an ear is hidden."""
    ears: tuple[Point | None, Point | None]
    """(subject's right, subject's left), None when not visible."""
    hands: tuple[HandObs, ...]
    n_poses: int

    @property
    def head_center(self) -> Point:
        return self.nose

    @property
    def tape_pts(self) -> tuple[Point, ...]:
        return tuple(((self.nose[0] + e[0]) / 2, (self.nose[1] + e[1]) / 2) for e in self.ears if e is not None)


def pose_observation(lms, n_poses: int, to_px, *, min_vis: float = 0.5, hand_min_vis: float = 0.5) -> PoseObs | None:
    """Reduce one pose's landmarks (objects with x, y, visibility) to a :class:`PoseObs`.

    Returns None when the nose is not visible. Pure, for tests.
    """
    def vis(i: int) -> bool:
        return getattr(lms[i], "visibility", 1.0) >= min_vis

    if not vis(P_NOSE):
        return None
    nose = to_px(lms[P_NOSE])
    ear_r = to_px(lms[P_EAR_R]) if vis(P_EAR_R) else None
    ear_l = to_px(lms[P_EAR_L]) if vis(P_EAR_L) else None
    if ear_r is not None and ear_l is not None:
        face_w = _dist(ear_r, ear_l)
    elif vis(P_EYE_L) and vis(P_EYE_R):
        face_w = 2.5 * _dist(to_px(lms[P_EYE_L]), to_px(lms[P_EYE_R]))
    elif ear_r is not None or ear_l is not None:
        face_w = 2.0 * _dist(nose, ear_r or ear_l)  # type: ignore[arg-type]
    else:
        return None
    if face_w <= 1.0:
        return None
    def hvis(i: int) -> bool:
        return getattr(lms[i], "visibility", 1.0) >= hand_min_vis

    hands = []
    for label, wi, ii, ti in (("Right", P_WRIST_R, P_INDEX_R, P_THUMB_R), ("Left", P_WRIST_L, P_INDEX_L, P_THUMB_L)):
        if not (hvis(wi) and hvis(ii)):
            continue
        index = to_px(lms[ii])
        thumb = to_px(lms[ti]) if hvis(ti) else index
        grip = ((index[0] + thumb[0]) / 2, (index[1] + thumb[1]) / 2)
        score = min(getattr(lms[wi], "visibility", 1.0), getattr(lms[ii], "visibility", 1.0))
        hands.append(HandObs(grip=grip, wrist=to_px(lms[wi]), pinch_px=_dist(index, thumb), score=float(score), label=label, coarse=True))
    return PoseObs(nose=nose, face_w=float(face_w), ears=(ear_r, ear_l), hands=tuple(hands), n_poses=n_poses)


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class FaceGate:
    """Is this face the patient's? Size is the cue: a visitor leaning over the bed is two to four
    times the patient's width in the frame (live false ALARM 2026-10-12, visitor filling the frame
    tracked as the patient). A face whose width differs from the recent baseline for its source
    by more than ``scale_max`` is rejected. Only the face-mesh width is stable enough to gate on;
    the body model's ear-to-ear width collapses in a profile view (2 to 77 px on one person), so a
    pose is instead rejected by position: its nose inside a recently rejected mesh face (see
    :func:`pose_near_rejected`). With no accepted face for ``relearn_s`` the baseline is dropped
    so a moved camera recovers. Pure, for tests."""

    def __init__(self, scale_max: float = 1.6, relearn_s: float = 30.0, window: int = 150) -> None:
        self.scale_max = scale_max
        self.relearn_s = relearn_s
        self._hist: dict[str, deque[float]] = {"mesh": deque(maxlen=window), "pose": deque(maxlen=window)}
        self._last_t: float | None = None

    def baseline(self, source: str) -> float | None:
        h = self._hist.get(source)
        return statistics.median(h) if h else None

    def check(self, source: str, face_w: float, t: float) -> bool:
        if self._last_t is not None and t - self._last_t > self.relearn_s:
            for h in self._hist.values():
                h.clear()
        if self.scale_max <= 0 or face_w <= 0:
            return True
        b = self.baseline(source)
        if b is None:
            return True
        return max(face_w / b, b / face_w) <= self.scale_max

    def accept(self, source: str, face_w: float, t: float) -> None:
        self._hist.setdefault(source, deque(maxlen=150)).append(face_w)
        self._last_t = t

    def reset(self) -> None:
        for h in self._hist.values():
            h.clear()
        self._last_t = None


def pose_near_rejected(nose: Point, rejected: list[tuple[float, Point, float]], t: float, memory_s: float) -> bool:
    """A body-model nose inside a mesh face rejected within ``memory_s`` belongs to that person."""
    return any(t - t0 <= memory_s and _dist(nose, c) <= fw for t0, c, fw in rejected)


class FeatureDeriver:
    """Stateful observation -> Features reducer. Deterministic for a given sequence."""

    def __init__(self, *, median_frames: int = 5, osc_window_s: float = 1.0, face_conf: float = 0.9,
                 velocity_alpha: float = 0.5, tube_memory_s: float = 10.0, memory_conf: float = 0.6,
                 hand_memory_s: float = 2.0) -> None:
        self.median_frames = median_frames
        self.osc_window_s = osc_window_s
        self.face_conf = face_conf
        self.velocity_alpha = velocity_alpha
        self.tube_memory_s = tube_memory_s
        self.memory_conf = memory_conf
        self.hand_memory_s = hand_memory_s
        self._last_hand: tuple[float, float, float | None] | None = None  # (t, raw_d, raw_tape)
        """"No hand seen" is not "hand away": a hand that vanishes near the face is usually occluded
        or missed by the model. Its last distance is held for ``hand_memory_s`` at reduced confidence."""
        self._last_face: tuple[float, Point, float, tuple[Point, ...]] | None = None  # (t, tube_pt, face_w, tape_pts)
        self._raw_d: deque[float] = deque(maxlen=median_frames)
        self._raw_tape: deque[float] = deque(maxlen=median_frames)
        self._d_hist: deque[tuple[float, float]] = deque()
        self._grip_hist: deque[tuple[float, float, float]] = deque()  # (t, gx/face_w, gy/face_w)
        self._prev: tuple[float, float] | None = None  # (t, smoothed d)
        self._v = 0.0

    def reset(self) -> None:
        self._raw_d.clear()
        self._raw_tape.clear()
        self._d_hist.clear()
        self._grip_hist.clear()
        self._prev = None
        self._v = 0.0
        self._last_face = None
        self._last_hand = None

    def update(self, obs: RawObservation, t: float) -> Features:
        n_persons = max(obs.n_faces, math.ceil(len(obs.hands) / 2))
        wrists = {h.label: h.wrist for h in obs.hands}

        tube_pt, face_w, base_conf = obs.tube_pt, obs.face_w, (obs.quality if obs.quality is not None else self.face_conf)
        tape_pts = obs.tape_pts
        remembered = False
        if tube_pt is not None and face_w:
            self._last_face = (t, tube_pt, face_w, tape_pts)
        elif obs.hands and not obs.n_rejected and self._last_face is not None and 0 <= t - self._last_face[0] <= self.tube_memory_s:
            # Face occluded, most likely by the hand itself. Keep measuring against the last anchor.
            # Not when someone else's face is in view: those are their hands, not the patient's.
            _, tube_pt, face_w, tape_pts = self._last_face
            base_conf = self.memory_conf
            remembered = True
        else:
            # Cannot see the patient. Do not touch smoothing history so a brief
            # dropout does not produce a velocity spike on recovery.
            self._prev = None
            return Features(t=t, d=None, conf=0.0, n_persons=n_persons,
                            wrist_l=wrists.get("Left"), wrist_r=wrists.get("Right"))

        assert tube_pt is not None and face_w
        tape_pt: Point | None = None
        coarse = obs.source == "pose"
        closest = None
        if obs.hands:
            closest = min(obs.hands, key=lambda h: _dist(h.grip, tube_pt))
            coarse = coarse or closest.coarse
            raw_d = _dist(closest.grip, tube_pt) / face_w
            pinch: float | None = None if coarse else closest.pinch_px / face_w
            conf = min(1.0, 0.5 * base_conf + 0.5 * closest.score) if not remembered else base_conf
            if tape_pts:
                tape_pt = min(tape_pts, key=lambda p: _dist(closest.grip, p))
                raw_tape: float | None = _dist(closest.grip, tape_pt) / face_w
            else:
                raw_tape = None
            self._last_hand = (t, raw_d, raw_tape)
        elif self._last_hand is not None and 0 <= t - self._last_hand[0] <= self.hand_memory_s:
            # Hand not seen this frame: hold its last distance rather than declaring it away.
            _, raw_d, raw_tape = self._last_hand
            pinch = None
            conf = min(base_conf, self.memory_conf)
            coarse = True
        else:
            raw_d = FAR
            raw_tape = FAR if tape_pts else None
            pinch = None
            conf = base_conf

        self._raw_d.append(raw_d)
        d = statistics.median(self._raw_d)
        if raw_tape is not None:
            self._raw_tape.append(raw_tape)
            d_tape: float | None = float(statistics.median(self._raw_tape))
        else:
            d_tape = None

        if self._prev is not None and t > self._prev[0]:
            inst = (d - self._prev[1]) / (t - self._prev[0])
            self._v = self.velocity_alpha * inst + (1 - self.velocity_alpha) * self._v
        else:
            self._v = 0.0
        self._prev = (t, d)

        self._d_hist.append((t, d))
        while self._d_hist and t - self._d_hist[0][0] > self.osc_window_s:
            self._d_hist.popleft()

        osc: float | None
        if closest is not None and not coarse:
            self._grip_hist.append((t, closest.grip[0] / face_w, closest.grip[1] / face_w))
            while self._grip_hist and t - self._grip_hist[0][0] > self.osc_window_s:
                self._grip_hist.popleft()
            if len(self._grip_hist) >= 2:
                pts = np.array([(x, y) for _, x, y in self._grip_hist])
                osc = float(np.sqrt(pts.var(axis=0).sum()))
            else:
                osc = 0.0
        elif closest is not None or (self._last_hand is not None and t - self._last_hand[0] <= self.hand_memory_s):
            # Coarse or remembered hand: oscillation and pinch are unknown, not zero.
            # The state machine's missing-evidence rule then lets dwell alone decide.
            self._grip_hist.clear()
            osc = None
        else:
            self._grip_hist.clear()
            osc = 0.0

        return Features(
            t=t, d=float(d), conf=float(conf), n_persons=n_persons,
            v_radial=float(self._v), pinch=pinch, osc=osc, tube_disp=None,
            wrist_l=wrists.get("Left"), wrist_r=wrists.get("Right"),
            tube_pt=tube_pt, face_w=face_w, d_tape=d_tape, tape_pt=tape_pt,
        )


Box = tuple[int, int, int, int]


def detection_window(track: tuple[float, float, float] | None, lost: int, base: Box, *,
                     scale: float = 3.0, min_px: int = 256, lost_frames: int = 15) -> Box:
    """Window to run detection on: a crop around the tracked face, else ``base``.

    ``track`` is (cx, cy, face_w) in full-frame pixels. Pure, for tests.
    """
    if track is None or lost >= lost_frames:
        return base
    cx, cy, fw = track
    half = max(min_px, scale * fw) / 2
    x0, y0 = max(base[0], int(cx - half)), max(base[1], int(cy - half))
    x1, y1 = min(base[2], int(cx + half)), min(base[3], int(cy + half))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return base
    return (x0, y0, x1, y1)


@dataclass
class MediaPipeConfig:
    hand_model: Path = Path("models/hand_landmarker.task")
    face_model: Path = Path("models/face_landmarker.task")
    num_hands: int = 4
    num_faces: int = 2
    min_detection_confidence: float = 0.4
    min_tracking_confidence: float = 0.5
    pose_model: Path | None = Path("models/pose_landmarker_lite.task")
    """None disables the pose stage."""
    pose_conf: float = 0.7
    """Base confidence when anchors come from pose instead of the face mesh."""
    pose_every_n: int = 1
    """Run pose on every n-th frame (it is the most expensive stage); the last result is reused between."""
    pose_detection_confidence: float = 0.25
    """Lower than the face threshold: on a small, distant, masked patient 0.4 misses most frames."""
    pose_track_scale: float = 10.0
    """Pose crop is wide (head + torso) so the body detector still sees a body."""
    pose_track_min_px: int = 512
    pose_hand_min_vis: float = 0.5
    """Wrist and index visibility a body-model hand needs. At 0.3 the model placed hidden hands
    (under the blanket) on the patient's face and dwell alarmed on nothing (2026-10-11)."""
    roi: Box | None = None
    """Static bed ROI in full-frame pixels (x0, y0, x1, y1). None = whole frame."""
    auto_zoom: bool = True
    track_scale: float = 3.0
    track_min_px: int = 256
    track_lost_frames: int = 15
    face_scale_max: float = 1.6
    """A face this many times larger or smaller than the tracked patient's face is someone else. 0 disables."""
    face_relearn_s: float = 30.0
    """No accepted face for this long drops the size baseline (camera moved)."""
    reject_memory_s: float = 1.0
    """A rejected face keeps rejecting a body-model nose at its position for this long (the mesh drops frames)."""
    tube_side: str = "center"  # center | left | right (subject's side)
    tape_side: str | None = None  # None = same as tube_side; center = both cheeks
    extra: dict = field(default_factory=dict)


class MediaPipePerceiver:
    """Runs MediaPipe pose + face + hand landmarkers in VIDEO mode and derives Features."""

    def __init__(self, cfg: MediaPipeConfig | None = None) -> None:
        import os

        os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe's startup chatter
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import mediapipe as mp  # heavy import, keep it local
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.cfg = cfg or MediaPipeConfig()
        for p in (self.cfg.hand_model, self.cfg.face_model):
            if not Path(p).exists():
                raise FileNotFoundError(f"model file missing: {p}. See README 'Models'.")
        self._mp = mp
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.cfg.hand_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.cfg.num_hands,
            min_hand_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        face_opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.cfg.face_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.cfg.num_faces,
            min_face_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        self._hands = vision.HandLandmarker.create_from_options(hand_opts)
        self._face = vision.FaceLandmarker.create_from_options(face_opts)
        self._pose = None
        if self.cfg.pose_model and Path(self.cfg.pose_model).exists():
            pose_opts = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(self.cfg.pose_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=self.cfg.num_faces,
                min_pose_detection_confidence=self.cfg.pose_detection_confidence,
                min_tracking_confidence=self.cfg.min_tracking_confidence,
            )
            self._pose = vision.PoseLandmarker.create_from_options(pose_opts)
        self._frame_i = 0
        self._last_pose: PoseObs | None = None
        self.last_pose: PoseObs | None = None
        self.last_pose_window: Box | None = None
        self.deriver = FeatureDeriver()
        self.face_gate = FaceGate(scale_max=self.cfg.face_scale_max, relearn_s=self.cfg.face_relearn_s)
        self._last_ts_ms = -1
        self.last_obs: RawObservation | None = None
        self.last_landmarks: tuple[list, list] = ([], [])
        self._track: tuple[float, float, float] | None = None  # (cx, cy, face_w) full-frame px
        self._lost = 0
        self._rejected: list[tuple[float, Point, float]] = []  # (t, center, face_w) of faces judged not the patient
        self.last_window: Box | None = None
        self.last_window_was_retry = False

    @property
    def _tube_index(self) -> int:
        return {"left": LM_NOSTRIL_LEFT, "right": LM_NOSTRIL_RIGHT}.get(self.cfg.tube_side, LM_SUBNASALE)

    @property
    def _tape_indices(self) -> tuple[int, ...]:
        side = self.cfg.tape_side or self.cfg.tube_side
        return {"left": (LM_CHEEK_LEFT,), "right": (LM_CHEEK_RIGHT,)}.get(side, (LM_CHEEK_RIGHT, LM_CHEEK_LEFT))

    def _next_ts(self, t: float) -> int:
        ts = int(round(t * 1000))
        if ts <= self._last_ts_ms:  # MediaPipe VIDEO mode needs strictly increasing timestamps
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def _detect(self, image_bgr: np.ndarray, win: Box, t: float):
        import cv2

        x0, y0, x1, y1 = win
        crop = image_bgr[y0:y1, x0:x1]
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = self._next_ts(t)
        face_res = self._face.detect_for_video(mp_img, ts)
        hand_res = self._hands.detect_for_video(mp_img, ts)
        return face_res, hand_res

    def _pose_in(self, image_bgr: np.ndarray, win: Box, t: float) -> PoseObs | None:
        import cv2

        x0, y0, x1, y1 = win
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        res = self._pose.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), self._next_ts(t))
        if not res.pose_landmarks:
            return None
        cw, ch = x1 - x0, y1 - y0
        for lms in res.pose_landmarks:
            p = pose_observation(lms, len(res.pose_landmarks), lambda lm: (x0 + lm.x * cw, y0 + lm.y * ch),
                                 hand_min_vis=self.cfg.pose_hand_min_vis)
            if p is not None:
                return p
        return None

    def _detect_pose(self, image_bgr: np.ndarray, base: Box, t: float) -> PoseObs | None:
        """Pose on a wide crop around the last known head (a small distant patient is missed on
        the full frame), retrying on the whole bed ROI when the crop finds nothing."""
        win = detection_window(self._track if self.cfg.auto_zoom else None, self._lost, base,
                               scale=self.cfg.pose_track_scale, min_px=self.cfg.pose_track_min_px,
                               lost_frames=self.cfg.track_lost_frames)
        pose = self._pose_in(image_bgr, win, t)
        self.last_pose_window = win
        if pose is None and win != base:
            pose = self._pose_in(image_bgr, base, t)
            self.last_pose_window = base
        return pose

    def observe(self, image_bgr: np.ndarray, t: float) -> RawObservation:
        h, w = image_bgr.shape[:2]
        base: Box = self.cfg.roi or (0, 0, w, h)
        base = (max(0, base[0]), max(0, base[1]), min(w, base[2]), min(h, base[3]))

        pose: PoseObs | None = None
        if self._pose is not None:
            if self._frame_i % max(1, self.cfg.pose_every_n) == 0:
                self._last_pose = self._detect_pose(image_bgr, base, t)
            pose = self._last_pose
        self._frame_i += 1
        self.last_pose = pose
        self._rejected = [r for r in self._rejected if t - r[0] <= self.cfg.reject_memory_s]
        pose_rejected = pose is not None and pose_near_rejected(pose.nose, self._rejected, t, self.cfg.reject_memory_s)

        track = self._track if self.cfg.auto_zoom else None
        if pose is not None and not pose_rejected and (track is None or self._lost > 0):
            # The body model knows where the head is even when the face mesh does not.
            track = (pose.nose[0], pose.nose[1], pose.face_w)
            lost = 0
        else:
            lost = self._lost
        win = detection_window(track, lost, base, scale=self.cfg.track_scale, min_px=self.cfg.track_min_px,
                               lost_frames=self.cfg.track_lost_frames)
        face_res, hand_res = self._detect(image_bgr, win, t)
        self.last_window_was_retry = False
        if not face_res.face_landmarks and win != base and pose is None:
            # Lost inside the zoom window and no body to go by: retry this frame on the whole bed ROI.
            win = base
            face_res, hand_res = self._detect(image_bgr, win, t)
            self.last_window_was_retry = True
        self.last_window = win
        self.last_landmarks = (face_res.face_landmarks, hand_res.hand_landmarks)

        x0, y0, x1, y1 = win
        cw, ch = x1 - x0, y1 - y0

        def px(lm) -> Point:  # crop-normalised landmark -> full-frame pixels
            return (x0 + lm.x * cw, y0 + lm.y * ch)

        tube_pt = None
        face_w = None
        tape_pts: tuple = ()
        quality: float | None = None
        source = "none"
        n_rejected = 0
        faces = []
        for lm in face_res.face_landmarks:
            right, left = px(lm[LM_FACE_RIGHT]), px(lm[LM_FACE_LEFT])
            fw = _dist(right, left)
            center = ((right[0] + left[0]) / 2, (right[1] + left[1]) / 2)
            if self.face_gate.check("mesh", fw, t):
                faces.append((lm, fw, center))
            else:
                n_rejected += 1
                self._rejected.append((t, center, fw))
        if pose is not None and not pose_rejected:
            pose_rejected = pose_near_rejected(pose.nose, self._rejected, t, self.cfg.reject_memory_s)
        if faces:
            # Several candidates: the one nearest the tracked head.
            if self._track is not None and len(faces) > 1:
                lm, face_w, center = min(faces, key=lambda f: _dist(f[2], self._track[:2]))
            else:
                lm, face_w, center = faces[0]
            tube_pt = px(lm[self._tube_index])
            tape_pts = tuple(px(lm[i]) for i in self._tape_indices)
            self._track = (center[0], center[1], face_w)
            self._lost = 0
            source = "mesh"
            self.face_gate.accept("mesh", face_w, t)
        elif pose is not None and not pose_rejected:
            # Coarse anchors from the body model.
            tube_pt, face_w = pose.nose, pose.face_w
            side = self.cfg.tape_side or self.cfg.tube_side
            ear_r, ear_l = pose.ears
            chosen = {"right": (ear_r,), "left": (ear_l,)}.get(side, (ear_r, ear_l))
            tape_pts = tuple(((pose.nose[0] + e[0]) / 2, (pose.nose[1] + e[1]) / 2) for e in chosen if e is not None)
            self._track = (pose.nose[0], pose.nose[1], pose.face_w)
            self._lost = 0
            quality = self.cfg.pose_conf
            source = "pose"
        else:
            if pose_rejected:
                n_rejected += 1
            self._lost += 1

        hands = []
        for lm, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
            thumb = px(lm[LM_THUMB_TIP])
            index = px(lm[LM_INDEX_TIP])
            hands.append(HandObs(
                grip=((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2),
                wrist=px(lm[LM_WRIST]),
                pinch_px=_dist(thumb, index),
                score=float(handed[0].score) if handed else 0.5,
                label=handed[0].category_name if handed else "Unknown",
            ))
        if not hands and pose is not None:
            hands = list(pose.hands)  # coarse hands from the body model
        n_faces = max(len(face_res.face_landmarks), pose.n_poses if pose is not None else 0)
        obs = RawObservation(tube_pt=tube_pt, face_w=face_w, hands=tuple(hands), n_faces=n_faces,
                             tape_pts=tape_pts, quality=quality, source=source, n_rejected=n_rejected)
        self.last_obs = obs
        return obs

    def process(self, image: np.ndarray, t: float) -> Features:
        return self.deriver.update(self.observe(image, t), t)

    def draw(self, image: np.ndarray, f: Features) -> None:
        import cv2

        obs = self.last_obs
        if obs is None:
            return
        if self.last_window is not None:
            x0, y0, x1, y1 = self.last_window
            full = (x1 - x0, y1 - y0) == (image.shape[1], image.shape[0]) and (x0, y0) == (0, 0)
            if not full:
                cv2.rectangle(image, (x0, y0), (x1, y1), (180, 180, 180), 1)
                cv2.putText(image, "zoom" + (" (retry: full ROI)" if self.last_window_was_retry else ""),
                            (x0 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        if obs.source == "pose" and self.last_pose is not None:
            for e in self.last_pose.ears:
                if e is not None:
                    cv2.line(image, (int(self.last_pose.nose[0]), int(self.last_pose.nose[1])), (int(e[0]), int(e[1])), (200, 200, 0), 1)
            if obs.tube_pt is not None:
                cv2.putText(image, "coarse anchors (pose)", (int(obs.tube_pt[0]) + 10, int(obs.tube_pt[1]) + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1, cv2.LINE_AA)
        remembered = obs.tube_pt is None and f.tube_pt is not None
        if f.tube_pt is not None and f.face_w:
            cx, cy = int(f.tube_pt[0]), int(f.tube_pt[1])
            colour = (160, 160, 160) if remembered else (255, 0, 255)
            cv2.circle(image, (cx, cy), 6, colour, 2)
            cv2.circle(image, (cx, cy), int(0.3 * f.face_w), (0, 140, 255), 1)  # d_grasp band
            cv2.circle(image, (cx, cy), int(0.6 * f.face_w), (0, 200, 255), 1)  # d_watch band
            if remembered:
                cv2.putText(image, "face occluded: using last anchor", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        for tp in (obs.tape_pts or ((f.tape_pt,) if f.tape_pt else ())):
            tx, ty = int(tp[0]), int(tp[1])
            cv2.rectangle(image, (tx - 6, ty - 6), (tx + 6, ty + 6), (0, 255, 255), 1)
            cv2.putText(image, "tape", (tx + 8, ty + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        for hd in obs.hands:
            g = (int(hd.grip[0]), int(hd.grip[1]))
            cv2.circle(image, g, 5, (0, 255, 0), -1)
            cv2.circle(image, (int(hd.wrist[0]), int(hd.wrist[1])), 4, (255, 255, 0), -1)
            cv2.line(image, g, (int(hd.wrist[0]), int(hd.wrist[1])), (255, 255, 0), 1)
            if f.tube_pt is not None:
                cv2.line(image, g, (int(f.tube_pt[0]), int(f.tube_pt[1])), (0, 255, 0), 1)

    @property
    def name(self) -> str:  # type: ignore[override]
        return "mediapipe pose+face+hands" if self._pose is not None else "mediapipe face+hands"

    def close(self) -> None:
        self._hands.close()
        self._face.close()
        if self._pose is not None:
            self._pose.close()
