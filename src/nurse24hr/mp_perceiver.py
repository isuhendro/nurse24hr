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

Which head is the patient
-------------------------
Every head seen in a frame is a :class:`Candidate`, numbered left to right.
By default the perceiver follows whichever face it acquired first (nearest the
tracked head when several are in view) and re-acquires freely after a loss.
:meth:`MediaPipePerceiver.select` / :meth:`select_at` *lock* onto one
candidate: from then on only a head within ``lock_radius`` face widths of the
patient's last known position is accepted, every other face is rejected like a
visitor, and the lock survives any length of occlusion. It never jumps to
another person on its own; a patient who moved far while unseen stays "not
visible" (the state machine raises FAULT, fail loud) until someone selects
again or clears the lock. ``survey`` mode runs detection on the whole bed ROI
instead of the zoom crop so every face is listed while the user chooses.

Finding people the zoom crop cannot see
---------------------------------------
Once a head is tracked, the per-frame models only look at a crop around it, so
a second person elsewhere in the bed is never detected, and a patient lying on
their side is missed even in a full-frame pass because the face detector
expects upright faces. A background *scan* therefore runs every
``survey_every_s`` (every frame in survey mode) on the whole bed ROI, once per
rotation in ``survey_rotations`` (upright and both sideways by default), using
separate IMAGE-mode models so the per-frame trackers are not disturbed. Heads
it finds are listed as candidates with the rotation they were found at; when
one is selected, the per-frame detection runs on the crop rotated the same
way (``_rot``), so a sideways patient is tracked with the full face mesh.
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
LM_MIDDLE_TIP = 12


@dataclass(frozen=True, slots=True)
class HandObs:
    grip: Point
    wrist: Point
    pinch_px: float
    score: float
    label: str  # "Left" / "Right" as reported by the model
    coarse: bool = False
    """True when the hand comes from the body model: position is usable, pinch and oscillation are not."""
    len_px: float = 0.0
    """Wrist to the farther of index and middle fingertip, in pixels. 0 = unknown (body-model hand)."""


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


@dataclass(frozen=True, slots=True)
class Candidate:
    """A head seen in one frame, offered for patient selection."""

    center: Point
    face_w: float
    source: str  # "mesh" | "pose"
    rot: int = 0
    """Rotation (0, 90, 180, 270 degrees clockwise) the image had to be turned by for the models to see this head."""


def rotate_image(img: np.ndarray, rot: int) -> np.ndarray:
    """``img`` turned ``rot`` degrees clockwise (0, 90, 180, 270)."""
    import cv2

    if rot == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def unrotate_point(xr: float, yr: float, rot: int, w: float, h: float) -> Point:
    """Map a point measured on an image turned ``rot`` degrees clockwise back onto the unturned
    image of size ``w`` x ``h``. Inverse of :func:`rotate_image` for continuous coordinates.
    Pure, for tests."""
    if rot == 90:
        return (yr, h - xr)
    if rot == 180:
        return (w - xr, h - yr)
    if rot == 270:
        return (w - yr, xr)
    return (xr, yr)


def scan_tiles(base: Box, tile: int) -> list[Box]:
    """Overlapping square windows of side ``tile`` (clamped to the ROI) covering ``base``, stepping by
    half a tile so a head on a tile edge is whole in its neighbour. ``tile`` 0 = just ``base``.
    Pure, for tests."""
    x0, y0, x1, y1 = base
    w, h = x1 - x0, y1 - y0
    if tile <= 0 or (tile >= w and tile >= h):
        return [base]
    tw, th = min(tile, w), min(tile, h)

    def starts(size: int, span: int) -> list[int]:
        step = max(1, size // 2)
        out = list(range(0, max(1, span - size + 1), step))
        if out[-1] + size < span:
            out.append(span - size)
        return out

    return [(x0 + sx, y0 + sy, x0 + sx + tw, y0 + sy + th) for sy in starts(th, h) for sx in starts(tw, w)]


def roll_rot(right: Point, left: Point) -> int:
    """Rotation (0, 90, 180, 270 clockwise) that would stand a head upright, from the subject's
    right-to-left line across the face (image coordinates, y down). An upright face has that line
    pointing +x; a face lying with its crown to the image's left has it pointing up, and the image
    must be turned 90 clockwise to right it. Pure, for tests."""
    a = math.degrees(math.atan2(left[1] - right[1], left[0] - right[0]))  # 0 = upright, -90 = line points up
    if -135 <= a < -45:
        return 90
    if 45 <= a < 135:
        return 270
    if a >= 135 or a < -135:
        return 180
    return 0


def crop_mapper(win: Box, rot: int):
    """Function taking a landmark (normalised x, y on the rotated crop of ``win``) to full-frame pixels."""
    x0, y0, x1, y1 = win
    cw, ch = x1 - x0, y1 - y0
    wr, hr = (ch, cw) if rot in (90, 270) else (cw, ch)

    def px(lm) -> Point:
        x, y = unrotate_point(lm.x * wr, lm.y * hr, rot, cw, ch)
        return (x0 + x, y0 + y)

    return px


def sane_width(track_w: float | None, baseline: float | None, min_px: float) -> float:
    """A head width to measure lock distances and crops with. The body model's ear-to-ear width
    collapses to a few pixels in profile; once that leaked into the tracked head the lock radius
    shrank to 8 px, every real face was rejected and the patient went "not visible" for minutes
    (FAULTs 2026-10-16). The face-mesh baseline wins when known; otherwise the tracked width, but
    never under ``min_px``. Pure, for tests."""
    if baseline:
        return float(baseline)
    return max(float(track_w or 0.0), float(min_px))


def lock_admits(center: Point, track: tuple[float, float, float] | None, locked: bool, radius: float) -> bool:
    """Is a head at ``center`` the locked patient? Only a head within ``radius`` face widths of the
    patient's last known position ``track`` (cx, cy, face_w) is. Unlocked, every head is admitted.
    Pure, for tests."""
    if not locked or track is None or radius <= 0:
        return True
    return _dist(center, track[:2]) <= radius * track[2]


def merge_candidates(faces: list[Candidate], pose_heads: list[Candidate] | Candidate | None) -> list[Candidate]:
    """All heads in a frame, left to right. A head is listed only when no earlier one already covers
    it (mesh faces first, then body-model heads), so one person is never offered twice, however
    many rotations found them. Pure, for tests."""
    if isinstance(pose_heads, Candidate):
        pose_heads = [pose_heads]
    cands: list[Candidate] = []
    for c in list(faces) + list(pose_heads or ()):
        if not any(_dist(c.center, k.center) <= max(k.face_w, c.face_w) for k in cands):
            cands.append(c)
    return sorted(cands, key=lambda c: c.center[0])


@dataclass(frozen=True, slots=True)
class SeenHead:
    t: float
    """When it was last seen."""
    hits: int
    """Sightings (weighted) at about this spot within the memory window."""
    cand: Candidate
    ttl: float = 1.0
    """How long after ``t`` it stays listed."""


def remember_candidates(hist: list[SeenHead], fresh: list[Candidate], t: float, memory_s: float,
                        min_sep: float = 96.0, weight: int = 1) -> list[SeenHead]:
    """Heads with a short memory, so the numbers do not blink when the models see a head in only
    some frames. A fresh head takes over the remembered one it overlaps (same person, moved a
    little; ``min_sep`` pixels at least, because the body model's head width collapses in profile)
    and inherits its hit count plus ``weight``; remembered heads unseen for their memory are
    dropped. Phantom heads from the body model jump about frame to frame, so they never build up
    hits. Pure, for tests."""
    kept = [h for h in hist if 0 <= t - h.t <= h.ttl]
    for c in fresh:
        same = [h for h in kept if _dist(h.cand.center, c.center) <= max(1.5 * h.cand.face_w, 1.5 * c.face_w, min_sep)]
        kept = [h for h in kept if h not in same]
        kept.append(SeenHead(t, weight + max((h.hits for h in same), default=0), c, memory_s))
    return kept


def plausible_hand(len_px: float, face_w: float | None, min_len: float) -> bool:
    """Is a hand-model hand big enough to be a hand? Wrist to fingertip is about 1.0 to 1.3 face
    widths on an adult and foreshortening rarely halves that, while the hand model's phantoms on a
    taped tube, a cannula or a gown collar are 0.1 to 0.3 (live false WARNs 2026-10-16). ``len_px``
    0 (body-model hand) or no face width to compare with: accept. Pure, for tests."""
    if len_px <= 0 or not face_w or min_len <= 0:
        return True
    return len_px >= min_len * face_w


def corroborated(hand: HandObs, face_w: float, sightings: list[tuple[float, Point, Point]], t: float,
                 memory_s: float, radius: float = 1.0) -> bool:
    """Does the body model agree there is a hand here? True when, in the last ``memory_s`` seconds,
    the body model placed a wrist within ``radius`` face widths of this hand's wrist, or a grip
    within ``radius`` of its grip. ``sightings`` are (t, wrist, grip) of body-model hands. The hand
    model hallucinates whole hands along a cheek or on a taped tube with nothing there, at a length
    and position a real hand can also have (false WARNs 2026-10-16); the body model never placed a
    hand near those, and did near every real grasp in every frame once a 1.5 s memory is allowed.
    ``memory_s`` 0 disables (always True). Pure, for tests."""
    if memory_s <= 0:
        return True
    return any(0 <= t - t0 <= memory_s and (_dist(w, hand.wrist) <= radius * face_w or _dist(g, hand.grip) <= radius * face_w)
               for t0, w, g in sightings)


def plausible_head(c: Candidate, roi: Box, min_px: float) -> bool:
    """Inside the bed ROI with a believable size. The body model, when it loses the person, places
    heads off the edge of the frame or a few pixels wide; those must not be offered for selection."""
    x, y = c.center
    return roi[0] <= x < roi[2] and roi[1] <= y < roi[3] and c.face_w >= min_px


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
    num_faces: int = 3
    min_detection_confidence: float = 0.4
    min_tracking_confidence: float = 0.5
    hand_detection_confidence: float = 0.6
    hand_tracking_confidence: float = 0.8
    """Hands get stricter thresholds than the face: at 0.4 / 0.5 the hand model kept a ghost hand on
    the nose and cheek for ten seconds after the real hands had left the face (false WARN
    2026-10-16, mask taken off); 0.6 / 0.8 dropped the ghost frames from 6 to 1 of 31 on that clip
    and kept more of the real-hand frames (9 of 12 against 7)."""
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
    hand_corroborate_s: float = 1.5
    """A hand-model hand within a face width of the tube counts only if the body model saw a wrist
    within a face width of it in the last this many seconds. 0 disables (hand model alone decides:
    more recall, but ghost hands on the cheek raise WARN)."""
    hand_min_len: float = 0.45
    """A hand-model hand shorter than this many face widths (wrist to fingertip) is not a hand:
    the model hallucinates small "hands" on the taped tube and the gown collar of a distant patient."""
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
    lock_radius: float = 2.0
    """Once a patient is selected, only a head within this many face widths of their last known position
    is the patient. 0 disables the radius (the lock then relies on face size alone)."""
    relock_radius: float = 4.0
    """Once the locked patient has been lost for ``track_lost_frames``, a head within this many face widths
    is accepted again if it lies the same way (same rotation) and passes the size gate: the camera was
    nudged 140 px and the patient sat 5 px outside ``lock_radius`` for minutes (FAULT 2026-10-16). A
    visitor leaning in is upright and larger, so this does not re-admit them."""
    candidate_memory_s: float = 1.0
    """A head stays listed (and numbered) for this long after the models last saw it."""
    candidate_min_px: float = 24.0
    """Heads narrower than this (or outside the bed ROI) are not offered for selection."""
    candidate_min_hits: int = 3
    """A head must be seen in this many frames at about the same spot before it is numbered."""
    survey_every_s: float = 1.0
    """Background scan of the whole bed ROI for other people this often (0 disables; survey mode scans every frame)."""
    survey_rotations: tuple[int, ...] = (0, 90, 270)
    """Clockwise rotations the scan tries, so a patient lying on their side is found too."""
    survey_cycle: bool = True
    """One rotation per scan, round robin (a third of the cost per scan; every orientation is still
    covered every ``len(survey_rotations)`` scans). False = all rotations in every scan."""
    survey_pose: bool = True
    """Run the body model in the scan as well (finds a masked or turned-away head the face mesh misses)."""
    survey_tile_px: int = 640
    """The scan looks at the bed in overlapping square tiles this big (0 = the whole ROI at once). The face
    detector shrinks whatever it is given to 128 px, so on a 1280 px frame a face under ~150 px is invisible
    to it; in a 640 px tile the same face is twice as large to the detector."""
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
            min_hand_detection_confidence=self.cfg.hand_detection_confidence,
            min_tracking_confidence=self.cfg.hand_tracking_confidence,
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
        self._last_poses: list[PoseObs] = []
        self.last_pose: PoseObs | None = None
        """The pose used for the patient this frame (the primary); other people's poses only feed the candidate list."""
        self.last_pose_window: Box | None = None
        self.deriver = FeatureDeriver()
        self.face_gate = FaceGate(scale_max=self.cfg.face_scale_max, relearn_s=self.cfg.face_relearn_s)
        self._last_ts_ms = -1
        self.last_obs: RawObservation | None = None
        self.last_landmarks: tuple[list, list] = ([], [])
        self.last_rejected_hands: list[HandObs] = []
        """Hand-model hands dropped this frame as too small to be hands (drawn for debugging)."""
        self.last_uncorroborated: list[HandObs] = []
        """Hand-model hands near the tube dropped this frame because the body model saw no wrist there."""
        self._pose_hands: deque[tuple[float, Point, Point]] = deque()  # (t, wrist, grip) of body-model hands
        self._track: tuple[float, float, float] | None = None  # (cx, cy, face_w) full-frame px
        self._lost = 0
        self._rejected: list[tuple[float, Point, float]] = []  # (t, center, face_w) of faces judged not the patient
        self.last_window: Box | None = None
        self.last_window_was_retry = False
        self.locked = False
        """True after :meth:`select`: follow that head only, never switch on our own."""
        self.survey = False
        """Detect on the whole bed ROI (no zoom crop) so every head is listed for selection."""
        self.candidates: list[Candidate] = []
        """Heads seen in the last frame, left to right. Numbered from 1 in the UI."""
        self.patient_index: int | None = None
        """Index into ``candidates`` of the head used as the patient this frame, if any."""
        self._cand_hist: list[SeenHead] = []
        self._heads_seen = 0  # faces or bodies the models reported (this frame or the last scan), listed or not
        self._scan_heads = 0
        self._rot = 0
        """Rotation the per-frame crop is turned by before detection: that of the tracked head."""
        self._scan_face = None
        self._scan_pose = None
        self._last_scan_t: float | None = None
        self._scan_i = 0
        self.last_scan: list[Candidate] = []
        """What the most recent background scan found, full-frame pixels."""
        self._t = 0.0

    # ---------------------------------------------------------- selection

    def select(self, n: int) -> Candidate | None:
        """Lock onto candidate ``n`` (1-based, left to right in the last frame). ``0`` clears the lock.
        Returns the chosen candidate, or None when there is no such head."""
        if n <= 0:
            self.clear_selection()
            return None
        if n > len(self.candidates):
            return None
        c = self.candidates[n - 1]
        self._lock_on(c)
        return c

    def select_at(self, x: float, y: float) -> Candidate | None:
        """Lock onto the head under full-frame pixel (x, y), e.g. a mouse click. None if no head is there."""
        near = [c for c in self.candidates if _dist((x, y), c.center) <= max(c.face_w, 40.0)]
        if not near:
            return None
        c = min(near, key=lambda c: _dist((x, y), c.center))
        self._lock_on(c)
        return c

    def clear_selection(self) -> None:
        """Back to automatic: follow whatever face is acquired, as before any selection."""
        self.locked = False
        self.survey = False

    def _lock_on(self, c: Candidate) -> None:
        self.locked = True
        self.survey = False
        self._track = (c.center[0], c.center[1], max(c.face_w, self.cfg.candidate_min_px))
        self._rot = c.rot
        self._lost = 0
        self._rejected.clear()
        self.face_gate.reset()
        self.face_gate.accept(c.source, c.face_w, self._t)
        self.deriver.reset()  # the previous person's smoothing history must not bleed into this one

    @property
    def lock_state(self) -> dict:
        """The lock as something to save: where the patient was last seen and how their head lies.
        ``{"locked": False}`` when following automatically."""
        if not self.locked or self._track is None:
            return {"locked": False}
        cx, cy, _ = self._track
        return {"locked": True, "cx": cx, "cy": cy, "face_w": self._ref_w(), "rot": self._rot}

    def restore_lock(self, state: dict) -> bool:
        """Re-lock from a saved :attr:`lock_state` (a fixed camera over a fixed bed: the patient is
        where they were). Returns False, and changes nothing, when the state holds no lock."""
        if not state.get("locked"):
            return False
        c = Candidate((float(state["cx"]), float(state["cy"])), float(state["face_w"]), "mesh", int(state.get("rot", 0)))
        self._lock_on(c)
        return True

    def _ref_w(self, fallback: float | None = None) -> float:
        """Head width for lock distances: the face-mesh baseline, else the tracked width with a floor."""
        return sane_width(self._track[2] if self._track else fallback, self.face_gate.baseline("mesh"), self.cfg.candidate_min_px)

    def _admits(self, center: Point) -> bool:
        track = (self._track[0], self._track[1], self._ref_w()) if self._track else None
        return lock_admits(center, track, self.locked, self.cfg.lock_radius)

    def _readmits(self, c: Candidate) -> bool:
        """Relock rule: the patient has been lost for a while and this head lies the same way, within
        ``relock_radius``. Size is checked by the caller's face gate for mesh faces."""
        if not self.locked or self._track is None or self._lost < self.cfg.track_lost_frames or c.rot != self._rot:
            return False
        track = (self._track[0], self._track[1], self._ref_w())
        return lock_admits(c.center, track, True, self.cfg.relock_radius)

    def _admit(self, c: Candidate) -> bool:
        return self._admits(c.center) or self._readmits(c)

    def _index_of(self, center: Point) -> int | None:
        """Index of the listed candidate at ``center``; None when it is not listed yet (too few hits)."""
        if not self.candidates:
            return None
        i = min(range(len(self.candidates)), key=lambda i: _dist(self.candidates[i].center, center))
        c = self.candidates[i]
        return i if _dist(c.center, center) <= max(1.5 * c.face_w, 96.0) else None

    @property
    def selection(self) -> dict:
        """For the HUD, the state feed and the nurse page."""
        return {"people": len(self.candidates), "patient": None if self.patient_index is None else self.patient_index + 1,
                "locked": self.locked, "survey": self.survey}

    @property
    def selection_label(self) -> str:
        who = f"#{self.patient_index + 1}" if self.patient_index is not None else "not visible"
        mode = "locked" if self.locked else "auto"
        label = f"patient {who} ({mode}), {len(self.candidates)} in view"
        if self.survey:
            return label + "  SURVEY: double-click a face or press 1-9, 0 = auto"
        if self._heads_seen > len(self.candidates):
            return label + "  (more people seen: press s to list them all)"
        return label

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

    def _mp_image(self, image_bgr: np.ndarray, win: Box, rot: int):
        import cv2

        x0, y0, x1, y1 = win
        rgb = cv2.cvtColor(rotate_image(image_bgr[y0:y1, x0:x1], rot), cv2.COLOR_BGR2RGB)
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def _detect(self, image_bgr: np.ndarray, win: Box, t: float, rot: int = 0):
        mp_img = self._mp_image(image_bgr, win, rot)
        ts = self._next_ts(t)
        face_res = self._face.detect_for_video(mp_img, ts)
        hand_res = self._hands.detect_for_video(mp_img, ts)
        return face_res, hand_res

    def _poses_from(self, res, win: Box, rot: int) -> list[PoseObs]:
        if not res.pose_landmarks:
            return []
        px = crop_mapper(win, rot)
        poses = []
        for lms in res.pose_landmarks:
            p = pose_observation(lms, len(res.pose_landmarks), px, hand_min_vis=self.cfg.pose_hand_min_vis)
            if p is not None:
                poses.append(p)
        return poses

    def _pose_in(self, image_bgr: np.ndarray, win: Box, t: float, rot: int = 0) -> list[PoseObs]:
        """Every body with a visible head in ``win``, in model order."""
        res = self._pose.detect_for_video(self._mp_image(image_bgr, win, rot), self._next_ts(t))
        return self._poses_from(res, win, rot)

    # --------------------------------------------------------------- scan

    def _scanners(self):
        """IMAGE-mode models for the background scan, created on first use. Separate from the
        per-frame VIDEO-mode trackers so scanning rotated copies does not disturb their tracking."""
        if self._scan_face is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._scan_face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(self.cfg.face_model)),
                running_mode=vision.RunningMode.IMAGE, num_faces=self.cfg.num_faces,
                min_face_detection_confidence=self.cfg.min_detection_confidence))
            if self.cfg.survey_pose and self._pose is not None:
                self._scan_pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(self.cfg.pose_model)),
                    running_mode=vision.RunningMode.IMAGE, num_poses=self.cfg.num_faces,
                    min_pose_detection_confidence=self.cfg.pose_detection_confidence))
        return self._scan_face, self._scan_pose

    def scan(self, image_bgr: np.ndarray, base: Box | None = None,
             rotations: tuple[int, ...] | None = None) -> list[Candidate]:
        """Every head in the bed ROI, trying each rotation in ``rotations`` (default: all of
        ``survey_rotations``). Full-frame pixels, left to right, each with the rotation it was found
        at. Mesh faces before body-model heads."""
        h, w = image_bgr.shape[:2]
        base = base or self.cfg.roi or (0, 0, w, h)
        face_lm, pose_lm = self._scanners()
        faces: list[Candidate] = []
        heads: list[Candidate] = []
        for rot in rotations or self.cfg.survey_rotations:
            for win in scan_tiles(base, self.cfg.survey_tile_px):
                mp_img = self._mp_image(image_bgr, win, rot)
                px = crop_mapper(win, rot)
                for lm in face_lm.detect(mp_img).face_landmarks:
                    right, left = px(lm[LM_FACE_RIGHT]), px(lm[LM_FACE_LEFT])
                    faces.append(Candidate(((right[0] + left[0]) / 2, (right[1] + left[1]) / 2), _dist(right, left), "mesh",
                                           roll_rot(right, left)))
            if pose_lm is not None:  # bodies are big: the whole ROI is enough
                mp_img = self._mp_image(image_bgr, base, rot)
                for p in self._poses_from(pose_lm.detect(mp_img), base, rot):
                    ear_r, ear_l = p.ears
                    heads.append(Candidate(p.nose, p.face_w, "pose", roll_rot(ear_r, ear_l) if ear_r and ear_l else rot))
        return merge_candidates(faces, heads)

    def _detect_pose(self, image_bgr: np.ndarray, base: Box, t: float) -> list[PoseObs]:
        """Pose on a wide crop around the last known head (a small distant patient is missed on
        the full frame), retrying on the whole bed ROI when the crop finds nothing."""
        win = detection_window(self._track if self.cfg.auto_zoom and not self.survey else None, self._lost, base,
                               scale=self.cfg.pose_track_scale, min_px=self.cfg.pose_track_min_px,
                               lost_frames=self.cfg.track_lost_frames)
        poses = self._pose_in(image_bgr, win, t, self._rot)
        self.last_pose_window = win
        if not poses and win != base:
            poses = self._pose_in(image_bgr, base, t, self._rot)
            self.last_pose_window = base
        return poses

    def _primary_pose(self, poses: list[PoseObs], t: float) -> PoseObs | None:
        """The body that is the patient: not inside a rejected face, admitted by the lock, and nearest
        the tracked head when several qualify. None when no body qualifies."""
        ok = [p for p in poses
              if not pose_near_rejected(p.nose, self._rejected, t, self.cfg.reject_memory_s) and self._admits(p.nose)]
        if not ok:
            return None
        if self._track is not None and len(ok) > 1:
            return min(ok, key=lambda p: _dist(p.nose, self._track[:2]))
        return ok[0]

    def observe(self, image_bgr: np.ndarray, t: float) -> RawObservation:
        h, w = image_bgr.shape[:2]
        base: Box = self.cfg.roi or (0, 0, w, h)
        base = (max(0, base[0]), max(0, base[1]), min(w, base[2]), min(h, base[3]))

        poses: list[PoseObs] = []
        if self._pose is not None:
            if self._frame_i % max(1, self.cfg.pose_every_n) == 0:
                self._last_poses = self._detect_pose(image_bgr, base, t)
            poses = self._last_poses
        self._frame_i += 1
        self._t = t
        self._rejected = [r for r in self._rejected if t - r[0] <= self.cfg.reject_memory_s]
        # Other people's bodies are listed for selection but never anchor the patient.
        pose = self._primary_pose(poses, t)
        pose_rejected = pose is None and bool(poses)
        if pose is None and poses:
            pose = poses[0]  # keep its hands / count for n_persons; it is treated as rejected below
        self.last_pose = pose

        track = self._track if self.cfg.auto_zoom and not self.survey else None
        if not self.survey and pose is not None and not pose_rejected and (track is None or self._lost > 0):
            # The body model knows where the head is even when the face mesh does not.
            track = (pose.nose[0], pose.nose[1], pose.face_w)
            lost = 0
        else:
            lost = self._lost
        win = detection_window(track, lost, base, scale=self.cfg.track_scale, min_px=self.cfg.track_min_px,
                               lost_frames=self.cfg.track_lost_frames)
        face_res, hand_res = self._detect(image_bgr, win, t, self._rot)
        self.last_window_was_retry = False
        if not face_res.face_landmarks and win != base and pose is None:
            # Lost inside the zoom window and no body to go by: retry this frame on the whole bed ROI.
            win = base
            face_res, hand_res = self._detect(image_bgr, win, t, self._rot)
            self.last_window_was_retry = True
        self.last_window = win
        self.last_landmarks = (face_res.face_landmarks, hand_res.hand_landmarks)
        px = crop_mapper(win, self._rot)  # crop-normalised landmark -> full-frame pixels

        # Background scan of the whole bed for people the crop cannot see, at every rotation.
        scanned: list[Candidate] | None = None
        due = self.cfg.survey_every_s > 0 and (self._last_scan_t is None or t - self._last_scan_t >= self.cfg.survey_every_s)
        if self.survey or due:
            rots = self.cfg.survey_rotations or (0,)
            if self.cfg.survey_cycle:
                rots = (rots[self._scan_i % len(rots)],)
            scanned = self.last_scan = self.scan(image_bgr, base, rots)
            self._last_scan_t = t
            self._scan_i += 1

        tube_pt = None
        face_w = None
        tape_pts: tuple = ()
        quality: float | None = None
        source = "none"
        n_rejected = 0
        faces = []
        seen: list[Candidate] = []
        for lm in face_res.face_landmarks:
            right, left = px(lm[LM_FACE_RIGHT]), px(lm[LM_FACE_LEFT])
            fw = _dist(right, left)
            center = ((right[0] + left[0]) / 2, (right[1] + left[1]) / 2)
            cand = Candidate(center, fw, "mesh", roll_rot(right, left))
            seen.append(cand)
            if self.face_gate.check("mesh", fw, t) and self._admit(cand):
                faces.append((lm, fw, center))
            else:
                # Wrong size for the patient, or (when locked) not where the patient is: someone else.
                n_rejected += 1
                self._rejected.append((t, center, fw))
        if poses:
            # Faces rejected just now may cover the body picked above: choose again.
            pose = self._primary_pose(poses, t)
            pose_rejected = pose is None
            if pose is None:
                pose = poses[0]
            self.last_pose = pose
        pose_heads = [Candidate(p.nose, p.face_w, "pose", roll_rot(*p.ears) if all(p.ears) else self._rot) for p in poses]
        fresh = [c for c in merge_candidates(seen, pose_heads) if plausible_head(c, base, self.cfg.candidate_min_px)]
        self._cand_hist = remember_candidates(self._cand_hist, fresh, t, self.cfg.candidate_memory_s)
        if scanned is not None:
            # Scan sightings are rarer than frames, so a mesh face counts as a full set of hits at
            # once and stays listed until well after the next scan; body-model heads still need to
            # be seen at the same spot by several scans (phantoms wander).
            cycle = len(self.cfg.survey_rotations) if self.cfg.survey_cycle else 1
            ttl = max(self.cfg.candidate_memory_s, 2.5 * cycle * self.cfg.survey_every_s)
            good = [c for c in scanned if plausible_head(c, base, self.cfg.candidate_min_px)]
            self._cand_hist = remember_candidates(self._cand_hist, [c for c in good if c.source == "mesh"], t, ttl,
                                                  weight=self.cfg.candidate_min_hits)
            self._cand_hist = remember_candidates(self._cand_hist, [c for c in good if c.source != "mesh"], t, ttl)
            self._scan_heads = len(good)
        patient_center: Point | None = None
        if faces:
            # Several candidates: the one nearest the tracked head.
            if self._track is not None and len(faces) > 1:
                lm, face_w, center = min(faces, key=lambda f: _dist(f[2], self._track[:2]))
            else:
                lm, face_w, center = faces[0]
            patient_center = center
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
            self._track = (pose.nose[0], pose.nose[1], self._ref_w(pose.face_w))  # not the collapsible pose width
            self._lost = 0
            quality = self.cfg.pose_conf
            source = "pose"
            patient_center = pose.nose
        else:
            if pose_rejected:
                n_rejected += 1
            self._lost += 1
            if scanned:
                # Nothing usable in the crop this frame: let the scan say where (and how turned)
                # the patient is, so next frame's crop looks there. Locked: only within the lock
                # radius. Auto: nearest to the last known head, or anyone once the track is stale.
                near = [c for c in scanned if self._admit(c) and plausible_head(c, base, self.cfg.candidate_min_px)]
                if self._track is not None and not (not self.locked and self._lost >= self.cfg.track_lost_frames):
                    reach = self.cfg.relock_radius if self._lost >= self.cfg.track_lost_frames else max(self.cfg.lock_radius, 1.0)
                    near = [c for c in near if _dist(c.center, self._track[:2]) <= reach * self._ref_w()]
                if near:
                    c = min(near, key=lambda c: _dist(c.center, self._track[:2])) if self._track is not None else near[0]
                    self._track = (c.center[0], c.center[1], c.face_w if c.source == "mesh" else self._ref_w(c.face_w))
                    self._rot = c.rot
                    self._lost = 0

        # Listed for selection: heads seen often enough, plus the head in use as the patient, always.
        self.candidates = sorted((h.cand for h in self._cand_hist if h.hits >= self.cfg.candidate_min_hits
                                  or (patient_center is not None and _dist(h.cand.center, patient_center) <= max(1.5 * h.cand.face_w, 96.0))),
                                 key=lambda c: c.center[0])
        self.patient_index = self._index_of(patient_center) if patient_center is not None else None

        hands = []
        for lm, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
            thumb = px(lm[LM_THUMB_TIP])
            index = px(lm[LM_INDEX_TIP])
            wrist = px(lm[LM_WRIST])
            hands.append(HandObs(
                grip=((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2),
                wrist=wrist,
                pinch_px=_dist(thumb, index),
                score=float(handed[0].score) if handed else 0.5,
                label=handed[0].category_name if handed else "Unknown",
                len_px=max(_dist(wrist, index), _dist(wrist, px(lm[LM_MIDDLE_TIP]))),
            ))
        ref_w = self.face_gate.baseline("mesh") or (face_w if source == "mesh" else None)
        self.last_rejected_hands = [h for h in hands if not plausible_hand(h.len_px, ref_w, self.cfg.hand_min_len)]
        hands = [h for h in hands if h not in self.last_rejected_hands]
        # Body-model wrists (any body in view) remembered for a short while, to vouch for hand-model
        # hands near the tube. Hands away from the face are kept regardless (they only feed n_persons).
        for q in poses:
            for ph in q.hands:
                self._pose_hands.append((t, ph.wrist, ph.grip))
        while self._pose_hands and t - self._pose_hands[0][0] > max(self.cfg.hand_corroborate_s, 0):
            self._pose_hands.popleft()
        self.last_uncorroborated = []
        if tube_pt is not None and face_w and self.cfg.hand_corroborate_s > 0:
            seen_hands = list(self._pose_hands)
            for h in hands:
                if _dist(h.grip, tube_pt) <= 1.0 * face_w and not corroborated(h, face_w, seen_hands, t, self.cfg.hand_corroborate_s):
                    self.last_uncorroborated.append(h)
            hands = [h for h in hands if h not in self.last_uncorroborated]
        if not hands and pose is not None:
            hands = list(pose.hands)  # coarse hands from the body model
        n_faces = max(len(face_res.face_landmarks), pose.n_poses if pose is not None else 0)
        self._heads_seen = max(n_faces, self._scan_heads)
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
        for i, c in enumerate(self.candidates):
            chosen = i == self.patient_index
            half = int(0.6 * c.face_w)
            cx, cy = int(c.center[0]), int(c.center[1])
            if chosen:
                colour, thick = (255, 0, 255), 2
            elif self.survey:
                colour, thick = (0, 255, 255), 2
            else:
                colour, thick = (140, 140, 140), 1
            cv2.rectangle(image, (cx - half, cy - half), (cx + half, cy + half), colour, thick)
            ly = cy - half - 8 if cy - half > 60 else cy - half + 22  # below the HUD banner, never off the top
            text = str(i + 1) + (" lying" if c.rot in (90, 270) else " upside down" if c.rot == 180 else "")
            cv2.putText(image, text, (cx - half + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, text, (cx - half + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
        if self.locked and self.patient_index is None and self._track is not None:
            tx, ty = int(self._track[0]), int(self._track[1])
            r = int(self.cfg.lock_radius * self._track[2]) if self.cfg.lock_radius > 0 else int(self._track[2])
            cv2.circle(image, (tx, ty), max(r, 8), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "locked patient last seen here", (tx - 90, ty - max(r, 8) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
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
        for hd, why in [(h, "not a hand") for h in self.last_rejected_hands] + [(h, "no body-model wrist") for h in self.last_uncorroborated]:
            g = (int(hd.grip[0]), int(hd.grip[1]))
            cv2.drawMarker(image, g, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 1, cv2.LINE_AA)
            cv2.putText(image, why, (g[0] + 8, g[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
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
        for m in (self._scan_face, self._scan_pose):
            if m is not None:
                m.close()
