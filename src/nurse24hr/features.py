"""Per-frame feature record produced by perception and consumed by the state machine.

All distances are normalised by face width so the logic is independent of
camera distance. Velocities are in face-widths per second.
"""

from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Features:
    """One frame's worth of derived measurements.

    Fields that perception could not compute are ``None``. The state machine
    treats ``d is None`` or ``conf < conf_min`` as a low-confidence frame.
    """

    t: float
    """Monotonic timestamp in seconds."""

    d: float | None
    """min(dist(wrist_l, tube_pt), dist(wrist_r, tube_pt)) / face_w, smoothed."""

    conf: float
    """Overall confidence in this frame's measurements, 0..1."""

    n_persons: int = 1
    """People detected inside the bed ROI. >1 suppresses WARN/ALARM."""

    v_radial: float | None = None
    """Velocity of the closer wrist *away from* tube_pt. Positive = moving away."""

    pinch: float | None = None
    """Thumb-tip to index-tip distance / face_w for the closer hand. Small = pinch grip."""

    osc: float | None = None
    """Oscillation of ``d`` over the trailing 1 s (std-dev). High = scratching / rubbing."""

    tube_disp: float | None = None
    """Displacement of tube_pt from its rolling baseline / face_w. Rises during a pull."""

    # Raw measurements, kept for logging and offline debugging only.
    wrist_l: Point | None = None
    wrist_r: Point | None = None
    tube_pt: Point | None = None
    face_w: float | None = None
