"""Debug heads-up display drawn onto frames for the live window."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from nurse24hr.events import EventMachine, State
from nurse24hr.features import Features

STATE_COLOURS: dict[State, tuple[int, int, int]] = {  # BGR
    State.IDLE: (90, 200, 90),
    State.WATCH: (0, 200, 255),
    State.WARN: (0, 140, 255),
    State.ALARM: (0, 0, 255),
    State.FAULT: (200, 0, 200),
}


@dataclass
class HudInfo:
    source: str
    perceiver: str
    fps: float
    frame_age_s: float
    dropped: int
    roi: tuple[int, int, int, int] | None
    paused: bool = False
    selection: str | None = None
    """Who the patient is, from the perceiver (``selection_label``)."""


def _fmt(v: float | None, spec: str = ".2f") -> str:
    return "-" if v is None else format(v, spec)


def draw_hud(img: np.ndarray, info: HudInfo, f: Features | None, m: EventMachine) -> None:
    """Draw ROI, state banner and metrics in place."""
    h, w = img.shape[:2]
    colour = STATE_COLOURS[m.state]

    if info.roi is not None:
        x0, y0, x1, y1 = info.roi
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 1)
        cv2.putText(img, "bed ROI", (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    # State banner
    cv2.rectangle(img, (0, 0), (w, 34), colour, -1)
    label = m.state.value + ("  (latched: press a in this window, type a+Enter in terminal, or ACK on the page)" if m.state.latched else "")
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if info.paused:
        cv2.putText(img, "PAUSED", (w - 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    lines = [
        f"src {info.source}  {info.fps:4.1f} fps  age {info.frame_age_s * 1000:4.0f} ms  dropped {info.dropped}",
        f"perception: {info.perceiver}",
    ]
    if info.selection:
        lines.append(info.selection)
    if f is not None:
        lines += [
            f"d {_fmt(f.d)}  d_tape {_fmt(f.d_tape)}  conf {f.conf:.2f}  persons {f.n_persons}",
            f"v_rad {_fmt(f.v_radial)}  pinch {_fmt(f.pinch)}  osc {_fmt(f.osc, '.3f')}  disp {_fmt(f.tube_disp)}",
        ]
    lines.append(
        f"grasp {m.grasp_dwell:4.1f}s  tape {m.tape_dwell:4.1f}s  warn {m.warn_dwell:4.1f}s  release {m.release_dwell:4.1f}s  lowconf {m.lowconf_dwell:4.1f}s"
    )
    y = 54
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20

    cv2.putText(img, "q quit  a ack  space pause  s survey faces  double-click/1-9 select patient  0 auto", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
