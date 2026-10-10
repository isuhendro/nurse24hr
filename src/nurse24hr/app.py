"""Run loop: capture -> perception -> state machine -> overlay.

Step 2 scope. Perception is the placeholder (so FAULT fires after ``t_fault``),
alarm output is a console line, and the window is optional (``--headless``).

Usage::

    nurse24hr --source webcam
    nurse24hr --source session.mp4 --fast --headless
    nurse24hr --source rtsp://camera/stream --roi 100,80,540,400
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import yaml

from nurse24hr.capture import CaptureThread, LatestFrame
from nurse24hr.config import Thresholds
from nurse24hr.events import Event, EventMachine
from nurse24hr.overlay import HudInfo, draw_hud
from nurse24hr.perception import NullPerceiver, Perceiver, crop_roi

WINDOW = "nurse24hr"


def parse_roi(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("roi must be x0,y0,x1,y1")
    return parts[0], parts[1], parts[2], parts[3]


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def run(source: str, *, thresholds: Thresholds, perceiver: Perceiver, roi: tuple[int, int, int, int] | None,
        fps: float, realtime: bool, headless: bool, loop: bool, max_frames: int | None = None,
        on_event=None) -> EventMachine:
    """Blocking run loop. Returns the machine when the source ends or the user quits."""
    slot = LatestFrame()
    cap = CaptureThread(source, slot, fps=fps, realtime=realtime, loop=loop)
    cap.start()
    machine = EventMachine(thresholds)
    on_event = on_event or (lambda e: print(str(e), flush=True))

    recent: deque[float] = deque(maxlen=30)
    paused = False
    processed = 0
    try:
        while True:
            if cap.error:
                print(f"capture error: {cap.error}", file=sys.stderr)
                break
            frame = slot.get(timeout=0.5)
            if frame is None:
                if cap.ended.is_set():
                    break
                if not headless and _handle_keys(machine, on_event, lambda: None) == "quit":
                    break
                continue

            if not paused:
                f = perceiver.process(crop_roi(frame.image, roi), frame.t)
                for ev in machine.step(f):
                    on_event(ev)
                processed += 1
            else:
                f = None

            now = time.monotonic()
            recent.append(now)
            fps_est = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else 0.0
            age = (now - cap.last_read_t) if cap.last_read_t else 0.0

            if not headless:
                img = frame.image.copy()
                if f is not None:
                    perceiver.draw(img, f)
                draw_hud(img, HudInfo(str(source), perceiver.name, fps_est, age, slot.dropped, roi, paused), f, machine)
                cv2.imshow(WINDOW, img)
                action = _handle_keys(machine, on_event, lambda: None)
                if action == "quit":
                    break
                if action == "pause":
                    paused = not paused

            if max_frames is not None and processed >= max_frames:
                break
    finally:
        cap.stop()
        cap.join(timeout=2.0)
        if not headless:
            cv2.destroyAllWindows()
    return machine


def _handle_keys(machine: EventMachine, on_event, _noop) -> str | None:
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        return "quit"
    if key == ord("a"):
        for ev in machine.ack(time.monotonic()):
            on_event(ev)
        return "ack"
    if key == ord(" "):
        return "pause"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nurse24hr", description="Feeding-tube pull-out monitor (prototype).")
    ap.add_argument("--source", default=None, help="webcam | <index> | <file> | rtsp://... (default from config)")
    ap.add_argument("--config", default="config.yaml", help="path to config.yaml")
    ap.add_argument("--roi", type=parse_roi, default=None, help="bed ROI x0,y0,x1,y1 (overrides config)")
    ap.add_argument("--fps", type=float, default=None, help="target capture fps (overrides config)")
    ap.add_argument("--headless", action="store_true", help="no window; events printed to stdout")
    ap.add_argument("--fast", action="store_true", help="file sources: do not pace playback to real time")
    ap.add_argument("--loop", action="store_true", help="file sources: loop forever")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N processed frames (testing)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    th = Thresholds.from_dict(cfg.get("thresholds", {}))
    capture_cfg = cfg.get("capture", {})
    source = args.source or capture_cfg.get("source", "webcam")
    fps = args.fps or float(capture_cfg.get("fps", 15))
    roi = args.roi or (tuple(capture_cfg["roi"]) if capture_cfg.get("roi") else None)

    def on_event(ev: Event) -> None:
        print(str(ev), flush=True)

    m = run(str(source), thresholds=th, perceiver=NullPerceiver(), roi=roi, fps=fps, realtime=not args.fast,
            headless=args.headless, loop=args.loop, max_frames=args.max_frames, on_event=on_event)
    print(f"final state: {m.state.value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
