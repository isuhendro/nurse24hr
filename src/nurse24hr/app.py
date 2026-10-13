"""Run loop: capture -> perception -> state machine -> alarm / recorder / notifiers, with watchdog.

Everything is wired here and nowhere else. Each component is optional and
fails independently of the alarm path:

* perception: MediaPipe hands + face (default), optionally wrapped by the tube
  detector when weights exist, or ``none`` (zero confidence -> FAULT, by design)
* alarm manager: sounds, latching, escalation, audit log
* recorder: pre/post clips on WARN / ALARM / FAULT
* notifiers: console (always), nurse web page (``--web``), MQTT (``--mqtt``)
* watchdog: separate thread probing capture, inference and frozen frames
* feature logger: Parquet per session (``--log``)

Usage::

    nurse24hr --source webcam --web
    nurse24hr --source session.mp4 --fast --headless --log out/features.parquet
    nurse24hr --source rtsp://camera/stream --roi 100,80,540,400 --mqtt
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import yaml

from nurse24hr.alarm import AfplaySounder, AlarmConfig, AlarmManager, NullSounder
from nurse24hr.capture import CaptureThread, LatestFrame, authorize_camera
from nurse24hr.config import Thresholds
from nurse24hr.events import Event, EventMachine, State
from nurse24hr.features_log import FeatureLogger
from nurse24hr.notify import ConsoleNotifier, FanoutNotifier, Notifier
from nurse24hr.overlay import HudInfo, draw_hud
from nurse24hr.perception import NullPerceiver, Perceiver
from nurse24hr.recorder import Recorder
from nurse24hr.watchdog import ProbeSample, Watchdog, WatchdogConfig, WatchdogThread, frame_signature

WINDOW = "nurse24hr"


def parse_roi(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("roi must be x0,y0,x1,y1")
    return parts[0], parts[1], parts[2], parts[3]


def load_config(path: str | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def build_perceiver(pcfg: dict[str, Any], backend: str | None = None, roi=None) -> Perceiver:
    backend = backend or pcfg.get("backend", "mediapipe")
    if backend == "none":
        return NullPerceiver()
    if backend != "mediapipe":
        raise SystemExit(f"unknown perception backend {backend!r}")
    from nurse24hr.mp_perceiver import MediaPipeConfig, MediaPipePerceiver

    base: Perceiver = MediaPipePerceiver(MediaPipeConfig(
        hand_model=Path(pcfg.get("hand_model", "models/hand_landmarker.task")),
        face_model=Path(pcfg.get("face_model", "models/face_landmarker.task")),
        tube_side=pcfg.get("tube_side", "center"),
        tape_side=pcfg.get("tape_side"),
        roi=tuple(roi) if roi else None,
        auto_zoom=bool(pcfg.get("auto_zoom", True)),
        pose_model=Path(pcfg["pose_model"]) if pcfg.get("pose_model") else None,
        pose_every_n=int(pcfg.get("pose_every_n", 1)),
        pose_detection_confidence=float(pcfg.get("pose_detection_confidence", 0.25)),
        pose_hand_min_vis=float(pcfg.get("pose_hand_min_vis", 0.5)),
        face_scale_max=float(pcfg.get("face_scale_max", 1.6)),
        face_relearn_s=float(pcfg.get("face_relearn_s", 30.0)),
        min_detection_confidence=float(pcfg.get("min_detection_confidence", 0.4)),
        num_hands=int(pcfg.get("num_hands", 4)),
        num_faces=int(pcfg.get("num_faces", 2)),
    ))
    weights = pcfg.get("tube_weights")
    if weights:
        from nurse24hr.tube_detector import TubeDetectorPerceiver

        return TubeDetectorPerceiver(weights, base)
    return base


@dataclass
class Pipeline:
    """Everything the loop needs, built by :func:`build_pipeline` so tests can swap parts."""

    thresholds: Thresholds
    perceiver: Perceiver
    notifier: Notifier
    alarm: AlarmManager
    recorder: Recorder | None = None
    logger: FeatureLogger | None = None
    watchdog_cfg: WatchdogConfig = field(default_factory=WatchdogConfig)
    web: Any = None
    mqtt: Any = None
    bed: str = "bed-01"
    roi: tuple[int, int, int, int] | None = None
    fps: float = 15.0

    def close(self) -> None:
        for part in (self.recorder, self.logger, self.alarm, self.web, self.mqtt):
            if part is None:
                continue
            try:
                part.close() if not hasattr(part, "stop") else part.stop()
            except Exception as exc:  # noqa: BLE001
                print(f"close {type(part).__name__}: {exc}", file=sys.stderr)
        if hasattr(self.perceiver, "close"):
            self.perceiver.close()


def build_pipeline(cfg: dict[str, Any], *, headless: bool, perceiver: Perceiver | None = None, log_path: str | None = None,
                   web: bool = False, mqtt: bool = False, roi=None, fps: float | None = None,
                   recorder_enabled: bool | None = None, verbose: bool = False) -> Pipeline:
    th = Thresholds.from_dict(cfg.get("thresholds", {}))
    bed = str(cfg.get("mqtt", {}).get("bed", "bed-01"))
    sinks: list[Notifier] = [ConsoleNotifier(exclude=set() if verbose else frozenset({"state"}))]
    web_client = mqtt_client = None

    web_cfg = cfg.get("web", {})
    if web or web_cfg.get("enabled"):
        from nurse24hr.web import WebClient

        web_client = WebClient(web_cfg.get("host", "127.0.0.1"), int(web_cfg.get("port", 8765)), bed=bed).start()
        sinks.append(web_client)
        print(f"nurse page: {web_client.url}", flush=True)

    mqtt_cfg = cfg.get("mqtt", {})
    if mqtt or mqtt_cfg.get("enabled"):
        from nurse24hr.mqtt import MqttNotifier

        mqtt_client = MqttNotifier(mqtt_cfg.get("broker", "localhost"), int(mqtt_cfg.get("port", 1883)), bed)
        sinks.append(mqtt_client)

    notifier = FanoutNotifier(*sinks)
    alarm = AlarmManager(notifier=notifier, sounder=NullSounder() if headless else AfplaySounder(),
                         cfg=AlarmConfig.from_dict(cfg.get("alarm")))

    rec_cfg = cfg.get("recorder", {})
    enabled = rec_cfg.get("enabled", True) if recorder_enabled is None else recorder_enabled
    cap_fps = fps or float(cfg.get("capture", {}).get("fps", 15))
    recorder = Recorder(rec_cfg.get("out_dir", "clips"), pre_s=float(rec_cfg.get("pre_s", 10)),
                        post_s=float(rec_cfg.get("post_s", 10)), fps=cap_fps, enabled=bool(enabled)) if enabled else None

    log_path = log_path or cfg.get("logging", {}).get("features")
    if log_path == "auto":
        log_path = f"logs/features-{time.strftime('%Y%m%d-%H%M%S')}.parquet"
    logger = FeatureLogger(Path(log_path)) if log_path else None
    if logger is not None:
        print(f"features log: {log_path}", flush=True)

    return Pipeline(
        thresholds=th,
        perceiver=perceiver or build_perceiver(cfg.get("perception", {})),
        notifier=notifier, alarm=alarm, recorder=recorder, logger=logger,
        watchdog_cfg=WatchdogConfig.from_dict(cfg.get("watchdog")),
        web=web_client, mqtt=mqtt_client, bed=bed, roi=roi, fps=cap_fps,
    )


def run(source: str, p: Pipeline, *, realtime: bool = True, headless: bool = False, loop: bool = False,
        max_frames: int | None = None, on_event=None) -> EventMachine:
    """Blocking run loop. Returns the machine when the source ends or the user quits."""
    slot = LatestFrame()
    machine = EventMachine(p.thresholds)
    auth_error = authorize_camera(source)
    if auth_error:
        raise SystemExit(f"capture error: {auth_error}")
    cap = CaptureThread(source, slot, fps=p.fps, realtime=realtime, loop=loop)
    recent: deque[float] = deque(maxlen=30)
    state_holder = {"last_features_wall": None, "sig": None, "fps": 0.0, "frame_t": 0.0}
    ack_requests: deque[dict[str, Any]] = deque()

    def emit(ev: Event) -> None:
        if on_event:
            on_event(ev)
        else:
            print(str(ev), flush=True)
        p.alarm.handle(ev)
        if p.recorder is not None and ev.kind == "transition" and ev.state in (State.WARN, State.ALARM, State.FAULT):
            p.recorder.trigger(ev)

    quit_flag = threading.Event()

    def request_ack(payload: dict[str, Any]) -> None:
        ack_requests.append(payload)  # applied on the loop thread, never from a network thread

    for client in (p.web, p.mqtt):
        if client is not None:
            client.on_ack(request_ack)
    _start_stdin_reader(request_ack, quit_flag)

    def do_ack(user: str) -> None:
        t = state_holder["frame_t"]
        was_latched = machine.state.latched or p.alarm.is_latched
        for ev in machine.ack(t):
            emit(ev)
        p.alarm.ack(t, user)
        print(f"ACK by {user}: " + ("cleared, sound stopped" if was_latched else "nothing latched"), flush=True)
        # The nurse has re-seated the tube: a detector baseline must restart or a
        # still-displaced tube would re-alarm immediately.
        if hasattr(p.perceiver, "reset_baseline"):
            p.perceiver.reset_baseline()

    def probe() -> ProbeSample:
        return ProbeSample(cap.last_read_t, cap.error, cap.ended.is_set(), state_holder["last_features_wall"],
                           state_holder["sig"], fps=state_holder["fps"], frames_read=cap.frames_read)

    def on_fault(fault) -> None:
        p.alarm.raise_fault(fault.kind, fault.reason, state_holder["frame_t"])

    watchdog = WatchdogThread(Watchdog(p.watchdog_cfg, p.notifier), probe, on_fault=on_fault)
    cap.start()
    watchdog.start()
    paused = False
    processed = 0
    last_state_pub = -1.0
    f = None
    try:
        while True:
            while ack_requests:
                do_ack(str(ack_requests.popleft().get("user", "remote")))
            if quit_flag.is_set():
                break
            if cap.error:
                print(f"capture error: {cap.error}", file=sys.stderr)
                break
            frame = slot.get(timeout=0.5)
            if frame is None:
                if cap.ended.is_set():
                    if cap.error:
                        print(f"capture error: {cap.error}", file=sys.stderr)
                    break
                p.alarm.tick(state_holder["frame_t"])
                if not headless and _handle_keys(do_ack) == "quit":
                    break
                continue

            state_holder["frame_t"] = frame.t
            state_holder["sig"] = frame_signature(frame.image)
            if not paused:
                # Full frame in, full-frame coordinates out: the perceiver owns the ROI / zoom crop.
                f = p.perceiver.process(frame.image, frame.t)
                state_holder["last_features_wall"] = time.monotonic()
                for ev in machine.step(f):
                    emit(ev)
                if p.logger is not None:
                    p.logger.log(f, machine.state.value)
                if p.recorder is not None:
                    p.recorder.push(frame, {"d": f.d, "conf": f.conf, "state": machine.state.value})
                processed += 1
            p.alarm.tick(frame.t)

            if frame.t - last_state_pub >= 1.0:
                last_state_pub = frame.t
                p.notifier.notify("state", {"bed": p.bed, "state": machine.state.value, "t": frame.t,
                                            "d": None if f is None else f.d, "conf": None if f is None else f.conf})

            now = time.monotonic()
            recent.append(now)
            state_holder["fps"] = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else 0.0
            age = (now - cap.last_read_t) if cap.last_read_t else 0.0

            if not headless:
                img = frame.image.copy()
                if f is not None:
                    p.perceiver.draw(img, f)
                draw_hud(img, HudInfo(str(source), p.perceiver.name, state_holder["fps"], age, slot.dropped, p.roi, paused), f, machine)
                cv2.imshow(WINDOW, img)
                action = _handle_keys(do_ack)
                if action == "quit":
                    break
                if action == "pause":
                    paused = not paused

            if max_frames is not None and processed >= max_frames:
                break
    finally:
        watchdog.stop()
        cap.stop()
        cap.join(timeout=2.0)
        if not headless:
            cv2.destroyAllWindows()
    return machine


def _start_stdin_reader(request_ack, quit_flag: threading.Event) -> None:
    """Accept ``a`` (ack) and ``q`` (quit) typed in the terminal, so acknowledging
    does not depend on the video window having keyboard focus."""
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return
    except (AttributeError, ValueError):
        return

    def reader() -> None:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in ("a", "ack"):
                request_ack({"user": "terminal"})
            elif cmd in ("q", "quit"):
                quit_flag.set()
                return

    threading.Thread(target=reader, name="stdin-ack", daemon=True).start()
    print("terminal: type a + Enter to acknowledge, q + Enter to quit", flush=True)


def _handle_keys(do_ack) -> str | None:
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        return "quit"
    if key == ord("a"):
        do_ack("local")
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
    ap.add_argument("--perceiver", choices=["mediapipe", "none"], default=None, help="override perception backend")
    ap.add_argument("--log", default=None, help="write per-frame Features to this Parquet file")
    ap.add_argument("--web", action="store_true", help="serve the nurse page (see config web.host/port)")
    ap.add_argument("--mqtt", action="store_true", help="publish to the MQTT broker in config")
    ap.add_argument("--no-record", action="store_true", help="disable clip recording")
    ap.add_argument("--headless", action="store_true", help="no window; events printed to stdout")
    ap.add_argument("--verbose", action="store_true", help="also print the 1 Hz state feed")
    ap.add_argument("--fast", action="store_true", help="file sources: do not pace playback to real time")
    ap.add_argument("--loop", action="store_true", help="file sources: loop forever")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N processed frames (testing)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    capture_cfg = cfg.get("capture", {})
    source = args.source or capture_cfg.get("source", "webcam")
    roi = args.roi or (tuple(capture_cfg["roi"]) if capture_cfg.get("roi") else None)
    perceiver = build_perceiver(cfg.get("perception", {}), args.perceiver, roi=roi)

    p = build_pipeline(cfg, headless=args.headless, perceiver=perceiver, log_path=args.log, web=args.web,
                       mqtt=args.mqtt, roi=roi, fps=args.fps, recorder_enabled=False if args.no_record else None,
                       verbose=args.verbose)
    try:
        m = run(str(source), p, realtime=not args.fast, headless=args.headless, loop=args.loop, max_frames=args.max_frames)
    finally:
        p.close()
    print(f"final state: {m.state.value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
