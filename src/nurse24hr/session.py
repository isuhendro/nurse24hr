"""Staged recording sessions (spec §7.3).

Runs the scripted protocol on screen, records the webcam to a video file,
logs Features to Parquet in the same pass, and writes the labels CSV from the
script itself, so no manual annotation is needed. Each rep of an action is a
cue window ("scratch nose NOW") followed by a rest window; the cue window is
the labeled segment.

Usage::

    nurse24hr-session --out data/sessions/s01 --subject alice --side left
    nurse24hr-session --out data/sessions/s02 --blocks 2,6,7   # only some blocks
    nurse24hr-session --dry-run                               # print the schedule

Keys during a session: space skips the current block, q aborts (files so far are kept).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from nurse24hr.features_log import FeatureLogger


@dataclass(frozen=True)
class Block:
    id: int
    instruction: str
    label: str
    reps: int = 1
    cue_s: float = 3.0
    rest_s: float = 2.0
    continuous_s: float | None = None
    """For a single long block (idle), the total duration instead of reps."""


PROTOCOL: list[Block] = [
    Block(1, "Lie still, eyes closed, small head movements", "idle", continuous_s=120),
    Block(2, "Scratch your nose / cheek / chin / eyebrow", "scratch", reps=10, cue_s=2.0),
    Block(3, "Rub your face with your palm, wipe your mouth", "rub", reps=10, cue_s=3.0),
    Block(4, "Adjust pillow, pull blanket up, roll over", "other", reps=10, cue_s=4.0),
    Block(5, "Touch the tube without gripping it", "touch", reps=10, cue_s=2.0),
    Block(10, "Pick at / peel the tape on your cheek", "tape", reps=10, cue_s=3.0),
    Block(6, "Pinch and HOLD the tube, then release", "grasp", reps=10, cue_s=3.5),
    Block(7, "Pinch and PULL the tube out (re-tape after)", "pull", reps=10, cue_s=3.0, rest_s=6.0),
    Block(8, "Second person: adjust the tube", "caregiver", reps=5, cue_s=6.0, rest_s=4.0),
    Block(9, "Pull the blanket over your face for 10 s", "occlusion", reps=3, cue_s=10.0, rest_s=4.0),
]


@dataclass(frozen=True)
class Step:
    t_start: float
    t_end: float
    label: str
    text: str
    is_rest: bool


def schedule(blocks: list[Block], *, lead_in_s: float = 5.0) -> list[Step]:
    """Expand blocks into a timeline. Only cue windows carry action labels; rests are labeled 'idle'."""
    steps: list[Step] = []
    t = 0.0
    steps.append(Step(0.0, lead_in_s, "idle", "Get ready", True))
    t = lead_in_s
    for b in blocks:
        if b.continuous_s is not None:
            steps.append(Step(t, t + b.continuous_s, b.label, f"[{b.id}] {b.instruction}", False))
            t += b.continuous_s
            continue
        for r in range(1, b.reps + 1):
            steps.append(Step(t, t + b.cue_s, b.label, f"[{b.id}] {b.instruction}  ({r}/{b.reps})  NOW", False))
            t += b.cue_s
            steps.append(Step(t, t + b.rest_s, "idle", "rest", True))
            t += b.rest_s
    return steps


def write_labels(path: Path, steps: list[Step]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_start", "t_end", "label"])
        for s in steps:
            w.writerow([f"{s.t_start:.3f}", f"{s.t_end:.3f}", s.label])


def select_blocks(spec: str | None) -> list[Block]:
    if not spec:
        return PROTOCOL
    ids = {int(x) for x in spec.split(",")}
    return [b for b in PROTOCOL if b.id in ids]


def run_session(out_dir: Path, *, source: str = "webcam", fps: int = 15, blocks: list[Block] | None = None,
                subject: str = "", side: str = "center", perceive: bool = True, lead_in_s: float = 5.0) -> dict:
    import cv2

    from nurse24hr.capture import CaptureThread, LatestFrame
    from nurse24hr.config import Thresholds
    from nurse24hr.events import EventMachine

    out_dir.mkdir(parents=True, exist_ok=True)
    steps = schedule(blocks or PROTOCOL, lead_in_s=lead_in_s)
    total = steps[-1].t_end

    slot = LatestFrame()
    cap = CaptureThread(source, slot, fps=fps, realtime=True)
    cap.start()
    first = slot.get(timeout=5.0)
    if first is None:
        cap.stop()
        raise RuntimeError(f"no frames from {source}: {cap.error}")
    h, w = first.image.shape[:2]
    video_path = out_dir / "video.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        video_path = out_dir / "video.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    perceiver = None
    if perceive:
        from nurse24hr.mp_perceiver import MediaPipeConfig, MediaPipePerceiver
        perceiver = MediaPipePerceiver(MediaPipeConfig(tube_side=side))
    machine = EventMachine(Thresholds())
    logger = FeatureLogger(out_dir / "features.parquet")

    t0 = time.monotonic()
    executed_steps: list[Step] = []
    skipped_until: float | None = None
    offset = 0.0  # time removed by skipping blocks
    n_frames = 0
    aborted = False
    try:
        idx = 0
        while idx < len(steps):
            frame = slot.get(timeout=1.0)
            if frame is None:
                if cap.error or cap.ended.is_set():
                    break
                continue
            now = time.monotonic() - t0
            t_sess = now - offset
            step = steps[idx]
            if t_sess >= step.t_end:
                executed_steps.append(step)
                idx += 1
                continue

            # Record frame with the session clock, so labels and features align.
            writer.write(frame.image)
            n_frames += 1
            if perceiver is not None:
                f = perceiver.process(frame.image, t_sess)
                machine.step(f)
                logger.log(f, machine.state.value)

            disp = frame.image.copy()
            remaining = step.t_end - t_sess
            colour = (120, 120, 120) if step.is_rest else (0, 0, 255)
            cv2.rectangle(disp, (0, 0), (w, 70), colour, -1)
            cv2.putText(disp, step.text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp, f"{remaining:4.1f}s   session {t_sess:6.1f}/{total:.0f}s   state {machine.state.value}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if perceiver is not None and perceiver.last_obs is not None:
                perceiver.draw(disp, f)
            cv2.imshow("nurse24hr session", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                aborted = True
                break
            if key == ord(" "):
                # skip to the end of the current block: advance idx past steps sharing the block text prefix
                prefix = step.text.split("]")[0]
                while idx < len(steps) and steps[idx].text.startswith(prefix):
                    idx += 1
                if idx < len(steps):
                    offset += steps[idx].t_start - t_sess
    finally:
        cap.stop()
        writer.release()
        logger.close()
        if perceiver is not None:
            perceiver.close()
        cv2.destroyAllWindows()

    # Labels for what actually ran, in session time. Skipped steps are omitted.
    write_labels(out_dir / "labels.csv", executed_steps)
    meta = {
        "subject": subject, "side": side, "source": source, "fps": fps, "frames": n_frames,
        "video": video_path.name, "features": "features.parquet", "labels": "labels.csv",
        "aborted": aborted, "blocks": [b.id for b in (blocks or PROTOCOL)], "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "session.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nurse24hr-session", description="Run a staged recording session.")
    ap.add_argument("--out", default=None, help="output directory (default data/sessions/<timestamp>)")
    ap.add_argument("--source", default="webcam")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--blocks", default=None, help="comma-separated block ids, default all")
    ap.add_argument("--subject", default="")
    ap.add_argument("--side", default="center", choices=["center", "left", "right"], help="which nostril the tube is taped to")
    ap.add_argument("--no-perceive", action="store_true", help="record video only, derive features later")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule and exit")
    args = ap.parse_args(argv)

    blocks = select_blocks(args.blocks)
    if args.dry_run:
        for s in schedule(blocks):
            print(f"{s.t_start:7.1f} - {s.t_end:7.1f}  {s.label:10} {s.text}")
        print(f"total {schedule(blocks)[-1].t_end / 60:.1f} min")
        return 0

    out = Path(args.out) if args.out else Path("data/sessions") / time.strftime("%Y%m%d-%H%M%S")
    meta = run_session(out, source=args.source, fps=args.fps, blocks=blocks, subject=args.subject, side=args.side,
                       perceive=not args.no_perceive)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
