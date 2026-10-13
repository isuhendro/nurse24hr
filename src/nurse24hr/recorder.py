"""Clip recorder: rolling pre-buffer plus post-roll around every WARN / ALARM / FAULT.

Why
---
Every alarm must be reviewable (spec G7), and the clips are the first real
training and tuning data we will ever have (spec §7.6). So the recorder keeps
the last ``pre_s`` seconds of frames in memory at all times and, when the state
machine escalates, snapshots that buffer and keeps collecting for ``post_s``
more seconds before writing one file per incident.

Design points
-------------
* Time comes from ``Frame.t`` (the source clock), never the wall clock, so a
  replayed file produces identical clips on every run.
* The ring buffer stores frame *references*, no copies. Capture already hands
  us a fresh array per frame, so this costs nothing beyond the deque itself.
* Encoding runs on a background worker fed by a queue. ``push`` and ``trigger``
  never touch the disk, so a slow SD card can never delay the alarm.
* One clip per incident: a second trigger during an active post-roll extends
  the post-roll and is recorded in the sidecar instead of starting a new file.
* Nothing here may raise into the run loop. Failures land in ``errors``.
"""

from __future__ import annotations

import json
import queue
import secrets
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nurse24hr.capture import Frame
from nurse24hr.events import Event

WRITER_CANDIDATES: tuple[tuple[str, str], ...] = (("avc1", ".mp4"), ("mp4v", ".mp4"), ("MJPG", ".avi"))
"""(fourcc, extension) tried in order until a writer opens."""

FEATURE_KEYS = ("d", "conf", "state")
"""Per-frame feature fields kept in the sidecar when the caller supplies them."""


def _event_dict(ev: Event) -> dict[str, Any]:
    return {"t": ev.t, "kind": ev.kind, "prev": ev.prev.value, "state": ev.state.value, "reason": ev.reason}


@dataclass
class Incident:
    clip_id: str
    events: list[Event]
    frames: list[Frame]
    features: list[dict[str, Any]]
    post_until: float
    extra: dict[str, Any] = field(default_factory=dict)

    def add_frame(self, frame: Frame, features: dict[str, Any] | None) -> None:
        self.frames.append(frame)
        if features is not None:
            self.features.append({"t": frame.t, **{k: features[k] for k in FEATURE_KEYS if k in features}})


class Recorder:
    """Rolling-buffer incident recorder. Call ``push`` every frame, ``trigger`` on escalation."""

    def __init__(self, out_dir: str | Path = "clips", *, pre_s: float = 10.0, post_s: float = 10.0,
                 fps: float = 15.0, enabled: bool = True) -> None:
        self.out_dir = Path(out_dir)
        self.pre_s = pre_s
        self.post_s = post_s
        self.fps = fps
        self.enabled = enabled
        self.errors: list[str] = []
        self.written: list[Path] = []
        """Paths of clips written so far (video files)."""

        self._maxlen = max(1, int(pre_s * fps * 1.5))
        self._ring: deque[tuple[Frame, dict[str, Any] | None]] = deque(maxlen=self._maxlen)
        self._active: Incident | None = None
        self._last_t: float | None = None

        self._queue: queue.Queue[Incident | None] = queue.Queue()
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run_writer, name="recorder-writer", daemon=True)
        self._closed = False
        if enabled:
            self._worker.start()

    # ------------------------------------------------------------------ public

    @property
    def buffered(self) -> int:
        """Frames currently in the pre-buffer."""
        return len(self._ring)

    @property
    def active(self) -> bool:
        """True while a post-roll is collecting frames."""
        return self._active is not None

    @property
    def pending(self) -> int:
        """Incidents queued for writing or being written right now."""
        with self._pending_lock:
            return self._pending

    def push(self, frame: Frame, features: dict[str, Any] | None = None) -> None:
        """Record one frame. Cheap: a deque append and a couple of comparisons."""
        if not self.enabled or self._closed:
            return
        try:
            self._last_t = frame.t
            self._ring.append((frame, features))
            cutoff = frame.t - self.pre_s
            while self._ring and self._ring[0][0].t < cutoff:
                self._ring.popleft()

            inc = self._active
            if inc is not None:
                inc.add_frame(frame, features)
                if frame.t >= inc.post_until:
                    self._finalize()
        except Exception as exc:  # noqa: BLE001 - recording must never take the alarm down
            self.errors.append(f"push: {type(exc).__name__}: {exc}")

    def trigger(self, event: Event, extra: dict[str, Any] | None = None) -> str:
        """Start (or extend) an incident clip around ``event``. Returns the clip id."""
        if not self.enabled or self._closed:
            return ""
        try:
            now = self._last_t if self._last_t is not None else event.t
            inc = self._active
            if inc is not None:
                inc.events.append(event)
                inc.post_until = max(inc.post_until, now + self.post_s)
                if extra:
                    inc.extra.update(extra)
                return inc.clip_id

            clip_id = f"{int(event.t * 1000):012d}_{event.state.value}_{secrets.token_hex(2)}"
            frames = [f for f, _ in self._ring]
            feats = [{"t": f.t, **{k: x[k] for k in FEATURE_KEYS if k in x}} for f, x in self._ring if x is not None]
            self._active = Incident(clip_id, [event], frames, feats, now + self.post_s, dict(extra or {}))
            return clip_id
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"trigger: {type(exc).__name__}: {exc}")
            return ""

    def close(self, timeout: float = 10.0) -> None:
        """Flush any active incident, write everything queued, stop the worker."""
        if self._closed:
            return
        self._closed = True
        if not self.enabled:
            return
        if self._active is not None:
            self._finalize()
        self._queue.put(None)
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            self.errors.append("close: writer thread did not finish in time")

    # ----------------------------------------------------------------- helpers

    def _finalize(self) -> None:
        inc = self._active
        self._active = None
        if inc is None:
            return
        with self._pending_lock:
            self._pending += 1
        self._queue.put(inc)

    def _run_writer(self) -> None:
        while True:
            inc = self._queue.get()
            if inc is None:
                return
            try:
                self._write(inc)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"write {inc.clip_id}: {type(exc).__name__}: {exc}")
            finally:
                with self._pending_lock:
                    self._pending -= 1

    def _open_writer(self, base: Path, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
        for fourcc, ext in WRITER_CANDIDATES:
            path = base.with_suffix(ext)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), self.fps, size)
            if writer.isOpened():
                return writer, path
            writer.release()
            if path.exists():
                path.unlink()
        raise RuntimeError(f"no video writer could be opened for {base.name} (tried {[c for c, _ in WRITER_CANDIDATES]})")

    def _write(self, inc: Incident) -> None:
        if not inc.frames:
            raise RuntimeError("incident has no frames")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        h, w = inc.frames[0].image.shape[:2]
        size = (w, h)
        writer, path = self._open_writer(self.out_dir / inc.clip_id, size)
        try:
            for frame in inc.frames:
                img = frame.image
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if img.shape[1] != w or img.shape[0] != h:
                    img = cv2.resize(img, size)
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                writer.write(img)
        finally:
            writer.release()

        first = inc.events[0]
        sidecar = {
            "clip_id": inc.clip_id,
            "event": _event_dict(first),
            "events": [_event_dict(e) for e in inc.events],
            "t_start": inc.frames[0].t,
            "t_end": inc.frames[-1].t,
            "n_frames": len(inc.frames),
            "fps": self.fps,
            "path": str(path),
            "extra": inc.extra,
            "features": inc.features,
        }
        (self.out_dir / f"{inc.clip_id}.json").write_text(json.dumps(sidecar, indent=1, default=str))
        self.written.append(path)
