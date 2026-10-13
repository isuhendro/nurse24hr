"""Capture thread and single-slot frame buffer.

The capture thread reads frames from a webcam, a video file or an RTSP URL and
writes each one into a :class:`LatestFrame` slot of size one. A slow consumer
therefore always sees the newest frame and stale frames are dropped, so
end-to-end latency never accumulates.

Timestamps
----------
* Live sources (webcam, RTSP): ``time.monotonic()`` at read.
* Files: ``frame_index / file_fps`` so a replay is deterministic and the state
  machine timers are the same on every run. By default playback is paced to
  real time so the overlay is watchable and stale frames are dropped like a
  live camera. ``realtime=False`` is offline analysis: it runs as fast as the
  consumer allows and applies back-pressure so **every** frame is processed.

The thread exposes ``last_read_t`` (wall clock) and ``frames_read`` so a
watchdog can detect a dead or frozen source.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    seq: int
    """Monotonic frame counter from the source."""
    t: float
    """Timestamp in seconds (source clock, see module docs)."""
    image: np.ndarray
    """BGR image as read from the source."""


class LatestFrame:
    """Thread-safe slot holding only the most recent frame."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._seen_seq = -1
        self.dropped = 0
        """Frames overwritten before any consumer took them."""

    def put(self, frame: Frame) -> None:
        """Overwrite the slot. A frame nobody took yet is counted as dropped."""
        with self._cond:
            if self._frame is not None and self._frame.seq > self._seen_seq:
                self.dropped += 1
            self._frame = frame
            self._cond.notify_all()

    def put_wait(self, frame: Frame, stop: threading.Event, poll_s: float = 0.05) -> bool:
        """Back-pressure variant: wait until the previous frame was taken, then put.

        Used for unpaced file replay so every frame is processed. Returns False
        if ``stop`` was set while waiting.
        """
        with self._cond:
            while self._frame is not None and self._frame.seq > self._seen_seq:
                if stop.is_set():
                    return False
                self._cond.wait(poll_s)
            self._frame = frame
            self._cond.notify_all()
            return True

    def get(self, timeout: float | None = None) -> Frame | None:
        """Return the newest frame not yet returned by ``get``, or None on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._frame is not None and self._frame.seq > self._seen_seq, timeout):
                return None
            assert self._frame is not None
            self._seen_seq = self._frame.seq
            self._cond.notify_all()  # wake a producer blocked in put_wait
            return self._frame

    def peek(self) -> Frame | None:
        with self._cond:
            return self._frame


def is_live_source(source: int | str) -> bool:
    return isinstance(source, int) or str(source).lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))


def authorize_camera(source: str | int) -> str | None:
    """macOS: trigger the camera permission prompt from the *calling* (main) thread.

    OpenCV's AVFoundation backend can only request authorisation on the main
    run loop; opened from a worker thread it fails with "can not spin main run
    loop from other thread". So the run loop opens the device once here, which
    shows the system prompt on first use, then sets
    ``OPENCV_AVFOUNDATION_SKIP_AUTH`` so the capture thread does not try again.
    Returns an error string if the device could not be opened, else None.
    """
    import os
    import sys

    src = parse_source(source)
    if sys.platform != "darwin" or not isinstance(src, int):
        return None
    cap = cv2.VideoCapture(src)
    ok = cap.isOpened()
    cap.release()
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
    if not ok:
        return ("cannot open camera. On macOS grant camera access to your terminal app in "
                "System Settings > Privacy & Security > Camera, then run again.")
    return None


def parse_source(source: str | int) -> int | str:
    """``"webcam"`` or a digit string -> device index; anything else is passed to OpenCV as-is."""
    if isinstance(source, int):
        return source
    if source == "webcam":
        return 0
    if source.isdigit():
        return int(source)
    return source


class CaptureThread(threading.Thread):
    """Reads frames from ``source`` into ``slot`` until stopped or the source ends."""

    def __init__(self, source: str | int, slot: LatestFrame, *, fps: float = 15.0,
                 realtime: bool = True, loop: bool = False, name: str = "capture") -> None:
        super().__init__(name=name, daemon=True)
        self.source = parse_source(source)
        self.is_file = isinstance(self.source, str) and not self.source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.slot = slot
        self.target_fps = fps
        self.realtime = realtime
        self.loop = loop and self.is_file
        self._stop_evt = threading.Event()
        self.frames_read = 0
        self.last_read_t: float | None = None
        """Wall-clock time of the last successful read. For the watchdog."""
        self.error: str | None = None
        self.ended = threading.Event()
        """Set when the source is exhausted (files) or failed."""
        self.source_fps: float | None = None
        self.size: tuple[int, int] | None = None

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # pragma: no cover - defensive; surfaced via .error
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.ended.set()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source {self.source!r}")
        if not self.is_file:
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = fps if fps and fps > 0 else None
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return cap

    def _run(self) -> None:
        cap = self._open()
        file_fps = self.source_fps or self.target_fps
        # Live sources that cannot be throttled by the driver are throttled here.
        min_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        seq = 0
        file_index = 0
        t_start = time.monotonic()
        last_emit = -float("inf")
        try:
            while not self._stop_evt.is_set():
                ok, img = cap.read()
                if not ok:
                    if self.loop:
                        cap.release()
                        cap = self._open()
                        t_start = time.monotonic() - file_index / file_fps
                        continue
                    if self.is_file:
                        return  # end of file
                    self.error = "read failed"
                    return
                now = time.monotonic()
                if self.is_file:
                    t = file_index / file_fps
                    file_index += 1
                    if self.realtime:
                        due = t_start + t
                        if due > now:
                            self._stop_evt.wait(due - now)
                            if self._stop_evt.is_set():
                                return
                else:
                    t = now
                    if now - last_emit < min_interval:
                        continue  # drop frames above target fps
                    last_emit = now
                self.frames_read += 1
                self.last_read_t = time.monotonic()
                frame = Frame(seq=seq, t=t, image=img)
                if self.is_file and not self.realtime:
                    # Offline analysis: every frame matters, so block until the consumer
                    # has taken the previous one instead of dropping it.
                    if not self.slot.put_wait(frame, self._stop_evt):
                        return
                else:
                    self.slot.put(frame)
                seq += 1
        finally:
            cap.release()
