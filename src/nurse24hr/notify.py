"""Outbound notification interface shared by the alarm manager, watchdog, run loop and MQTT.

A ``Notifier`` receives a short topic (``event``, ``ack``, ``escalation``,
``heartbeat``, ``state``) and a JSON-serialisable payload. The MQTT notifier
maps ``topic`` to ``nurse24hr/<bed>/<topic>``. Everything that produces
notifications depends only on this protocol, so tests can capture them and
the run loop can fan out to several sinks.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Protocol


class Notifier(Protocol):
    def notify(self, topic: str, payload: dict[str, Any]) -> None: ...


class ConsoleNotifier:
    """Prints one JSON line per notification. Default sink for the prototype.

    ``exclude`` drops chatty topics (the 1 Hz ``state`` feed by default).
    """

    def __init__(self, stream=None, exclude: set[str] | frozenset[str] = frozenset({"state"})) -> None:
        self.stream = stream or sys.stdout
        self.exclude = set(exclude)

    def notify(self, topic: str, payload: dict[str, Any]) -> None:
        if topic in self.exclude:
            return
        print(json.dumps({"topic": topic, **payload}, default=str), file=self.stream, flush=True)


class RecordingNotifier:
    """Keeps every notification in memory. For tests."""

    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    def notify(self, topic: str, payload: dict[str, Any]) -> None:
        self.items.append((topic, dict(payload)))

    def of(self, topic: str) -> list[dict[str, Any]]:
        return [p for t, p in self.items if t == topic]


class FanoutNotifier:
    """Forwards to several notifiers; one failing sink never blocks the others."""

    def __init__(self, *sinks: Notifier) -> None:
        self.sinks = list(sinks)
        self.errors: list[str] = []

    def notify(self, topic: str, payload: dict[str, Any]) -> None:
        for s in self.sinks:
            try:
                s.notify(topic, payload)
            except Exception as exc:  # noqa: BLE001 - a sink must never take the alarm down
                self.errors.append(f"{type(s).__name__}: {exc}")
