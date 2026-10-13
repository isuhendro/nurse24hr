"""MQTT sink for notifications and inbound nurse acknowledgements (TS-001 §6).

Topics, per bed::

    nurse24hr/<bed>/state       edge -> client   throttled to 1 Hz, retained
    nurse24hr/<bed>/event       edge -> client   QoS 1
    nurse24hr/<bed>/escalation  edge -> client   QoS 1
    nurse24hr/<bed>/heartbeat   edge -> station  QoS 1
    nurse24hr/<bed>/ack         client -> edge   {event_id, user, t}

Design rules
------------
* ``notify`` never raises. The alarm path must keep working when the broker
  is down; failures are counted in ``errors`` and paho reconnects on its own.
* ``paho-mqtt`` is imported lazily so the rest of the package (state machine,
  capture, tests) does not depend on it. Inject a duck-typed ``client`` to
  test without a broker or the library.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import Any

from nurse24hr.notify import Notifier

AckCallback = Callable[[dict[str, Any]], None]


class MqttNotifier(Notifier):
    """Publishes notifications to an MQTT broker and dispatches inbound acks."""

    def __init__(self, broker: str = "localhost", port: int = 1883, bed: str = "bed-01", *,
                 client: Any | None = None, qos: int = 1, state_interval: float = 1.0,
                 clock: Callable[[], float] = time.monotonic, keepalive: int = 60) -> None:
        self.bed = bed
        self.qos = qos
        self.state_interval = state_interval
        self._clock = clock
        self._last_state_t = -math.inf
        self._ack_callbacks: list[AckCallback] = []
        self.errors: list[str] = []
        self.published = 0

        if client is None:
            try:
                import paho.mqtt.client as mqtt  # noqa: PLC0415 - deliberate lazy import
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError(
                    "MqttNotifier needs paho-mqtt. Install it with `uv add paho-mqtt` "
                    "or use ConsoleNotifier / WebClient instead."
                ) from exc
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"nurse24hr-{bed}")
        self.client = client

        # Re-subscribe on every (re)connect so acks survive broker restarts.
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        try:
            connect_async = getattr(self.client, "connect_async", None)
            if connect_async is not None:
                connect_async(broker, port, keepalive)
            else:
                self.client.connect(broker, port, keepalive)
            self.client.loop_start()
        except Exception as exc:  # noqa: BLE001 - the alarm must not die with the broker
            self.errors.append(f"connect: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------- outbound

    def topic(self, name: str) -> str:
        return f"nurse24hr/{self.bed}/{name}"

    def notify(self, topic: str, payload: dict[str, Any]) -> None:
        if topic == "state":
            now = self._clock()
            if now - self._last_state_t < self.state_interval:
                return
            self._last_state_t = now
        try:
            data = json.dumps(payload, default=str)
            result = self.client.publish(self.topic(topic), data, qos=self.qos, retain=(topic == "state"))
            rc = getattr(result, "rc", 0)
            if rc != 0:
                self.errors.append(f"publish {topic}: rc={rc}")
            else:
                self.published += 1
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"publish {topic}: {type(exc).__name__}: {exc}")

    # -------------------------------------------------------------- inbound

    def on_ack(self, callback: AckCallback) -> None:
        """Register a callback for ``{event_id, user, t}`` acks from nurses."""
        self._ack_callbacks.append(callback)

    def _on_connect(self, client: Any, userdata: Any = None, flags: Any = None,
                    reason_code: Any = None, properties: Any = None) -> None:
        try:
            client.subscribe(self.topic("ack"), self.qos)
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"subscribe: {type(exc).__name__}: {exc}")

    def _on_message(self, client: Any, userdata: Any, message: Any) -> None:
        if getattr(message, "topic", None) != self.topic("ack"):
            return
        try:
            raw = message.payload
            ack = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
            if not isinstance(ack, dict):
                raise ValueError("ack payload must be a JSON object")
            ack.setdefault("t", time.time())
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"bad ack: {type(exc).__name__}: {exc}")
            return
        for cb in list(self._ack_callbacks):
            try:
                cb(ack)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"ack callback: {type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- close

    def close(self) -> None:
        for name in ("loop_stop", "disconnect"):
            fn = getattr(self.client, name, None)
            if fn is None:
                continue
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{name}: {type(exc).__name__}: {exc}")
