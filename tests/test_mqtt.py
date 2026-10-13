"""MqttNotifier tests with a fake client. No broker, no paho required."""

import builtins
import json
import sys
from dataclasses import dataclass, field

import pytest

from nurse24hr.mqtt import MqttNotifier


@dataclass
class Result:
    rc: int = 0


@dataclass
class Message:
    topic: str
    payload: bytes


@dataclass
class FakeClient:
    publishes: list[tuple[str, str, int, bool]] = field(default_factory=list)
    subscriptions: list[tuple[str, int]] = field(default_factory=list)
    connected: tuple | None = None
    loop_started: bool = False
    loop_stopped: bool = False
    disconnected: bool = False
    fail_publish: bool = False
    on_connect = None
    on_message = None

    def connect_async(self, host, port, keepalive):
        self.connected = (host, port, keepalive)

    def loop_start(self):
        self.loop_started = True
        if self.on_connect:
            self.on_connect(self, None, {}, 0, None)

    def publish(self, topic, payload, qos=0, retain=False):
        if self.fail_publish:
            raise OSError("socket closed")
        self.publishes.append((topic, payload, qos, retain))
        return Result(0)

    def subscribe(self, topic, qos=0):
        self.subscriptions.append((topic, qos))

    def loop_stop(self):
        self.loop_stopped = True

    def disconnect(self):
        self.disconnected = True

    def deliver(self, topic: str, payload):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.on_message(self, None, Message(topic, body))


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(**kw):
    fc = FakeClient()
    clock = Clock()
    n = MqttNotifier("broker.local", 1883, "bed-07", client=fc, clock=clock, **kw)
    return n, fc, clock


def test_connects_and_subscribes_to_ack_on_connect():
    n, fc, _ = make()
    assert fc.connected == ("broker.local", 1883, 60)
    assert fc.loop_started
    assert fc.subscriptions == [("nurse24hr/bed-07/ack", 1)]
    assert n.errors == []


def test_topic_mapping_qos_and_payload():
    n, fc, _ = make()
    n.notify("event", {"level": "ALARM", "reason": "pull"})
    topic, payload, qos, retain = fc.publishes[-1]
    assert topic == "nurse24hr/bed-07/event"
    assert json.loads(payload) == {"level": "ALARM", "reason": "pull"}
    assert qos == 1 and retain is False
    assert n.published == 1


def test_state_is_throttled_to_one_hz_and_retained():
    n, fc, clock = make()
    for i in range(10):
        n.notify("state", {"state": "WATCH", "i": i})
        clock.t += 0.1
    states = [p for p in fc.publishes if p[0].endswith("/state")]
    assert len(states) == 1
    assert states[0][3] is True
    clock.t += 1.0
    n.notify("state", {"state": "WARN"})
    states = [p for p in fc.publishes if p[0].endswith("/state")]
    assert len(states) == 2
    assert json.loads(states[-1][1])["state"] == "WARN"


def test_other_topics_are_not_throttled():
    n, fc, _ = make()
    for _ in range(5):
        n.notify("heartbeat", {"fps": 15})
    assert len([p for p in fc.publishes if p[0].endswith("/heartbeat")]) == 5


def test_publish_failure_is_recorded_not_raised():
    n, fc, _ = make()
    fc.fail_publish = True
    n.notify("event", {"x": 1})
    assert n.published == 0
    assert n.errors and "publish event" in n.errors[0]


def test_nonzero_rc_is_recorded():
    n, fc, _ = make()
    fc.publish = lambda *a, **k: Result(4)
    n.notify("event", {"x": 1})
    assert n.errors == ["publish event: rc=4"]


def test_ack_dispatch():
    n, fc, _ = make()
    got = []
    n.on_ack(got.append)
    n.on_ack(lambda a: got.append(("second", a["user"])))
    fc.deliver("nurse24hr/bed-07/ack", {"event_id": "e1", "user": "ana", "t": 5.0})
    assert got[0] == {"event_id": "e1", "user": "ana", "t": 5.0}
    assert got[1] == ("second", "ana")


def test_ack_without_t_gets_timestamp():
    n, fc, _ = make()
    got = []
    n.on_ack(got.append)
    fc.deliver("nurse24hr/bed-07/ack", {"user": "bo"})
    assert got[0]["user"] == "bo" and isinstance(got[0]["t"], float)


def test_malformed_ack_is_counted_not_raised():
    n, fc, _ = make()
    got = []
    n.on_ack(got.append)
    fc.deliver("nurse24hr/bed-07/ack", b"not json")
    fc.deliver("nurse24hr/bed-07/ack", b"[1,2]")
    assert got == []
    assert len(n.errors) == 2 and all("bad ack" in e for e in n.errors)


def test_messages_on_other_topics_are_ignored():
    n, fc, _ = make()
    got = []
    n.on_ack(got.append)
    fc.deliver("nurse24hr/bed-99/ack", {"user": "x"})
    assert got == [] and n.errors == []


def test_failing_ack_callback_does_not_stop_others():
    n, fc, _ = make()
    got = []

    def bad(_):
        raise RuntimeError("boom")

    n.on_ack(bad)
    n.on_ack(got.append)
    fc.deliver("nurse24hr/bed-07/ack", {"user": "x"})
    assert len(got) == 1 and any("ack callback" in e for e in n.errors)


def test_close_stops_loop_and_disconnects():
    n, fc, _ = make()
    n.close()
    assert fc.loop_stopped and fc.disconnected


def test_connect_failure_is_recorded_not_raised():
    fc = FakeClient()

    def boom(*a):
        raise ConnectionRefusedError("no broker")

    fc.connect_async = boom
    n = MqttNotifier(client=fc)
    assert n.errors and "connect" in n.errors[0]
    n.notify("event", {"x": 1})  # still does not raise


def test_lazy_import_error_mentions_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "paho", None)
    monkeypatch.setitem(sys.modules, "paho.mqtt", None)
    monkeypatch.setitem(sys.modules, "paho.mqtt.client", None)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("paho"):
            raise ImportError("No module named 'paho'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="paho-mqtt"):
        MqttNotifier()
