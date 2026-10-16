"""WebClient tests: ephemeral port, real HTTP, SSE over urllib."""

import json
import threading
import time
import urllib.request

import pytest

from nurse24hr.web import WebClient


@pytest.fixture
def web():
    acks = []
    client = WebClient("127.0.0.1", 0, bed="bed-03", on_ack=acks.append).start()
    client.test_acks = acks  # type: ignore[attr-defined]
    yield client
    client.stop()


def get(url, timeout=3):
    return urllib.request.urlopen(url, timeout=timeout)


def read_sse_data(resp, timeout=3):
    """Read lines until a ``data:`` line arrives and return its JSON."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = resp.readline()
        if not line:
            break
        if line.startswith(b"data:"):
            return json.loads(line[5:].strip())
    raise AssertionError("no SSE data line received")


def test_index_serves_page_with_ack_button_and_bed(web):
    with get(web.url) as r:
        assert r.status == 200
        body = r.read().decode()
    assert "ACK" in body
    assert "bed-03" in body
    assert "EventSource('/events')" in body


def test_unknown_path_is_404(web):
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(web.url + "nope")
    assert exc.value.code == 404


def test_events_stream_replays_history_then_pushes_live(web):
    web.notify("state", {"state": "WATCH"})
    resp = get(web.url + "events", timeout=3)
    assert resp.headers["Content-Type"].startswith("text/event-stream")
    first = read_sse_data(resp)
    assert first == {"topic": "state", "state": "WATCH"}

    def push():
        time.sleep(0.2)
        web.notify("event", {"level": "ALARM", "reason": "pull"})

    threading.Thread(target=push, daemon=True).start()
    live = read_sse_data(resp)
    assert live["topic"] == "event" and live["level"] == "ALARM"
    resp.close()


def test_post_ack_calls_callback_with_user(web):
    req = urllib.request.Request(
        web.url + "ack", data=json.dumps({"user": "ana", "event_id": "e9"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with get(req) as r:
        assert r.status == 200
        body = json.loads(r.read())
    assert body["user"] == "ana" and body["event_id"] == "e9"
    assert len(web.test_acks) == 1
    assert web.test_acks[0]["user"] == "ana" and web.test_acks[0]["event_id"] == "e9"
    assert isinstance(web.test_acks[0]["t"], float)


def test_post_ack_without_user_defaults_to_nurse(web):
    req = urllib.request.Request(web.url + "ack", data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    with get(req):
        pass
    assert web.test_acks[0]["user"] == "nurse"


def test_post_ack_bad_json_is_400(web):
    req = urllib.request.Request(web.url + "ack", data=b"[1", headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(req)
    assert exc.value.code == 400
    assert web.test_acks == []


def test_additional_on_ack_callbacks_and_failures_are_isolated(web):
    got = []

    def bad(_):
        raise RuntimeError("boom")

    web.on_ack(bad)
    web.on_ack(got.append)
    req = urllib.request.Request(web.url + "ack", data=b'{"user":"x"}', headers={"Content-Type": "application/json"}, method="POST")
    with get(req):
        pass
    assert len(got) == 1 and len(web.test_acks) == 1
    assert any("ack callback" in e for e in web.errors)


def test_stop_terminates_server_and_open_streams():
    client = WebClient("127.0.0.1", 0).start()
    resp = get(client.url + "events", timeout=3)
    t0 = time.monotonic()
    client.stop()
    assert time.monotonic() - t0 < 3
    assert resp.readline() == b""  # server closed the stream
    with pytest.raises((urllib.error.URLError, ConnectionError, OSError)):
        get(client.url, timeout=1)


def post(web, path, body):
    req = urllib.request.Request(web.url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return get(req)


def test_post_select_reaches_on_select_callbacks(web):
    got = []
    web.on_select(got.append)
    with post(web, "select", {"n": 2}) as r:
        assert json.loads(r.read())["n"] == 2
    with post(web, "select", {"n": 0}):
        pass
    with post(web, "select", {"survey": True}) as r:
        assert json.loads(r.read())["survey"] is True
    assert [g.get("n", g.get("survey")) for g in got] == [2, 0, True]
    assert web.test_acks == [], "selection is not an ack"


def test_post_select_rejects_nonsense(web):
    got = []
    web.on_select(got.append)
    for body in ({}, {"n": -1}, {"n": "2"}, {"n": 2.5}, {"n": True}):
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(web, "select", body)
        assert exc.value.code == 400
    assert got == []


def test_index_page_has_patient_selector(web):
    with get(web.url) as r:
        body = r.read().decode()
    assert 'id="people"' in body and "/select" in body
