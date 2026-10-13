"""AlarmManager tests. Silent sounder, recording notifier, injected clock."""

import json
from pathlib import Path

import pytest

from nurse24hr.alarm import AfplaySounder, AlarmConfig, AlarmManager, NullSounder
from nurse24hr.events import Event, State
from nurse24hr.notify import RecordingNotifier


def ev(t, prev, state, reason="r", kind="transition"):
    return Event(t, kind, prev, state, reason)


@pytest.fixture
def mgr(tmp_path):
    snd = tmp_path / "alarm.wav"
    snd.write_bytes(b"RIFF")
    flt = tmp_path / "fault.wav"
    flt.write_bytes(b"RIFF")
    cfg = AlarmConfig(alarm_sound=str(snd), fault_sound=str(flt), escalate_after_s=30.0,
                      audit_log=str(tmp_path / "logs" / "audit.jsonl"))
    n = RecordingNotifier()
    m = AlarmManager(n, NullSounder(), cfg)
    yield m
    m.close()


def test_level_mapping(mgr):
    mgr.handle(ev(1, State.IDLE, State.WATCH))
    mgr.handle(ev(2, State.WATCH, State.WARN))
    mgr.handle(ev(3, State.WARN, State.ALARM))
    levels = [p["level"] for p in mgr.notifier.of("event")]
    assert levels == ["info", "warn", "alarm"]
    p = mgr.notifier.of("event")[2]
    assert p["state"] == "ALARM" and p["prev"] == "WARN" and p["t"] == 3 and p["reason"] == "r"
    assert len(p["event_id"]) == 36


def test_fault_level(mgr):
    mgr.handle(ev(5, State.IDLE, State.FAULT, "no view"))
    assert mgr.notifier.of("event")[0]["level"] == "fault"
    assert mgr.is_latched and mgr.latched.level == "fault"
    assert mgr.sounder.current == (mgr.cfg.fault_sound, True)


def test_warn_is_silent_at_bedside_by_default(mgr):
    mgr.handle(ev(2, State.WATCH, State.WARN))
    assert mgr.sounder.played == []
    assert not mgr.is_latched
    assert mgr.notifier.of("event")[0]["level"] == "warn"


def test_warn_sound_when_configured(tmp_path):
    chime = tmp_path / "chime.wav"
    chime.write_bytes(b"x")
    m = AlarmManager(RecordingNotifier(), NullSounder(), AlarmConfig(warn_sound=str(chime), audit_log=None))
    m.handle(ev(2, State.WATCH, State.WARN))
    assert m.sounder.played == [(str(chime), False)]
    assert not m.is_latched


def test_alarm_latches_loops_and_ack_clears(mgr):
    eid = mgr.handle(ev(3, State.WARN, State.ALARM))
    assert mgr.is_latched
    assert mgr.sounder.current == (mgr.cfg.alarm_sound, True)
    # an IDLE transition cannot happen while latched, but even if fed, sound stays on
    mgr.handle(ev(4, State.ALARM, State.IDLE))
    assert mgr.sounder.current is not None
    assert mgr.ack(10.0, user="nurse-1") is True
    assert not mgr.is_latched
    assert mgr.sounder.current is None and mgr.sounder.stops >= 1
    ack = mgr.notifier.of("ack")
    assert ack == [{"event_id": eid, "user": "nurse-1", "t": 10.0, "latched_for_s": 7.0}]
    assert mgr.ack(11.0) is False
    assert len(mgr.notifier.of("ack")) == 1


def test_machine_ack_event_clears(mgr):
    mgr.handle(ev(3, State.WARN, State.ALARM))
    mgr.handle(ev(9, State.ALARM, State.IDLE, "acknowledged", kind="ack"))
    assert not mgr.is_latched
    assert mgr.notifier.of("ack")[0]["user"] == "machine"


def test_idle_transition_stops_sound_when_not_latched(tmp_path):
    chime = tmp_path / "chime.wav"
    chime.write_bytes(b"x")
    m = AlarmManager(RecordingNotifier(), NullSounder(), AlarmConfig(warn_sound=str(chime), audit_log=None))
    m.handle(ev(2, State.WATCH, State.WARN))
    m.handle(ev(5, State.WARN, State.IDLE))
    assert m.sounder.stops == 1


def test_escalation_timing_and_repeat(mgr):
    mgr.handle(ev(3.0, State.WARN, State.ALARM))
    for t in (10.0, 32.9):
        mgr.tick(t)
    assert mgr.notifier.of("escalation") == []
    mgr.tick(33.0)
    esc = mgr.notifier.of("escalation")
    assert len(esc) == 1 and esc[0]["n"] == 1 and esc[0]["unacked_for_s"] == 30.0
    mgr.tick(50.0)
    assert len(mgr.notifier.of("escalation")) == 1
    mgr.tick(63.0)
    assert len(mgr.notifier.of("escalation")) == 2
    assert mgr.notifier.of("escalation")[1]["n"] == 2
    mgr.ack(64.0)
    mgr.tick(200.0)
    assert len(mgr.notifier.of("escalation")) == 2


def test_tick_catches_up_multiple_missed_escalations(mgr):
    mgr.handle(ev(0.0, State.WARN, State.ALARM))
    mgr.tick(95.0)
    assert [e["n"] for e in mgr.notifier.of("escalation")] == [1, 2, 3]


def test_no_escalation_without_latch(mgr):
    mgr.handle(ev(0.0, State.IDLE, State.WATCH))
    mgr.tick(1000.0)
    assert mgr.notifier.of("escalation") == []


def test_suppressed_event_notified(mgr):
    mgr.handle(Event(4.0, "suppressed", State.WATCH, State.WATCH, "WARN suppressed, 2 persons in ROI"))
    p = mgr.notifier.of("event")[0]
    assert p["level"] == "suppressed" and p["state"] == "WATCH"
    assert mgr.sounder.played == [] and not mgr.is_latched


def test_missing_sound_file_warned_once(tmp_path):
    cfg = AlarmConfig(alarm_sound=str(tmp_path / "nope.wav"), audit_log=None)
    m = AlarmManager(RecordingNotifier(), NullSounder(), cfg)
    m.handle(ev(1, State.WARN, State.ALARM))
    m.ack(2)
    m.handle(ev(3, State.WARN, State.ALARM))
    warnings = [p for p in m.notifier.of("event") if p["level"] == "warning"]
    assert len(warnings) == 1 and "nope.wav" in warnings[0]["reason"]
    assert m.sounder.played == []
    assert m.is_latched  # still latched and notified even without sound


def test_raise_fault_latches_and_acks(mgr):
    eid = mgr.raise_fault("camera_loss", "no frame for 2.1s", 7.0)
    assert mgr.is_latched and mgr.latched.event_id == eid
    assert mgr.sounder.current == (mgr.cfg.fault_sound, True)
    p = mgr.notifier.of("event")[0]
    assert p["level"] == "fault" and p["kind"] == "camera_loss" and p["prev"] is None
    mgr.tick(37.0)
    assert len(mgr.notifier.of("escalation")) == 1
    assert mgr.ack(38.0)
    assert mgr.notifier.of("ack")[0]["event_id"] == eid


def test_raise_fault_keeps_existing_alarm_latch(mgr):
    eid = mgr.handle(ev(3, State.WARN, State.ALARM))
    mgr.raise_fault("inference_stall", "no features", 4.0)
    assert mgr.latched.event_id == eid and mgr.latched.level == "alarm"
    assert mgr.sounder.current == (mgr.cfg.alarm_sound, True)
    assert len(mgr.notifier.of("event")) == 2


def test_audit_jsonl(mgr):
    eid = mgr.handle(ev(3.0, State.WARN, State.ALARM, "pull"))
    mgr.tick(33.0)
    mgr.ack(34.0, user="n")
    mgr.raise_fault("camera_loss", "x", 40.0)
    lines = [json.loads(l) for l in Path(mgr.cfg.audit_log).read_text().splitlines()]
    assert [l["entry"] for l in lines] == ["event", "escalation", "ack", "fault"]
    assert lines[0]["event_id"] == eid and lines[0]["reason"] == "pull" and lines[0]["level"] == "alarm"
    assert lines[1]["n"] == 1
    assert lines[2]["user"] == "n"
    assert lines[3]["kind"] == "camera_loss" and lines[3]["reason"] == "x"


def test_audit_disabled(tmp_path):
    m = AlarmManager(RecordingNotifier(), NullSounder(), AlarmConfig(audit_log=None))
    m.handle(ev(1, State.IDLE, State.WATCH))
    assert not (tmp_path / "logs").exists()
    assert m._audit_fh is None


def test_config_from_dict_matches_yaml_section():
    cfg = AlarmConfig.from_dict({"warn_sound": None, "alarm_sound": "sounds/alarm.wav",
                                 "fault_sound": "sounds/fault.wav", "escalate_after_s": 30})
    assert cfg.escalate_after_s == 30.0 and cfg.alarm_sound == "sounds/alarm.wav"
    with pytest.raises(KeyError):
        AlarmConfig.from_dict({"alarm_sounds": "x"})


def test_afplay_sounder_missing_binary_never_raises(tmp_path):
    s = AfplaySounder(binary="definitely-not-a-real-binary-xyz")
    s.play(str(tmp_path / "a.wav"), loop=True)
    s.tick()
    s.stop()
    assert s.error and "not found" in s.error
    assert s.proc is None
