"""Parquet logging, replay scoring, sweeps and the session schedule."""

import csv
from dataclasses import replace

import pytest

from nurse24hr.config import Thresholds
from nurse24hr.events import State
from nurse24hr.features import Features
from nurse24hr.features_log import FeatureLogger, read_features, read_states
from nurse24hr.replay import main as replay_main
from nurse24hr.replay import parse_sweep, read_labels, replay_with_ack, score, sweep
from nurse24hr.session import PROTOCOL, Block, schedule, select_blocks, write_labels
from nurse24hr.simulate import generate


def test_parquet_roundtrip(tmp_path):
    sc = generate("grasp_and_pull", 1)
    path = tmp_path / "f.parquet"
    with FeatureLogger(path, flush_every=7) as lg:
        for i, f in enumerate(sc.features):
            lg.log(f, "IDLE" if i < 10 else "WATCH")
    back = read_features(path)
    assert len(back) == len(sc.features)
    assert back[5].t == sc.features[5].t
    assert back[5].d == pytest.approx(sc.features[5].d)
    assert back[5].pinch == pytest.approx(sc.features[5].pinch)
    assert read_states(path)[:11] == ["IDLE"] * 10 + ["WATCH"]


def test_parquet_handles_none_fields(tmp_path):
    path = tmp_path / "f.parquet"
    with FeatureLogger(path) as lg:
        lg.log(Features(t=0.0, d=None, conf=0.0, n_persons=0), None)
        lg.log(Features(t=0.1, d=0.5, conf=0.9, wrist_l=(1.0, 2.0), tube_pt=(3.0, 4.0), face_w=100.0), "WATCH")
    a, b = read_features(path)
    assert a.d is None and a.wrist_l is None
    assert b.wrist_l == (1.0, 2.0) and b.tube_pt == (3.0, 4.0) and b.face_w == 100.0


def _session(tmp_path):
    """Concatenate scenarios into one 'session' with labels, return (features, segments)."""
    feats, segs = [], []
    t_off = 0.0
    for name, seed in [("idle", 0), ("nose_scratch_short", 1), ("grasp_and_pull", 2), ("long_face_rub", 3), ("grasp_and_pull", 4)]:
        sc = generate(name, seed)
        for f in sc.features:
            feats.append(replace(f, t=f.t + t_off))
        for t0, t1, lab in sc.labels:
            segs.append((t0 + t_off, t1 + t_off, lab))
        t_off += sc.features[-1].t + 1 / 15 + 2.0  # gap between scenarios
    return feats, segs


def test_replay_with_ack_observes_events_after_an_alarm(tmp_path):
    feats, segs = _session(tmp_path)
    evs = replay_with_ack(feats, Thresholds())
    alarms = [e for e in evs if e.kind == "transition" and e.state is State.ALARM]
    acks = [e for e in evs if e.kind == "ack"]
    assert len(alarms) == 2 and len(acks) == 2


def test_score_recall_tta_and_false_alarm_attribution(tmp_path):
    feats, segs = _session(tmp_path)
    evs = replay_with_ack(feats, Thresholds())
    m = score(evs, segs, feats[-1].t)
    assert m.n_pull == 2 and m.recall == 1.0
    assert m.median_tta is not None and m.median_tta <= 1.5
    assert m.false_alarms == 0
    # with pull detection disabled the pulls become misses, not false alarms
    m2 = score([e for e in evs if e.state is not State.ALARM], segs, feats[-1].t)
    assert m2.recall == 0.0 and m2.false_alarms == 0
    # an alarm during a scratch is attributed to the scratch label
    scratch_t = next(t0 for t0, t1, lab in segs if lab == "scratch") + 0.1
    fake = [type(evs[0])(scratch_t, "transition", State.WARN, State.ALARM, "x")]
    m3 = score(fake, segs, feats[-1].t)
    assert m3.false_alarms_by_label == {"scratch": 1}
    assert m3.false_alarms_per_hour > 0
    assert "recall" in m3.summary()


def test_sweep_and_parse():
    grid = parse_sweep(["t_grasp=1.0,2.0", "missing_evidence_counts=true,false"])
    assert grid == {"t_grasp": [1.0, 2.0], "missing_evidence_counts": [True, False]}
    feats, segs = _session(None)
    rows = sweep(feats, segs, Thresholds(), {"t_grasp": [1.0, 1.5]}, feats[-1].t)
    assert len(rows) == 2 and all(m.n_pull == 2 for _, m in rows)


def test_replay_cli(tmp_path, capsys):
    feats, segs = _session(tmp_path)
    pq = tmp_path / "s.parquet"
    with FeatureLogger(pq) as lg:
        for f in feats:
            lg.log(f)
    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_start", "t_end", "label"])
        w.writerows(segs)
    assert read_labels(labels) == [(float(a), float(b), c) for a, b, c in segs]
    assert replay_main([str(pq), "--labels", str(labels), "--quiet", "--config", "/nonexistent.yaml"]) == 0
    out = capsys.readouterr().out
    assert "recall 1.00 (2/2)" in out
    assert replay_main([str(pq), "--labels", str(labels), "--sweep", "t_grasp=1.0,1.5", "--config", "/nonexistent.yaml"]) == 0
    assert capsys.readouterr().out.count("recall") == 2


def test_schedule_is_contiguous_and_labels_cues_only():
    steps = schedule(PROTOCOL)
    for a, b in zip(steps, steps[1:]):
        assert b.t_start == pytest.approx(a.t_end)
    cues = [s for s in steps if not s.is_rest]
    assert sum(1 for s in cues if s.label == "pull") == 10
    assert sum(1 for s in cues if s.label == "scratch") == 10
    assert all(s.label == "idle" for s in steps if s.is_rest)
    assert 20 * 60 > steps[-1].t_end > 8 * 60  # whole protocol is a reasonable session length


def test_select_blocks_and_write_labels(tmp_path):
    blocks = select_blocks("2,7")
    assert [b.id for b in blocks] == [2, 7]
    steps = schedule(blocks, lead_in_s=1.0)
    path = tmp_path / "labels.csv"
    write_labels(path, steps)
    segs = read_labels(path)
    assert segs[0] == (0.0, 1.0, "idle")
    assert {lab for _, _, lab in segs} == {"idle", "scratch", "pull"}


def test_custom_block_continuous():
    steps = schedule([Block(1, "x", "idle", continuous_s=30)], lead_in_s=0.0)
    assert len(steps) == 2 and steps[1].t_end == 30
