"""Saving and restoring the patient lock across runs (app helpers, no camera or model)."""

import json

from nurse24hr.app import load_patient, save_patient


class FakeSelector:
    def __init__(self, state):
        self.lock_state = state
        self.restored = None

    def restore_lock(self, state):
        self.restored = state
        return bool(state.get("locked"))


def test_save_then_load_restores_a_lock(tmp_path):
    path = tmp_path / "state" / "patient.json"
    save_patient(FakeSelector({"locked": True, "cx": 900.0, "cy": 400.0, "face_w": 150.0, "rot": 270}), path)
    on_disk = json.loads(path.read_text())
    assert on_disk["locked"] and on_disk["rot"] == 270 and "saved_at" in on_disk
    sel = FakeSelector({"locked": False})
    assert load_patient(sel, path)["cx"] == 900.0
    assert sel.restored["face_w"] == 150.0


def test_clearing_overwrites_and_restores_nothing(tmp_path):
    path = tmp_path / "patient.json"
    save_patient(FakeSelector({"locked": True, "cx": 1.0, "cy": 2.0, "face_w": 3.0, "rot": 0}), path)
    save_patient(FakeSelector({"locked": False}), path)
    sel = FakeSelector({"locked": False})
    assert load_patient(sel, path) is None
    assert sel.restored is None


def test_missing_or_broken_file_is_ignored(tmp_path, capsys):
    sel = FakeSelector({"locked": False})
    assert load_patient(sel, tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_patient(sel, bad) is None
    assert "bad.json" in capsys.readouterr().err
    assert load_patient(None, bad) is None and save_patient(None, bad) is None
