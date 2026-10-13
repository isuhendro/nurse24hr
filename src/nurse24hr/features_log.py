"""Parquet logging of per-frame Features (plus the machine state after each frame).

Every session writes one Parquet file. Rows are flushed in row groups so a
crash loses at most ``flush_every`` frames. ``read_features`` turns a file
back into ``Features`` objects for :mod:`nurse24hr.replay`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nurse24hr.features import Features

SCHEMA = pa.schema([
    ("t", pa.float64()),
    ("d", pa.float64()),
    ("conf", pa.float64()),
    ("n_persons", pa.int32()),
    ("v_radial", pa.float64()),
    ("pinch", pa.float64()),
    ("osc", pa.float64()),
    ("tube_disp", pa.float64()),
    ("wrist_l_x", pa.float64()), ("wrist_l_y", pa.float64()),
    ("wrist_r_x", pa.float64()), ("wrist_r_y", pa.float64()),
    ("tube_x", pa.float64()), ("tube_y", pa.float64()),
    ("face_w", pa.float64()),
    ("d_tape", pa.float64()),
    ("tape_x", pa.float64()), ("tape_y", pa.float64()),
    ("state", pa.string()),
])


def _xy(p):
    return (None, None) if p is None else (float(p[0]), float(p[1]))


def row_from(f: Features, state: str | None) -> dict:
    lx, ly = _xy(f.wrist_l)
    rx, ry = _xy(f.wrist_r)
    tx, ty = _xy(f.tube_pt)
    px, py = _xy(f.tape_pt)
    return {
        "t": f.t, "d": f.d, "conf": f.conf, "n_persons": f.n_persons,
        "v_radial": f.v_radial, "pinch": f.pinch, "osc": f.osc, "tube_disp": f.tube_disp,
        "wrist_l_x": lx, "wrist_l_y": ly, "wrist_r_x": rx, "wrist_r_y": ry,
        "tube_x": tx, "tube_y": ty, "face_w": f.face_w, "d_tape": f.d_tape, "tape_x": px, "tape_y": py, "state": state,
    }


@dataclass
class FeatureLogger:
    path: Path
    flush_every: int = 150

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(str(self.path), SCHEMA)
        self._rows: list[dict] = []
        self.count = 0

    def log(self, f: Features, state: str | None = None) -> None:
        self._rows.append(row_from(f, state))
        self.count += 1
        if len(self._rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=SCHEMA)
        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()

    def __enter__(self) -> FeatureLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_features(path: str | Path) -> list[Features]:
    table = pq.read_table(str(path))
    out: list[Features] = []
    for r in table.to_pylist():
        def pt(x, y):
            return None if x is None or y is None else (x, y)
        out.append(Features(
            t=r["t"], d=r["d"], conf=r["conf"], n_persons=r["n_persons"] or 0,
            v_radial=r["v_radial"], pinch=r["pinch"], osc=r["osc"], tube_disp=r["tube_disp"],
            wrist_l=pt(r["wrist_l_x"], r["wrist_l_y"]), wrist_r=pt(r["wrist_r_x"], r["wrist_r_y"]),
            tube_pt=pt(r["tube_x"], r["tube_y"]), face_w=r["face_w"],
            d_tape=r.get("d_tape"), tape_pt=pt(r.get("tape_x"), r.get("tape_y")),
        ))
    return out


def read_states(path: str | Path) -> list[str | None]:
    return pq.read_table(str(path), columns=["state"]).column("state").to_pylist()
