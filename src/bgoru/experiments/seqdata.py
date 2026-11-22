"""Ham SoccerNet sekanslarını (MOT gt.txt + gameinfo) kare kare okuma yardımcıları."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bgoru.config import DATA

SN = DATA / "soccernet"


def load_meta(seq: str) -> dict:
    m = json.loads((SN / "raw" / seq / "meta.json").read_text(encoding="utf-8"))
    m["tracks"] = {int(k): v for k, v in m["tracks"].items()}
    return m


def load_gt(seq: str) -> dict[int, list]:
    rows = defaultdict(list)
    for line in (SN / "raw" / seq / "gt.txt").read_text().strip().splitlines():
        p = line.split(",")
        f, tid = int(p[0]), int(p[1])
        x, y, w, h = map(float, p[2:6])
        rows[f].append((tid, np.array([x, y, x + w, y + h], np.float32)))
    return rows


def frame_objects(meta: dict, rows: dict, f: int):
    """Dönüş: players(list of (tid, box, team 0=left/1=right, is_gk)), refs(list box), ball(box|None)"""
    players, refs, ball = [], [], None
    for tid, box in rows.get(f, []):
        t = meta["tracks"].get(tid)
        if t is None:
            continue
        if t["cls"] == 0 and t["team"] in ("left", "right"):
            players.append((tid, box, 0 if t["team"] == "left" else 1, t["goalkeeper"]))
        elif t["cls"] == 1:
            refs.append(box)
        elif t["cls"] == 2:
            ball = box
    return players, refs, ball


def dense_sequences() -> list[str]:
    d = SN / "dense"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def dense_frames(seq: str) -> list[tuple[int, Path]]:
    return [(int(p.stem), p) for p in sorted((SN / "dense" / seq / "img1").glob("*.jpg"))]


def sparse_test_frames() -> dict[str, list[tuple[int, Path]]]:
    out = defaultdict(list)
    for p in sorted((SN / "images" / "test").glob("*.jpg")):
        seq, f = p.stem.rsplit("_", 1)
        out[seq].append((int(f), p))
    return dict(out)
