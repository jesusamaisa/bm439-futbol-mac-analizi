"""Takım ataması: iki formanın ayrılması ve yerel saha maskesinin yeşil formayı koruması."""
from __future__ import annotations

import numpy as np
import pytest

from bgoru.team import TeamAssigner


def synthetic_frame(colors, n_per_team=6, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((400, 900, 3), np.uint8)
    img[:] = (40, 140, 50)  # çim (BGR)
    boxes, gt = [], []
    for t, col in enumerate(colors):
        for k in range(n_per_team):
            x0 = 20 + (t * n_per_team + k) * 70
            y0 = 100 + int(rng.integers(0, 50))
            img[y0:y0 + 120, x0:x0 + 40] = col
            boxes.append([x0, y0, x0 + 40, y0 + 120]); gt.append(t)
    return img, np.array(boxes, np.float32), np.array(gt)


@pytest.mark.parametrize("method", ["proposed_local", "proposed", "baseline"])
def test_team_assignment_separates_two_kits(method):
    ta = TeamAssigner(method, vote=method.startswith("proposed"))
    pred = None
    for f in range(12):
        img, boxes, gt = synthetic_frame([(230, 230, 230), (40, 40, 180)], seed=f)
        pred, _, _ = ta.assign(img, boxes, np.arange(len(boxes)))
    acc = max(np.mean(pred == gt), np.mean(pred == 1 - gt))
    assert acc == 1.0


def test_local_mask_keeps_green_kit():
    """Yeşil formalı takım (doğrulama maçındaki durum) yerel saha maskesiyle ayrılabilmeli."""
    ta = TeamAssigner("proposed_local", vote=True)
    pred = None
    for f in range(12):
        img, boxes, gt = synthetic_frame([(235, 235, 235), (60, 170, 70)], seed=f)  # beyaz vs açık yeşil forma
        pred, _, _ = ta.assign(img, boxes, np.arange(len(boxes)))
    acc = max(np.mean(pred == gt), np.mean(pred == 1 - gt))
    assert acc == 1.0
