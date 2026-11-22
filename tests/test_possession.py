"""Top sahipliği ve olay motoru: pas, araya girme, histerezis, pas/dk."""
from __future__ import annotations

import numpy as np
import pytest

from bgoru.possession import PlayerObs, PossessionEngine, ball_player_dist


def box(x, y, h=100.0, w=40.0):
    """Ayak noktası (x, y) olan oyuncu kutusu."""
    return np.array([x - w / 2, y - h, x + w / 2, y], np.float32)


def feet_ball(x, y):
    return np.array([x, y - 5.0])


# ------------------------------------------------------------------ top sahipliği / olaylar
def run_script(script, fps=25.0):
    """script: kare listesi; her kare (oyuncular[(tid, team, x, y)], top (x,y) | None)"""
    e = PossessionEngine(fps)
    for f, (players, ball) in enumerate(script):
        obs = [PlayerObs(t, tm, box(x, y), None) for t, tm, x, y in players]
        e.update(f, obs, None if ball is None else np.asarray(ball, float))
    e.finalize()
    return e


def test_ball_player_distance_is_scale_free():
    b = box(100, 500, h=100)
    assert ball_player_dist(np.array([100, 495]), b) == 0.0
    near = ball_player_dist(np.array([100 + 20 + 30, 495]), b)        # kutudan 30 px uzak, boy 100
    far_small = ball_player_dist(np.array([100 + 10 + 15, 495]), box(100, 500, h=50, w=20))  # yarı ölçek
    assert near == pytest.approx(far_small, rel=1e-6)


def test_pass_between_teammates():
    A, B = (1, 0, 100, 500), (2, 0, 600, 500)
    script = [([A, B], feet_ball(100, 500))] * 10                    # A topla
    script += [([A, B], (350, 450))] * 10                            # top havada
    script += [([A, B], feet_ball(600, 500))] * 10                   # B karşıladı
    e = run_script(script)
    types = [ev["type"] for ev in e.events]
    assert "PASS" in types
    ev = next(ev for ev in e.events if ev["type"] == "PASS")
    assert (ev["tid"], ev["to"], ev["team"]) == (1, 2, 0)
    assert e.snapshot()["possession_pct"][0] == pytest.approx(100.0)


def test_interception_by_opponent():
    A, C = (1, 0, 100, 500), (3, 1, 600, 500)
    script = [([A, C], feet_ball(100, 500))] * 10 + [([A, C], (350, 450))] * 10 + [([A, C], feet_ball(600, 500))] * 10
    e = run_script(script)
    types = {ev["type"] for ev in e.events}
    assert {"INTERCEPT", "PASS_FAIL"} <= types
    pct = e.snapshot()["possession_pct"]
    assert pct[0] > 0 and pct[1] > 0 and pct[0] + pct[1] == pytest.approx(100.0)


def test_hysteresis_ignores_single_frame_touch():
    A, C = (1, 0, 100, 500), (3, 1, 300, 500)
    script = [([A, C], feet_ball(100, 500))] * 10 + [([A, C], feet_ball(300, 500))] * 1 + \
             [([A, C], feet_ball(100, 500))] * 10
    e = run_script(script)
    assert not any(ev["type"] in ("INTERCEPT", "TACKLE", "PICKUP") for ev in e.events)


def test_pass_frequency_per_minute():
    A, B = (1, 0, 100, 500), (2, 0, 600, 500)
    script = []
    for k in range(6):  # 6 pas, her biri 1 sn
        src, dst = (A, B) if k % 2 == 0 else (B, A)
        script += [([A, B], feet_ball(src[2], 500))] * 12 + [([A, B], (350, 450))] * 6 + \
                  [([A, B], feet_ball(dst[2], 500))] * 7
    e = run_script(script)
    snap = e.snapshot()
    assert snap["passes"][0] >= 5
    assert snap["passes_per_min"][0] == pytest.approx(snap["passes"][0] / (snap["elapsed_s"] / 60), rel=0.05)
