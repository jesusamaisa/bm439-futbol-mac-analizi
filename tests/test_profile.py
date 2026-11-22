"""Profilleme: CIES kategori sektörleri, yüzdelik puanlar ve profil yönleri."""
from __future__ import annotations

from bgoru.profiling import build_profiles, category_of, percentile_scores


def test_category_sectors():
    assert category_of(180, 0.8, 10) == "Bitirici Forvet"
    assert category_of(0, 0.8, 10) == "Top Kesici / Ön Libero"
    assert category_of(270, 0.8, 10) == "Orta Yapan Kanat"
    assert category_of(45, 0.05, 10).startswith("Dengeli")
    assert category_of(45, 0.8, 1) == "Yetersiz veri"


def test_percentiles_and_profiles():
    pc = percentile_scores({1: 1.0, 2: 2.0, 3: 3.0, 4: None})
    assert pc[1] == 0 and pc[3] == 99 and pc[4] is None
    players = {
        1: {"frames": 1500, "team": 0, "events": {"SHOT": 4, "DRIBBLE": 3, "KEY_PASS": 2}, "poss_frames": 100,
            "kin": {"distance_m": 300, "max_speed_kmh": 30}},
        2: {"frames": 1500, "team": 1, "events": {"TACKLE": 4, "INTERCEPT": 4, "PASS": 6}, "poss_frames": 80,
            "kin": {"distance_m": 250, "max_speed_kmh": 25}},
        3: {"frames": 1500, "team": 0, "events": {"PASS": 6, "PICKUP": 1}, "poss_frames": 60,
            "kin": {"distance_m": 280, "max_speed_kmh": 27}},
    }
    pr = build_profiles(players, fps=25)
    assert pr[1]["offensive"] > 0 > pr[2]["offensive"]       # hücumcu sola, savunmacı sağa
    assert pr[1]["attributes"]["HUCUM"] == 99
    assert 0 <= pr[2]["overall"] <= 99
