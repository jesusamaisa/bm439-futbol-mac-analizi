"""Oyuncu teknik profilleme (CIES Football Observatory yönteminden uyarlanmış) ve
karakter kartı özellik puanları.

1. Her oyuncu için aksiyon sıklığı (dakika başına) hesaplanır.
2. Sıklıklar, aynı kesitteki tüm oyuncuların ortalamasına bölünerek göreli hale
   getirilir (pas gibi sık aksiyonların baskınlığı önlenir).
3. Göreli sıklıklar aksiyon birim vektörleriyle ağırlıklandırılıp toplanır; vektörün
   yönü profil kategorisini, büyüklüğü uzmanlaşma derecesini verir.
4. Kart özellikleri (HIZ, PAS, SAVUNMA …) aynı kesitteki oyunculara göre yüzdelik
   (percentile, 0–99) olarak verilir; GENEL puan profile göre ağırlıklı ortalamadır.
"""
from __future__ import annotations

import math

import numpy as np

from bgoru.config import CONFIGS, load_yaml

PROFILE_CFG = load_yaml(CONFIGS / "profile_vectors.yaml")
ATTRS = ["HIZ", "HAREKET", "PAS", "ISABET", "SURME", "SAVUNMA", "HUCUM", "KONTROL"]
ATTR_TR = {"HIZ": "Hız", "HAREKET": "Hareket", "PAS": "Pas", "ISABET": "Pas İsabeti", "SURME": "Top Sürme",
           "SAVUNMA": "Savunma", "HUCUM": "Hücum", "KONTROL": "Top Kontrolü"}


def unit(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)])


def category_of(angle: float, mag: float, n_events: int, cfg=PROFILE_CFG) -> str:
    if n_events < cfg["min_events"]:
        return "Yetersiz veri"
    if mag < cfg["min_magnitude"]:
        return cfg["balanced_name"]
    a = angle % 360
    best = min(cfg["categories"], key=lambda c: min(abs(a - c["center"]), 360 - abs(a - c["center"])))
    return best["name"]


def percentile_scores(values: dict[int, float | None]) -> dict[int, int | None]:
    valid = {k: v for k, v in values.items() if v is not None}
    if not valid:
        return {k: None for k in values}
    arr = np.array(list(valid.values()), float)
    out = {}
    for k, v in values.items():
        if v is None:
            out[k] = None
        elif len(arr) == 1:
            out[k] = 50
        else:
            out[k] = int(round(99 * ((arr < v).sum() + 0.5 * ((arr == v).sum() - 1)) / (len(arr) - 1)))
    return out


def build_profiles(players: dict[int, dict], fps: float, cfg=PROFILE_CFG) -> dict[int, dict]:
    """players: tid -> {frames, team, events{type:count}, poss_frames, kin{distance_m,max_speed_kmh,...}}"""
    acts = cfg["actions"]
    elig = {t: p for t, p in players.items() if p["frames"] / fps / 60 >= cfg["min_minutes"]}
    if not elig:
        return {}
    mins = {t: p["frames"] / fps / 60 for t, p in elig.items()}
    rates = {t: {a: p["events"].get(a, 0) / mins[t] for a in acts} for t, p in elig.items()}
    mean = {a: np.mean([r[a] for r in rates.values()]) for a in acts}

    prof = {}
    for t, p in elig.items():
        rel = {a: (rates[t][a] / mean[a]) if mean[a] > 0 else 0.0 for a in acts}
        tot = sum(rel.values())
        v = sum((rel[a] * unit(acts[a]["angle"]) for a in acts), np.zeros(2)) / tot if tot > 0 else np.zeros(2)
        ang = math.degrees(math.atan2(v[1], v[0])) % 360
        n_ev = int(sum(p["events"].get(a, 0) for a in acts))
        prof[t] = {"vector": v.round(3).tolist(), "angle": round(ang, 1), "magnitude": round(float(np.linalg.norm(v)), 3),
                   "category": category_of(ang, float(np.linalg.norm(v)), n_ev, cfg), "n_events": n_ev,
                   "minutes": round(mins[t], 2), "offensive": round(float(-v[0]), 3), "centrality": round(float(v[1]), 3)}

    # --- kart özellikleri (ham değerler -> yüzdelik)
    ev = lambda t, k: elig[t]["events"].get(k, 0)
    raw = {a: {} for a in ATTRS}
    for t, p in elig.items():
        m = mins[t]
        kin = p.get("kin") or {}
        has_kin = kin.get("distance_m", 0) > 0
        raw["HIZ"][t] = kin.get("max_speed_kmh") if has_kin else None
        raw["HAREKET"][t] = kin.get("distance_m") / m if has_kin else None
        raw["PAS"][t] = ev(t, "PASS") / m
        att = ev(t, "PASS") + ev(t, "PASS_FAIL")
        raw["ISABET"][t] = ev(t, "PASS") / att if att else None
        raw["SURME"][t] = (ev(t, "CARRY") + 2 * ev(t, "DRIBBLE")) / m
        raw["SAVUNMA"][t] = (ev(t, "TACKLE") + ev(t, "INTERCEPT") + ev(t, "PICKUP")) / m
        raw["HUCUM"][t] = (2 * ev(t, "SHOT") + 1.5 * ev(t, "KEY_PASS") + ev(t, "CROSS") + ev(t, "DRIBBLE")) / m
        raw["KONTROL"][t] = p.get("poss_frames", 0) / fps / m
    pct = {a: percentile_scores(raw[a]) for a in ATTRS}
    weights = cfg["overall_weights"]
    for t in prof:
        scores = {a: pct[a][t] for a in ATTRS}
        w = weights.get(prof[t]["category"], weights["default"])
        num = sum(w[a] * s for a, s in scores.items() if s is not None)
        den = sum(w[a] for a, s in scores.items() if s is not None)
        prof[t]["attributes"] = scores
        prof[t]["raw"] = {a: (None if raw[a][t] is None else round(float(raw[a][t]), 3)) for a in ATTRS}
        prof[t]["overall"] = int(round(num / den)) if den else None
    return prof
