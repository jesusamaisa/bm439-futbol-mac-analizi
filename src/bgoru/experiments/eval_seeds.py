"""Tohum tekrarı (eğitim varyansı): E3 ve A1'in seed 0 ve seed 1 eğitimleri aynı test protokolüyle.

* Her eğitim için doğrulama ve test top AP@0.5, AP_m, oyuncu AP@0.5, mAP@[.5:.95] (eval_detector ile aynı fonksiyonlar)
* Yapılandırma başına ortalama ± standart sapma (n = tohum sayısı, ddof = 1)
* Sekans bazlı eşleştirilmiş bootstrap (1000): her tohum için E3−A1 ve tohum ortalamalarının farkı

Son sistem dedektörü (A1, seed 0) ve aşağı akış sonuçları değişmez; tekrar eğitimleri yalnızca varyans tahmini içindir.
Çıktı: results/detection/seed_repeats.csv, results/detection/seed_repeats.json

Kullanım: python -m bgoru.experiments.eval_seeds
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from bgoru.config import RESULTS, WEIGHTS
from bgoru.experiments.common import MODELS, ap, ap_range, load_split, predict_split, save_json
from bgoru.experiments.eval_detector import bootstrap, gt_cls_for
from bgoru.experiments.registry import log_experiment

PAIRS = {"E3_ft1280_cp": ["E3_ft1280_cp", "E3_ft1280_cp_s1"], "A1_ft1280": ["A1_ft1280", "A1_ft1280_s1"]}


def ci(v: np.ndarray) -> dict:
    return {"mean": float(np.nanmean(v)), "lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5)),
            "p_le_0": float(np.mean(v <= 0))}


def main():
    t0 = time.time()
    val, test = load_split("val"), load_split("test")
    rows, preds, gts = [], {}, {}
    for cfg, mids in PAIRS.items():
        for seed, mid in enumerate(mids):
            if not (WEIGHTS / MODELS[mid][0]).exists():
                print(f"[skip] {mid}: ağırlık yok")
                continue
            pv, pt = predict_split(mid, "val", val), predict_split(mid, "test", test)
            g = {c: gt_cls_for(mid, c) for c in (0, 1, 2)}
            row = {"config": cfg, "seed": seed, "model_id": mid, "val_ball_AP50": ap(val, pv, 2, 0.5, g[2]),
                   "ball_AP50": ap(test, pt, 2, 0.5, g[2]), "ball_APc": ap(test, pt, 2, "center", g[2]),
                   "player_AP50": ap(test, pt, 0, 0.5, g[0]),
                   "mAP50_95": float(np.mean([ap_range(test, pt, c, g[c]) for c in (0, 1, 2)]))}
            rows.append(row)
            preds[mid], gts[mid] = pt, g[2]
            print(f"[✓] {mid}: val top AP50={row['val_ball_AP50']:.4f}  test top AP50={row['ball_AP50']:.4f}")
    df = pd.DataFrame(rows)
    out = RESULTS / "detection"
    df.to_csv(out / "seed_repeats.csv", index=False)

    summ = {}
    for cfg, g in df.groupby("config"):
        summ[cfg] = {"n_seeds": int(len(g))}
        for m in ("ball_AP50", "ball_APc", "val_ball_AP50", "player_AP50", "mAP50_95"):
            summ[cfg][m] = {"mean": float(g[m].mean()), "std": float(g[m].std(ddof=1)) if len(g) > 1 else None,
                            "values": [float(x) for x in g.sort_values("seed")[m]]}
    # aynı yeniden örneklemelerle (tek çağrı) tüm modeller -> eşleştirilmiş farklar
    bs = bootstrap(test, preds, 2, gts, n_boot=1000)
    diffs = {}
    for s in (0, 1):
        e, a = PAIRS["E3_ft1280_cp"][s], PAIRS["A1_ft1280"][s]
        if e in bs and a in bs:
            diffs[f"seed{s}: E3 - A1"] = ci(bs[e] - bs[a])
    if all(m in bs for ms in PAIRS.values() for m in ms):
        me = np.mean([bs[m] for m in PAIRS["E3_ft1280_cp"]], 0)
        ma = np.mean([bs[m] for m in PAIRS["A1_ft1280"]], 0)
        diffs["tohum ortalaması: E3 - A1"] = ci(me - ma)
    res = {"summary": summ, "paired_differences": diffs,
           "note": "Kopyala-yapıştır verisi (seed 0) sabittir; yalnızca eğitim tohumu değişir."}
    save_json(res, out / "seed_repeats.json")
    log_experiment("eval_seed_repeats", "eval_detector", "E3/A1 x seed {0,1}", {"pairs": PAIRS},
                   "test (resmî SoccerNet test, 49 sekans, her 25. kare)",
                   {k: {m: round(v[m]["mean"], 4) for m in ("ball_AP50", "val_ball_AP50")} for k, v in summ.items()}
                   | {k: round(v["mean"], 4) for k, v in diffs.items()}, time.time() - t0)
    print(df.round(4).to_string(index=False))
    print(diffs)


if __name__ == "__main__":
    main()
