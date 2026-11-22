"""Ek eğitim verisi deneyi: G1 (A1 + SoccerNet-GSR valid, oyun 2, 3, 5) ile A1'in aynı protokolle karşılaştırılması.

* Doğrulama ve test: top AP@0.5, top AP_m, oyuncu / hakem AP@0.5, mAP@[.5:.95] (eval_detector ile aynı fonksiyonlar)
* Sekans bazlı eşleştirilmiş bootstrap (1000): G1 − A1 (tohum 0) ve G1 − A1 tohum ortalaması
* Bağlam: A1'in iki tohumu arasındaki fark, aynı ayarlarla eğitim varyansının büyüklüğünü gösterir
* Adım eşleşmeli kontrol C1 (A1 verisi, 50 epoch ≈ G1'in gradyan adımı, çizelge oranları aynı):
  G1 − C1 = eşit adımda ek verinin etkisi, C1 − A1 = yalnızca daha uzun eğitimin etkisi

Test kümesi (oyun 7, 8, 11) ve doğrulama kümesi (oyun 4) değişmez; ek veri bu maçlarla kesişmez
(data/soccernet_gsr/manifests/leak_check.json).
Çıktı: results/detection/gsr_extra.csv, results/detection/gsr_extra.json

Kullanım: python -m bgoru.experiments.eval_gsr
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from bgoru.config import RESULTS, WEIGHTS
from bgoru.experiments.common import MODELS, ap, ap_range, load_split, predict_split, save_json
from bgoru.experiments.eval_detector import bootstrap, gt_cls_for
from bgoru.experiments.registry import log_experiment

RUNS = [("A1_ft1280", "A1 (tohum 0)"), ("A1_ft1280_s1", "A1 (tohum 1)"), ("G1_ft1280_gsr", "G1 = A1 + GSR (tohum 0)"),
        ("C1_ft1280_ep50", "C1 = A1, 50 epoch (adım eşleşmeli kontrol)")]


def ci(v: np.ndarray) -> dict:
    return {"mean": float(np.nanmean(v)), "lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5)),
            "p_le_0": float(np.mean(v <= 0))}


def main():
    t0 = time.time()
    val, test = load_split("val"), load_split("test")
    rows, preds, gts = [], {}, {}
    for mid, label in RUNS:
        if not (WEIGHTS / MODELS[mid][0]).exists():
            print(f"[skip] {mid}: ağırlık yok")
            continue
        pv, pt = predict_split(mid, "val", val), predict_split(mid, "test", test)
        g = {c: gt_cls_for(mid, c) for c in (0, 1, 2)}
        row = {"model_id": mid, "label": label,
               "val_ball_AP50": ap(val, pv, 2, 0.5, g[2]), "val_mAP50_95": float(np.mean([ap_range(val, pv, c, g[c]) for c in (0, 1, 2)])),
               "ball_AP50": ap(test, pt, 2, 0.5, g[2]), "ball_APc": ap(test, pt, 2, "center", g[2]),
               "player_AP50": ap(test, pt, 0, 0.5, g[0]), "referee_AP50": ap(test, pt, 1, 0.5, g[1]),
               "mAP50_95": float(np.mean([ap_range(test, pt, c, g[c]) for c in (0, 1, 2)]))}
        rows.append(row)
        preds[mid], gts[mid] = pt, g[2]
        print(f"[✓] {mid}: val top AP50={row['val_ball_AP50']:.4f}  test top AP50={row['ball_AP50']:.4f}")
    df = pd.DataFrame(rows)
    out = RESULTS / "detection"
    df.to_csv(out / "gsr_extra.csv", index=False)

    bs = bootstrap(test, preds, 2, gts, n_boot=1000)  # aynı yeniden örneklemeler -> eşleştirilmiş farklar
    diffs = {}
    if "G1_ft1280_gsr" in bs and "A1_ft1280" in bs:
        diffs["G1 - A1 (tohum 0)"] = ci(bs["G1_ft1280_gsr"] - bs["A1_ft1280"])
        if "A1_ft1280_s1" in bs:
            diffs["G1 - A1 tohum ortalaması"] = ci(bs["G1_ft1280_gsr"] - (bs["A1_ft1280"] + bs["A1_ft1280_s1"]) / 2)
    if "A1_ft1280_s1" in bs and "A1_ft1280" in bs:
        diffs["A1 tohum 1 - tohum 0 (eğitim varyansı)"] = ci(bs["A1_ft1280_s1"] - bs["A1_ft1280"])
    if "C1_ft1280_ep50" in bs:
        if "G1_ft1280_gsr" in bs:
            diffs["G1 - C1 (eşit adımda ek veri etkisi)"] = ci(bs["G1_ft1280_gsr"] - bs["C1_ft1280_ep50"])
        if "A1_ft1280" in bs:
            diffs["C1 - A1 (daha uzun eğitim etkisi)"] = ci(bs["C1_ft1280_ep50"] - bs["A1_ft1280"])
    res = {"rows": rows, "paired_differences_ball_AP50": diffs,
           "ball_AP50_bootstrap_CI95": {m: ci(v) for m, v in bs.items()},
           "train_images": {"A1": 1950, "G1": 4850, "C1": 1950},
           "train_games": {"A1": [6, 9], "G1": [2, 3, 5, 6, 9], "C1": [6, 9]},
           "train_iterations": {"A1": 488 * 20, "G1": 1213 * 20, "C1": 488 * 50},
           "note": "Aynı ayarlar (1280 px, SGD, seed 0, batch 4, nbs 64). A1 ve G1 20 epoch: yalnızca eğitim verisi farklı, "
                   "bu yüzden G1 yaklaşık 2,5 kat daha çok gradyan adımı görür. C1 = A1 verisi ile 50 epoch "
                   "(G1 ile eşit adım; ısınma 7,5 epoch ve son 25 epoch mozaiksiz, yani çizelge oranları A1/G1 ile aynı)."}
    save_json(res, out / "gsr_extra.json")
    log_experiment("eval_gsr_extra", "eval_detector", "A1 vs G1 (A1 + SoccerNet-GSR valid) vs C1 (adım kontrolü)",
                   {"runs": [r["model_id"] for r in rows]},
                   "test (resmî SoccerNet test, 49 sekans, her 25. kare)",
                   {r["model_id"]: {"ball_AP50": round(r["ball_AP50"], 4), "val_ball_AP50": round(r["val_ball_AP50"], 4)}
                    for r in rows} | {k: round(v["mean"], 4) for k, v in diffs.items()}, time.time() - t0)
    print(df.round(4).to_string(index=False))
    for k, v in diffs.items():
        print(f"{k}: {v['mean']:+.4f} [%95 GA {v['lo']:+.4f}; {v['hi']:+.4f}]  P(≤0)={v['p_le_0']:.3f}")


if __name__ == "__main__":
    main()
