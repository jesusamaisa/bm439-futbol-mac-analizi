"""RQ1 — Dedektör karşılaştırması (aynı test protokolü).

Protokol
  1. Her model için doğrulama (val) ve test kümelerinde tahmin üretilir (önbelleğe alınır).
  2. Birden fazla girdi boyutu olan referans modeller (E1, R) için girdi boyutu
     **doğrulama** top AP@0.5 değerine göre seçilir.
  3. Top/oyuncu güven eşikleri **doğrulama** kümesinde F1'i en büyükleyen değerdir.
  4. Test: sınıf bazında AP@0.5, AP@[.5:.95]; seçilen eşikte P/R/F1; top boyutu kovalarına
     göre recall; sekans düzeyinde bootstrap (1000 örnek) %95 güven aralığı ve E3−E2 farkı.

Kullanım:  python -m bgoru.experiments.eval_detector [--models E2_ft640 E3_ft1280_cp ...]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from bgoru.config import RESULTS
from bgoru.experiments.common import (MODELS, ap, ap_from, ap_range, best_conf_f1, collect, load_split,
                                      predict_split, prf_at, save_json)
from bgoru.experiments.registry import log_experiment

GROUPS = {"E1": ["E1_coco640", "E1_coco1280"], "R": ["R_hmzbo640", "R_hmzbo1280"]}
LABEL = {"E1": "E1 Temel (COCO YOLOv8s, ince ayarsız)", "E2_ft640": "E2 Alternatif (ince ayar 640)",
         "E3_ft1280_cp": "E3 Önerilen (1280 + kopyala-yapıştır)", "A1_ft1280": "A1 Ablation (1280, artırımsız)",
         "A2_ft640_cp": "A2 Ablation (640 + kopyala-yapıştır)", "R": "R Referans (Hmzbo YOLOv8L, taktik kamera)",
         "G1_ft1280_gsr": "G1 Son sistem (A1 + SoccerNet-GSR ek eğitim verisi)",
         "C1_ft1280_ep50": "C1 Kontrol (A1 verisi, G1 ile eşit adım: 50 epoch)"}


def gt_cls_for(model_id, c):
    # COCO 'person' hakemi ayırt edemez: taban modelin oyuncu sınıfı oyuncu+hakem GT'sine karşı ölçülür
    if model_id.startswith("E1") and c == 0:
        return [0, 1]
    return [c]


def size_bucket_recall(items, preds, conf, iou_thr=0.5):
    out = {}
    for name, lo, hi in (("<10px", 0, 10), ("10-20px", 10, 20), (">=20px", 20, 1e9)):
        tp = n = 0
        for it, p in zip(items, preds):
            g = it["boxes"][it["cls"] == 2]
            if not len(g):
                continue
            sz = np.maximum(g[:, 2] - g[:, 0], g[:, 3] - g[:, 1])
            sel = (sz >= lo) & (sz < hi)
            if not sel.any():
                continue
            from bgoru.experiments.common import match_image
            pp = p[(p[:, 5] == 2) & (p[:, 4] >= conf)]
            _, used = match_image(pp, g, iou_thr)
            tp += int(used[sel].sum()); n += int(sel.sum())
        out[name] = {"recall": tp / max(n, 1), "n": n}
    return out


def bootstrap(items, preds_by_model, cls, gt_map, n_boot=1000, seed=0):
    seqs = sorted({it["seq"] for it in items})
    idx_of = {s: [i for i, it in enumerate(items) if it["seq"] == s] for s in seqs}
    # sekans başına önceden (skor, tp) topla -> hızlı yeniden örnekleme
    per = {}
    for m, preds in preds_by_model.items():
        per[m] = {}
        for s in seqs:
            sc, tp, _, n = collect(items, preds, cls, 0.5, gt_map[m], idx_of[s])
            per[m][s] = (sc, tp, n)
    rng = np.random.default_rng(seed)
    res = {m: [] for m in preds_by_model}
    for _ in range(n_boot):
        samp = rng.choice(seqs, len(seqs), replace=True)
        for m in preds_by_model:
            sc = np.concatenate([per[m][s][0] for s in samp]); tp = np.concatenate([per[m][s][1] for s in samp])
            n = sum(per[m][s][2] for s in samp)
            res[m].append(ap_from(sc, tp, n))
    return {m: np.array(v) for m, v in res.items()}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--models", nargs="*", default=["E1", "E2_ft640", "E3_ft1280_cp", "A1_ft1280", "A2_ft640_cp", "R",
                                                        "G1_ft1280_gsr", "C1_ft1280_ep50"])
    ap_.add_argument("--n-boot", type=int, default=1000)
    args = ap_.parse_args()
    val, test = load_split("val"), load_split("test")
    print(f"[i] val={len(val)} görüntü, test={len(test)} görüntü, test sekans={len({i['seq'] for i in test})}")

    rows, chosen, test_preds, gt_maps = [], {}, {}, {}
    for key in args.models:
        cands = GROUPS.get(key, [key])
        best, best_ap = None, -1
        for mid in cands:  # girdi boyutu seçimi: yalnızca doğrulama
            t0 = time.time()
            pv = predict_split(mid, "val", val)
            a = ap(val, pv, 2, 0.5, gt_cls_for(mid, 2))
            print(f"  {mid}: val ball AP50={a:.4f} ({time.time() - t0:.0f}s)")
            if a > best_ap:
                best, best_ap = mid, a
        mid = best
        chosen[key] = mid
        pv = predict_split(mid, "val", val)
        thr = {c: best_conf_f1(val, pv, c, 0.5, gt_cls_for(mid, c)) for c in (0, 2)}
        t0 = time.time()
        pt = predict_split(mid, "test", test)
        test_preds[key] = pt
        gt_maps[key] = gt_cls_for(mid, 2)
        row = {"config": key, "model_id": mid, "label": LABEL.get(key, key)}
        for c, nm in ((2, "ball"), (0, "player"), (1, "referee")):
            if mid.startswith("E1") and c == 1:
                row[f"{nm}_AP50"] = None; row[f"{nm}_AP50_95"] = None
                continue
            row[f"{nm}_AP50"] = ap(test, pt, c, 0.5, gt_cls_for(mid, c))
            row[f"{nm}_AP50_95"] = ap_range(test, pt, c, gt_cls_for(mid, c))
        # yardımcı metrik: merkez uzaklığı <= 2 top çapı (etiket ötelemesine dayanıklı), tüm modellere aynı
        row["ball_APc"] = ap(test, pt, 2, "center", gt_cls_for(mid, 2))
        cls_eval = [0, 2] if mid.startswith("E1") else [0, 1, 2]
        row["mAP50"] = float(np.mean([row[f"{n}_AP50"] for c, n in ((0, "player"), (1, "referee"), (2, "ball")) if c in cls_eval]))
        row["mAP50_95"] = float(np.mean([row[f"{n}_AP50_95"] for c, n in ((0, "player"), (1, "referee"), (2, "ball")) if c in cls_eval]))
        for c, nm in ((2, "ball"), (0, "player")):
            m = prf_at(test, pt, c, thr[c], 0.5, gt_cls_for(mid, c))
            row[f"{nm}_conf_thr"] = thr[c]
            row.update({f"{nm}_P": m["P"], f"{nm}_R": m["R"], f"{nm}_F1": m["F1"]})
        sb = size_bucket_recall(test, pt, thr[2])
        for k, v in sb.items():
            row[f"ball_R_{k}"] = v["recall"]
        row["val_ball_AP50"] = ap(val, pv, 2, 0.5, gt_cls_for(mid, 2))
        rows.append(row)
        print(f"[✓] {key} ({mid}): test ball AP50={row['ball_AP50']:.4f} player AP50={row['player_AP50']:.4f}")
        log_experiment(f"eval_{key}", "eval_detector", mid, {"imgsz": MODELS[mid][2], "conf_thr": thr},
                       "test (resmî SoccerNet test, 49 sekans, her 25. kare)",
                       {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                        if k not in ("config", "model_id", "label")}, time.time() - t0)

    df = pd.DataFrame(rows)
    out = RESULTS / "detection"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "detection_results.csv", index=False)

    # bootstrap güven aralıkları (top AP50, sekans düzeyinde yeniden örnekleme)
    bs = bootstrap(test, test_preds, 2, gt_maps, n_boot=args.n_boot)
    ci = {m: {"mean": float(np.nanmean(v)), "lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5))}
          for m, v in bs.items()}
    diffs = {}
    for a, b in (("E3_ft1280_cp", "E2_ft640"), ("E3_ft1280_cp", "A1_ft1280"), ("E3_ft1280_cp", "A2_ft640_cp"),
                 ("E2_ft640", "E1"), ("A1_ft1280", "E2_ft640"),
                 # ek eğitim verisi: toplam etki, eşit adımda veri etkisi, yalnızca uzun eğitimin etkisi
                 ("G1_ft1280_gsr", "A1_ft1280"), ("G1_ft1280_gsr", "C1_ft1280_ep50"), ("C1_ft1280_ep50", "A1_ft1280"),
                 ("G1_ft1280_gsr", "E3_ft1280_cp")):
        if a in bs and b in bs:
            d = bs[a] - bs[b]
            diffs[f"{a} - {b}"] = {"mean": float(np.nanmean(d)), "lo": float(np.nanpercentile(d, 2.5)),
                                   "hi": float(np.nanpercentile(d, 97.5)), "p_le_0": float(np.mean(d <= 0))}
    save_json({"chosen_input_size": chosen, "ball_AP50_bootstrap_CI95": ci, "paired_differences": diffs},
              out / "detection_bootstrap.json")
    print(df[["config", "ball_AP50", "ball_APc", "ball_R", "player_AP50", "referee_AP50", "mAP50", "mAP50_95"]].round(4).to_string())
    print(diffs)


if __name__ == "__main__":
    main()
