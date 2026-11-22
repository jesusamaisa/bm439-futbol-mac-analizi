"""Zorlu koşul (dayanıklılık) testi.

Test görüntülerine deterministik bozulmalar uygulanır ve modeller aynı protokolle ölçülür:
  * night  : gece/düşük ışık benzetimi — gama 2.0 karartma, %45 parlaklık, mavi kayma, Gauss gürültüsü (σ=6)
  * blur   : 15 px yatay hareket bulanıklığı (hızlı kamera kaydırma)
  * jpeg   : JPEG kalite 15 (düşük bant genişlikli yayın)
  * small  : top boyutu kovaları (ayrıca eval_detector'da) — burada ek olarak %50 küçültme (uzak kamera)
Güven eşikleri doğrulamada seçilen eşiklerdir (detection_results.csv), testte yeniden seçilmez.

Kullanım: python -m bgoru.experiments.robustness
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd

from bgoru.config import RESULTS
from bgoru.experiments.common import ap, load_split, predict_split, prf_at
from bgoru.experiments.registry import log_experiment


def night(img, i):
    rng = np.random.default_rng(i)
    f = img.astype(np.float32) / 255.0
    f = np.power(f, 2.0) * 0.45
    f[..., 0] *= 1.15  # mavi kayma (BGR)
    f = f * 255 + rng.normal(0, 6, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def blur(img, i):
    k = np.zeros((15, 15), np.float32)
    k[7, :] = 1 / 15
    return cv2.filter2D(img, -1, k)


def jpeg(img, i):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 15])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def downscale(img, i):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(small, 0, h - h // 2, 0, w - w // 2, cv2.BORDER_CONSTANT, value=(40, 110, 50))
    return canvas


CONDITIONS = {"clean": None, "night": night, "blur": blur, "jpeg": jpeg, "far_half": downscale}


def scale_items(items, s=0.5):
    out = []
    for it in items:
        out.append({**it, "boxes": it["boxes"] * s})
    return out


def main(models=("E1_coco1280", "E2_ft640", "E3_ft1280_cp", "A1_ft1280", "A2_ft640_cp", "G1_ft1280_gsr")):
    test = load_split("test")
    det = pd.read_csv(RESULTS / "detection" / "detection_results.csv")
    chosen = dict(zip(det["model_id"], det.to_dict("records")))
    rows = []
    for mid in models:
        if mid not in chosen:
            # E1 için doğrulamada seçilen girdi boyutu kullanılır
            alt = [k for k in chosen if k.startswith(mid.split("_")[0])]
            if not alt:
                continue
            mid = alt[0]
        r = chosen[mid]
        for cond, fn in CONDITIONS.items():
            t0 = time.time()
            items = scale_items(test) if cond == "far_half" else test
            preds = predict_split(mid, "test", test, transform=fn, tag="" if fn is None else f"_{cond}")
            gt_p = [0, 1] if mid.startswith("E1") else [0]
            row = {"model": mid, "condition": cond,
                   "ball_AP50": ap(items, preds, 2, 0.5), "player_AP50": ap(items, preds, 0, 0.5, gt_p),
                   **{f"ball_{k}": v for k, v in prf_at(items, preds, 2, r["ball_conf_thr"]).items() if k in ("P", "R", "F1")}}
            rows.append(row)
            print(f"  {mid:14s} {cond:9s} ball AP50={row['ball_AP50']:.4f} R={row['ball_R']:.3f} "
                  f"player AP50={row['player_AP50']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            log_experiment(f"rob_{mid}_{cond}", "robustness", mid, {"condition": cond}, "test (bozulmuş)",
                           {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}, time.time() - t0)
    df = pd.DataFrame(rows)
    out = RESULTS / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "robustness_results.csv", index=False)
    print(df.pivot(index="model", columns="condition", values="ball_AP50").round(4).to_string())


if __name__ == "__main__":
    main()
