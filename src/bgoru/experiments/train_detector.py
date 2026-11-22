"""Dedektör eğitimi (E2, E3, A1, A2).

Kullanım:
    python -m bgoru.experiments.train_detector --config configs/exp/E3_ft1280_cp.yaml

Model seçimi yalnızca DOĞRULAMA kümesi üzerinden yapılır (Ultralytics best.pt =
doğrulama fitness'ı en yüksek epoch). Test kümesi eğitim sırasında hiç kullanılmaz.
"""
from __future__ import annotations

import argparse
import shutil
import time

import pandas as pd

from bgoru.config import EXPERIMENTS, WEIGHTS, dataset_yaml, load_yaml
from bgoru.experiments.registry import log_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="yapılandırmayı geçersiz kıl (hızlı deneme)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.epochs:
        cfg["epochs"] = args.epochs

    from ultralytics import YOLO

    run_dir = EXPERIMENTS / "runs"
    data_yaml = dataset_yaml(cfg["data"], run_dir / "_data")
    model = YOLO(str(WEIGHTS / cfg["model"]) if (WEIGHTS / cfg["model"]).exists() else cfg["model"])
    t0 = time.time()
    model.train(
        data=str(data_yaml), imgsz=cfg["imgsz"], epochs=cfg["epochs"], batch=cfg["batch"], nbs=cfg["nbs"],
        optimizer=cfg["optimizer"], lr0=cfg["lr0"], seed=cfg["seed"], deterministic=cfg["deterministic"],
        cache=cfg.get("cache", False), workers=cfg.get("workers", 4), patience=cfg.get("patience", 100),
        project=str(run_dir), name=cfg["id"], exist_ok=True, plots=True, verbose=True, amp=True,
        # isteğe bağlı çizelge alanları (yalnızca adım eşleşmeli kontrol kullanır; yoksa Ultralytics varsayılanı)
        **{k: cfg[k] for k in ("warmup_epochs", "close_mosaic") if k in cfg},
    )
    dur = time.time() - t0
    out = run_dir / cfg["id"]
    best = out / "weights" / "best.pt"
    WEIGHTS.mkdir(exist_ok=True)
    shutil.copy2(best, WEIGHTS / f"bgoru_{cfg['id']}.pt")

    res = pd.read_csv(out / "results.csv")
    res.columns = [c.strip() for c in res.columns]
    fit = 0.1 * res["metrics/mAP50(B)"] + 0.9 * res["metrics/mAP50-95(B)"]
    b = res.loc[fit.idxmax()]
    metrics = {"best_epoch": int(b["epoch"]), "val_mAP50": round(float(b["metrics/mAP50(B)"]), 4),
               "val_mAP50_95": round(float(b["metrics/mAP50-95(B)"]), 4),
               "val_P": round(float(b["metrics/precision(B)"]), 4), "val_R": round(float(b["metrics/recall(B)"]), 4)}
    log_experiment(cfg["id"], "train_detector", cfg["model"], cfg, "train/val (gameID bazlı)", metrics, dur,
                   cfg.get("description", ""))
    print(f"[✓] {cfg['id']} bitti: {metrics}  süre={dur / 60:.1f} dk")


if __name__ == "__main__":
    main()
