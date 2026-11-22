"""Proje genelinde kullanılan yollar ve yardımcılar (sabit kişisel yol yoktur)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
WEIGHTS = ROOT / "weights"
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"

CLASS_NAMES = ["player", "referee", "ball"]

# Son sistem dedektörü: doğrulama top AP@0.5'e göre seçildi (results/detection/detection_results.csv).
# G1 = A1 ayarları + SoccerNet-GSR ek eğitim verisi (oyun 2, 3, 5); bkz. results/detection/gsr_extra.json
FINAL_DETECTOR = "G1_ft1280_gsr"
FINAL_DET_WEIGHTS = f"bgoru_{FINAL_DETECTOR}.pt"
# Doğrulamada F1'i en büyükleyen güven eşikleri (G1: top 0,203, oyuncu 0,321), 0,05 adımına yuvarlanmış
FINAL_CONF_BALL = 0.20
FINAL_CONF_PLAYER = 0.30
PLAYER, REFEREE, BALL = 0, 1, 2
SEED = 0


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def dataset_yaml(name: str, out_dir: Path) -> Path:
    """configs/data/<name>.yaml içindeki göreli yolları mutlak yollara çevirip
    ultralytics'in okuyacağı geçici bir veri yaml'ı üretir."""
    cfg = load_yaml(CONFIGS / "data" / f"{name}.yaml")
    out = {"path": str(resolve(cfg["path"])), "names": {i: n for i, n in enumerate(CLASS_NAMES)}}
    for k in ("train", "val", "test"):
        if k in cfg:
            out[k] = cfg[k]  # 'path'e göre göreli
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"data_{name}.yaml"
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(out, fh, allow_unicode=True)
    return p
