"""Deney kayıt defteri: her deneyin kimliği, modeli, yapılandırması, veri ayrımı,
metrikleri ve süresi experiments/registry.csv (+ JSON) dosyasına yazılır."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from bgoru.config import EXPERIMENTS

FIELDS = ["exp_id", "task", "model", "config", "data_split", "metrics", "duration_s", "timestamp", "notes"]


def log_experiment(exp_id: str, task: str, model: str, config: dict | str, data_split: str,
                   metrics: dict, duration_s: float, notes: str = "") -> None:
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS / "registry.csv"
    rows = []
    if path.exists():
        with open(path, encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if not (r["exp_id"] == exp_id and r["task"] == task)]
    rows.append({
        "exp_id": exp_id, "task": task, "model": model,
        "config": config if isinstance(config, str) else json.dumps(config, ensure_ascii=False),
        "data_split": data_split, "metrics": json.dumps(metrics, ensure_ascii=False),
        "duration_s": f"{duration_s:.1f}", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": notes,
    })
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(EXPERIMENTS / "registry.json", "w", encoding="utf-8") as fh:
        json.dump([{**r, "metrics": json.loads(r["metrics"])} for r in rows], fh, ensure_ascii=False, indent=1)
