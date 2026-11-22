#!/usr/bin/env bash
# Tüm dedektör deneylerini sırayla eğitir (proje kökünden çalıştırın).
export PYTHONUTF8=1 PYTHONPATH=src
for cfg in E3_ft1280_cp E2_ft640 A1_ft1280 A2_ft640_cp; do
  if [ -f "weights/bgoru_${cfg}.pt" ]; then echo "[skip] $cfg"; continue; fi
  python -u -m bgoru.experiments.train_detector --config "configs/exp/${cfg}.yaml" > "experiments/logs/train_${cfg}.log" 2>&1 || echo "[fail] $cfg"
done
