#!/usr/bin/env bash
# Tohum tekrarı: A1 ve E3'ü seed 1 ile yeniden eğitir (sırayla, proje kökünden). Varyans tahmini içindir;
# son sistem dedektörü (A1, seed 0) ve aşağı akış sonuçları değişmez.
export PYTHONUTF8=1 PYTHONPATH=src
for cfg in A1_ft1280_s1 E3_ft1280_cp_s1; do
  if [ -f "weights/bgoru_${cfg}.pt" ]; then echo "[skip] $cfg"; continue; fi
  echo "[start] $cfg $(date)"
  python -u -m bgoru.experiments.train_detector --config "configs/exp/${cfg}.yaml" > "experiments/logs/train_${cfg}.log" 2>&1 || echo "[fail] $cfg"
  echo "[end] $cfg $(date)"
done
echo "[done] $(date)"
