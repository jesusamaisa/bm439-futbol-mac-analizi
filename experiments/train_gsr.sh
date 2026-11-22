#!/usr/bin/env bash
# Ek eğitim verisi deneyi: A1 ile aynı ayarlar, eğitim kümesi = SoccerNet-Tracking train + SoccerNet-GSR valid
# (oyun 2, 3, 5). Doğrulama/test değişmez. Önce: python -m bgoru.data.prepare_gsr
# Sonra: python -m bgoru.experiments.eval_gsr
export PYTHONUTF8=1 PYTHONPATH=src
cfg=G1_ft1280_gsr
if [ -f "weights/bgoru_${cfg}.pt" ]; then echo "[skip] $cfg"; exit 0; fi
echo "[start] $cfg $(date)"
python -u -m bgoru.experiments.train_detector --config "configs/exp/${cfg}.yaml" > "experiments/logs/train_${cfg}.log" 2>&1 || echo "[fail] $cfg"
echo "[end] $cfg $(date)"
echo "[done] $(date)"
