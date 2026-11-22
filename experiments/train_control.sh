#!/usr/bin/env bash
# Adım eşleşmeli kontrol: C1 = A1 verisi (1950 kare), 50 epoch -> G1 (4850 kare x 20 epoch) ile eşit gradyan adımı.
# G1 - C1 farkı, eşit eğitim süresinde ek verinin etkisini; C1 - A1 farkı yalnızca daha uzun eğitimin etkisini ölçer.
# Eğitim bitince karşılaştırma (eval_gsr) kendiliğinden çalışır. Proje kökünden: bash experiments/train_control.sh
export PYTHONUTF8=1 PYTHONPATH=src
cfg=C1_ft1280_ep50
if [ -f "weights/bgoru_${cfg}.pt" ]; then
  echo "[skip] $cfg"
else
  echo "[start] $cfg $(date)"
  python -u -m bgoru.experiments.train_detector --config "configs/exp/${cfg}.yaml" > "experiments/logs/train_${cfg}.log" 2>&1 || echo "[fail] $cfg"
  echo "[end] $cfg $(date)"
fi
rm -f yolo11n.pt  # Ultralytics AMP denetiminin indirdiği model
if [ -f "weights/bgoru_${cfg}.pt" ]; then
  echo "[eval] $(date)"
  python -m bgoru.experiments.eval_gsr > experiments/logs/eval_gsr.log 2>&1 && echo "[eval-ok]" || echo "[eval-fail]"
fi
echo "[done] $(date)"
