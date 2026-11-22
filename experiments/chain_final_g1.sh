#!/usr/bin/env bash
# Son sistem dedektörü A1 -> G1 (config.FINAL_DETECTOR) geçişinden sonra aşağı akış çıktılarını yeniden üretir.
# Önce: python -m bgoru.experiments.eval_detector (G1 ve C1 dahil). Hata analizi E3 üzerindedir, değişmez.
# Rapor bu betikten sonra: python report/build_report.py && report/to_pdf.ps1 && python report/update_readme.py
set -u
export PYTHONUTF8=1 PYTHONPATH=src
L=experiments/logs
step() { echo "[$(date +%H:%M:%S)] $*"; }

step "robustness (G1 eklendi; diğer modeller önbellekten)"
python -u -m bgoru.experiments.robustness > $L/robustness.log 2>&1 || step "HATA robustness"
step "benchmark (GPU yalnız bu işte)"
python -u -m bgoru.experiments.benchmark > $L/benchmark.log 2>&1 || step "HATA benchmark"
step "eval_possession V7_G1_final"
python -u -m bgoru.experiments.eval_possession --variants V7_G1_final > $L/eval_possession_g1.log 2>&1 || step "HATA eval_possession"
step "make_demo"
python -u -m bgoru.experiments.make_demo > $L/make_demo.log 2>&1 || step "HATA make_demo"
step "make_figures"
python -u -m bgoru.experiments.make_figures > $L/make_figures.log 2>&1 || step "HATA make_figures"
step "make_showcase"
python -u -m bgoru.experiments.make_showcase > $L/make_showcase.log 2>&1 || step "HATA make_showcase"
step "make_narration --tts"
python -u -m bgoru.experiments.make_narration --tts > $L/make_narration.log 2>&1 || step "HATA make_narration"
rm -f yolo11n.pt
step "BİTTİ"
