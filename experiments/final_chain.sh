#!/usr/bin/env bash
# Tüm eğitimler bittikten sonra değerlendirme -> rapor zinciri (proje kökünden çalıştırın).
set -u
export PYTHONUTF8=1 PYTHONPATH=src
L=experiments/logs
step() { echo "[$(date +%H:%M:%S)] $*"; }

step "eval_detector (tüm modeller, 1000 bootstrap)"
python -u -m bgoru.experiments.eval_detector --n-boot 1000 > $L/eval_detector.log 2>&1 || step "HATA eval_detector"
step "robustness (eksik modeller)"
python -u -m bgoru.experiments.robustness > $L/robustness.log 2>&1 || step "HATA robustness"
step "benchmark (GPU yalnız bu işte)"
python -u -m bgoru.experiments.benchmark > $L/benchmark.log 2>&1 || step "HATA benchmark"
step "failure_cases"
python -u -m bgoru.experiments.failure_cases > $L/failure_cases.log 2>&1 || step "HATA failure_cases"
step "make_demo"
python -u -m bgoru.experiments.make_demo > $L/make_demo.log 2>&1 || step "HATA make_demo"
step "make_figures"
python -u -m bgoru.experiments.make_figures > $L/make_figures.log 2>&1 || step "HATA make_figures"
step "make_showcase"
python -u -m bgoru.experiments.make_showcase > $L/make_showcase.log 2>&1 || step "HATA make_showcase"
step "cli env"
python -m bgoru.cli env > /dev/null 2>&1
step "rapor"
python report/build_report.py > $L/report.log 2>&1 || step "HATA rapor"
powershell -NoProfile -ExecutionPolicy Bypass -File report/to_pdf.ps1 >> $L/report.log 2>&1 || step "HATA pdf"
python report/update_readme.py >> $L/report.log 2>&1 || step "HATA readme"
step "BİTTİ"
