#!/usr/bin/env bash
# Eksik eğitimleri tamamlar, ardından değerlendirme -> rapor zincirini çalıştırır (oturumdan bağımsız başlatılır).
cd "$(dirname "$0")/.."
echo "[$(date +%H:%M:%S)] eğitimler" 
bash experiments/train_all.sh
echo "[$(date +%H:%M:%S)] zincir"
bash experiments/final_chain.sh
echo "[$(date +%H:%M:%S)] TUM_ISLER_BITTI"
