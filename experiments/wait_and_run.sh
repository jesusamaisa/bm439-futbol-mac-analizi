#!/usr/bin/env bash
# Son eğitim (A2) ağırlığı yazılana ve hiçbir train_detector python süreci kalmayana kadar bekler, sonra zinciri çalıştırır.
cd "$(dirname "$0")/.."
busy() {
  powershell -NoProfile -Command "\$p = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*bgoru.experiments.train_detector*' }; if (\$p) { exit 0 } else { exit 1 }"
}
until [ -f weights/bgoru_A2_ft640_cp.pt ] && ! busy; do sleep 30; done
echo "[$(date +%H:%M:%S)] eğitimler bitti, zincir başlıyor"
bash experiments/final_chain.sh
