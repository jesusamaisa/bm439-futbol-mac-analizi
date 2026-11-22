"""Teslim paketi: ortak koşullardaki klasör düzeniyle tek bir ZIP (teslim/BilgisayarliGoru_teslim.zip).

Eklenenler: README.md, ADIMLAR.md, src/, app/, configs/, environment/, tests/, experiments/ (loglar, kayıt
defteri, eğitim eğrileri), results/, report/, demo/, weights/ ve veri ayrımını belgeleyen manifest dosyaları.
Dışarıda bırakılanlar: ham veri (data/ görüntüleri, ~2,5 GB — README'deki "Veri edinme" adımlarıyla yeniden
üretilir), önbellekler, arayüz deneme çıktıları, ara tahmin dosyaları, eğitim checkpoint kopyaları ve
demo_runs içindeki video kopyaları (aynıları demo/ altında).

Ağırlıklar: varsayılan olarak yalnızca son sistemin çalışması için gerekenler (A1 dedektörü + saha anahtar noktası
modeli). --all-weights tüm ablation ağırlıklarını, --no-weights hiçbirini eklemez.

Kullanım: python -m bgoru.package [--all-weights | --no-weights] [--out teslim/BilgisayarliGoru_teslim.zip]
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
import zipfile
from pathlib import Path

from bgoru.config import FINAL_DET_WEIGHTS, ROOT

INCLUDE = ["README.md", "ADIMLAR.md", ".gitignore", "src", "app", "configs", "environment", "tests", "experiments",
           "results", "report", "demo", "weights", "data/soccernet/manifests", "data/soccernet_cp/copy_paste_info.json",
           "data/soccernet_gsr/manifests", "data/soccernet_gsr/KAYNAK.md"]
EXCLUDE = ["*/__pycache__/*", "*.pyc", "*/.pytest_cache/*", "*/~$*", "experiments/preds/*",
           "experiments/runs/*/weights/*", "experiments/runs/*/events.out.tfevents.*",  # adı makine adını içerir; eğriler results.csv'de
           "results/app_runs/*", "results/_smoke/*", "results/demo_runs/*/analiz.mp4",
           "report/_tmp*", "demo/_gosterim_tmp.mp4"]
FINAL_WEIGHTS = [FINAL_DET_WEIGHTS, "hmzbo_keypoints_yolov8m.pt", "README.md"]
STORED = {".mp4", ".pt", ".png", ".jpg", ".jpeg", ".zip", ".pdf", ".docx"}  # zaten sıkıştırılmış biçimler


def files(weights: str) -> list[Path]:
    out = []
    for item in INCLUDE:
        p = ROOT / item
        cand = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file()) if p.exists() else []
        for f in cand:
            rel = f.relative_to(ROOT).as_posix()
            if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch("/" + rel, pat) for pat in EXCLUDE):
                continue
            if rel.startswith("weights/") and (weights == "none" and f.name != "README.md"
                                               or weights == "final" and f.name not in FINAL_WEIGHTS):
                continue
            out.append(f)
    return out


def warn_report():
    """Rapor hâlâ yer tutucu içeriyorsa veya PDF, DOCX'ten eskiyse uyarır (paket yine de üretilir)."""
    rep = ROOT / "report"
    sys.path.insert(0, str(rep))
    from build_report import open_fields  # noqa: E402

    left = open_fields(rep / "rapor.docx")
    if left:
        print(f"[!] Raporda doldurulmamış {len(left)} alan var: {'; '.join(left)}\n"
              "    report/ogrenci_bilgileri.yaml dosyasını doldurup raporu yeniden üretin.")
    for src in (rep / "rapor.docx", rep / "ogrenci_bilgileri.yaml"):
        if src.exists() and (rep / "rapor.pdf").stat().st_mtime < src.stat().st_mtime - 1:
            print(f"[!] rapor.pdf, {src.name} dosyasından eski: report/to_pdf.ps1 ile PDF'i yeniden üretin.")
            break


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all-weights", action="store_true")
    g.add_argument("--no-weights", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "teslim" / "BilgisayarliGoru_teslim.zip"))
    a = ap.parse_args()
    weights = "all" if a.all_weights else "none" if a.no_weights else "final"
    warn_report()
    fs = files(weights)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    with zipfile.ZipFile(out, "w") as z:
        for f in fs:
            rel = f.relative_to(ROOT).as_posix()
            comp = zipfile.ZIP_STORED if f.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            z.write(f, f"BilgisayarliGoru/{rel}", compress_type=comp)
            top = rel.split("/")[0]
            sizes[top] = sizes.get(top, 0) + f.stat().st_size
    for k, v in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14} {v / 2**20:8.1f} MB")
    print(f"[✓] {out}  ({len(fs)} dosya, {out.stat().st_size / 2**20:.1f} MB, ağırlıklar: {weights})")


if __name__ == "__main__":
    main()
