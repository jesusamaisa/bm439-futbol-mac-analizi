"""Proje raporunu (DOCX) ders şablonunun stilleriyle üretir; tüm sayılar results/ altındaki
deney çıktılarından okunur (elle yazılmış ölçüm yoktur).

Kullanım (proje kökünden):
    python report/build_report.py           -> report/rapor.docx
    powershell -File report/to_pdf.ps1      -> report/rapor.pdf (Microsoft Word gerekir)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "report"))
from docx_helpers import Report  # noqa: E402

R = ROOT / "results"
FIGS = R / "figures"


def f3(x):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.3f}"


def f1(x):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.1f}"


def load():
    d = {}
    p = R / "detection" / "detection_results.csv"
    d["det"] = pd.read_csv(p).set_index("config") if p.exists() else None
    p = R / "detection" / "detection_bootstrap.json"
    d["boot"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    p = R / "team" / "team_results.csv"
    d["team"] = pd.read_csv(p) if p.exists() else None
    p = R / "possession" / "possession_results.csv"
    d["poss"] = pd.read_csv(p).set_index("variant") if p.exists() else None
    p = R / "robustness" / "robustness_results.csv"
    d["rob"] = pd.read_csv(p) if p.exists() else None
    p = R / "benchmark" / "model_benchmark.csv"
    d["bench"] = pd.read_csv(p).set_index("model") if p.exists() else None
    p = R / "benchmark" / "pipeline_benchmark.csv"
    d["pipe"] = pd.read_csv(p).iloc[0] if p.exists() else None
    p = R / "failures" / "failures.csv"
    d["fail"] = pd.read_csv(p) if p.exists() else None
    p = R / "failures" / "failure_totals.csv"
    d["fail_tot"] = pd.read_csv(p).iloc[0] if p.exists() else None
    p = ROOT / "environment" / "env_info.json"
    d["env"] = json.loads(p.read_text(encoding="utf-8"))
    p = ROOT / "experiments" / "registry.csv"
    d["reg"] = pd.read_csv(p) if p.exists() else None
    p = R / "demo_runs" / "demo_summary.json"
    d["demo"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    p = ROOT / "data" / "soccernet_cp" / "copy_paste_info.json"
    d["cp"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    p = R / "detection" / "seed_repeats.json"  # tohum tekrarı (E3, A1 x seed 0/1)
    d["seeds"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    p = R / "detection" / "seed_repeats.csv"
    d["seeds_df"] = pd.read_csv(p) if p.exists() else None
    p = R / "detection" / "gsr_extra.json"  # ek eğitim verisi deneyi (A1, C1, G1)
    d["gsr"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    p = ROOT / "report" / "ogrenci_bilgileri.yaml"  # kapak / ekip katkıları / yapay zekâ doğrulama notu
    d["ogr"] = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}
    return d


def build(d, out: Path):
    from content import write_all  # rapor metni ayrı modülde

    rep = Report(ROOT / "report" / "_template" / "sablon.docx")
    write_all(rep, d)
    rep.save(out)
    print("[✓]", out)
    left = open_fields(out)
    if left:
        print(f"[!] Doldurulmamış {len(left)} alan var (report/ogrenci_bilgileri.yaml): " + "; ".join(left))


def open_fields(docx_path: Path) -> list[str]:
    """Teslimden önce kaldırılması gereken yer tutucular (ortak koşul: 'Şablondaki yer tutucuları kaldırın')."""
    from docx import Document

    doc = Document(str(docx_path))
    out = []
    for t in doc.tables:  # kapak ve ekip tablosu: satır etiketiyle
        for row in t.rows:
            cells = [c.text for c in row.cells]
            if any("………" in c or "Doldurunuz" in c for c in cells[1:]):
                out.append(f"{cells[0]} (katkı)" if t.rows[0].cells[0].text == "Üye" else cells[0])
    for p in doc.paragraphs:
        if "[Öğrenci doğrulaması" in p.text:
            out.append("Yapay zekâ doğrulama notu")
        elif "[Doldurunuz" in p.text:
            out.append("Ekip katkıları")
    return list(dict.fromkeys(out))


if __name__ == "__main__":
    build(load(), ROOT / "report" / "rapor.docx")
