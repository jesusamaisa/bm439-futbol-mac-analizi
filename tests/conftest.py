"""Ortak test ayarı: proje kaynak kodu (src/) içe aktarma yoluna eklenir.

Çalıştırma (proje kökünden):  python -m pytest -q tests
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
