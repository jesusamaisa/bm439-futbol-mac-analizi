"""Veri hazırlama: SoccerNet-GSR etiketlerinin YOLO biçimine dönüştürülmesi."""
from __future__ import annotations

import pytest


def test_gsr_labels_to_yolo_classes_and_clipping():
    """SoccerNet-GSR -> YOLO: kaleci oyuncu sınıfına, 'other' dışarıda; kutular görüntüye kırpılır."""
    from bgoru.data.prepare_gsr import CAT2CLS, yolo_lines
    box = lambda x, y, w, h: {"x": x, "y": y, "w": w, "h": h}
    objs = [{"category_id": 1, "bbox_image": box(100, 200, 40, 80)},     # oyuncu
            {"category_id": 2, "bbox_image": box(10, 10, 20, 40)},       # kaleci -> oyuncu
            {"category_id": 3, "bbox_image": box(1900, 1000, 40, 100)},  # hakem, sağ-alt kenardan taşar
            {"category_id": 4, "bbox_image": box(960, 540, 10, 10)},     # top
            {"category_id": 7, "bbox_image": box(0, 0, 50, 50)}]         # other -> yok sayılır
    lines = yolo_lines(objs, 1920, 1080)
    assert [ln.split()[0] for ln in lines] == ["0", "0", "1", "2"]
    assert CAT2CLS == {1: 0, 2: 0, 3: 1, 4: 2}
    cx, cy, w, h = map(float, lines[2].split()[1:])
    assert cx + w / 2 == pytest.approx(1.0) and cy + h / 2 == pytest.approx(1.0)  # kırpıldı
    assert float(lines[3].split()[3]) == pytest.approx(10 / 1920, abs=1e-6)  # 6 ondalık
