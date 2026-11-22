"""YOLOv8 tabanlı nesne tespiti ve sınıf eşleme.

Desteklenen ağırlık türleri:
  * ``bgoru``  — bu projede SoccerNet üzerinde ince ayarlanan modeller (0 player, 1 referee, 2 ball)
  * ``hmzbo``  — referans projenin taktik kamera modeli (aynı sınıf sırası)
  * ``coco``   — COCO ön-eğitimli YOLOv8 (0 person -> player, 32 sports ball -> ball; hakem yok)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bgoru.config import BALL, PLAYER, REFEREE, WEIGHTS

COCO_MAP = {0: PLAYER, 32: BALL}


def use_half(device: str | int | None = None) -> bool:
    """FP16 (yarı hassasiyet) yalnızca CUDA GPU'da açılır.

    CPU'da yarı hassasiyet emüle edildiği için çıkarım çok yavaşlar: 1280 px'de kare başına 26,9 sn'ye karşı
    FP32 ile 0,32 sn (G1, i5-13500H, 8 iş parçacığı). GPU'da ise FP16 hızlıdır ve sonuçları değiştirmez.
    """
    if device is not None and str(device).lower() == "cpu":
        return False
    import torch

    return torch.cuda.is_available()


@dataclass
class Detections:
    xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    conf: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    cls: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))

    def __len__(self):
        return len(self.conf)

    def select(self, mask) -> "Detections":
        return Detections(self.xyxy[mask], self.conf[mask], self.cls[mask])

    def of_class(self, c: int) -> "Detections":
        return self.select(self.cls == c)


def infer_kind(weights: str) -> str:
    w = str(weights).lower()
    if "hmzbo" in w:
        return "hmzbo"
    if "bgoru" in w or "best" in w or "last" in w:
        return "bgoru"
    return "coco"


class Detector:
    def __init__(self, weights: str, kind: str | None = None, imgsz: int = 1280,
                 conf_player: float = 0.30, conf_referee: float = 0.30, conf_ball: float = 0.15,
                 device: str | int | None = None, half: bool | None = None):
        from ultralytics import YOLO

        p = Path(weights)
        if not p.is_absolute() and (WEIGHTS / p).exists():
            p = WEIGHTS / p
        self.model = YOLO(str(p))
        self.kind = kind or infer_kind(str(weights))
        self.imgsz = imgsz
        self.thr = {PLAYER: conf_player, REFEREE: conf_referee, BALL: conf_ball}
        self.device = device
        # None: GPU varsa FP16, yoksa FP32; açıkça True verilse de CPU'da FP16 kullanılmaz
        self.half = use_half(device) if half is None else (half and use_half(device))

    def map_classes(self, cls: np.ndarray) -> np.ndarray:
        if self.kind == "coco":
            return np.array([COCO_MAP.get(int(c), -1) for c in cls], dtype=np.int64)
        return cls.astype(np.int64)

    def __call__(self, frame: np.ndarray) -> Detections:
        return self.predict_batch([frame])[0]

    def predict_batch(self, frames: list[np.ndarray]) -> list[Detections]:
        min_thr = min(self.thr.values())
        classes = [0, 32] if self.kind == "coco" else None
        res = self.model.predict(frames, imgsz=self.imgsz, conf=min_thr, verbose=False, half=self.half,
                                 device=self.device, classes=classes)
        out = []
        for r in res:
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy().astype(np.float32)
            conf = b.conf.cpu().numpy().astype(np.float32)
            cls = self.map_classes(b.cls.cpu().numpy())
            keep = cls >= 0
            keep &= conf >= np.array([self.thr.get(int(c), 1.0) for c in cls], dtype=np.float32) if len(cls) else keep
            out.append(Detections(xyxy[keep], conf[keep], cls[keep]))
        return out
