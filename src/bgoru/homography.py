"""Saha anahtar noktaları -> homografi -> metre cinsinden taktik harita.

Anahtar nokta modeli (referans proje, YOLOv8M) 28 saha noktasını kutu olarak tespit eder.
Bu modül her noktanın gerçek FIFA ölçülerindeki (105 x 68 m) konumunu kullanarak
görüntü düzleminden saha düzlemine homografi hesaplar. Böylece hız ve mesafe
**metre** cinsinden ölçülebilir (referans projedeki piksel tabanlı harita ölçekli değildir).

Homografi güncelleme koşulu (referans videodaki fikir + iyileştirme):
  * önceki karede de görülen ortak nokta sayısı >= ``min_common`` ise ve bu noktaların
    ortalama yer değiştirmesi ``tol_px`` eşiğinin altındaysa eski matris korunur
    (titremeyi ve gereksiz hesaplamayı önler);
  * aksi halde RANSAC ile yeni matris hesaplanır, geri izdüşüm hatası kontrol edilir ve
    önceki matrisle üstel olarak yumuşatılır.
"""
from __future__ import annotations

import cv2
import numpy as np

from bgoru.config import WEIGHTS

PITCH_LEN, PITCH_WID = 105.0, 68.0
_GW6, _GW18 = 18.32 / 2, 40.32 / 2
_ARC = float(np.sqrt(9.15 ** 2 - 5.5 ** 2))
# (enine konum xw, boyuna konum yl) — referans modelin dikey saha düzeni
_V = {
    "TLC": (0, 0), "TRC": (68, 0), "TR6MC": (34 + _GW6, 5.5), "TL6MC": (34 - _GW6, 5.5),
    "TR6ML": (34 + _GW6, 0), "TL6ML": (34 - _GW6, 0), "TR18MC": (34 + _GW18, 16.5), "TL18MC": (34 - _GW18, 16.5),
    "TR18ML": (34 + _GW18, 0), "TL18ML": (34 - _GW18, 0), "TRArc": (34 + _ARC, 16.5), "TLArc": (34 - _ARC, 16.5),
    "RML": (68, 52.5), "RMC": (34 + 9.15, 52.5), "LMC": (34 - 9.15, 52.5), "LML": (0, 52.5),
    "BLC": (0, 105), "BRC": (68, 105), "BR6MC": (34 + _GW6, 99.5), "BL6MC": (34 - _GW6, 99.5),
    "BR6ML": (34 + _GW6, 105), "BL6ML": (34 - _GW6, 105), "BR18MC": (34 + _GW18, 88.5), "BL18MC": (34 - _GW18, 88.5),
    "BR18ML": (34 + _GW18, 105), "BL18ML": (34 - _GW18, 105), "BRArc": (34 + _ARC, 88.5), "BLArc": (34 - _ARC, 88.5),
}
KP_NAMES = ["TLC", "TRC", "TR6MC", "TL6MC", "TR6ML", "TL6ML", "TR18MC", "TL18MC", "TR18ML", "TL18ML", "TRArc",
            "TLArc", "RML", "RMC", "LMC", "LML", "BLC", "BRC", "BR6MC", "BL6MC", "BR6ML", "BL6ML", "BR18MC",
            "BL18MC", "BR18ML", "BL18ML", "BRArc", "BLArc"]
# yatay taktik harita: X = boyuna (0..105), Y = enine (0..68)
KP_METERS = {k: (float(v[1]), float(v[0])) for k, v in _V.items()}


class PitchHomography:
    def __init__(self, weights: str = "hmzbo_keypoints_yolov8m.pt", conf: float = 0.5, imgsz: int = 640,
                 tol_px: float = 4.0, min_common: int = 4, max_age: int = 250, smooth: float = 0.6,
                 max_reproj_m: float = 2.5, half: bool | None = None):
        from ultralytics import YOLO

        from bgoru.detection import use_half

        self.model = YOLO(str(WEIGHTS / weights))
        self.conf, self.imgsz = conf, imgsz
        self.half = use_half() if half is None else (half and use_half())  # FP16 yalnızca GPU'da
        self.tol_px, self.min_common, self.max_age = tol_px, min_common, max_age
        self.smooth, self.max_reproj_m = smooth, max_reproj_m
        self.H: np.ndarray | None = None
        self.age = 0
        self.prev_kps: dict[int, np.ndarray] = {}
        self.last_kps: dict[int, np.ndarray] = {}
        self.n_updates = 0

    def detect(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False, half=self.half)[0]
        kps: dict[int, tuple[float, np.ndarray]] = {}
        for b, c, k in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            k = int(k)
            if k not in kps or c > kps[k][0]:
                kps[k] = (float(c), b[:2].astype(np.float64))
        return {k: v[1] for k, v in kps.items()}

    def _compute(self, kps: dict[int, np.ndarray]) -> np.ndarray | None:
        ids = sorted(kps)
        src = np.array([kps[i] for i in ids], np.float64)
        dst = np.array([KP_METERS[KP_NAMES[i]] for i in ids], np.float64)
        if len(ids) >= 5:
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 1.5)
        else:
            H, inl = cv2.findHomography(src, dst, 0)
        if H is None or abs(H[2, 2]) < 1e-12:
            return None
        H = H / H[2, 2]
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(proj - dst, axis=1)
        if inl is not None and len(ids) >= 5:
            err = err[inl.ravel().astype(bool)]
        if not len(err) or np.median(err) > self.max_reproj_m:
            return None
        return H

    def update(self, frame: np.ndarray) -> bool:
        kps = self.detect(frame)
        self.last_kps = kps
        self.age += 1
        if len(kps) >= 4:
            need = self.H is None
            if not need:
                common = set(kps) & set(self.prev_kps)
                if len(common) >= self.min_common:
                    disp = np.mean([np.linalg.norm(kps[c] - self.prev_kps[c]) for c in common])
                    need = disp > self.tol_px
                else:
                    need = True
            if need:
                H = self._compute(kps)
                if H is not None:
                    if self.H is not None and self.smooth > 0:
                        H = self.smooth * H + (1 - self.smooth) * self.H
                        H = H / H[2, 2]
                    self.H, self.age = H, 0
                    self.n_updates += 1
                    self.prev_kps = kps
            else:
                self.age = 0
        if self.age > self.max_age:
            self.H = None
        return self.H is not None

    def to_pitch(self, pts: np.ndarray) -> np.ndarray | None:
        if self.H is None or len(pts) == 0:
            return None
        return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), self.H).reshape(-1, 2)


def draw_pitch(scale: float = 6.0, margin: int = 18, bg=(38, 120, 56), line=(235, 235, 235)) -> np.ndarray:
    """Metre ölçekli yatay taktik harita görüntüsü üretir."""
    W, H = int(PITCH_LEN * scale + 2 * margin), int(PITCH_WID * scale + 2 * margin)
    img = np.full((H, W, 3), bg, np.uint8)
    for i in range(0, 11):  # çim şeritleri
        x0 = int(margin + i * PITCH_LEN / 10 * scale)
        if i % 2 == 0 and i < 10:
            cv2.rectangle(img, (x0, margin), (int(x0 + PITCH_LEN / 10 * scale), H - margin),
                          tuple(min(255, c + 10) for c in bg), -1)

    def P(x, y):
        return int(margin + x * scale), int(margin + y * scale)

    t = max(1, int(scale / 4))
    cv2.rectangle(img, P(0, 0), P(PITCH_LEN, PITCH_WID), line, t)
    cv2.line(img, P(52.5, 0), P(52.5, 68), line, t)
    cv2.circle(img, P(52.5, 34), int(9.15 * scale), line, t)
    cv2.circle(img, P(52.5, 34), max(2, t + 1), line, -1)
    for x0, sgn in ((0, 1), (105, -1)):
        cv2.rectangle(img, P(x0, 34 - _GW18), P(x0 + sgn * 16.5, 34 + _GW18), line, t)
        cv2.rectangle(img, P(x0, 34 - _GW6), P(x0 + sgn * 5.5, 34 + _GW6), line, t)
        cv2.circle(img, P(x0 + sgn * 11, 34), max(2, t + 1), line, -1)
        cv2.ellipse(img, P(x0 + sgn * 11, 34), (int(9.15 * scale), int(9.15 * scale)), 0,
                    -53 if sgn == 1 else 127, 53 if sgn == 1 else 233, line, t)
        cv2.rectangle(img, P(x0, 34 - 3.66), P(x0 - sgn * 1.5, 34 + 3.66), line, t)
    return img


def pitch_to_px(pts_m: np.ndarray, scale: float = 6.0, margin: int = 18) -> np.ndarray:
    return (np.asarray(pts_m) * scale + margin).astype(int)
