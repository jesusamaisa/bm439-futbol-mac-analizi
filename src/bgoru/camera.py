"""Kamera hareketi telafisi (homografi yokken yedek).

Yayın görüntülerinde saha anahtar noktaları bulunamazsa oyuncu hareketi, kameranın
kaydırma (pan) hareketinden ayrılmalıdır. Oyuncu kutuları dışında kalan köşe
noktaları Lucas–Kanade optik akışıyla izlenir ve medyan öteleme kamera hareketi kabul edilir.
"""
from __future__ import annotations

import cv2
import numpy as np


class CameraMotion:
    def __init__(self, max_corners: int = 300, scale: float = 0.5):
        self.prev_gray = None
        self.offset = np.zeros(2)  # birikimli kamera ötelemesi (piksel)
        self.max_corners = max_corners
        self.s = scale

    def update(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(cv2.resize(frame, None, fx=self.s, fy=self.s), cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None:
            mask = np.full(g.shape, 255, np.uint8)
            for b in boxes:
                x0, y0, x1, y1 = (np.asarray(b) * self.s).astype(int)
                mask[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4] = 0
            p0 = cv2.goodFeaturesToTrack(self.prev_gray, self.max_corners, 0.01, 8, mask=mask)
            if p0 is not None and len(p0) >= 10:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, g, p0, None)
                ok = st.ravel() == 1
                if ok.sum() >= 8:
                    d = np.median((p1[ok] - p0[ok]).reshape(-1, 2), 0) / self.s
                    self.offset += d
        self.prev_gray = g
        return self.offset.copy()
