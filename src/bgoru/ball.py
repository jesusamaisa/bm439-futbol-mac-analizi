"""Tek top seçimi ve nedensel (causal) kısa boşluk doldurma.

Sahada tek top bulunduğundan her karede en fazla bir top tutulur: aday tespitler
güven skoru ve önceki konuma yakınlıkla puanlanır. Top birkaç kare tespit
edilemezse son hız vektörüyle ``max_gap`` kareye kadar konum tahmin edilir
(gerçek zamanlı çalışma için yalnızca geçmiş kareler kullanılır).
"""
from __future__ import annotations

from collections import deque

import numpy as np


class BallTracker:
    def __init__(self, fps: float = 25.0, max_gap_s: float = 0.4, gate_px: float = 220.0, fill: bool = True,
                 trail_s: float = 3.0):
        self.max_gap = max(1, int(round(fps * max_gap_s)))
        self.gate = gate_px
        self.fill = fill
        self.last: np.ndarray | None = None
        self.vel = np.zeros(2)
        self.gap = 0
        self.trail_img: deque = deque(maxlen=int(fps * trail_s))
        self.trail_m: deque = deque(maxlen=int(fps * trail_s))

    def update(self, boxes: np.ndarray, conf: np.ndarray):
        """Dönüş: (merkez[x,y] | None, tahmin_mi, kutu | None)"""
        if len(boxes):
            c = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
            if self.last is not None and self.gap <= self.max_gap * 3:
                pred = self.last + self.vel * (self.gap + 1)
                d = np.linalg.norm(c - pred, axis=1)
                score = conf - 0.5 * d / self.gate
                score[d > self.gate * (1 + self.gap)] = -np.inf
                i = int(np.argmax(score))
                if not np.isfinite(score[i]):
                    i = int(np.argmax(conf))
            else:
                i = int(np.argmax(conf))
            new = c[i]
            if self.last is not None and self.gap <= self.max_gap:
                v = (new - self.last) / (self.gap + 1)
                self.vel = 0.6 * v + 0.4 * self.vel
            else:
                self.vel = np.zeros(2)
            self.last, self.gap = new, 0
            self.trail_img.append(tuple(new))
            return new, False, boxes[i]
        self.gap += 1
        if self.fill and self.last is not None and self.gap <= self.max_gap:
            return self.last + self.vel * self.gap, True, None
        return None, False, None
