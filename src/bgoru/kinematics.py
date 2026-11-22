"""Koşu mesafesi ve hız ölçümü.

Konum kaynağı öncelik sırası:
  1. homografi ile metre cinsinden saha koordinatı (doğru ölçek)
  2. homografi yoksa: kamera hareketi telafili piksel konumu, oyuncu kutu yüksekliği
     ≈ 1,8 m varsayımıyla metreye çevrilir (yaklaşık; raporda belirtilir)
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

PLAYER_HEIGHT_M = 1.8
MAX_SPEED_KMH = 38.0


class Kinematics:
    def __init__(self, fps: float = 25.0, window_s: float = 0.5, ema: float = 0.35):
        self.fps = fps
        self.win = max(2, int(round(fps * window_s)))
        self.ema = ema
        self.hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.win + 1))
        self.smooth: dict[int, np.ndarray] = {}
        self.distance: dict[int, float] = defaultdict(float)
        self.max_speed: dict[int, float] = defaultdict(float)
        self.speed_sum: dict[int, float] = defaultdict(float)
        self.speed_n: dict[int, int] = defaultdict(int)
        self.current_speed: dict[int, float] = {}
        self.approx = False

    def update(self, frame_idx: int, tid: int, pos_m: np.ndarray):
        s = self.smooth.get(tid)
        s = pos_m.copy() if s is None else self.ema * pos_m + (1 - self.ema) * s
        h = self.hist[tid]
        if h and frame_idx - h[-1][0] > self.fps:  # uzun kayıp sonrası sıfırla
            h.clear()
            s = pos_m.copy()
        self.smooth[tid] = s
        if h:
            step = np.linalg.norm(s - h[-1][1])
            dt = (frame_idx - h[-1][0]) / self.fps
            if dt > 0 and step / dt * 3.6 <= MAX_SPEED_KMH:
                self.distance[tid] += step
        h.append((frame_idx, s.copy()))
        if len(h) > self.win // 2:
            dt = (h[-1][0] - h[0][0]) / self.fps
            if dt > 0:
                v = np.linalg.norm(h[-1][1] - h[0][1]) / dt * 3.6
                if v <= MAX_SPEED_KMH:
                    self.current_speed[tid] = v
                    self.max_speed[tid] = max(self.max_speed[tid], v)
                    self.speed_sum[tid] += v
                    self.speed_n[tid] += 1

    def summary(self, tid: int) -> dict:
        n = self.speed_n.get(tid, 0)
        return {"distance_m": round(self.distance.get(tid, 0.0), 1),
                "max_speed_kmh": round(self.max_speed.get(tid, 0.0), 1),
                "avg_speed_kmh": round(self.speed_sum.get(tid, 0.0) / n, 1) if n else 0.0,
                "approx": self.approx}
