"""ByteTrack ile oyuncu/hakem kimlik takibi (supervision kütüphanesi)."""
from __future__ import annotations

import numpy as np
import supervision as sv

from bgoru.config import BALL
from bgoru.detection import Detections


class PlayerTracker:
    def __init__(self, fps: float = 25.0, lost_seconds: float = 2.0):
        self.tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=int(fps * lost_seconds),
                                    minimum_matching_threshold=0.8, frame_rate=int(round(fps)))

    def update(self, det: Detections) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Top dışındaki tespitleri izler. Dönüş: xyxy, conf, cls, track_id"""
        d = det.select(det.cls != BALL)
        svd = sv.Detections(xyxy=d.xyxy.astype(np.float32), confidence=d.conf.astype(np.float32),
                            class_id=d.cls.astype(int))
        t = self.tracker.update_with_detections(svd)
        if len(t) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int), np.zeros(0, int)
        return t.xyxy, t.confidence, t.class_id, t.tracker_id
