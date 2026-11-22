"""Takım ataması.

Üç yöntem aynı arayüzle sunulur (RQ2 karşılaştırması için):

* ``baseline``  — tüm sınırlayıcı kutunun ortalama RGB rengi + K-Means(2), ilk karede eğitilir.
* ``alternative`` — kutunun üst yarısında piksel düzeyinde K-Means(2) ile arka plan/forma ayrımı
  (köşe pikselleri arka plan kabul edilir), forma rengi + K-Means(2) (yaygın açık kaynak yaklaşımı).
* ``proposed``  — gövde merkez kırpması + HSV çim maskesi + LAB uzayında medyan renk;
  takım merkezleri ilk ``fit_frames`` karede toplanan örneklerle, aykırı değerler (kaleci vb.)
  ayıklanarak bulunur; nihai karar her iz (track) için birikimli **çoğunluk oylaması** ile verilir.

İsteğe bağlı olarak kullanıcı takım renklerini verirse (videodaki yöntem), LAB uzaklığı ile
en yakın renge atama yapılır.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans


def _clip_box(box, W, H):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


def grass_mask_hsv(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 32) & (h <= 90) & (s >= 40) & (v >= 40)


def feat_full_mean_rgb(img, box):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = _clip_box(box, W, H)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop.reshape(-1, 3)[:, ::-1].mean(0)  # BGR->RGB


def feat_tophalf_kmeans(img, box):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = _clip_box(box, W, H)
    crop = img[y0:y0 + max(1, (y1 - y0) // 2), x0:x1]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    px = crop.reshape(-1, 3).astype(np.float32)
    km = KMeans(2, n_init=1, random_state=0).fit(px)
    lab = km.labels_.reshape(crop.shape[:2])
    corners = [lab[0, 0], lab[0, -1], lab[-1, 0], lab[-1, -1]]
    bg = max(set(corners), key=corners.count)
    return km.cluster_centers_[1 - bg][::-1]


def _to_lab(px_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(px_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lab[:, 0] *= 100 / 255.0
    lab[:, 1:] -= 128.0
    return lab


def local_pitch_lab(img, x0, y0, x1, y1, pad: float = 0.35) -> np.ndarray | None:
    """Kutunun hemen çevresindeki halkadan (kutu dışı) yerel saha rengi (LAB medyanı)."""
    H, W = img.shape[:2]
    w, h = x1 - x0, y1 - y0
    X0, Y0 = max(0, int(x0 - pad * w)), max(0, int(y0 + 0.3 * h))
    X1, Y1 = min(W, int(x1 + pad * w)), min(H, int(y1 + 0.15 * h))
    ring = img[Y0:Y1, X0:X1].copy()
    if ring.size == 0:
        return None
    m = np.ones(ring.shape[:2], bool)
    m[max(0, y0 - Y0):max(0, y1 - Y0), max(0, x0 - X0):max(0, x1 - X0)] = False
    if m.sum() < 10:
        return None
    return np.median(_to_lab(ring[m]), 0)


def feat_proposed(img, box, use_mask: bool | str = True, use_lab: bool = True, local_thr: float = 12.0):
    """use_mask: True/'hsv' = sabit HSV çim aralığı, 'local' = yerel saha rengine LAB uzaklığı, False = maske yok"""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = _clip_box(box, W, H)
    w, h = x1 - x0, y1 - y0
    if w < 4 or h < 8:
        return None
    # gövde bölgesi: genişliğin ortadaki %50'si, yüksekliğin %15–%55 aralığı
    cx0, cx1 = x0 + int(0.25 * w), x0 + int(0.75 * w)
    cy0, cy1 = y0 + int(0.15 * h), y0 + int(0.55 * h)
    crop = img[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None
    px = crop.reshape(-1, 3)
    if use_mask == "local":
        bg = local_pitch_lab(img, x0, y0, x1, y1)
        if bg is not None:
            keep = np.linalg.norm(_to_lab(px) - bg, axis=1) > local_thr
            if keep.sum() >= 8:
                px = px[keep]
    elif use_mask:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        keep = ~grass_mask_hsv(hsv)
        if keep.sum() >= 8:
            px = px[keep]
    if use_lab:
        lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        lab[:, 0] *= 100 / 255.0
        lab[:, 1:] -= 128.0
        return np.median(lab, 0)
    return np.median(px[:, ::-1].astype(np.float32), 0)


FEATURES = {
    "baseline": lambda img, b: feat_full_mean_rgb(img, b),
    "alternative": lambda img, b: feat_tophalf_kmeans(img, b),
    "proposed": lambda img, b: feat_proposed(img, b),
    "proposed_nomask": lambda img, b: feat_proposed(img, b, use_mask=False),
    "proposed_rgb": lambda img, b: feat_proposed(img, b, use_lab=False),
    "proposed_local": lambda img, b: feat_proposed(img, b, use_mask="local"),
    "proposed_local_rgb": lambda img, b: feat_proposed(img, b, use_mask="local", use_lab=False),
}


def bgr_to_lab(color_bgr) -> np.ndarray:
    lab = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    lab[0] *= 100 / 255.0
    lab[1:] -= 128.0
    return lab


class TeamAssigner:
    """Kare kare çalışan takım atayıcı.

    method: baseline | alternative | proposed | proposed_nomask | proposed_rgb
    vote:   iz-tabanlı çoğunluk oylaması (yalnızca track id verildiğinde)
    """

    def __init__(self, method: str = "proposed", vote: bool = True, fit_frames: int = 10,
                 min_samples: int = 12, team_colors_bgr: list | None = None, outlier_z: float = 3.0,
                 centers: list | None = None, spread: float | None = None, balanced: bool = True,
                 min_tracks: int = 6, min_share: float = 0.2, max_wait_frames: int = 150):
        self.method = method
        self.feat = FEATURES[method]
        self.vote = vote and method.startswith("proposed")
        self.fit_frames = fit_frames if method.startswith("proposed") else 1
        self.min_samples = min_samples if method.startswith("proposed") else 2
        self.outlier_z = outlier_z
        self.samples: list[np.ndarray] = []
        self.frames_seen = 0
        self.centers: np.ndarray | None = None
        self.spread: float = 1.0
        self.votes: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(2))
        # v2 (dengeli kümeleme): yeterli sayıda farklı oyuncu görülmeden ve iki küme dengeli
        # (her biri >= min_share) olmadan merkezler sabitlenmez; küçük küme (kaleci/hakem) atılır.
        self.balanced = balanced and method.startswith("proposed")
        self.min_tracks, self.min_share, self.max_wait = min_tracks, min_share, max_wait_frames
        self.seen_tracks: set[int] = set()
        if team_colors_bgr is not None:
            self.centers = np.stack([bgr_to_lab(c) for c in team_colors_bgr[:2]])
            self.spread = 15.0
        if centers is not None:  # önceden algılanmış (ve kullanıcının adlandırdığı) takım merkezleri
            self.centers = np.asarray(centers, np.float32)
            self.spread = float(spread or 15.0)

    @property
    def ready(self) -> bool:
        return self.centers is not None

    def _fit(self, force: bool = False) -> bool:
        X = np.stack(self.samples)
        km = KMeans(2, n_init=10, random_state=0).fit(X)
        if self.balanced:
            for _ in range(2):  # en fazla iki küçük küme (ör. kaleci / hakem) ayıklanır
                share = np.bincount(km.labels_, minlength=2) / len(X)
                if share.min() >= self.min_share:
                    break
                X = X[km.labels_ == int(np.argmax(share))]
                if len(X) < self.min_samples:
                    return False
                km = KMeans(2, n_init=10, random_state=0).fit(X)
            share = np.bincount(km.labels_, minlength=2) / len(X)
            if share.min() < self.min_share and not force:
                return False
        if self.method.startswith("proposed"):
            d = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
            med, mad = np.median(d), np.median(np.abs(d - np.median(d))) + 1e-6
            inl = d < med + self.outlier_z * 1.4826 * mad
            if inl.sum() >= 4 and len(set(km.labels_[inl])) == 2:
                km = KMeans(2, n_init=10, random_state=0).fit(X[inl])
                d = np.linalg.norm(X[inl] - km.cluster_centers_[km.labels_], axis=1)
            self.spread = float(np.median(d) + 1e-6)
        # takım 0 = daha açık renkli küme (kararlı sıralama için)
        order = np.argsort([-c[0] for c in km.cluster_centers_])
        self.centers = km.cluster_centers_[order]
        return True

    def assign(self, img: np.ndarray, boxes: np.ndarray, track_ids: np.ndarray | None = None):
        """Dönüş: (team[N] ∈ {0,1,-1}, feats[N,3] veya NaN, outlier[N] bool)"""
        feats = []
        for b in boxes:
            f = self.feat(img, b)
            feats.append(f if f is not None else np.full(3, np.nan, np.float32))
        feats = np.array(feats, np.float32).reshape(-1, 3)
        if self.centers is None:
            valid = ~np.isnan(feats).any(1)
            self.samples.extend(list(feats[valid]))
            self.frames_seen += 1
            if track_ids is not None:
                self.seen_tracks.update(int(t) for t in track_ids)
            enough_tracks = (not self.balanced or track_ids is None or len(self.seen_tracks) >= self.min_tracks
                             or self.frames_seen >= self.max_wait)
            if self.frames_seen >= self.fit_frames and len(self.samples) >= self.min_samples and enough_tracks:
                self._fit(force=self.frames_seen >= self.max_wait)
            elif self.frames_seen >= self.fit_frames and len(self.samples) >= 2 and not self.method.startswith("proposed"):
                self._fit()
            if self.centers is None:
                return np.full(len(boxes), -1), feats, np.zeros(len(boxes), bool)
        team = np.full(len(boxes), -1)
        outlier = np.zeros(len(boxes), bool)
        for i, f in enumerate(feats):
            if np.isnan(f).any():
                continue
            d = np.linalg.norm(self.centers - f, axis=1)
            team[i] = int(np.argmin(d))
            outlier[i] = d.min() > 4.0 * self.spread
            if self.vote and track_ids is not None and not outlier[i]:
                # yakınlık ağırlıklı oy: belirsiz kararlar daha az oy alır
                margin = abs(d[0] - d[1]) / (d.sum() + 1e-6)
                self.votes[int(track_ids[i])][team[i]] += 0.5 + margin
        if self.vote and track_ids is not None:
            for i, t in enumerate(track_ids):
                v = self.votes.get(int(t))
                if v is not None and v.sum() > 0:
                    team[i] = int(np.argmax(v))
        return team, feats, outlier
