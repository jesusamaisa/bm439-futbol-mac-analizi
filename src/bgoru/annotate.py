"""Kare üstü çizimler: oyuncu elipsleri, top işareti, canlı istatistik paneli (HUD) ve
taktik harita."""
from __future__ import annotations

import cv2
import numpy as np

from bgoru.homography import draw_pitch, pitch_to_px
from bgoru.possession import EVENT_TR

FONT = cv2.FONT_HERSHEY_SIMPLEX
REF_COLOR = (0, 215, 255)

_TR = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def tr(s: str) -> str:
    """OpenCV Hershey yazı tipi Türkçe karakter çizemediği için ASCII'ye indirger."""
    return s.translate(_TR)


def text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1, shadow=True):
    s = tr(s)
    if shadow:
        cv2.putText(img, s, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_player(img, box, color, label=None, possessor=False, palette=None):
    x0, y0, x1, y1 = [int(v) for v in box]
    cx, w = (x0 + x1) // 2, x1 - x0
    cv2.ellipse(img, (cx, y1), (max(6, int(w * 0.6)), max(3, int(0.22 * w))), 0, -45, 235, color, 2, cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.42, 1)
        cv2.rectangle(img, (cx - tw // 2 - 3, y1 + 6), (cx + tw // 2 + 3, y1 + 8 + th + 4), color, -1)
        bright = sum(color) > 450
        cv2.putText(img, label, (cx - tw // 2, y1 + 8 + th), FONT, 0.42, (0, 0, 0) if bright else (255, 255, 255), 1,
                    cv2.LINE_AA)
    if possessor:
        pts = np.array([[cx, y0 - 6], [cx - 8, y0 - 20], [cx + 8, y0 - 20]], np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 255))
        cv2.polylines(img, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)
    if palette is not None:
        cv2.rectangle(img, (x1 + 2, y0), (x1 + 10, y0 + 8), tuple(int(c) for c in palette), -1)


def draw_ball(img, xy, predicted=False):
    x, y = int(xy[0]), int(xy[1])
    pts = np.array([[x, y - 8], [x - 8, y - 22], [x + 8, y - 22]], np.int32)
    col = (0, 255, 0) if not predicted else (160, 160, 160)
    cv2.fillPoly(img, [pts], col)
    cv2.polylines(img, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(img, snap, team_names, team_bgr, events_tail=3):
    H, W = img.shape[:2]
    s = W / 1920
    pw, ph = int(560 * s), int(205 * s)
    x0, y0 = int(20 * s), int(20 * s)
    over = img.copy()
    cv2.rectangle(over, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 20), -1)
    cv2.addWeighted(over, 0.62, img, 0.38, 0, img)
    fs = 0.62 * s
    text(img, "BILGISAYARLI GORU  |  Canli Mac Analizi", (x0 + int(12 * s), y0 + int(26 * s)), 0.55 * s, (200, 230, 255), 1)
    pct = snap["possession_pct"]
    bx, by, bw, bh = x0 + int(12 * s), y0 + int(48 * s), pw - int(24 * s), int(26 * s)
    split = int(bw * pct[0] / 100)
    cv2.rectangle(img, (bx, by), (bx + split, by + bh), team_bgr[0], -1)
    cv2.rectangle(img, (bx + split, by), (bx + bw, by + bh), team_bgr[1], -1)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
    def ink(bg):
        return ((20, 20, 20), False) if sum(bg) > 400 else ((255, 255, 255), True)

    c0, sh0 = ink(team_bgr[0])
    text(img, f"{team_names[0]} %{pct[0]:.0f}", (bx + 6, by + int(19 * s)), fs, c0, 2, shadow=sh0)
    r = f"%{pct[1]:.0f} {team_names[1]}"
    (tw, _), _ = cv2.getTextSize(tr(r), FONT, fs, 2)
    c1, sh1 = ink(team_bgr[1])
    text(img, r, (bx + bw - tw - 6, by + int(19 * s)), fs, c1, 2, shadow=sh1)
    wp = snap["possession_pct_window"]
    y = by + bh + int(24 * s)
    text(img, f"Top hakimiyeti (son 60 sn): %{wp[0]:.0f} - %{wp[1]:.0f}", (bx, y), 0.52 * s)
    ps, ppm = snap["passes"], snap["passes_per_min"]
    acc = ["-" if a is None else f"%{a:.0f}" for a in snap["pass_accuracy"]]
    y += int(24 * s)
    text(img, f"Pas: {ps[0]} ({ppm[0]:.1f}/dk, isabet {acc[0]})  |  {ps[1]} ({ppm[1]:.1f}/dk, isabet {acc[1]})",
         (bx, y), 0.52 * s)
    y += int(24 * s)
    text(img, f"Gecen sure: {snap['elapsed_s']:.0f} sn   Pas sikligi (60 sn): "
              f"{snap['passes_per_min_window'][0]:.1f} / {snap['passes_per_min_window'][1]:.1f} pas/dk", (bx, y), 0.5 * s,
         (190, 190, 190))
    for ev in snap["last_events"][-events_tail:][::-1]:
        y += int(22 * s)
        col = team_bgr[ev["team"]] if ev["team"] in (0, 1) else (160, 160, 160)
        sq = int(11 * s)
        cv2.rectangle(img, (bx, y - sq), (bx + sq, y), col, -1)
        cv2.rectangle(img, (bx, y - sq), (bx + sq, y), (255, 255, 255), 1)
        to = f" -> #{ev['to']}" if "to" in ev and ev["type"] in ("PASS", "KEY_PASS", "PASS_FAIL") else ""
        text(img, f"{ev['t']:6.1f}s  #{ev['tid']}{to}  {EVENT_TR[ev['type']]}", (bx + sq + 6, y), 0.5 * s)
    return img


class TacticalMap:
    def __init__(self, scale: float = 5.0):
        self.scale = scale
        self.base = draw_pitch(scale=scale)

    def render(self, players_m, teams, team_bgr, refs_m=None, ball_trail_m=None, possessor_idx=None):
        m = self.base.copy()
        if ball_trail_m is not None and len(ball_trail_m) > 1:
            pts = pitch_to_px(np.array(ball_trail_m), self.scale)
            for i in range(1, len(pts)):
                a = i / len(pts)
                cv2.line(m, tuple(pts[i - 1]), tuple(pts[i]), (int(255 * a), int(255 * a), int(255 * a)), 2, cv2.LINE_AA)
        if refs_m is not None:
            for p in pitch_to_px(refs_m, self.scale):
                cv2.circle(m, tuple(p), 6, REF_COLOR, -1)
                cv2.circle(m, tuple(p), 6, (0, 0, 0), 1)
        if players_m is not None:
            for i, (p, t) in enumerate(zip(pitch_to_px(players_m, self.scale), teams)):
                col = team_bgr[t] if t in (0, 1) else (200, 200, 200)
                cv2.circle(m, tuple(p), 8, col, -1)
                cv2.circle(m, tuple(p), 8, (0, 0, 255) if i == possessor_idx else (0, 0, 0), 2 if i == possessor_idx else 1)
        if ball_trail_m is not None and len(ball_trail_m):
            b = pitch_to_px(np.array(ball_trail_m[-1:]), self.scale)[0]
            cv2.circle(m, tuple(b), 5, (255, 255, 255), -1)
            cv2.circle(m, tuple(b), 5, (0, 0, 0), 1)
        return m


def overlay_map(img, m, frac=0.30, margin=16):
    H, W = img.shape[:2]
    w = int(W * frac)
    h = int(m.shape[0] * w / m.shape[1])
    mm = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    y0, x0 = H - h - margin, W - w - margin
    img[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(img[y0:y0 + h, x0:x0 + w], 0.15, mm, 0.85, 0)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
    return img
