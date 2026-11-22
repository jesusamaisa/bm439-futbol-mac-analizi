"""Oyuncu karakter kartı ve profil haritası çizimi (matplotlib)."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Wedge  # noqa: E402

from bgoru.profiling import ATTR_TR, ATTRS, PROFILE_CFG, unit  # noqa: E402

BG_TOP, BG_BOT = np.array([13, 27, 42]) / 255, np.array([27, 38, 59]) / 255
TXT, MUTED, ACCENT, GRID = "#E8EEF6", "#8FA3BF", "#4CC9F0", "#3A4A63"
FIG_W, FIG_H = 16.0, 9.0


def _rgb(bgr):
    return tuple(np.array(bgr[::-1]) / 255.0)


def _square_axes(fig, cx, cy, size_in):
    """Şekil koordinatında (cx, cy) merkezli, size_in inç kenarlı kare eksen."""
    w, h = size_in / FIG_W, size_in / FIG_H
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def _ring(fig, cx, cy, pct, color, label, value_txt, size_in=1.15):
    ax = _square_axes(fig, cx, cy, size_in)
    ax.add_patch(Wedge((0, 0), 0.95, 0, 360, width=0.2, color="#2E3D55"))
    if pct is not None:
        ax.add_patch(Wedge((0, 0), 0.95, 90 - 360 * min(pct, 100) / 100, 90, width=0.2, color=color))
    ax.text(0, 0, value_txt, ha="center", va="center", color=TXT, fontsize=14, fontweight="bold")
    fig.text(cx, cy - size_in / FIG_H / 2 - 0.012, label, ha="center", va="top", color=MUTED, fontsize=8.5)


def render_card(p: dict, prof: dict, out: Path, team_name: str, team_bgr, photo_bgr=None,
                segment: str = "", approx_kin: bool = False) -> Path:
    """p: oyuncu özeti (Analyzer.player_summaries), prof: profiling.build_profiles çıktısı"""
    tcol = _rgb(team_bgr)
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=100)
    bg = fig.add_axes([0, 0, 1, 1])
    grad = np.linspace(0, 1, 256)[:, None, None] * (BG_BOT - BG_TOP) + BG_TOP
    bg.imshow(np.repeat(grad, 2, 1), aspect="auto", extent=[0, 1, 0, 1])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.set_axis_off()
    bg.add_patch(plt.Rectangle((0, 0.975), 1, 0.025, color=tcol))
    bg.add_patch(plt.Rectangle((0, 0), 0.215, 0.975, color=tcol, alpha=0.10))

    # --- fotoğraf (en-boy oranı korunur)
    ph = fig.add_axes([0.015, 0.20, 0.185, 0.66])
    ph.set_axis_off()
    if photo_bgr is not None and photo_bgr.size:
        img = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        sc = max(1.0, 420 / h)
        img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_CUBIC)
        ph.imshow(img, aspect="equal", interpolation="lanczos")
    else:
        ph.text(0.5, 0.5, "fotoğraf yok", color=MUTED, ha="center", va="center", transform=ph.transAxes)
    bg.text(0.1075, 0.12, p["display_id"], color=tcol, fontsize=54, fontweight="bold", ha="center", va="center")
    bg.text(0.1075, 0.05, team_name.upper(), color=TXT, fontsize=13, fontweight="bold", ha="center", va="center")

    # --- başlık
    bg.text(0.235, 0.905, p["name"].upper(), color=TXT, fontsize=32, fontweight="bold", va="center")
    bg.text(0.235, 0.845, f"{team_name}   |   Profil: {prof['category']}   |   {prof['minutes']:.1f} dk görüntüde   |   "
                          f"{prof['n_events']} aksiyon", color=MUTED, fontsize=12.5, va="center")

    # --- GENEL puan + halkalar (kare eksenler)
    ga = _square_axes(fig, 0.295, 0.66, 1.55)
    ink = "#0B1522" if (0.299 * tcol[0] + 0.587 * tcol[1] + 0.114 * tcol[2]) > 0.6 else "white"
    ga.add_patch(Circle((0, 0), 0.95, color=tcol))
    ga.add_patch(Circle((0, 0), 0.95, fill=False, ec="white", lw=2))
    ga.text(0, 0.12, "—" if prof["overall"] is None else str(prof["overall"]), color=ink, fontsize=34,
            fontweight="bold", ha="center", va="center")
    ga.text(0, -0.45, "GENEL", color=ink, fontsize=10, fontweight="bold", ha="center", va="center")
    acc = prof["raw"].get("ISABET")
    duel_w, duel_l = p["events"].get("TACKLE", 0), p["events"].get("DUEL_LOST", 0)
    duel = duel_w / (duel_w + duel_l) * 100 if duel_w + duel_l else None
    share = p.get("team_poss_share")
    _ring(fig, 0.255, 0.43, None if acc is None else acc * 100, ACCENT, "PAS İSABETİ",
          "—" if acc is None else f"%{acc * 100:.0f}")
    _ring(fig, 0.335, 0.43, share, "#F72585", "TAKIM İÇİ TOP PAYI", "—" if share is None else f"%{share:.0f}")
    _ring(fig, 0.295, 0.245, duel, "#B5E48C", "İKİLİ KAZANMA", "—" if duel is None else f"%{duel:.0f}")

    # --- radar
    rd = fig.add_axes([0.425, 0.29, 0.23, 0.41], projection="polar")
    vals = [prof["attributes"][a] for a in ATTRS]
    v = [0 if x is None else x for x in vals]
    ang = np.linspace(0, 2 * np.pi, len(ATTRS), endpoint=False).tolist()
    rd.set_facecolor((1, 1, 1, 0.03))
    rd.plot(ang + ang[:1], v + v[:1], color=tcol, lw=2.5)
    rd.fill(ang + ang[:1], v + v[:1], color=tcol, alpha=0.35)
    rd.scatter(ang, v, color="white", s=18, zorder=5)
    rd.set_ylim(0, 100)
    rd.set_yticks([25, 50, 75, 100])
    rd.set_yticklabels(["25", "50", "75", ""], color=MUTED, fontsize=7)
    rd.set_xticks(ang)
    rd.set_xticklabels([f"{ATTR_TR[a]}\n{'—' if x is None else x}" for a, x in zip(ATTRS, vals)], color=TXT, fontsize=10)
    rd.tick_params(axis="x", pad=10)
    rd.grid(color=GRID, lw=0.8)
    rd.spines["polar"].set_color(GRID)
    bg.text(0.54, 0.80, "ÖZELLİK PUANLARI (yüzdelik, aynı kesitteki oyunculara göre)", color=MUTED, fontsize=9,
            ha="center")

    # --- CIES pusulası
    cp = _square_axes(fig, 0.835, 0.53, 3.3)
    cp.set_xlim(-1.3, 1.3)
    cp.set_ylim(-1.3, 1.3)
    cp.add_patch(Circle((0, 0), 1, fill=False, ec=GRID, lw=1))
    cp.add_patch(Circle((0, 0), 0.5, fill=False, ec="#2E3D55", lw=0.8, ls="--"))
    for (x, y, t, c, r) in ((-1.2, 0, "HÜCUM", "#F28482", 90), (1.2, 0, "SAVUNMA", "#4EA8DE", -90),
                            (0, 1.18, "MERKEZ", "#B5E48C", 0), (0, -1.18, "KANAT", "#F6BD60", 0)):
        cp.text(x, y, t, color=c, fontsize=10, fontweight="bold", ha="center", va="center", rotation=r)
    cp.plot([-1, 1], [0, 0], color=GRID, lw=0.8)
    cp.plot([0, 0], [-1, 1], color=GRID, lw=0.8)
    for a, spec in PROFILE_CFG["actions"].items():
        u = unit(spec["angle"]) * 0.95
        cp.plot([0, u[0]], [0, u[1]], color="#2E3D55", lw=1)
    vec = np.array(prof["vector"])
    nv = float(np.linalg.norm(vec))
    if nv > 0:
        tip = vec / nv * min(1.0, 0.3 + nv)
        cp.annotate("", xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=tcol, lw=4, mutation_scale=24))
    cp.plot(0, 0, "o", color="white", ms=5)
    bg.text(0.835, 0.80, "TEKNİK PROFİL (CIES yöntemi)", color=MUTED, fontsize=9, ha="center")
    bg.text(0.835, 0.30, prof["category"], color=TXT, fontsize=15, fontweight="bold", ha="center")

    # --- istatistik tablosu
    kin = p.get("kin") or {}
    star = "*" if approx_kin else ""
    has_kin = kin.get("distance_m", 0) > 0
    ev = p["events"]
    cols = [("KOŞU MESAFESİ", f"{kin.get('distance_m', 0):.0f} m{star}" if has_kin else "—"),
            ("MAKS. HIZ", f"{kin.get('max_speed_kmh', 0):.1f} km/sa{star}" if has_kin else "—"),
            ("ORT. HIZ", f"{kin.get('avg_speed_kmh', 0):.1f} km/sa{star}" if has_kin else "—"),
            ("PAS (başarılı/top.)", f"{ev.get('PASS', 0)} / {ev.get('PASS', 0) + ev.get('PASS_FAIL', 0)}"),
            ("TOPLA SÜRE", f"{p.get('poss_s', 0):.1f} sn"),
            ("TOP KAZANMA", f"{ev.get('TACKLE', 0) + ev.get('INTERCEPT', 0) + ev.get('PICKUP', 0)}"),
            ("ŞUT / KİLİT PAS", f"{ev.get('SHOT', 0)} / {ev.get('KEY_PASS', 0)}"),
            ("ÇALIM / TAŞIMA", f"{ev.get('DRIBBLE', 0)} / {ev.get('CARRY', 0)}")]
    x0, x1 = 0.395, 0.985
    w = (x1 - x0) / len(cols)
    bg.add_patch(FancyBboxPatch((x0 - 0.012, 0.065), x1 - x0 + 0.012, 0.17, boxstyle="round,pad=0,rounding_size=0.01",
                                color="#0B1522", alpha=0.75))
    bg.text(x0, 0.205, "BU KESİT", color=ACCENT, fontsize=11, fontweight="bold")
    for i, (k, val) in enumerate(cols):
        bg.text(x0 + i * w, 0.16, k, color=MUTED, fontsize=8)
        bg.text(x0 + i * w, 0.10, val, color=TXT, fontsize=12.5, fontweight="bold")
    foot = f"{segment}   ·   Bilgisayarlı Görü — otomatik video analizi"
    if approx_kin:
        foot += "   ·   * saha homografisi yok: kamera telafili yaklaşık ölçüm"
    bg.text(x0, 0.03, foot, color=MUTED, fontsize=8.5)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=BG_TOP)
    plt.close(fig)
    return out


def render_profile_map(profiles: dict[int, dict], players: dict[int, dict], team_names, team_bgr, out: Path) -> Path:
    """Tüm oyuncuları CIES düzleminde gösterir (kategorilere ayrım görseli)."""
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    fig.patch.set_facecolor(BG_TOP)
    ax.set_facecolor(BG_TOP)
    lim = 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    for c in PROFILE_CFG["categories"]:
        a = math.radians(c["center"] + 22.5)
        ax.plot([0, math.cos(a) * lim * 1.5], [0, math.sin(a) * lim * 1.5], color="#2E3D55", lw=0.8)
        u = unit(c["center"]) * 0.93
        ax.text(u[0], u[1], c["name"], color=MUTED, fontsize=8, ha="center", va="center")
    ax.add_patch(Circle((0, 0), PROFILE_CFG["min_magnitude"], color="#2E3D55", alpha=0.6))
    ax.text(0, 0, "Dengeli", color=MUTED, fontsize=7, ha="center", va="center")
    for t, pr in profiles.items():
        team = players[t]["team"]
        col = _rgb(team_bgr[team]) if team in (0, 1) else (0.7, 0.7, 0.7)
        v = np.array(pr["vector"])
        ax.scatter(v[0], v[1], s=60 + 30 * pr["n_events"] ** 0.5, color=col, edgecolor="white", lw=1, zorder=4)
        ax.text(v[0], v[1] + 0.045, players[t]["display_id"], color=TXT, fontsize=8, ha="center", zorder=5)
    ax.set_xlabel("← HÜCUM            SAVUNMA →", color=TXT)
    ax.set_ylabel("← KANAT            MERKEZ →", color=TXT)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    handles = [plt.Line2D([], [], marker="o", ls="", color=_rgb(team_bgr[i]), label=team_names[i]) for i in (0, 1)]
    ax.legend(handles=handles, facecolor=tuple(BG_BOT), labelcolor=TXT, loc="lower right")
    ax.set_title("Oyuncu Teknik Profil Haritası", color=TXT, fontsize=14, fontweight="bold")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG_TOP, bbox_inches="tight")
    plt.close(fig)
    return out
