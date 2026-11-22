"""Gösterim çıktılarını üretir (demo/ klasörü).

* başarılı örnek: taktik kamera demo videoları (homografi + taktik harita + kartlar)
* başarısız örnek: SoccerNet yayın görüntüsü sekansı (yoğun test karelerinden video)

Kullanım: python -m bgoru.experiments.make_demo
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

from bgoru.config import ROOT
from bgoru.experiments.seqdata import dense_frames
from bgoru.pipeline import AnalyzerConfig, analyze_video, detect_team_colors

DEMO = ROOT / "demo"


def frames_to_video(seq: str, fps: float = 12.5) -> str:
    out = DEMO / "input" / f"soccernet_{seq}.mp4"
    if out.exists():
        return str(out)
    fr = dense_frames(seq)
    h, w = cv2.imread(str(fr[0][1])).shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _, p in fr:
        vw.write(cv2.imread(str(p)))
    vw.release()
    return str(out)


def save_frame(video, out, at: float = 0.7, width: int = 1280):
    """Rapordaki açıklamalı çıktı karesi: videonun %70'indeki kare, 1280 px genişliğe ölçeklenir."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * at))
    ok, fr = cap.read()
    cap.release()
    if ok:
        h, w = fr.shape[:2]
        cv2.imwrite(str(out), cv2.resize(fr, (width, round(h * width / w)), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])


def run(video, name, team_names, homography=True):
    cfg = AnalyzerConfig(team_names=team_names, use_homography=homography)
    tc = detect_team_colors(video, cfg)
    if tc:
        cfg.team_centers, cfg.team_spread = tc["centers"], tc["spread"]
    out = ROOT / "results" / "demo_runs" / name
    s = analyze_video(video, out, cfg)
    f = s["final"]
    print(f"[✓] {name}: hakimiyet {f['possession_pct']}, pas {f['passes']}, pas/dk {f['passes_per_min']}, "
          f"kart {len(s['cards'])}, homografi kapsamı {s['pitch_coverage']}, {s.get('wall_fps')} FPS")
    return s, out


def main():
    import os

    import torch

    if torch.cuda.is_available() and os.environ.get("BGORU_GPU_FRAC"):  # eğitimle GPU paylaşırken sınır
        torch.cuda.set_per_process_memory_fraction(float(os.environ["BGORU_GPU_FRAC"]))
    (DEMO / "kartlar").mkdir(parents=True, exist_ok=True)
    results = {}
    # takım sırası: TeamAssigner açık renkli formayı 0. takım yapar (demo 1'de İsviçre beyaz)
    for vid, name, teams in ((DEMO / "input" / "taktik_demo1.mp4", "taktik_demo1", ["İsviçre", "Fransa"]),
                             (DEMO / "input" / "taktik_demo2.mp4", "taktik_demo2", ["Takım A", "Takım B"])):
        s, out = run(str(vid), name, teams)
        results[name] = {k: s[k] for k in ("final", "pitch_coverage", "wall_fps", "timing_ms_per_frame")}
        if s.get("video"):
            shutil.copy2(s["video"], DEMO / f"basarili_{name}.mp4")
        for old in (DEMO / "kartlar").glob(f"{name}_kart_*.png"):  # önceki çalıştırmadan kalan kartlar
            old.unlink()
        for p in list(s["cards"].values())[:6]:  # bu çalıştırmanın en yüksek GENEL puanlı 6 kartı
            shutil.copy2(p, DEMO / "kartlar" / f"{name}_{Path(p).name}")
        if (out / "profil_haritasi.png").exists():
            shutil.copy2(out / "profil_haritasi.png", DEMO / "kartlar" / f"{name}_profil_haritasi.png")
        if name == "taktik_demo2" and s.get("video"):
            save_frame(s["video"], ROOT / "results" / "figures" / "demo_frame.jpg")
    seq = "SNMOT-118"
    v = frames_to_video(seq)
    s, out = run(v, f"soccernet_{seq}", ["Sol takım", "Sağ takım"], homography=True)
    results[f"soccernet_{seq}"] = {k: s[k] for k in ("final", "pitch_coverage", "wall_fps", "timing_ms_per_frame")}
    if s.get("video"):
        shutil.copy2(s["video"], DEMO / "basarisiz_ornek.mp4")
    (ROOT / "results" / "demo_runs" / "demo_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
