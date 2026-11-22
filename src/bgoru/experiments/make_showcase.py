"""3–5 dakikalık uygulama gösterim videosu (demo/gosterim_videosu.mp4).

Akış: başlık → girdi → model çıkarımı + anlık istatistikler (taktik demo 1–2) → sonuçların yorumu
(profil haritası, karakter kartları, test sonuçları) → başarısız örnek → kapanış.
Türkçe karakterler için yazılar PIL + DejaVuSans ile çizilir. Sesli anlatım içermez.

Kullanım: python -m bgoru.experiments.make_showcase
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from bgoru.config import FINAL_DETECTOR, ROOT, RESULTS
from bgoru.pipeline import to_h264

W, H, FPS = 1280, 720, 25
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
FONT_B = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
BG = (42, 27, 13)  # BGR koyu lacivert
DEMO = ROOT / "demo"


def font(sz, bold=False):
    return ImageFont.truetype(str(FONT_B if bold else FONT), sz)


def draw_text(img, lines, start_y=None, center=True):
    """lines: [(metin, boyut, kalın, renk_rgb)]"""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    hs = [font(s, b).getbbox(t)[3] + 14 for t, s, b, _ in lines]
    y = start_y if start_y is not None else (H - sum(hs)) // 2
    for (t, s, b, c), h in zip(lines, hs):
        f = font(s, b)
        w = d.textlength(t, font=f)
        d.text(((W - w) / 2 if center else 60, y), t, font=f, fill=c)
        y += h
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def card(lines, seconds):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.rectangle(img, (0, 0), (W, 10), (235, 140, 52), -1)
    img = draw_text(img, lines)
    return [img] * int(seconds * FPS)


def caption(frame, text):
    over = frame.copy()
    cv2.rectangle(over, (0, H - 70), (W, H), (0, 0, 0), -1)
    frame = cv2.addWeighted(over, 0.6, frame, 0.4, 0)
    return draw_text(frame, [(text, 24, False, (255, 255, 255))], start_y=H - 58)


def video_frames(path, cap_text=None, max_s=None, fps_out=FPS):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    frames, t = [], 0.0
    step = fps / fps_out
    i, nxt = 0, 0.0
    cap_img = None
    while True:
        ok, f = cap.read()
        if not ok or (max_s and t > max_s):
            break
        while i >= nxt:  # basit kare hızı dönüşümü
            fr = cv2.resize(f, (W, H))
            if cap_text:
                fr = caption(fr, cap_text)
            frames.append(fr)
            nxt += step
            t = len(frames) / fps_out
        i += 1
    cap.release()
    return frames


def image_slide(path, seconds, title=None):
    img = cv2.imread(str(path))
    if img is None:
        return []
    canvas = np.full((H, W, 3), BG, np.uint8)
    top = 70 if title else 20
    s = min((W - 40) / img.shape[1], (H - top - 20) / img.shape[0])
    im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    y0, x0 = top + (H - top - 20 - im.shape[0]) // 2, (W - im.shape[1]) // 2
    canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    if title:
        canvas = draw_text(canvas, [(title, 30, True, (255, 255, 255))], start_y=18)
    return [canvas] * int(seconds * FPS)


def main():
    seq, marks = [], []
    mark = lambda k: marks.append({"id": k, "start_s": round(len(seq) / FPS, 2)})  # anlatım zamanlaması için
    mark("baslik")
    seq += card([("Bilgisayarlı Görü", 64, True, (255, 255, 255)),
                 ("Futbol videolarında anlık top hakimiyeti, pas sıklığı,", 30, False, (200, 220, 255)),
                 ("oyuncu profili ve karakter kartı", 30, False, (200, 220, 255)),
                 ("BM439 Bilgisayarlı Görme — Proje Gösterimi", 22, False, (160, 175, 200))], 7)
    mark("girdi_kart")
    seq += card([("1 · Girdi", 46, True, (255, 255, 255)),
                 ("Tek kameralı maç videosu (taktik veya yayın kamerası)", 28, False, (220, 230, 245)),
                 ("Arayüzde demo seçilir veya kendi MP4 dosyanız yüklenir;", 24, False, (190, 200, 220)),
                 ("ilk karelerden forma renkleri öğrenilip takım adları eşlenir.", 24, False, (190, 200, 220))], 6)
    mark("girdi_video")
    seq += video_frames(DEMO / "input" / "taktik_demo1.mp4", "Girdi: ham taktik kamera görüntüsü", max_s=6)
    mark("cikarim_kart")
    seq += card([("2 · Model çıkarımı ve anlık istatistikler", 42, True, (255, 255, 255)),
                 ("YOLOv8s (1280 px) → ByteTrack → takım ataması → saha homografisi", 26, False, (220, 230, 245)),
                 ("→ top sahipliği ve olaylar → canlı panel + taktik harita", 26, False, (220, 230, 245))], 6)
    mark("demo1")
    seq += video_frames(DEMO / "basarili_taktik_demo1.mp4",
                        "Sol üst: kümülatif ve son 60 sn top hakimiyeti, pas/dk, isabet, olay akışı")
    mark("demo2")
    seq += video_frames(DEMO / "basarili_taktik_demo2.mp4",
                        "Sağ alt: metre ölçekli taktik harita ve top izi · kırmızı üçgen = top sahibi")
    mark("yorum_kart")
    seq += card([("3 · Sonuçların yorumu", 46, True, (255, 255, 255)),
                 ("Oyuncular CIES yöntemiyle teknik profillere ayrılır;", 28, False, (220, 230, 245)),
                 ("her oyuncu için karakter kartı üretilir.", 28, False, (220, 230, 245))], 5)
    kart = DEMO / "kartlar"
    mark("profil_haritasi")
    seq += image_slide(kart / "taktik_demo2_profil_haritasi.png", 7, "Profil haritası: Hücum↔Savunma, Merkez↔Kanat")
    for p in sorted(kart.glob("taktik_demo2_kart_*.png"))[:3]:
        mark(f"kart_{p.stem.split('_')[-1]}")
        seq += image_slide(p, 6, "Karakter kartı (yüzdelik puanlar, aynı kesitteki oyunculara göre)")
    fig = RESULTS / "figures"
    mark("fig_tespit")
    seq += image_slide(fig / "fig_detection_results.png", 9, "Test sonuçları: top AP@0.5 ve top boyutuna göre recall")
    mark("fig_karsilastirma")
    seq += image_slide(fig / "fig_compare_models.png", 8, "Aynı test karelerinde modellerin top tespitleri")
    mark("fig_hakimiyet")
    seq += image_slide(fig / "fig_possession.png", 8, "Hakimiyet % hatası ve pas olayı F1 (vekil referansa göre)")
    mark("fig_dayaniklilik")
    seq += image_slide(fig / "fig_robustness.png", 8, "Zorlu koşullar: gece, bulanıklık, JPEG, uzak kamera")
    mark("basarisiz_kart")
    seq += card([("4 · Başarısız örnek", 46, True, (255, 255, 255)),
                 ("Yayın kamerası, kalabalık ceza sahası: saha noktaları bulunamaz,", 26, False, (220, 230, 245)),
                 ("top oyuncu kalabalığında kaybolur, sahiplik yanlış oyuncuya geçebilir.", 26, False, (220, 230, 245))], 6)
    mark("basarisiz_video")
    seq += video_frames(DEMO / "basarisiz_ornek.mp4",
                        "Taktik harita yok (homografi bulunamadı); top kayboldukça sahiplik ve paslar hatalanıyor")
    mark("hata_ornekleri")
    seq += image_slide(RESULTS / "failures" / "contact_sheet_1.jpg", 8,
                       "İncelenen başarısız örnekler: etiket ötelemesi, bulanıklık, elde tutulan top, örtülme")
    det = pd.read_csv(RESULTS / "detection" / "detection_results.csv").set_index("config")["ball_AP50"]
    tr = lambda x: f"{x:.3f}".replace(".", ",")
    mark("ozet")
    seq += card([("Özet", 46, True, (255, 255, 255)),
                 ("1280 px ince ayar ve 3 maçlık ek eğitim verisi top tespitini iyileştirir:", 26, False,
                  (220, 230, 245)),
                 (f"top AP@0,5  {tr(det['E1'])} (COCO) → {tr(det['A1_ft1280'])} (A1) → "
                  f"{tr(det[FINAL_DETECTOR])} (G1, son sistem)", 26, True, (255, 255, 255)),
                 ("Top kopyala-yapıştır artırımı ise ablation'da olumsuz sonuç verdi", 24, False, (220, 230, 245)),
                 (f"(E3 {tr(det['E3_ft1280_cp'])} < A1); tüm sayılar raporda ve experiments/registry.csv içindedir.",
                  22, False, (190, 200, 220)),
                 ("Bilgisayarlı Görü", 30, True, (235, 140, 52))], 7)
    out = DEMO / "_gosterim_tmp.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in seq:
        vw.write(f)
    vw.release()
    final = to_h264(out, DEMO / "gosterim_videosu.mp4")
    (RESULTS / "demo_runs").mkdir(parents=True, exist_ok=True)
    (RESULTS / "demo_runs" / "gosterim_zaman_cizelgesi.json").write_text(
        json.dumps({"fps": FPS, "duration_s": round(len(seq) / FPS, 2), "segments": marks}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[✓] {final}  süre = {len(seq) / FPS / 60:.2f} dk")


if __name__ == "__main__":
    main()
