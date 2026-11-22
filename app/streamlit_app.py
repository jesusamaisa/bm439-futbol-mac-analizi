"""Bilgisayarlı Görü — Futbol analiz web arayüzü.

Çalıştırma (proje kökünden):
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if os.environ.get("BGORU_GPU_FRAC"):  # GPU başka bir işle (ör. eğitim) paylaşılırken bellek payını sınırla
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ["BGORU_GPU_FRAC"]))

from bgoru.card import render_card, render_profile_map  # noqa: E402
from bgoru.config import FINAL_CONF_BALL, FINAL_CONF_PLAYER, FINAL_DET_WEIGHTS  # noqa: E402
from bgoru.pipeline import Analyzer, AnalyzerConfig, to_h264  # noqa: E402
from bgoru.possession import EVENT_TR  # noqa: E402
from bgoru.profiling import build_profiles  # noqa: E402

st.set_page_config(page_title="Bilgisayarlı Görü — Futbol Analizi", page_icon="⚽", layout="wide")


def _img_kw() -> dict:
    """st.image tam genişlik parametresi: Streamlit >= 1.40 'use_container_width', öncesi 'use_column_width'."""
    import inspect

    return {"use_container_width": True} if "use_container_width" in inspect.signature(st.image).parameters         else {"use_column_width": True}


IMG_KW = _img_kw()

DEMOS = {
    "Taktik kamera demo 1 (İsviçre – Fransa)": (ROOT / "demo" / "input" / "taktik_demo1.mp4", ["İsviçre", "Fransa"]),
    "Taktik kamera demo 2": (ROOT / "demo" / "input" / "taktik_demo2.mp4", ["Takım A", "Takım B"]),
}
WEIGHT_OPTIONS = {
    "Son sistem: YOLOv8s 1280 px + ek veri (G1, doğrulamada en iyi)": (FINAL_DET_WEIGHTS, 1280),
    "YOLOv8s 1280 px, yalnızca SoccerNet-Tracking (A1)": ("bgoru_A1_ft1280.pt", 1280),
    "YOLOv8s 1280 + top kopyala-yapıştır (E3)": ("bgoru_E3_ft1280_cp.pt", 1280),
    "Alternatif: YOLOv8s 640 ince ayar (E2)": ("bgoru_E2_ft640.pt", 640),
    "Referans: Hmzbo YOLOv8L (taktik kamera verisi)": ("hmzbo_players_yolov8l.pt", 640),
    "Temel: COCO YOLOv8s (ince ayarsız)": ("yolov8s.pt", 1280),
}


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def bgr_to_hex(c):
    return "#{:02x}{:02x}{:02x}".format(int(c[2]), int(c[1]), int(c[0]))


# ------------------------------------------------------------------ kenar çubuğu
st.sidebar.title("⚽ Bilgisayarlı Görü")
st.sidebar.caption("Futbol videosu analizi · anlık top hakimiyeti · pas sıklığı · oyuncu profili ve kartı")

src = st.sidebar.radio("Video kaynağı", list(DEMOS) + ["Kendi videom (yükle)"])
if src == "Kendi videom (yükle)":
    up = st.sidebar.file_uploader("MP4 / AVI / MOV", type=["mp4", "avi", "mov", "mkv"])
    video_path, default_names = None, ["Takım A", "Takım B"]
    if up is not None:
        tmp = Path(tempfile.gettempdir()) / f"bgoru_{up.name}"
        tmp.write_bytes(up.read())
        video_path = tmp
else:
    video_path, default_names = DEMOS[src]

mdl = st.sidebar.expander("Model ve eşikler")
wsel = mdl.selectbox("Oyuncu/top dedektörü", list(WEIGHT_OPTIONS))
conf_ball = mdl.slider("Top güven eşiği", 0.01, 0.6, FINAL_CONF_BALL, 0.01)
conf_player = mdl.slider("Oyuncu güven eşiği", 0.05, 0.8, FINAL_CONF_PLAYER, 0.05)
use_h = mdl.checkbox("Saha homografisi (taktik harita, metre ölçümü)", True)
kp_every = mdl.slider("Anahtar nokta tespiti her N karede", 1, 10, 1)
dist_thr = mdl.slider("Top sahipliği mesafe eşiği (× oyuncu boyu)", 0.2, 1.0, 0.45, 0.05)
team_method = mdl.selectbox("Takım ataması", ["proposed_local", "baseline", "alternative"],
                            format_func={"proposed_local": "Önerilen (LAB + yerel saha maskesi + iz oylaması)",
                                         "baseline": "Temel (ortalama RGB + K-Means)",
                                         "alternative": "Alternatif (üst yarı K-Means)"}.get)

viz = st.sidebar.expander("Görselleştirme")
show_map = viz.checkbox("Taktik harita", True)
show_hud = viz.checkbox("Canlı istatistik paneli", True)
show_kp = viz.checkbox("Saha anahtar noktaları", False)
show_pal = viz.checkbox("Renk paleti", False)
max_s = viz.slider("En fazla analiz süresi (sn, 0 = tümü)", 0, 300, 0, 10)
save_video = viz.checkbox("Çıktı videosunu kaydet", True)
show_every = viz.slider("Ekrana her N karede bir çiz (hız için)", 1, 10, 2)
start_s = viz.number_input("Başlangıç saniyesi", 0.0, 3600.0, 0.0, 5.0)

# ------------------------------------------------------------------ takımlar ve renkler
tab_set = st.sidebar.expander("Takımlar ve renkler", expanded=True)
color_mode = tab_set.radio("Takım renkleri", ["Otomatik algıla (önerilen)", "Elle seç (videodaki yöntem)"],
                           help="Otomatik: ilk karelerdeki oyuncuların forma renkleri (yerel saha rengi maskelenerek) LAB uzayında 2 kümeye ayrılır.")
manual_colors = color_mode.startswith("Elle")
c1 = c2 = None
if manual_colors:
    n1 = tab_set.text_input("Takım 1 adı", default_names[0])
    c1 = tab_set.color_picker("Takım 1 forma rengi", "#1f3a93")
    n2 = tab_set.text_input("Takım 2 adı", default_names[1])
    c2 = tab_set.color_picker("Takım 2 forma rengi", "#e5e5e5")
else:
    key = f"tc_{video_path}_{start_s}"
    if tab_set.button("🎨 Takım renklerini algıla", disabled=video_path is None):
        from bgoru.pipeline import detect_team_colors

        w_, sz_ = WEIGHT_OPTIONS[wsel]
        with st.spinner("İlk karelerden forma renkleri öğreniliyor..."):
            st.session_state[key] = detect_team_colors(video_path, AnalyzerConfig(det_weights=w_, det_imgsz=sz_),
                                                       start_s=start_s)
    tc = st.session_state.get(key)
    if tc:
        swap = tab_set.checkbox("↔ Renk–takım eşleşmesini değiştir", False)
        order = [1, 0] if swap else [0, 1]
        names_in = []
        for slot, k in enumerate(order):
            c = tc["colors_bgr"][k]
            tab_set.markdown(f"<div style='display:flex;align-items:center;gap:8px'><div style='width:26px;height:26px;"
                             f"border-radius:6px;border:1px solid #888;background:{bgr_to_hex(c)}'></div>"
                             f"<b>Takım {slot + 1} rengi</b></div>", unsafe_allow_html=True)
            ex = tc["examples"].get(k) or tc["examples"].get(str(k)) or []
            if ex:
                tab_set.image([cv2.cvtColor(cv2.resize(e, (40, 80)), cv2.COLOR_BGR2RGB) for e in ex[:6]], width=40)
            names_in.append(tab_set.text_input(f"Takım {slot + 1} adı", default_names[slot], key=f"nm{slot}"))
        n1, n2 = names_in
        st.session_state["team_centers"] = [tc["centers"][k] for k in order]
        st.session_state["team_spread"] = tc["spread"]
    else:
        tab_set.caption("Algılama yapılmazsa renkler analiz sırasında otomatik öğrenilir; takım sırası açık renk → koyu renktir.")
        n1 = tab_set.text_input("Takım 1 adı (açık renkli forma)", default_names[0])
        n2 = tab_set.text_input("Takım 2 adı (koyu renkli forma)", default_names[1])
        st.session_state["team_centers"] = None

# ------------------------------------------------------------------ ana sayfa
st.title("Bilgisayarlı Görü — Futbol Maç Analizi")
tabs = st.tabs(["🎥 Canlı analiz", "📊 Takım istatistikleri", "🧭 Oyuncu profilleri", "🃏 Karakter kartları", "ℹ️ Nasıl çalışır"])

with tabs[4]:
    st.markdown("""
**Akış:** YOLOv8 ile oyuncu/hakem/top tespiti → ByteTrack ile kimlik takibi → LAB renk + çim maskesi + iz oylaması ile
takım ataması → saha anahtar noktalarından homografi (metre cinsinden taktik harita) → **anlık top hakimiyeti** ve
**pas / araya girme / top kapma / top kazanma / top taşıma / çalım / şut / orta** olayları → CIES yöntemiyle **teknik
profil** → **karakter kartı**.

* **Top hakimiyeti %:** topa en yakın oyuncu (ayak bölgesine uzaklık / oyuncu boyu < eşik), 0,12 sn onay süresiyle
  top sahibi olur; takım kontrolü son top sahibinin takımındadır.
* **Pas sıklığı:** başarılı paslar / geçen dakika (kümülatif) ve son 60 sn penceresi.
* **Profil:** her aksiyonun Hücum↔Savunma ve Merkez↔Kanat eksenlerindeki vektörlerinin, kesit ortalamasına göre
  göreli sıklıklarla ağırlıklı toplamı.
* **Kart puanları:** aynı kesitteki oyunculara göre yüzdelik (0–99); GENEL = profile göre ağırlıklı ortalama.
""")

if "result" not in st.session_state:
    st.session_state.result = None

with tabs[0]:
    colA, colB = st.columns([3, 1.25])
    frame_ph = colA.empty()
    prog = colA.progress(0.0, text="Hazır")
    with colB:
        st.subheader("Canlı istatistikler")
        m_poss = st.empty()
        m_win = st.empty()
        m_pass = st.empty()
        m_ppm = st.empty()
        chart_ph = st.empty()
        ev_ph = st.empty()
    start = st.button("▶ Analizi başlat", type="primary", disabled=video_path is None)
    stop = st.button("■ Durdur (o ana kadarki sonuçlar kaydedilir)")
    if stop:
        st.session_state.stop = True
    last = st.session_state.get("result")
    if last is not None and not start:  # durdurma/yeniden çizim sonrası son çalıştırmanın durumu
        ls = last["summary"]
        st.info(f"Son analiz {'durduruldu' if ls.get('interrupted') else 'tamamlandı'}: {ls['frames']} kare "
                f"({ls['final']['elapsed_s']:.0f} sn) işlendi, {len(ls['cards'])} kart üretildi · çıktı: "
                f"`{Path(last['out_dir']).relative_to(ROOT)}`. Sonuçlar diğer sekmelerdedir.")

    if start and video_path is not None:
        st.session_state.stop = False
        w, imgsz = WEIGHT_OPTIONS[wsel]
        cfg = AnalyzerConfig(det_weights=w, det_imgsz=imgsz, conf_ball=conf_ball, conf_player=conf_player,
                             use_homography=use_h, kp_every=kp_every, team_method=team_method,
                             team_colors_bgr=[hex_to_bgr(c1), hex_to_bgr(c2)] if manual_colors else None,
                             team_names=[n1, n2], dist_thr=dist_thr, show_hud=show_hud, show_map=show_map,
                             show_keypoints=show_kp, show_palette=show_pal,
                             team_centers=None if manual_colors else st.session_state.get("team_centers"),
                             team_spread=st.session_state.get("team_spread"))
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if start_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        limit = int(max_s * fps) if max_s else total
        with st.spinner("Modeller yükleniyor..."):
            an = Analyzer(cfg, fps)
        out_dir = ROOT / "results" / "app_runs" / time.strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        writer, tmp = None, out_dir / "_tmp.mp4"
        hist = []
        t0 = time.time()
        interrupted = True
        try:
            while an.f < limit and not st.session_state.get("stop"):
                ok, frame = cap.read()
                if not ok:
                    break
                out, snap = an.process(frame)
                if save_video:
                    if writer is None:
                        h = int(out.shape[0] * 1280 / out.shape[1])
                        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, h))
                    writer.write(cv2.resize(out, (1280, int(out.shape[0] * 1280 / out.shape[1]))))
                hist.append({"sn": an.f / fps, n1: snap["possession_pct"][0], n2: snap["possession_pct"][1]})
                if an.f % show_every == 0:
                    frame_ph.image(cv2.cvtColor(cv2.resize(out, (1280, int(out.shape[0] * 1280 / out.shape[1]))),
                                                cv2.COLOR_BGR2RGB), channels="RGB", **IMG_KW)
                    p, wp = snap["possession_pct"], snap["possession_pct_window"]
                    m_poss.metric(f"Top hakimiyeti · {n1} – {n2}", f"%{p[0]:.0f} – %{p[1]:.0f}")
                    m_win.metric("Hakimiyet · son 60 sn", f"%{wp[0]:.0f} – %{wp[1]:.0f}")
                    acc = ["–" if a is None else f"%{a:.0f}" for a in snap["pass_accuracy"]]
                    m_pass.metric(f"Pas · isabet {acc[0]} / {acc[1]}", f"{snap['passes'][0]} – {snap['passes'][1]}")
                    ppm = snap["passes_per_min"]
                    m_ppm.metric("Pas sıklığı (pas/dk)", f"{ppm[0]:.1f} – {ppm[1]:.1f}")
                    if an.f % (show_every * 10) == 0:
                        chart_ph.line_chart(pd.DataFrame(hist).set_index("sn"), height=180)
                    evs = snap["last_events"][::-1]
                    ev_ph.markdown("\n".join(f"- `{e['t']:.1f}s` **#{e['tid']}** "
                                             f"{'→ #' + str(e['to']) + ' ' if 'to' in e and e['type'] in ('PASS', 'PASS_FAIL') else ''}"
                                             f"{EVENT_TR[e['type']]} ({[n1, n2][e['team']] if e['team'] in (0, 1) else '?'})"
                                             for e in evs) or "_henüz olay yok_")
                    el = time.time() - t0
                    prog.progress(min(an.f / limit, 1.0), text=f"Kare {an.f}/{limit} · {an.f / max(el, 1e-6):.1f} FPS")
            interrupted = False
        finally:
            # 'Durdur' veya sayfa etkileşimi çalışmayı kesse bile o ana kadarki sonuçlar kaydedilir
            cap.release()
            if writer is not None:
                writer.release()
            if an.f > 0:
                wall = time.time() - t0  # yalnızca kare işleme (kart üretimi ve video kodlama hariç)
                summary = an.finalize(out_dir, Path(video_path).stem)
                if writer is not None:
                    summary["video"] = str(to_h264(tmp, out_dir / "analiz.mp4"))
                summary["hist"] = hist
                summary["interrupted"] = interrupted
                summary["wall_fps"] = round(an.f / max(wall, 1e-6), 2)
                (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                                      encoding="utf-8")
                st.session_state.result = {"summary": summary, "out_dir": str(out_dir), "an": an}
        prog.progress(1.0, text=f"Bitti · {an.f} kare · çıktı: {out_dir.relative_to(ROOT)}")

res = st.session_state.result
with tabs[1]:
    if res is None:
        st.info("Önce **Canlı analiz** sekmesinden bir analiz çalıştırın.")
    else:
        s = res["summary"]
        f = s["final"]
        names = s["team_names"]
        c = st.columns(4)
        c[0].metric("Top hakimiyeti", f"%{f['possession_pct'][0]:.0f} – %{f['possession_pct'][1]:.0f}")
        c[1].metric("Pas", f"{f['passes'][0]} – {f['passes'][1]}")
        c[2].metric("Pas/dk", f"{f['passes_per_min'][0]:.1f} – {f['passes_per_min'][1]:.1f}")
        acc = ["–" if a is None else f"%{a:.0f}" for a in f["pass_accuracy"]]
        c[3].metric("Pas isabeti", f"{acc[0]} – {acc[1]}")
        st.caption(f"Analiz edilen kesit: {f['elapsed_s']:.0f} sn · saha homografisi kapsamı %{s['pitch_coverage'] * 100:.0f}"
                   f" · ortalama işlem hızı {s.get('wall_fps', '–')} FPS")
        if s.get("hist"):
            st.subheader("Top hakimiyetinin zaman içindeki değişimi (kümülatif)")
            st.line_chart(pd.DataFrame(s["hist"]).set_index("sn"))
        ev = pd.read_csv(Path(res["out_dir"]) / "events.csv")
        if len(ev):
            ev["takım"] = ev["team"].map({0: names[0], 1: names[1]})
            st.subheader("Olay dağılımı")
            st.bar_chart(ev.pivot_table(index="name", columns="takım", values="frame", aggfunc="count").fillna(0))
            st.subheader("Olay kaydı")
            st.dataframe(ev[["time_s", "name", "player", "to", "takım"]], use_container_width=True, height=300)
        if s.get("video") and Path(s["video"]).exists():
            st.subheader("Açıklamalı çıktı videosu")
            st.video(s["video"])

with tabs[2]:
    if res is None:
        st.info("Önce analiz çalıştırın.")
    else:
        s = res["summary"]
        pm = Path(res["out_dir"]) / "profil_haritasi.png"
        cA, cB = st.columns([1.1, 1])
        if pm.exists():
            cA.image(str(pm), **IMG_KW)
        rows = []
        for tid, pr in s["profiles"].items():
            p = s["players"][str(tid)] if str(tid) in s["players"] else s["players"][tid]
            rows.append({"Oyuncu": p["display_id"], "Takım": s["team_names"][p["team"]] if p["team"] in (0, 1) else "?",
                         "Kategori": pr["category"], "GENEL": pr["overall"], "Hücum eğilimi": pr["offensive"],
                         "Merkez eğilimi": pr["centrality"], "Aksiyon": pr["n_events"], "Dakika": pr["minutes"]})
        if rows:
            df = pd.DataFrame(rows).sort_values("GENEL", ascending=False)
            cB.dataframe(df, use_container_width=True, height=520)
            st.subheader("Kategori dağılımı")
            st.bar_chart(df.groupby(["Kategori", "Takım"]).size().unstack(fill_value=0))

with tabs[3]:
    if res is None:
        st.info("Önce analiz çalıştırın.")
    else:
        s, an = res["summary"], res["an"]
        cards = s["cards"]
        if not cards:
            st.warning("Kart üretmek için yeterli süre görünen oyuncu yok (en az 9 sn).")
        else:
            st.caption("Oyuncu kimliği yayın görüntüsünden bilinemediği için iz numarasıyla gösterilir; aşağıdan isim verip kartı yenileyebilirsiniz.")
            ids = sorted(cards, key=lambda t: -(s["profiles"][t]["overall"] or 0))
            sel = st.selectbox("Oyuncu", ids, format_func=lambda t: f"#{t} — {s['profiles'][t]['category']} (GENEL {s['profiles'][t]['overall']})")
            new_name = st.text_input("Oyuncu adı", s["players"][sel]["name"])
            if st.button("Kartı bu isimle yenile"):
                p = dict(s["players"][sel]); p["name"] = new_name
                t = p["team"]
                render_card(p, s["profiles"][sel], Path(cards[sel]), s["team_names"][t] if t in (0, 1) else "?",
                            tuple(s["team_colors_bgr"][t]) if t in (0, 1) else (180, 180, 180),
                            an.players[sel]["photo"], f"{s['segment']} · {s['final']['elapsed_s'] / 60:.1f} dk kesit",
                            approx_kin=s["kinematics_approx"])
                s["players"][sel]["name"] = new_name
            st.image(cards[sel], **IMG_KW)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for t, pth in cards.items():
                    z.write(pth, Path(pth).name)
                z.write(Path(res["out_dir"]) / "summary.json", "summary.json")
                z.write(Path(res["out_dir"]) / "events.csv", "events.csv")
            st.download_button("Tüm kartları ve verileri indir (ZIP)", buf.getvalue(), "bilgisayarli_goru_kartlar.zip")
            st.subheader("Tüm kartlar")
            cols = st.columns(2)
            for i, t in enumerate(ids):
                cols[i % 2].image(cards[t], **IMG_KW)
