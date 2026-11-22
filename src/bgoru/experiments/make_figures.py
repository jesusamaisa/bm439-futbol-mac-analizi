"""Rapor şekilleri (results/figures/*.png). Girdisi henüz üretilmemiş şekiller atlanır.

Renk: doğrulanmış kategorik palet sabit sırayla (mavi, turuncu, su yeşili, sarı, menekşe);
tek seriden oluşan çubuklarda nötr gri + önerilen yöntem vurgusu. Değerler doğrudan etiketlenir.
Kullanım: python -m bgoru.experiments.make_figures
"""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from bgoru.config import DATA, EXPERIMENTS, RESULTS  # noqa: E402

FIG = RESULTS / "figures"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4"]
NEUTRAL, HIGHLIGHT = "#9b9a94", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})
NAMES = {"E1": "E1 Temel\n(COCO)", "E2_ft640": "E2 Alternatif\n(640)", "A2_ft640_cp": "A2 Abl.\n(640+KY)",
         "A1_ft1280": "A1 Abl.\n(1280)", "E3_ft1280_cp": "E3 Önerilen\n(1280+KY)", "R": "R Referans\n(Hmzbo L)",
         "G1_ft1280_gsr": "G1 Son\n(+ek veri)"}


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)
    print("  ->", name)


def fig_data():
    root = DATA / "soccernet"
    if not (root / "labels" / "test").exists():
        return
    counts, sizes = {}, []
    for sp in ("train", "val", "test"):
        c = np.zeros(3, int)
        for f in (root / "labels" / sp).glob("*.txt"):
            for line in f.read_text().splitlines():
                if line.strip():
                    p = line.split()
                    c[int(p[0])] += 1
                    if p[0] == "2" and sp == "train":
                        sizes.append(max(float(p[3]) * 1920, float(p[4]) * 1080))
        counts[sp] = c
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(3)
    for k, (sp, lab) in enumerate((("train", "Eğitim"), ("val", "Doğrulama"), ("test", "Test"))):
        b = ax[0].bar(x + (k - 1) * 0.26, counts[sp], 0.24, color=S[k], label=lab)
        ax[0].bar_label(b, fmt="%d", fontsize=7, color=INK2, padding=1)
    ax[0].set_xticks(x, ["Oyuncu", "Hakem", "Top"])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Örnek sayısı (log)")
    ax[0].legend(loc="upper right")
    ax[0].set_title("(a) Sınıf dağılımı", loc="left", fontsize=10)
    ax[1].hist(sizes, bins=np.arange(0, 60, 2), color=S[0], edgecolor="white", linewidth=1)
    ax[1].axvline(np.median(sizes), color=INK, lw=1, ls="--")
    ax[1].text(np.median(sizes) + 1, ax[1].get_ylim()[1] * 0.9, f"medyan {np.median(sizes):.0f} px", fontsize=8, color=INK)
    ax[1].set_xlabel("Top kutusunun uzun kenarı (px, 1920×1080)")
    ax[1].set_ylabel("Eğitim örneği")
    ax[1].set_title("(b) Top boyutu dağılımı (eğitim)", loc="left", fontsize=10)
    _save(fig, "fig_data_distribution.png")


def fig_training():
    runs = [("E2_ft640", "E2 (640)"), ("A2_ft640_cp", "A2 (640+KY)"), ("A1_ft1280", "A1 (1280)"), ("E3_ft1280_cp", "E3 (1280+KY)"),
            ("G1_ft1280_gsr", "G1 (A1+ek veri)")]
    have = [(r, l) for r, l in runs if (EXPERIMENTS / "runs" / r / "results.csv").exists()]
    if not have:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ends = []
    for k, (r, lab) in enumerate(have):
        df = pd.read_csv(EXPERIMENTS / "runs" / r / "results.csv")
        df.columns = [c.strip() for c in df.columns]
        col = S[[x[0] for x in runs].index(r)]
        ax[0].plot(df["epoch"], df["metrics/mAP50(B)"], color=col, lw=2, label=lab)
        ends.append([float(df["metrics/mAP50(B)"].iloc[-1]), df["epoch"].iloc[-1], lab])
        ax[1].plot(df["epoch"], df["train/box_loss"], color=col, lw=2, label=lab)
        ax[1].plot(df["epoch"], df["val/box_loss"], color=col, lw=1.2, ls="--")
    ends.sort()  # doğrudan etiketler: çakışmasın diye en az 0,014 aralıkla yukarı itilir
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.014)
    for y, x, lab in ends:
        ax[0].text(x + 0.3, y, lab, fontsize=7, color=INK2, va="center")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Doğrulama mAP@0.5 (↑)")
    ax[0].set_title("(a) Doğrulama mAP@0.5", loc="left", fontsize=10)
    ax[0].set_xlim(right=ax[0].get_xlim()[1] + 4)
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Kutu kaybı (↓)")
    ax[1].set_title("(b) Eğitim (düz) / doğrulama (kesikli) kutu kaybı", loc="left", fontsize=10)
    ax[1].legend(fontsize=8)
    _save(fig, "fig_training_curves.png")


def fig_detection():
    p = RESULTS / "detection" / "detection_results.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).set_index("config")
    bs = json.loads((RESULTS / "detection" / "detection_bootstrap.json").read_text(encoding="utf-8"))
    ci = bs["ball_AP50_bootstrap_CI95"]
    order = [k for k in ["E1", "E2_ft640", "A2_ft640_cp", "A1_ft1280", "E3_ft1280_cp", "G1_ft1280_gsr", "R"] if k in df.index]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.45, 1]})
    vals = [df.loc[k, "ball_AP50"] for k in order]
    err = np.array([[vals[i] - ci[k]["lo"], ci[k]["hi"] - vals[i]] for i, k in enumerate(order)]).T
    cols = [HIGHLIGHT if k == "G1_ft1280_gsr" else S[0] if k == "E3_ft1280_cp" else NEUTRAL for k in order]
    b = ax[0].bar(range(len(order)), vals, color=cols, width=0.6, yerr=err, capsize=3, error_kw={"lw": 1, "ecolor": INK2})
    for i, v in enumerate(vals):
        ax[0].text(i, v + err[1][i] + 0.01, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax[0].set_xticks(range(len(order)), [NAMES[k] for k in order], fontsize=7)
    ax[0].set_ylabel("Top AP@0.5 (↑)")
    ax[0].set_title("(a) Test top AP@0.5, %95 bootstrap GA", loc="left", fontsize=10)
    ks = [k for k in ["E1", "E2_ft640", "E3_ft1280_cp"] if k in df.index]
    buckets = ["<10px", "10-20px", ">=20px"]
    x = np.arange(3)
    for j, k in enumerate(ks):
        v = [df.loc[k, f"ball_R_{bk}"] for bk in buckets]
        bb = ax[1].bar(x + (j - 1) * 0.26, v, 0.24, color=S[j], label=NAMES[k].replace("\n", " "))
        ax[1].bar_label(bb, fmt="%.2f", fontsize=7, color=INK2, padding=1)
    ax[1].set_xticks(x, ["< 10 px", "10–20 px", "≥ 20 px"])
    ax[1].set_ylabel("Top recall (doğrulama eşiğinde, ↑)")
    ax[1].set_title("(b) Top boyutuna göre recall", loc="left", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].set_ylim(0, 1.05)
    _save(fig, "fig_detection_results.png")


def fig_robustness():
    p = RESULTS / "robustness" / "robustness_results.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    conds = ["clean", "night", "blur", "jpeg", "far_half"]
    lab = {"clean": "Temiz", "night": "Gece", "blur": "Hareket\nbulanıklığı", "jpeg": "JPEG q=15", "far_half": "Uzak (×0,5)"}
    models = list(dict.fromkeys(df["model"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    x = np.arange(len(conds))
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        for a, metric in enumerate(("ball_AP50", "player_AP50")):
            v = [df[(df.model == m) & (df.condition == c)][metric].values[0] for c in conds]
            nm = {"E1_coco1280": "E1 COCO", "E1_coco640": "E1 COCO", "E2_ft640": "E2 (640)", "E3_ft1280_cp": "E3 (1280+KY)",
                  "A1_ft1280": "A1 (1280)", "A2_ft640_cp": "A2 (640+KY)", "G1_ft1280_gsr": "G1 (son)"}.get(m, m)
            ax[a].bar(x + (j - (len(models) - 1) / 2) * w, v, w * 0.92, color=S[j], label=nm)
    for a, t in enumerate(("(a) Top AP@0.5", "(b) Oyuncu AP@0.5")):
        ax[a].set_xticks(x, [lab[c] for c in conds], fontsize=8)
        ax[a].set_title(t + " — zorlu koşullar", loc="left", fontsize=10)
    ax[0].legend(fontsize=7, ncol=2)
    _save(fig, "fig_robustness.png")


def fig_team():
    p = RESULTS / "team" / "team_results.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    lab = {"T1_baseline": "T1 Temel", "T2_alternative": "T2 Alternatif", "T3_proposed_v1": "T3 Önerilen v1",
           "T5_local_mask": "T5 Önerilen", "T5a_no_vote": "−oylama", "T5b_no_balance": "−dengeli küme",
           "T5c_rgb_not_lab": "LAB→RGB", "T5d_hsv_mask": "sabit HSV maske", "T5e_no_mask": "maskesiz"}
    cfgs = [c for c in lab if c in set(df["config"])]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = np.arange(len(cfgs))
    sets = (("val", "Doğrulama (18 sek.)"), ("dense", "Yoğun test (10 sek.)"), ("sparse", "Seyrek test (49 sek.)"))
    for j, (st, nm) in enumerate(sets):
        d = df[df.set == st].set_index("config")
        v = [d.loc[c, "acc_mean"] if c in d.index else np.nan for c in cfgs]
        ax.bar(x + (j - 1) * 0.27, v, 0.25, color=S[j], label=nm)
    ax.set_xticks(x, [lab[c] for c in cfgs], fontsize=8)
    ax.set_ylabel("Takım doğruluğu (↑)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8, loc="lower left", ncol=3)
    ax.set_title("Takım ataması doğruluğu (GT kutular, kaleciler hariç)", loc="left", fontsize=10)
    _save(fig, "fig_team_accuracy.png")


def fig_possession():
    p = RESULTS / "possession" / "possession_results.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    lab = {"V0_E1": "V0 E1", "V1_E2": "V1 E2", "V2_E3_full": "V2 Tam", "V3_E3_team_baseline": "V3 −takım",
           "V4_E3_no_ballfill": "V4 −doldurma", "V5_E3_no_hysteresis": "V5 −histerezis", "V6_A1_final": "V6 A1",
           "V7_G1_final": "V7 G1 (son)"}
    mets = [("poss_err_pp_mean", "Hakimiyet % hatası (yp, ↓)"), ("control_agreement_mean", "Kare kontrol uyumu (↑)"),
            ("pass_F1_mean", "Pas olayı F1 (↑)")]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    for a, (m, t) in enumerate(mets):
        cols = [HIGHLIGHT if v == "V7_G1_final" else NEUTRAL for v in df["variant"]]
        b = ax[a].bar(range(len(df)), df[m], color=cols, width=0.62)
        ax[a].bar_label(b, fmt="%.2f" if "err" in m else "%.3f", fontsize=7, color=INK2, padding=1)
        ax[a].set_xticks(range(len(df)), [lab.get(v, v) for v in df["variant"]], fontsize=7, rotation=30, ha="right")
        ax[a].set_title(t, loc="left", fontsize=9.5)
    _save(fig, "fig_possession.png")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.6); ax.axis("off")
    boxes = [(0.1, 1.5, "Video karesi\n1920×1080"), (1.75, 2.35, "YOLOv8s (1280)\noyuncu·hakem·top"),
             (1.75, 0.65, "YOLOv8M saha\nanahtar noktaları"), (3.6, 2.35, "ByteTrack\nkimlik takibi"),
             (3.6, 0.65, "Homografi\n(RANSAC + güncelleme\nkoşulu, metre)"), (5.45, 2.35, "Takım ataması\nLAB + çim maskesi\n+ iz oylaması"),
             (5.45, 0.65, "Top izleyici\ntek top + kısa\nboşluk doldurma"), (7.3, 1.5, "Top sahipliği\n(histerezis) →\nolay kuralları"),
             (9.15, 2.35, "ANLIK: hakimiyet %,\npas/dk, olay akışı\n(HUD + taktik harita)"),
             (9.15, 0.65, "Kesit sonu: CIES\nprofil + yüzdelik\n→ karakter kartı")]
    pos = {}
    for x, y, t in boxes:
        hl = "hakimiyet" in t or "CIES" in t
        ax.add_patch(FancyBboxPatch((x, y), 1.6, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#fdf0ea" if hl else "#eef4fc", ec=HIGHLIGHT if hl else S[0], lw=1.2))
        ax.text(x + 0.8, y + 0.45, t, ha="center", va="center", fontsize=7.2, color=INK)
        pos[t.split("\n")[0]] = (x, y)
    def arr(a, b):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch((x0 + 1.6, y0 + 0.45), (x1, y1 + 0.45), arrowstyle="-|>", mutation_scale=10,
                                     color=INK2, lw=1, connectionstyle="arc3,rad=0"))
    for a, b in (("Video karesi", "YOLOv8s (1280)"), ("Video karesi", "YOLOv8M saha"), ("YOLOv8s (1280)", "ByteTrack"),
                 ("YOLOv8M saha", "Homografi"), ("ByteTrack", "Takım ataması"), ("YOLOv8s (1280)", "Top izleyici"),
                 ("Homografi", "Top izleyici"), ("Takım ataması", "Top sahipliği"), ("Top izleyici", "Top sahipliği"),
                 ("Top sahipliği", "ANLIK: hakimiyet %,"), ("Top sahipliği", "Kesit sonu: CIES")):
        arr(a, b)
    _save(fig, "fig_architecture.png")


def fig_example():
    """Aynı test karesi: girdi + GT etiketleri | önerilen modelin tahmini; köşede top yakınlaştırması."""
    from bgoru.experiments.common import load_split, predict_split

    det = RESULTS / "detection" / "detection_results.csv"
    if not det.exists():
        return
    thr = pd.read_csv(det).set_index("model_id")
    test = load_split("test")
    preds = predict_split("E3_ft1280_cp", "test", test)
    i = 300
    it, p = test[i], preds[i]
    img = cv2.imread(str(it["path"]))
    gtv, prv = img.copy(), img.copy()
    col = {0: (214, 120, 42), 1: (0, 161, 237), 2: (52, 104, 235)}
    for b, c in zip(it["boxes"], it["cls"]):
        cv2.rectangle(gtv, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), col[int(c)], 3)
    tb, tp_ = thr.loc["E3_ft1280_cp", "ball_conf_thr"], thr.loc["E3_ft1280_cp", "player_conf_thr"]
    for q in p:
        if q[4] >= (tb if q[5] == 2 else tp_):
            cv2.rectangle(prv, tuple(q[:2].astype(int)), tuple(q[2:4].astype(int)), col[int(q[5])], 3)
    balls = it["boxes"][it["cls"] == 2]
    if len(balls):  # top çevresi x5 yakınlaştırma, sağ alt köşeye
        cx, cy = ((balls[0, :2] + balls[0, 2:]) / 2).astype(int)
        for im in (gtv, prv):
            z = im[max(0, cy - 40):cy + 40, max(0, cx - 60):cx + 60].copy()
            z = cv2.resize(z, (z.shape[1] * 4, z.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
            H_, W_ = im.shape[:2]
            im[H_ - z.shape[0] - 10:H_ - 10, W_ - z.shape[1] - 10:W_ - 10] = z
            cv2.rectangle(im, (W_ - z.shape[1] - 12, H_ - z.shape[0] - 12), (W_ - 8, H_ - 8), (255, 255, 255), 3)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for a, (im, t) in zip(ax, ((gtv, "(a) Girdi + etiket (GT); sağ alt: top ×4"), (prv, "(b) E3 tahmini (doğrulama eşiği)"))):
        a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); a.axis("off"); a.set_title(t, loc="left", fontsize=10)
    _save(fig, "fig_example_prediction.png")


def fig_compare():
    """Aynı (rastgele, tohum 0, farklı sekanslardan) test karelerinde E1 / E2 / E3 / G1 (son sistem) top tespitleri."""
    import random

    from bgoru.experiments.common import EXPERIMENTS as EXP, load_split

    det = RESULTS / "detection" / "detection_results.csv"
    if not det.exists():
        return
    thr = pd.read_csv(det).set_index("model_id")
    models = [m for m in ("E1_coco1280", "E2_ft640", "E3_ft1280_cp", "G1_ft1280_gsr") if m in thr.index
              and (EXP / "preds" / f"{m}_test.npz").exists()]
    if len(models) < 2:
        return
    test = load_split("test")
    preds = {m: list(np.load(EXP / "preds" / f"{m}_test.npz", allow_pickle=True)["preds"]) for m in models}
    rng = random.Random(0)
    by_seq = {}
    for i, it in enumerate(test):
        if (it["cls"] == 2).any():
            by_seq.setdefault(it["seq"], []).append(i)
    seqs = sorted(by_seq)
    rng.shuffle(seqs)
    idx = [rng.choice(by_seq[s_]) for s_ in seqs[:5]]
    name = {"E1_coco1280": "E1 COCO", "E2_ft640": "E2 (640)", "E3_ft1280_cp": "E3 (1280+KY)", "G1_ft1280_gsr": "G1 (son)"}
    fig, axes = plt.subplots(len(models), len(idx), figsize=(2.3 * len(idx), 2.3 * len(models)))
    for c, i in enumerate(idx):
        it = test[i]
        img = cv2.imread(str(it["path"]))
        g = it["boxes"][it["cls"] == 2][0]
        cx, cy = ((g[:2] + g[2:]) / 2).astype(int)
        x0, y0 = max(0, cx - 70), max(0, cy - 70)
        for r_, m in enumerate(models):
            im = img.copy()
            cv2.rectangle(im, tuple(g[:2].astype(int)), tuple(g[2:].astype(int)), (60, 200, 60), 2)
            q = preds[m][i]
            q = q[(q[:, 5] == 2) & (q[:, 4] >= thr.loc[m, "ball_conf_thr"])]
            for b in q:
                cv2.rectangle(im, tuple(b[:2].astype(int)), tuple(b[2:4].astype(int)), (40, 40, 230), 2)
            crop = im[y0:y0 + 140, x0:x0 + 140]
            ax = axes[r_, c]
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name[m], fontsize=9)
            if r_ == 0:
                ax.set_title(it["seq"], fontsize=8)
    fig.suptitle("Aynı test kareleri: yeşil = GT top, kırmızı = doğrulama eşiği üstündeki top tahmini", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_compare_models.png")


def fig_extra_data():
    """Ek eğitim verisi deneyi (eval_gsr): (a) doğrulama mAP@[.5:.95] - gradyan adımı, (b) test top AP@0.5 ve GA."""
    p = RESULTS / "detection" / "gsr_extra.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    ci = d.get("ball_AP50_bootstrap_CI95", {})
    rows = {r["model_id"]: r for r in d["rows"]}
    imgs = d["train_images"]
    runs = [("A1_ft1280", "A1 (20 ep.)", S[1], "-", imgs["A1"]), ("A1_ft1280_s1", "A1 tohum 1", S[1], "--", imgs["A1"]),
            ("C1_ft1280_ep50", "C1 (A1 verisi, 50 ep.)", S[2], "-", imgs.get("C1", imgs["A1"])),
            ("G1_ft1280_gsr", "G1 (A1 + GSR, 20 ep.)", S[0], "-", imgs["G1"])]
    runs = [r for r in runs if r[0] in rows and (EXPERIMENTS / "runs" / r[0] / "results.csv").exists()]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5), gridspec_kw={"width_ratios": [1.35, 1]})
    for mid, lab, col, ls, n in runs:
        df = pd.read_csv(EXPERIMENTS / "runs" / mid / "results.csv")
        df.columns = [c.strip() for c in df.columns]
        it = df["epoch"] * int(np.ceil(n / 4)) / 1000  # batch 4 -> iterasyon/epoch
        ax[0].plot(it, df["metrics/mAP50-95(B)"], color=col, lw=2 if ls == "-" else 1.4, ls=ls, label=lab)
        if ls == "-":
            ax[0].text(it.iloc[-1] + 0.3, df["metrics/mAP50-95(B)"].iloc[-1], f"{df['metrics/mAP50-95(B)'].iloc[-1]:.3f}",
                       fontsize=7, color=INK2, va="center")
    ax[0].set_xlabel("Eğitim iterasyonu (bin, batch 4)"); ax[0].set_ylabel("Doğrulama mAP@[.5:.95] (↑)")
    ax[0].set_title("(a) Doğrulama eğrileri, adım eksenine göre", loc="left", fontsize=10)
    ax[0].set_xlim(right=ax[0].get_xlim()[1] + 2)
    ax[0].legend(fontsize=7.5, loc="lower right")
    bars = [r for r in runs if r[3] == "-"]
    vals = [rows[m]["ball_AP50"] for m, *_ in bars]
    err = np.array([[v - ci[m]["lo"], ci[m]["hi"] - v] if m in ci else [0, 0] for v, (m, *_) in zip(vals, bars)]).T
    ax[1].bar(range(len(bars)), vals, color=[c for _, _, c, *_ in bars], width=0.55, yerr=err, capsize=3,
              error_kw={"lw": 1, "ecolor": INK2})
    for i, v in enumerate(vals):
        ax[1].text(i, v + err[1][i] + 0.008, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    if "A1_ft1280_s1" in rows:
        ax[1].plot(0, rows["A1_ft1280_s1"]["ball_AP50"], "o", ms=6, mfc="white", mec=INK, mew=1.2, label="A1 tohum 1")
        ax[1].legend(fontsize=7.5, loc="lower right")
    ax[1].set_xticks(range(len(bars)), [lab.replace(" (", "\n(") for _, lab, *_ in bars], fontsize=7.5)
    ax[1].set_ylabel("Test top AP@0.5 (↑)"); ax[1].set_ylim(0, max(vals) * 1.3)
    ax[1].set_title("(b) Test top AP@0.5, %95 bootstrap GA", loc="left", fontsize=10)
    _save(fig, "fig_extra_data.png")


def main():
    for f in (fig_data, fig_training, fig_detection, fig_robustness, fig_team, fig_possession, fig_architecture,
              fig_example, fig_compare, fig_extra_data):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"  [!] {f.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
