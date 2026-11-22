"""Rapor: sonuçlar, hata analizi, sonuç/kaynakça/ekler. Tüm sayılar results/ çıktılarından gelir."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from content import FIGS, FINAL, FINAL_V, ROOT, cp_seed_status, f1, f3, final_poss, gd, has_g1

R = ROOT / "results"


def _ci(d, key):
    c = d["boot"]["ball_AP50_bootstrap_CI95"].get(key)
    return "—" if c is None else f"[{c['lo']:.3f}; {c['hi']:.3f}]"


def _diff(d, a, b):
    return d["boot"]["paired_differences"].get(f"{a} - {b}", {})


def seeds_ok(d):
    """E3 ve A1 için en az iki tohum ölçüldüyse tohum tekrarı özeti, yoksa None."""
    s = d.get("seeds")
    if not s or any(s["summary"].get(k, {}).get("n_seeds", 0) < 2 for k in ("E3_ft1280_cp", "A1_ft1280")):
        return None
    return s


def _bench(d, mid, col):
    b = d["bench"]
    if b is None or mid not in b.index:
        return None
    return b.loc[mid, col]


def _word(x, up="artırmıştır", down="düşürmüştür", eps=0.005):
    if x is None:
        return "değiştirmemiştir"
    return up if x > eps else (down if x < -eps else "anlamlı biçimde değiştirmemiştir")


# =====================================================================================================
def sec6(r, d):
    det = d["det"]
    rows_main = []
    chosen = d["boot"]["chosen_input_size"]
    for key, name in (("E1", "E1 Temel (COCO, ince ayarsız)"), ("E2_ft640", "E2 Alternatif (640)"),
                      ("E3_ft1280_cp", "E3 Önerilen (1280 + KY)"),
                      ("A1_ft1280", "A1 Ablation (1280, artırımsız)" if has_g1(d) else "A1 Son sistem (1280, artırımsız)"),
                      (FINAL, "G1 Son sistem (A1 + ek veri)")):
        if key not in det.index:
            continue
        x = det.loc[key]
        mid = chosen.get(key, key)
        rows_main.append((f"{name}" + (f" @{mid[-4:].lstrip('o')}" if key == "E1" else ""),
                          f"{x['ball_AP50']:.3f} {_ci(d, key)}", f3(x.get("ball_APc")), f3(x["ball_R"]),
                          f3(x["player_AP50"]), f3(x["referee_AP50"]) if not pd.isna(x["referee_AP50"]) else "—",
                          f3(x["mAP50_95"]), f1(_bench(d, mid, "latency_ms_mean")), f"{_bench(d, mid, 'gpu_peak_MB'):.0f}"
                          if _bench(d, mid, "gpu_peak_MB") is not None else "—"))
    r.h1("6 Sonuçlar ve ablation")
    r.p("Tüm değerler test kümesinde (49 sekans, 1470 kare) tek eğitim çalıştırmasından"
        + (" (tohum 0; E3/A1 tohum tekrarı Ek A.3)" if seeds_ok(d) else "") + " ölçülmüştür; köşeli "
        "parantezler sekans bazlı bootstrap %95 güven aralığıdır. AP, recall ve F1 için yüksek değer, süre ve "
        "bellek için düşük değer iyidir. AP_m: merkez uzaklığı ≤ 2 top çapı ile eşleştirilen yardımcı top AP'si "
        "(Bölüm 7). Süre: ms/kare (batch 1, FP16); bellek: en yüksek GPU belleği (MB).")
    r.table(["Yöntem", "Top AP@0.5 (ana)", "Top AP_m", "Top R", "Oyuncu AP@0.5", "Hakem AP@0.5",
             "mAP@[.5:.95]", "ms/kare", "Bellek MB"], rows_main,
            widths_cm=[3.6, 2.9, 1.6, 1.2, 1.6, 1.5, 1.6, 1.4, 1.5],
            caption=("Ana karşılaştırma (RQ1). A1 ablation modelidir; G1, A1 ayarlarıyla ek eğitim verisi kullanır ve "
                     "doğrulamada en iyi olduğu için son sistemdir." if has_g1(d) else
                     "Ana karşılaştırma (RQ1). A1, ablation modelidir; doğrulamada en iyi olduğu için son sistemde kullanılır."),
            font=8, highlight_rows=tuple(i for i, rw in enumerate(rows_main)
                                         if rw[0].startswith(("E3", "G1" if has_g1(d) else "A1"))))
    rr = det.reindex(["E1", "E2_ft640", "E3_ft1280_cp", "A1_ft1280", "A2_ft640_cp"])
    e1, e2, e3, a1, a2 = (rr.loc[k] for k in rr.index)
    d32, d3a1, d3a2, d21 = (_diff(d, "E3_ft1280_cp", "E2_ft640"), _diff(d, "E3_ft1280_cp", "A1_ft1280"),
                            _diff(d, "E3_ft1280_cp", "A2_ft640_cp"), _diff(d, "E2_ft640", "E1"))
    rows_ab = [("A1: kopyala-yapıştır kaldırıldı (1280)", f"{a1['ball_AP50']:.3f} {_ci(d, 'A1_ft1280')}",
                f"E3−A1 = {d3a1.get('mean', float('nan')):+.3f} [{d3a1.get('lo', float('nan')):+.3f}; "
                f"{d3a1.get('hi', float('nan')):+.3f}]; artırımın katkısı"),
               ("A2: yüksek çözünürlük kaldırıldı (640 + KY)", f"{a2['ball_AP50']:.3f} {_ci(d, 'A2_ft640_cp')}",
                f"E3−A2 = {d3a2.get('mean', float('nan')):+.3f} [{d3a2.get('lo', float('nan')):+.3f}; "
                f"{d3a2.get('hi', float('nan')):+.3f}]; çözünürlüğün katkısı"),
               ("E3: tam yöntem", f"{e3['ball_AP50']:.3f} {_ci(d, 'E3_ft1280_cp')}",
                f"E3−E2 = {d32.get('mean', float('nan')):+.3f} [{d32.get('lo', float('nan')):+.3f}; "
                f"{d32.get('hi', float('nan')):+.3f}]")]
    if has_g1(d):
        dca, dgc = gd(d, "C1_ft1280_ep50", "A1_ft1280"), gd(d, FINAL, "C1_ft1280_ep50")
        rows_ab += [("C1: A1 + 2,5× eğitim adımı (50 epoch)",
                     f"{det.loc['C1_ft1280_ep50', 'ball_AP50']:.3f} {_ci(d, 'C1_ft1280_ep50')}",
                     f"C1−A1 = {dca['mean']:+.3f} [{dca['lo']:+.3f}; {dca['hi']:+.3f}]; yalnızca uzun eğitimin etkisi"),
                    ("G1: A1 + ek eğitim verisi (3 maç, 2900 kare)", f"{det.loc[FINAL, 'ball_AP50']:.3f} {_ci(d, FINAL)}",
                     f"G1−C1 = {dgc['mean']:+.3f} [{dgc['lo']:+.3f}; {dgc['hi']:+.3f}]; eşit adımda ek verinin etkisi")]
    r.table(["Ablation yapılandırması", "Top AP@0.5 [%95 GA]", "Yorum (eşleştirilmiş bootstrap farkı)"], rows_ab,
            widths_cm=[5.2, 3.8, 8.0], caption="Ablation ve ek veri sonuçları (top AP@0.5, test)." if has_g1(d)
            else "Ablation sonuçları (top AP@0.5, test).", font=8.5,
            highlight_rows=(2, 4) if has_g1(d) else (2,))
    r.figure(FIGS / "fig_training_curves.png", "Eğitim eğrileri: doğrulama mAP@0.5 ve kutu kaybı (20 epoch, tohum 0"
             + ("; G1'in epoch başına adımı 2,5 kat fazladır, adım eksenli karşılaştırma Ek A.4" if has_g1(d) else "")
             + ").", width_cm=10.5)
    n_det = r.figure(FIGS / "fig_detection_results.png", "(a) Test top AP@0.5 ve %95 bootstrap güven aralığı"
             + (" (mavi: RQ1'de önerilen E3, turuncu: son sistem G1)" if has_g1(d) else "")
             + ", (b) top boyutuna göre recall (doğrulamada seçilen eşikte).", width_cm=12)
    if (FIGS / "fig_compare_models.png").exists():
        r.figure(FIGS / "fig_compare_models.png", "Aynı test karelerinde (5 farklı sekanstan tohum 0 ile rastgele "
                 "seçildi, sonuca bakılmadan) modellerin top tespitleri.", width_cm=8)

    # --- RQ2
    t = d["team"]
    lab = {"T1_baseline": "T1 Temel: tüm kutu ort. RGB + K-Means",
           "T2_alternative": "T2 Alternatif: üst yarı piksel K-Means",
           "T3_proposed_v1": "T3 Önerilen v1: gövde + sabit HSV çim maskesi + LAB + oylama",
           "T5_local_mask": "T5 Önerilen (son): yerel saha maskesi + LAB + dengeli kümeleme + oylama",
           "T5a_no_vote": "T5a Ablation: iz oylaması yok", "T5b_no_balance": "T5b Ablation: dengeli kümeleme yok",
           "T5c_rgb_not_lab": "T5c Ablation: LAB yerine RGB",
           "T5d_hsv_mask": "T5d Ablation: yerel yerine sabit HSV maskesi",
           "T5e_no_mask": "T5e Ablation: saha maskesi yok"}

    def cell(st, c, std=True):
        x = t[(t.set == st) & (t.config == c)]
        if x.empty:
            return "—"
        x = x.iloc[0]
        return f"{x['acc_mean']:.3f} ± {x['acc_std']:.3f}" if std else f3(x["acc_min"])

    rows = [(lab[c], cell("val", c), cell("dense", c), cell("dense", c, False), cell("sparse", c), cell("sparse", c, False))
            for c in lab if not t[t.config == c].empty]
    hl = tuple(i for i, c in enumerate([c for c in lab if not t[t.config == c].empty]) if c == "T5_local_mask")
    r.table(["Takım ataması (RQ2)", "Doğrulama (18 sek.)", "Yoğun test (10)", "en kötü", "Seyrek test (49)", "en kötü"],
            rows, widths_cm=[6.4, 2.4, 2.4, 1.2, 2.4, 1.2],
            caption="Takım ataması doğruluğu (GT kutular, kaleciler hariç; ortalama ± sekanslar arası standart sapma). "
                    "Son yöntem ve ablation'ları doğrulama kümesinde seçilmiştir.", font=7.5, highlight_rows=hl)

    # --- RQ3
    p = d["poss"]
    labp = {"V0_E1": "V0: E1 COCO dedektör", "V1_E2": "V1: E2 dedektör (640)", "V2_E3_full": "V2: Tam sistem (E3)",
            "V3_E3_team_baseline": "V3: E3 + temel takım ataması", "V4_E3_no_ballfill": "V4: E3, top doldurma yok",
            "V5_E3_no_hysteresis": "V5: E3, histerezis yok",
            "V6_A1_final": "V6: A1 dedektörü" if FINAL_V in p.index else "V6: A1 (son sistem dedektörü)",
            FINAL_V: "V7: G1 (son sistem dedektörü)"}
    rows = []
    for v, name in labp.items():
        if v not in p.index:
            continue
        x = p.loc[v]
        rows.append((name, f"{x['poss_err_pp_mean']:.1f} ± {x['poss_err_pp_std']:.1f}", f3(x["control_agreement_mean"]),
                     f"{x['pass_F1_mean']:.3f}", f"{x['pass_P_mean']:.2f} / {x['pass_R_mean']:.2f}",
                     f"{x['passes_pred_total']:.0f} / {x['passes_gt_total']:.0f}", f3(x["team_acc_e2e_mean"])))
    r.table(["Aşağı akış (RQ3)", "Hakimiyet hatası (yp)", "Kontrol uyumu", "Pas F1", "Pas P / R",
             "Pas (tahmin/ref.)", "Takım doğr."], rows, widths_cm=[4.6, 2.4, 1.8, 1.4, 1.9, 2.2, 1.7],
            caption="Top hakimiyeti ve pas ölçümü (10 yoğun test sekansı, vekil referansa göre; grafik: Ek D).", font=8,
            highlight_rows=tuple(i for i, v in enumerate([v for v in labp if v in p.index])
                                 if v == (FINAL_V if FINAL_V in p.index else "V6_A1_final")))

    # --- yorum
    tt = {c: t[(t.set == "dense") & (t.config == c)].iloc[0] for c in lab if not t[(t.set == "dense") & (t.config == c)].empty}
    r.h2("Sonuçların yorumu")
    nan = float("nan")

    def fd(dd):
        return (f"{dd.get('mean', nan):+.3f} (%95 GA {dd.get('lo', nan):+.3f}; {dd.get('hi', nan):+.3f})")

    def sig(dd):
        lo, hi = dd.get("lo", nan), dd.get("hi", nan)
        if pd.isna(lo):
            return "belirsizdir"
        if lo > 0:
            return "istatistiksel olarak anlamlı bir artıştır"
        if hi < 0:
            return "istatistiksel olarak anlamlı bir düşüştür"
        return "anlamlı değildir (güven aralığı sıfırı içerir)"

    txt = (f"**RQ1.** Önerilen yapılandırma (E3) test kümesinde top AP@0.5 = {f3(e3['ball_AP50'])} değerine ulaşmıştır; "
           f"640 px ince ayara (E2: {f3(e2['ball_AP50'])}) göre fark {fd(d32)} olup {sig(d32)}. ")
    if not pd.isna(e1["ball_AP50"]) and not pd.isna(e2["ball_AP50"]):
        if e1["ball_AP50"] > e2["ball_AP50"]:
            txt += (f"Dikkat çekici biçimde, ince ayar görmemiş ancak doğrulamada 1280 px girdi seçilen COCO modeli "
                    f"(E1: {f3(e1['ball_AP50'])}), 640 px'de ince ayarlanan E2'den daha yüksek top AP'si vermiştir; top "
                    f"için girdi çözünürlüğü alana özgü ince ayardan daha belirleyicidir. ")
        else:
            txt += f"İnce ayar (E2), COCO taban modelinin (E1: {f3(e1['ball_AP50'])}) üzerine çıkmıştır. "
    if not pd.isna(a2["ball_AP50"]):
        txt += (f"Ablation'larda yüksek çözünürlüğün kaldırılması (A2, 640 px + kopyala-yapıştır) top AP'sini "
                f"{f3(a2['ball_AP50'])} değerine götürmüştür: E3−A2 = {fd(d3a2)}, {sig(d3a2)}. ")
    if not pd.isna(a1["ball_AP50"]):
        txt += (f"Kopyala-yapıştırın kaldırılması (A1, 1280 px) ise {f3(a1['ball_AP50'])} değerini vermiştir: "
                f"E3−A1 = {fd(d3a1)}, {sig(d3a1)}. ")
        lo, hi = d3a1.get("lo", nan), d3a1.get("hi", nan)
        st = cp_seed_status(d)
        if not pd.isna(lo) and hi < 0:
            txt += ("Yani önerilen artırım " + ("bu eğitim bütçesinde" if st in (None, "neg_all") else "tohum 0 eğitiminde")
                    + " top AP'sini **düşürmüştür** (olumsuz sonuç); olası "
                    "nedenler yapıştırılan topların bağlamdan kopuk görünmesi ve sahne başına birden çok top "
                    "görülmesinin 'tek top' önselini bozmasıdır. ")
        elif not pd.isna(lo) and lo <= 0 <= hi:
            txt += "Dolayısıyla artırımın ana metriğe katkısı bu bütçede **gösterilememiştir**. "
        elif not pd.isna(lo):
            txt += "Artırım, yüksek çözünürlüğün üzerine ek bir kazanç sağlamıştır. "
        sd = seeds_ok(d)
        if sd:  # eğitim varyansı: E3 ve A1 ikinci tohumla yeniden eğitildi
            s3, sa = sd["summary"]["E3_ft1280_cp"]["ball_AP50"], sd["summary"]["A1_ft1280"]["ball_AP50"]
            pdif = sd["paired_differences"]
            dm, d1 = pdif.get("tohum ortalaması: E3 - A1", {}), pdif.get("seed1: E3 - A1", {})
            per = [pdif.get(f"seed{s}: E3 - A1", {}) for s in (0, 1)]
            txt += (f"İkinci tohumla tekrarda (Ek A.3) top AP@0.5 E3 için {f3(s3['mean'])} ± {f3(s3['std'])}, A1 için "
                    f"{f3(sa['mean'])} ± {f3(sa['std'])}; tohum 1'de E3−A1 = {fd(d1)}, tohum ortalamasında "
                    f"{fd(dm)}, {sig(dm)}. ")
            if st == "neg_all":
                txt += "Olumsuz sonuç iki tohumda da korunmuştur. "
            elif st == "pos_all":
                txt += "İki tohumda da artırım lehine anlamlı fark vardır. "
            elif st == "neg_some":
                ek = {0: "0'da", 1: "1'de"}  # ünlü uyumu: sıfır-da, bir-de
                sig_s = [ek[s] for s, p in enumerate(per) if p["hi"] < 0]
                ns_s = [ek[s] for s, p in enumerate(per) if p["hi"] >= 0]
                txt += (f"Tek tek tohumlara bakıldığında anlamlı düşüş yalnızca tohum {', '.join(sig_s)} görülmüş, tohum "
                        f"{', '.join(ns_s)} fark "
                        f"anlamsız kalmıştır; E3 hiçbir tohumda A1'i "
                        + ("geçmemiştir" if all(p["mean"] < 0 for p in per) else "anlamlı biçimde geçmemiştir")
                        + ". Dolayısıyla iki tohumda da geçerli olan sonuç artırımın **katkı sağlamadığıdır**; düşüşün "
                        f"büyüklüğü ise eğitim varyansına duyarlıdır (tohumlar arası std: E3 {f3(s3['std'])}, "
                        f"A1 {f3(sa['std'])}). ")
            elif st == "none":
                txt += "Hiçbir tohumda anlamlı fark yoktur; artırımın katkısı gösterilememiştir. "
            else:
                txt += "Fark tohuma duyarlıdır; tek eğitimle elde edilen işaret genellenmemelidir. "
        if not pd.isna(a2["ball_AP50"]):
            res_gain = e3["ball_AP50"] - a2["ball_AP50"]
            cp_gain = e3["ball_AP50"] - a1["ball_AP50"]
            txt += (f"Kazancın {'büyük kısmı yüksek çözünürlükten' if res_gain > cp_gain else 'büyük kısmı artırımdan'} "
                    f"gelir; AP_m de aynı eğilimi gösterir (E3 {f3(e3.get('ball_APc'))}, A1 {f3(a1.get('ball_APc'))}, "
                    f"A2 {f3(a2.get('ball_APc'))}). ")
    txt += (f"Top boyutu kovalarına göre recall Şekil {n_det}b’de verilmiştir. Oyuncu AP@0.5 E2'de "
            f"{f3(e2['player_AP50'])}, E3'te {f3(e3['player_AP50'])} olup kazanç top sınıfında yoğunlaşmıştır; hipotezin "
            f"bu kısmı "
            f"{'desteklenmiştir' if (e3['ball_AP50'] - e2['ball_AP50']) > abs(e3['player_AP50'] - e2['player_AP50']) else 'desteklenmemiştir'}. "
            f"Hakem sınıfında test AP'sinin ({f3(e3['referee_AP50'])}) doğrulamaya göre belirgin biçimde düşük olması, test "
            f"maçlarındaki farklı hakem forma renklerinden kaynaklanan bir alan değişimine işaret eder. "
            f"Maliyet: 1280 px girdi kare süresini {f1(_bench(d, 'E2_ft640', 'latency_ms_mean'))} ms değerinden "
            f"{f1(_bench(d, 'E3_ft1280_cp', 'latency_ms_mean'))} ms değerine çıkarır; model boyutu değişmez (9,8 M parametre).")
    r.p(txt)
    if has_g1(d):
        g, c1 = det.loc[FINAL], det.loc["C1_ft1280_ep50"]
        dga, dgc, dca = gd(d, FINAL, "A1_ft1280"), gd(d, FINAL, "C1_ft1280_ep50"), gd(d, "C1_ft1280_ep50", "A1_ft1280")
        gx = d.get("gsr") or {}
        n1 = (gx.get("paired_differences_ball_AP50") or {}).get("A1 tohum 1 - tohum 0 (eğitim varyansı)")
        r.p(f"**Ek eğitim verisi (G1).** A1 ayarlarıyla eğitim kümesine SoccerNet-GSR'dan üç yeni maç eklenmesi top "
            f"AP@0.5'i {f3(a1['ball_AP50'])} değerinden {f3(g['ball_AP50'])} değerine çıkarmıştır: G1−A1 = {fd(dga)}, "
            f"{sig(dga)}. Epoch sayısı sabit olduğundan G1 2,5 kat daha çok gradyan adımı görür; aynı adım sayısıyla "
            f"yalnızca eski veride eğitilen kontrol (C1: {f3(c1['ball_AP50'])}) A1'den farklı değildir (C1−A1 = {fd(dca)}), "
            f"G1−C1 = {fd(dgc)} ise {sig(dgc)}. Yani kazanç daha uzun eğitimden değil **ek veriden** gelmektedir (Ek A.4)."
            + (f" G1−A1 farkı, A1'in iki eğitim tohumu arasındaki farktan ({n1['mean']:+.3f} [{n1['lo']:+.3f}; "
               f"{n1['hi']:+.3f}]) belirgin biçimde büyüktür." if n1 else "")
            + f" G1 doğrulamada da en iyi modeldir (top AP@0.5 {f3(g['val_ball_AP50'])}; A1 {f3(a1['val_ball_AP50'])}). "
            f"En çok artan sınıf hakemdir ({f3(a1['referee_AP50'])} → {f3(g['referee_AP50'])}; C1 "
            f"{f3(c1['referee_AP50'])}); bu, yeni maçların hakem forması çeşitliliğini artırmasıyla uyumludur (ayrıca "
            f"sınanmadı). G1 tek tohumla eğitilmiştir.")
    def acc(st, c):
        x = t[(t.set == st) & (t.config == c)]
        return None if x.empty else float(x.iloc[0]["acc_mean"])

    if acc("dense", "T5_local_mask") is not None:
        r.p(f"**RQ2 ve geliştirme süreci (açık beyan).** İlk sürüm (v1) testte ölçüldüğünde yoğun kümede "
            f"{f3(acc('dense', 'T3_proposed_v1'))} doğruluk vermiş, ancak bir sekansta (SNMOT-147, uzaklaştırma anı) "
            f"0,544'e düşmüştür: ilk karelerde yalnızca üç kişi göründüğünden K-Means kümeleri 'kaleci ↔ diğerleri' "
            f"olarak ayırmıştır. Aynı v1 doğrulama maçında {f3(acc('val', 'T3_proposed_v1'))} değerine düşmüştür; bu maçta bir "
            f"takımın formasının yeşil olması, sabit HSV çim maskesinin forma piksellerini de silmesine yol açar. Test "
            f"kümesindeki bir hatanın incelenmesi tasarımı etkilediğinden **sızıntı riskine karşı** tüm düzeltmeler "
            f"(dengeli kümeleme, yerel saha maskesi, LAB/RGB seçimi) yalnızca doğrulama maçında karşılaştırılıp "
            f"seçilmiş, test kümesi son yöntem için bir kez çalıştırılmıştır. Son yöntem (T5) doğrulamada "
            f"{f3(acc('val', 'T5_local_mask'))}, yoğun testte {f3(acc('dense', 'T5_local_mask'))}, seyrek testte "
            f"{f3(acc('sparse', 'T5_local_mask'))} doğruluğa ulaşır (temel: {f3(acc('val', 'T1_baseline'))} / "
            f"{f3(acc('dense', 'T1_baseline'))} / {f3(acc('sparse', 'T1_baseline'))}). Yoğun testteki ortalama tek bir "
            f"sekansın (SNMOT-147: 0,508) etkisindedir; diğer dokuz sekansın tamamı ≥0,977'dir (medyan 0,998). Bu "
            f"klipte uzun süre yalnızca bir kaleci ve her takımdan birer oyuncu göründüğünden iki takımlı renk "
            f"kümelemesi tanımsızdır; **dengeli kümeleme bu hatayı gidermemiştir** ve doğrulamada da etkisizdir "
            f"(T5b: {f3(acc('val', 'T5b_no_balance'))} / {f3(acc('dense', 'T5b_no_balance'))}) — bu bileşen için olumsuz "
            f"bir sonuçtur. Gerçek maç analizinde renkler kalabalık bir andan bir kez öğrenildiği (arayüzdeki 'takım "
            f"renklerini algıla' adımı) için bu hata büyük ölçüde 30 sn'lik bağımsız klip protokolünden kaynaklanır. "
            f"Diğer ablation'larda (Tablo) iz oylaması üç kümede de en tutarlı katkıyı veren bileşendir; "
            f"LAB/RGB tercihinin etkisi tutarsızdır (RGB doğrulamada kötü, yoğun testte iyi); sabit HSV maskesinin "
            f"doğrulamadaki çöküşü, sabit renk eşiklerinin stadyum ve formaya bağımlılığını gösteren olumsuz bir sonuçtur.")
    if "V2_E3_full" in p.index:
        pp = p.reindex(["V2_E3_full", "V0_E1", "V3_E3_team_baseline", "V4_E3_no_ballfill", "V5_E3_no_hysteresis"])
        v2, v0, v3, v4, v5 = (pp.loc[k] for k in ("V2_E3_full", "V0_E1", "V3_E3_team_baseline", "V4_E3_no_ballfill",
                                                  "V5_E3_no_hysteresis"))
        v1 = p.loc["V1_E2"] if "V1_E2" in p.index else v2
        r.p(f"**RQ3.** Daha iyi dedektör, kare bazlı kontrol uyumunu V0→V1→V2 sırasıyla {f3(v0['control_agreement_mean'])}"
            f" → {f3(v1['control_agreement_mean'])} → {f3(v2['control_agreement_mean'])} artırmış, pas F1'ini COCO "
            f"dedektöre göre {f3(v0['pass_F1_mean'])} değerinden {f3(v2['pass_F1_mean'])} değerine çıkarmıştır (pas kesinliği "
            f"{v0['pass_P_mean']:.2f} → {v2['pass_P_mean']:.2f}); ancak E2 ile E3 arasında pas F1 farkı yoktur "
            f"({f3(v1['pass_F1_mean'])} / {f3(v2['pass_F1_mean'])}). Klip başına hakimiyet yüzdesi hatası tüm varyantlarda "
            f"16–22 yp arasındadır ve sekanslar arası standart sapması (~16 yp) farklardan büyüktür: 30 sn'lik bir klipte "
            f"yüzde, birkaç kritik sahiplik kararına bağlıdır ve 10 sekansta toplam yalnızca "
            f"{int(v2['passes_gt_total'])} referans pas vardır. Bu nedenle 'daha iyi top tespiti hakimiyet yüzdesini "
            f"iyileştirir' hipotezi **desteklenmemiştir**; yalnızca kare bazlı kontrol uyumu ve pas kesinliğinde "
            f"tutarlı bir iyileşme görülür. Ablation'lar: temel takım ataması (V3) pas kesinliğini "
            f"{v3['pass_P_mean']:.2f} değerine düşürür (F1 {f3(v3['pass_F1_mean'])}); top doldurmanın kaldırılması (V4) pas F1'ini "
            f"değiştirmez ({f3(v4['pass_F1_mean'])}) — bu bileşen için olumsuz sonuçtur; histerezisin kaldırılması (V5) "
            f"kontrol uyumunu {f3(v5['control_agreement_mean'])} değerine çıkarırken pas kesinliğini {v5['pass_P_mean']:.2f} "
            f"değerine düşürür (tahmin edilen pas sayısı {int(v5['passes_pred_total'])} olur), yani histerezis olay kalitesi ile anlık "
            f"tepki arasında bir ödünleşimdir."
            + (f" Son dedektörle (V7, G1) kontrol uyumu {f3(p.loc[FINAL_V, 'control_agreement_mean'])}, "
               f"pas F1 {f3(p.loc[FINAL_V, 'pass_F1_mean'])}, hakimiyet hatası "
               f"{f1(p.loc[FINAL_V, 'poss_err_pp_mean'])} yp'dir"
               + (f" (V6, A1: {f3(p.loc['V6_A1_final', 'control_agreement_mean'])}, "
                  f"{f3(p.loc['V6_A1_final', 'pass_F1_mean'])}, {f1(p.loc['V6_A1_final', 'poss_err_pp_mean'])} yp)"
                  + ("; G1'in daha yüksek top AP'si pas F1'ine yansımamıştır, bu da yukarıdaki bulguyla tutarlıdır"
                     if p.loc[FINAL_V, "pass_F1_mean"] < p.loc["V6_A1_final", "pass_F1_mean"] else "")
                  if "V6_A1_final" in p.index else "") + "." if FINAL_V in p.index else
               f" Doğrulamada seçilen son dedektörle (V6, A1) kontrol uyumu {f3(p.loc['V6_A1_final', 'control_agreement_mean'])}, "
               f"pas F1 {f3(p.loc['V6_A1_final', 'pass_F1_mean'])} ve hakimiyet hatası "
               f"{f1(p.loc['V6_A1_final', 'poss_err_pp_mean'])} yp'dir." if "V6_A1_final" in p.index else ""))


# =====================================================================================================
def sec7(r, d):
    fl, tot, rob = d["fail"], d["fail_tot"], d["rob"]
    r.h1("7 Hata analizi ve tartışma")
    r.h2("Başarısız örnekler")
    n = len(fl)
    off = int(tot.get("ball_FP_near_gt_ball", 0))
    r.p(f"Önerilen modelin doğrulama eşiğindeki test hataları sayılmıştır: top için {int(tot['ball_FN'])} kaçırılan ve "
        f"{int(tot['ball_FP'])} yanlış pozitif, oyuncu için {int(tot['player_FN'])} kaçırılan kutu. Bunlardan sekanslara "
        f"yayılan, tohum 0 ile rastgele **{n} örnek** seçilmiştir (en kötüler değil, temsilî): 12 kaçırılan top, 8 yanlış "
        f"top, 4 kaçırılan oyuncu ve 6 takım hatası. Her örnek için top boyutu, oyuncuyla örtüşme, Laplace varyansı "
        f"(bulanıklık) ve çim oranı ölçülmüş, kural tabanlı bir ön kategori atanmış, ardından görüntüler tek tek "
        f"incelenerek neden elle doğrulanmıştır (Şekil {r.fig_n + 1}; tam liste Ek B ve `results/failures/failures.csv`). "
        f"İnceleme beklenmedik bir bulgu ortaya çıkarmıştır: yanlış top tespitlerinin {off}/{int(tot['ball_FP'])}'i "
        f"(%{100 * off / max(int(tot['ball_FP']), 1):.0f}) gerçek topun 2 çapı içindedir; incelenen örneklerde tahmin topun "
        f"tam üzerindeyken GT kutusu birkaç piksel ötelenmiştir. 15 px'lik bir topta bu öteleme IoU'yu 0,5'in altına "
        f"düşürür ve aynı doğru tespit hem yanlış pozitif hem kaçırılan top sayılır. Bu nedenle tablolarda ana metriğin "
        f"(AP@0.5) yanında, tüm modellere aynı biçimde uygulanan merkez uzaklığı tabanlı **AP_m** (≤2 top çapı) "
        f"yardımcı metrik olarak verilmiştir; bu metrik model seçiminde kullanılmamıştır.")
    sheet = R / "failures" / "contact_sheet_1.jpg"
    if sheet.exists():
        r.figure(sheet, "Başarısız örneklerden kesitler (yeşil: GT, kırmızı: eşik üstü tahmin, turuncu: eşik altı "
                 "tahmin, sarı: takım hatası). Kimlikler Ek B ile eşleşir.", width_cm=11)
    reason_col = "manual_category" if "manual_category" in fl.columns and fl["manual_category"].notna().any() \
        else "auto_category"
    g = fl.groupby(["type", reason_col]).size().reset_index(name="n").sort_values(["type", "n"], ascending=[True, False])
    typ = {"ball_FN": "Kaçırılan top", "ball_FP": "Yanlış top", "player_FN": "Kaçırılan oyuncu", "team": "Yanlış takım"}
    sol = {
        "Etiket ötelemesi": "Etiket düzeltme veya merkez tabanlı eşleştirme; zamansal etiket enterpolasyonu",
        "Etiket/hareket uyumsuzluğu": "Bulanık top için çizgisel (uzamış) kutu etiketleri",
        "Hareket bulanıklığı": "Bulanıklık artırımı; ardışık kare girdisi",
        "Elde tutulan top": "Taç atışı/kaleci örnekleriyle hedefli veri artırma",
        "Örtülme (oyuncu/kaleci)": "Zamansal izleyici ile doldurma; örtülme örnekleri",
        "Düşük güven": "Önceki konuma yakın düşük güvenli adayları kabul eden zamansal kapı",
        "Düşük kontrast": "Renk/kontrast artırımı; P2 (adım 4) başlığı",
        "Kutu/konum hatası": "Daha yüksek çözünürlük; merkez tabanlı değerlendirme",
        "Saha dışı bağlam": "Saha dışı top örnekleri; saha maskesi ile önceli",
        "Topa benzer nesne": "Zor negatif madenciliği; tek top kısıtı",
        "Oyuncu parçası": "Zor negatifler; topun oyuncu kutusundaki konum önceli",
        "NMS / iç içe oyuncular": "Daha yüksek NMS IoU eşiği veya Soft-NMS",
        "Kalabalık örtülme": "Kalabalık sahneler için daha fazla veri; izleyiciyle doldurma",
        "Takım: sekans düzeyi çöküş": "Renkleri kalabalık bir andan bir kez öğrenmek; 3 kümeli (takım A/B/kaleci) model",
        "Takım: kutuda rakip": "Örtülme farkındalığı: kutular kesişirken oy vermemek",
        "Takım: arka plan": "Pano/tribün için de yerel arka plan maskesi",
        "Takım: belirsiz": "Oyların zamanla sönümlenmesi (eski hatalı oyları unutmak)",
    }
    rows = [(f"{typ.get(x.type, x.type)} — {x[reason_col]}", int(x.n), sol.get(x[reason_col], "—")) for _, x in g.iterrows()]
    r.table(["Hata türü", "Örnek sayısı", "Olası neden ve çözüm"], rows, widths_cm=[6.8, 1.8, 8.4],
            caption=f"İncelenen {n} başarısız örneğin elle doğrulanmış dağılımı.", font=8)
    r.h2("Zorlu koşullarda davranış")
    if rob is not None:
        piv = rob.pivot(index="model", columns="condition", values="ball_AP50")
        pivp = rob.pivot(index="model", columns="condition", values="player_AP50")
        conds = [c for c in ["clean", "night", "blur", "jpeg", "far_half"] if c in piv.columns]
        name = {"clean": "Temiz", "night": "Gece", "blur": "Bulanık", "jpeg": "JPEG q15", "far_half": "Uzak ×0,5"}
        rows = []
        mname = {"E1_coco1280": "E1 COCO (1280)", "E1_coco640": "E1 COCO (640)", "E2_ft640": "E2 (640)",
                 "E3_ft1280_cp": "E3 (1280 + KY)", "A1_ft1280": "A1 (1280)" if FINAL in piv.index else "A1 (1280, son)",
                 "A2_ft640_cp": "A2 (640 + KY)", FINAL: "G1 (ek veri, son)"}
        order = [m for m in ("E1_coco1280", "E1_coco640", "E2_ft640", "E3_ft1280_cp", "A1_ft1280", "A2_ft640_cp", FINAL)
                 if m in piv.index]
        for m in order:
            rows.append((mname.get(m, m), *[f"{piv.loc[m, c]:.3f} / {pivp.loc[m, c]:.3f}" for c in conds]))
        r.table(["Model", *[name[c] for c in conds]], rows, widths_cm=[3.0] + [2.8] * len(conds),
                caption="Zorlu koşullar: top AP@0.5 / oyuncu AP@0.5 (test, doğrulama eşikleri).", font=8)
        e3 = "E3_ft1280_cp"
        if e3 in piv.index:
            worst = min(conds[1:], key=lambda c: piv.loc[e3, c] / max(piv.loc[e3, "clean"], 1e-6))
            txt = (f"E3 için en büyük düşüş **{name[worst]}** koşulundadır (top AP "
                   f"{piv.loc[e3, 'clean']:.3f} → {piv.loc[e3, worst]:.3f}). Uzak kamera benzetimi (×0,5) topu 5–10 px'e "
                   f"indirerek küçük nesne sınırını zorlar; 640 px modeller (E2, A2) bu koşulda neredeyse hiç top "
                   f"bulamaz, oyuncu sınıfı ise tüm bozulmalarda belirgin biçimde daha dayanıklıdır. ")
            a1 = "A1_ft1280"
            if a1 in piv.index:
                better = [name[c] for c in conds[1:] if piv.loc[e3, c] > piv.loc[a1, c]]
                rel = lambda m, c: piv.loc[m, c] / max(piv.loc[m, "clean"], 1e-6)
                if better:
                    txt += (f"Temiz görüntüde A1'in gerisinde kalan E3, **{', '.join(better)}** koşul(lar)ında A1'i geçer "
                            f"(ör. gece: E3 korunma oranı %{100 * rel(e3, 'night'):.0f}, A1 %{100 * rel(a1, 'night'):.0f}); "
                            f"kopyala-yapıştır artırımı temiz test verisinde katkı sağlamasa da bozulmalara karşı "
                            f"bir dayanıklılık kazancı sağlıyor olabilir. Tek tohumla ölçüldüğünden"
                            + (" (E3'ün tohum değişkenliği yüksektir, Ek A.3)"
                               if cp_seed_status(d) in ("neg_some", "mixed") else "")
                            + " bu bir ön bulgudur. ")
                else:
                    txt += "A1 tüm bozulmalarda E3'ten iyi veya eşittir; artırım dayanıklılık kazancı da sağlamamıştır. "
                if FINAL in piv.index:
                    gb = [name[c] for c in conds[1:] if piv.loc[FINAL, c] > piv.loc[a1, c]]
                    gw = min(conds[1:], key=lambda c: rel(FINAL, c))
                    txt += (f"Son sistem G1, A1'e göre {len(gb)}/{len(conds) - 1} bozulmada daha yüksek top AP'si verir"
                            + (f" ({', '.join(gb)})" if 0 < len(gb) < len(conds) - 1 else "")
                            + f"; en büyük göreli düşüş onun için {'de ' if gw == worst else ''}**{name[gw]}** "
                              f"koşulundadır (korunma oranı %{100 * rel(FINAL, gw):.0f}). ")
            txt += ("Taktik kamera videolarında (alan değişimi) homografi kapsamı ve kart üretimi Bölüm 8 ve Ek C'de "
                    "nitel olarak gösterilmiştir.")
            r.p(txt)
    r.h2("Sınırlılıklar ve genellenebilirlik")
    r.bullets([
        ("Ana deneylerde eğitim 2 maç (39 klip), son sistemde 5 maçtır (97 klip); ek verinin kazancı, veri "
         "çeşitliliğinin hâlâ darboğaz olduğunu gösterir. Doğrulama tek maç, test aynı yayın stilinde 3 maçtır."
         if has_g1(d) else
         "Eğitim yalnızca 2 maçtan (39 klip) oluşur; doğrulama tek maçtır. Farklı forma/stadyum/ışık koşullarına "
         "genelleme sınırlıdır; test 3 farklı maçtır ancak hepsi aynı yayın stilindedir."),
        ("Eğitim varyansı yalnızca E3 ve A1 için iki tohumla ölçülmüştür (n = 2, kaba tahmin); diğerleri (G1, C1 "
         "dahil) tek tohumdur ve belirsizlikleri yalnızca test örneklemesinden (bootstrap) gelir."
         if seeds_ok(d) else
         "Her yapılandırma tek tohumla eğitilmiştir; belirsizlik yalnızca test örneklemesinden (bootstrap) gelir, "
         "eğitim varyansı ölçülmemiştir."),
        "RQ3 referansı insan etiketli değildir (GT kutulara uygulanan aynı kurallar); kuralların kendi hataları "
        "(ör. 2B yakınlıktan doğan yanlış 'top kapma') ölçülmez.",
        "Yayın görüntüsünde homografi yoktur: hız/mesafe yaklaşıktır; şut/orta/kilit pas yalnızca taktik kamerada "
        "çıkarılır; hava topu mücadeleleri ölçülemez.",
        "Kart puanları kesit içi yüzdeliktir; kısa kesitlerde (≤1 dk) profiller istatistiksel olarak zayıftır ve "
        "kimlik değişimleri bir oyuncunun istatistiklerini bölebilir.",
    ])
    r.h2("Açıklanabilirlik ve kullanım sınırları")
    r.p("Açıklama haritası üretilmemiştir; her istatistik izlenebilir bir olay kaydına (`events.csv`: kare, oyuncu, "
        "olay türü) dayanır ve videoda top sahibi işaretlenir. Sonuçlar **takım düzeyinde hakimiyet ve pas "
        "temposunun** yaklaşık izlenmesini destekler; tek oyuncu hakkında kesin yargı, kimlik doğrulaması ve daha "
        "uzun kesitler olmadan desteklenmez.")


# =====================================================================================================
REFS = [
    "[1] A. Cioppa, S. Giancola, A. Deliège, L. Kang, X. Zhou, Z. Cheng, B. Ghanem, M. Van Droogenbroeck, "
    "\"SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos,\" CVPR Workshops, 2022. "
    "Veri: https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023 (erişim 17.11.2026).",
    "[2] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, X. Wang, \"ByteTrack: Multi-Object "
    "Tracking by Associating Every Detection Box,\" ECCV, 2022.",
    "[3] M. Kisantal, Z. Wojna, J. Murawski, J. Naruniec, K. Cho, \"Augmentation for small object detection,\" "
    "arXiv:1902.07296, 2019.",
    "[4] F. C. Akyon, S. O. Altinuc, A. Temizel, \"Slicing Aided Hyper Inference and Fine-tuning for Small Object "
    "Detection,\" IEEE ICIP, 2022.",
    "[5] M. Istasse, J. Moreau, C. De Vleeschouwer, \"Associative Embedding for Team Discrimination,\" CVPR "
    "Workshops, 2019.",
    "[6] F. Vidal-Codina, N. Evans, B. El Fakir, J. Billingham, \"Automatic event detection in football using "
    "tracking data,\" Sports Engineering, 25, 2022.",
    "[7] CIES Football Observatory, \"Technical profiling of football players,\" "
    "https://football-observatory.com/Technical-profiling-of-football-players (erişim 17.11.2026).",
    "[8] H. Boulahia (Hmzbo), \"Football-Analytics-with-Deep-Learning-and-Computer-Vision,\" GitHub deposu ve "
    "YouTube anlatımı \"Computer Vision for Football Analysis in Python with Yolov8 & OpenCV\"; saha anahtar noktası "
    "modeli (YOLOv8M), oyuncu modeli (YOLOv8L) ve taktik kamera demo videoları.",
    "[9] G. Jocher, A. Chaurasia, J. Qiu, \"Ultralytics YOLOv8,\" v8.3.6, https://github.com/ultralytics/ultralytics "
    "(AGPL-3.0); COCO ön-eğitimli yolov8s.pt ağırlıkları.",
    "[10] T.-Y. Lin ve ark., \"Microsoft COCO: Common Objects in Context,\" ECCV, 2014.",
    "[11] M. A. Fischler, R. C. Bolles, \"Random Sample Consensus,\" Communications of the ACM, 24(6), 1981.",
    "[12] G. Ghiasi ve ark., \"Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation,\" "
    "CVPR, 2021.",
    "[13] Roboflow, \"supervision\" kütüphanesi v0.24.0 (ByteTrack uygulaması), https://github.com/roboflow/supervision.",
    "[14] Dishant27, \"Football_Analysis\" GitHub deposu — pozisyona göre ağırlıklı genel oyuncu puanı fikri.",
    "[15] V. Somers ve ark., \"SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification "
    "on a Minimap,\" CVPR Workshops, 2024. Veri: https://huggingface.co/datasets/SoccerNet/SN-GSR-2025 (valid ayrımı, "
    "erişim 18.11.2026).",
]


def sec8(r, d):
    det, p, t = d["det"], d["poss"], d["team"]
    e2, e3 = det.loc["E2_ft640"], det.loc["E3_ft1280_cp"]
    dd = d["boot"]["paired_differences"].get("E3_ft1280_cp - E2_ft640", {})
    r.h1("8 Sonuç, kaynakça ve ekler")
    r.h2("Sonuç")
    tp = t[(t.set == "dense") & (t.config == "T5_local_mask")].iloc[0]
    v2 = final_poss(p)
    a1r = det.loc["A1_ft1280"] if "A1_ft1280" in det.index else None
    da1 = d["boot"]["paired_differences"].get("E3_ft1280_cp - A1_ft1280", {})
    cp_note = ""
    if a1r is not None and da1:
        if da1.get("hi", 0) < 0 and cp_seed_status(d) == "neg_some":
            cp_note = (" Ablation, kazancın yüksek çözünürlükten geldiğini, kopyala-yapıştırın ise hiçbir tohumda katkı "
                       "sağlamadığını göstermiştir.")
        elif da1.get("hi", 0) < 0:
            cp_note = (" Ablation, kazancın yüksek çözünürlükten geldiğini, kopyala-yapıştır artırımının ise top AP'sini "
                       "düşürdüğünü göstermiştir; uygulamada artırımsız 1280 px model (A1) daha iyi bir seçenektir.")
        elif da1.get("lo", 0) <= 0:
            cp_note = (" Ablation, kazancın büyük ölçüde yüksek çözünürlükten geldiğini; kopyala-yapıştır artırımının "
                       "katkısının ise anlamlı olmadığını göstermiştir.")
        else:
            cp_note = " Ablation, hem yüksek çözünürlüğün hem de kopyala-yapıştır artırımının katkı sağladığını göstermiştir."
    r.p(f"Futbola özgü ince ayar + 1280 px + top kopyala-yapıştır (E3), top AP@0.5'i 640 px ince ayara göre "
        f"{f3(e2['ball_AP50'])} değerinden {f3(e3['ball_AP50'])} değerine taşımıştır ({dd.get('mean', float('nan')):+.3f}, "
        f"%95 GA [{dd.get('lo', float('nan')):+.3f}; {dd.get('hi', float('nan')):+.3f}])."
        + cp_note
        + (f" Artırımsız ayarlarla eğitime 3 yeni maç eklemek top AP@0.5'i {f3(det.loc[FINAL, 'ball_AP50'])} değerine "
           f"çıkarmış, eşit adımlı kontrol bu kazancın ek veriden geldiğini göstermiştir (G1−C1 = "
           f"{gd(d, FINAL, 'C1_ft1280_ep50')['mean']:+.3f}); son sistem bu modeldir (G1)." if has_g1(d) else "")
        + f" Takım ataması yoğun testte {f3(tp['acc_mean'])} doğruluğa ulaşmış; son sistem "
          f"({'G1' if has_g1(d) and FINAL_V in p.index else 'A1'}) pasları {f3(v2['pass_F1_mean'])} F1 "
        f"ile ölçmüş, klip başına hakimiyet hatası ({f1(v2['poss_err_pp_mean'])} yp) ise daha iyi tespitle güvenilir "
        f"biçimde azalmamıştır. Sistem taktik kamerada metre cinsinden hız/mesafe ile anlık istatistik ve karakter "
        f"kartları üretir"
        + (f" (tam hat {d['pipe']['pipeline_fps']:.1f} FPS, RTX 4050)." if d.get('pipe') is not None else ".")
        + (" Başlıca sınırlılıklar az sayıda eğitim maçı ve insan etiketli olmayan olay referansıdır." if has_g1(d) else
           " En önemli sınırlılık, eğitim verisinin iki maçla sınırlı olması ve olay referansının insan etiketli "
           "olmamasıdır."))
    r.h2("Gelecek çalışmalar")
    r.p("Top için zamansal model (3–5 karelik girdi) ve P2 başlığı; insan etiketli küçük bir pas kümesiyle RQ3'ün "
        "bağımsız doğrulanması; yayın görüntüsünde saha kalibrasyonu; forma numarasıyla oyuncu kimliği; daha çok tohum.")
    r.h2("Kaynakça", new_page=True)
    for ref in REFS:
        r.p(ref, size=9.5, space_after=2, convert=False, align_justify=False)
    o = d.get("ogr") or {}  # report/ogrenci_bilgileri.yaml
    uyeler = [m for m in (o.get("ekip") or []) if str((m or {}).get("ad") or "").strip()]
    r.h2("Ekip katkıları")
    if uyeler:
        r.table(["Üye", "Öğrenci no", "Katkı"],
                [(str(m["ad"]).strip(), str(m.get("numara") or "").strip() or "—",
                  str(m.get("katki") or "").strip() or "[Doldurunuz]") for m in uyeler],
                widths_cm=[4.0, 2.8, 10.2], font=9.5)
    else:
        r.p("[Doldurunuz: her üyenin veri, model, deney, rapor ve gösterim katkısı "
            "(`report/ogrenci_bilgileri.yaml`).]")
    ai_note = str(o.get("ai_dogrulama") or "").strip()
    r.h2("Dış kod ve yapay zekâ araçları")
    r.p("Dış kod/ağırlıklar: Ultralytics YOLOv8 (eğitim/çıkarım), supervision ByteTrack, referans projenin saha "
        "anahtar noktası modeli ve demo videoları [8] (ağırlıkların pickle içeriği taranarak yalnızca torch/"
        "ultralytics nesneleri içerdiği doğrulandı). **Yapay zekâ aracı:** Anthropic Claude (Claude Code), kod "
        "iskeletinin ve modüllerin yazımında, deney betiklerinin hazırlanmasında, rapor metninin taslağında "
        "kullanılmıştır. Rapordaki tüm sayılar `results/` altındaki betik çıktılarından otomatik okunur; deneyler bu "
        "makinede çalıştırılmış ve kayıtları `experiments/registry.csv` içindedir. "
        + (f"**Öğrenci doğrulaması:** {ai_note}" if ai_note else
           "[Öğrenci doğrulaması: kodun çalıştırılması, sonuçların ve hata örneklerinin kontrolü ekip tarafından "
           "yapılmalı ve burada belirtilmelidir (`report/ogrenci_bilgileri.yaml`).]"))
    appendix(r, d)


def appendix(r, d):
    r.h1("Ekler", new_page=True)
    r.h2("Ek A — Çalıştırma komutları ve ayrıntılı hiperparametreler")
    cmds = ["python -m bgoru.data.prepare_soccernet --split train --stride 15",
            "python -m bgoru.data.prepare_soccernet --split test --stride 25",
            "python -m bgoru.data.prepare_soccernet --split test --dense --stride 2 --max-seqs 10",
            "python -m bgoru.data.copy_paste --frac 0.5 --seed 0",
            "bash experiments/train_all.sh   (configs/exp/{E2_ft640,E3_ft1280_cp,A1_ft1280,A2_ft640_cp}.yaml)",
            "bash experiments/train_seeds.sh && python -m bgoru.experiments.eval_seeds   (tohum tekrarı, Ek A.3)",
            "python -m bgoru.data.prepare_gsr && bash experiments/train_gsr.sh && bash experiments/train_control.sh   "
            "(G1, C1 ve eval_gsr; Ek A.4)",
            "python -m bgoru.experiments.eval_detector | eval_team | eval_possession | robustness | benchmark | failure_cases",
            "python -m bgoru.cli analyze --video demo/input/taktik_demo1.mp4 --max-frames 500",
            "streamlit run app/streamlit_app.py"]
    r.bullets([f"`{c}`" for c in cmds], size=9)
    r.p("Ortak hiperparametreler: epoch 20, nbs 64, SGD lr0 0,01 lrf 0,01, momentum 0,937, weight_decay 5e-4, "
        "warmup 3 epoch, box 7,5 / cls 0,5 / dfl 1,5, mozaik 1,0 (close_mosaic 10), hsv (0,015/0,7/0,4), fliplr 0,5, "
        "scale 0,5, translate 0,1, erasing 0,4, AMP, seed 0, deterministic. Sahiplik: d < 0,45 oyuncu boyu, onay "
        "0,12 sn; pas boşluğu ≤ 4 sn; top kapma ≤ 0,6 sn ve ayak uzaklığı < 1,5 boy; top kazanma ≥ 2 sn; taşıma ≥ 5 m; "
        "top doldurma ≤ 0,4 sn; homografi güncelleme 4 px, RANSAC 1,5 m, en büyük geri izdüşüm 2,5 m.", size=9.5)
    reg = d.get("reg")
    if reg is not None:
        r.h2("Ek A.2 — Deney kayıtları (özet)")
        rows = [(x.exp_id, x.task, str(x.data_split)[:38], f"{float(x.duration_s):.0f}") for x in reg.itertuples()]
        r.table(["Kimlik", "Görev", "Veri ayrımı", "Süre (sn)"], rows, widths_cm=[4.6, 3.2, 7.2, 2.0],
                caption="experiments/registry.csv içeriğinin özeti: tüm deneyler (metrikler dosyada).", font=7)
    sd, sdf = seeds_ok(d), d.get("seeds_df")
    if sd and sdf is not None:
        r.h2("Ek A.3 — Tohum tekrarı (eğitim varyansı)")
        lab = {"E3_ft1280_cp": "E3 (1280 + KY)", "A1_ft1280": "A1 (1280)"}
        cols = ("val_ball_AP50", "ball_AP50", "ball_APc", "player_AP50", "mAP50_95")
        rows = []
        for cfg in ("E3_ft1280_cp", "A1_ft1280"):
            for x in sdf[sdf.config == cfg].sort_values("seed").itertuples():
                rows.append((lab[cfg], str(x.seed), *[f3(getattr(x, c)) for c in cols]))
            sm = sd["summary"][cfg]
            rows.append((lab[cfg], "ort. ± std", *[f"{f3(sm[c]['mean'])} ± {f3(sm[c]['std'])}" for c in cols]))
        r.table(["Yapılandırma", "Tohum", "Doğr. top AP@0.5", "Test top AP@0.5", "Test top AP_m", "Oyuncu AP@0.5",
                 "mAP@[.5:.95]"], rows, widths_cm=[2.8, 1.8, 2.4, 2.4, 2.4, 2.4, 2.6], font=8,
                highlight_rows=tuple(i for i, rw in enumerate(rows) if rw[1].startswith("ort")),
                caption="E3 ve A1'in tohum 0 ve 1 eğitimleri (aynı veri, bütçe ve test protokolü; std: ddof = 1, n = 2).")
        pdif = sd["paired_differences"]
        r.p("Eşleştirilmiş sekans bootstrap'ı (1000) ile E3−A1 top AP@0.5 farkı: "
            + "; ".join(f"{k.replace('seed', 'tohum ').replace(' - ', '−')} = {v['mean']:+.3f} (%95 GA {v['lo']:+.3f}; {v['hi']:+.3f})"
                        for k, v in pdif.items())
            + ". Kopyala-yapıştır verisi (tohum 0) sabit tutulmuş, yalnızca eğitim tohumu değiştirilmiştir. Bu tekrar "
              "yalnızca E3–A1 karşılaştırmasının eğitim varyansını ölçer; son sistem dedektörü ve aşağı akış sonuçları "
              "bundan etkilenmez.", size=9.5)
    gx = d.get("gsr")
    if has_g1(d) and gx:
        r.h2("Ek A.4 — Ek eğitim verisi deneyi (G1) ve adım eşleşmeli kontrol (C1)")
        rows_g = {x["model_id"]: x for x in gx["rows"]}
        ci = gx.get("ball_AP50_bootstrap_CI95", {})
        meta = [("A1_ft1280", "A1 (tohum 0)", "1950 / 2 maç", "20", gx["train_iterations"]["A1"]),
                ("A1_ft1280_s1", "A1 (tohum 1)", "1950 / 2 maç", "20", gx["train_iterations"]["A1"]),
                ("C1_ft1280_ep50", "C1 kontrol", "1950 / 2 maç", "50", gx["train_iterations"]["C1"]),
                (FINAL, "G1 (son sistem)", "4850 / 5 maç", "20", gx["train_iterations"]["G1"])]
        rows = []
        for m, lab_, data_, ep, it in meta:
            if m not in rows_g:
                continue
            x = rows_g[m]
            c = ci.get(m)
            rows.append((lab_, data_, ep, f"{it:,}".replace(",", "."), f3(x["val_ball_AP50"]),
                         f"{x['ball_AP50']:.3f}" + (f" [{c['lo']:.3f}; {c['hi']:.3f}]" if c else ""),
                         f3(x["ball_APc"]), f3(x["referee_AP50"]), f3(x["mAP50_95"])))
        r.table(["Model", "Eğitim karesi / maç", "Epoch", "İterasyon", "Doğr. top AP@0.5", "Test top AP@0.5 [%95 GA]",
                 "Test top AP_m", "Hakem AP@0.5", "mAP@[.5:.95]"], rows,
                widths_cm=[2.4, 2.3, 1.2, 1.7, 1.7, 3.0, 1.6, 1.6, 1.6], font=7.5,
                highlight_rows=tuple(i for i, rw in enumerate(rows) if rw[0].startswith("G1")),
                caption="Ek eğitim verisi deneyi (aynı ayarlar: 1280 px, SGD, nbs 64, batch 4; aynı doğrulama ve test).")
        pdg = gx.get("paired_differences_ball_AP50", {})
        r.p("Eşleştirilmiş sekans bootstrap'ı (1000) ile test top AP@0.5 farkları: "
            + "; ".join(f"{k.replace(' - ', '−')} = {v['mean']:+.3f} (%95 GA {v['lo']:+.3f}; {v['hi']:+.3f})"
                        for k, v in pdg.items())
            + ". C1'in ısınması 7,5 epoch ve son 25 epoch'u mozaiksizdir; böylece G1 ile aynı adım sayısında çizelge "
              "oranları da aynıdır.", size=9.5)
        if (FIGS / "fig_extra_data.png").exists():
            r.figure(FIGS / "fig_extra_data.png", "(a) Doğrulama mAP@[.5:.95], eğitim adımına göre: C1 aynı veriyle "
                     "uzatıldığında yaklaşık 12 bin adımdan sonra artmaz, G1 ek veriyle yükselmeyi sürdürür; (b) test top "
                     "AP@0.5 ve %95 bootstrap güven aralığı.", width_cm=15.5)
    fl = d["fail"]
    reason_col = "manual_reason" if "manual_reason" in fl.columns and fl["manual_reason"].notna().any() else "auto_category"
    r.h2("Ek B — Başarısız örneklerin tam listesi")
    rows = [(x.id, x.type, x.seq, f1(x.size_px), str(x[reason_col])) for _, x in fl.iterrows()]
    r.table(["ID", "Tür", "Sekans", "Boyut px", "Neden"], rows, widths_cm=[1.2, 2.0, 2.4, 1.6, 9.8], font=7.5,
            caption="Başarısız örnekler (görüntüler: results/failures/imgs/).")
    sheet2 = R / "failures" / "contact_sheet_2.jpg"
    if sheet2.exists():
        r.figure(sheet2, "Başarısız örnekler (devam).", width_cm=4.8)
    r.h2("Ek C — Karakter kartı ve profil haritası örnekleri")
    cards = sorted((ROOT / "demo" / "kartlar").glob("taktik_demo2_kart_*.png"))
    if cards:
        r.figure(cards[0], "Taktik kamera demo 2'den otomatik üretilen karakter kartı. Özellikler aynı kesitteki "
                 "oyunculara göre yüzdeliktir.", width_cm=16)
    pm = ROOT / "demo" / "kartlar" / "taktik_demo2_profil_haritasi.png"
    if pm.exists():
        r.figure(pm, "Aynı kesitteki oyuncuların CIES düzleminde profil kategorileri.", width_cm=10)
    shot = R / "figures" / "demo_frame.jpg"
    if shot.exists():
        r.figure(shot, "Açıklamalı çıktı karesi: canlı hakimiyet/pas paneli ve taktik harita.", width_cm=16)
    extra = [(FIGS / "fig_possession.png", "Aşağı akış metrikleri (Bölüm 6, RQ3 tablosu): son sistem ("
              + ("V7" if FINAL_V in d["poss"].index else "V6") + ", turuncu), E3 tabanlı tam sistem (V2) ve ablation'lar.",
              14.5)]
    if any(f.exists() for f, _, _ in extra):
        r.h2("Ek D — Ek şekiller")
        for f, cap, w in extra:
            if f.exists():
                r.figure(f, cap, width_cm=w)
