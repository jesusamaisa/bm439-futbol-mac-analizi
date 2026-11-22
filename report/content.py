"""Rapor metni. Sayısal değerler build_report.load() ile results/ klasöründen gelir."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"


def f3(x):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.3f}"


def f1(x):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.1f}"


def pct(x):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"%{x * 100:.1f}"


FINAL = "G1_ft1280_gsr"  # son sistem dedektörü (src/bgoru/config.py: FINAL_DETECTOR)
FINAL_V = "V7_G1_final"  # son sistemle aşağı akış varyantı (eval_possession)


def final_poss(poss):
    """Son sistemin aşağı akış satırı; yoksa sırasıyla V6 (A1) ve V2 (E3)."""
    for v in (FINAL_V, "V6_A1_final", "V2_E3_full"):
        if v in poss.index:
            return poss.loc[v]


def has_g1(d) -> bool:
    """Ek eğitim verisi deneyi (G1, C1) ölçüldü mü."""
    return d["det"] is not None and all(k in d["det"].index for k in (FINAL, "C1_ft1280_ep50", "A1_ft1280"))         and f"{FINAL} - C1_ft1280_ep50" in d["boot"]["paired_differences"]


def gd(d, a, b):
    return d["boot"]["paired_differences"].get(f"{a} - {b}", {})


def _seeds2(d) -> bool:
    """E3 ve A1 için en az iki eğitim tohumu ölçüldü mü (results/detection/seed_repeats.json)."""
    s = d.get("seeds")
    return bool(s) and all(s["summary"].get(k, {}).get("n_seeds", 0) >= 2 for k in ("E3_ft1280_cp", "A1_ft1280"))


def cp_seed_status(d):
    """Kopyala-yapıştır etkisinin (E3−A1, test top AP@0.5) tohumlar arası durumu.

    None: tohum tekrarı yok · 'neg_all'/'pos_all': her tohumda anlamlı düşüş/artış · 'neg_some': en az bir tohumda
    anlamlı düşüş, hiçbirinde anlamlı artış yok · 'none': hiçbir tohumda anlamlı fark yok · 'mixed': diğer durumlar.
    """
    if not _seeds2(d):
        return None
    per = [d["seeds"]["paired_differences"].get(f"seed{s}: E3 - A1") for s in (0, 1)]
    if any(p is None for p in per):
        return None
    neg, pos = [p["hi"] < 0 for p in per], [p["lo"] > 0 for p in per]
    if all(neg):
        return "neg_all"
    if all(pos):
        return "pos_all"
    if any(neg) and not any(pos):
        return "neg_some"
    if not any(neg) and not any(pos):
        return "none"
    return "mixed"


def a1_val_better_all_seeds(d) -> bool:
    """A1, her tohumda doğrulama top AP@0.5'te E3'ten yüksek mi (model seçimi doğrulamaya dayanır)."""
    sdf = d.get("seeds_df")
    if sdf is None:
        return False
    v = sdf.pivot(index="seed", columns="config", values="val_ball_AP50")
    return bool(len(v) >= 2 and (v["A1_ft1280"] > v["E3_ft1280_cp"]).all())


# =====================================================================================================
def cover(r, d):
    r.title("Bilgisayarlı Görü")
    r.p("**Futbol Videolarında Anlık Top Hakimiyeti, Pas Sıklığı ve Oyuncu Teknik Profili Çıkarımı**",
        align_justify=False, size=14)
    r.p("")
    o = d.get("ogr") or {}  # report/ogrenci_bilgileri.yaml; boş alanlar noktalı satır olarak kalır
    g = lambda k, dots="………………………………………………": str(o.get(k) or "").strip() or dots
    uyeler = "; ".join(f"{str(m['ad']).strip()} ({str(m.get('numara') or '').strip() or '…'})"
                       for m in (o.get("ekip") or []) if str((m or {}).get("ad") or "").strip())
    rows = [("Proje başlığı", "Bilgisayarlı Görü: Futbol Videolarında Anlık Top Hakimiyeti, Pas Sıklığı ve Oyuncu "
                              "Teknik Profili Çıkarımı"),
            ("Ders ve dönem", f"BM439 Bilgisayarlı Görme — {g('donem', '………………………')}"),
            ("Proje konusu ve numarası",
             f"Futbol video analizi (nesne tespiti, takip, olay çıkarımı) — No: {g('proje_no', '………')}"),
            ("Ekip üyeleri ve öğrenci numaraları", uyeler or "………………………………………………"),
            ("Öğretim elemanı", g("ogretim_elemani")),
            ("Teslim tarihi", g("teslim_tarihi", "………………………")),
            ("Kod ve gösterim bağlantısı", g("baglanti"))]
    r.table(["Alan", "Bilgi"], rows, widths_cm=[5.5, 11.5], font=10.5)
    r.page_break()


# =====================================================================================================
def sec1(r, d):
    det, poss, team = d["det"], d["poss"], d["team"]
    e1, e2, e3 = det.loc["E1"], det.loc["E2_ft640"], det.loc["E3_ft1280_cp"]
    bo = d["boot"]["paired_differences"]
    diff32 = bo.get("E3_ft1280_cp - E2_ft640", {})
    tp = team[(team.set == "dense") & (team.config == "T5_local_mask")].iloc[0]
    tb = team[(team.set == "dense") & (team.config == "T1_baseline")].iloc[0]
    pv = final_poss(poss)
    a1 = det.loc["A1_ft1280"] if "A1_ft1280" in det.index else None
    d31 = bo.get("E3_ft1280_cp - A1_ft1280", {})
    abl = ""
    if a1 is not None and d31:
        st = cp_seed_status(d)
        if d31.get("hi", 0) < 0 and st == "neg_some":
            sp = d["seeds"]["paired_differences"]
            d1, dm = sp["seed1: E3 - A1"], sp["tohum ortalaması: E3 - A1"]
            abl = (f"Ablation, kazancın yüksek çözünürlükten geldiğini, kopyala-yapıştırın ise hiçbir tohumda katkı "
                   f"sağlamadığını göstermiştir (artırımsız A1: {f3(a1['ball_AP50'])}; E3−A1 tohum 0'da "
                   f"{d31.get('mean'):+.3f} [{d31.get('lo'):+.3f}; {d31.get('hi'):+.3f}], tohum 1'de {d1['mean']:+.3f} "
                   f"[{d1['lo']:+.3f}; {d1['hi']:+.3f}]). ")
        elif d31.get("hi", 0) < 0:
            abl = (f"Ablation, kazancın tamamen yüksek çözünürlükten geldiğini, top kopyala-yapıştır artırımının ise "
                   f"AP@0.5'i anlamlı biçimde **düşürdüğünü** göstermiştir (artırımsız 1280 px, A1: "
                   f"{f3(a1['ball_AP50'])}; E3−A1 = {d31.get('mean'):+.3f}, %95 GA [{d31.get('lo'):+.3f}; "
                   f"{d31.get('hi'):+.3f}]); bu nedenle sonraki adımda A1 ayarları kullanılmıştır. ")
        else:
            abl = (f"Ablation'da artırımsız 1280 px model (A1) {f3(a1['ball_AP50'])} değerini vermiştir "
                   f"(E3−A1 = {d31.get('mean'):+.3f}). ")
    if has_g1(d):
        g, dgc = det.loc[FINAL], gd(d, FINAL, "C1_ft1280_ep50")
        abl += (f"A1 ayarlarıyla eğitime SoccerNet-GSR'dan üç yeni maç (2900 kare) eklemek top AP@0.5'i "
                f"{f3(g['ball_AP50'])} değerine çıkarmıştır; eşit adımlı kontrole göre fark {dgc['mean']:+.3f} "
                f"[{dgc['lo']:+.3f}; {dgc['hi']:+.3f}] olduğundan kazanç ek veriden gelir. Doğrulamada en iyi olan bu "
                f"model (G1) son sistemdir. ")
    r.h1("1 Özet ve giriş")
    r.h2("Özet")
    r.p(f"Bu çalışma, futbol videolarından oyuncu, hakem ve topu tespit edip takip eden, takımları ayıran, "
        f"**top hakimiyeti yüzdesini ve pas sıklığını anlık** ölçen ve oyuncular için teknik profil ile **karakter "
        f"kartı** üreten uçtan uca bir sistem sunar. Temel darboğaz top tespitidir (SoccerNet-Tracking yayın "
        f"görüntülerinde topların %76'sı 20 pikselden küçüktür); ana araştırma sorusu buna odaklanır. Resmî "
        f"SoccerNet-Tracking-2023 ayrımı korunmuş, doğrulama kümesi maç bazında ayrılmış, tüm eşikler doğrulamada "
        f"seçilmiştir. Aynı test protokolünde (49 sekans, 3 maç) COCO YOLOv8s (temel), 640 px ince ayar (alternatif) "
        f"ve önerilen 1280 px + top kopyala-yapıştır karşılaştırılmıştır. Top AP@0.5 temel yöntemde "
        f"{f3(e1['ball_AP50'])}, alternatifte {f3(e2['ball_AP50'])}, önerilen yöntemde {f3(e3['ball_AP50'])} "
        f"olarak ölçülmüştür (E3−E2 farkı {f3(diff32.get('mean'))}, %95 GA [{f3(diff32.get('lo'))}; "
        f"{f3(diff32.get('hi'))}]). " + abl + f"Önerilen takım ataması (yerel saha maskesi + LAB + iz oylaması) "
        f"yoğun testte doğruluğu {f3(tb['acc_mean'])} değerinden {f3(tp['acc_mean'])} değerine çıkarmıştır. "
        f"Son sistem, GT kutulardan üretilen vekil referansa göre pasları {f3(pv['pass_F1_mean'])} F1 ve kare bazlı "
        f"top kontrolünü {f3(pv['control_agreement_mean'])} uyumla ölçmüş; klip başına hakimiyet yüzdesi hatası "
        f"({f1(pv['poss_err_pp_mean'])} yp) ise güvenilir bir iyileşme göstermemiştir.")
    r.h2("Anahtar kelimeler")
    r.p("futbol analitiği; küçük nesne tespiti; YOLOv8; çoklu nesne takibi; top hakimiyeti; oyuncu profilleme")
    r.h2("Problem ve kullanım senaryosu")
    r.p("**Girdi:** tek kameralı futbol videosu (yayın veya taktik kamera, 720p–1080p). **Çıktı:** (i) açıklamalı "
        "video — oyuncu kimlikleri, takım renkleri, top, top sahibi ve taktik harita; (ii) her kare güncellenen "
        "takım bazlı **top hakimiyeti yüzdesi** (kümülatif ve son 60 sn), **pas sayısı, pas/dakika ve pas isabeti**; "
        "(iii) olay kaydı (pas, başarısız pas, araya girme, top kapma, top kazanma, top taşıma, çalım, şut, orta, "
        "kilit pas); (iv) oyuncu başına teknik profil kategorisi ve karakter kartı (Şekil 1). Sistem; antrenörlerin "
        "maçın belirli bir kesitinde (ör. son 10 dakika) takımın topa ne kadar sahip olduğunu ve pas temposunu "
        "anlık izlemesi, maç sonrası oyuncu raporları üretmesi ve amatör maç kayıtlarının etiketlenmesi ihtiyacını "
        "hedefler.")
    r.figure(FIGS / "fig_example_prediction.png", "Problemin girdisi, etiketi ve beklenen çıktısı: SoccerNet test "
             "karesi (a) GT kutuları (mavi oyuncu, sarı hakem, kırmızı top) ve (b) önerilen modelin (E3) tahmini.",
             width_cm=13)
    r.h2("Araştırma sorusu ve hipotez")
    r.p("**RQ1 (ana):** SoccerNet-Tracking test kümesinde, futbola özgü ince ayar + yüksek çözünürlüklü girdi "
        "(1280 px) + top kopyala-yapıştır artırımı, COCO ön-eğitimli YOLOv8s'e ve standart 640 px ince ayara göre "
        "**top sınıfı AP@0.5** değerini ne kadar artırır? *Hipotez:* top küçük ve az temsil edilen bir sınıf "
        "olduğundan çözünürlük artışı ve top örneklerinin çoğaltılması top AP'sini belirgin biçimde artırır; "
        "oyuncu sınıfındaki kazanç sınırlı kalır. **RQ2:** Gövde kırpması + yerel saha maskesi + LAB + iz oylaması "
        "takım atama doğruluğunu artırır mı? **RQ3:** Tespit ve takım hataları hakimiyet yüzdesi ve pas ölçümüne ne "
        "kadar yansır? Ek deneyde, eğitim kümesine yeni maçlar eklemenin etkisi eğitim süresinden ayrıştırılarak "
        "ölçülmüştür. Ana metrik top AP@0.5 (↑); yardımcı metrikler oyuncu/hakem AP@0.5, mAP@[.5:.95], top "
        "recall, ms/kare (↓) ve GPU belleğidir (↓).")
    r.h2("Kapsam ve katkılar")
    r.bullets([
        "Yalnızca gereken kareleri indiren seçici SoccerNet hazırlayıcı ve **maç bazlı sızıntısız** ayrım.",
        "Özgün veri hazırlama bileşeni olarak tasarlanıp ablation ile sınanan, yalnızca çim bölgelerine dairesel "
        "yumuşak maskeyle **top kopyala-yapıştır** artırımı (eğitim bütçesi sabit) — sonuç olumsuzdur ve raporlanmıştır.",
        "Aynı yayın kliplerine dayanan SoccerNet-GSR'dan klip düzeyinde sızıntı denetimiyle seçilen **3 maçlık ek "
        "eğitim verisi** ve etkisinin adım sayısı eşlenmiş bir kontrol modeliyle ayrıştırılması.",
        "Kural tabanlı, perspektiften bağımsız (oyuncu boyuna ölçekli) **anlık top hakimiyeti ve olay** motoru; "
        "taktik kamerada metre cinsinden homografi ile hız/mesafe.",
        "CIES yönteminden uyarlanan aksiyon vektörleriyle **profil kategorileri** ve yüzdelik tabanlı **karakter "
        "kartı**; Streamlit arayüzü.",
        "Kapsam dışı: oyuncu kimliğinin (isim/forma no) tanınması, hava topu mücadeleleri, insan etiketli olay "
        "verisiyle doğrulama (vekil referans kullanılmıştır)."])


# =====================================================================================================
def sec2(r, d):
    r.h1("2 İlgili çalışmalar")
    rows = [
        ("[1] Cioppa ve ark., 2022", "SoccerNet-Tracking: 200 yayın klibi (30 sn, 1080p), oyuncu/hakem/top kutuları, "
         "iz kimlikleri ve takım bilgisi; MOT kıyaslaması.", "Olay (pas) etiketi yok; top küçük ve sık örtülü. "
         "Projenin verisi ve resmî ayrımı."),
        ("[2] Zhang ve ark., 2022 (ByteTrack)", "Düşük güvenli kutuları da eşleştiren iki aşamalı ilişkilendirme; "
         "MOT17/20.", "Benzer formalı oyuncularda kimlik değişimi. Oyuncu takibinde kullanıldı."),
        ("[3] Kisantal ve ark., 2019", "Küçük nesneler için aşırı örnekleme ve kopyala-yapıştır; COCO küçük nesne AP "
         "artışı.", "Maske gerektirir, bağlamdan bağımsız yapıştırır. Topu yalnızca çime, maskesiz ve dairesel "
         "harmanlama ile yapıştırarak uyarlandı."),
        ("[4] Akyon ve ark., 2022 (SAHI)", "Dilimleyerek çıkarım ve ince ayar ile küçük nesne tespiti.",
         "Çıkarım maliyeti dilim sayısıyla artar; gerçek zamanlılık için tek geçişli yüksek çözünürlük tercih edildi."),
        ("[5] Istasse ve ark., 2019", "Takım ayrımı için ilişkisel gömme (associative embedding); basketbol.",
         "Eğitim verisi ister. Eğitimsiz, renk tabanlı ve iz oylamalı yöntem seçildi."),
        ("[6] Vidal-Codina ve ark., 2022", "İzleme (tracking) verisinden kural tabanlı pas/şut olay tespiti.",
         "Konumlar optik izleme sisteminden gelir. Projede konumlar videodan kestirilir; kurallar benzer mantıkla "
         "uyarlandı."),
        ("[7] CIES Football Observatory", "Aksiyon türlerinin Hücum–Savunma / Merkez–Kanat eksenlerinde vektörlerle "
         "temsiliyle teknik profil.", "Olay verisi (Opta tipi) kullanır. Videodan çıkarılan olaylarla uygulandı."),
    ]
    r.table(["Kaynak", "Yöntem ve veri", "Sınırlılık ve ilişki"], rows, widths_cm=[3.6, 6.6, 6.8],
            caption="İlgili çalışmaların sentezi.", font=8.5)
    r.h2("Literatürdeki ihtiyaç")
    r.p("Yayın görüntülerinde top, oyunculara göre çok küçük (çoğunlukla 10–20 px) ve az örneklidir; genel amaçlı "
        "dedektörler ve standart 640 px girdi bu ölçekte bilgi kaybeder. Olay tespiti çalışmaları [6] ise hassas "
        "izleme verisi varsayar. Videodan hakimiyet/pas çıkarımında hatanın büyük kısmı top tespitinden "
        "kaynaklandığından, araştırma sorusu küçük nesne tespitine ve bu hatanın aşağı akış metriklerine "
        "yansımasına odaklanmıştır.")
    r.h2("Yöntem seçiminin gerekçesi")
    r.p("Anlık çalışma gereksinimi tek aşamalı, çapa-içermeyen YOLOv8'i öne çıkarır: dizüstü GPU'da 1280 px'de bile "
        "kare başına onlarca milisaniyede çalışır ve COCO ön-eğitimiyle küçük veriyle ince ayarlanabilir. "
        "Transformer tabanlı dedektörler (DETR ailesi) küçük nesnelerde ve 6 GB bellekte bu bütçeye sığmaz. "
        "Model boyutu sabit tutulmuş (YOLOv8s), iyileştirme veri ve girdi çözünürlüğü üzerinden yapılmıştır.")


# =====================================================================================================
def sec3(r, d):
    cp = d["cp"] or {}
    r.h1("3 Veri seti ve hazırlık")
    r.h2("Veri kaynağı")
    r.p("SoccerNet-Tracking-2023 [1], Hugging Face deposu `SoccerNet/SN-Tracking-2023` (train.zip 9,58 GB, "
        "test.zip 8,71 GB; erişim tarihi 17.11.2026). Veri SoccerNet'in araştırma amaçlı kullanım koşullarına "
        "tabidir; bu nedenle teslim paketinde ham görüntüler yerine **indirme betiği ve ayrım manifestleri** "
        "verilmiştir. Arşivlerin tamamı indirilmemiş; ZIP merkezi dizini okunup yalnızca seçilen karelerin bayt "
        "aralıkları HTTP Range ile çekilmiştir (`bgoru/data/remote_zip.py`). Taktik kamera gösterimi için referans "
        "projenin [8] iki demo videosu ve saha anahtar noktası modeli kullanılmıştır (yalnızca nitel test)."
        + (" **Ek eğitim verisi:** SoccerNet-GSR-2025 [15] (`SoccerNet/SN-GSR-2025`, `valid.zip`, 11,2 GB, SHA-256 "
           "doğrulandı, erişim 18.11.2026; depo lisansı GPL v3, görüntüler aynı araştırma koşullarına tabi); kare başına "
           "kutuları aynı sınıflara eşlenmiştir (`bgoru/data/prepare_gsr.py`)." if has_g1(d) else ""))
    r.h2("Veri ve anotasyon yapısı")
    r.p("Örnek birimi 1920×1080 bir karedir. Her sekans 750 karelik (30 sn, 25 fps) bir klip; etiketler MOT "
        "biçiminde eksen hizalı kutu + iz kimliğidir. `gameinfo.ini` her izi *oyuncu (sol/sağ takım)*, *kaleci*, "
        "*hakem*, *top* veya *diğer* olarak tanımlar. Sınıflar: 0 = oyuncu (kaleciler dahil), 1 = hakem, 2 = top; "
        "*diğer* izleri atılmıştır. Takım etiketi (sol/sağ) RQ2 için ayrıca saklanmıştır. Eğitim karelerinin %97'sinde "
        "top etiketlidir; top kutusunun uzun kenarının medyanı 15 px, %12'si 10 px'den küçüktür "
        "(Şekil 2b). Olay (pas) etiketi bulunmadığından RQ3 için vekil referans üretilmiştir (Bölüm 5).")
    rows = [("Eğitim", "1950 kare (her 15. kare)", "39 sekans / 2 maç (ID 6, 9)", "Resmî train ayrımı, maç bazlı"),
            ("Doğrulama", "900 kare (her 15. kare)", "18 sekans / 1 maç (ID 4)", "Resmî train'den tohum 0 ile maç seçimi"),
            ("Test", "1470 kare (her 25. kare)", "49 sekans / 3 maç (ID 7, 8, 11)", "Resmî test ayrımı, dokunulmadı"),
            ("Yoğun test (RQ2–3)", "3750 kare (12,5 fps)", "10 test sekansı", "Test sekanslarından rastgele (tohum 0)")]
    if has_g1(d):
        rows.insert(1, ("Ek eğitim (G1)", "2900 kare (her 15. kare)", "58 sekans / 3 maç (ID 2, 3, 5)",
                        "SoccerNet-GSR valid; kümelerimizle kesişmez"))
    r.table(["Ayrım", "Örnek sayısı", "Grup sayısı", "Ayrım kuralı"], rows, widths_cm=[3.2, 4.0, 4.6, 5.2],
            caption="Veri ayrımları (manifest: data/soccernet/manifests/*_split.json).", font=8.5)
    r.h2("Sızıntının önlenmesi")
    r.p("Aynı maçtaki klipler benzer forma, stadyum ve ışık koşulu paylaştığından ayrım **maç kimliği (gameID)** "
        "üzerinden yapılmıştır: eğitim (6, 9), doğrulama (4) ve test (7, 8, 11) maçları kesişmez. Bir sekanstan "
        "türetilen tüm kareler tek bir ayrımda kalır. Kopyala-yapıştır top bankası yalnızca eğitim karelerinden "
        "oluşturulmuştur. Model seçimi (en iyi epoch), girdi boyutu seçimi ve güven eşikleri yalnızca doğrulama "
        "kümesinde yapılmış; test kümesi her yapılandırma için bir kez değerlendirilmiştir."
        + (" SoccerNet-GSR aynı yayın kliplerinden türetildiği için klipler başlangıç zamanı ve aksiyon konumuyla "
           "eşleştirilmiştir: GSR'ın train ayrımı bizim eğitim ve doğrulama kliplerimizle, test ayrımı ise test "
           "kümemizle aynıdır. Bu nedenle yalnızca hiçbir kümemizle kesişmeyen valid ayrımı (maç 2, 3, 5) eğitime "
           "eklenmiş; dönüştürücü kesişme bulursa durur (`data/soccernet_gsr/manifests/leak_check.json`)."
           if has_g1(d) else ""))
    r.h2("Ön işleme ve artırma")
    r.p("Kareler oran korunarak 640 veya 1280 px'e harf kutusu (letterbox) ile ölçeklenir, [0,1] aralığına "
        "normalize edilir. Tüm ince ayarlarda Ultralytics varsayılan çevrimiçi artırımları aynıdır: mozaik (son 10 "
        "epoch kapalı), HSV renk oynatma (h 0,015 / s 0,7 / v 0,4), yatay çevirme (0,5), ölçek (±0,5), öteleme "
        "(0,1), rastgele silme (0,4). **Önerilen çevrimdışı artırım:** eğitim görüntülerinin "
        f"%{int(cp.get('frac', 0.5) * 100)}'sine ({cp.get('augmented_images', '—')} görüntü) eğitim kümesinden "
        f"toplanan {cp.get('bank_size', '—')} top kesitinden 1–3 adet, ×0,8–1,25 ölçekle, HSV çim maskesi içinde "
        f"ve mevcut nesnelerle çakışmayan konumlara dairesel yumuşak maskeyle yapıştırılmıştır (toplam "
        f"{cp.get('pasted_balls', '—')} top). Görüntü sayısı ve epoch sayısı değişmediğinden eğitim bütçesi sabittir.")
    r.figure(FIGS / "fig_data_distribution.png", "(a) Ayrımlara göre sınıf dağılımı (log ölçek), (b) eğitimdeki "
             "top boyutu dağılımı: toplar küçük ve az temsil edilen sınıftır.", width_cm=12.5)


# =====================================================================================================
def sec4(r, d):
    r.h1("4 Yöntem ve mimari")
    r.h2("Sistem akışı")
    r.p("Her kare için (Şekil 3): (1) YOLOv8s oyuncu/hakem/top kutularını üretir; (2) ByteTrack [2] oyunculara "
        "kalıcı kimlik verir; (3) takım ataması gövde bölgesinden çim maskeli LAB rengiyle yapılır ve iz boyunca "
        "oylanır; (4) taktik kamerada YOLOv8M saha modeli 28 anahtar noktayı bulur, RANSAC homografisi görüntüyü "
        "105×68 m sahaya izdüşürür; (5) top izleyici tek topu seçer ve 0,4 sn'ye kadar boşlukları sabit hızla "
        "doldurur; (6) top sahipliği ve olay kuralları anlık istatistikleri günceller; (7) kesit sonunda profil "
        "ve kartlar üretilir.")
    r.figure(FIGS / "fig_architecture.png", "Sistem mimarisi. Turuncu kutular kullanıcıya sunulan çıktılardır.",
             width_cm=14)
    rows = [("Girdi ve boyutlar", "1920×1080 RGB kare → harf kutusu 1280×1280 (E3, A1, G1) veya 640×640 (E2, A2)"),
            ("Backbone", "YOLOv8s CSPDarknet (C2f blokları, SPPF); COCO ön-eğitimli, 9,84 M parametre"),
            ("Neck veya füzyon", "PAN-FPN; P3/P4/P5 (adım 8/16/32) öznitelik birleştirme"),
            ("Head ve çıktı", "Ayrık (decoupled), çapa-içermeyen başlık; 3 sınıf; DFL kutu regresyonu; NMS IoU 0,7"),
            ("Önerilen bileşen", "Yüksek çözünürlük (1280) + top kopyala-yapıştır artırımı; çıkarımda tek top "
                                 "seçimi ve kısa boşluk doldurma; LAB + çim maskesi + iz oylamalı takım ataması")]
    r.table(["Bileşen", "Projedeki karşılığı"], rows, widths_cm=[3.8, 13.2], caption="Model bileşenleri.", font=8.5)
    r.h2("Temel yöntem, alternatif ve iyileştirme")
    r.p("**E1 Temel:** Ultralytics YOLOv8s (v8.3.6) COCO ağırlıkları, ince ayarsız; *person* → oyuncu, *sports "
        "ball* → top (hakem sınıfı yok, oyuncu AP'si oyuncu+hakem GT'sine karşı ölçülür); girdi boyutu "
        "{640, 1280} arasından doğrulamada seçilir. **E2 Alternatif:** aynı ağırlıklardan SoccerNet ince ayarı, "
        "640 px. **E3 Önerilen:** aynı ince ayar, 1280 px + kopyala-yapıştır. **A1/A2** ablation'larında E3'ün "
        "bir bileşeni çıkarılır. **R Referans:** referans projenin taktik kamera verisiyle eğitilmiş YOLOv8L "
        "modeli [8]; farklı ön eğitim verisi ve model boyutu nedeniyle yalnızca alan değişimi referansıdır. "
        + ("**G1 Ek veri:** A1'in tüm ayarları, eğitim kümesine 3 maç (2900 kare) eklenmiş. Epoch sayısı sabit "
           "kaldığından G1 yaklaşık 2,5 kat daha çok gradyan adımı görür; bu yüzden **C1 Kontrol** modeli A1 verisiyle "
           "aynı adım sayısına kadar (50 epoch; ısınma ve mozaiksiz son bölüm oranları aynı) eğitilmiştir. "
           if has_g1(d) else "")
        + "**Son sistem:** uygulamada kullanılacak dedektör, yalnızca doğrulama top AP@0.5 değerine göre seçilmiştir ("
        + ("G1" if has_g1(d) else "A1") + ").")
    r.h2("Öğrenme stratejisi ve kayıp")
    r.p("Tüm katmanlar eğitilir (dondurma yok); COCO'dan 313/391 tensör aktarılır, sınıf başlığı yeniden "
        "başlatılır. Kayıp = 7,5·CIoU + 1,5·DFL + 0,5·BCE (sınıf). SGD (momentum 0,937, ağırlık çürümesi 5e-4), "
        "lr0 = 0,01, 3 epoch ısınma, doğrusal azalma (lrf 0,01), 20 epoch, nominal batch 64 (gradyan biriktirme; "
        "fiziksel batch 1280 px'de 4, 640 px'de 16), karışık hassasiyet (AMP), tohum 0, deterministik mod. "
        "Damıtma veya sözde etiketleme kullanılmamıştır.")
    r.h2("Çıkarım ve göreve özel işlemler")
    r.p("**Eşikler:** sınıf başına güven eşiği doğrulamada F1'i en büyükleyen değerdir. **Takip:** ByteTrack "
        "(etkinleştirme 0,25, eşleşme 0,8, 2 sn kayıp tamponu). **Takım:** gövde kırpması (genişliğin orta %50'si, "
        "yüksekliğin %15–55'i), HSV çim maskesi, LAB medyan rengi; ilk 10 karede K-Means(2) + MAD tabanlı aykırı "
        "ayıklama (kaleciler), iz başına güven ağırlıklı oy. **Top sahipliği:** d = uzaklık(top, ayak bölgesi) / "
        "oyuncu boyu; d < 0,45 olan en yakın oyuncu 0,12 sn boyunca aday kalırsa sahip olur (histerezis). "
        "**Olaylar:** sahiplik A→B geçişinde aynı takım → pas; farklı takım ve ≤0,6 sn + yakın → top kapma; "
        "≥2 sn boşluk → top kazanma; diğer → araya girme + başarısız pas. Homografi varsa ≥5 m taşıma, rakip "
        "baskısında korunan taşıma (çalım), kale ağzına giden top (şut), kanattan ceza sahasına top (orta) ve "
        "ceza sahasında karşılanan/şutla biten pas (kilit pas) eklenir. **Homografi:** ortak ≥4 noktanın ortalama "
        "yer değiştirmesi 4 px'i aşmadıkça matris korunur, aşınca RANSAC ile yenilenir, medyan geri izdüşüm "
        "hatası 2,5 m'yi aşan matris reddedilir ve 0,6 katsayısıyla yumuşatılır. **Profil:** aksiyon sıklıkları "
        "kesit ortalamasına bölünüp CIES eksenlerindeki birim vektörlerle toplanır; açı 8 kategoriden birini, "
        "büyüklük <0,18 ise *Dengeli*'yi verir. Kart özellikleri aynı kesitteki oyunculara göre yüzdeliktir; "
        "GENEL puan profile göre ağırlıklı ortalamadır (Ek C).")


# =====================================================================================================
def sec5(r, d):
    env = d["env"]
    r.h1("5 Deney düzeni")
    rows = [("Donanım, işletim sistemi, GPU belleği", f"Intel Core i5-13500H, 31,7 GB RAM, {env['gpu']} "
             f"({env['gpu_mem_gb']:.0f} GB); {env['os']}"),
            ("Python, framework, CUDA", f"Python `{env['python']}`, PyTorch `{env['torch']}`, Ultralytics "
             f"`{env['ultralytics']}`, CUDA `{env['cuda_build']}` (sürücü `{env.get('nvidia_driver', '—')}`), cuDNN "
             f"`{env['cudnn']}`"),
            ("Girdi boyutu ve batch size", "640 (batch 16) / 1280 (batch 4); nominal batch 64"),
            ("Optimizer, öğrenme oranı, scheduler", "SGD, lr0 0,01, momentum 0,937, 3 epoch ısınma, doğrusal azalma"),
            ("Epoch, erken durdurma, regularizasyon", "20 epoch (sabit bütçe; C1 kontrolü 50), erken durdurma yok, "
                                                     "ağırlık çürümesi 5e-4, veri artırma"),
            ("Tohumlar ve tekrar sayısı",
             "Tohum 0, deterministik; E3 ve A1 ayrıca tohum 1 ile (Ek A.3), diğerleri 1 eğitim; test belirsizliği "
             "sekans bazlı bootstrap ile (1000 örnek)" if _seeds2(d) else
             "Tohum 0, deterministik; her yapılandırma 1 eğitim (bütçe nedeniyle); "
             "belirsizlik sekans bazlı bootstrap ile (1000 örnek)"),
            ("Model seçimi ve eşik belirleme", "Doğrulama verisi (en iyi epoch, girdi boyutu, güven eşikleri)")]
    r.table(["Eğitim ve ortam bilgisi", "Değer"], rows, widths_cm=[5.5, 11.5], caption="Deney ortamı.", font=8.5)
    r.h2("Metrikler ve hesaplama protokolü")
    r.p("Tespit, COCO tarzı 101 noktalı enterpolasyonla **sınıf bazında AP@0.5** ve AP@[.5:.95] ile ölçülür; tüm "
        "modeller aynı kodla (aynı NMS: IoU 0,7, en fazla 300 kutu, güven ≥ 0,001) değerlendirilir. Top küçük "
        "olduğundan AP'ye ek olarak doğrulamada seçilen eşikte recall ve top boyutu kovalarına (<10, 10–20, ≥20 px) "
        "göre recall raporlanır. Belirsizlik, test sekansları yeniden örneklenerek (1000 bootstrap) %95 güven "
        "aralığı ve **eşleştirilmiş fark** olarak verilir. RQ2'de GT kutular ve GT iz kimlikleri kullanılarak "
        "takım doğruluğu (sekans başına en iyi etiket permütasyonu, kaleciler hariç; ilk 10 ısınma karesi tüm "
        "yöntemlerde dışarıda) ölçülür. RQ3'te **vekil referans**, aynı hakimiyet/olay kurallarının GT oyuncu "
        "kutuları, GT takımları ve GT top kutusuyla çalıştırılmasıyla elde edilir; ölçülen hata algılama, takip "
        "ve takım ataması hatalarının yansımasıdır. Metrikler: hakimiyet % mutlak hatası (yüzde puanı, ↓), kare "
        "bazlı kontrol uyumu (↑), pas olayı F1 (±1 sn, aynı takım, ↑).")
    rows = [("Karşılaştırma (E1, E2, E3)", "Ön eğitim / ince ayar, girdi boyutu, artırım", "Model (YOLOv8s), veri, "
             "epoch, nominal batch, optimizer, tohum, test protokolü"),
            ("Ablation 1 (A1)", "Kopyala-yapıştır kaldırıldı (1280 px)", "E3'ün diğer tüm ayarları"),
            ("Ablation 2 (A2)", "Yüksek çözünürlük kaldırıldı (640 px + kopyala-yapıştır)", "E3'ün diğer tüm ayarları"),
            *([("Ek veri (G1) ve adım kontrolü (C1)", "Eğitime 3 maç eklendi (G1); yalnızca eğitim süresi ×2,5 (C1)",
                "A1'in diğer tüm ayarları, doğrulama ve test")] if has_g1(d) else []),
            ("Ablation (takım, aşağı akış)", "İz oylaması / çim maskesi / LAB; zamansal top doldurma; histerezis",
             "Aynı dedektör, eşik, sekanslar"),
            ("Zorlu koşul", "Gece, hareket bulanıklığı, JPEG q=15, uzak kamera (×0,5), top boyutu kovaları",
             "Aynı modeller ve doğrulama eşikleri")]
    r.table(["Deney", "Değişen unsur", "Sabit koşullar"], rows, widths_cm=[3.8, 6.4, 6.8],
            caption="Deney tasarımı.", font=8.5)
    r.h2("Verimlilik ölçümü")
    r.p(f"Cihaz: {env['gpu']}; batch 1, FP16; 20 ısınma + 200 ölçüm; süre `model.predict` uçtan uca (ön işleme + "
        "ileri geçiş + NMS), `torch.cuda.synchronize` ile; bellek `torch.cuda.max_memory_allocated`. Tam hat "
        "(tespit + takip + takım + homografi + olaylar + çizim) taktik kamera videosunda 20 ısınma karesinden "
        "sonra 280 karede ölçülmüştür.")


def write_all(r, d):
    cover(r, d)
    sec1(r, d)
    sec2(r, d)
    sec3(r, d)
    sec4(r, d)
    sec5(r, d)
    from content_results import sec6, sec7, sec8
    sec6(r, d)
    sec7(r, d)
    sec8(r, d)
