# Bilgisayarlı Görü — Proje Adımları

**Proje:** Futbol maç videolarından oyuncu/hakem/top tespiti, takım ataması, taktik harita,
**anlık top hakimiyeti ve pas sıklığı ölçümü**, oyuncuların **teknik profillere (kategorilere)
ayrılması** ve her oyuncu için **karakter kartı** üretimi.

Bu belge projenin baştan sona tüm adımlarını, BM439 *Proje Ortak Koşulları* ile eşleştirerek listeler.
Her adımın yanında projede hangi dosyanın o adımı gerçekleştirdiği yazılıdır.

---

## 0. Kaynakların incelenmesi

| Kaynak | Ne alındı | Ne eklendi / değiştirildi |
|---|---|---|
| YouTube: *Computer Vision for Football Analysis in Python with YOLOv8 & OpenCV* (Hmzbo) | YOLOv8 ile oyuncu/hakem/top + saha anahtar noktası tespiti, homografi ile taktik harita, LAB renk uzaklığı ile takım tahmini, homografi güncelleme koşulu, top izi | Homografi metre cinsinden gerçek saha ölçülerine taşındı (hız/mesafe için), takım ataması otomatik (manuel renk seçimi gerektirmez) ve iz-tabanlı oylama eklendi |
| github.com/Dishant27/Football_Analysis | Bu repo görüntü işleme değil, FIFA veri seti üzerinde **oyuncu reytingi (Overall) ve kadro seçimi** yapar | Karakter kartındaki **GENEL puanın**, profile göre ağırlıklı özelliklerden hesaplanması fikri buradan alındı |
| CIES Football Observatory — *Technical profiling of football players* | Aksiyon türlerinin (Şut, Kilit Pas, Çalım, Orta, Pas, Top Kapma, Araya Girme, Top Kazanma…) **Hücum↔Savunma** ve **Merkez↔Kanat** eksenlerinde vektörlerle temsil edilmesi | Videodan **otomatik çıkarılan** aksiyon olaylarıyla aynı vektör yöntemi uygulandı → oyuncular 8 profil kategorisine + "Dengeli" sınıfına ayrıldı |
| Football Manager / oyuncu analiz kartı görselleri | Radar grafiği, yüzdelik (percentile) çubukları, maç istatistik tablosu | Kart, aynı kesitteki diğer oyunculara göre yüzdelik puanlarla otomatik üretilir |

---

## 1. Araştırma sorusu ve hipotez (Ortak koşul: *Araştırma sorusu*)

- **Görev:** Yayın (broadcast) ve taktik kamera futbol videolarında nesne tespiti → takım ataması → top hakimiyeti / pas olayı çıkarımı.
- **Veri seti:** SoccerNet-Tracking-2023 (resmî train/test ayrımı, 1920×1080, MOT formatı, takım etiketli).
- **Ana araştırma sorusu (RQ1):** Futbola özgü ince ayar + **yüksek çözünürlüklü girdi (1280 px)** + **top kopyala-yapıştır artırımı**, COCO ön-eğitimli YOLOv8s'e ve standart 640 px ince ayara göre **top sınıfı AP@0.5** değerini ne kadar artırır?
  - Ana metrik: Top AP@0.5 (↑). Yardımcı: oyuncu/hakem AP@0.5, mAP@0.5:0.95, top recall, ms/kare (↓), VRAM (↓).
  - Hipotez: Top küçük bir nesne olduğundan (çoğu < 20 px) çözünürlük artışı ve top örneklerinin çoğaltılması top recall'unu belirgin biçimde artırır; oyuncu sınıfında kazanç sınırlı kalır.
- **Destekleyici soru (RQ2):** Merkez kırpma + çim maskesi + LAB renk uzayı + **iz-tabanlı çoğunluk oylaması**, takım atama doğruluğunu temel K-Means yöntemine göre artırır mı? (Metrik: oyuncu-kutu takım doğruluğu ↑)
- **Aşağı akış sorusu (RQ3):** Tespit/takım hataları **top hakimiyeti yüzdesi** ve **pas sayısı** ölçümüne ne kadar yansır? (Metrik: hakimiyet % MAE ↓, pas olayı F1 ↑)

## 2. Veri hazırlama ve ayrım (Ortak koşul: *Veri ayrımı*)

1. `src/bgoru/data/remote_zip.py` — 18 GB'lık arşivleri tamamen indirmeden, HTTP Range ile yalnızca gerekli kareleri çeker.
2. `src/bgoru/data/prepare_soccernet.py`
   - `--split train --stride 15` → eğitim/doğrulama (her 15. kare, 0,6 sn aralık)
   - `--split test --stride 25` → test (her 25. kare)
   - `--split test --dense --stride 2 --max-seqs 10` → top hakimiyeti testi için ardışık kareler (12,5 fps)
3. **Sızıntı önleme:** Resmî train/test korunur. Doğrulama kümesi eğitimden **gameID (maç) bazında** ayrılır; aynı maçın klipleri iki kümeye düşmez. Eşikler (güven eşiği vb.) yalnızca doğrulama kümesinde seçilir.
4. Sınıflar: `0=player` (oyuncu + kaleci), `1=referee`, `2=ball`. Takım etiketi (sol/sağ) ham `gameinfo.ini` dosyasından korunur.
5. Çıktılar: `data/soccernet/manifests/*_manifest.csv` ve `*_split.json` (ayrım manifesti).

## 3. Yöntemler (Ortak koşul: *Yöntem karşılaştırması*, *Özgün çalışma*)

| Kimlik | Yapılandırma | Açıklama |
|---|---|---|
| **E1 – Temel** | YOLOv8s, COCO ağırlıkları, ince ayar yok | `person→player`, `sports ball→ball`; hakem sınıfı yok |
| **E2 – Alternatif** | YOLOv8s, SoccerNet ince ayar, 640 px | Standart transfer öğrenme |
| **E3 – Önerilen** | YOLOv8s, ince ayar, **1280 px + top kopyala-yapıştır** | Küçük nesne odaklı iyileştirme |
| A1 – Ablation | E3 − kopyala-yapıştır (1280 px) | Artırımın katkısı |
| A2 – Ablation | E3 − yüksek çözünürlük (640 px + kopyala-yapıştır) | Çözünürlüğün katkısı |
| R – Referans | Hmzbo YOLOv8L (taktik kamera verisiyle eğitilmiş) | Alan değişimi referansı |
| **G1 – Son sistem** | A1 ayarları + **SoccerNet-GSR ek eğitim verisi** (3 maç, 2900 kare) | Ek verinin katkısı |
| C1 – Kontrol | A1 verisi, G1 ile eşit gradyan adımı (50 epoch) | Kazancı eğitim süresinden ayırır |

- Tüm ince ayarlarda aynı veri, aynı tohum (0; E3 ve A1 ayrıca tohum 1), aynı epoch (20), aynı nominal batch (64, gradyan biriktirme ile), aynı optimizer (SGD, lr0 0,01).
- Veri ayrımı (gerçekleşen): eğitim = 2 maç / 39 sekans / 1950 kare, doğrulama = 1 maç / 18 sekans / 900 kare, test = 3 maç / 49 sekans / 1470 kare (+ 10 sekanslık 3750 karelik yoğun test). Maçlar kesişmez.
- **Sonuç (ölçüldü):** kazancın tamamı 1280 px'den gelir; kopyala-yapıştır top AP@0.5'e katkı sağlamamıştır.
  Tohum 0'da E3−A1 = −0,043 [−0,066; −0,022] (anlamlı düşüş); ikinci tohumla tekrarda −0,003 [−0,028; +0,021]
  (anlamsız), iki tohum ortalamasında −0,023 [−0,043; −0,005]. Yani düşüşün büyüklüğü eğitim varyansına duyarlıdır
  (E3 std 0,029, A1 std 0,001). Sonraki adımda doğrulamada iki tohumda da E3'ten iyi olan **A1** ayarları kullanılır.
- **Tohum tekrarı:** `bash experiments/train_seeds.sh && python -m bgoru.experiments.eval_seeds` → A1 ve E3 tohum 1 ile
  yeniden eğitildi (her biri ~50 dk, RTX 4050); sonuçlar `results/detection/seed_repeats.{csv,json}`, rapor Ek A.3.
- **Ek eğitim verisi (G1):** SoccerNet-GSR-2025 `valid` (maç 2, 3, 5; aynı yayın kliplerinden, kümelerimizle kesişmez — `data/soccernet_gsr/KAYNAK.md`). Test top AP@0.5: A1 0,353 → **G1 0,391** (G1−A1 = +0,037 [+0,012; +0,064]). Eşit adımlı kontrol C1 0,347: G1−C1 = +0,044 [+0,021; +0,067], C1−A1 = −0,006 [−0,025; +0,013] → kazanç ek veriden gelir, uzun eğitimden değil. Hakem AP@0.5 0,70 → 0,93. G1 doğrulamada da en iyidir (0,359 > 0,313/0,332) ve **son sistem dedektörüdür** (`src/bgoru/config.py: FINAL_DETECTOR`; eşikler top 0,20, oyuncu 0,30). Komutlar: `python -m bgoru.data.prepare_gsr`, `bash experiments/train_gsr.sh`, `bash experiments/train_control.sh`; sonuçlar `results/detection/gsr_extra.{csv,json}`, rapor Ek A.4.
- **Özgün bileşen:** `src/bgoru/data/copy_paste.py` — eğitim kümesindeki top kesitleri, HSV çim maskesi üzerinde rastgele konumlara yumuşak kenarla yapıştırılır. Görüntü sayısı değişmez (eğitim bütçesi sabit).

## 4. Uygulama hattı (pipeline)

`src/bgoru/` modülleri:

| Modül | Görev |
|---|---|
| `detection.py` | YOLO çıkarımı, sınıf eşleme (COCO / Hmzbo / bizim model) |
| `tracking.py` | ByteTrack ile oyuncu kimliği takibi |
| `team.py` | Takım ataması: K-Means (temel), forma+çim maskesi (alternatif), **LAB palet + iz oylaması (önerilen)** |
| `homography.py` | Saha anahtar noktaları → metre cinsinden taktik harita; güncelleme koşulu + RANSAC |
| `camera.py` | Anahtar nokta yoksa optik akışla kamera hareketi telafisi |
| `ball.py` | Tek top seçimi, kısa boşluk enterpolasyonu, top izi |
| `possession.py` | **Anlık top hakimiyeti %** (kümülatif + son 60 sn), hakimiyet histerezisi |
| `events.py` | Pas, başarısız pas/araya girme, top kapma, top kazanma, top taşıma/çalım, şut, orta, kilit pas |
| `kinematics.py` | Koşu mesafesi, ortalama/maksimum hız |
| `profiling.py` | CIES tarzı aksiyon vektörleri → profil kategorisi |
| `card.py` | **Oyuncu karakter kartı** (radar, yüzdelik çubuklar, maç tablosu, profil pusulası) |
| `annotate.py` | Video üstü HUD: hakimiyet çubuğu, pas/dk, son olaylar, taktik harita |
| `pipeline.py` | Hepsini kare kare çalıştıran ana döngü; `detect_team_colors()` ile takım rengi algılama |

Arayüz: `app/streamlit_app.py` (video yükleme → takım renklerini algıla ve adlandır → canlı analiz → takım istatistikleri → profil haritası → karakter kartları, ZIP indirme).
Komut satırı: `python -m bgoru.cli analyze --video <yol>`.

## 5. Deneyler (Ortak koşul: *Kontrollü deneyler*, *Ablation*)

1. `python -m bgoru.experiments.train_detector --config configs/exp/<ID>.yaml` — E2, E3, A1, A2, G1, C1 eğitimi
2. `python -m bgoru.experiments.eval_detector` — tüm yapılandırmalar aynı test protokolünde (+ dizi bazlı bootstrap güven aralığı)
3. `python -m bgoru.experiments.eval_team` — RQ2 (temel / alternatif / önerilen + 2 ablation)
4. `python -m bgoru.experiments.eval_possession` — RQ3 (hakimiyet ve pas, enterpolasyon ablation'ı)
5. Tüm sonuçlar `experiments/registry.csv` (kimlik, model, yapılandırma, ayrım, metrikler, süre) ve `results/` altında.

## 6. Hata ve dayanıklılık analizi (Ortak koşul: *Hata ve dayanıklılık analizi*)

1. `python -m bgoru.experiments.robustness` — sentetik **gece** (gama + gürültü), **hareket bulanıklığı**, **JPEG sıkıştırma**; top boyutu kovaları (küçük nesne).
2. `python -m bgoru.experiments.failure_cases` — en az 20 başarısız örnek (kaçırılan top, yanlış pozitif, yanlış takım) kırpılıp numaralandırılır; nedenleri `results/failures/failures.csv` içinde.
3. Alan değişimi: taktik kamera demo videolarında nitel test.

## 7. Başarı ve maliyet (Ortak koşul: *Başarı ve maliyet*)

`python -m bgoru.experiments.benchmark` — parametre sayısı, dosya boyutu (MB), ms/kare (batch=1, 20 ısınma + 200 tekrar), en yüksek GPU belleği; cihaz: RTX 4050 Laptop 6 GB.

## 8. Tekrarlanabilirlik (Ortak koşul: *Teslim gereklilikleri*)

- `environment/requirements.txt`, `environment/environment.yml`, `environment/env_info.json` (otomatik)
- Tohum: 0. Sabit kişisel dosya yolu yok; tüm yollar proje köküne göre.
- `README.md` içinde veri edinme → kurulum → eğitim → değerlendirme → çıkarım komutları.

## 9. Rapor ve gösterim

- `report/rapor.docx` + `report/rapor.pdf` (şablon yapısında, 8–12 sayfa ana metin)
- `demo/` — açıklamalı çıktı videoları (başarılı + başarısız örnek) ve `DEMO_SENARYOSU.md` (3–5 dk sunum akışı)

## 10. Gerçekleşme durumu (son kontrol)

| Ortak koşul | Karşılığı | Durum |
|---|---|---|
| Araştırma sorusu + ana metrik | RQ1 top AP@0.5; RQ2 takım doğruluğu; RQ3 hakimiyet % hatası ve pas F1 | ✔ |
| ≥3 yapılandırma, aynı test protokolü | E1 / E2 / E3 (+ A1, A2, R, G1, C1) — `eval_detector` | ✔ |
| Özgün iyileştirme | Top kopyala-yapıştır (ablation: **olumsuz**, iki tohumda da A1 ≥ E3; anlamlı düşüş yalnız tohum 0'da) + 1280 px (asıl kazanç); yerel saha maskeli takım ataması | ✔ |
| Veri ayrımı, sızıntı önleme | Maç bazlı train/val/test; eşikler doğrulamada | ✔ (`results/data/`) |
| ≥2 ablation | A1, A2 (dedektör); T5a–T5e (takım); V3–V5 (aşağı akış) | ✔ |
| ≥20 başarısız örnek + zorlu koşul | 30 örnek (`results/failures/`), gece/bulanık/JPEG/uzak | ✔ |
| Model boyutu, süre, GPU belleği | `benchmark` (RTX 4050, batch 1, FP16) | ✔ |
| Kaynak ve yapay zekâ beyanı | Rapor "Dış kod ve yapay zekâ araçları" | ✔ (ekip katkıları öğrencide) |
| Çalıştırılabilir kod, README, ortam | `README.md`, `environment/` | ✔ |
| Deney kayıtları | `experiments/registry.csv/json`, `experiments/runs/*/results.csv` | ✔ |
| Ek eğitim verisi (ek) | G1 = A1 + SoccerNet-GSR (3 maç), son sistem; eşit adımlı kontrol C1 ile veri etkisi ayrıştırıldı — `results/detection/gsr_extra.*`, rapor Ek A.4 | ✔ |
| Eğitim varyansı (ek) | E3 ve A1 tohum 1 ile yeniden eğitildi; ort. ± std ve eşleştirilmiş farklar `results/detection/seed_repeats.*`, rapor Ek A.3 | ✔ |
| 8–12 sayfa rapor (PDF + DOCX) | `report/rapor.docx`, `report/rapor.pdf` | ✔ |
| 3–5 dk gösterim | `demo/gosterim_videosu.mp4` (3,3 dk) + zamanlanmış `ANLATIM_METNI.md` + sentetik sesli kopya `gosterim_videosu_anlatimli.mp4`; `DEMO_SENARYOSU.md` (kendi sesli kaydı öğrenci alır) | ✔ |
| Teslim paketi | `python -m bgoru.package` → `teslim/BilgisayarliGoru_teslim.zip` (ham veri hariç; rapor yer tutucuları kaldıysa uyarır) | ✔ |
| Kod temiz kopyada çalışıyor | Teslim ZIP'i boş bir klasöre açılıp README'deki gibi birim testleri (şu an 14) ve örnek çıkarım (`bgoru.cli analyze`) çalıştırıldı | ✔ |

**Öğrencinin tamamlaması gerekenler:** rapor kapağındaki ad/numara/öğretim elemanı/tarih alanları,
"Ekip katkıları" bölümü ve yapay zekâ kullanımının doğrulama notu — hepsi `report/ogrenci_bilgileri.yaml`
dosyasına yazılır, ardından `python report/build_report.py` ve `report/to_pdf.ps1` ile rapor yeniden üretilir —
ve 3–5 dakikalık gösterim kaydı.
