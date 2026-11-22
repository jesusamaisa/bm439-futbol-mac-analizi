# Bilgisayarlı Görü — Futbol Maç Analizi, Oyuncu Profilleme ve Karakter Kartı

Futbol videolarından **oyuncu / hakem / top tespiti**, **takım ataması**, **taktik harita**,
**anlık top hakimiyeti yüzdesi ve pas sıklığı** ölçümü, oyuncuların **teknik profil kategorilerine**
ayrılması ve her oyuncu için **karakter kartı** üretimi yapan uçtan uca sistem.

BM439 Bilgisayarlı Görme dersi projesidir. Tüm adımlar için [ADIMLAR.md](ADIMLAR.md), rapor için
[report/rapor.pdf](report/rapor.pdf) dosyasına bakın.

<!-- SONUCLAR -->
## Ölçülen sonuçlar (test kümesi)

| Yapılandırma | Top AP@0.5 [%95 GA] | Top AP_m* | Oyuncu AP@0.5 | Hakem AP@0.5 | mAP@[.5:.95] | ms/kare |
|---|---|---|---|---|---|---|
| E1 Temel — COCO YOLOv8s | 0.202 [0.160, 0.247] | 0.457 | 0.863 | — | 0.282 | 14.2 |
| E2 Alternatif — ince ayar 640 | 0.158 [0.127, 0.195] | 0.437 | 0.925 | 0.686 | 0.331 | 7.9 |
| A2 Ablation — 640 + kopyala-yapıştır | 0.149 [0.117, 0.188] | 0.467 | 0.914 | 0.693 | 0.328 | 7.7 |
| E3 Önerilen iyileştirme — 1280 + kopyala-yapıştır | 0.309 [0.263, 0.359] | 0.691 | 0.943 | 0.683 | 0.370 | 13.2 |
| A1 Ablation — 1280, artırımsız | 0.353 [0.303, 0.408] | 0.689 | 0.939 | 0.699 | 0.386 | 13.1 |
| C1 Kontrol — A1 verisi, G1 ile eşit adım (50 epoch) | 0.347 [0.299, 0.401] | 0.680 | 0.943 | 0.688 | 0.387 | — |
| **G1 — A1 + SoccerNet-GSR ek eğitim verisi (son sistem; doğrulamada en iyi)** | 0.391 [0.343, 0.443] | 0.723 | 0.966 | 0.929 | 0.455 | 13.3 |
| R Referans — Hmzbo YOLOv8L | 0.007 [0.003, 0.014] | 0.035 | 0.564 | 0.076 | 0.091 | — |

*AP_m: merkez uzaklığı ≤ 2 top çapı ile eşleştirme (etiket ötelemesine dayanıklı yardımcı metrik).

Tohum tekrarı (eğitim varyansı, 2 tohum, ort. ± std): top AP@0.5 E3 **0.330 ± 0.029**, A1 **0.353 ± 0.001**; tohum 1'de E3−A1 = -0.003 [-0.028, +0.021], tohum ortalamasında -0.023 [-0.043, -0.005] (`results/detection/seed_repeats.json`).

Ek eğitim verisi (SoccerNet-GSR valid, 3 yeni maç, 2900 kare): G1−A1 = +0.037 [+0.012, +0.064]; aynı adım sayısıyla eğitilen kontrole göre G1−C1 = +0.044 [+0.021, +0.067], C1−A1 = -0.006 [-0.025, +0.013]. Kazanç daha uzun eğitimden değil ek veriden gelir (`results/detection/gsr_extra.json`).

| Takım ataması | Doğrulama | Yoğun test | Seyrek test |
|---|---|---|---|
| Temel (ort. RGB + K-Means) | 0.871 | 0.897 | 0.855 |
| Önerilen v1 (sabit HSV maskesi) | 0.622 | 0.950 | 0.995 |
| **Önerilen (yerel saha maskesi)** | 0.998 | 0.947 | 0.995 |

Kopyala-yapıştır artırımı ablation'da top AP@0.5'e katkı sağlamadığı için (tohum 0'da E3−A1 anlamlı biçimde < 0, tohum 1'de fark anlamsız) sonraki adımda doğrulamada iki tohumda da daha iyi olan A1 ayarları kullanılır. Bu ayarlarla ek veriyle eğitilen G1 doğrulamada en iyi olduğundan son sistemde G1 kullanılır. Son sistem, 10 yoğun test sekansında vekil referansa göre pas olaylarını **F1 = 0.660**, kare bazlı top kontrolünü **0.753** uyumla ölçer; klip başına hakimiyet yüzdesi hatası 18.5 yüzde puanıdır (30 sn'lik kliplerde yüksek sapmalı). Ayrıntılar, ablation'lar ve hata analizi: `report/rapor.pdf`, tüm kayıtlar: `experiments/registry.csv`.

Tam hat hızı (taktik kamera, NVIDIA GeForce RTX 4050 Laptop GPU): **26.3 FPS**.

<!-- /SONUCLAR -->

## Klasör düzeni

```
README.md  ADIMLAR.md
src/bgoru/            kaynak kod (paket)
  data/               SoccerNet seçici indirme + YOLO dönüşümü, top kopyala-yapıştır artırımı
  experiments/        eğitim, değerlendirme, dayanıklılık, verimlilik, hata analizi, şekiller
  detection.py tracking.py team.py homography.py ball.py camera.py kinematics.py
  possession.py profiling.py card.py annotate.py pipeline.py cli.py
app/streamlit_app.py  web arayüzü
tests/                GPU gerektirmeyen birim testleri
configs/              veri ve deney yapılandırmaları, profil vektörleri
environment/          requirements.txt, environment.yml, env_info.json
experiments/          registry.csv/json (tüm deney kayıtları), runs/ (eğitim eğrileri), logs/
results/              tablolar (CSV/JSON), şekiller, hata örnekleri, örnek kartlar
report/               rapor.docx + rapor.pdf, rapor üretim betiği
demo/                 girdi videoları, çıktı videoları, gösterim senaryosu
weights/              model ağırlıkları + indirme yönergesi (weights/README.md)
```

## Kurulum

Gereksinim: Python 3.12, NVIDIA GPU (CUDA 11.8 derlemesi ile test edildi). GPU'suz bilgisayarda CPU kurulumu için aşağıdaki "Başka bir bilgisayarda çalıştırma" bölümüne bakın (~2 kare/sn).

```bash
pip install -r environment/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

Tüm komutlar **proje kökünden** ve `PYTHONPATH=src` ile çalıştırılır (Windows PowerShell: `$env:PYTHONPATH="src"`).

### Başka bir bilgisayarda çalıştırma (teslim paketiyle)

`teslim/BilgisayarliGoru_teslim.zip` kod, son sistem ağırlıkları, demo videoları, sonuçlar ve raporu içerir (ham veri
hariç); test, arayüz ve örnek çıkarım için ek indirme gerekmez.

1. ZIP'i boşluk ve Türkçe karakter içermeyen bir klasöre açın (ör. `C:\proje`). **Python 3.12** kurulu olmalıdır
   (3.13 için PyTorch 2.4.1 paketi yoktur).
2. Sanal ortam ve paketler (PowerShell, proje kökünde):

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   # NVIDIA GPU varsa:
   pip install -r environment/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
   # GPU yoksa (CPU):
   pip install -r environment/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
   $env:PYTHONPATH="src"; $env:PYTHONUTF8="1"
   ```

3. Doğrulama: `python -m pytest -q tests` (14 test), `python -c "import torch; print(torch.cuda.is_available())"`
   (GPU görünüyor mu), ardından `streamlit run app/streamlit_app.py`. (`bgoru.cli env` komutunu çalıştırmayın: rapordaki
   deney ortamı bilgisini, `environment/env_info.json`, bu bilgisayarınkiyle değiştirir.)

### GitHub deposundan kurulum

Depoda kod, yapılandırmalar, deney kayıtları, sonuçlar ve rapor bulunur; model ağırlıkları ve videolar (`*.pt`,
`*.mp4`) boyut nedeniyle depoya değil, deponun **Releases → v1.0** ekine konmuştur. Klonladıktan sonra:

| Release dosyası | Konulacağı yer | Gerekli mi |
|---|---|---|
| `bgoru_G1_ft1280_gsr.pt` (son sistem dedektörü) | `weights/` | Evet |
| `hmzbo_keypoints_yolov8m.pt` (saha anahtar noktaları) | `weights/` | Evet |
| `taktik_demo1.mp4`, `taktik_demo2.mp4`, `soccernet_SNMOT-118.mp4` | `demo/input/` | Arayüzdeki demolar için |
| `gosterim_videosu*.mp4`, `basarili_*.mp4`, `basarisiz_ornek.mp4` | `demo/` | Hayır (gösterim çıktıları) |

Ardından yukarıdaki 2. ve 3. adımlar (sanal ortam, paketler, testler) aynen uygulanır.

**GPU'suz bilgisayar:** Yarı hassasiyet (FP16) yalnızca CUDA varken açılır (`bgoru.detection.use_half`); CPU'da
FP16 emüle edildiği için 1280 px tespit kare başına 26,9 sn sürerken FP32 ile 0,32 sn sürer. Ölçülen tam hat hızı
(i5-13500H, GPU gizli): **~2 kare/sn** (tespit 277 ms + saha anahtar noktaları 196 ms); RTX 4050'de ~26 kare/sn.
CPU'da arayüzde "Anahtar nokta tespiti her N karede" değerini 5–10 yapmak veya kısa bir kesit seçmek önerilir.
`benchmark` ölçümü yalnızca CUDA'da çalışır; PDF dönüştürme Microsoft Word, sesli anlatım Windows gerektirir
(ikisi de test için gerekmez).

## Veri edinme

SoccerNet-Tracking-2023 (Hugging Face `SoccerNet/SN-Tracking-2023`, herkese açık train/test). Arşivlerin
tamamı ~18 GB'tır; betik HTTP Range istekleriyle yalnızca gereken kareleri indirir (~2 GB):

```bash
python -m bgoru.data.prepare_soccernet --split train --stride 15        # 2850 kare -> train (2 maç) / val (1 maç)
python -m bgoru.data.prepare_soccernet --split test  --stride 25        # 1470 kare, 49 sekans, 3 maç
python -m bgoru.data.prepare_soccernet --split test  --dense --stride 2 --max-seqs 10   # 3750 ardışık kare
python -m bgoru.data.copy_paste                                         # top kopyala-yapıştır eğitim kümesi
```

Ayrım manifestleri: `data/soccernet/manifests/*_manifest.csv`, `*_split.json` (kopyaları `results/data/`).
Doğrulama kümesi eğitimden **maç (gameID) bazında** ayrılır; test resmî ayrımdır ve eğitim/eşik seçiminde kullanılmaz.

**Ek eğitim verisi (son sistem G1 için): SoccerNet-GSR-2025 `valid`.** GSR, SoccerNet-Tracking kliplerinin üzerine kuruludur; GSR train/test bizim train+val/test kliplerimizle aynıdır, yalnızca `valid` ayrımı 3 yeni maçtan (oyun 2, 3, 5) 58 klip getirir. Ayrıntılar ve sızıntı denetimi: `data/soccernet_gsr/KAYNAK.md`.

```bash
curl -L -C - -o data/soccernet_gsr/valid.zip https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/valid.zip   # 11,2 GB
python -m bgoru.data.prepare_gsr --stride 15     # YOLO'ya dönüştürür; test/val maçlarıyla kesişirse durur
```

Bu veriyle eğitim için yapılandırmada `data: soccernet_gsr` kullanılır (`configs/data/soccernet_gsr.yaml`: train = SoccerNet-Tracking train + GSR valid; doğrulama ve test değişmez).

Referans projenin saha anahtar noktası modeli ve taktik kamera demo videoları için `weights/README.md`.

## Eğitim, değerlendirme, analiz

```bash
bash experiments/train_all.sh                        # E3, E2, A1, A2 (her biri 20 epoch, tohum 0)
bash experiments/train_gsr.sh                        # G1 = A1 ayarları + GSR ek verisi (son sistem)
bash experiments/train_control.sh                    # C1 = A1 verisi, G1 ile eşit adım (50 epoch) + eval_gsr
python -m bgoru.experiments.eval_detector            # RQ1: aynı test protokolü + bootstrap GA
python -m bgoru.experiments.eval_team                # RQ2: takım ataması + ablation
python -m bgoru.experiments.eval_possession          # RQ3: hakimiyet % ve pas doğruluğu + ablation
python -m bgoru.experiments.robustness               # gece / bulanıklık / JPEG / uzak kamera
python -m bgoru.experiments.benchmark                # ms/kare, FPS, GPU belleği, parametre
python -m bgoru.experiments.failure_cases            # 20+ başarısız örnek
python -m bgoru.experiments.make_figures             # rapor şekilleri
```

## Raporu üretme

Rapordaki tüm sayılar `results/` çıktılarından okunur. Kapak bilgileri (dönem, ekip, öğretim elemanı, tarih,
bağlantı), "Ekip katkıları" tablosu ve yapay zekâ doğrulama notu `report/ogrenci_bilgileri.yaml` dosyasından gelir;
dosyayı doldurup şu iki komutu çalıştırın:

```bash
python report/build_report.py                                   # report/rapor.docx
powershell -ExecutionPolicy Bypass -File report/to_pdf.ps1      # report/rapor.pdf (Microsoft Word gerekir)
python report/update_readme.py                                  # README sonuç bölümü
```

## Gösterim videosu

```bash
python -m bgoru.experiments.make_demo           # demo/ başarılı ve başarısız örnek videoları + kartlar (GPU, ~5 dk)
python -m bgoru.experiments.make_showcase       # demo/gosterim_videosu.mp4 (3,3 dk montaj, sessiz) + zaman çizelgesi
python -m bgoru.experiments.make_narration      # demo/ANLATIM_METNI.md: zamanlanmış anlatım metni
python -m bgoru.experiments.make_narration --tts  # + Windows Türkçe sesiyle anlatımlı kopya ve .srt altyazı
```

## Teslim paketi

```bash
python -m bgoru.package                # teslim/BilgisayarliGoru_teslim.zip (~310 MB; yalnızca son sistem ağırlıkları)
python -m bgoru.package --no-weights   # ağırlıksız (~215 MB); --all-weights: tüm ablation ağırlıkları
```

Ham veri (`data/` görüntüleri, ~2,5 GB) pakete girmez; "Veri edinme" adımlarıyla yeniden üretilir. Veri ayrımını
belgeleyen manifest dosyaları pakettedir.

## Birim testleri (GPU gerekmez)

```bash
python -m pytest -q tests      # 14 test: pas/araya girme kuralları, histerezis, pas/dk, saha ölçüleri, profil, takım ataması, anlatım sayıları, GSR dönüştürme
```

## Çıkarım (küçük örnek)

```bash
python -m bgoru.cli analyze --video demo/input/taktik_demo1.mp4 --out results/ornek --teams "İsviçre" "Fransa" --max-frames 500
```

Varsayılan dedektör `bgoru_G1_ft1280_gsr.pt`'dir (`src/bgoru/config.py: FINAL_DETECTOR`; doğrulama kümesinde top AP'si en
yüksek model). G1, A1 ayarlarıyla (1280 px, artırımsız; kopyala-yapıştır ablation'da katkı sağlamadı) SoccerNet-Tracking
eğitim kümesine GSR'dan 3 maç eklenerek eğitilmiştir. Varsayılan güven eşikleri doğrulamada seçilmiştir (top 0,20, oyuncu 0,30).

Çıktı: `analiz.mp4` (açıklamalı video), `summary.json`, `events.csv`, `possession_timeline.csv`,
`cards/kart_<id>.png`, `profil_haritasi.png`.

Web arayüzü:

```bash
streamlit run app/streamlit_app.py
```

## Sorun giderme (Windows)

* **Eğitimde RAM taşması:** Ultralytics `cache=ram` her veri yükleyici işçisine önbelleğin kopyasını verir;
  bu yüzden yapılandırmalarda `cache: false` kullanılır.
* **Eğitim sürerken arayüz/analiz çökerse (0xC0000005):** eğitim süreci ve işçileri sanal bellek (commit)
  sınırını doldurabilir; yeni PyTorch süreci DLL yükleyemez. Eğitim bitince çalıştırın veya Windows sayfa
  dosyasını büyütün. GPU'yu eğitimle paylaşırken `BGORU_GPU_FRAC=0.2` ile analiz sürecinin bellek payı sınırlanabilir.
* Konsolda Türkçe karakter hatası için `PYTHONUTF8=1` kullanın.

## Kaynaklar

* SoccerNet-Tracking — Cioppa ve ark., *SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos*, CVPRW 2022.
* Ultralytics YOLOv8 (AGPL-3.0); ByteTrack (Zhang ve ark., ECCV 2022) — `supervision` uygulaması.
* Hmzbo, *Football-Analytics-with-Deep-Learning-and-Computer-Vision* (GitHub) — saha anahtar noktası modeli, taktik kamera demo videoları, homografi güncelleme fikri.
* CIES Football Observatory — *Technical profiling of football players* (aksiyon vektörleriyle profil).
* Dishant27, *Football_Analysis* (GitHub) — pozisyona göre ağırlıklı genel oyuncu puanı fikri.

Yapay zekâ aracı kullanımı ve doğrulama: rapor, "Dış kod ve yapay zekâ araçları" bölümü.
