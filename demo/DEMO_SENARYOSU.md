# Uygulama Gösterimi — 3–5 dakikalık senaryo

Gösterim, `streamlit run app/streamlit_app.py` ile açılan web arayüzü üzerinden canlı yapılır ya da
ekran kaydı alınır (Windows: `Win + Alt + R`). Hazır çıktı videoları bu klasördedir:

| Dosya | İçerik |
|---|---|
| `demo/gosterim_videosu.mp4` | **Hazır ~3 dk gösterim montajı** (girdi → çıkarım → yorum → başarısız örnek; sessiz — üzerine anlatım eklenebilir) |
| `demo/ANLATIM_METNI.md` | Montajın zaman çizelgesine göre okunacak anlatım metni (kendi sesinizle kayıt için) |
| `demo/gosterim_videosu_anlatimli.mp4` | Aynı montaj, Windows Türkçe sentetik sesiyle (Microsoft Tolga) anlatımlı; altyazı: `gosterim_videosu_anlatimli.srt` |
| `demo/basarili_taktik_demo1.mp4` | Başarılı örnek — taktik kamera, homografi + taktik harita, canlı hakimiyet/pas paneli |
| `demo/basarili_taktik_demo2.mp4` | Başarılı örnek — 46 sn taktik kamera, 30+ karakter kartı |
| `demo/basarisiz_ornek.mp4` | Başarısız örnek — yayın kamerası, kalabalık ceza sahası: top kaybı ve takım karışması |
| `demo/kartlar/` | Örnek karakter kartları ve profil haritası |

## Akış (≈ 4 dk)

| Süre | Ekranda | Anlatım |
|---|---|---|
| 0:00–0:30 | README'deki akış şeması (`results/figures/fig_architecture.png`) | Problem: videodan **anlık top hakimiyeti, pas sıklığı**, oyuncu profili ve kartı. Girdi = video, çıktı = açıklamalı video + istatistik + kartlar. |
| 0:30–1:00 | Kenar çubuğu: *Taktik kamera demo 1* seçilir, **🎨 Takım renklerini algıla** | **Girdi yükleme:** demo seçilir ya da "Kendi videom" ile MP4 yüklenir. İlk karelerden forma renkleri LAB uzayında kümelenir, takım adları renklerle eşlenir. |
| 1:00–2:15 | **▶ Analizi başlat** | **Model çıkarımı:** YOLOv8s (1280) tespit, ByteTrack kimlik, taktik harita. Sağ panelde **top hakimiyeti % (kümülatif + son 60 sn)**, **pas sayısı / pas/dk / isabet** ve olay akışı (pas, araya girme, top kapma…) kare kare güncellenir. |
| 2:15–3:00 | *Takım istatistikleri* ve *Oyuncu profilleri* sekmeleri | **Sonucun yorumu:** hakimiyetin zaman içindeki eğrisi; olay dağılımı; CIES düzleminde oyuncuların Hücum↔Savunma / Merkez↔Kanat konumları ve kategorileri. |
| 3:00–3:30 | *Karakter kartları* sekmesi, bir oyuncuya isim verilip kart yenilenir | Kart: GENEL puan, radar (yüzdelik), profil pusulası, koşu mesafesi/hız, pas, top kazanma. ZIP olarak indirme. |
| 3:30–4:15 | `demo/basarisiz_ornek.mp4` | **Başarısız örnek:** yayın açısında top oyuncu kalabalığında kayboluyor; sahiplik yanlış oyuncuya geçip hatalı "top kapma" üretiliyor; homografi olmadığından hız/mesafe yaklaşık. Nedenler raporun Hata Analizi bölümünde (20+ örnek). |
| 4:15–4:30 | `results/` tabloları | Ölçülen sonuçlar ve sınırlılıklar; tüm deneylerin kaydı `experiments/registry.csv`. |

## Kontrol listesi
- [ ] `weights/` içinde son sistem dedektörü `bgoru_G1_ft1280_gsr.pt` ve `hmzbo_keypoints_yolov8m.pt` var
- [ ] GPU sürücüsü çalışıyor (`python -m bgoru.cli env`)
- [ ] Demo videoları `demo/input/` altında
