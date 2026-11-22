# SoccerNet-GSR-2025 — ek eğitim verisi (kaynak notu)

| Alan | Değer |
|---|---|
| Veri seti | SoccerNet Game State Reconstruction (GSR) 2025 — `valid` ayrımı |
| Kaynak | Hugging Face `SoccerNet/SN-GSR-2025` → `valid.zip` (herkese açık, erişim kapısı yok) |
| Erişim tarihi | 2026-11-18 |
| Dosya | `valid.zip`, 11 169 497 921 bayt |
| SHA-256 | `d60e517e0bfbce83d46a2805842d73bc98bbb65e84ba76be7b55af9bb3047a50` (Hugging Face LFS kaydı) |
| Lisans etiketi | Hugging Face deposunda `gpl-3.0`; görüntüler SoccerNet yayın kliplerinden gelir, yalnızca araştırma/eğitim amaçlı kullanılır |
| Makale | Somers ve ark., *SoccerNet Game State Reconstruction*, CVPR Workshops (CVsports) 2024 |
| İçerik | 58 klip × 750 kare (30 sn, 25 FPS, 1920×1080); oyuncu/kaleci/hakem/top kutuları, iz kimliği, takım (sol/sağ), forma numarası, saha çizgileri ve kamera parametreleri |

## Neden yalnızca `valid`?

SoccerNet-GSR, SoccerNet-Tracking kliplerinin üzerine kuruludur. Klip kimlikleri (clipStart, actionPosition) ile
eşleştirince:

| GSR ayrımı | Klip | Oyun | Bizim kümelerimizle ilişkisi |
|---|---|---|---|
| train | 57 | 4, 6, 9 | Bizim train (6, 9) + val (4) ile aynı klipler |
| test | 49 | 7, 8, 11 | **Bizim test kümemizle aynı — eğitimde kullanılmaz** |
| valid | 58 | 2, 3, 5 | Yeni maçlar; hiçbir kümemizle kesişmez |
| challenge | 58 | — | Etiketleri gizli |

`python -m bgoru.data.prepare_gsr` dönüştürmeden önce bu kesişmeyi yeniden denetler (`manifests/leak_check.json`)
ve kesişme varsa durur. Dönüştürücü, aynı klibin GSR etiketlerinden SoccerNet-Tracking ile **birebir aynı** YOLO
satırlarını üretir (SNGS-062/170, 6 karede tüm satırlar eşleşti).

## Klasörler

- `valid.zip` — ham arşiv (açılmadan saklanır; dönüştürücü doğrudan okur)
- `images/train`, `labels/train` — YOLO biçiminde seçilen kareler (varsayılan her 15. kare)
- `raw/SNGS-xxx/` — SoccerNet-Tracking biçiminde `gt.txt`, `gameinfo.ini`, `seqinfo.ini` ve `meta.json`
- `manifests/` — kare listesi, ayrım özeti, sızıntı denetimi
