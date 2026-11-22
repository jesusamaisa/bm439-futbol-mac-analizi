# Model ağırlıkları

| Dosya | Açıklama | Nasıl elde edilir |
|---|---|---|
| `bgoru_G1_ft1280_gsr.pt` | **Son sistem** dedektörü (YOLOv8s, 1280 px, A1 ayarları + SoccerNet-GSR ek eğitim verisi) — doğrulama top AP'sine göre seçildi; arayüz/CLI varsayılanı | `bash experiments/train_gsr.sh` (veya teslim paketinden) |
| `bgoru_A1_ft1280.pt` | Ablation (1280 px, artırımsız; yalnızca SoccerNet-Tracking) — G1'in ayarları | `bash experiments/train_all.sh` |
| `bgoru_C1_ft1280_ep50.pt` | Kontrol (A1 verisi, G1 ile eşit adım: 50 epoch) | `bash experiments/train_control.sh` |
| `bgoru_E3_ft1280_cp.pt` | Önerilen iyileştirme (1280 px + top kopyala-yapıştır) — ablation'da A1'in gerisinde kaldı | `bash experiments/train_all.sh` |
| `bgoru_E2_ft640.pt` | Alternatif (YOLOv8s, 640 px) | aynı |
| `bgoru_A2_ft640_cp.pt` | Ablation modeli (640 px + kopyala-yapıştır) | aynı |
| `yolov8s.pt` | COCO ön-eğitimli YOLOv8s (Ultralytics, AGPL-3.0) | ilk kullanımda otomatik iner |
| `hmzbo_keypoints_yolov8m.pt` | Saha anahtar noktası modeli (28 nokta) — referans proje | aşağıdaki komut |
| `hmzbo_players_yolov8l.pt` | Referans projenin taktik-kamera oyuncu modeli (R) | aşağıdaki komut |

Referans proje ağırlıkları (Hmzbo, *Football-Analytics-with-Deep-Learning-and-Computer-Vision*):

```bash
B=https://github.com/Hmzbo/Football-Analytics-with-Deep-Learning-and-Computer-Vision/raw/master
curl -L -o weights/hmzbo_players_yolov8l.pt   "$B/models/Yolo8L%20Players/weights/best.pt"
curl -L -o weights/hmzbo_keypoints_yolov8m.pt "$B/models/Yolo8M%20Field%20Keypoints/weights/best.pt"
```

Not: İndirilen `.pt` dosyalarının pickle içeriği tarandı; yalnızca `torch`, `ultralytics`,
`collections` ve `set` nesneleri içerdiği doğrulandı.
