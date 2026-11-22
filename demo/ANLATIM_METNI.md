# Gösterim videosu — anlatım metni

`demo/gosterim_videosu.mp4` (3.3 dk) üzerine kendi sesinizle anlatım kaydı için zamanlanmış metin (~323 kelime). Sayılar `results/` çıktılarından üretilmiştir; metni değiştirirseniz `src/bgoru/experiments/make_narration.py` içindeki `BLOCKS` listesini düzenleyip betiği yeniden çalıştırın.

Windows'ta kayıt: videoyu tam ekran oynatıp `Win + Alt + R` (Xbox Game Bar) ile mikrofonlu kayıt alın ya da Clipchamp'te videonun üzerine ses kaydı ekleyin.

| Süre | Ekranda | Okunacak metin |
|---|---|---|
| 0:00–0:07 | Başlık | Bilgisayarlı Görü: futbol videolarından anlık top hakimiyeti, pas sıklığı ve oyuncu kartları. |
| 0:07–0:19 | Girdi, Ham girdi videosu | Girdi, tek kameralı bir maç videosu. Arayüzde demo seçiliyor ya da video yükleniyor; ilk karelerden forma renkleri öğrenilip takımlarla eşleniyor. |
| 0:19–0:39 | Model çıkarımı, Taktik demo 1 (canlı panel) | Her karede YOLOv8s dedektörü oyuncuları, hakemi ve topu buluyor; ByteTrack oyunculara kimlik veriyor, forma rengiyle takım atanıyor ve saha noktalarından homografi hesaplanıyor. Sol üstteki panelde top hakimiyeti, pas sayısı, dakika başına pas ve olay akışı kare kare güncelleniyor. |
| 0:39–1:26 | Taktik demo 2 (taktik harita) | İkinci örnekte sağ alttaki taktik haritada oyuncuların ve topun metre cinsinden konumu görülüyor; kırmızı üçgen o anki top sahibini gösteriyor. Top sahibi, topun oyuncunun ayak bölgesine uzaklığıyla ve kısa bir gecikme süresiyle belirleniyor. Sahiplik aynı takımda el değiştirince pas, rakibe geçince araya girme ya da top kaybı sayılıyor; şut, orta ve top taşıma da topun hızına ve konumuna göre ayrılıyor. Tüm hat, RTX 4050 ekran kartında saniyede yaklaşık 26 kare işliyor. |
| 1:26–1:38 | Sonuçların yorumu, Profil haritası | Analiz bitince her oyuncunun eylem sıklıkları CIES yöntemindeki gibi bir vektöre dönüştürülüyor ve oyuncular sekiz teknik profilden birine ya da dengeli sınıfına atanıyor. |
| 1:38–1:56 | Karakter kartları | Her oyuncu için bir karakter kartı üretiliyor. Hız, hareket, pas, top sürme, savunma, hücum ve top kontrolü puanları aynı kesitteki oyunculara göre yüzdelik olarak hesaplanıyor; genel puan ise oyuncunun profiline göre ağırlıklandırılıyor. |
| 1:56–2:13 | Tespit sonuçları, Model karşılaştırması | Test kümesinde top tespiti, ince ayarsız modelde 0,202, ek veriyle eğitilen son modelde 0,391 AP değerine ulaştı. Top kopyala-yapıştır artırımı ise 0,309 ile geride kaldı; bu bir olumsuz sonuç. |
| 2:13–2:29 | Hakimiyet ve pas sonuçları, Zorlu koşullar | Son sistem pasları 0,660 F1 ile buluyor; ancak kısa kliplerde hakimiyet yüzdesi hatası yüksek kalıyor. Son sistemde uzak kamera ve bulanıklık koşullarında top tespiti belirgin biçimde düşüyor; gece ve JPEG sıkıştırması ise az etkiliyor. |
| 2:29–3:05 | Başarısız örnek, Başarısız örnek videosu | Başarısız örnekte yayın kamerası ve kalabalık bir ceza sahası var. Saha noktaları bulunamadığı için taktik harita çizilemiyor ve hızlar yalnızca yaklaşık hesaplanabiliyor. Top oyuncu kalabalığında kaybolduğunda sahiplik yanlış oyuncuya geçiyor; bu da hatalı pas ve top kapma olaylarına yol açıyor. |
| 3:05–3:13 | Hata örnekleri | İncelediğimiz 30 başarısız örnekte başlıca nedenler etiket kayması, hareket bulanıklığı, elde tutulan top ve örtülme. |
| 3:13–3:20 | Özet | Tüm sayılar raporda ve deney kayıtlarında yer alıyor. Teşekkürler. |
