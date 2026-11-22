# Deney akışı

```mermaid
flowchart TD
    SN[("SoccerNet maç videoları")] --> SPL["Maçlara göre ayırma<br/>eğitim · doğrulama · test"]
    GSR[("Ek veri: 3 yeni maç")] --> CHK{"Test maçlarıyla<br/>çakışıyor mu?"}
    CHK -- hayır --> TR
    SPL --> TR["Modelleri eğit<br/>(farklı çözünürlük, artırım, veri)"]
    TR --> EV["Test kümesinde karşılaştır"]
    EV --> SEL{"Doğrulamada en iyi model?"}
    SEL --> FIN(["Son sistem: G1"])
    EV --> RES[("Sonuçlar ve şekiller")]
    RES --> REP["Rapor ve gösterim videosu"]
```

| Model | Fark | Test top AP@0.5 |
|---|---|---|
| E1 | Hazır model, eğitimsiz | 0,202 |
| E2 | 640 px | 0,158 |
| E3 | 1280 px + top kopyala-yapıştır | 0,309 |
| A1 | 1280 px | 0,353 |
| **G1** | **1280 px + 3 maç ek veri (son sistem)** | **0,391** |
