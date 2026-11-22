# Top sahipliği ve olaylar

## Sahiplik

Topa en yakın oyuncu hemen sahip sayılmaz; kısa bir süre aday kalması gerekir. Böylece topun
yanından geçen oyuncu sahipliği değiştirmez.

```mermaid
stateDiagram-v2
    [*] --> SahipYok
    SahipYok: Sahip yok
    Onay: Aday onayı
    Sahip: Sahip = oyuncu A

    SahipYok --> Onay: topa yakın oyuncu B
    Sahip --> Onay: başka oyuncu B topa yaklaştı
    Onay --> Sahip: B kısa süre aday kaldı → yeni sahip B
    Onay --> Sahip: aday kayboldu → A devam eder
    Onay --> SahipYok: aday kayboldu
```

## Top el değiştirince

```mermaid
flowchart TD
    N["Top A'dan B'ye geçti"] --> ST{"Aynı takım mı?"}
    ST -- evet --> PASS["Pas"]
    ST -- hayır --> Q{"Nasıl geçti?"}
    Q -- "yakın temas" --> TK["Top kapma"]
    Q -- "top uzun süre sahipsiz" --> PK["Top kazanma"]
    Q -- "diğer" --> INT["Araya girme"]
```
