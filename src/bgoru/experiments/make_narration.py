"""Gösterim videosu için zamanlanmış anlatım metni ve (isteğe bağlı) sesli anlatımlı kopya.

* `demo/ANLATIM_METNI.md`  — gosterim_videosu.mp4 zaman çizelgesine göre okunacak metin (kendi sesinizle kayıt için)
* `--tts` ile ayrıca: Windows'un Türkçe sesiyle (Microsoft Tolga, tr-TR) seslendirilmiş
  `demo/gosterim_videosu_anlatimli.mp4` ve altyazı dosyası `demo/gosterim_videosu_anlatimli.srt`

Metindeki tüm sayılar results/ çıktılarından okunur (elle sayı yazılmaz). Zaman çizelgesi
`python -m bgoru.experiments.make_showcase` tarafından results/demo_runs/gosterim_zaman_cizelgesi.json'a yazılır.

Kullanım: python -m bgoru.experiments.make_narration [--tts]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from bgoru.config import FINAL_DETECTOR, RESULTS, ROOT

DEMO = ROOT / "demo"
TIMELINE = RESULTS / "demo_runs" / "gosterim_zaman_cizelgesi.json"
LEAD_S, GAP_S, MAX_RATE = 0.4, 0.3, 1.25  # blok başı gecikme, sonraki bloğa pay, en yüksek konuşma hızı

SEG_TR = {"baslik": "Başlık", "girdi_kart": "Girdi", "girdi_video": "Ham girdi videosu",
          "cikarim_kart": "Model çıkarımı", "demo1": "Taktik demo 1 (canlı panel)", "demo2": "Taktik demo 2 (taktik harita)",
          "yorum_kart": "Sonuçların yorumu", "profil_haritasi": "Profil haritası", "kart": "Karakter kartları",
          "fig_tespit": "Tespit sonuçları", "fig_karsilastirma": "Model karşılaştırması",
          "fig_hakimiyet": "Hakimiyet ve pas sonuçları", "fig_dayaniklilik": "Zorlu koşullar",
          "basarisiz_kart": "Başarısız örnek", "basarisiz_video": "Başarısız örnek videosu",
          "hata_ornekleri": "Hata örnekleri", "ozet": "Özet"}

# Seslendirmede okunuşu belirsiz kısaltmalar (yalnızca TTS metninde değiştirilir)
SPOKEN = {"YOLOv8s": "Yolo ve sekiz", "ByteTrack": "Bayt Trek", "CIES": "Siyes", "RTX 4050": "ar ti eks kırk elli",
          " F1 ": " ef bir ", " AP ": " a pe ", "JPEG": "jeypeg"}

BIR = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
ON = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def tamsayi_oku(n: int) -> str:
    """0–999 999 arası tam sayının Türkçe okunuşu."""
    if n == 0:
        return "sıfır"
    out = []
    if n >= 1000:
        b = n // 1000
        out.append("bin" if b == 1 else f"{tamsayi_oku(b)} bin")
        n %= 1000
    if n >= 100:
        out.append("yüz" if n // 100 == 1 else f"{BIR[n // 100]} yüz")
        n %= 100
    out += [w for w in (ON[n // 10], BIR[n % 10]) if w]
    return " ".join(out)


def sayi(x: float, nd: int) -> tuple[str, str]:
    """(gösterim, okunuş): 0.353 -> ('0,353', 'sıfır virgül üç yüz elli üç')."""
    s = f"{x:.{nd}f}"
    ip, _, fp = s.partition(".")
    disp = s.replace(".", ",")
    if not fp:
        return disp, tamsayi_oku(int(ip))
    lead = len(fp) - len(fp.lstrip("0"))
    frac = " ".join(["sıfır"] * lead + ([tamsayi_oku(int(fp))] if fp.strip("0") else []))
    return disp, f"{tamsayi_oku(int(ip))} virgül {frac}"


def numbers() -> dict[str, tuple[str, str]]:
    sys.path.insert(0, str(ROOT / "report"))
    from build_report import load  # noqa: E402  (rapor ile aynı kaynaklar)

    d = load()
    det, poss = d["det"]["ball_AP50"], d["poss"]
    v = poss.loc["V7_G1_final"] if "V7_G1_final" in poss.index else poss.loc["V2_E3_full"]
    n_fail = len(d["fail"]) if d["fail"] is not None else 0
    # zorlu koşullar: son sistemde (G1) temiz görüntüye göre top AP'si %20'den fazla düşenler "belirgin"
    rob = d["rob"][d["rob"].model == FINAL_DETECTOR].set_index("condition")["ball_AP50"]
    cname = {"blur": "bulanıklık", "far_half": "uzak kamera", "night": "gece", "jpeg": "JPEG sıkıştırması"}
    rel = {c: rob[c] / rob["clean"] for c in cname if c in rob.index}
    big = [cname[c] for c in sorted(rel, key=rel.get) if rel[c] < 0.8]
    small = [cname[c] for c in sorted(rel, key=rel.get) if rel[c] >= 0.8]
    ve = lambda xs: ", ".join(xs[:-1]) + " ve " + xs[-1] if len(xs) > 1 else (xs[0] if xs else "")
    rob_txt = (f"Son sistemde {ve(big)} koşullarında top tespiti belirgin biçimde düşüyor"
               + (f"; {ve(small)} ise az etkiliyor." if small else "."))
    return {"E1": sayi(det["E1"], 3), "G1": sayi(det[FINAL_DETECTOR], 3), "E3": sayi(det["E3_ft1280_cp"], 3),
            "F1": sayi(v["pass_F1_mean"], 3), "FPS": sayi(round(float(d["pipe"]["pipeline_fps"])), 0),
            "NFAIL": sayi(n_fail, 0), "ROB": (rob_txt, rob_txt)}


# (başlangıç bölümü, metin şablonu) — blok, bir sonraki bloğun bölümüne kadar sürebilir
BLOCKS = [
    ("baslik", "Bilgisayarlı Görü: futbol videolarından anlık top hakimiyeti, pas sıklığı ve oyuncu kartları."),
    ("girdi_kart", "Girdi, tek kameralı bir maç videosu. Arayüzde demo seçiliyor ya da video yükleniyor; ilk "
                   "karelerden forma renkleri öğrenilip takımlarla eşleniyor."),
    ("cikarim_kart", "Her karede YOLOv8s dedektörü oyuncuları, hakemi ve topu buluyor; ByteTrack oyunculara kimlik "
                     "veriyor, forma rengiyle takım atanıyor ve saha noktalarından homografi hesaplanıyor. Sol üstteki "
                     "panelde top hakimiyeti, pas sayısı, dakika başına pas ve olay akışı kare kare güncelleniyor."),
    ("demo2", "İkinci örnekte sağ alttaki taktik haritada oyuncuların ve topun metre cinsinden konumu görülüyor; "
              "kırmızı üçgen o anki top sahibini gösteriyor. Top sahibi, topun oyuncunun ayak bölgesine uzaklığıyla ve "
              "kısa bir gecikme süresiyle belirleniyor. Sahiplik aynı takımda el değiştirince pas, rakibe geçince "
              "araya girme ya da top kaybı sayılıyor; şut, orta ve top taşıma da topun hızına ve konumuna göre "
              "ayrılıyor. Tüm hat, RTX 4050 ekran kartında saniyede yaklaşık {FPS} kare işliyor."),
    ("yorum_kart", "Analiz bitince her oyuncunun eylem sıklıkları CIES yöntemindeki gibi bir vektöre dönüştürülüyor "
                   "ve oyuncular sekiz teknik profilden birine ya da dengeli sınıfına atanıyor."),
    ("kart_", "Her oyuncu için bir karakter kartı üretiliyor. Hız, hareket, pas, top sürme, savunma, hücum ve top "
              "kontrolü puanları aynı kesitteki oyunculara göre yüzdelik olarak hesaplanıyor; genel puan ise "
              "oyuncunun profiline göre ağırlıklandırılıyor."),
    ("fig_tespit", "Test kümesinde top tespiti, ince ayarsız modelde {E1}, ek veriyle eğitilen son modelde {G1} AP "
                   "değerine ulaştı. Top kopyala-yapıştır artırımı ise {E3} ile geride kaldı; bu bir olumsuz sonuç."),
    ("fig_hakimiyet", "Son sistem pasları {F1} F1 ile buluyor; ancak kısa kliplerde hakimiyet yüzdesi hatası yüksek "
                      "kalıyor. {ROB}"),
    ("basarisiz_kart", "Başarısız örnekte yayın kamerası ve kalabalık bir ceza sahası var. Saha noktaları "
                       "bulunamadığı için taktik harita çizilemiyor ve hızlar yalnızca yaklaşık hesaplanabiliyor. Top "
                       "oyuncu kalabalığında kaybolduğunda sahiplik yanlış oyuncuya geçiyor; bu da hatalı pas ve top "
                       "kapma olaylarına yol açıyor."),
    ("hata_ornekleri", "İncelediğimiz {NFAIL} başarısız örnekte başlıca nedenler etiket kayması, hareket "
                       "bulanıklığı, elde tutulan top ve örtülme."),
    ("ozet", "Tüm sayılar raporda ve deney kayıtlarında yer alıyor. Teşekkürler."),
]


def plan(tl: dict, nums: dict) -> list[dict]:
    segs = tl["segments"]
    starts = []
    for key, _ in BLOCKS:
        i = next(i for i, s in enumerate(segs) if s["id"].startswith(key))
        starts.append(i)
    out = []
    for b, ((key, tmpl), i) in enumerate(zip(BLOCKS, starts)):
        end_i = starts[b + 1] if b + 1 < len(starts) else None
        t0 = segs[i]["start_s"]
        t1 = segs[end_i]["start_s"] if end_i is not None else tl["duration_s"]
        names = [SEG_TR.get("kart" if s["id"].startswith("kart_") else s["id"], s["id"]) for s in segs[i:end_i]]
        shown = list(dict.fromkeys(names))  # sırayı koruyarak tekrarları at
        spoken = " " + tmpl.format(**{k: v[1] for k, v in nums.items()}) + " "
        for a, r in SPOKEN.items():
            spoken = spoken.replace(a, r)
        out.append({"t0": t0, "t1": t1, "shown": shown, "text": tmpl.format(**{k: v[0] for k, v in nums.items()}),
                    "spoken": spoken.strip()})
    return out


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def write_markdown(blocks: list[dict], tl: dict, path: Path):
    words = sum(len(b["text"].split()) for b in blocks)
    lines = ["# Gösterim videosu — anlatım metni", "",
             f"`demo/gosterim_videosu.mp4` ({tl['duration_s'] / 60:.1f} dk) üzerine kendi sesinizle anlatım kaydı için "
             f"zamanlanmış metin (~{words} kelime). Sayılar `results/` çıktılarından üretilmiştir; metni değiştirirseniz "
             "`src/bgoru/experiments/make_narration.py` içindeki `BLOCKS` listesini düzenleyip betiği yeniden çalıştırın.",
             "", "Windows'ta kayıt: videoyu tam ekran oynatıp `Win + Alt + R` (Xbox Game Bar) ile mikrofonlu kayıt alın "
             "ya da Clipchamp'te videonun üzerine ses kaydı ekleyin.", "",
             "| Süre | Ekranda | Okunacak metin |", "|---|---|---|"]
    for b in blocks:
        lines.append(f"| {mmss(b['t0'])}–{mmss(b['t1'])} | {', '.join(b['shown'])} | {b['text']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[✓] {path}")


PS_TTS = r"""
param([string]$jobs)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and
    $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, $type) { $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $t.Wait(-1) | Out-Null; $t.Result }
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.Language -eq 'tr-TR' } | Select-Object -First 1
if ($null -eq $voice) { throw "Türkçe (tr-TR) ses yüklü değil: Ayarlar > Zaman ve dil > Konuşma" }
$synth.Voice = $voice
foreach ($j in (Get-Content -Raw -Encoding UTF8 $jobs | ConvertFrom-Json)) {
    $synth.Options.SpeakingRate = [double]$j.rate
    $s = Await ($synth.SynthesizeTextToStreamAsync([string]$j.text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
    $r = New-Object Windows.Storage.Streams.DataReader($s.GetInputStreamAt(0))
    $n = Await ($r.LoadAsync([uint32]$s.Size)) ([uint32])
    $b = New-Object byte[] $n; $r.ReadBytes($b); [IO.File]::WriteAllBytes([string]$j.out, $b)
}
Write-Output $voice.DisplayName
"""


def synth(jobs: list[dict], tmp: Path) -> str:
    ps, jp = tmp / "tts.ps1", tmp / "jobs.json"
    ps.write_text(PS_TTS, encoding="utf-8-sig")
    jp.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps), "-jobs", str(jp)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)
    return r.stdout.strip()


def read_wav(p: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(p)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "16 bit mono bekleniyor"
        return np.frombuffer(w.readframes(w.getnframes()), np.int16), w.getframerate()


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def make_tts(blocks: list[dict], tl: dict):
    import imageio_ffmpeg

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rates = [1.0] * len(blocks)
        for attempt in range(3):  # pencereye sığmayan bloklar hızlandırılarak yeniden sentezlenir
            jobs = [{"text": b["spoken"], "rate": rates[k], "out": str(tmp / f"b{k:02d}.wav")}
                    for k, b in enumerate(blocks)]
            voice = synth(jobs, tmp)
            clips = [read_wav(Path(j["out"])) for j in jobs]
            sr = clips[0][1]
            over = []
            for k, (b, (a, _)) in enumerate(zip(blocks, clips)):
                win = b["t1"] - b["t0"] - LEAD_S - GAP_S
                dur = len(a) / sr / 1.0
                b["dur"] = dur
                if dur > win:
                    over.append(k)
                    rates[k] = min(MAX_RATE, rates[k] * dur / win * 1.03)
            if not over:
                break
        for k in over:
            print(f"[!] blok {k} pencereye sığmıyor ({blocks[k]['dur']:.1f} sn); metni kısaltın")
        track = np.zeros(int(np.ceil(tl["duration_s"] * sr)), np.int32)
        for b, (a, _) in zip(blocks, clips):
            i0 = int((b["t0"] + LEAD_S) * sr)
            seg = a[: max(0, len(track) - i0)]
            track[i0:i0 + len(seg)] += seg
        wav = tmp / "anlatim.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(np.clip(track, -32768, 32767).astype(np.int16).tobytes())
        out = DEMO / "gosterim_videosu_anlatimli.mp4"
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(DEMO / "gosterim_videosu.mp4"),
               "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
               "-shortest", str(out)]
        subprocess.run(cmd, check=True)
    srt = ["\n".join([str(k + 1), f"{srt_time(b['t0'] + LEAD_S)} --> {srt_time(b['t0'] + LEAD_S + b['dur'])}",
                      b["text"], ""]) for k, b in enumerate(blocks)]
    (DEMO / "gosterim_videosu_anlatimli.srt").write_text("\n".join(srt), encoding="utf-8")
    fill = sum(b["dur"] for b in blocks) / tl["duration_s"]
    print(f"[✓] {out}  (ses: {voice}, hız: {', '.join(f'{r:.2f}' for r in rates)}, konuşma doluluğu %{fill * 100:.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", action="store_true", help="Windows Türkçe sesiyle anlatımlı video üret")
    args = ap.parse_args()
    if not TIMELINE.exists():
        sys.exit("Önce: python -m bgoru.experiments.make_showcase (zaman çizelgesi yok)")
    tl = json.loads(TIMELINE.read_text(encoding="utf-8"))
    blocks = plan(tl, numbers())
    write_markdown(blocks, tl, DEMO / "ANLATIM_METNI.md")
    if args.tts:
        make_tts(blocks, tl)


if __name__ == "__main__":
    main()
