"""Verimlilik ölçümü: parametre sayısı, GFLOPs, dosya boyutu, çıkarım süresi ve GPU belleği.

Protokol: tek görüntü (batch=1), FP16, CUDA; 20 ısınma + 200 ölçüm; süre = model.predict
(ön işleme + ileri geçiş + NMS) uçtan uca, torch.cuda.synchronize ile. Bellek = ölçüm sırasında
torch.cuda.max_memory_allocated (PyTorch ayırıcısı). Ayrıca tam analiz hattının (tespit + takip +
takım + homografi + hakimiyet + çizim) taktik kamera videosundaki kare hızı ölçülür.

Kullanım: python -m bgoru.experiments.benchmark
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import torch

from bgoru.config import FINAL_DETECTOR, RESULTS, ROOT, WEIGHTS
from bgoru.experiments.common import MODELS
from bgoru.experiments.registry import log_experiment


def bench_model(mid, img, warm=20, reps=200):
    from ultralytics import YOLO

    w, kind, imgsz = MODELS[mid]
    m = YOLO(str(WEIGHTS / w))
    n_par = sum(p.numel() for p in m.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops

        gf = get_flops(m.model, imgsz)
    except Exception:  # noqa: BLE001
        gf = float("nan")
    for _ in range(warm):
        m.predict(img, imgsz=imgsz, half=True, verbose=False, conf=0.1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        m.predict(img, imgsz=imgsz, half=True, verbose=False, conf=0.1)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    mem = torch.cuda.max_memory_allocated() / 1024 ** 2
    return {"model": mid, "imgsz": imgsz, "params_M": n_par / 1e6, "GFLOPs": gf,
            "file_MB": (WEIGHTS / w).stat().st_size / 1024 ** 2, "latency_ms_mean": float(np.mean(ts)),
            "latency_ms_std": float(np.std(ts)), "latency_ms_p95": float(np.percentile(ts, 95)),
            "fps": 1000 / float(np.mean(ts)), "gpu_peak_MB": mem, "batch": 1, "precision": "FP16"}


def bench_pipeline(video, frames=300):
    from bgoru.pipeline import AnalyzerConfig, Analyzer

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    an = Analyzer(AnalyzerConfig(), fps)
    imgs = []
    while len(imgs) < frames:
        ok, f = cap.read()
        if not ok:
            break
        imgs.append(f)
    for f in imgs[:20]:
        an.process(f)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for f in imgs[20:]:
        an.process(f)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = len(imgs) - 20
    stages = {k: v / an.f * 1000 for k, v in an.timing.items()}
    return {"pipeline_fps": n / dt, "pipeline_ms": dt / n * 1000, "frames": n,
            **{f"stage_ms_{k}": round(v, 2) for k, v in stages.items()}}


def main():
    if not torch.cuda.is_available():
        raise SystemExit("[!] benchmark yalnızca CUDA GPU'da çalışır (rapordaki ölçüm protokolü: FP16, CUDA). "
                         "CPU'da uygulama/CLI çalışır ancak bu ölçüm yapılmaz.")
    img = cv2.imread(str(next((ROOT / "data" / "soccernet" / "images" / "test").glob("*.jpg"))))
    rows = []
    dev = torch.cuda.get_device_name(0)
    for mid in ["E1_coco640", "E1_coco1280", "E2_ft640", "E3_ft1280_cp", "A1_ft1280", "A2_ft640_cp", "R_hmzbo640",
                "G1_ft1280_gsr"]:
        if not (WEIGHTS / MODELS[mid][0]).exists():
            continue
        r = bench_model(mid, img)
        r["device"] = dev
        rows.append(r)
        print(f"  {mid:14s} {r['latency_ms_mean']:.1f}±{r['latency_ms_std']:.1f} ms  {r['fps']:.1f} FPS  "
              f"{r['gpu_peak_MB']:.0f} MB  {r['params_M']:.2f}M param  {r['GFLOPs']:.1f} GFLOPs", flush=True)
        log_experiment(f"bench_{mid}", "benchmark", MODELS[mid][0], {"imgsz": r["imgsz"], "batch": 1, "half": True},
                       "1 test görüntüsü x 200 tekrar", {k: (round(v, 3) if isinstance(v, float) else v)
                                                        for k, v in r.items()}, 0.0, dev)
    out = RESULTS / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "model_benchmark.csv", index=False)
    vid = ROOT / "demo" / "input" / "taktik_demo1.mp4"
    if vid.exists():
        p = bench_pipeline(vid)
        p["device"] = dev
        pd.DataFrame([p]).to_csv(out / "pipeline_benchmark.csv", index=False)
        print("  tam hat:", p)
        log_experiment("bench_pipeline", "benchmark", f"{FINAL_DETECTOR} + keypoints + ByteTrack", {"video": vid.name}, "taktik_demo1",
                       {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}, 0.0, dev)


if __name__ == "__main__":
    main()
