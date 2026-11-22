"""Komut satırı arayüzü.

    python -m bgoru.cli analyze --video demo/input/taktik_demo1.mp4 --out results/analiz_demo1 \
        --teams Fransa Isvicre
    python -m bgoru.cli env          # ortam bilgisini environment/env_info.json dosyasına yazar
"""
from __future__ import annotations

import argparse
import json
import platform
import sys

from bgoru.config import FINAL_CONF_BALL, FINAL_DET_WEIGHTS, ROOT


def cmd_analyze(a):
    from bgoru.pipeline import AnalyzerConfig, analyze_video

    cfg = AnalyzerConfig(det_weights=a.weights, det_imgsz=a.imgsz, use_homography=not a.no_homography,
                         team_method=a.team_method, ball_fill=not a.no_ball_fill, team_names=a.teams,
                         show_keypoints=a.show_keypoints, show_palette=a.show_palette, conf_ball=a.conf_ball)
    last = {"f": -1}

    def cb(an, out, snap):
        if an.f % 50 == 0 and an.f != last["f"]:
            last["f"] = an.f
            p = snap["possession_pct"]
            print(f"  kare {an.f:5d} | hakimiyet {p[0]:5.1f}% - {p[1]:5.1f}% | pas {snap['passes']} "
                  f"| pas/dk {snap['passes_per_min']}", flush=True)
        return True

    s = analyze_video(a.video, a.out, cfg, max_frames=a.max_frames, start_s=a.start, callback=cb)
    f = s["final"]
    print(json.dumps({"hakimiyet_%": f["possession_pct"], "pas": f["passes"], "pas_dk": f["passes_per_min"],
                      "pas_isabet_%": f["pass_accuracy"], "fps": s.get("wall_fps"), "kart_sayisi": len(s["cards"]),
                      "video": s.get("video")}, ensure_ascii=False, indent=1))


def cmd_env(_):
    import cv2
    import numpy
    import torch
    import ultralytics

    info = {
        "os": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2) if torch.cuda.is_available() else None,
        "cpu": platform.processor(), "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
        "numpy": numpy.__version__,
    }
    try:
        import subprocess

        info["nvidia_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                               capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    for m in ("supervision", "sklearn", "streamlit", "matplotlib", "pandas", "skimage"):
        try:
            info[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            pass
    p = ROOT / "environment" / "env_info.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser(prog="bgoru")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="bir videoyu analiz et")
    a.add_argument("--video", required=True)
    a.add_argument("--out", default=str(ROOT / "results" / "analiz"))
    a.add_argument("--weights", default=FINAL_DET_WEIGHTS)
    a.add_argument("--imgsz", type=int, default=1280)
    a.add_argument("--conf-ball", type=float, default=FINAL_CONF_BALL)
    a.add_argument("--team-method", default="proposed_local")
    a.add_argument("--teams", nargs=2, default=["Takım A", "Takım B"])
    a.add_argument("--no-homography", action="store_true")
    a.add_argument("--no-ball-fill", action="store_true")
    a.add_argument("--show-keypoints", action="store_true")
    a.add_argument("--show-palette", action="store_true")
    a.add_argument("--max-frames", type=int, default=None)
    a.add_argument("--start", type=float, default=0.0, help="başlangıç saniyesi")
    a.set_defaults(fn=cmd_analyze)
    e = sub.add_parser("env", help="ortam bilgisini kaydet")
    e.set_defaults(fn=cmd_env)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
