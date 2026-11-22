"""SoccerNet-Tracking-2023 -> YOLO formatı dönüştürücü (seçici indirme ile).

Kullanım (proje kökünden):
    python -m bgoru.data.prepare_soccernet --split train --stride 15
    python -m bgoru.data.prepare_soccernet --split test  --stride 25
    python -m bgoru.data.prepare_soccernet --split test  --dense --stride 2 --max-seqs 10

Sınıflar: 0=player (oyuncu + kaleci), 1=referee (hakem), 2=ball (top)
Ham MOT etiketleri (gt.txt) ve gameinfo.ini her sekans için data/soccernet/raw/<SEQ>/
altına kopyalanır; takım ataması ve top hakimiyeti değerlendirmesi bunları kullanır.

Veri sızıntısı önlemi: resmî train/test ayrımı korunur. Doğrulama kümesi, eğitim
ayrımından *oyun kimliği (gameID)* bazında ayrılır; aynı maçın klipleri iki farklı
kümeye düşmez.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from bgoru.data.remote_zip import fetch_member, open_remote_zip

HF_URL = "https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023/resolve/main/{split}.zip"
CLASS_NAMES = ["player", "referee", "ball"]
ROOT = Path(__file__).resolve().parents[3]


def tracklet_class(desc: str) -> tuple[int, str]:
    """gameinfo.ini açıklamasını (ör. 'player team left;4') sınıf + takım etiketine çevirir."""
    d = desc.strip().lower()
    role = d.split(";")[0].strip()
    team = "left" if "team left" in role else ("right" if "team right" in role else "none")
    if role.startswith("ball"):
        return 2, "none"
    if role.startswith("referee"):
        return 1, "none"
    if role.startswith("player") or role.startswith("goalkeeper"):
        return 0, team
    return -1, "none"  # 'other' vb. -> eğitimde yok sayılır


def parse_gameinfo(text: str) -> dict:
    cp = configparser.ConfigParser()
    cp.read_string(text)
    s = cp["Sequence"]
    tracks = {}
    for k, v in s.items():
        if k.startswith("trackletid_"):
            tid = int(k.split("_")[1])
            cls, team = tracklet_class(v)
            role = v.split(";")[0].strip()
            jersey = v.split(";")[1].strip() if ";" in v else ""
            tracks[tid] = {"cls": cls, "team": team, "role": role, "jersey": jersey,
                           "goalkeeper": role.lower().startswith("goalkeeper")}
    return {"name": s.get("name"), "gameID": int(s.get("gameid", -1)),
            "actionClass": s.get("actionclass", ""), "gameTimeStart": s.get("gametimestart", ""),
            "tracks": tracks}


def yolo_lines(rows, tracks, W, H):
    out = []
    for (tid, x, y, w, h) in rows:
        t = tracks.get(tid)
        if t is None or t["cls"] < 0:
            continue
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
        out.append(f"{t['cls']} {cx:.6f} {cy:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--stride", type=int, default=15, help="her N karede bir kare al")
    ap.add_argument("--dense", action="store_true", help="aşağı akış (top hakimiyeti) testi için yoğun kare kümesi")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2, help="eğitim oyunlarından doğrulamaya ayrılacak oran")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "data" / "soccernet"))
    args = ap.parse_args()

    out = Path(args.out)
    url = HF_URL.format(split=args.split)
    print(f"[i] Merkezi dizin okunuyor: {url}")
    z = open_remote_zip(url)
    infos = {i.filename: i for i in z.infolist()}
    seqs = sorted({f.split("/")[1] for f in infos if f.count("/") >= 2 and f.split("/")[1].startswith("SNMOT")})
    print(f"[i] {len(seqs)} sekans bulundu")

    # 1) küçük metin dosyalarını oku (gameinfo, seqinfo, gt)
    meta = {}
    for seq in seqs:
        base = f"{args.split}/{seq}/"
        gi = parse_gameinfo(z.read(infos[base + "gameinfo.ini"]).decode())
        si = configparser.ConfigParser()
        si.read_string(z.read(infos[base + "seqinfo.ini"]).decode())
        W, H = int(si["Sequence"]["imWidth"]), int(si["Sequence"]["imHeight"])
        n = int(si["Sequence"]["seqLength"])
        gt_txt = z.read(infos[base + "gt/gt.txt"]).decode()
        raw_dir = out / "raw" / seq
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "gt.txt").write_text(gt_txt)
        (raw_dir / "gameinfo.ini").write_text(z.read(infos[base + "gameinfo.ini"]).decode())
        (raw_dir / "seqinfo.ini").write_text(z.read(infos[base + "seqinfo.ini"]).decode())
        (raw_dir / "meta.json").write_text(json.dumps({**{k: v for k, v in gi.items() if k != "tracks"},
                                                         "split": args.split, "W": W, "H": H, "n": n,
                                                         "tracks": gi["tracks"]}, ensure_ascii=False, indent=1))
        rows = defaultdict(list)
        for line in gt_txt.strip().splitlines():
            p = line.split(",")
            rows[int(p[0])].append((int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
        meta[seq] = {"gi": gi, "W": W, "H": H, "n": n, "rows": rows}

    # 2) ayrım: test -> test ; train -> oyun bazlı train/val
    if args.split == "train":
        games = sorted({m["gi"]["gameID"] for m in meta.values()})
        rng = random.Random(args.seed)
        rng.shuffle(games)
        n_val = max(1, round(len(games) * args.val_frac))
        val_games = set(games[:n_val])
        split_of = {s: ("val" if m["gi"]["gameID"] in val_games else "train") for s, m in meta.items()}
    else:
        split_of = {s: "test" for s in meta}

    sel_seqs = seqs[: args.max_seqs] if args.max_seqs else seqs
    if args.dense:
        # aşağı akış değerlendirmesi: sekans başına ardışık kareler
        rng = random.Random(args.seed)
        sel_seqs = sorted(rng.sample(seqs, args.max_seqs)) if args.max_seqs else seqs

    jobs = []
    for seq in sel_seqs:
        m = meta[seq]
        for f in range(1, m["n"] + 1, args.stride):
            name = f"{args.split}/{seq}/img1/{f:06d}.jpg"
            if name not in infos:
                continue
            if args.dense:
                dst = out / "dense" / seq / "img1" / f"{f:06d}.jpg"
                lbl = None
            else:
                sp = split_of[seq]
                stem = f"{seq}_{f:06d}"
                dst = out / "images" / sp / f"{stem}.jpg"
                lbl = out / "labels" / sp / f"{stem}.txt"
            jobs.append((seq, f, name, dst, lbl))

    print(f"[i] {len(jobs)} kare indirilecek ({args.workers} iş parçacığı)")
    local = threading.local()

    def work(job):
        seq, f, name, dst, lbl = job
        if not hasattr(local, "s"):
            local.s = requests.Session()
        if not dst.exists():
            data = fetch_member(url, infos[name], local.s)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        if lbl is not None:
            m = meta[seq]
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("\n".join(yolo_lines(m["rows"].get(f, []), m["gi"]["tracks"], m["W"], m["H"])))
        return seq, f

    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for fu in as_completed(futs):
            fu.result()
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(jobs)}", flush=True)

    # 3) manifest
    man = out / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}{'_dense' if args.dense else ''}"
    with open(man / f"{tag}_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "sequence", "gameID", "frame", "split", "actionClass"])
        for seq, f, name, dst, lbl in jobs:
            sp = "dense_test" if args.dense else split_of[seq]
            w.writerow([dst.relative_to(out).as_posix(), seq, meta[seq]["gi"]["gameID"], f, sp,
                        meta[seq]["gi"]["actionClass"]])
    summary = defaultdict(lambda: {"sequences": set(), "games": set(), "images": 0})
    for seq, f, name, dst, lbl in jobs:
        sp = "dense_test" if args.dense else split_of[seq]
        summary[sp]["sequences"].add(seq)
        summary[sp]["games"].add(meta[seq]["gi"]["gameID"])
        summary[sp]["images"] += 1
    js = {k: {"sequences": sorted(v["sequences"]), "games": sorted(v["games"]), "images": v["images"]}
          for k, v in summary.items()}
    (man / f"{tag}_split.json").write_text(json.dumps(js, indent=1))
    for k, v in js.items():
        print(f"[✓] {k}: {v['images']} kare, {len(v['sequences'])} sekans, {len(v['games'])} oyun")


if __name__ == "__main__":
    main()
