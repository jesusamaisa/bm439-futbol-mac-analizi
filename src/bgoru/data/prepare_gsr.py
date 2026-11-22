"""SoccerNet-GSR-2025 'valid' ayrımı -> ek YOLO eğitim verisi (yerel ZIP arşivinden).

SoccerNet-GSR, SoccerNet-Tracking kliplerinin üzerine kuruludur (aynı 30 sn'lik klipler, ek saha/kimlik etiketleri):
    GSR train = bizim train + val (oyun 4, 6, 9) · GSR test = bizim test (oyun 7, 8, 11) · GSR valid = 58 yeni klip.
Yalnızca GSR valid yeni maçlardan gelir; bu betik yalnızca onu dönüştürür ve oyun kimliklerinin / kliplerin bizim test
ve doğrulama kümelerimizle kesişmediğini doğrular — kesişme varsa hiçbir şey yazmadan durur.

Sınıflar prepare_soccernet ile aynıdır: 0=player (oyuncu + kaleci), 1=referee, 2=ball; 'other' yok sayılır.

Kullanım (proje kökünden):
    python -m bgoru.data.prepare_gsr                      # her 15. kare (mevcut eğitim kümesiyle aynı örnekleme)
    python -m bgoru.data.prepare_gsr --stride 5           # daha yoğun örnekleme
Çıktı: data/soccernet_gsr/{images,labels}/train/, raw/<SNGS-xxx>/ (gt.txt, gameinfo.ini, seqinfo.ini, meta.json),
       manifests/ (train_manifest.csv, train_split.json, leak_check.json)
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAT2CLS = {1: 0, 2: 0, 3: 1, 4: 2}  # GSR kategori -> proje sınıfı (player, goalkeeper -> 0; referee -> 1; ball -> 2)
CAT2ROLE = {1: "player", 2: "goalkeeper", 3: "referee", 4: "ball", 7: "other"}


def frame_of(file_name: str) -> int:
    return int(Path(file_name).stem)


def yolo_lines(objs: list[dict], W: int, H: int) -> list[str]:
    """GSR nesne etiketleri (bbox_image: sol-üst x, y, w, h piksel) -> YOLO satırları (görüntüye kırpılmış)."""
    out = []
    for a in objs:
        c = CAT2CLS.get(a["category_id"])
        b = a.get("bbox_image")
        if c is None or not b:
            continue
        x0, y0 = max(0.0, float(b["x"])), max(0.0, float(b["y"]))
        x1, y1 = min(W, float(b["x"]) + float(b["w"])), min(H, float(b["y"]) + float(b["h"]))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        out.append(f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return out


def our_split_info() -> tuple[dict, set]:
    """Bizim ayrımlarımızın oyunları ve klip kimlikleri (clipStart, actionPosition)."""
    games = {}
    for f in ("train_split.json", "test_split.json"):
        p = ROOT / "data" / "soccernet" / "manifests" / f
        if p.exists():
            for k, v in json.loads(p.read_text(encoding="utf-8")).items():
                games[k] = set(int(g) for g in v["games"])
    clips = set()
    for gi in (ROOT / "data" / "soccernet" / "raw").glob("*/gameinfo.ini"):
        cp = configparser.ConfigParser()
        cp.read(gi)
        s = cp["Sequence"]
        clips.add((str(s.get("clipStart")), str(s.get("actionPosition"))))
    return games, clips


def raw_files(info: dict, objs_by_frame: dict, W: int, H: int, n: int) -> dict[str, str]:
    """Takım ataması / hakimiyet araçlarının okuduğu SoccerNet-Tracking biçiminde ham dosyalar."""
    tracks, gt = {}, []
    for f in sorted(objs_by_frame):
        for a in objs_by_frame[f]:
            b, tid = a.get("bbox_image"), a.get("track_id")
            if not b or tid is None:
                continue
            at = a.get("attributes") or {}
            role = CAT2ROLE.get(a["category_id"], "other")
            team = at.get("team")
            desc = (f"{role} team {team}" if role in ("player", "goalkeeper") and team else role)
            jersey = at.get("jersey") or ("1" if role == "ball" else "")
            tracks.setdefault(int(tid), f"{desc};{jersey}")
            gt.append(f"{f},{int(tid)},{float(b['x']):.1f},{float(b['y']):.1f},{float(b['w']):.1f},{float(b['h']):.1f},1,-1,-1,-1")
    gi = ["[Sequence]", f"name={info['name']}", f"gameID={info['game_id']}",
          f"actionPosition={info.get('action_position', '')}", f"actionClass={info.get('action_class', '')}",
          f"visibility={info.get('visibility', '')}", f"clipStart={info.get('clip_start', '')}",
          f"gameTimeStart={info.get('game_time_start', '')}", f"clipStop={info.get('clip_stop', '')}",
          f"gameTimeStop={info.get('game_time_stop', '')}", f"num_tracklets={len(tracks)}"]
    gi += [f"trackletID_{t}= {d}" for t, d in sorted(tracks.items())]
    si = ["[Sequence]", f"name={info['name']}", "imDir=img1", f"frameRate={info.get('frame_rate', 25)}",
          f"seqLength={n}", f"imWidth={W}", f"imHeight={H}", "imExt=.jpg"]
    return {"gt.txt": "\n".join(gt) + "\n", "gameinfo.ini": "\n".join(gi) + "\n", "seqinfo.ini": "\n".join(si) + "\n"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ROOT / "data" / "soccernet_gsr" / "valid.zip"))
    ap.add_argument("--stride", type=int, default=15, help="her N karede bir kare (mevcut eğitim kümesi: 15)")
    ap.add_argument("--out", default=str(ROOT / "data" / "soccernet_gsr"))
    args = ap.parse_args()
    out = Path(args.out)
    z = zipfile.ZipFile(args.zip)
    names = set(z.namelist())
    label_files = sorted(n for n in names if n.endswith("Labels-GameState.json"))
    print(f"[i] {len(label_files)} sekans etiket dosyası: {args.zip}")

    # 1) etiketleri oku + sızıntı denetimi (yazmadan önce)
    seqs = {}
    for lf in label_files:
        d = json.loads(z.read(lf))
        seqs[lf.split("/")[0]] = d
    ours, our_clips = our_split_info()
    gsr_games = {int(d["info"]["game_id"]) for d in seqs.values()}
    protected = ours.get("test", set()) | ours.get("val", set())
    clip_hits = sorted(s for s, d in seqs.items()
                       if (str(d["info"].get("clip_start")), str(d["info"].get("action_position"))) in our_clips)
    leak = {"gsr_games": sorted(gsr_games), "our_games": {k: sorted(v) for k, v in ours.items()},
            "overlap_with_test_or_val_games": sorted(gsr_games & protected),
            "overlap_with_train_games": sorted(gsr_games & ours.get("train", set())),
            "clips_identical_to_ours": clip_hits}
    print("[i] sızıntı denetimi:", json.dumps(leak))
    if leak["overlap_with_test_or_val_games"] or clip_hits:
        raise SystemExit("[x] GSR klipleri test/doğrulama maçlarımızla veya kliplerimizle kesişiyor; dönüştürme durduruldu.")

    # 2) dönüştür
    img_dir, lbl_dir = out / "images" / "train", out / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rows, stats = [], defaultdict(int)
    for seq, d in sorted(seqs.items()):
        info = d["info"]
        imgs = {im["image_id"]: im for im in d["images"]}
        objs = defaultdict(list)
        for a in d["annotations"]:
            if a.get("supercategory") == "object" and a["image_id"] in imgs:
                objs[frame_of(imgs[a["image_id"]]["file_name"])].append(a)
        W, H = int(d["images"][0]["width"]), int(d["images"][0]["height"])
        n = int(info.get("seq_length", len(d["images"])))
        raw = out / "raw" / seq
        raw.mkdir(parents=True, exist_ok=True)
        for fn, txt in raw_files(info, objs, W, H, n).items():
            (raw / fn).write_text(txt, encoding="utf-8")
        (raw / "meta.json").write_text(json.dumps({"name": seq, "gameID": int(info["game_id"]),
                                                   "actionClass": info.get("action_class", ""),
                                                   "gameTimeStart": info.get("game_time_start", ""),
                                                   "split": "gsr_valid", "W": W, "H": H, "n": n},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
        for im in sorted(d["images"], key=lambda x: x["file_name"]):
            f = frame_of(im["file_name"])
            if (f - 1) % args.stride or not im.get("is_labeled", True):
                continue
            member = f"{seq}/{info.get('im_dir', 'img1')}/{im['file_name']}"
            if member not in names:
                stats["missing_image"] += 1
                continue
            if im.get("ignore_regions_x"):
                stats["frames_with_ignore_regions"] += 1
            stem = f"{seq}_{f:06d}"
            dst = img_dir / f"{stem}.jpg"
            if not dst.exists():
                dst.write_bytes(z.read(member))
            lines = yolo_lines(objs.get(f, []), W, H)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            for ln in lines:
                stats[("player", "referee", "ball")[int(ln[0])]] += 1
            stats["images"] += 1
            stats["images_with_ball"] += any(ln.startswith("2 ") for ln in lines)
            rows.append([f"images/train/{stem}.jpg", seq, int(info["game_id"]), f, "train_gsr", info.get("action_class", "")])
        print(f"    {seq}: oyun {info['game_id']}, {sum(1 for r in rows if r[1] == seq)} kare", flush=True)

    # 3) manifest ve özet
    man = out / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    with open(man / "train_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "sequence", "gameID", "frame", "split", "actionClass"])
        w.writerows(rows)
    split = {"train_gsr": {"sequences": sorted({r[1] for r in rows}), "games": sorted({r[2] for r in rows}),
                           "images": len(rows), "stride": args.stride, "boxes": {k: stats[k] for k in ("player", "referee", "ball")},
                           "images_with_ball": stats["images_with_ball"],
                           "frames_with_ignore_regions": stats["frames_with_ignore_regions"],
                           "missing_image": stats["missing_image"]}}
    (man / "train_split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    (man / "leak_check.json").write_text(json.dumps(leak, indent=1), encoding="utf-8")
    s = split["train_gsr"]
    print(f"[✓] {s['images']} kare, {len(s['sequences'])} sekans, {len(s['games'])} oyun {s['games']}; kutular {s['boxes']}; "
          f"topu görünen kare {s['images_with_ball']}")


if __name__ == "__main__":
    main()
