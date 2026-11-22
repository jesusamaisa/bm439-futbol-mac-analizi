"""Top odaklı kopyala-yapıştır artırımı (projenin özgün veri hazırlama bileşeni).

Gerekçe: Topun SoccerNet karelerinde çoğu zaman 20 pikselden küçük olması ve kare
başına en fazla bir örnek bulunması, top sınıfını hem küçük hem de az temsil edilen
bir sınıf yapar. Bu betik eğitim kümesindeki gerçek top kesitlerini toplar ve
seçilen eğitim görüntülerine, **yalnızca çim olan bölgelere** dairesel yumuşak
maske ile yapıştırır.

Eğitim bütçesini sabit tutmak için görüntü sayısı DEĞİŞMEZ: görüntülerin `--frac`
kadarı artırılmış sürümüyle değiştirilir, kalanı aynen bağlanır (hard link).
Doğrulama ve test kümelerine dokunulmaz (sızıntı yok); top bankası yalnızca eğitim
ayrımından toplanır.

Kullanım:
    python -m bgoru.data.copy_paste --src data/soccernet --dst data/soccernet_cp --frac 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BALL = 2


def read_labels(p: Path) -> list[list[float]]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().strip().splitlines():
        if line.strip():
            c, *xywh = line.split()
            rows.append([int(c), *map(float, xywh)])
    return rows


def grass_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (32, 40, 40), (90, 255, 255))
    return cv2.erode(m, np.ones((9, 9), np.uint8))


def build_ball_bank(img_dir: Path, lbl_dir: Path, min_px: int = 5, margin: int = 2, limit: int = 4000, seed: int = 0):
    rng = random.Random(seed)
    files = sorted(lbl_dir.glob("*.txt"))
    rng.shuffle(files)
    bank = []
    for lf in files:
        balls = [r for r in read_labels(lf) if r[0] == BALL]
        if not balls:
            continue
        img = cv2.imread(str(img_dir / (lf.stem + ".jpg")))
        if img is None:
            continue
        H, W = img.shape[:2]
        for _, cx, cy, w, h in balls:
            bw, bh = w * W, h * H
            if min(bw, bh) < min_px:
                continue
            x0 = int(max(0, cx * W - bw / 2 - margin)); y0 = int(max(0, cy * H - bh / 2 - margin))
            x1 = int(min(W, cx * W + bw / 2 + margin)); y1 = int(min(H, cy * H + bh / 2 + margin))
            crop = img[y0:y1, x0:x1].copy()
            if crop.size:
                bank.append({"crop": crop, "src": lf.stem, "w": bw, "h": bh})
        if len(bank) >= limit:
            break
    return bank


def soft_circle(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r = np.sqrt(((yy - cy) / max(h / 2, 1)) ** 2 + ((xx - cx) / max(w / 2, 1)) ** 2)
    a = np.clip((1.0 - r) / 0.35, 0, 1)  # kenarda yumuşak geçiş
    return a[..., None]


def paste_balls(img, rows, bank, rng: random.Random, n_min=1, n_max=3, scale=(0.8, 1.25)):
    H, W = img.shape[:2]
    gm = grass_mask(img)
    occupied = np.zeros((H, W), np.uint8)
    for c, cx, cy, w, h in rows:  # mevcut nesnelerin üstüne yapıştırma
        x0, y0 = int((cx - w / 2) * W) - 6, int((cy - h / 2) * H) - 6
        x1, y1 = int((cx + w / 2) * W) + 6, int((cy + h / 2) * H) + 6
        occupied[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1
    ys, xs = np.nonzero((gm > 0) & (occupied == 0))
    if len(xs) == 0:
        return img, rows, 0
    added = 0
    for _ in range(rng.randint(n_min, n_max)):
        b = bank[rng.randrange(len(bank))]
        s = rng.uniform(*scale)
        crop = cv2.resize(b["crop"], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        ch, cw = crop.shape[:2]
        if ch < 3 or cw < 3:
            continue
        k = rng.randrange(len(xs))
        px, py = int(xs[k]) - cw // 2, int(ys[k]) - ch // 2
        if px < 0 or py < 0 or px + cw >= W or py + ch >= H:
            continue
        if occupied[py:py + ch, px:px + cw].any():
            continue
        alpha = soft_circle(ch, cw)
        roi = img[py:py + ch, px:px + cw].astype(np.float32)
        img[py:py + ch, px:px + cw] = (alpha * crop + (1 - alpha) * roi).astype(np.uint8)
        occupied[py - 6:py + ch + 6, px - 6:px + cw + 6] = 1
        rows.append([BALL, (px + cw / 2) / W, (py + ch / 2) / H, cw / W, ch / H])
        added += 1
    return img, rows, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "soccernet"))
    ap.add_argument("--dst", default=str(ROOT / "data" / "soccernet_cp"))
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    rng = random.Random(args.seed)
    si, sl = src / "images" / "train", src / "labels" / "train"
    di, dl = dst / "images" / "train", dst / "labels" / "train"
    if dst.exists():
        shutil.rmtree(dst)
    di.mkdir(parents=True); dl.mkdir(parents=True)

    bank = build_ball_bank(si, sl, seed=args.seed)
    print(f"[i] Top bankası: {len(bank)} kesit (yalnızca eğitim ayrımı)")
    imgs = sorted(si.glob("*.jpg"))
    chosen = set(rng.sample(range(len(imgs)), int(len(imgs) * args.frac)))
    n_added = 0
    for i, ip in enumerate(imgs):
        lp = sl / (ip.stem + ".txt")
        if i in chosen:
            img = cv2.imread(str(ip))
            rows = read_labels(lp)
            img, rows, a = paste_balls(img, rows, bank, rng)
            n_added += a
            cv2.imwrite(str(di / ip.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (dl / lp.name).write_text("\n".join(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in rows))
        else:
            try:
                os.link(ip, di / ip.name)
            except OSError:
                shutil.copy2(ip, di / ip.name)
            shutil.copy2(lp, dl / lp.name) if lp.exists() else (dl / lp.name).write_text("")
    info = {"images": len(imgs), "augmented_images": len(chosen), "pasted_balls": n_added,
            "bank_size": len(bank), "frac": args.frac, "seed": args.seed}
    (dst / "copy_paste_info.json").write_text(json.dumps(info, indent=1))
    print(f"[✓] {info}")


if __name__ == "__main__":
    main()
