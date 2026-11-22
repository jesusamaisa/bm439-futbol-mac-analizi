"""Hata analizi: önerilen sistemin (E3 + önerilen takım ataması) test kümesindeki başarısız
örneklerini çıkarır, ölçülebilir özniteliklerini hesaplar ve kural tabanlı bir ön kategori atar.

Çıktılar (results/failures/):
  * imgs/F01.jpg ...      bağlamlı kırpıntılar (yeşil = GT, kırmızı = tahmin, mavi = eşik altı tahmin)
  * failures.csv          kimlik, tür, sekans, kare, öznitelikler, otomatik kategori
  * contact_sheet_*.jpg   rapor için toplu görünüm

Seçim: sekanslara yayılmış ve rastgele (tohum 0) — en kötü örnekler seçilmez, temsilîdir.
Kullanım: python -m bgoru.experiments.failure_cases
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bgoru.config import RESULTS
from bgoru.experiments.common import box_iou, load_split, match_image, predict_split
from bgoru.experiments.seqdata import SN

OUT = RESULTS / "failures"
MODEL = "E3_ft1280_cp"


def lap_var(img, box, pad=6):
    x0, y0, x1, y1 = box.astype(int)
    c = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    if c.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def green_ratio(img, box, pad=20):
    x0, y0, x1, y1 = box.astype(int)
    c = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    if c.size == 0:
        return 0.0
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 0] >= 32) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 40)).mean())


def crop_ctx(img, box, size=180, draw=()):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(size // 2, int(max(box[2] - box[0], box[3] - box[1]) * 0.8))
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
    vis = img.copy()
    for b, col in draw:
        cv2.rectangle(vis, tuple(b[:2].astype(int)), tuple(b[2:4].astype(int)), col, 1)
    c = vis[y0:y1, x0:x1]
    return cv2.resize(c, (320, 320), interpolation=cv2.INTER_NEAREST if c.shape[0] < 160 else cv2.INTER_LINEAR)


def categorize_fn(size, best_iou, best_conf_any, thr, near_player, blur, green, near_pred=False):
    if near_pred:
        return "Etiket ötelemesi (eşik üstü tahmin yakında, IoU<0.5)"
    if best_conf_any is not None and best_conf_any < thr:
        return "Eşik altında tespit (düşük güven)"
    if 0.1 <= best_iou < 0.5:
        return "Konum/kutu hatası (IoU<0.5)"
    if near_player:
        return "Oyuncuyla örtüşme / ayak temasında"
    if size < 10:
        return "Çok küçük top (<10 px)"
    if blur < 30:
        return "Hareket bulanıklığı"
    if green < 0.35:
        return "Saha dışı / tribün önünde"
    return "Karmaşık arka plan / diğer"


def categorize_fp(in_player, green, size, near_gt_ball=False):
    if near_gt_ball:
        return "Etiket ötelemesi (tahmin topun üstünde, GT kutusu kaymış)"
    if in_player:
        return "Oyuncu parçası (ayak/kafa/ayakkabı)"
    if green < 0.35:
        return "Tribün / reklam panosu / saha dışı"
    if size < 8:
        return "Saha çizgisi veya beyaz leke"
    return "Topa benzer nesne (yedek top, krampon, çizgi kesişimi)"


def main(n_ball_fn=12, n_ball_fp=8, n_player=4, n_team=6, seed=0):
    rng = random.Random(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "imgs").mkdir(exist_ok=True)
    test = load_split("test")
    preds = predict_split(MODEL, "test", test)
    det = pd.read_csv(RESULTS / "detection" / "detection_results.csv")
    r = det[det["model_id"] == MODEL].iloc[0]
    thr_b, thr_p = float(r["ball_conf_thr"]), float(r["player_conf_thr"])

    cands = {"ball_FN": [], "ball_FP": [], "player_FN": []}
    for i, (it, p) in enumerate(zip(test, preds)):
        gb = it["boxes"][it["cls"] == 2]
        pb_all = p[p[:, 5] == 2]
        pb = pb_all[pb_all[:, 4] >= thr_b]
        tp, used = match_image(pb, gb, 0.5)
        for j in np.where(~used)[0]:
            cands["ball_FN"].append((i, gb[j]))
        for k in np.where(~tp)[0]:
            cands["ball_FP"].append((i, pb[k]))
        gp = it["boxes"][it["cls"] == 0]
        pp = p[(p[:, 5] == 0) & (p[:, 4] >= thr_p)]
        _, usedp = match_image(pp, gp, 0.5)
        for j in np.where(~usedp)[0]:
            cands["player_FN"].append((i, gp[j]))

    def spread_pick(lst, n):
        by_seq = {}
        for x in lst:
            by_seq.setdefault(test[x[0]]["seq"], []).append(x)
        seqs = sorted(by_seq)
        rng.shuffle(seqs)
        out = []
        while len(out) < n and any(by_seq.values()):
            for s in seqs:
                if by_seq[s] and len(out) < n:
                    out.append(by_seq[s].pop(rng.randrange(len(by_seq[s]))))
        return out

    rows, fid = [], 0
    for kind, n in (("ball_FN", n_ball_fn), ("ball_FP", n_ball_fp), ("player_FN", n_player)):
        for i, box in spread_pick(cands[kind], n):
            fid += 1
            it, p = test[i], preds[i]
            img = cv2.imread(str(it["path"]))
            b = box[:4]
            size = float(max(b[2] - b[0], b[3] - b[1]))
            players = it["boxes"][it["cls"] == 0]
            ovl = float(box_iou(b[None], players).max()) if len(players) else 0.0
            near = False
            if len(players):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                near = bool(((players[:, 0] - 4 <= cx) & (cx <= players[:, 2] + 4) & (players[:, 1] <= cy) &
                             (cy <= players[:, 3] + 6)).any())
            blur, green = lap_var(img, b), green_ratio(img, b)
            cls = 2 if kind.startswith("ball") else 0
            pc = p[p[:, 5] == cls]
            best_iou, best_conf = 0.0, None
            if len(pc) and kind.endswith("FN"):
                ious = box_iou(b[None], pc[:, :4])[0]
                k = int(np.argmax(ious))
                best_iou = float(ious[k])
                good = pc[ious >= 0.5]
                best_conf = float(good[:, 4].max()) if len(good) else None
            draw = [(g, (0, 255, 0)) for g in it["boxes"][it["cls"] == cls]]
            draw += [(q, (0, 0, 255) if q[4] >= (thr_b if cls == 2 else thr_p) else (255, 128, 0)) for q in pc if q[4] >= 0.05]
            cv2.imwrite(str(OUT / "imgs" / f"F{fid:02d}.jpg"), crop_ctx(img, b, draw=draw))
            from bgoru.experiments.common import match_image_center
            if kind == "ball_FN":
                pab = pc[pc[:, 4] >= thr_b]
                near_pred = bool(match_image_center(pab, b[None])[1].any()) if len(pab) else False
                cat = categorize_fn(size, best_iou, best_conf, thr_b, near, blur, green, near_pred)
            elif kind == "ball_FP":
                gb = it["boxes"][it["cls"] == 2]
                near_gt = bool(match_image_center(box[None], gb)[0].any()) if len(gb) else False
                cat = categorize_fp(near, green, size, near_gt)
            else:
                cat = "Örtülme (oyuncu kalabalığı)" if ovl > 0.2 else ("Kısmi görünür / kadraj kenarı"
                                                                         if b[0] < 5 or b[2] > 1915 or b[3] > 1075
                                                                         else "Küçük/uzak oyuncu veya bulanıklık")
            rows.append({"id": f"F{fid:02d}", "type": kind, "seq": it["seq"], "image": it["path"].name,
                         "size_px": round(size, 1), "player_overlap_iou": round(ovl, 2), "near_player": near,
                         "laplacian_var": round(blur, 1), "green_ratio": round(green, 2),
                         "best_iou": round(best_iou, 2), "best_conf_iou50": best_conf, "auto_category": cat})

    # takım ataması hataları (eval_team önerilen yöntem kayıtları)
    team_errs = []
    for f in sorted((RESULTS / "team").glob("recs_*.csv")):
        df = pd.read_csv(f)
        df = df[~df["gk"]]
        if df.empty:
            continue
        flip = (df["pred"] == 1 - df["gt"]).mean() > (df["pred"] == df["gt"]).mean()
        pred = 1 - df["pred"] if flip else df["pred"]
        bad = df[pred != df["gt"]]
        seq = f.stem.replace("recs_", "")
        for _, rr in bad.iterrows():
            team_errs.append((seq, int(rr["frame"]), int(rr["tid"]), int(rr["gt"])))
    rng.shuffle(team_errs)
    by_seq = {}
    for e in team_errs:
        by_seq.setdefault(e[0], []).append(e)
    picked, seen = [], set()
    while len(picked) < n_team and any(by_seq.values()):
        for sq in sorted(by_seq, key=lambda q: -len(by_seq[q])):
            while by_seq[sq]:
                e = by_seq[sq].pop()
                if (e[0], e[2]) not in seen:
                    seen.add((e[0], e[2])); picked.append(e)
                    break
            if len(picked) >= n_team:
                break
    for seq, frame, tid, gt in picked:
        path = SN / "dense" / seq / "img1" / f"{frame:06d}.jpg"
        img = cv2.imread(str(path))
        from bgoru.experiments.seqdata import load_gt
        box = next(b for t, b in load_gt(seq)[frame] if t == tid)
        fid += 1
        cv2.imwrite(str(OUT / "imgs" / f"F{fid:02d}.jpg"), crop_ctx(img, box, draw=[(box, (0, 255, 255))]))
        rows.append({"id": f"F{fid:02d}", "type": "team", "seq": seq, "image": path.name,
                     "size_px": round(float(box[3] - box[1]), 1), "player_overlap_iou": None, "near_player": None,
                     "laplacian_var": round(lap_var(img, box), 1), "green_ratio": round(green_ratio(img, box), 2),
                     "best_iou": None, "best_conf_iou50": None, "auto_category": "Takım ataması hatası"})

    df = pd.DataFrame(rows)
    # görüntüler tek tek incelenerek yazılan gerekçeler (results/failures/manual_review.csv)
    mr = OUT / "manual_review.csv"
    if mr.exists():
        m = pd.read_csv(mr)
        df = df.merge(m, on="id", how="left")
    else:
        df["manual_category"] = ""
        df["manual_reason"] = ""
    df.to_csv(OUT / "failures.csv", index=False)
    counts = {k: len(v) for k, v in cands.items()}
    from bgoru.experiments.common import match_image_center
    off_fp = sum(bool(match_image_center(b[None], test[i]["boxes"][test[i]["cls"] == 2])[0].any())
                 for i, b in cands["ball_FP"] if (test[i]["cls"] == 2).any())
    counts["ball_FP_near_gt_ball"] = int(off_fp)
    pd.DataFrame([counts]).to_csv(OUT / "failure_totals.csv", index=False)
    # temas tabakası
    ims = []
    for rrow in rows:
        im = cv2.imread(str(OUT / "imgs" / f"{rrow['id']}.jpg"))
        im = cv2.resize(im, (240, 240))
        cv2.rectangle(im, (0, 0), (240, 22), (0, 0, 0), -1)
        cv2.putText(im, f"{rrow['id']} {rrow['type']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        ims.append(im)
    cols = 7
    for s in range(0, len(ims), cols * 4):
        chunk = ims[s:s + cols * 4]
        if len(chunk) < cols:  # tek ve eksik satır: siyah dolgu yerine yalnızca mevcut kareler
            grid = np.hstack(chunk)
        else:
            while len(chunk) % cols:
                chunk.append(np.zeros_like(ims[0]))
            grid = np.vstack([np.hstack(chunk[r:r + cols]) for r in range(0, len(chunk), cols)])
        cv2.imwrite(str(OUT / f"contact_sheet_{s // (cols * 4) + 1}.jpg"), grid)
    print(f"[✓] {len(rows)} başarısız örnek; toplam aday sayıları: {counts}")
    print(df.groupby(["type", "auto_category"]).size().to_string())


if __name__ == "__main__":
    main()
