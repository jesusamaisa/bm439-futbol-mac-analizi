"""RQ3 — Top hakimiyeti yüzdesi ve pas ölçümünün doğruluğu (aşağı akış).

Referans (vekil GT): Aynı PossessionEngine kuralları, GT oyuncu kutuları + GT iz kimlikleri +
GT takım etiketleri + GT top kutusu üzerinde çalıştırılır. Böylece ölçülen hata, yalnızca
algılama/takip/takım ataması hatalarının hakimiyet ve pas ölçümüne yansımasıdır.
(Not: Bu referans, insan tarafından etiketlenmiş olay verisi değildir; raporda belirtilir.)

Varyantlar (aynı 10 yoğun test sekansı, 12,5 fps):
  V0 E1 COCO + önerilen takım + top doldurma
  V1 E2 (640)  + önerilen takım + top doldurma
  V2 E3 (önerilen tam sistem)
  V3 E3 + temel (K-Means) takım ataması          -> takım bileşeni ablation'ı
  V4 E3 + top doldurma YOK                        -> zamansal bileşen ablation'ı
  V5 E3 + histerezis YOK (onay = 1 kare)          -> karar kuralı ablation'ı
  V6 A1 (yalnızca SoccerNet-Tracking ile eğitilmiş 1280 px dedektör)
  V7 G1 = son sistem (A1 + SoccerNet-GSR ek eğitim verisi; config.FINAL_DETECTOR)

Metrikler: hakimiyet % mutlak hatası (yüzde puanı, ↓), kare bazlı kontrol uyumu (↑),
pas olayı F1 (±1 sn, aynı takım, ↑), pas sayısı mutlak hatası (↓), uçtan uca takım doğruluğu (↑).
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd

from bgoru.ball import BallTracker
from bgoru.config import RESULTS, WEIGHTS
from bgoru.experiments.common import MODELS, box_iou, save_json
from bgoru.experiments.registry import log_experiment
from bgoru.experiments.seqdata import dense_frames, dense_sequences, frame_objects, load_gt, load_meta
from bgoru.possession import PlayerObs, PossessionEngine
from bgoru.team import TeamAssigner
from bgoru.tracking import PlayerTracker

FPS = 12.5
COCO_MAP = {0: 0, 32: 2}
VARIANTS = {
    "V0_E1": dict(model="E1_coco1280", team="proposed_local", fill=True, confirm_s=0.12),
    "V1_E2": dict(model="E2_ft640", team="proposed_local", fill=True, confirm_s=0.12),
    "V2_E3_full": dict(model="E3_ft1280_cp", team="proposed_local", fill=True, confirm_s=0.12),
    "V3_E3_team_baseline": dict(model="E3_ft1280_cp", team="baseline", fill=True, confirm_s=0.12),
    "V4_E3_no_ballfill": dict(model="E3_ft1280_cp", team="proposed_local", fill=False, confirm_s=0.12),
    "V5_E3_no_hysteresis": dict(model="E3_ft1280_cp", team="proposed_local", fill=True, confirm_s=0.0),
    "V6_A1_final": dict(model="A1_ft1280", team="proposed_local", fill=True, confirm_s=0.12),
    "V7_G1_final": dict(model="G1_ft1280_gsr", team="proposed_local", fill=True, confirm_s=0.12),
}


def load_thresholds():
    p = RESULTS / "detection" / "detection_results.csv"
    thr = {}
    if p.exists():
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            thr[r["model_id"]] = {"ball": float(r["ball_conf_thr"]), "player": float(r["player_conf_thr"])}
    return thr


def cached_detections(model_id: str, seq: str, frames):
    from ultralytics import YOLO

    from bgoru.detection import use_half
    from bgoru.experiments.common import gpu_cap

    gpu_cap()

    out = RESULTS.parent / "experiments" / "preds" / f"dense_{model_id}_{seq}.npz"
    if out.exists():
        return list(np.load(out, allow_pickle=True)["preds"])
    w, kind, imgsz = MODELS[model_id]
    model = YOLO(str(WEIGHTS / w))
    preds = []
    bs = 4
    for i in range(0, len(frames), bs):
        imgs = [cv2.imread(str(p)) for _, p in frames[i:i + bs]]
        res = model.predict(imgs, imgsz=imgsz, conf=0.05, iou=0.7, half=use_half(), verbose=False,
                            classes=[0, 32] if kind == "coco" else None)
        for r in res:
            c = r.boxes.cls.cpu().numpy().astype(int)
            if kind == "coco":
                c = np.array([COCO_MAP[int(x)] for x in c], int)
            preds.append(np.concatenate([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None], c[:, None]], 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, preds=np.array(preds, dtype=object))
    return preds


def run_reference(frames, meta, rows):
    eng = PossessionEngine(FPS)
    for k, (f, _) in enumerate(frames):
        players, _, ball = frame_objects(meta, rows, f)
        obs = [PlayerObs(tid, team, box, None) for tid, box, team, _ in players]
        bc = None if ball is None else np.array([(ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2])
        eng.update(k, obs, bc)
    eng.finalize()
    return eng


def run_predicted(frames, preds, cfg, thr, meta, rows):
    eng = PossessionEngine(FPS, confirm_s=cfg["confirm_s"]) if cfg["confirm_s"] > 0 else PossessionEngine(FPS, confirm_s=1e-6)
    tracker = PlayerTracker(FPS)
    ta = TeamAssigner(cfg["team"], vote=cfg["team"].startswith("proposed"))
    bt = BallTracker(FPS, fill=cfg["fill"])
    pairs = np.zeros((2, 2))  # (pred team, gt team) eşleşme sayımı
    team_hits = []
    from bgoru.detection import Detections

    for k, ((f, path), p) in enumerate(zip(frames, preds)):
        p = np.asarray(p, np.float32).reshape(-1, 6)
        keep = ((p[:, 5] == 2) & (p[:, 4] >= thr["ball"])) | ((p[:, 5] != 2) & (p[:, 4] >= thr["player"]))
        p = p[keep]
        det = Detections(p[:, :4], p[:, 4], p[:, 5].astype(int))
        xyxy, conf, cls, tids = tracker.update(det)
        pl = np.where(cls == 0)[0]
        teams = np.full(len(xyxy), -1)
        if len(pl):
            img = cv2.imread(str(path))
            t, _, _ = ta.assign(img, xyxy[pl], tids[pl])
            teams[pl] = t
        b = det.of_class(2)
        bxy, _, _ = bt.update(b.xyxy, b.conf)
        obs = [PlayerObs(int(tids[i]), int(teams[i]), xyxy[i], None) for i in pl]
        eng.update(k, obs, bxy)
        gplayers, _, _ = frame_objects(meta, rows, f)
        if len(pl) and gplayers:
            g = np.stack([x[1] for x in gplayers]); gt_t = np.array([x[2] for x in gplayers])
            iou = box_iou(xyxy[pl], g)
            for a, i in enumerate(pl):
                j = int(np.argmax(iou[a]))
                if iou[a, j] >= 0.5 and teams[i] in (0, 1):
                    pairs[teams[i], gt_t[j]] += 1
                    team_hits.append((teams[i], gt_t[j]))
    eng.finalize()
    swap = pairs[0, 1] + pairs[1, 0] > pairs[0, 0] + pairs[1, 1]
    th = np.array(team_hits).reshape(-1, 2)
    team_acc = float(np.mean((1 - th[:, 0] if swap else th[:, 0]) == th[:, 1])) if len(th) else np.nan
    return eng, swap, team_acc


def match_events(ev_p, ev_g, tol_s=1.0):
    used = set()
    tp = 0
    for e in ev_p:
        best = None
        for j, g in enumerate(ev_g):
            if j in used or g["team"] != e["team"]:
                continue
            d = abs(g["t"] - e["t"])
            if d <= tol_s and (best is None or d < best[0]):
                best = (d, j)
        if best is not None:
            used.add(best[1]); tp += 1
    p = tp / max(len(ev_p), 1); r = tp / max(len(ev_g), 1)
    return {"P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-9) if (ev_p or ev_g) else 1.0}


def compare(ref: PossessionEngine, pred: PossessionEngine, swap: bool):
    def mp(t):
        return (1 - t) if swap and t in (0, 1) else t
    rs, ps = ref.snapshot(), pred.snapshot()
    gt_left = rs["possession_pct"][0]
    pr_left = ps["possession_pct"][1 if swap else 0]
    rt = dict(ref.timeline); pt = {f: mp(t) for f, t in pred.timeline}
    common = [f for f in rt if f in pt]
    agree = float(np.mean([rt[f] == pt[f] for f in common])) if common else 0.0
    cover = len(common) / max(len(rt), 1)
    gp = [e for e in ref.events if e["type"] == "PASS"]
    pp = [{**e, "team": mp(e["team"])} for e in pred.events if e["type"] == "PASS"]
    m = match_events(pp, gp)
    return {"poss_err_pp": abs(pr_left - gt_left), "gt_left_pct": gt_left, "pred_left_pct": pr_left,
            "control_agreement": agree * cover, "control_coverage": cover, "pass_gt": len(gp), "pass_pred": len(pp),
            "pass_count_err": abs(len(pp) - len(gp)), "pass_P": m["P"], "pass_R": m["R"], "pass_F1": m["F1"]}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), help="yalnızca bu varyantları çalıştır (sonuçlar birleştirilir)")
    args = ap.parse_args()
    out = RESULTS / "possession"
    out.mkdir(parents=True, exist_ok=True)
    seqs = dense_sequences()
    thr_all = load_thresholds()
    rows, ref_cache = [], {}
    for vid, cfg in ((v, VARIANTS[v]) for v in args.variants):
        t0 = time.time()
        thr = thr_all.get(cfg["model"], {"ball": 0.15, "player": 0.3})
        per = []
        for seq in seqs:
            meta, gt = load_meta(seq), load_gt(seq)
            frames = dense_frames(seq)
            if seq not in ref_cache:
                ref_cache[seq] = run_reference(frames, meta, gt)
            preds = cached_detections(cfg["model"], seq, frames)
            eng, swap, tacc = run_predicted(frames, preds, cfg, thr, meta, gt)
            r = compare(ref_cache[seq], eng, swap)
            r.update({"variant": vid, "seq": seq, "team_acc_e2e": tacc})
            per.append(r)
        df = pd.DataFrame(per)
        agg = {"variant": vid, **{f"{k}_mean": float(df[k].mean()) for k in
                                  ("poss_err_pp", "control_agreement", "pass_F1", "pass_P", "pass_R", "pass_count_err",
                                   "team_acc_e2e")},
               "poss_err_pp_std": float(df["poss_err_pp"].std()), "pass_F1_std": float(df["pass_F1"].std()),
               "passes_gt_total": int(df["pass_gt"].sum()), "passes_pred_total": int(df["pass_pred"].sum()),
               "n_seq": len(df)}
        rows.append(agg)
        df.to_csv(out / f"per_seq_{vid}.csv", index=False)
        dur = time.time() - t0
        print(f"[✓] {vid}: hakimiyet hatası={agg['poss_err_pp_mean']:.2f} pp, kontrol uyumu={agg['control_agreement_mean']:.3f},"
              f" pas F1={agg['pass_F1_mean']:.3f}, takım={agg['team_acc_e2e_mean']:.3f} ({dur:.0f}s)", flush=True)
        log_experiment(vid, "eval_possession", cfg["model"], {**cfg, "thr": thr},
                       f"test dense ({len(seqs)} sekans, 12.5 fps), vekil GT",
                       {k: (round(v, 4) if isinstance(v, float) else v) for k, v in agg.items() if k != "variant"}, dur)
    new = pd.DataFrame(rows)
    prev = out / "possession_results.csv"
    if prev.exists() and set(args.variants) != set(VARIANTS):
        old_df = pd.read_csv(prev)
        new = pd.concat([old_df[~old_df["variant"].isin(new["variant"])], new], ignore_index=True)
        new["_o"] = new["variant"].map({v: i for i, v in enumerate(VARIANTS)})
        new = new.sort_values("_o").drop(columns="_o")
    new.to_csv(prev, index=False)
    ref_summary = {s: {"possession_pct": e.snapshot()["possession_pct"], "passes": e.snapshot()["passes"],
                       "events": len(e.events)} for s, e in ref_cache.items()}
    save_json(ref_summary, out / "reference_summary.json")


if __name__ == "__main__":
    main()
