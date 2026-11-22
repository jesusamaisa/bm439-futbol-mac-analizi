"""RQ2 — Takım ataması karşılaştırması ve ablation.

Tespit hatasının etkisini ayırmak için GT oyuncu kutuları ve GT iz kimlikleri kullanılır.
Her yöntem sekans başına sıfırdan başlar; ilk ``warmup`` kare (yöntemlerin renk
merkezlerini öğrendiği ısınma) TÜM yöntemler için değerlendirme dışıdır.
Kümelemede takım numarası keyfi olduğundan sekans başına en iyi etiket permütasyonu alınır.

Kümeler:
  * dense  : 10 test sekansı, 12,5 fps ardışık kare (aşağı akış senaryosu)
  * sparse : 49 test sekansının tamamı, 1 fps (genelleme)

Kullanım:  python -m bgoru.experiments.eval_team
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd

from bgoru.config import RESULTS
from bgoru.experiments.common import save_json
from bgoru.experiments.registry import log_experiment
from bgoru.experiments.seqdata import dense_frames, dense_sequences, frame_objects, load_gt, load_meta, sparse_test_frames
from bgoru.team import TeamAssigner

CONFIGS = {
    "T1_baseline": dict(method="baseline", vote=False),
    "T2_alternative": dict(method="alternative", vote=False),
    "T3_proposed_v1": dict(method="proposed", vote=True, balanced=False),
    "T4_proposed": dict(method="proposed", vote=True, balanced=True),
    "T4a_no_vote": dict(method="proposed", vote=False, balanced=True),
    "T4b_no_grass_mask": dict(method="proposed_nomask", vote=True, balanced=True),
    "T4c_rgb_not_lab": dict(method="proposed_rgb", vote=True, balanced=True),
    "T5_local_mask": dict(method="proposed_local", vote=True, balanced=True),
    "T5a_no_vote": dict(method="proposed_local", vote=False, balanced=True),
    "T5b_no_balance": dict(method="proposed_local", vote=True, balanced=False),
    "T5c_rgb_not_lab": dict(method="proposed_local_rgb", vote=True, balanced=True),
    "T5d_hsv_mask": dict(method="proposed", vote=True, balanced=True),
    "T5e_no_mask": dict(method="proposed_nomask", vote=True, balanced=True),
}
LABEL = {"T1_baseline": "Temel: tüm kutu ort. RGB + K-Means", "T2_alternative": "Alternatif: üst yarı piksel K-Means",
         "T3_proposed_v1": "Önerilen v1: gövde + çim maskesi + LAB + iz oylaması (ilk sürüm)",
         "T4_proposed": "Önerilen v2: v1 + dengeli kümeleme (>=6 iz, küçük küme ayıklama)",
         "T4a_no_vote": "Ablation: oylama yok", "T4b_no_grass_mask": "Ablation: çim maskesi yok",
         "T4c_rgb_not_lab": "Ablation: LAB yerine RGB",
         "T5_local_mask": "Önerilen (son): gövde + YEREL saha maskesi + LAB + dengeli kümeleme + iz oylaması",
         "T5a_no_vote": "Ablation: iz oylaması yok", "T5b_no_balance": "Ablation: dengeli kümeleme yok",
         "T5c_rgb_not_lab": "Ablation: LAB yerine RGB", "T5d_hsv_mask": "Ablation: yerel maske yerine sabit HSV maskesi",
         "T5e_no_mask": "Ablation: saha maskesi yok"}


def run_sequence(frames, meta, rows, cfg, warmup=10):
    ta = TeamAssigner(cfg["method"], vote=cfg["vote"], balanced=cfg.get("balanced", True))
    recs = []  # (frame_idx, tid, gt_team, pred_team, is_gk)
    for k, (f, path) in enumerate(frames):
        players, _, _ = frame_objects(meta, rows, f)
        if not players:
            continue
        img = cv2.imread(str(path))
        boxes = np.stack([p[1] for p in players])
        tids = np.array([p[0] for p in players])
        pred, _, _ = ta.assign(img, boxes, tids)
        if k >= warmup:
            for (tid, _, gt, gk), pr in zip(players, pred):
                recs.append((f, tid, gt, int(pr), gk))
    return recs


def seq_accuracy(recs, include_gk=False):
    r = [x for x in recs if include_gk or not x[4]]
    if not r:
        return np.nan, 0
    gt = np.array([x[2] for x in r]); pr = np.array([x[3] for x in r])
    acc = max(np.mean(pr == gt), np.mean(pr == 1 - gt))  # en iyi permütasyon (-1 = yanlış)
    return float(acc), len(r)


def val_frames() -> dict:
    from bgoru.experiments.seqdata import SN
    out = {}
    for p in sorted((SN / "images" / "val").glob("*.jpg")):
        seq, f = p.stem.rsplit("_", 1)
        out.setdefault(seq, []).append((int(f), p))
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["val", "dense", "sparse"])
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()
    out = RESULTS / "team"
    out.mkdir(parents=True, exist_ok=True)
    all_sets = {"val": val_frames, "dense": lambda: {s: dense_frames(s) for s in dense_sequences()},
                "sparse": sparse_test_frames}
    old_rows = pd.read_csv(out / "team_results.csv") if (out / "team_results.csv").exists() else pd.DataFrame()
    old_det = pd.read_csv(out / "team_per_sequence.csv") if (out / "team_per_sequence.csv").exists() else pd.DataFrame()
    rows, details = [], []
    for set_name in args.sets:
        seqs = all_sets[set_name]()
        if not seqs:
            continue
        for cid in args.configs:
            cfg = CONFIGS[cid]
            t0 = time.time()
            accs, accs_gk, n_tot = [], [], 0
            for seq, frames in seqs.items():
                meta, gt = load_meta(seq), load_gt(seq)
                recs = run_sequence(frames, meta, gt, cfg, warmup=10)
                a, n = seq_accuracy(recs)
                a2, _ = seq_accuracy(recs, include_gk=True)
                accs.append(a); accs_gk.append(a2); n_tot += n
                details.append({"set": set_name, "config": cid, "seq": seq, "acc": a, "acc_with_gk": a2, "n": n})
                if set_name == "dense" and cid == "T5_local_mask":
                    pd.DataFrame(recs, columns=["frame", "tid", "gt", "pred", "gk"]).to_csv(out / f"recs_{seq}.csv",
                                                                                         index=False)
            dur = time.time() - t0
            row = {"set": set_name, "config": cid, "label": LABEL[cid], "acc_mean": float(np.nanmean(accs)),
                   "acc_std": float(np.nanstd(accs)), "acc_min": float(np.nanmin(accs)),
                   "acc_with_gk_mean": float(np.nanmean(accs_gk)), "n_seq": len(accs), "n_boxes": n_tot,
                   "ms_per_box": dur / max(n_tot, 1) * 1000}
            rows.append(row)
            print(f"[{set_name}] {cid:18s} acc={row['acc_mean']:.4f}±{row['acc_std']:.4f} (min {row['acc_min']:.3f}) "
                  f"gk-dahil={row['acc_with_gk_mean']:.4f} n={n_tot} {dur:.0f}s", flush=True)
            log_experiment(f"{cid}_{set_name}", "eval_team", cfg["method"], cfg,
                           f"{set_name} ({len(accs)} sekans), GT kutular",
                           {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                            if k not in ("set", "config", "label")}, dur, LABEL[cid])
    new = pd.DataFrame(rows)
    idx = set(zip(new["set"], new["config"])) if len(new) else set()

    def keep(df):
        if not len(df):
            return df
        return df[[(a, b) not in idx for a, b in zip(df["set"], df["config"])]]

    res = pd.concat([keep(old_rows), new], ignore_index=True)
    det = pd.concat([keep(old_det), pd.DataFrame(details)], ignore_index=True)
    res.to_csv(out / "team_results.csv", index=False)
    det.to_csv(out / "team_per_sequence.csv", index=False)
    save_json(res.to_dict("records"), out / "team_results.json")


if __name__ == "__main__":
    main()
