"""Deneylerde ortak kullanılan veri yükleme, tahmin önbelleği ve metrik fonksiyonları.

Tüm yapılandırmalar (COCO taban modeli dahil) AYNI kod yolu ile değerlendirilir:
aynı görüntüler, aynı NMS ayarı (IoU 0.7, en fazla 300 kutu, güven >= 0.001),
aynı eşleştirme ve aynı AP hesabı (COCO tarzı 101 noktalı enterpolasyon).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bgoru.config import DATA, EXPERIMENTS, WEIGHTS

IMG_W, IMG_H = 1920, 1080

MODELS = {
    # kimlik: (ağırlık, tür, girdi boyutu)
    "E1_coco640": ("yolov8s.pt", "coco", 640),
    "E1_coco1280": ("yolov8s.pt", "coco", 1280),
    "E2_ft640": ("bgoru_E2_ft640.pt", "bgoru", 640),
    "E3_ft1280_cp": ("bgoru_E3_ft1280_cp.pt", "bgoru", 1280),
    "A1_ft1280": ("bgoru_A1_ft1280.pt", "bgoru", 1280),
    "A2_ft640_cp": ("bgoru_A2_ft640_cp.pt", "bgoru", 640),
    "E3_ft1280_cp_s1": ("bgoru_E3_ft1280_cp_s1.pt", "bgoru", 1280),  # tohum tekrarı (seed 1)
    "A1_ft1280_s1": ("bgoru_A1_ft1280_s1.pt", "bgoru", 1280),
    "G1_ft1280_gsr": ("bgoru_G1_ft1280_gsr.pt", "bgoru", 1280),  # A1 + SoccerNet-GSR valid ek eğitim verisi
    "C1_ft1280_ep50": ("bgoru_C1_ft1280_ep50.pt", "bgoru", 1280),  # kontrol: A1 verisi, G1 ile eşit adım
    "R_hmzbo640": ("hmzbo_players_yolov8l.pt", "hmzbo", 640),
    "R_hmzbo1280": ("hmzbo_players_yolov8l.pt", "hmzbo", 1280),
}
COCO_MAP = {0: 0, 32: 2}


def load_split(split: str, root: Path = DATA / "soccernet"):
    """Dönüş: list[dict(path, seq, boxes[N,4] piksel, cls[N])]"""
    items = []
    for ip in sorted((root / "images" / split).glob("*.jpg")):
        lp = root / "labels" / split / (ip.stem + ".txt")
        rows = [list(map(float, l.split())) for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
        a = np.array(rows, np.float32).reshape(-1, 5)
        boxes = np.stack([(a[:, 1] - a[:, 3] / 2) * IMG_W, (a[:, 2] - a[:, 4] / 2) * IMG_H,
                          (a[:, 1] + a[:, 3] / 2) * IMG_W, (a[:, 2] + a[:, 4] / 2) * IMG_H], 1)
        items.append({"path": ip, "seq": ip.stem.rsplit("_", 1)[0], "boxes": boxes, "cls": a[:, 0].astype(int)})
    return items


def gpu_cap():
    """Eğitimle GPU paylaşılırken (BGORU_GPU_FRAC) bu sürecin bellek payını sınırlar."""
    import os

    import torch

    if torch.cuda.is_available() and os.environ.get("BGORU_GPU_FRAC") and not getattr(gpu_cap, "done", False):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["BGORU_GPU_FRAC"]))
        gpu_cap.done = True


def predict_split(model_id: str, split: str, items=None, transform=None, tag: str = "", batch: int | None = None,
                  cache: bool = True):
    """Tahminleri hesaplar ve experiments/preds/<model>_<split><tag>.npz içinde önbelleğe alır.
    Not: toplu (batch) boyutu yalnızca hızı etkiler, sonuçları değiştirmez."""
    import os

    import cv2
    from ultralytics import YOLO

    from bgoru.detection import use_half

    gpu_cap()
    batch = batch or int(os.environ.get("BGORU_PRED_BATCH", 8))

    out = EXPERIMENTS / "preds" / f"{model_id}_{split}{tag}.npz"
    if cache and out.exists():
        z = np.load(out, allow_pickle=True)
        return list(z["preds"])
    items = items or load_split(split)
    w, kind, imgsz = MODELS[model_id]
    model = YOLO(str(WEIGHTS / w))
    classes = [0, 32] if kind == "coco" else None
    preds = []
    for i in range(0, len(items), batch):
        imgs = [cv2.imread(str(it["path"])) for it in items[i:i + batch]]
        if transform is not None:
            imgs = [transform(im, j + i) for j, im in enumerate(imgs)]
        res = model.predict(imgs, imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, half=use_half(), verbose=False,
                            classes=classes)
        for r in res:
            b = r.boxes
            c = b.cls.cpu().numpy().astype(int)
            if kind == "coco":
                c = np.array([COCO_MAP[int(x)] for x in c], int)
            preds.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], c[:, None]], 1)
                         .astype(np.float32))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, preds=np.array(preds, dtype=object))
    return preds


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match_image(pred: np.ndarray, gt_boxes: np.ndarray, iou_thr: float):
    """Açgözlü eşleştirme (skora göre). Dönüş: tp[bool, len(pred)], gt_matched[bool]"""
    order = np.argsort(-pred[:, 4]) if len(pred) else np.zeros(0, int)
    tp = np.zeros(len(pred), bool)
    used = np.zeros(len(gt_boxes), bool)
    if len(pred) and len(gt_boxes):
        ious = box_iou(pred[order, :4], gt_boxes)
        for k, pi in enumerate(order):
            cand = np.where(~used & (ious[k] >= iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(ious[k, cand])]
                used[j] = True
                tp[pi] = True
    return tp, used


def match_image_center(pred: np.ndarray, gt_boxes: np.ndarray, factor: float = 2.0, min_px: float = 16.0):
    """Küçük nesneler için yardımcı eşleştirme: merkez uzaklığı <= factor x GT boyutu (en az min_px).
    Etiket kutusunun birkaç piksel ötelenmesine karşı IoU'dan daha az duyarlıdır."""
    order = np.argsort(-pred[:, 4]) if len(pred) else np.zeros(0, int)
    tp = np.zeros(len(pred), bool)
    used = np.zeros(len(gt_boxes), bool)
    if len(pred) and len(gt_boxes):
        pc = np.stack([(pred[order, 0] + pred[order, 2]) / 2, (pred[order, 1] + pred[order, 3]) / 2], 1)
        gc = np.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2, (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2], 1)
        gs = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], gt_boxes[:, 3] - gt_boxes[:, 1])
        tol = np.maximum(factor * gs, min_px)
        d = np.linalg.norm(pc[:, None] - gc[None], axis=2)
        for k, pi in enumerate(order):
            cand = np.where(~used & (d[k] <= tol))[0]
            if len(cand):
                j = cand[np.argmin(d[k, cand])]
                used[j] = True
                tp[pi] = True
    return tp, used


def collect(items, preds, cls: int, iou_thr, gt_cls=None, img_idx=None):
    """Bir sınıf için (skor, tp, görüntü) dizileri ve GT sayısı. iou_thr="center" -> merkez uzaklığı eşleştirmesi."""
    gt_cls = gt_cls if gt_cls is not None else [cls]
    idx = range(len(items)) if img_idx is None else img_idx
    S, T, I, n_gt = [], [], [], 0
    for i in idx:
        it, p = items[i], preds[i]
        p = p[p[:, 5] == cls]
        g = it["boxes"][np.isin(it["cls"], gt_cls)]
        n_gt += len(g)
        tp, _ = match_image_center(p, g) if iou_thr == "center" else match_image(p, g, iou_thr)
        S.append(p[:, 4]); T.append(tp); I.append(np.full(len(p), i))
    if not S:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0, int), 0
    return np.concatenate(S), np.concatenate(T), np.concatenate(I), n_gt


def ap_from(scores, tps, n_gt) -> float:
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores, kind="stable")
    tp = tps[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def ap(items, preds, cls, iou_thr=0.5, gt_cls=None, img_idx=None) -> float:
    s, t, _, n = collect(items, preds, cls, iou_thr, gt_cls, img_idx)
    return ap_from(s, t, n)


def ap_range(items, preds, cls, gt_cls=None) -> float:
    return float(np.nanmean([ap(items, preds, cls, t, gt_cls) for t in np.arange(0.5, 0.96, 0.05)]))


def prf_at(items, preds, cls, conf, iou_thr=0.5, gt_cls=None):
    s, t, _, n = collect(items, preds, cls, iou_thr, gt_cls)
    k = s >= conf
    tp = int(t[k].sum()); fp = int(k.sum() - tp)
    p = tp / max(tp + fp, 1); r = tp / max(n, 1)
    return {"P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-9), "TP": tp, "FP": fp, "FN": n - tp, "n_gt": n}


def best_conf_f1(items, preds, cls, iou_thr=0.5, gt_cls=None):
    """Doğrulama kümesinde F1'i en büyükleyen güven eşiği."""
    s, t, _, n = collect(items, preds, cls, iou_thr, gt_cls)
    if n == 0 or len(s) == 0:
        return 0.25
    o = np.argsort(-s)
    ctp = np.cumsum(t[o]); k = np.arange(1, len(o) + 1)
    f1 = 2 * ctp / (k + n)
    return float(s[o][int(np.argmax(f1))])


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
