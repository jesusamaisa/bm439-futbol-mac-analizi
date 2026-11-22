"""Uçtan uca analiz hattı: kare kare tespit -> takip -> takım -> homografi -> top ->
anlık hakimiyet/olaylar -> çizim; video sonunda profiller ve karakter kartları."""
from __future__ import annotations

import csv
import json
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from bgoru.annotate import TacticalMap, draw_ball, draw_hud, draw_player, overlay_map, text
from bgoru.ball import BallTracker
from bgoru.camera import CameraMotion
from bgoru.card import render_card, render_profile_map
from bgoru.config import BALL, FINAL_CONF_BALL, FINAL_CONF_PLAYER, FINAL_DET_WEIGHTS, PLAYER, REFEREE
from bgoru.detection import Detector
from bgoru.homography import PitchHomography
from bgoru.kinematics import PLAYER_HEIGHT_M, Kinematics
from bgoru.possession import PlayerObs, PossessionEngine
from bgoru.profiling import build_profiles
from bgoru.team import TeamAssigner
from bgoru.tracking import PlayerTracker

DEFAULT_TEAM_BGR = [(235, 235, 235), (60, 60, 220)]


@dataclass
class AnalyzerConfig:
    det_weights: str = FINAL_DET_WEIGHTS      # doğrulama top AP'sine göre seçildi (G1; bkz. config.FINAL_DETECTOR)
    det_imgsz: int = 1280
    conf_player: float = FINAL_CONF_PLAYER
    conf_referee: float = 0.30
    conf_ball: float = FINAL_CONF_BALL
    use_homography: bool = True
    kp_weights: str = "hmzbo_keypoints_yolov8m.pt"
    kp_conf: float = 0.5
    kp_every: int = 1
    kp_tol_px: float = 4.0
    team_method: str = "proposed_local"   # doğrulamada seçildi (results/team)
    team_colors_bgr: list | None = None
    team_centers: list | None = None      # detect_team_colors() çıktısı (LAB), isteğe bağlı
    team_spread: float | None = None
    team_names: list = field(default_factory=lambda: ["Takım A", "Takım B"])
    ball_fill: bool = True
    dist_thr: float = 0.45
    show_hud: bool = True
    show_map: bool = True
    show_keypoints: bool = False
    show_palette: bool = False
    min_card_minutes: float = 0.15


def lab_to_bgr(lab) -> tuple:
    l, a, b = lab
    px = np.uint8([[[np.clip(l * 255 / 100, 0, 255), np.clip(a + 128, 0, 255), np.clip(b + 128, 0, 255)]]])
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0])


def iou_max(box, others):
    if len(others) == 0:
        return 0.0
    x0 = np.maximum(box[0], others[:, 0]); y0 = np.maximum(box[1], others[:, 1])
    x1 = np.minimum(box[2], others[:, 2]); y1 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return float((inter / (a + b - inter + 1e-6)).max())


class Analyzer:
    def __init__(self, cfg: AnalyzerConfig, fps: float = 25.0):
        self.cfg, self.fps = cfg, fps
        self.det = Detector(cfg.det_weights, imgsz=cfg.det_imgsz, conf_player=cfg.conf_player,
                            conf_referee=cfg.conf_referee, conf_ball=cfg.conf_ball)
        self.hom = PitchHomography(cfg.kp_weights, conf=cfg.kp_conf, tol_px=cfg.kp_tol_px) if cfg.use_homography else None
        self.tracker = PlayerTracker(fps)
        self.team = TeamAssigner(cfg.team_method, team_colors_bgr=cfg.team_colors_bgr, centers=cfg.team_centers,
                                 spread=cfg.team_spread)
        self.ball = BallTracker(fps, fill=cfg.ball_fill)
        self.cam = CameraMotion()
        self.kin = Kinematics(fps)
        self.pos = PossessionEngine(fps, dist_thr=cfg.dist_thr)
        self.tmap = TacticalMap()
        self.players: dict[int, dict] = {}
        self.ball_trail_m: deque = deque(maxlen=int(fps * 3))
        self.team_bgr = list(cfg.team_colors_bgr) if cfg.team_colors_bgr else list(DEFAULT_TEAM_BGR)
        self._colors_set = False
        self.f = 0
        self.pitch_frames = 0
        self.last_ball_m, self.last_ball_f = None, -10 ** 9
        self.src_of: dict[int, str] = {}
        self.timing = defaultdict(float)

    # -------------------------------------------------------------------------------------------
    def _team_colors(self):
        if self._colors_set:
            return
        if self.cfg.team_colors_bgr is None and self.team.centers is not None and self.team.method.startswith("proposed") \
                and self.team.centers.shape[1] == 3 and "rgb" not in self.team.method:
            c0, c1 = (np.array(lab_to_bgr(c), float) for c in self.team.centers)
            dist = float(np.linalg.norm(c0 - c1))
            if dist < 60:  # neredeyse aynı renk -> 2. takım için ayırt edici sabit renk
                c1 = np.array([60, 60, 220], float)
            elif dist < 120:  # benzer tonlar (ör. açık mavi / lacivert): açık takımı aç, koyu takımı koyulaştır
                c0 = c0 + 0.55 * (255 - c0)
                c1 = c1 * 0.55
            self.team_bgr = [tuple(int(v) for v in c0), tuple(int(v) for v in c1)]
            self._colors_set = True

    def _player_rec(self, tid):
        if tid not in self.players:
            self.players[tid] = {"frames": 0, "team_votes": np.zeros(2), "cls_votes": np.zeros(2), "photo": None,
                                 "photo_area": 0.0, "first_f": self.f, "last_f": self.f, "h_hist": deque(maxlen=50)}
        return self.players[tid]

    def process(self, frame: np.ndarray):
        cfg, f = self.cfg, self.f
        t0 = time.perf_counter()
        det = self.det(frame)
        t1 = time.perf_counter()
        xyxy, conf, cls, tids = self.tracker.update(det)
        pl = np.where(cls == PLAYER)[0]
        rf = np.where(cls == REFEREE)[0]
        teams = np.full(len(xyxy), -1)
        feats = np.full((len(xyxy), 3), np.nan)
        if len(pl):
            t, ft, _ = self.team.assign(frame, xyxy[pl], tids[pl])
            teams[pl], feats[pl] = t, ft
            self._team_colors()
        t2 = time.perf_counter()

        has_pitch = False
        if self.hom is not None:
            if f % max(1, cfg.kp_every) == 0:
                has_pitch = self.hom.update(frame)
            else:
                has_pitch = self.hom.H is not None
        t3 = time.perf_counter()
        feet = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], 1) if len(xyxy) else np.zeros((0, 2))
        if has_pitch:
            pos = self.hom.to_pitch(feet) if len(feet) else np.zeros((0, 2))
            src = "pitch"
            self.pitch_frames += 1
        else:
            off = self.cam.update(frame, xyxy)
            pos = np.zeros((len(xyxy), 2))
            for i in range(len(xyxy)):
                rec = self._player_rec(int(tids[i]))
                rec["h_hist"].append(xyxy[i, 3] - xyxy[i, 1])
                hm = float(np.median(rec["h_hist"]))
                pos[i] = (feet[i] - off) * PLAYER_HEIGHT_M / max(hm, 1.0)
            src = "approx"
            self.kin.approx = True

        bd = det.of_class(BALL)
        bxy, bpred, _ = self.ball.update(bd.xyxy, bd.conf)
        ball_m = None
        if bxy is not None and has_pitch:
            bm = self.hom.to_pitch(bxy[None])[0]
            # fiziksel olabilirlik: saha sınırları içinde ve topun en yüksek hızından (≈45 m/s) hızlı sıçramıyor
            inside = -5 <= bm[0] <= 110 and -5 <= bm[1] <= 73
            ok_speed = True
            if self.last_ball_m is not None:
                dt = max(f - self.last_ball_f, 1) / self.fps
                ok_speed = np.linalg.norm(bm - self.last_ball_m) / dt <= 45.0 or f - self.last_ball_f > self.fps
            if inside and ok_speed:
                ball_m = bm
                self.last_ball_m, self.last_ball_f = bm, f
                self.ball_trail_m.append(ball_m)

        obs = []
        for i in pl:
            tid = int(tids[i])
            rec = self._player_rec(tid)
            rec["frames"] += 1
            rec["last_f"] = f
            if teams[i] in (0, 1):
                rec["team_votes"][teams[i]] += 1
            area = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
            if area > rec["photo_area"] and iou_max(xyxy[i], np.delete(xyxy, i, 0)) < 0.05:
                x0, y0, x1, y1 = xyxy[i].astype(int)
                pad = int(0.08 * (y1 - y0))
                rec["photo"] = frame[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad].copy()
                rec["photo_area"] = area
            if self.src_of.get(tid) not in (None, src):
                self.kin.hist[tid].clear()
                self.kin.smooth.pop(tid, None)
            self.src_of[tid] = src
            self.kin.update(f, tid, pos[i])
            obs.append(PlayerObs(tid, int(teams[i]), xyxy[i], pos[i]))
        snap = self.pos.update(f, obs, bxy, ball_m, metric=True, pitch=has_pitch)
        t4 = time.perf_counter()

        out = frame.copy()
        for i in rf:
            draw_player(out, xyxy[i], (0, 215, 255), "HAKEM")
        for i in pl:
            t = teams[i]
            col = self.team_bgr[t] if t in (0, 1) else (180, 180, 180)
            pal = None
            if cfg.show_palette and not np.isnan(feats[i]).any() and self.team.method.startswith("proposed")                     and "rgb" not in self.team.method:
                pal = lab_to_bgr(feats[i])
            draw_player(out, xyxy[i], col, f"{int(tids[i])}", possessor=(snap["possessor"] == int(tids[i])), palette=pal)
        if bxy is not None:
            draw_ball(out, bxy, bpred)
        if cfg.show_keypoints and self.hom is not None:
            for k, p in self.hom.last_kps.items():
                cv2.circle(out, tuple(int(v) for v in p), 6, (255, 0, 255), -1)
        if cfg.show_map and has_pitch:
            prs = [i for i in pl]
            pidx = next((j for j, i in enumerate(prs) if int(tids[i]) == snap["possessor"]), None)
            m = self.tmap.render(pos[prs] if prs else None, teams[prs], self.team_bgr,
                                 pos[rf] if len(rf) else None, list(self.ball_trail_m), pidx)
            overlay_map(out, m)
        elif cfg.show_map:
            text(out, "Taktik harita: saha noktalari yetersiz", (out.shape[1] - 520, out.shape[0] - 20), 0.6)
        if cfg.show_hud:
            draw_hud(out, snap, cfg.team_names, self.team_bgr)
        t5 = time.perf_counter()
        for k, v in (("detect", t1 - t0), ("track_team", t2 - t1), ("homography", t3 - t2), ("logic", t4 - t3),
                     ("draw", t5 - t4)):
            self.timing[k] += v
        self.f += 1
        return out, snap

    # -------------------------------------------------------------------------------------------
    def player_summaries(self) -> dict[int, dict]:
        res = {}
        team_poss = np.zeros(2)
        for tid, rec in self.players.items():
            if rec["frames"] / self.fps / 60 < self.cfg.min_card_minutes:
                continue
            team = int(np.argmax(rec["team_votes"])) if rec["team_votes"].sum() else -1
            ev = {k: int(v) for k, v in self.pos.stats.get(tid, {}).items() if k != "poss_frames"}
            pf = float(self.pos.stats.get(tid, {}).get("poss_frames", 0))
            if team in (0, 1):
                team_poss[team] += pf
            res[tid] = {"tid": tid, "display_id": f"#{tid}", "name": f"Oyuncu {tid}", "team": team,
                        "frames": rec["frames"], "events": ev, "poss_frames": pf, "poss_s": round(pf / self.fps, 1),
                        "kin": self.kin.summary(tid)}
        for p in res.values():
            p["team_poss_share"] = round(p["poss_frames"] / team_poss[p["team"]] * 100, 1) \
                if p["team"] in (0, 1) and team_poss[p["team"]] > 0 else None
        return res

    def finalize(self, out_dir: Path, segment_name: str = "", names: dict[int, str] | None = None,
                 make_cards: bool = True) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        snap = self.pos.finalize()
        players = self.player_summaries()
        for tid, nm in (names or {}).items():
            if tid in players:
                players[tid]["name"] = nm
        profiles = build_profiles(players, self.fps)
        approx = self.pitch_frames < 0.5 * max(self.f, 1)
        cards = {}
        if make_cards:
            for old in (out_dir / "cards").glob("kart_*.png"):  # aynı klasördeki önceki çalıştırmaların kartları
                old.unlink()
            seg = f"{segment_name}  ·  {snap['elapsed_s'] / 60:.1f} dk kesit"
            for tid, pr in sorted(profiles.items(), key=lambda kv: -(kv[1]["overall"] or 0)):
                p = players[tid]
                tb = self.team_bgr[p["team"]] if p["team"] in (0, 1) else (180, 180, 180)
                tn = self.cfg.team_names[p["team"]] if p["team"] in (0, 1) else "Takım ?"
                cards[tid] = str(render_card(p, pr, out_dir / "cards" / f"kart_{tid}.png", tn, tb,
                                             self.players[tid]["photo"], seg, approx_kin=approx))
            if profiles:
                render_profile_map(profiles, players, self.cfg.team_names, self.team_bgr, out_dir / "profil_haritasi.png")
        with open(out_dir / "events.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "time_s", "type", "name", "player", "team", "to"])
            for e in self.pos.events:
                w.writerow([e["f"], e["t"], e["type"], e["name"], e["tid"], e["team"], e.get("to", "")])
        with open(out_dir / "possession_timeline.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "control_team"])
            w.writerows(self.pos.timeline)
        n = max(self.f, 1)
        summary = {
            "segment": segment_name, "frames": self.f, "fps": self.fps, "team_names": self.cfg.team_names,
            "team_colors_bgr": [list(map(int, c)) for c in self.team_bgr], "final": snap,
            "pitch_coverage": round(self.pitch_frames / n, 3), "kinematics_approx": approx,
            "timing_ms_per_frame": {k: round(v / n * 1000, 2) for k, v in self.timing.items()},
            "players": {int(k): v for k, v in players.items()}, "profiles": {int(k): v for k, v in profiles.items()},
            "cards": cards, "config": asdict(self.cfg),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                              encoding="utf-8")
        return summary


def detect_team_colors(video, cfg: AnalyzerConfig | None = None, n_frames: int = 12, step: int = 5,
                       start_s: float = 0.0) -> dict:
    """Videonun ilk karelerinden takım renk merkezlerini öğrenir (kullanıcı takım adlarını
    renklere göre eşleyebilsin diye). Dönüş: centers (LAB), spread, colors_bgr, örnek kesitler."""
    cfg = cfg or AnalyzerConfig()
    det = Detector(cfg.det_weights, imgsz=cfg.det_imgsz, conf_player=cfg.conf_player)
    ta = TeamAssigner(cfg.team_method if cfg.team_method.startswith("proposed") else "proposed_local", vote=False,
                      fit_frames=n_frames, balanced=True)
    cap = cv2.VideoCapture(str(video))
    if start_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    examples: dict[int, list] = {0: [], 1: []}
    frames, k = [], 0
    while len(frames) < n_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if k % step == 0:
            frames.append(fr)
        k += 1
    cap.release()
    for fr in frames:
        d = det(fr).of_class(PLAYER)
        ta.assign(fr, d.xyxy)
    if ta.centers is None:
        return {}
    fr = frames[-1]
    d = det(fr).of_class(PLAYER)
    team, _, outl = ta.assign(fr, d.xyxy)
    for b, t, o in zip(d.xyxy, team, outl):
        if t in (0, 1) and not o and len(examples[t]) < 6:
            x0, y0, x1, y1 = b.astype(int)
            examples[t].append(fr[max(0, y0):y1, max(0, x0):x1].copy())
    return {"centers": ta.centers.tolist(), "spread": ta.spread,
            "colors_bgr": [lab_to_bgr(c) for c in ta.centers], "examples": examples}


def to_h264(src: Path, dst: Path) -> Path:
    """Tarayıcıda oynatılabilir H.264 MP4'e dönüştürür (imageio-ffmpeg)."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "veryfast", "-crf", "23", str(dst)], check=True)
        src.unlink(missing_ok=True)
        return dst
    except Exception:  # noqa: BLE001
        return src


def analyze_video(video: str | Path, out_dir: str | Path, cfg: AnalyzerConfig | None = None,
                  max_frames: int | None = None, start_s: float = 0.0, write_video: bool = True,
                  callback=None, out_width: int = 1280) -> dict:
    cfg = cfg or AnalyzerConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    an = Analyzer(cfg, fps)
    writer, tmp = None, out_dir / "_annotated_tmp.mp4"
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and an.f >= max_frames):
            break
        out, snap = an.process(frame)
        if write_video:
            if writer is None:
                h = int(out.shape[0] * out_width / out.shape[1])
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, h))
            writer.write(cv2.resize(out, (out_width, int(out.shape[0] * out_width / out.shape[1]))))
        if callback is not None and callback(an, out, snap) is False:
            break
    cap.release()
    if writer is not None:
        writer.release()
    wall = time.time() - t0
    summary = an.finalize(out_dir, Path(video).stem)
    summary["wall_fps"] = round(an.f / max(wall, 1e-6), 2)
    if writer is not None:
        summary["video"] = str(to_h264(tmp, out_dir / "analiz.mp4"))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                          encoding="utf-8")
    return summary
