"""Anlık top hakimiyeti ve olay (pas, araya girme, top kapma, top kazanma, top taşıma,
çalım, şut, orta, kilit pas) tespiti.

Top sahipliği kuralı
--------------------
Topun merkezi ile oyuncu kutusunun alt üçte biri (ayak bölgesi) arasındaki uzaklık,
oyuncunun kutu yüksekliğine bölünerek ölçeklenir (perspektiften bağımsız):

    d = mesafe(top, ayak_bölgesi) / kutu_yüksekliği

``d < dist_thr`` olan en yakın oyuncu adaydır. Aday ``confirm_s`` süresince değişmezse top
sahibi olur (histerezis); aksi takdirde önceki sahip korunur. Takım kontrolü, son top
sahibinin takımındadır — top hakimiyeti yüzdesi bu kontrol karelerinden hesaplanır.

Olay kuralları (bir sahiplik dönemi A -> B'ye geçtiğinde)
--------------------------------------------------------
* aynı takım, A≠B, boşluk ≤ max_pass_s            -> PAS (başarılı)
* farklı takım, boşluk ≤ tackle_gap_s, oyuncular yakın -> TOP KAPMA (B) / İKİLİ KAYIP (A)
* farklı takım, boşluk ≥ pickup_gap_s             -> TOP KAZANMA (B) / TOP KAYBI (A)
* farklı takım, diğer                             -> ARAYA GİRME (B) / BAŞARISIZ PAS (A)
* dönem içinde ≥ carry_m taşınan top             -> TOP TAŞIMA; rakip yakınken korunduysa ÇALIM
* saha koordinatı varsa: kaleye ≤35 m'den bırakılıp ≥12 m/s ile kale ağzına giden top -> ŞUT; kanattan ceza sahasına -> ORTA;
  ceza sahasında karşılanan / şutla biten pas      -> KİLİT PAS
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

EVENT_TR = {
    "PASS": "Pas", "PASS_FAIL": "Başarısız pas", "INTERCEPT": "Araya girme", "TACKLE": "Top kapma",
    "DUEL_LOST": "İkili kayıp", "PICKUP": "Top kazanma", "LOST_BALL": "Top kaybı", "CARRY": "Top taşıma",
    "DRIBBLE": "Çalım", "SHOT": "Şut", "CROSS": "Orta", "KEY_PASS": "Kilit pas",
}


@dataclass
class PlayerObs:
    tid: int
    team: int
    box: np.ndarray            # xyxy (görüntü pikseli)
    pos_m: np.ndarray | None   # metre (saha veya yaklaşık)


def ball_player_dist(ball: np.ndarray, box: np.ndarray) -> float:
    x0, y0, x1, y1 = box
    h = max(y1 - y0, 1.0)
    ry0 = y1 - h / 3.0
    bx, by = ball
    dx = max(x0 - bx, 0.0, bx - x1)
    dy = max(ry0 - by, 0.0, by - y1)
    return float(np.hypot(dx, dy) / h)


def feet_dist(a: np.ndarray, b: np.ndarray) -> float:
    fa = np.array([(a[0] + a[2]) / 2, a[3]])
    fb = np.array([(b[0] + b[2]) / 2, b[3]])
    h = max((a[3] - a[1] + b[3] - b[1]) / 2, 1.0)
    return float(np.linalg.norm(fa - fb) / h)


class PossessionEngine:
    def __init__(self, fps: float = 25.0, dist_thr: float = 0.45, confirm_s: float = 0.12,
                 window_s: float = 60.0, max_pass_s: float = 4.0, tackle_gap_s: float = 0.6,
                 pickup_gap_s: float = 2.0, carry_m: float = 5.0, shot_watch_s: float = 2.5):
        self.fps = fps
        self.dist_thr = dist_thr
        self.confirm = max(1, int(round(fps * confirm_s)))
        self.window = int(fps * window_s)
        self.max_pass = fps * max_pass_s
        self.tackle_gap = fps * tackle_gap_s
        self.pickup_gap = fps * pickup_gap_s
        self.carry_m = carry_m
        self.shot_watch = int(fps * shot_watch_s)

        self.possessor: dict | None = None
        self.cand, self.cand_n = None, 0
        self.team_frames = np.zeros(2)
        self.win_q: deque = deque()
        self.events: list[dict] = []
        self.pass_q: deque = deque()
        self.stats: dict[int, dict] = defaultdict(lambda: defaultdict(float))
        self.first_f: int | None = None
        self.last_f = 0
        self.team_x = np.zeros(2)
        self.team_xn = np.zeros(2)
        self.release: dict | None = None
        self.timeline: list[tuple[int, int]] = []  # (kare, kontrol eden takım)
        self.prev_ball: tuple[int, np.ndarray] | None = None

    # ------------------------------------------------------------------ yardımcılar
    def _emit(self, f, etype, tid, team, **kw):
        ev = {"f": int(f), "t": round(f / self.fps, 2), "type": etype, "name": EVENT_TR[etype],
              "tid": int(tid), "team": int(team), **kw}
        self.events.append(ev)
        self.stats[int(tid)][etype] += 1
        return ev

    def attacks_right(self, team: int) -> bool | None:
        if self.team_xn.min() < 25:
            return None
        mx = self.team_x / self.team_xn
        return bool(mx[team] < mx[1 - team])

    def _in_opp_box(self, team, pos):
        ar = self.attacks_right(team)
        if ar is None or pos is None:
            return False
        x, y = pos
        return (x > 105 - 16.5 if ar else x < 16.5) and 13.84 <= y <= 54.16

    # ------------------------------------------------------------------ dönem işlemleri
    def _close_spell(self, sp):
        if sp["start_m"] is not None and sp["last_m"] is not None and sp["metric"]:
            moved = float(np.linalg.norm(sp["last_m"] - sp["start_m"]))
            if moved >= self.carry_m:
                self._emit(sp["last_f"], "CARRY", sp["tid"], sp["team"], dist_m=round(moved, 1))
                if sp["opp_close_f"] is not None and sp["last_f"] - sp["opp_close_f"] >= 0.5 * self.fps:
                    self._emit(sp["last_f"], "DRIBBLE", sp["tid"], sp["team"])

    def _new_spell(self, f, p: PlayerObs, metric: bool, pitch: bool):
        prev = self.possessor
        start_f = f - self.confirm + 1
        new = {"tid": int(p.tid), "team": int(p.team), "start_f": start_f, "last_f": f, "box": p.box.copy(),
               "start_m": None if p.pos_m is None else p.pos_m.copy(),
               "last_m": None if p.pos_m is None else p.pos_m.copy(), "metric": metric, "pitch": pitch,
               "opp_close_f": None, "from_pass": None}
        if prev is not None:
            self._close_spell(prev)
            gap = start_f - prev["last_f"]
            if prev["team"] == new["team"] and prev["tid"] != new["tid"]:
                if gap <= self.max_pass:
                    ev = self._emit(start_f, "PASS", prev["tid"], prev["team"], to=new["tid"], success=True)
                    self.pass_q.append((start_f, prev["team"]))
                    new["from_pass"] = ev
                    if pitch and self._in_opp_box(prev["team"], new["start_m"]):
                        self._emit(start_f, "KEY_PASS", prev["tid"], prev["team"], to=new["tid"])
                        ev["key"] = True
                    self._maybe_cross(prev, new, start_f)
            elif prev["team"] != new["team"]:
                close = feet_dist(prev["box"], new["box"]) < 1.5
                if gap <= self.tackle_gap and close:
                    self._emit(start_f, "TACKLE", new["tid"], new["team"], vs=prev["tid"])
                    self._emit(start_f, "DUEL_LOST", prev["tid"], prev["team"], vs=new["tid"])
                elif gap >= self.pickup_gap:
                    self._emit(start_f, "PICKUP", new["tid"], new["team"])
                    self._emit(start_f, "LOST_BALL", prev["tid"], prev["team"])
                else:
                    self._emit(start_f, "INTERCEPT", new["tid"], new["team"], vs=prev["tid"])
                    self._emit(start_f, "PASS_FAIL", prev["tid"], prev["team"], to=new["tid"])
                    self._maybe_cross(prev, new, start_f)
        self.possessor = new
        self.release = None

    def _maybe_cross(self, prev, new, f):
        if not (prev["pitch"] and prev["last_m"] is not None and new["start_m"] is not None):
            return
        ar = self.attacks_right(prev["team"])
        if ar is None:
            return
        x, y = prev["last_m"]
        wide = y < 13.84 or y > 54.16
        final_third = x > 70 if ar else x < 35
        if wide and final_third and self._in_opp_box(prev["team"], new["start_m"]):
            self._emit(f, "CROSS", prev["tid"], prev["team"])

    # ------------------------------------------------------------------ ana güncelleme
    def update(self, f: int, players: list[PlayerObs], ball: np.ndarray | None, ball_m: np.ndarray | None = None,
               metric: bool = False, pitch: bool = False) -> dict:
        """metric: pos_m metre cinsinden mi (yaklaşık dahil); pitch: gerçek saha koordinatı mı (homografi)"""
        if self.first_f is None:
            self.first_f = f
        self.last_f = f
        if pitch:
            for p in players:
                if p.pos_m is not None and p.team in (0, 1):
                    self.team_x[p.team] += p.pos_m[0]
                    self.team_xn[p.team] += 1

        cand, dmin = None, np.inf
        if ball is not None:
            for p in players:
                if p.team not in (0, 1):
                    continue
                d = ball_player_dist(ball, p.box)
                if d < dmin:
                    cand, dmin = p, d
            if dmin >= self.dist_thr:
                cand = None
        if cand is not None:
            if self.possessor is not None and cand.tid == self.possessor["tid"]:
                sp = self.possessor
                sp["last_f"], sp["box"] = f, cand.box.copy()
                if cand.pos_m is not None:
                    sp["last_m"] = cand.pos_m.copy()
                    if sp["start_m"] is None:
                        sp["start_m"] = cand.pos_m.copy()
                self.cand, self.cand_n = None, 0
                self.release = None
            else:
                self.cand_n = self.cand_n + 1 if self.cand == cand.tid else 1
                self.cand = cand.tid
                if self.cand_n >= self.confirm:
                    self._new_spell(f, cand, metric, pitch)
                    self.cand, self.cand_n = None, 0
        else:
            self.cand, self.cand_n = None, 0

        sp = self.possessor
        if sp is not None:
            in_contact = f - sp["last_f"] <= self.confirm
            if in_contact:
                self.stats[sp["tid"]]["poss_frames"] += 1
                for p in players:  # rakip baskısı (çalım için)
                    if p.team in (0, 1) and p.team != sp["team"] and feet_dist(p.box, sp["box"]) < 1.0:
                        if sp["opp_close_f"] is None:
                            sp["opp_close_f"] = f
                        break
            elif self.release is None and pitch:
                self.release = {"f": f, "tid": sp["tid"], "team": sp["team"], "done": False,
                                "pos": None if sp["last_m"] is None else sp["last_m"].copy()}
            team = sp["team"]
            self.team_frames[team] += 1
            self.win_q.append((f, team))
            self.timeline.append((f, team))
        while self.win_q and self.win_q[0][0] <= f - self.window:
            self.win_q.popleft()
        while self.pass_q and self.pass_q[0][0] <= f - self.window:
            self.pass_q.popleft()

        # şut takibi (yalnızca saha koordinatı varken)
        r = self.release
        speed = None
        if ball_m is not None and self.prev_ball is not None and f > self.prev_ball[0]:
            speed = float(np.linalg.norm(ball_m - self.prev_ball[1]) / ((f - self.prev_ball[0]) / self.fps))
        if ball_m is not None:
            self.prev_ball = (f, ball_m.copy())
        if r is not None and not r["done"] and pitch and ball_m is not None and f - r["f"] <= self.shot_watch:
            ar = self.attacks_right(r["team"])
            if ar is not None:
                x, y = ball_m
                goal_x = 105.0 if ar else 0.0
                near = r["pos"] is None or abs(r["pos"][0] - goal_x) <= 35.0  # şutlar kaleye ≤35 m'den
                fast = speed is not None and speed >= 12.0                       # ≥ 43 km/sa
                if (x > 105 - 3.0 if ar else x < 3.0) and 26.0 <= y <= 42.0 and near and fast:
                    self._emit(f, "SHOT", r["tid"], r["team"])
                    r["done"] = True
                    fp = sp.get("from_pass") if sp else None
                    if fp is not None and not fp.get("key") and f - fp["f"] <= 5 * self.fps:
                        self._emit(f, "KEY_PASS", fp["tid"], fp["team"], to=fp["to"])
                        fp["key"] = True
        return self.snapshot()

    # ------------------------------------------------------------------ canlı istatistikler
    def snapshot(self) -> dict:
        tot = self.team_frames.sum()
        pct = (self.team_frames / tot * 100) if tot else np.array([50.0, 50.0])
        wc = np.bincount([t for _, t in self.win_q], minlength=2).astype(float)
        wpct = wc / wc.sum() * 100 if wc.sum() else pct
        mins = max((self.last_f - (self.first_f or 0) + 1) / self.fps / 60.0, 1e-6)
        passes = np.zeros(2)
        fails = np.zeros(2)
        for e in self.events:
            if e["type"] == "PASS":
                passes[e["team"]] += 1
            elif e["type"] == "PASS_FAIL":
                fails[e["team"]] += 1
        wp = np.bincount([t for _, t in self.pass_q], minlength=2).astype(float)
        win_min = min(mins, self.window / self.fps / 60.0)
        return {
            "possession_pct": pct.round(1).tolist(),
            "possession_pct_window": wpct.round(1).tolist(),
            "passes": passes.astype(int).tolist(),
            "pass_fail": fails.astype(int).tolist(),
            "pass_accuracy": [round(float(passes[i] / (passes[i] + fails[i]) * 100), 1) if passes[i] + fails[i] else None
                              for i in range(2)],
            "passes_per_min": (passes / mins).round(2).tolist(),
            "passes_per_min_window": (wp / max(win_min, 1e-6)).round(2).tolist(),
            "elapsed_s": round(mins * 60, 1),
            "possessor": None if self.possessor is None else self.possessor["tid"],
            "control_team": None if self.possessor is None else self.possessor["team"],
            "last_events": self.events[-4:],
        }

    def finalize(self):
        if self.possessor is not None:
            self._close_spell(self.possessor)
        return self.snapshot()
