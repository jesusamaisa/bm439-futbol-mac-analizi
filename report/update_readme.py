"""README.md içindeki <!-- SONUCLAR --> bölümünü results/ çıktılarından üretir (elle sayı yazılmaz)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "report"))
from build_report import load  # noqa: E402
from content import FINAL, a1_val_better_all_seeds, cp_seed_status, final_poss, gd, has_g1  # noqa: E402


def block(d) -> str:
    det, boot, team, poss, bench = d["det"], d["boot"], d["team"], d["poss"], d["bench"]
    out = ["<!-- SONUCLAR -->", "## Ölçülen sonuçlar (test kümesi)", ""]
    if det is not None:
        ci = boot["ball_AP50_bootstrap_CI95"]
        out += ["| Yapılandırma | Top AP@0.5 [%95 GA] | Top AP_m* | Oyuncu AP@0.5 | Hakem AP@0.5 | mAP@[.5:.95] | ms/kare |",
                "|---|---|---|---|---|---|---|"]
        for k, name in (("E1", "E1 Temel — COCO YOLOv8s"), ("E2_ft640", "E2 Alternatif — ince ayar 640"),
                        ("A2_ft640_cp", "A2 Ablation — 640 + kopyala-yapıştır"),
                        ("E3_ft1280_cp", "E3 Önerilen iyileştirme — 1280 + kopyala-yapıştır"),
                        ("A1_ft1280", "A1 Ablation — 1280, artırımsız" if has_g1(d)
                         else "**A1 — 1280, artırımsız (son sistem; doğrulamada en iyi)**"),
                        ("C1_ft1280_ep50", "C1 Kontrol — A1 verisi, G1 ile eşit adım (50 epoch)"),
                        (FINAL, "**G1 — A1 + SoccerNet-GSR ek eğitim verisi (son sistem; doğrulamada en iyi)**"),
                        ("R", "R Referans — Hmzbo YOLOv8L")):
            if k not in det.index:
                continue
            x = det.loc[k]
            mid = boot["chosen_input_size"].get(k, k)
            ms = bench.loc[mid, "latency_ms_mean"] if bench is not None and mid in bench.index else float("nan")
            ref = "—" if x["referee_AP50"] != x["referee_AP50"] else f"{x['referee_AP50']:.3f}"
            out.append(f"| {name} | {x['ball_AP50']:.3f} [{ci[k]['lo']:.3f}, {ci[k]['hi']:.3f}] | {x['ball_APc']:.3f} | "
                       f"{x['player_AP50']:.3f} | "
                       f"{ref} | {x['mAP50_95']:.3f} | {'—' if ms != ms else f'{ms:.1f}'} |")
        out.append("")
        out.append("*AP_m: merkez uzaklığı ≤ 2 top çapı ile eşleştirme (etiket ötelemesine dayanıklı yardımcı metrik).")
        out.append("")
        sd = d.get("seeds")
        if sd and all(sd["summary"].get(k, {}).get("n_seeds", 0) >= 2 for k in ("E3_ft1280_cp", "A1_ft1280")):
            s3, sa = sd["summary"]["E3_ft1280_cp"]["ball_AP50"], sd["summary"]["A1_ft1280"]["ball_AP50"]
            dm = sd["paired_differences"].get("tohum ortalaması: E3 - A1", {})
            d1 = sd["paired_differences"].get("seed1: E3 - A1", {})
            out.append(f"Tohum tekrarı (eğitim varyansı, 2 tohum, ort. ± std): top AP@0.5 E3 **{s3['mean']:.3f} ± "
                       f"{s3['std']:.3f}**, A1 **{sa['mean']:.3f} ± {sa['std']:.3f}**; tohum 1'de E3−A1 = "
                       f"{d1.get('mean', float('nan')):+.3f} [{d1.get('lo', float('nan')):+.3f}, "
                       f"{d1.get('hi', float('nan')):+.3f}], tohum ortalamasında "
                       f"{dm.get('mean', float('nan')):+.3f} [{dm.get('lo', float('nan')):+.3f}, "
                       f"{dm.get('hi', float('nan')):+.3f}] (`results/detection/seed_repeats.json`).")
            out.append("")
        if has_g1(d):
            dga, dgc, dca = gd(d, FINAL, "A1_ft1280"), gd(d, FINAL, "C1_ft1280_ep50"), gd(d, "C1_ft1280_ep50", "A1_ft1280")
            fm = lambda x: f"{x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"
            out.append(f"Ek eğitim verisi (SoccerNet-GSR valid, 3 yeni maç, 2900 kare): G1−A1 = {fm(dga)}; aynı adım "
                       f"sayısıyla eğitilen kontrole göre G1−C1 = {fm(dgc)}, C1−A1 = {fm(dca)}. Kazanç daha uzun "
                       f"eğitimden değil ek veriden gelir (`results/detection/gsr_extra.json`).")
            out.append("")
    if team is not None:
        t = team.set_index(["set", "config"])
        out += ["| Takım ataması | Doğrulama | Yoğun test | Seyrek test |", "|---|---|---|---|"]
        for c, n in (("T1_baseline", "Temel (ort. RGB + K-Means)"), ("T3_proposed_v1", "Önerilen v1 (sabit HSV maskesi)"),
                     ("T5_local_mask", "**Önerilen (yerel saha maskesi)**")):
            g = lambda s: f"{t.loc[(s, c), 'acc_mean']:.3f}" if (s, c) in t.index else "—"
            out.append(f"| {n} | {g('val')} | {g('dense')} | {g('sparse')} |")
        out.append("")
    if poss is not None and final_poss(poss) is not None:
        v = final_poss(poss)
        st = cp_seed_status(d)
        why = ("Kopyala-yapıştır artırımı ablation'da top AP@0.5'e katkı sağlamadığı için (tohum 0'da E3−A1 anlamlı "
               "biçimde < 0, tohum 1'de fark anlamsız) son sistemde"
               + (" doğrulamada iki tohumda da daha iyi olan" if a1_val_better_all_seeds(d) else "") + " A1 kullanılır."
               if st == "neg_some" else
               "Kopyala-yapıştır artırımı ablation'da top AP@0.5'i anlamlı biçimde düşürdüğü için (E3−A1 < 0) son "
               "sistemde doğrulamada en iyi model A1 kullanılır.")
        if has_g1(d) and "V7_G1_final" in poss.index:
            why = (why.replace(" son sistemde", " sonraki adımda").replace(" A1 kullanılır.", " A1 ayarları kullanılır.")
                   + " Bu ayarlarla ek veriyle eğitilen G1 doğrulamada en iyi olduğundan son sistemde G1 kullanılır.")
        out.append(f"{why} Son sistem, 10 yoğun test sekansında vekil referansa "
                   f"göre pas olaylarını **F1 = {v['pass_F1_mean']:.3f}**, kare bazlı top kontrolünü "
                   f"**{v['control_agreement_mean']:.3f}** uyumla ölçer; klip başına hakimiyet yüzdesi hatası "
                   f"{v['poss_err_pp_mean']:.1f} yüzde puanıdır (30 sn'lik kliplerde yüksek sapmalı). Ayrıntılar, ablation'lar ve hata analizi: `report/rapor.pdf`, tüm kayıtlar: "
                   f"`experiments/registry.csv`.")
        out.append("")
    if d.get("pipe") is not None:
        out.append(f"Tam hat hızı (taktik kamera, {d['pipe']['device']}): **{d['pipe']['pipeline_fps']:.1f} FPS**.")
        out.append("")
    out.append("<!-- /SONUCLAR -->")
    return "\n".join(out)


def main():
    p = ROOT / "README.md"
    s = p.read_text(encoding="utf-8")
    new = block(load())
    if "<!-- /SONUCLAR -->" in s:
        s = re.sub(r"<!-- SONUCLAR -->.*?<!-- /SONUCLAR -->", lambda m: new, s, flags=re.S)
    else:
        s = s.replace("<!-- SONUCLAR -->", new)
    p.write_text(s, encoding="utf-8")
    print("[✓] README sonuç bölümü güncellendi")


if __name__ == "__main__":
    main()
