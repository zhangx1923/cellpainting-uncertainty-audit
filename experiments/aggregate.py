#!/usr/bin/env python3
"""Aggregate LOSO JSON results into summary tables (CSV + Markdown) for the paper.

Usage: aggregate.py [--dir experiments/results/loso] [--tag ""] [--out experiments/results/summary/main]
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
ROOT = os.environ.get("TRUST_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=f"{ROOT}/experiments/results/loso")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default=f"{ROOT}/experiments/results/summary/main")
ap.add_argument("--pk", default="raw")
ap.add_argument("--score", default="nlp")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

files = sorted(glob.glob(f"{args.dir}/*_s*{args.tag}.json"))
files = [f for f in files if f.endswith(f"{args.tag}.json") and (args.tag or "_s" in os.path.basename(f) and os.path.basename(f).count("_") == 2)]
rows, conf_rows, sel_rows, plate_rows = [], [], [], []
for f in files:
    r = json.load(open(f)); ho, seed = r["holdout"], r["seed"]
    m = r["metrics"][args.pk]
    rows.append({"holdout": ho, "seed": seed, "K": r["K"], "n_active": r["n_active_compounds"], "T": r["temperature"],
                 "ood_plates": r["n"]["ood_plates"], "n_ood": r["n"]["ood"],
                 **{f"id_{g}_{k}": m[f"idt_{g}"].get(k) for g in ["trt", "act", "pos"] for k in ["acc", "top5", "nll", "ece", "brier", "aurc"]},
                 **{f"ood_{g}_{k}": m[f"ood_{g}"].get(k) for g in ["trt", "act", "pos"] for k in ["acc", "top5", "nll", "ece", "brier", "aurc"]},
                 "id_false_naming": r["false_naming_rate"]["idt"], "ood_false_naming": r["false_naming_rate"]["ood"],
                 "id_dmso_inactive_recall": r["inactive_recall"]["idt"], "ood_dmso_inactive_recall": r["inactive_recall"]["ood"],
                 "dmso_domain_acc": r["dmso_domain_acc"], "frac_flagged": r["canary"]["ood_frac_plates_flagged"],
                 "gate_thr": r["canary"]["gate_thr_pos_acc"], "cal_nll": r["clf_info"].get("best_val_nll")})
    for key, e in r["conformal"].items():
        pk, sc, a = key.split("_")[0], key.split("_")[1], float(key.rsplit("_a", 1)[1])
        if pk != args.pk: continue
        d = {"holdout": ho, "seed": seed, "score": sc, "alpha": a, "id_trt_cov": e["id_trt_cov"], "id_trt_size": e["id_trt_size"],
             "id_act_cov": e["id_act_cov"], "id_act_size": e["id_act_size"], "c_pos_id": e["c_pos_id"], "c_pos_ood_at_qstd": e["c_pos_ood_at_qstd"],
             "id_trt_cov_v5": e.get("id_trt_cov_v5"), "id_trt_size_v5": e.get("id_trt_size_v5"), "id_act_cov_v5": e.get("id_act_cov_v5"), "id_act_size_v5": e.get("id_act_size_v5"), "ood_frac_looks_active_trt": e.get("ood_frac_looks_active_trt")}
        for v in ["std", "oracle", "v1_poscon", "v2_match", "v3_dmsoW", "v4_gated", "v5_actmond"]:
            for g in ["trt", "act"]:
                d[f"ood_{g}_cov_{v}"] = e.get(f"ood_{g}_cov_{v}"); d[f"ood_{g}_size_{v}"] = e.get(f"ood_{g}_size_{v}")
        for v in ["std", "v2_match"]:
            for fl in ["unflagged", "flagged"]:
                d[f"ood_trt_cov_{v}_{fl}"] = e.get(f"ood_trt_cov_{v}_{fl}"); d[f"ood_trt_size_{v}_{fl}"] = e.get(f"ood_trt_size_{v}_{fl}")
        conf_rows.append(d)
        if sc == args.score and a == 0.1:
            for p, cov in e["per_plate_std_trt_cov"].items():
                ps = r["canary"]["ood_plates"][p]
                plate_rows.append({"holdout": ho, "seed": seed, "plate": p, "std_trt_cov": cov, **{k: ps[k] for k in ps}, "flagged": ps["pos_acc"] is not None and ps["pos_acc"] < r["canary"]["gate_thr_pos_acc"],
                                   "id_pos_acc_mean": r["canary"]["id_pos_acc_mean"]})
    s = r["selective_named"][args.pk]
    for risk in [0.1, 0.2]:
        sel_rows.append({"holdout": ho, "seed": seed, "target_risk": risk, "frac_named_id": s["frac_named_id"], "frac_named_ood": s["frac_named_ood"],
                         "risk_named_id_nothr": s["risk_named_id_nothr"], "risk_named_ood_nothr": s["risk_named_ood_nothr"],
                         "id_risk": s[f"id_r{risk}"]["risk"], "id_cov": s[f"id_r{risk}"]["coverage"],
                         "ood_risk": s[f"ood_r{risk}"]["risk"], "ood_cov": s[f"ood_r{risk}"]["coverage"],
                         "ood_canary_risk": s[f"ood_canary_r{risk}"]["risk"], "ood_canary_cov": s[f"ood_canary_r{risk}"]["coverage"],
                         "ood_oracle_risk": s[f"ood_oracle_r{risk}"]["risk"], "ood_oracle_cov": s[f"ood_oracle_r{risk}"]["coverage"]})

M = pd.DataFrame(rows); C = pd.DataFrame(conf_rows); S = pd.DataFrame(sel_rows); PLT = pd.DataFrame(plate_rows)
M.to_csv(f"{args.out}/metrics_per_fold.csv", index=False); C.to_csv(f"{args.out}/conformal_per_fold.csv", index=False)
S.to_csv(f"{args.out}/selective_per_fold.csv", index=False); PLT.to_csv(f"{args.out}/plate_canary.csv", index=False)

def fmt(x, d=3): return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"
lines = [f"# Summary ({args.dir}, tag='{args.tag}', probs={args.pk}, score={args.score})", "", f"Folds: {M.holdout.nunique()} sources × {M.seed.nunique()} seeds = {len(M)} runs", ""]
# Table 1: metrics per source (mean over seeds)
g = M.groupby("holdout").mean(numeric_only=True)
lines += ["## Table 1 — ID vs OOD metrics per held-out lab (mean over seeds)", "",
          "| lab | plates | K | acc trt ID→OOD | acc active ID→OOD | ECE active ID→OOD | NLL active ID→OOD | AURC active ID→OOD | false-naming ID→OOD | poscon acc ID→OOD | flagged |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for ho, x in g.iterrows():
    lines.append(f"| {ho} | {int(x.ood_plates)} | {x.K:.0f} | {fmt(x.id_trt_acc)}→{fmt(x.ood_trt_acc)} | {fmt(x.id_act_acc)}→{fmt(x.ood_act_acc)} | {fmt(x.id_act_ece)}→{fmt(x.ood_act_ece)} | {fmt(x.id_act_nll,2)}→{fmt(x.ood_act_nll,2)} | {fmt(x.id_act_aurc)}→{fmt(x.ood_act_aurc)} | {fmt(x.id_false_naming)}→{fmt(x.ood_false_naming)} | {fmt(x.id_pos_acc)}→{fmt(x.ood_pos_acc)} | {fmt(x.frac_flagged,2)} |")
mm = M.mean(numeric_only=True)
lines.append(f"| **mean** | | | {fmt(mm.id_trt_acc)}→{fmt(mm.ood_trt_acc)} | {fmt(mm.id_act_acc)}→{fmt(mm.ood_act_acc)} | {fmt(mm.id_act_ece)}→{fmt(mm.ood_act_ece)} | {fmt(mm.id_act_nll,2)}→{fmt(mm.ood_act_nll,2)} | {fmt(mm.id_act_aurc)}→{fmt(mm.ood_act_aurc)} | {fmt(mm.id_false_naming)}→{fmt(mm.ood_false_naming)} | {fmt(mm.id_pos_acc)}→{fmt(mm.ood_pos_acc)} | {fmt(mm.frac_flagged,2)} |")
# Table 2: conformal
for grp in ["trt", "act"]:
    for sc in sorted(C.score.unique()):
        lines += ["", f"## Table 2{grp}/{sc} — conformal coverage / set size on OOD {'treatment wells' if grp=='trt' else 'active-compound treatment wells'} ({sc} score); mean over folds (±sd over folds)", "",
                  "| α | ID cov/size | std | v1 poscon | v2 match | v3 DMSO-w | v4 gated | v5 act-Mondrian | oracle | folds std under | folds v5 under |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in sorted(C.alpha.unique()):
            x = C[(C.score == sc) & (C.alpha == a)]
            def cs(v):
                c, s = x[f"ood_{grp}_cov_{v}"], x[f"ood_{grp}_size_{v}"]
                return f"{c.mean():.3f}±{c.std():.3f} / {s.mean():.1f}"
            under = lambda v: int((x[f"ood_{grp}_cov_{v}"] < 1 - a - 0.02).sum())
            def cs2(v):
                c, s_ = x[f"ood_{grp}_cov_{v}"], x[f"ood_{grp}_size_{v}"]
                return f"{c.mean():.3f}±{c.std():.3f} / {s_.mean():.1f}" if c.notna().any() else "—"
            lines.append(f"| {a} | {x[f'id_{grp}_cov'].mean():.3f} / {x[f'id_{grp}_size'].mean():.1f} | {cs('std')} | {cs('v1_poscon')} | {cs('v2_match')} | {cs('v3_dmsoW')} | {cs('v4_gated')} | {cs2('v5_actmond')} | {cs('oracle')} | {under('std')}/{len(x)} | {int((x[f'ood_{grp}_cov_v5_actmond'] < 1 - a - 0.02).sum()) if x[f'ood_{grp}_cov_v5_actmond'].notna().any() else '—'}/{len(x)} |")
# Table 3: selective
lines += ["", "## Table 3 — selective prediction among NAMED calls (prediction ≠ inactive), threshold chosen on ID calibration for target risk", "",
          "| target risk | ID risk / coverage | OOD risk / coverage (ID threshold) | OOD canary-retuned risk / coverage | OOD oracle risk / coverage | folds with OOD risk > target+0.05 |", "|---|---|---|---|---|---|"]
for risk in [0.1, 0.2]:
    x = S[S.target_risk == risk]
    lines.append(f"| {risk} | {x.id_risk.mean():.3f} / {x.id_cov.mean():.3f} | {x.ood_risk.mean():.3f} / {x.ood_cov.mean():.3f} | {x.ood_canary_risk.mean():.3f} / {x.ood_canary_cov.mean():.3f} | {x.ood_oracle_risk.mean():.3f} / {x.ood_oracle_cov.mean():.3f} | {int((x.ood_risk > risk + 0.05).sum())}/{len(x)} |")
lines += ["", f"Unthresholded risk among named calls: ID {S.risk_named_id_nothr.mean():.3f}, OOD {S.risk_named_ood_nothr.mean():.3f}; fraction of treatment wells named: ID {S.frac_named_id.mean():.3f}, OOD {S.frac_named_ood.mean():.3f}."]
# Table 4: canary correlation at plate level
if len(PLT):
    P2 = PLT.dropna(subset=["pos_acc", "std_trt_cov"])
    from scipy.stats import spearmanr, pearsonr
    rs, ps = spearmanr(P2.pos_acc, P2.std_trt_cov); rt, pt = spearmanr(P2.pos_acc, P2.trt_acc)
    ra, pa = spearmanr(P2.pos_acc, P2.act_acc)
    lines += ["", "## Table 4 — plate-level canary analysis (OOD plates, std CP at α=0.1)", "",
              f"n plates = {len(P2)}; Spearman(poscon acc, std CP trt coverage) = {rs:.3f} (p={ps:.1e}); Spearman(poscon acc, trt acc) = {rt:.3f} (p={pt:.1e}); Spearman(poscon acc, active acc) = {ra:.3f} (p={pa:.1e})",
              f"flagged plates: {int(P2.flagged.sum())}/{len(P2)}; mean std-CP coverage flagged {P2[P2.flagged].std_trt_cov.mean():.3f} vs unflagged {P2[~P2.flagged].std_trt_cov.mean():.3f}; mean active acc flagged {P2[P2.flagged].act_acc.mean():.3f} vs unflagged {P2[~P2.flagged].act_acc.mean():.3f}"]
open(f"{args.out}/SUMMARY.md", "w").write("\n".join(lines)); print("\n".join(lines))
