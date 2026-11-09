#!/usr/bin/env python3
"""Generate LaTeX tables from aggregated summaries -> paper/tables/*.tex"""
import argparse, json, os
import numpy as np, pandas as pd
ROOT = os.environ.get("TRUST_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--main", default=f"{ROOT}/experiments/results/summary/main")
ap.add_argument("--root", default=f"{ROOT}/experiments/results/summary")
ap.add_argument("--out", default=f"{ROOT}/paper/tables")
ap.add_argument("--score", default="nlp")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
M = pd.read_csv(f"{a.main}/metrics_per_fold.csv"); C = pd.read_csv(f"{a.main}/conformal_per_fold.csv"); S = pd.read_csv(f"{a.main}/selective_per_fold.csv")
labs = sorted(M.holdout.unique(), key=lambda s: int(s.split("_")[1]))
g = M.groupby("holdout").mean(numeric_only=True).loc[labs]
f3 = lambda v: f"{v:.3f}"; f2 = lambda v: f"{v:.2f}"
# ---- Table: per-lab audit ----
L = [r"\begin{tabular}{lrrcccccc}", r"\toprule",
     r"Held-out lab & plates & $K$ & \multicolumn{2}{c}{acc.\ (active)} & \multicolumn{2}{c}{ECE (active)} & AURC (active) & poscon acc. \\",
     r" & & & ID & unseen & ID & unseen & ID$\rightarrow$unseen & ID$\rightarrow$unseen \\", r"\midrule"]
for lab, x in g.iterrows():
    L.append(f"{lab.replace('source_', 'L')} & {int(x.ood_plates)} & {x.K:.0f} & {f3(x.id_act_acc)} & {f3(x.ood_act_acc)} & {f3(x.id_act_ece)} & {f3(x.ood_act_ece)} & {f2(x.id_act_aurc)}$\\rightarrow${f2(x.ood_act_aurc)} & {f2(x.id_pos_acc)}$\\rightarrow${f2(x.ood_pos_acc)} \\\\")
mm = M.mean(numeric_only=True)
L += [r"\midrule", f"mean & {int(g.ood_plates.sum())} & {mm.K:.0f} & {f3(mm.id_act_acc)} & {f3(mm.ood_act_acc)} & {f3(mm.id_act_ece)} & {f3(mm.ood_act_ece)} & {f2(mm.id_act_aurc)}$\\rightarrow${f2(mm.ood_act_aurc)} & {f2(mm.id_pos_acc)}$\\rightarrow${f2(mm.ood_pos_acc)} \\\\", r"\bottomrule", r"\end{tabular}"]
open(f"{a.out}/tab_perlab.tex", "w").write("\n".join(L))
# ---- Table: conformal variants ----
names = [("std", "Standard split CP (ID calibration)"), ("v1_poscon", "Positive-control Mondrian"), ("v2_match", "Control-coverage matching"),
         ("v3_dmsoW", "DMSO-weighted CP (label-free)"), ("v4_gated", "Canary-gated matching"), ("v5_actmond", "Activity-split Mondrian (both controls)"), ("oracle", "Oracle, marginal (labelled unseen-lab trt.\\ wells)")]
L = [r"\begin{tabular}{lcccc}", r"\toprule", r"Calibration on the unseen lab & \multicolumn{2}{c}{all treatment wells} & \multicolumn{2}{c}{active-compound wells} \\",
     r" & coverage & size & coverage & size \\", r"\midrule"]
x = C[(C.score == a.score) & (C.alpha == 0.1)]
L.append(f"\\emph{{In-distribution plates (reference)}} & {f3(x.id_trt_cov.mean())} & {x.id_trt_size.mean():.1f} & {f3(x.id_act_cov.mean())} & {x.id_act_size.mean():.1f} \\\\")
for v, nm in names:
    if f"ood_trt_cov_{v}" not in x or x[f"ood_trt_cov_{v}"].isna().all(): continue
    L.append(f"{nm} & {f3(x[f'ood_trt_cov_{v}'].mean())}$\\pm${f2(x[f'ood_trt_cov_{v}'].std())} & {x[f'ood_trt_size_{v}'].mean():.1f} & {f3(x[f'ood_act_cov_{v}'].mean())}$\\pm${f2(x[f'ood_act_cov_{v}'].std())} & {x[f'ood_act_size_{v}'].mean():.1f} \\\\")
try:  # active-calibrated oracle: quantile of labelled unseen-lab active-compound wells (analysis_revision.py)
    O = pd.read_csv(f"{a.root}/revision_main/oracles_per_fold.csv"); o = O[O.alpha == 0.1]
    L.insert(-1, f"Plate-standardised CP (V6) & {f3(o.trt_cov_v6.mean())}$\\pm${f2(o.trt_cov_v6.std())} & {o.trt_size_v6.mean():.1f} & {f3(o.act_cov_v6.mean())}$\\pm${f2(o.act_cov_v6.std())} & {o.act_size_v6.mean():.1f} \\\\")
    L.append(f"Oracle, active-calibrated (labelled unseen-lab active wells) & {f3(o.trt_cov_oracle_act.mean())}$\\pm${f2(o.trt_cov_oracle_act.std())} & {o.trt_size_oracle_act.mean():.1f} & {f3(o.act_cov_oracle_act.mean())}$\\pm${f2(o.act_cov_oracle_act.std())} & {o.act_size_oracle_act.mean():.1f} \\\\")
except FileNotFoundError: pass
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{a.out}/tab_cp.tex", "w").write("\n".join(L))
# ---- Table: per-lab coverage, gate and selective risk (alpha = 0.1, 10% risk target) ----
try:
    O = pd.read_csv(f"{a.root}/revision_main/oracles_per_fold.csv"); o = O[O.alpha == 0.1].groupby("holdout").mean(numeric_only=True)
    x = C[(C.score == a.score) & (C.alpha == 0.1)].groupby("holdout").mean(numeric_only=True)
    s_ = S[S.target_risk == 0.1].groupby("holdout").mean(numeric_only=True)
    Pc = pd.read_csv(f"{a.main}/plate_canary.csv"); gt = Pc.groupby("holdout").agg(n=("flagged", "size"), f=("flagged", "sum"))
    L = [r"\begin{tabular}{lcccccccc}", r"\toprule", r"Held-out & flagged & \multicolumn{5}{c}{CP coverage, active wells ($\alpha=0.1$)} & \multicolumn{2}{c}{risk, 10\% target} \\",
         r"lab & plate evals & std & V5 & V1 & V6 (size) & oracle-act (size) & ID thr. & canary \\", r"\midrule"]
    fx = lambda v: "--" if v != v else f3(v)
    for lab in labs:
        L.append(f"{lab.replace('source_', 'L')} & {int(gt.loc[lab, 'f'])}/{int(gt.loc[lab, 'n'])} & {f3(x.loc[lab, 'ood_act_cov_std'])} & {f3(x.loc[lab, 'ood_act_cov_v5_actmond'])} & {f3(x.loc[lab, 'ood_act_cov_v1_poscon'])} & {f3(o.loc[lab, 'act_cov_v6'])} ({o.loc[lab, 'act_size_v6']:.0f}) & {f3(o.loc[lab, 'act_cov_oracle_act'])} ({o.loc[lab, 'act_size_oracle_act']:.0f}) & {fx(s_.loc[lab, 'ood_risk'])} & {fx(s_.loc[lab, 'ood_canary_risk'])} \\\\")
    med = lambda c, d=x: f3(d.loc[labs, c].median())
    L += [r"\midrule", f"median of labs & -- & {med('ood_act_cov_std')} & {med('ood_act_cov_v5_actmond')} & {med('ood_act_cov_v1_poscon')} & {f3(o.loc[labs, 'act_cov_v6'].median())} ({o.loc[labs, 'act_size_v6'].median():.0f}) & {f3(o.loc[labs, 'act_cov_oracle_act'].median())} ({o.loc[labs, 'act_size_oracle_act'].median():.0f}) & {f3(s_.loc[labs, 'ood_risk'].median())} & {f3(s_.loc[labs, 'ood_canary_risk'].median())} \\\\",
          r"\bottomrule", r"\end{tabular}"]
    open(f"{a.out}/tab_perlab_cov.tex", "w").write("\n".join(L))
except FileNotFoundError: pass
# ---- Table: selective ----
L = [r"\begin{tabular}{lcccc}", r"\toprule", r"Threshold rule & \multicolumn{2}{c}{target risk 10\%} & \multicolumn{2}{c}{target risk 20\%} \\", r" & risk & coverage & risk & coverage \\", r"\midrule"]
s1, s2 = S[S.target_risk == 0.1], S[S.target_risk == 0.2]
rows = [("ID threshold, in-distribution plates", "id_risk", "id_cov"), ("ID threshold, unseen lab", "ood_risk", "ood_cov"), ("Canary-retuned threshold, unseen lab", "ood_canary_risk", "ood_canary_cov"), ("Oracle threshold, unseen lab", "ood_oracle_risk", "ood_oracle_cov")]
for nm, rk, cv in rows:
    L.append(f"{nm} & {f3(s1[rk].mean())} & {f3(s1[cv].mean())} & {f3(s2[rk].mean())} & {f3(s2[cv].mean())} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{a.out}/tab_selective.tex", "w").write("\n".join(L))
# ---- Table: baselines ----
L = [r"\begin{tabular}{lrccccc}", r"\toprule", r"Representation / normalisation & runs & acc.\ active ID & acc.\ active unseen & ECE active unseen & CP cov.\ trt unseen & CP cov.\ active unseen \\", r"\midrule"]
for t, nm in [("harmony", "Harmony consortium profiles (transductive)"), ("negcon9", "CellProfiler, per-plate DMSO (inductive)"), ("allwell9", "CellProfiler, per-plate all-well (inductive)"), ("global9", "CellProfiler, global training statistics")]:
    try:
        Mt = pd.read_csv(f"{a.root}/{t}/metrics_per_fold.csv"); Ct = pd.read_csv(f"{a.root}/{t}/conformal_per_fold.csv"); ct = Ct[(Ct.score == a.score) & (Ct.alpha == 0.1)]
        L.append(f"{nm} & {len(Mt)} & {f3(Mt.id_act_acc.mean())} & {f3(Mt.ood_act_acc.mean())} & {f3(Mt.ood_act_ece.mean())} & {f3(ct.ood_trt_cov_std.mean())} & {f3(ct.ood_act_cov_std.mean())} \\\\")
    except FileNotFoundError: pass
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{a.out}/tab_baselines.tex", "w").write("\n".join(L))
# ---- Table: principal comparison across classifiers (256 PCs default, 64 PCs, logistic regression) ----
def col(t, rev):
    try:
        Mt = pd.read_csv(f"{a.root}/{t}/metrics_per_fold.csv"); Ct = pd.read_csv(f"{a.root}/{t}/conformal_per_fold.csv"); ct = Ct[(Ct.score == a.score) & (Ct.alpha == 0.1)]
        St = pd.read_csv(f"{a.root}/{t}/selective_per_fold.csv"); st = St[St.target_risk == 0.1]
    except FileNotFoundError: return None
    from scipy.stats import spearmanr
    Pt = pd.read_csv(f"{a.root}/{t}/plate_canary.csv").groupby(["holdout", "plate"]).mean(numeric_only=True)
    sz = lambda c, s: "--" if c not in ct or ct[c].isna().all() else f"{f3(ct[c].mean())} ({ct[s].mean():.1f})"
    d = {"runs": str(len(Mt)), "calnll": f3(Mt.cal_nll.mean()) if "cal_nll" in Mt and Mt.cal_nll.notna().all() else "--",
         "accid": f3(Mt.id_act_acc.mean()), "accood": f3(Mt.ood_act_acc.mean()), "eceood": f3(Mt.ood_act_ece.mean()), "aurcood": f3(Mt.ood_act_aurc.mean()),
         "covid": f3(ct.id_act_cov.mean()), "std": sz("ood_act_cov_std", "ood_act_size_std"), "v5": sz("ood_act_cov_v5_actmond", "ood_act_size_v5_actmond"), "v1": sz("ood_act_cov_v1_poscon", "ood_act_size_v1_poscon"),
         "orc": "--", "v6": "--", "risk": f3(st.ood_risk.mean()), "can": f3(st.ood_canary_risk.mean()), "rho": f2(spearmanr(Pt.pos_acc, Pt.act_acc)[0])}
    if rev:
        try:
            O = pd.read_csv(f"{a.root}/{rev}/oracles_per_fold.csv"); o = O[O.alpha == 0.1]; d["orc"] = f"{f3(o.act_cov_oracle_act.mean())} ({o.act_size_oracle_act.mean():.1f})"
            d["v6"] = f"{f3(o.act_cov_v6.mean())} ({o.act_size_v6.mean():.1f})"
        except FileNotFoundError: pass
    return d
cols = [("MLP, 256 PCs (default)", col("main", "revision_main")), ("MLP, 64 PCs", col("pcs64full", "revision_pcs64")), ("Log.\\ reg., 256 PCs, seed 0", col("logreg", None))]
cols = [(n, d) for n, d in cols if d is not None]
rows = [("runs", "runs"), ("calibration-plate NLL (model selection)", "calnll"), ("acc.\\ active, ID", "accid"), ("acc.\\ active, unseen", "accood"), ("ECE active, unseen", "eceood"),
        ("AURC active, unseen", "aurcood"), ("CP cov.\\ active, ID plates", "covid"), ("CP cov.\\ active, standard (size)", "std"), ("CP cov.\\ active, V5 (size)", "v5"),
        ("CP cov.\\ active, V1 (size)", "v1"), ("CP cov.\\ active, V6 (size)", "v6"), ("CP cov.\\ active, active oracle (size)", "orc"), ("risk at 10\\% target, ID threshold", "risk"), ("risk at 10\\% target, canary", "can"),
        ("canary $\\rho$ (plates, seed-averaged)", "rho")]
L = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", "Unseen-lab metric & " + " & ".join(n for n, _ in cols) + r" \\", r"\midrule"]
for nm, k in rows: L.append(nm + " & " + " & ".join(d[k] for _, d in cols) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{a.out}/tab_ablation.tex", "w").write("\n".join(L))
print("tables written")
# ---- Table: model-side repairs (seed 0, all labs; screen_compare.py) ----
try:
    T = pd.read_csv(f"{a.root}/repair/tag_means.csv", index_col=0); Tj = json.load(open(f"{a.root}/repair/tests.json"))["tests"]
    order = [("_base", "default MLP"), ("_ens5", "deep ensemble (5)"), ("_aug3", "plate augmentation 0.3"), ("_aug5", "plate augmentation 0.5"), ("_aug3ens5", "augmentation 0.3 + ensemble"), ("_sphere", "per-lab DMSO whitening"), ("_alignlab", "control-anchored alignment")]
    order = [(t, n) for t, n in order if t in T.index]
    def cell(t, c, d=3):
        v = f"{T.loc[t, c]:.{d}f}" if d else f"{T.loc[t, c]:.1f}"
        k = f"{t}|{c}"
        if t != "_base" and k in Tj and Tj[k]["p"] == Tj[k]["p"] and Tj[k]["p"] < 0.05: v += "$^{*}$"
        return v
    L = [r"\begin{tabular}{lcccccccc}", r"\toprule", r"Model & cal. & acc. & ECE & AURC & std CP & V1 & oracle & risk 10\% \\", r" & NLL (ID) & active & active & active & cov. & cov. & size & canary \\", r"\midrule"]
    for t, n in order:
        L.append(f"{n} & {T.loc[t, 'cal_nll']:.3f} & {cell(t, 'acc_act')} & {cell(t, 'ece_act')} & {cell(t, 'aurc_act')} & {cell(t, 'act_std')} & {cell(t, 'act_v1_poscon')} & {cell(t, 'actsz_oracle_act', 0)} & {cell(t, 'risk10_can')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{a.out}/tab_repair.tex", "w").write("\n".join(L))
except FileNotFoundError: pass
