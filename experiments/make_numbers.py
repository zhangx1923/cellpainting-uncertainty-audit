#!/usr/bin/env python3
"""Emit LaTeX macros (paper/numbers.tex) from aggregated summaries so every number in the paper traces to a CSV."""
import os
import argparse, json, glob
import numpy as np, pandas as pd
ROOT = os.environ.get("TRUST_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--main", default=f"{ROOT}/experiments/results/summary/main")
ap.add_argument("--root", default=f"{ROOT}/experiments/results/summary")
ap.add_argument("--out", default=f"{ROOT}/paper/numbers.tex")
ap.add_argument("--score", default="nlp")
a = ap.parse_args()

M = pd.read_csv(f"{a.main}/metrics_per_fold.csv"); C = pd.read_csv(f"{a.main}/conformal_per_fold.csv")
S = pd.read_csv(f"{a.main}/selective_per_fold.csv"); P = pd.read_csv(f"{a.main}/plate_canary.csv")
_sp = f"{ROOT}/data/target2_summary.json"  # written by prepare_data.py; a copy ships with the summaries
summ = json.load(open(_sp if os.path.exists(_sp) else f"{a.root}/target2_summary.json"))
mac = {}
def m(name, val, d=3):
    mac[name] = f"{val:.{d}f}" if isinstance(val, (float, np.floating)) else str(val)
def pct(name, val, d=0): mac[name] = f"{100*val:.{d}f}"

m("nLabs", M.holdout.nunique()); m("nSeeds", M.seed.nunique()); m("nRuns", len(M))
m("nPlates", summ["n_plates"]); m("nWells", f"{summ['n_wells']:,}"); m("nFeaturesRaw", f"{summ['n_features_raw']:,}"); m("nFeatures", f"{summ['n_features_kept']:,}")
m("nBatches", summ["n_batches"]); m("nTrtCompounds", summ["n_trt_compounds"]); m("nCtrlCompounds", summ["n_control_compounds"]); m("nTrtWithMoa", summ["trt_with_moa"])
m("nActiveMean", f"{(M.K - 1).mean():.0f}"); m("nActiveMin", int((M.K - 1).min())); m("nActiveMax", int((M.K - 1).max()))
mm = M.mean(numeric_only=True)
for k in ["id_trt_acc", "ood_trt_acc", "id_act_acc", "ood_act_acc", "id_act_ece", "ood_act_ece", "id_act_nll", "ood_act_nll", "id_act_aurc", "ood_act_aurc", "id_pos_acc", "ood_pos_acc", "id_false_naming", "ood_false_naming", "frac_flagged", "id_act_top5", "ood_act_top5"]:
    m(k.replace("_", "").replace("top5", "topfive"), mm[k], 2 if "nll" in k else 3)
g = M.groupby("holdout").mean(numeric_only=True)
m("accActOODmin", g.ood_act_acc.min()); m("accActOODmax", g.ood_act_acc.max()); m("labActOODmin", g.ood_act_acc.idxmin().replace("source_", "L")); m("labActOODmax", g.ood_act_acc.idxmax().replace("source_", "L"))
m("eceActOODmax", g.ood_act_ece.max()); m("eceActOODmin", g.ood_act_ece.min())
m("accActOODLnine", g.loc["source_9", "ood_act_acc"]); m("eceActOODLnine", g.loc["source_9", "ood_act_ece"]); m("posAccOODLnine", g.loc["source_9", "ood_pos_acc"]); m("flaggedLnine", g.loc["source_9", "frac_flagged"], 2)
m("nLabsAccDropGtTen", int(((g.id_act_acc - g.ood_act_acc) > 0.10).sum())); m("nLabsEceUp", int((g.ood_act_ece > g.id_act_ece).sum()))
# conformal
for grp in ["trt", "act"]:
    for alpha in [0.05, 0.1, 0.2]:
        x = C[(C.score == a.score) & (C.alpha == alpha)]; tag = {0.05: "Five", 0.1: "Ten", 0.2: "Twenty"}[alpha]; G = grp.capitalize()
        m(f"cp{G}IDcov{tag}", x[f"id_{grp}_cov"].mean()); m(f"cp{G}IDsize{tag}", x[f"id_{grp}_size"].mean(), 1)
        for v, nm in [("std", "Std"), ("oracle", "Oracle"), ("v1_poscon", "Vone"), ("v2_match", "Vtwo"), ("v3_dmsoW", "Vthree"), ("v4_gated", "Vfour"), ("v5_actmond", "Vfive")]:
            col = f"ood_{grp}_cov_{v}"
            if col in x and x[col].notna().any():
                m(f"cp{G}{nm}cov{tag}", x[col].mean()); m(f"cp{G}{nm}sd{tag}", x[col].std()); m(f"cp{G}{nm}size{tag}", x[f"ood_{grp}_size_{v}"].mean(), 1)
                m(f"cp{G}{nm}under{tag}", int((x[col] < 1 - alpha - 0.02).sum()))
        if f"id_{grp}_cov_v5" in x and x[f"id_{grp}_cov_v5"].notna().any():
            m(f"cp{G}IDcovVfive{tag}", x[f"id_{grp}_cov_v5"].mean()); m(f"cp{G}IDsizeVfive{tag}", x[f"id_{grp}_size_v5"].mean(), 1)
x = C[(C.score == a.score) & (C.alpha == 0.1)]
m("cpTrtStdMinCov", x.ood_trt_cov_std.min()); m("cpTrtStdMaxCov", x.ood_trt_cov_std.max())
m("cpActStdMinCov", x.ood_act_cov_std.min()); m("cpActStdMaxCov", x.ood_act_cov_std.max())
m("fracLooksActive", x.ood_frac_looks_active_trt.mean() if "ood_frac_looks_active_trt" in x and x.ood_frac_looks_active_trt.notna().any() else float("nan"))
# selective
for risk, tag in [(0.1, "Ten"), (0.2, "Twenty")]:
    s = S[S.target_risk == risk]
    m(f"selIDrisk{tag}", s.id_risk.mean()); m(f"selIDcov{tag}", s.id_cov.mean()); m(f"selOODrisk{tag}", s.ood_risk.mean()); m(f"selOODcov{tag}", s.ood_cov.mean())
    m(f"selCanRisk{tag}", s.ood_canary_risk.mean()); m(f"selCanCov{tag}", s.ood_canary_cov.mean()); m(f"selOrRisk{tag}", s.ood_oracle_risk.mean()); m(f"selOrCov{tag}", s.ood_oracle_cov.mean())
    m(f"selFail{tag}", int((s.ood_risk > risk + 0.05).sum())); m(f"selOODriskMax{tag}", s.ood_risk.max()); m(f"selOODriskMin{tag}", s.ood_risk.min())
m("riskNamedIDnothr", S.risk_named_id_nothr.mean()); m("riskNamedOODnothr", S.risk_named_ood_nothr.mean()); m("fracNamedID", S.frac_named_id.mean()); m("fracNamedOOD", S.frac_named_ood.mean())
# canary
from scipy.stats import spearmanr
P2 = P.dropna(subset=["pos_acc", "std_trt_cov", "act_acc"])
m("nOODPlates", len(P2)); m("spearmanPosAct", spearmanr(P2.pos_acc, P2.act_acc)[0]); m("spearmanPosTrt", spearmanr(P2.pos_acc, P2.trt_acc)[0]); m("spearmanPosCov", spearmanr(P2.pos_acc, P2.std_trt_cov)[0])
m("nFlagged", int(P2.flagged.sum())); m("covFlagged", P2[P2.flagged].std_trt_cov.mean()); m("covUnflagged", P2[~P2.flagged].std_trt_cov.mean()); m("accFlagged", P2[P2.flagged].act_acc.mean()); m("accUnflagged", P2[~P2.flagged].act_acc.mean())
# baselines / ablations
for t in ["harmony", "negcon9", "global9", "allwell9", "logreg", "pcs64"]:
    try:
        Mt = pd.read_csv(f"{a.root}/{t}/metrics_per_fold.csv"); Ct = pd.read_csv(f"{a.root}/{t}/conformal_per_fold.csv"); ct = Ct[(Ct.score == a.score) & (Ct.alpha == 0.1)]
        T = t.replace("9", "Nine").replace("64", "Sixtyfour").capitalize()
        m(f"{T}nRuns", len(Mt)); m(f"{T}accTrtID", Mt.id_trt_acc.mean()); m(f"{T}accTrtOOD", Mt.ood_trt_acc.mean()); m(f"{T}accActID", Mt.id_act_acc.mean()); m(f"{T}accActOOD", Mt.ood_act_acc.mean())
        m(f"{T}eceActID", Mt.id_act_ece.mean()); m(f"{T}eceActOOD", Mt.ood_act_ece.mean()); m(f"{T}cpTrtIDcov", ct.id_trt_cov.mean()); m(f"{T}cpTrtStdCov", ct.ood_trt_cov_std.mean()); m(f"{T}cpActStdCov", ct.ood_act_cov_std.mean()); m(f"{T}cpActStdSize", ct.ood_act_size_std.mean(), 1)
        m(f"{T}cpActVtwoCov", ct.ood_act_cov_v2_match.mean()); m(f"{T}cpActOracleCov", ct.ood_act_cov_oracle.mean()); m(f"{T}posAccOOD", Mt.ood_pos_acc.mean()); m(f"{T}aurcActOOD", Mt.ood_act_aurc.mean())
    except FileNotFoundError:
        pass

# paired tests across runs (ID vs OOD), Wilcoxon signed-rank
from scipy.stats import wilcoxon
def pw(name, a_, b_):
    ok = a_.notna() & b_.notna(); a_, b_ = a_[ok], b_[ok]
    try: pv = wilcoxon(a_, b_).pvalue
    except ValueError: pv = float("nan")
    mac[name] = "$<$0.001" if pv < 0.001 else (f"{pv:.3f}" if pv == pv else "n/a")
pw("pAccAct", M.id_act_acc, M.ood_act_acc); pw("pEceAct", M.id_act_ece, M.ood_act_ece); pw("pAurcAct", M.id_act_aurc, M.ood_act_aurc); pw("pPosAcc", M.id_pos_acc, M.ood_pos_acc)
x10 = C[(C.score == a.score) & (C.alpha == 0.1)]
pw("pCovAct", x10.id_act_cov, x10.ood_act_cov_std); pw("pCovTrt", x10.id_trt_cov, x10.ood_trt_cov_std)
if "ood_act_cov_v5_actmond" in x10 and x10.ood_act_cov_v5_actmond.notna().any(): pw("pCovActVfiveVsStd", x10.ood_act_cov_v5_actmond, x10.ood_act_cov_std)
pw("pCovActVoneVsStd", x10.ood_act_cov_v1_poscon, x10.ood_act_cov_std)
s10 = S[S.target_risk == 0.1]; pw("pSelRisk", s10.id_risk, s10.ood_risk); pw("pSelCanary", s10.ood_risk, s10.ood_canary_risk)

# temperature-scaling baseline (same runs, TS probabilities)
try:
    Mts = pd.read_csv(f"{a.root}/main_ts/metrics_per_fold.csv"); Cts = pd.read_csv(f"{a.root}/main_ts/conformal_per_fold.csv"); cts = Cts[(Cts.score == a.score) & (Cts.alpha == 0.1)]
    m("tsEceActID", Mts.id_act_ece.mean()); m("tsEceActOOD", Mts.ood_act_ece.mean()); m("tsNllActOOD", Mts.ood_act_nll.mean(), 2)
    m("tsCpTrtStdCov", cts.ood_trt_cov_std.mean()); m("tsCpActStdCov", cts.ood_act_cov_std.mean())
except FileNotFoundError:
    pass

# anchor-target gap and activity-split robustness (from per-well outputs)
try:
    A = pd.read_csv(f"{a.root}/anchor_gap.csv"); B = pd.read_csv(f"{a.root}/activity_shift.csv")
    m("ksPosVsActive", A.ks_pos_vs_active.mean()); m("ksPosVsLooksActive", A.ks_pos_vs_looksactive.mean()); m("ksDmsoVsLooksInactive", A.ks_dmso_vs_looksinactive.mean()); m("ksPosVsAllTrt", A.ks_pos_vs_alltrt.mean())
    m("gapVoneActive", A.gap_v1_active.mean()); m("gapPosLooksActive", A.gap_posanchor_looksactive.mean()); m("gapDmsoLooksInactive", A.gap_dmsoanchor_looksinactive.mean())
    m("boundOkVone", int(A.bound_ok_v1.sum())); m("boundOkLooksActive", int(A.bound_ok_looksactive.sum())); m("boundOkLooksInactive", int(A.bound_ok_looksinactive.sum())); m("nAnchorRuns", len(A))
    bo, bi = B[B.split == "ood"], B[B.split == "idt"]
    m("actLookActiveID", bi.frac_active_wells_look_active.mean()); m("actLookActiveOOD", bo.frac_active_wells_look_active.mean())
    m("inactLookActiveID", bi.frac_inactive_wells_look_active.mean()); m("inactLookActiveOOD", bo.frac_inactive_wells_look_active.mean())
    m("dmsoLookActiveOOD", bo.frac_dmso_look_active.mean()); m("posLookActiveOOD", bo.frac_pos_look_active.mean())
    gb = bo.groupby("holdout").frac_active_wells_look_active.mean(); m("actLookActiveOODmin", gb.min()); m("labActLookActiveOODmin", gb.idxmin().replace("source_", "L")); m("actLookActiveOODmax", gb.max())
except FileNotFoundError:
    pass

# 99th-percentile activity threshold ablation
try:
    Mt = pd.read_csv(f"{a.root}/pct99/metrics_per_fold.csv"); Ct = pd.read_csv(f"{a.root}/pct99/conformal_per_fold.csv"); ct = Ct[(Ct.score == a.score) & (Ct.alpha == 0.1)]
    m("PctnninetynineNactive", f"{(Mt.K - 1).mean():.0f}"); m("PctnninetynineaccActID", Mt.id_act_acc.mean()); m("PctnninetynineaccActOOD", Mt.ood_act_acc.mean())
    m("PctnninetynineeceActID", Mt.id_act_ece.mean()); m("PctnninetynineeceActOOD", Mt.ood_act_ece.mean()); m("PctnninetyninecpActStdCov", ct.ood_act_cov_std.mean()); m("PctnninetyninecpTrtStdCov", ct.ood_trt_cov_std.mean())
except FileNotFoundError:
    pass

# revision analyses: oracles, laboratory-clustered statistics, clustered canary (analysis_revision.py)
def fp(v): return "$<$0.001" if v < 0.001 else f"$=${v:.3f}"
TESTS = {"acc_act": "AccAct", "ece_act": "EceAct", "aurc_act": "AurcAct", "cov_act_id_vs_std": "CovIdStd", "cov_act_std_vs_v5": "CovStdVfive",
         "cov_act_std_vs_v1": "CovStdVone", "risk_id_vs_ood": "RiskIdOod", "risk_ood_vs_canary": "RiskOodCan", "cov_act_v5_vs_oracle_act": "CovVfiveOrAct", "cov_act_v1_vs_oracle_act": "CovVoneOrAct",
         "cov_act_std_vs_v6": "CovStdVsix", "cov_act_v1_vs_v6": "CovVoneVsix", "cov_act_v6_vs_oracle_act": "CovVsixOrAct", "size_act_v6_vs_oracle_at_v6": "SizeVsixOr", "risk_idthr_vs_platestd": "RiskSelStd"}
def revision(path, pre):
    R = json.load(open(path))
    for al, tag in [("0.05", "Five"), ("0.1", "Ten"), ("0.2", "Twenty")]:
        o = R["oracle_alpha"][al]
        m(f"{pre}cpActOrActcov{tag}", o["act_cov_oracle_act"]); m(f"{pre}cpActOrActsize{tag}", o["act_size_oracle_act"], 1)
        m(f"{pre}cpTrtOrActcov{tag}", o["trt_cov_oracle_act"]); m(f"{pre}cpTrtOrActsize{tag}", o["trt_size_oracle_act"], 1)
        m(f"{pre}cpActOrActXfcov{tag}", o["act_cov_oracle_act_xf"]); m(f"{pre}cpActOrActXfsize{tag}", o["act_size_oracle_act_xf"], 1)
        m(f"{pre}cpActOrMarXfcov{tag}", o["act_cov_oracle_mar_xf"]); m(f"{pre}cpTrtOrMarXfcov{tag}", o["trt_cov_oracle_mar_xf"])
        m(f"{pre}orSizeAtVone{tag}", o["act_size_oracle_at_v1"], 1); m(f"{pre}orSizeAtVfive{tag}", o["act_size_oracle_at_v5"], 1)
        if "act_cov_v6" in o:  # V6 plate-standardised CP
            m(f"{pre}cpActVsixcov{tag}", o["act_cov_v6"]); m(f"{pre}cpActVsixsize{tag}", o["act_size_v6"], 1); m(f"{pre}cpTrtVsixcov{tag}", o["trt_cov_v6"]); m(f"{pre}cpTrtVsixsize{tag}", o["trt_size_v6"], 1)
            m(f"{pre}cpActIDcovVsix{tag}", o["id_act_cov_v6"]); m(f"{pre}cpActIDsizeVsix{tag}", o["id_act_size_v6"], 1); m(f"{pre}orSizeAtVsix{tag}", o["act_size_oracle_at_v6"], 1)
            m(f"{pre}cpActVsixUnflagcov{tag}", o["act_cov_v6_unflagged"])
    m(f"{pre}cpActOrActsdTen", R["oracle_alpha_sd"]["0.1"]["act_cov_oracle_act"])
    if "act_cov_v6" in R["oracle_alpha_sd"]["0.1"]: m(f"{pre}cpActVsixsdTen", R["oracle_alpha_sd"]["0.1"]["act_cov_v6"])
    for k, nm in TESTS.items():
        t = R["lab_tests"][k]
        m(f"{pre}lab{nm}A", t["mean_a"]); m(f"{pre}lab{nm}B", t["mean_b"]); m(f"{pre}lab{nm}Diff", t["diff"]); m(f"{pre}lab{nm}Lo", t["diff_ci"][0]); m(f"{pre}lab{nm}Hi", t["diff_ci"][1])
        m(f"{pre}lab{nm}Nup", t["n_labs_b_gt_a"]); m(f"{pre}lab{nm}Ndown", t["n_labs_b_lt_a"]); m(f"{pre}lab{nm}N", t["n_labs"]); mac[f"{pre}lab{nm}P"] = fp(t["p_lab_wilcoxon"]) if t["p_lab_wilcoxon"] == t["p_lab_wilcoxon"] else "n/a"
        m(f"{pre}lab{nm}ANoNine", t["mean_a_no9"]); m(f"{pre}lab{nm}BNoNine", t["mean_b_no9"]); m(f"{pre}lab{nm}LoNoNine", t["diff_ci_no9"][0]); m(f"{pre}lab{nm}HiNoNine", t["diff_ci_no9"][1]); mac[f"{pre}lab{nm}PNoNine"] = fp(t["p_lab_wilcoxon_no9"])
        m(f"{pre}lab{nm}MedA", t["median_a"]); m(f"{pre}lab{nm}MedB", t["median_b"])
    c = R["canary"]
    m(f"{pre}rhoPlate", c["rho_plate_seedavg"], 2); m(f"{pre}rhoPlateLo", c["rho_ci_labboot"][0], 2); m(f"{pre}rhoPlateHi", c["rho_ci_labboot"][1], 2); m(f"{pre}rhoPlateNoNine", c["rho_plate_seedavg_no9"], 2)
    m(f"{pre}rhoCovPlate", c["rho_cov_plate_seedavg"], 2); m(f"{pre}rhoWithinPooled", c["rho_within_lab_pooled"], 2); m(f"{pre}rhoWithinMedian", c["rho_within_lab_median"], 2)
    m(f"{pre}nLabsWithinPos", c["n_labs_within_pos"]); m(f"{pre}nLabsWithin", sum(v == v for v in c["within_lab"].values())); m(f"{pre}rhoLab", c["rho_lab_level"], 2); m(f"{pre}rhoLabNoNine", c["rho_lab_level_no9"], 2)
    m(f"{pre}nPlatesSeedAvg", c["n_plates"]); m(f"{pre}nFlaggedNoNine", c["n_flagged_no9"]); m(f"{pre}nEvalsNoNine", c["n_evals_no9"]); m(f"{pre}nLabsWithFlag", c["n_labs_with_flag"])
    m(f"{pre}accFlaggedNoNine", c["acc_flagged_no9"]); m(f"{pre}accUnflaggedNoNine", c["acc_unflagged_no9"]); m(f"{pre}nFlaggedLnine", c["flagged_per_lab"].get("source_9", 0))
try: revision(f"{a.root}/revision_main/revision.json", "")
except FileNotFoundError: pass
try: revision(f"{a.root}/revision_pcs64/revision.json", "Sf")
except FileNotFoundError: pass
s10n = S[(S.target_risk == 0.1) & (S.holdout != "source_9")]
m("selOODriskTenNoNine", s10n.ood_risk.mean()); m("selCanRiskTenNoNine", s10n.ood_canary_risk.mean()); m("selCanCovTenNoNine", s10n.ood_canary_cov.mean())
c9 = C[(C.score == a.score) & (C.alpha == 0.1) & (C.holdout == "source_9")]; m("covVfiveLnine", c9.ood_act_cov_v5_actmond.mean()); m("covVoneLnine", c9.ood_act_cov_v1_poscon.mean())
s9 = S[(S.target_risk == 0.1) & (S.holdout == "source_9")]; pct("fracNamedLnine", s9.frac_named_ood.mean(), 1); m("selCanNnanTen", int(S[S.target_risk == 0.1].ood_canary_risk.isna().sum()))
# PCA width chosen on in-distribution calibration NLL only (deployable model selection)
for t, T in [("main", ""), ("pcs64full", "Sf")]:
    try:
        Mt = pd.read_csv(f"{a.root}/{t}/metrics_per_fold.csv")
        if "cal_nll" in Mt: m(f"{T}calNll", Mt.cal_nll.mean())
    except FileNotFoundError: pass
try:
    A_, B_ = pd.read_csv(f"{a.root}/main/metrics_per_fold.csv"), pd.read_csv(f"{a.root}/pcs64full/metrics_per_fold.csv")
    j = A_.merge(B_, on=["holdout", "seed"], suffixes=("", "_b")); m("nCalNllSfBetter", int((j.cal_nll_b < j.cal_nll).sum())); m("nCalNllDefBetter", int((j.cal_nll_b > j.cal_nll).sum())); m("nCalNllPairs", len(j))
except (FileNotFoundError, KeyError): pass

# model-side repairs (seed 0, all labs; screen_compare.py --out summary/repair)
try:
    T = pd.read_csv(f"{a.root}/repair/tag_means.csv", index_col=0); Tj = json.load(open(f"{a.root}/repair/tests.json"))["tests"]
    TG = {"_base": "Base", "_ens5": "Ens", "_aug3": "Aug", "_aug5": "Augf", "_aug3ens5": "AugEns", "_sphere": "Sph", "_alignlab": "Align"}
    MT = {"acc_act": "Acc", "ece_act": "Ece", "aurc_act": "Aurc", "act_std": "Std", "act_v1_poscon": "Vone", "actsz_oracle_act": "OrSz", "risk10_can": "Can", "risk10": "Risk"}
    for t, tn in TG.items():
        if t not in T.index: continue
        for c, cn in MT.items():
            if c in T.columns: m(f"rep{tn}{cn}", T.loc[t, c], 1 if cn == "OrSz" else 3)
            k = f"{t}|{c}"
            if k in Tj:
                m(f"rep{tn}{cn}Nup", Tj[k]["n_up"]); m(f"rep{tn}{cn}Ndown", Tj[k]["n_down"]); mac[f"rep{tn}{cn}P"] = fp(Tj[k]["p"]) if Tj[k]["p"] == Tj[k]["p"] else "n/a"
except FileNotFoundError: pass

# 3-seed confirmation of the selected model-side repair (augmentation + ensemble) against the default (summary/repair3)
try:
    T = pd.read_csv(f"{a.root}/repair3/tag_means.csv", index_col=0); Tj = json.load(open(f"{a.root}/repair3/tests.json"))["tests"]
    MT = {"acc_act_id": "AccId", "ece_act_id": "EceId", "acc_act": "Acc", "ece_act": "Ece", "aurc_act": "Aurc", "act_std": "Std", "act_v1_poscon": "Vone", "act_v6_platestd": "Vsix", "actsz_v6_platestd": "VsixSz",
          "actsz_oracle_act": "OrSz", "risk10": "Risk", "risk10_can": "Can", "cal_nll": "CalNll", "rho_plate": "Rho"}
    for t, tn in [("_base", "Base"), ("_aug3ens5", "Rob")]:
        for c, cn in MT.items():
            if c in T.columns and t in T.index: m(f"rpt{cn}{tn}", T.loc[t, c], 1 if cn in ("OrSz", "VsixSz") else (2 if cn == "Rho" else 3))
    for c, cn in MT.items():
        k = f"_aug3ens5|{c}"
        if k in Tj:
            m(f"rpt{cn}Nup", Tj[k]["n_up"]); m(f"rpt{cn}Ndown", Tj[k]["n_down"]); m(f"rpt{cn}N", Tj[k]["n"]); mac[f"rpt{cn}P"] = fp(Tj[k]["p"]) if Tj[k]["p"] == Tj[k]["p"] else "n/a"
except FileNotFoundError: pass

lines = ["% AUTO-GENERATED by experiments/make_numbers.py — do not edit by hand"]
for k, v in mac.items():
    lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
open(a.out, "w").write("\n".join(lines) + "\n"); print(len(mac), "macros ->", a.out)
