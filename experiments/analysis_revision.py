#!/usr/bin/env python3
"""Revision analyses from per-well outputs and per-fold summaries.

  1. oracles: marginal oracle (labelled unseen-lab treatment wells) vs active-calibrated oracle (labelled unseen-lab
     active-compound wells), in-sample and plate cross-fitted; oracle set size at the coverage V1/V5 reach.
  2. laboratory-clustered statistics: seeds averaged within laboratory, then 11 laboratory means -> bootstrap CI over
     laboratories, exact Wilcoxon signed-rank on paired laboratory means, sign counts, leave-L9-out means.
  3. canary: plate-level Spearman with seeds averaged per plate, laboratory-cluster bootstrap CI, without L9, within-lab.
  4. per-laboratory table of coverage (std, V1, V5, V6, oracles) and selective risk.
  5. V6 plate-standardised CP (score minus the plate's own positive-control conformal quantile, thresholded at the
     conformal quantile of that standardised score on CAL active-compound wells), on unseen-lab and ID-test plates.
Usage: analysis_revision.py --wells results/loso_v5 --summary results/summary/main --out results/summary/revision_main
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
ROOT = os.environ.get("TRUST_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--wells", default=f"{ROOT}/experiments/results/loso_v5")
ap.add_argument("--summary", default=f"{ROOT}/experiments/results/summary/main")
ap.add_argument("--out", default=f"{ROOT}/experiments/results/summary/revision_main")
ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ALPHAS = [0.05, 0.1, 0.2]

def nlp(P, y): return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))
def quantile(s, alpha):
    n = len(s)
    if n == 0: return float("inf")
    lvl = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(s, lvl, method="higher")) if lvl <= 1 else float("inf")
def plate_q(s, pos, pl, alpha):
    out = np.full(len(s), np.inf)
    for p in np.unique(pl):
        m = pl == p; out[m] = quantile(s[m & pos], alpha)
    return out
def cov_size(L, q, y):  # L = -log P, q scalar or per-well
    S = L <= np.broadcast_to(np.asarray(q, float).reshape(-1, 1) if np.ndim(q) else q, L.shape)
    return float(S[np.arange(len(y)), y].mean()), float(S.sum(1).mean())

# ---------------- 1. oracles from per-well outputs ----------------
rows = []
for f in sorted(glob.glob(f"{a.wells}/*_wells.npz")):
    ho, seed = re.match(r"(source_\d+)_s(\d+)", os.path.basename(f)).groups(); seed = int(seed)
    d = np.load(f, allow_pickle=True); inact = int(np.where(d["classes"] == "inactive")[0][0])
    Po = d["ood_P"].astype(np.float64); yo = d["ood_y"]; Lo = -np.log(np.clip(Po, 1e-12, 1)); so = nlp(Po, yo)
    trt = ~d["ood_pos"] & ~d["ood_dmso"]; act = trt & (yo != inact); pos = d["ood_pos"]; dm = d["ood_dmso"]
    Pc = d["cal_P"].astype(np.float64); yc = d["cal_y"]; sc = nlp(Pc, yc); ctrt = ~d["cal_pos"] & ~d["cal_dmso"]
    pl = d["ood_plate"].astype(str); plates = np.unique(pl); rng = np.random.default_rng(seed)
    cact = ctrt & (yc != inact); cpl = d["cal_plate"].astype(str)
    Pi = d["idt_P"].astype(np.float64); yi = d["idt_y"]; Li = -np.log(np.clip(Pi, 1e-12, 1)); si = nlp(Pi, yi)
    itrt = ~d["idt_pos"] & ~d["idt_dmso"]; iact = itrt & (yi != inact)
    perm = rng.permutation(plates); half = np.isin(pl, perm[: len(perm) // 2])
    looks = d["ood_pheno"] > float(d["dmso_thr"])
    # selective analogue of V6: confidence standardised by the median positive-control confidence of its own plate,
    # threshold for 10% risk among named calls chosen on CAL; compared with the plain ID threshold
    def named(P_, y_, pos_, dm_, pl_):
        pred = P_.argmax(1); conf = P_.max(1); nm = ~pos_ & ~dm_ & (pred != inact); loc = np.zeros(len(y_))
        for p_ in np.unique(pl_):
            m_ = pl_ == p_; loc[m_] = np.median(conf[m_ & pos_])
        return conf, pred != y_, nm, loc
    def thr_for_risk(c_, e_, risk):
        o_ = np.argsort(-c_); er = np.cumsum(e_[o_]) / np.arange(1, len(o_) + 1); ok_ = np.where(er <= risk)[0]
        return float(c_[o_][ok_.max()]) if len(ok_) else float("inf")
    cc, ce, cn, cl = named(Pc, yc, d["cal_pos"], d["cal_dmso"], cpl)
    oc, oe, on_, ol = named(Po, yo, pos, dm, pl)
    k1 = on_ & (oc >= thr_for_risk(cc[cn], ce[cn], 0.1)); k2 = on_ & ((oc - ol) >= thr_for_risk((cc - cl)[cn], ce[cn], 0.1))
    sel_std = {"sel_risk_idthr": float(oe[k1].mean()) if k1.any() else np.nan, "sel_risk_platestd": float(oe[k2].mean()) if k2.any() else np.nan,
               "sel_cov_platestd": float(k2.sum() / max(on_.sum(), 1))}
    for alpha in ALPHAS:
        r = {"holdout": ho, "seed": seed, "alpha": alpha, "n_act": int(act.sum()), "n_ood_plates": len(plates), **sel_std}
        q_std = quantile(sc[ctrt], alpha); q_v1 = quantile(so[pos], alpha)
        q_v5 = np.where(looks, q_v1, quantile(so[dm], alpha))
        for nm, q in [("std", q_std), ("v1", q_v1)]:
            r[f"act_cov_{nm}"], r[f"act_size_{nm}"] = cov_size(Lo[act], q, yo[act]); r[f"trt_cov_{nm}"], r[f"trt_size_{nm}"] = cov_size(Lo[trt], q, yo[trt])
        r["act_cov_v5"], r["act_size_v5"] = cov_size(Lo[act], q_v5[act], yo[act]); r["trt_cov_v5"], r["trt_size_v5"] = cov_size(Lo[trt], q_v5[trt], yo[trt])
        r["recomp_act_cov_v1"], r["recomp_act_cov_v5"], r["recomp_act_cov_std"] = r["act_cov_v1"], r["act_cov_v5"], r["act_cov_std"]
        # use the values logged at full precision by loso.py for the deployable rules (per-well outputs are stored as float16)
        e = json.load(open(f.replace("_wells.npz", ".json")))["conformal"][f"raw_nlp_a{alpha}"]
        for nm, v in [("std", "std"), ("v1", "v1_poscon"), ("v5", "v5_actmond")]:
            for g_ in ["act", "trt"]: r[f"{g_}_cov_{nm}"], r[f"{g_}_size_{nm}"] = e[f"ood_{g_}_cov_{v}"], e[f"ood_{g_}_size_{v}"]
        # in-sample oracles (labels of the very wells evaluated; non-deployable reference)
        for nm, grp in [("oracle_mar", trt), ("oracle_act", act)]:
            q = quantile(so[grp], alpha)
            r[f"act_cov_{nm}"], r[f"act_size_{nm}"] = cov_size(Lo[act], q, yo[act]); r[f"trt_cov_{nm}"], r[f"trt_size_{nm}"] = cov_size(Lo[trt], q, yo[trt])
        # plate cross-fitted oracles: calibrate on labelled wells of the other half of the unseen lab's plates
        for nm, grp in [("oracle_mar_xf", trt), ("oracle_act_xf", act)]:
            hit_a, sz_a, hit_t, sz_t = [], [], [], []
            for h in [half, ~half]:
                q = quantile(so[grp & ~h], alpha); ea, et = act & h, trt & h
                S = Lo <= q
                hit_a.append(S[ea][np.arange(ea.sum()), yo[ea]]); sz_a.append(S[ea].sum(1)); hit_t.append(S[et][np.arange(et.sum()), yo[et]]); sz_t.append(S[et].sum(1))
            r[f"act_cov_{nm}"], r[f"act_size_{nm}"] = float(np.concatenate(hit_a).mean()), float(np.concatenate(sz_a).mean())
            r[f"trt_cov_{nm}"], r[f"trt_size_{nm}"] = float(np.concatenate(hit_t).mean()), float(np.concatenate(sz_t).mean())
        # V6 plate-standardised CP (deployable: CAL labels + the deployed plates' positive-control identities)
        t6 = quantile((sc - plate_q(sc, d["cal_pos"], cpl, alpha))[cact], alpha); r["t_v6"] = t6
        q6 = plate_q(so, pos, pl, alpha) + t6
        r["act_cov_v6"], r["act_size_v6"] = cov_size(Lo[act], q6[act], yo[act]); r["trt_cov_v6"], r["trt_size_v6"] = cov_size(Lo[trt], q6[trt], yo[trt])
        q6i = plate_q(si, d["idt_pos"], d["idt_plate"].astype(str), alpha) + t6
        r["id_act_cov_v6"], r["id_act_size_v6"] = cov_size(Li[iact], q6i[iact], yi[iact]); r["id_trt_cov_v6"], r["id_trt_size_v6"] = cov_size(Li[itrt], q6i[itrt], yi[itrt])
        # V6 restricted to plates the canary gate passes (flagged plates are reported, not called)
        ok = ~d["flag_mask"].astype(bool)
        r["act_cov_v6_unflagged"] = cov_size(Lo[act & ok], q6[act & ok], yo[act & ok])[0] if (act & ok).any() else np.nan
        r["frac_act_unflagged"] = float((act & ok).sum() / max(act.sum(), 1))
        # efficiency: oracle set size on active wells at the coverage V1 / V5 / V6 actually reach
        for nm in ["v1", "v5", "v6"]:
            c = r[f"act_cov_{nm}"]; q = float(np.quantile(so[act], min(1.0, c), method="higher"))
            r[f"act_size_oracle_at_{nm}"] = cov_size(Lo[act], q, yo[act])[1]
        rows.append(r)
O = pd.DataFrame(rows); O.to_csv(f"{a.out}/oracles_per_fold.csv", index=False)

# check against the per-fold JSON-derived summary (float16 storage -> tiny differences allowed)
C = pd.read_csv(f"{a.summary}/conformal_per_fold.csv"); C = C[C.score == "nlp"]
chk = O.merge(C, on=["holdout", "seed", "alpha"])
dev = {k: float((chk[k1] - chk[k2]).abs().max()) for k, k1, k2 in [("std", "recomp_act_cov_std", "ood_act_cov_std"), ("v1", "recomp_act_cov_v1", "ood_act_cov_v1_poscon"), ("v5", "recomp_act_cov_v5", "ood_act_cov_v5_actmond"), ("oracle", "trt_cov_oracle_mar", "ood_trt_cov_oracle"), ("oracle_mean", "trt_cov_oracle_mar", "ood_trt_cov_oracle")]}
dev["oracle_mean"] = float((chk.trt_cov_oracle_mar - chk.ood_trt_cov_oracle).mean())
print("max |recomputed - logged| coverage:", dev)

# ---------------- 2. laboratory-clustered statistics ----------------
M = pd.read_csv(f"{a.summary}/metrics_per_fold.csv"); S = pd.read_csv(f"{a.summary}/selective_per_fold.csv"); PC = pd.read_csv(f"{a.summary}/plate_canary.csv")
o10 = O[O.alpha == 0.1]
lab = M.groupby("holdout").mean(numeric_only=True).join(o10.groupby("holdout").mean(numeric_only=True), rsuffix="_o")
s10 = S[S.target_risk == 0.1].groupby("holdout").mean(numeric_only=True)
lab = lab.join(s10[["id_risk", "ood_risk", "ood_canary_risk", "ood_oracle_risk", "ood_cov", "ood_canary_cov"]])
cid = C[C.alpha == 0.1].groupby("holdout").mean(numeric_only=True)[["id_act_cov", "id_trt_cov"]]
lab = lab.join(cid)
B = np.random.default_rng(0).integers(0, len(lab), (a.boot, len(lab)))
def boot_ci(v):
    v = np.asarray(v, float); bm = v[B].mean(1); return [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
def paired(name, x, y):
    x, y = lab[x].to_numpy(float), lab[y].to_numpy(float); dlt = y - x
    ok = np.isfinite(dlt); p = float(wilcoxon(x[ok], y[ok]).pvalue) if ok.all() else float("nan")
    no9 = (lab.index != "source_9") & ok
    return {"name": name, "mean_a": float(x.mean()), "mean_b": float(y.mean()), "diff": float(dlt.mean()), "diff_ci": boot_ci(dlt),
            "ci_a": boot_ci(x), "ci_b": boot_ci(y), "p_lab_wilcoxon": p, "n_labs_b_gt_a": int((dlt[ok] > 0).sum()), "n_labs_b_lt_a": int((dlt[ok] < 0).sum()), "n_labs": int(ok.sum()),
            "median_a": float(np.median(x[ok])), "median_b": float(np.median(y[ok])),
            "mean_a_no9": float(x[no9].mean()), "mean_b_no9": float(y[no9].mean()), "diff_ci_no9": [float(np.percentile(dlt[no9][np.random.default_rng(1).integers(0, no9.sum(), (a.boot, no9.sum()))].mean(1), q)) for q in (2.5, 97.5)],
            "p_lab_wilcoxon_no9": float(wilcoxon(x[no9], y[no9]).pvalue)}
tests = [paired("acc_act", "id_act_acc", "ood_act_acc"), paired("ece_act", "id_act_ece", "ood_act_ece"), paired("aurc_act", "id_act_aurc", "ood_act_aurc"),
         paired("cov_act_id_vs_std", "id_act_cov", "act_cov_std"), paired("cov_act_std_vs_v5", "act_cov_std", "act_cov_v5"), paired("cov_act_std_vs_v1", "act_cov_std", "act_cov_v1"),
         paired("risk_id_vs_ood", "id_risk", "ood_risk"), paired("risk_ood_vs_canary", "ood_risk", "ood_canary_risk"),
         paired("cov_act_v5_vs_oracle_act", "act_cov_v5", "act_cov_oracle_act"), paired("cov_act_v1_vs_oracle_act", "act_cov_v1", "act_cov_oracle_act"),
         paired("cov_act_std_vs_v6", "act_cov_std", "act_cov_v6"), paired("cov_act_v1_vs_v6", "act_cov_v1", "act_cov_v6"), paired("cov_act_v6_vs_oracle_act", "act_cov_v6", "act_cov_oracle_act"),
         paired("size_act_v6_vs_oracle_at_v6", "act_size_v6", "act_size_oracle_at_v6"), paired("risk_idthr_vs_platestd", "sel_risk_idthr", "sel_risk_platestd")]
lab.to_csv(f"{a.out}/per_lab.csv")

# ---------------- 3. canary with clustering ----------------
P1 = PC.groupby(["holdout", "plate"]).mean(numeric_only=True).reset_index()   # seeds averaged per plate
labs = np.array(sorted(P1.holdout.unique()))
def rho(df, y="act_acc"): return float(spearmanr(df.pos_acc, df[y])[0])
from scipy.stats import rankdata
grp = [P1[P1.holdout == l][["pos_acc", "act_acc"]].to_numpy() for l in labs]; Bc = np.random.default_rng(0).integers(0, len(labs), (a.boot, len(labs)))
br = []
for idx in Bc:
    Z = np.vstack([grp[i] for i in idx]); r0 = np.corrcoef(rankdata(Z[:, 0]), rankdata(Z[:, 1]))[0, 1]; br.append(r0)
within = {l: rho(P1[P1.holdout == l]) for l in labs if (P1.holdout == l).sum() >= 4}
dm_ = P1.copy(); dm_[["pos_acc", "act_acc"]] = dm_.groupby("holdout")[["pos_acc", "act_acc"]].rank(); dm_[["pos_acc", "act_acc"]] -= dm_.groupby("holdout")[["pos_acc", "act_acc"]].transform("mean")
gate = PC.groupby("holdout").agg(n=("flagged", "size"), flagged=("flagged", "sum"))
PCn9 = PC[PC.holdout != "source_9"]
canary = {"n_plates": int(len(P1)), "rho_plate_seedavg": rho(P1), "rho_ci_labboot": [float(np.nanpercentile(br, 2.5)), float(np.nanpercentile(br, 97.5))],
          "rho_plate_seedavg_no9": rho(P1[P1.holdout != "source_9"]), "rho_cov_plate_seedavg": rho(P1, "std_trt_cov"),
          "rho_within_lab_pooled": float(spearmanr(dm_.pos_acc, dm_.act_acc)[0]), "rho_within_lab_median": float(np.nanmedian(list(within.values()))),
          "n_labs_within_pos": int(sum(v > 0 for v in within.values())), "n_labs_within": len(within), "within_lab": within,
          "rho_lab_level": float(spearmanr(lab.ood_pos_acc, lab.ood_act_acc)[0]),
          "rho_lab_level_no9": float(spearmanr(lab.ood_pos_acc[lab.index != "source_9"], lab.ood_act_acc[lab.index != "source_9"])[0]),
          "flagged_per_lab": gate.flagged.astype(int).to_dict(), "n_flagged_no9": int(PCn9.flagged.sum()), "n_evals_no9": int(len(PCn9)),
          "acc_flagged_no9": float(PCn9[PCn9.flagged].act_acc.mean()) if PCn9.flagged.any() else None, "acc_unflagged_no9": float(PCn9[~PCn9.flagged].act_acc.mean()),
          "n_labs_with_flag": int((gate.flagged > 0).sum())}

summ = {"oracle_alpha": {str(al): O[O.alpha == al].drop(columns=["holdout", "seed", "alpha"]).mean().to_dict() for al in ALPHAS},
        "oracle_alpha_sd": {str(al): O[O.alpha == al].drop(columns=["holdout", "seed", "alpha"]).std().to_dict() for al in ALPHAS},
        "check_max_dev": dev, "lab_tests": {t["name"]: t for t in tests}, "canary": canary}
json.dump(summ, open(f"{a.out}/revision.json", "w"), indent=1)
print(json.dumps({k: summ[k] for k in ["check_max_dev"]}, indent=1))
print(O[O.alpha == 0.1][[c for c in O.columns if c.startswith("act_") or c.startswith("trt_")]].mean().round(3).to_string())
for t in tests: print(f"{t['name']:28s} {t['mean_a']:.3f} -> {t['mean_b']:.3f}  diff CI {t['diff_ci'][0]:+.3f},{t['diff_ci'][1]:+.3f}  p={t['p_lab_wilcoxon']:.4f}  {t['n_labs_b_gt_a']}/{t['n_labs']} up  | no L9 {t['mean_a_no9']:.3f}->{t['mean_b_no9']:.3f} CI {t['diff_ci_no9'][0]:+.3f},{t['diff_ci_no9'][1]:+.3f} p={t['p_lab_wilcoxon_no9']:.4f}")
print(json.dumps({k: v for k, v in canary.items() if k != "within_lab"}, indent=1)); print("within-lab rho:", {k: round(v, 2) for k, v in within.items()})
