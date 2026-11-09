#!/usr/bin/env python3
"""Post-hoc analyses from per-well outputs (loso_v5/*_wells.npz):
  (1) anchor-target distribution gap: total-variation / KS distance between the score distributions of the anchor wells
      (positive controls; DMSO) and the target wells they calibrate (active-looking treatment wells; inactive-looking),
      and the coverage-gap bound of Barber et al. (2023): |cov - (1-alpha)| <= d_TV(anchor, target).
  (2) robustness of the training-plate activity split under lab shift: on unseen-lab plates, fraction of wells of
      training-ACTIVE compounds that look active (pheno > thr) and fraction of training-INACTIVE compound wells that look active.
Writes experiments/results/summary/anchor_gap.csv and activity_shift.csv
"""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
ROOT = os.environ.get("TRUST_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D = f"{ROOT}/experiments/results/loso_v5"; OUT = f"{ROOT}/experiments/results/summary"
ALPHA = 0.1

def nlp(P, y): return -np.log(np.clip(P[np.arange(len(y)), y].astype(np.float64), 1e-12, 1))
def quantile(s, alpha):
    n = len(s); lvl = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(s, lvl, method="higher")) if lvl <= 1 else float("inf")
def tv_dist(a, b, grid=None):
    grid = np.unique(np.r_[a, b]) if grid is None else grid
    Fa = np.searchsorted(np.sort(a), grid, side="right") / len(a); Fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.abs(Fa - Fb).max())  # Kolmogorov distance (lower bound on TV between score laws)

rows, arows = [], []
for f in sorted(glob.glob(f"{D}/*_wells.npz")):
    z = np.load(f, allow_pickle=True); r = json.load(open(f.replace("_wells.npz", ".json")))
    ho, seed = r["holdout"], r["seed"]; classes = z["classes"]; inact = int(np.where(classes == "inactive")[0][0]); thr = float(z["dmso_thr"])
    P, y, pos, dm, ph = z["ood_P"].astype(np.float32), z["ood_y"], z["ood_pos"], z["ood_dmso"], z["ood_pheno"]
    trt = ~pos & ~dm; act = trt & (y != inact); inactrt = trt & (y == inact); looks = ph > thr
    s = nlp(P, y)
    # anchor-target gaps on the unseen lab
    d_pos_act = ks_2samp(s[pos], s[act]).statistic if act.sum() and pos.sum() else np.nan
    d_pos_looksact = ks_2samp(s[pos], s[trt & looks]).statistic if (trt & looks).sum() else np.nan
    d_dm_looksinact = ks_2samp(s[dm], s[trt & ~looks]).statistic if (trt & ~looks).sum() else np.nan
    d_pos_trt = ks_2samp(s[pos], s[trt]).statistic
    # realised coverage gaps for V1 (poscon anchor on active wells) and V5 pieces
    q_pos = quantile(s[pos], ALPHA); q_dm = quantile(s[dm], ALPHA)
    cov_v1_act = float((s[act] <= q_pos).mean()); cov_pos_looksact = float((s[trt & looks] <= q_pos).mean()) if (trt & looks).sum() else np.nan
    cov_dm_looksinact = float((s[trt & ~looks] <= q_dm).mean()) if (trt & ~looks).sum() else np.nan
    rows.append({"holdout": ho, "seed": seed, "ks_pos_vs_active": d_pos_act, "ks_pos_vs_looksactive": d_pos_looksact, "ks_dmso_vs_looksinactive": d_dm_looksinact, "ks_pos_vs_alltrt": d_pos_trt,
                 "cov_v1_active": cov_v1_act, "gap_v1_active": cov_v1_act - (1 - ALPHA), "cov_posanchor_looksactive": cov_pos_looksact, "gap_posanchor_looksactive": cov_pos_looksact - (1 - ALPHA),
                 "cov_dmsoanchor_looksinactive": cov_dm_looksinact, "gap_dmsoanchor_looksinactive": cov_dm_looksinact - (1 - ALPHA),
                 "bound_ok_v1": abs(cov_v1_act - (1 - ALPHA)) <= d_pos_act + 1.0 / max(pos.sum(), 1), "bound_ok_looksactive": abs(cov_pos_looksact - (1 - ALPHA)) <= d_pos_looksact + 1.0 / max(pos.sum(), 1),
                 "bound_ok_looksinactive": abs(cov_dm_looksinact - (1 - ALPHA)) <= d_dm_looksinact + 1.0 / max(dm.sum(), 1)})
    # activity split under shift (OOD) vs in-distribution (idt)
    for split in ["ood", "idt"]:
        P_, y_, pos_, dm_, ph_ = z[f"{split}_P"], z[f"{split}_y"], z[f"{split}_pos"], z[f"{split}_dmso"], z[f"{split}_pheno"]
        trt_ = ~pos_ & ~dm_; act_ = trt_ & (y_ != inact); ina_ = trt_ & (y_ == inact); lk = ph_ > thr
        arows.append({"holdout": ho, "seed": seed, "split": split, "frac_active_wells_look_active": float(lk[act_].mean()), "frac_inactive_wells_look_active": float(lk[ina_].mean()),
                      "frac_dmso_look_active": float(lk[dm_].mean()), "frac_pos_look_active": float(lk[pos_].mean()), "n_act": int(act_.sum())})
A = pd.DataFrame(rows); B = pd.DataFrame(arows)
A.to_csv(f"{OUT}/anchor_gap.csv", index=False); B.to_csv(f"{OUT}/activity_shift.csv", index=False)
print(A.describe().loc[["mean", "std", "min", "max"]].T.round(3).to_string())
print(B.groupby("split").mean(numeric_only=True).round(3).to_string())
print("bound holds (v1 active):", int(A.bound_ok_v1.sum()), "/", len(A), "| looks-active:", int(A.bound_ok_looksactive.sum()), "| looks-inactive:", int(A.bound_ok_looksinactive.sum()))
