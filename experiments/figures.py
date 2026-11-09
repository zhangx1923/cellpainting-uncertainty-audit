#!/usr/bin/env python3
"""Paper figures from aggregated summaries. Usage: figures.py --summary experiments/results/summary/main --out paper/figures"""
import argparse, os, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.environ.get("TRUST_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--summary", default=f"{ROOT}/experiments/results/summary/main")
ap.add_argument("--out", default=f"{ROOT}/paper/figures")
ap.add_argument("--score", default="nlp")
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 7.5, "axes.titlepad": 3, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
C_ID, C_OOD, C_ACC = "#2563EB", "#DC2626", "#6B7280"
M = pd.read_csv(f"{args.summary}/metrics_per_fold.csv"); C = pd.read_csv(f"{args.summary}/conformal_per_fold.csv")
S = pd.read_csv(f"{args.summary}/selective_per_fold.csv"); P = pd.read_csv(f"{args.summary}/plate_canary.csv")
try:  # labelled oracles recomputed from per-well outputs (analysis_revision.py)
    O = pd.read_csv(f"{os.path.dirname(args.summary)}/revision_{os.path.basename(args.summary)}/oracles_per_fold.csv")
    O = O.rename(columns={"act_cov_oracle_act": "ood_act_cov_oracle_act", "trt_cov_oracle_act": "ood_trt_cov_oracle_act", "act_size_oracle_act": "ood_act_size_oracle_act", "trt_size_oracle_act": "ood_trt_size_oracle_act",
                          "act_cov_v6": "ood_act_cov_v6_platestd", "trt_cov_v6": "ood_trt_cov_v6_platestd", "act_size_v6": "ood_act_size_v6_platestd", "trt_size_v6": "ood_trt_size_v6_platestd"})
    C = C.merge(O[["holdout", "seed", "alpha"] + [c for c in O.columns if c.startswith("ood_")]].assign(score=args.score), on=["holdout", "seed", "alpha", "score"], how="left")
except FileNotFoundError: pass
labs = sorted(M.holdout.unique(), key=lambda s: int(s.split("_")[1])); lab_short = [s.replace("source_", "L") for s in labs]
g = M.groupby("holdout"); gm = g.mean(numeric_only=True).loc[labs]; gs = g.std(numeric_only=True).loc[labs]

# ---- Fig 2: ID vs OOD trust metrics per lab ----
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.75))
panels = [("id_act_acc", "ood_act_acc", "Accuracy (active compounds)"), ("id_act_ece", "ood_act_ece", "ECE (active compounds)"),
          ("id_act_aurc", "ood_act_aurc", "AURC (active compounds)"), ("id_pos_acc", "ood_pos_acc", "Positive-control accuracy")]
x = np.arange(len(labs))
for ax, (a, b, title) in zip(axes, panels):
    ax.vlines(x, gm[a], gm[b], color=C_ACC, lw=1, zorder=1)
    ax.scatter(x, gm[a], s=14, color=C_ID, label="in-distribution plates", zorder=3)
    ax.scatter(x, gm[b], s=14, color=C_OOD, marker="s", label="unseen lab", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(lab_short, rotation=60); ax.set_title(title); ax.grid(axis="y", lw=0.3, alpha=0.5)
axes[0].legend(loc="upper right", frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1.02)); axes[0].set_ylim(-0.02, 1.0); axes[1].set_ylim(0, 1.0); axes[2].set_ylim(0, 1.02); axes[3].set_ylim(0.2, 1.0)
fig.tight_layout(w_pad=0.6); fig.savefig(f"{args.out}/fig_audit.pdf"); plt.close(fig)

# ---- Fig 3: conformal coverage per lab, std vs variants, alpha=0.1, treatment wells and active wells ----
c = C[(C.score == args.score) & (C.alpha == 0.1)].groupby("holdout").mean(numeric_only=True).loc[labs]
variants = [("std", "standard split CP", "#111827", "o"), ("v2_match", "V2 control-coverage matching", "#059669", "^"), ("v5_actmond", "V5 activity-split Mondrian", "#7C3AED", "D"),
            ("v1_poscon", "V1 positive-control Mondrian", "#D97706", "s"), ("v6_platestd", "V6 plate-standardised CP", "#DC2626", "v"), ("oracle", "oracle, marginal (labelled)", "#9CA3AF", "x"), ("oracle_act", "oracle, active-calibrated (labelled)", "#9CA3AF", "+")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05), sharey=True)
for ax, grp, title in zip(axes, ["trt", "act"], ["All treatment wells (deployment target)", "Active-compound wells (training-defined label)"]):
    ax.axhline(0.9, color="k", lw=0.8, ls="--"); ax.axhspan(0.88, 0.92, color="k", alpha=0.05)
    ax.plot(x, c[f"id_{grp}_cov"], color=C_ID, lw=1, ls=":", label="std CP, in-distribution plates")
    for v, name, col, mk in variants:
        if f"ood_{grp}_cov_{v}" in c and c[f"ood_{grp}_cov_{v}"].notna().any():
            ax.plot(x, c[f"ood_{grp}_cov_{v}"], color=col, marker=mk, ms=3.5, lw=1, label=name if "oracle" in name else name + " (unseen lab)")
    ax.set_xticks(x); ax.set_xticklabels(lab_short, rotation=60); ax.set_title(title); ax.grid(axis="y", lw=0.3, alpha=0.5)
axes[0].set_ylabel("empirical coverage (α = 0.1)"); axes[0].set_ylim(0, 1.02)
h, l = axes[1].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(w_pad=0.6, rect=(0, 0.14, 1, 1)); fig.savefig(f"{args.out}/fig_coverage.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Fig 4: coverage vs alpha and set size (mean over folds) ----
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.75), gridspec_kw={"width_ratios": [1, 1, 1, 0.75]})
alphas = sorted(C.alpha.unique())
for ax, grp, title in zip(axes[:2], ["trt", "act"], ["coverage: treatment wells", "coverage: active wells"]):
    ax.plot([1 - a for a in alphas], [1 - a for a in alphas], "k--", lw=0.8, label="nominal")
    for v, name, col, mk in variants:
        col_ = f"ood_{grp}_cov_{v}"
        if col_ in C and C[col_].notna().any():
            y = [C[(C.score == args.score) & (C.alpha == a)][col_].mean() for a in alphas]
            ax.plot([1 - a for a in alphas], y, color=col, marker=mk, ms=3.5, lw=1, label=name)
    ax.set_xlabel("nominal coverage 1−α"); ax.set_title(title); ax.grid(lw=0.3, alpha=0.5)
axes[0].set_ylabel("coverage, unseen lab")
ax = axes[2]
for v, name, col, mk in variants:
    col_ = f"ood_act_size_{v}"
    if col_ in C and C[col_].notna().any():
        y = [C[(C.score == args.score) & (C.alpha == a)][col_].mean() for a in alphas]
        ax.plot([1 - a for a in alphas], y, color=col, marker=mk, ms=3.5, lw=1, label=name)
ax.set_xlabel("nominal coverage 1−α"); ax.set_title("set size: active wells"); ax.grid(lw=0.3, alpha=0.5)
h, l = axes[2].get_legend_handles_labels(); axes[3].axis("off"); axes[3].legend(h, l, loc="center left", frameon=False, fontsize=6, borderaxespad=0)
fig.tight_layout(w_pad=0.5); fig.savefig(f"{args.out}/fig_alpha.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Fig 5: canary scatter (plate level) + selective prediction ----
fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95))
ax = axes[0]; P2 = P.dropna(subset=["pos_acc", "act_acc"]).groupby(["holdout", "plate"]).mean(numeric_only=True).reset_index()  # seeds averaged per plate
fl = P2.flagged >= 0.5
from scipy.stats import spearmanr
ax.scatter(P2.pos_acc, P2.act_acc, s=6, c=np.where(fl, C_OOD, C_ACC), alpha=0.7)
ax.set_xlabel("positive-control acc. (plate)"); ax.set_ylabel("active-compound acc. (plate)"); ax.set_title("Canary vs accuracy"); ax.grid(lw=0.3, alpha=0.5)
ax.text(0.03, 0.93, f"n={len(P2)} plates (seed-avg.)\nρ={spearmanr(P2.pos_acc, P2.act_acc)[0]:.2f}; flagged: {int(fl.sum())}", transform=ax.transAxes, fontsize=6.5, va="top")
ax = axes[1]
ax.scatter(P2.pos_acc, P2.std_trt_cov, s=6, c=np.where(fl, C_OOD, C_ACC), alpha=0.7); ax.axhline(0.9, color="k", ls="--", lw=0.8)
ax.set_xlabel("positive-control acc. (plate)"); ax.set_ylabel("std-CP coverage (plate, α=0.1)"); ax.set_title("Canary vs coverage"); ax.grid(lw=0.3, alpha=0.5)
ax = axes[2]
s1 = S[S.target_risk == 0.1]; s2 = S[S.target_risk == 0.2]
labels_ = ["ID thr.\nID plates", "ID thr.\nunseen", "canary\nunseen", "oracle\nunseen"]
vals10 = [s1.id_risk.mean(), s1.ood_risk.mean(), s1.ood_canary_risk.mean(), s1.ood_oracle_risk.mean()]
vals20 = [s2.id_risk.mean(), s2.ood_risk.mean(), s2.ood_canary_risk.mean(), s2.ood_oracle_risk.mean()]
xx = np.arange(4); ax.bar(xx - 0.18, vals10, 0.36, color=C_ID, label="target risk 10%"); ax.bar(xx + 0.18, vals20, 0.36, color=C_OOD, label="target risk 20%")
ax.axhline(0.1, color=C_ID, ls="--", lw=0.8); ax.axhline(0.2, color=C_OOD, ls="--", lw=0.8)
ax.set_xticks(xx); ax.set_xticklabels(labels_, fontsize=6); ax.set_ylabel("risk among named calls"); ax.set_title("Selective prediction"); ax.legend(frameon=False, fontsize=6, loc="upper left")
fig.tight_layout(w_pad=0.8); fig.savefig(f"{args.out}/fig_canary_selective.pdf"); plt.close(fig)
print("figures written to", args.out)
