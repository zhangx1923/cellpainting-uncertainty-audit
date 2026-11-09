#!/usr/bin/env python3
"""Compare repair strategies from leave-one-lab-out JSONs: one row per tag, means over the labs present for every tag.
Usage: screen_compare.py --dir experiments/results/strat [--tags _base _ens5 ...]"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--tags", nargs="*"); ap.add_argument("--ref", default="_base")
ap.add_argument("--alpha", default="0.1"); ap.add_argument("--out", default=None, help="write lab means and tests here"); ap.add_argument("--seeds", nargs="*", type=int); args = ap.parse_args()
rows = []
for f in glob.glob(f"{args.dir}/source_*_s*.json"):
    m = re.match(r"(source_\d+)_s(\d+)(.*)\.json", os.path.basename(f)); ho, seed, tag = m.group(1), int(m.group(2)), m.group(3)
    if args.tags and tag not in args.tags: continue
    if args.seeds and seed not in args.seeds: continue
    r = json.load(open(f)); mt = r["metrics"]["raw"]; c = r["conformal"][f"raw_nlp_a{args.alpha}"]; s = r["selective_named"]["raw"]
    pl = r["canary"]["ood_plates"]; pa = [v["pos_acc"] for v in pl.values() if v["act_acc"] is not None]; aa = [v["act_acc"] for v in pl.values() if v["act_acc"] is not None]
    row = {"tag": tag, "lab": ho, "seed": seed, "acc_act_id": mt["idt_act"]["acc"], "ece_act_id": mt["idt_act"]["ece"], "acc_act": mt["ood_act"]["acc"], "ece_act": mt["ood_act"]["ece"], "aurc_act": mt["ood_act"]["aurc"],
           "nll_act": mt["ood_act"]["nll"], "acc_pos": mt["ood_pos"]["acc"], "cal_nll": r["clf_info"].get("best_val_nll"),
           "risk10": s["ood_r0.1"]["risk"], "risk10_can": s["ood_canary_r0.1"]["risk"], "cov10_can": s["ood_canary_r0.1"]["coverage"],
           "flag_frac": r["canary"]["ood_frac_plates_flagged"], "rho_plate": spearmanr(pa, aa)[0] if len(pa) > 2 else np.nan}
    for v in ["std", "v1_poscon", "v5_actmond", "v6_platestd", "oracle_act", "oracle"]:
        row[f"act_{v}"] = c.get(f"ood_act_cov_{v}"); row[f"actsz_{v}"] = c.get(f"ood_act_size_{v}"); row[f"trt_{v}"] = c.get(f"ood_trt_cov_{v}")
    rows.append(row)
D = pd.DataFrame(rows)
labs = set.intersection(*[set(g.lab) for _, g in D.groupby("tag")]); D = D[D.lab.isin(labs)]
L = D.groupby(["tag", "lab"]).mean(numeric_only=True)
cols = ["acc_act_id", "ece_act_id", "acc_act", "ece_act", "aurc_act", "act_std", "act_v1_poscon", "actsz_v1_poscon", "act_v5_actmond", "actsz_v5_actmond", "act_v6_platestd", "actsz_v6_platestd", "trt_v6_platestd",
        "act_oracle_act", "actsz_oracle_act", "risk10", "risk10_can", "acc_pos", "cal_nll", "rho_plate"]
cols = [c_ for c_ in cols if c_ in L.columns]; T = L.groupby("tag")[cols].mean()
pd.set_option("display.width", 250); print(f"labs: {len(labs)}"); print(T.round(3).T.to_string())
tests = {}
if args.ref in T.index:
    print(f"\nlab-level Wilcoxon vs {args.ref} (n up / n labs, p):")
    for tag in T.index:
        if tag == args.ref: continue
        out = []
        for c_ in [k for k in ["acc_act", "ece_act", "aurc_act", "act_std", "act_v1_poscon", "act_v6_platestd", "actsz_v6_platestd", "actsz_oracle_act", "risk10", "risk10_can", "rho_plate"] if k in L.columns]:
            a, b = L.loc[args.ref][c_], L.loc[tag][c_]; d = (b - a).dropna()
            p = wilcoxon(d).pvalue if len(d) > 5 and (d != 0).any() else np.nan
            out.append(f"{c_} {int((d > 0).sum())}/{len(d)} p={p:.3f}"); tests[f"{tag}|{c_}"] = {"n_up": int((d > 0).sum()), "n_down": int((d < 0).sum()), "n": int(len(d)), "p": float(p), "diff": float(d.mean())}
        print(f"{tag:14s} " + " | ".join(out))
if args.out:
    os.makedirs(args.out, exist_ok=True); L.to_csv(f"{args.out}/lab_means.csv"); T.to_csv(f"{args.out}/tag_means.csv")
    json.dump({"labs": sorted(labs), "tests": tests}, open(f"{args.out}/tests.json", "w"), indent=1)
