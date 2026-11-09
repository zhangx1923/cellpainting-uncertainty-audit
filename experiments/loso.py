#!/usr/bin/env python3
"""Leave-one-source-out (LOSO) trust audit on JUMP-CP TARGET2 profiles — v3.

Per fold (held-out source = deployment lab):
  1. plates of training sources split 80/20 (by plate) into TRAIN and ID plates; ID plates split half/half into
     CAL (calibration) and IDTEST.
  2. features: --feat cp  -> CellProfiler profiles, per-plate DMSO robust z-scoring (--norm negcon | allwell | global),
                              feature selection + PCA fit on TRAIN
               --feat harmony -> consortium Harmony-corrected profiles (transductive over all labs; 113 plates / 9 sources)
  3. phenotypic activity from TRAIN plates only: compound ACTIVE if median DMSO-standardised squared distance of its
     TRAIN wells > the --active-pct percentile of TRAIN DMSO wells. Inactive compounds + DMSO -> one class "inactive".
  4. classifier: MLP (GPU) with early stopping on CAL, or multinomial logistic regression. Temperature scaling (TS) on CAL.
  5. trust metrics on IDTEST vs OOD: treatment wells (deployment target, "trt") and active-compound treatment wells ("act").
  6. conformal prediction, score s = -log p_y ("nlp", same sets as LAC) and APS, alpha in ALPHAS, calibrated on CAL trt wells:
       std        standard split CP (ID calibration)              [baseline]
       oracle     calibrated on OOD trt wells (not deployable)     [upper bound]
       v1_poscon  Mondrian: quantile from OOD plates' positive-control wells
       v2_match   control-coverage matching: threshold at which OOD positive controls reach the coverage ID positive
                  controls had at the ID treatment threshold
       v3_dmsoW   weighted CP with importance weights from a DMSO-only domain classifier (label-free)
       v4_gated   canary gate: per OOD plate, if the plate's positive-control accuracy is within the ID range
                  (>= mean - 2 sd over ID plates) apply v2_match (per source), else fall back to std and FLAG the plate
       v5_actmond activity-split Mondrian: positive-control quantile for active-looking wells, DMSO quantile otherwise
       v6_platestd plate-standardised CP: s - q_plate(positive controls), thresholded at the CAL active-well quantile
       oracle_act calibrated on OOD active-compound wells (not deployable)
     Coverage/size reported on OOD trt wells and OOD active-trt wells; plus per-plate statistics.
  7. selective prediction: confidence threshold chosen on CAL to reach target risk (10%, 20%) among NAMED calls
     (prediction != inactive) -> realised OOD risk/coverage; canary re-tuned threshold using OOD poscon wells.
Outputs <outdir>/<source>_s<seed><tag>.json
"""
import json, sys, time, argparse, os
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ROOT = os.environ.get("TRUST_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALPHAS = [0.05, 0.1, 0.2]
HARMONY = f"{ROOT}/data/target2_harmony.parquet"

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--feat", default="cp", choices=["cp", "harmony"])
    ap.add_argument("--norm", default="negcon", choices=["negcon", "allwell", "global", "none"])
    ap.add_argument("--clf", default="mlp", choices=["mlp", "logreg"])
    ap.add_argument("--pcs", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--active-pct", type=float, default=95.0)
    ap.add_argument("--no-active-merge", action="store_true")
    ap.add_argument("--sources", nargs="+", default=None, help="restrict training+holdout sources (e.g. Harmony-covered)")
    ap.add_argument("--outdir", default=f"{ROOT}/experiments/results/loso")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-wells", action="store_true")
    # repair strategies (defaults reproduce the main sweep)
    ap.add_argument("--ens", type=int, default=1, help="deep ensemble size (MLP)")
    ap.add_argument("--aug", type=float, default=0.0, help="plate-level random affine augmentation strength during training")
    ap.add_argument("--sphere", default="none", choices=["none", "lab"], help="per-lab DMSO whitening (label-free)")
    ap.add_argument("--align", default="none", choices=["none", "lab", "plate"], help="control-anchored per-component affine alignment")
    ap.add_argument("--align-lam", type=float, default=1.0)
    return ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- models ----------------
def mlp_fit(Xtr, ytr, Xval, yval, K, hidden=1024, epochs=200, lr=1e-3, wd=1e-3, patience=10, seed=0, aug=0.0, gtr=None):
    torch.manual_seed(seed)
    Xt = torch.tensor(Xtr, device=DEV); yt = torch.tensor(ytr, device=DEV)
    if aug > 0:  # plate-level nuisance augmentation: one random per-component scale and shift per training plate and epoch
        gt = torch.tensor(gtr, device=DEV); n_g = int(gtr.max()) + 1
    Xv = torch.tensor(Xval, device=DEV); yv = torch.tensor(yval, device=DEV)
    model = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(hidden, K)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = 1e9, None, 0; n = len(Xt); bs = 1024; ep = 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=DEV)
        if aug > 0:
            sc = torch.exp(aug * torch.randn(n_g, Xt.shape[1], device=DEV)); sh = aug * torch.randn(n_g, Xt.shape[1], device=DEV)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb = Xt[b] * sc[gt[b]] + sh[gt[b]] if aug > 0 else Xt[b]
            loss = torch.nn.functional.cross_entropy(model(xb), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = torch.nn.functional.cross_entropy(model(Xv), yv).item()
        if vl < best - 1e-4:
            best, bad = vl, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); model.eval()
    def logits(X):
        with torch.no_grad():
            return np.vstack([model(torch.tensor(X[i:i + 8192], device=DEV)).cpu().numpy() for i in range(0, len(X), 8192)])
    return logits, {"epochs_run": ep + 1, "best_val_nll": best}

def ens_fit(Xtr, ytr, Xval, yval, K, n_ens=1, seed=0, **kw):
    """Deep ensemble: average of member softmax outputs; returns log mean probability as 'logits'."""
    fns, infos = zip(*[mlp_fit(Xtr, ytr, Xval, yval, K, seed=seed * 100 + i, **kw) for i in range(n_ens)])
    def logits(X):
        Pm = np.mean([softmax(f(X)) for f in fns], axis=0); return np.log(np.clip(Pm, 1e-12, 1))
    Lv = logits(Xval); info = {"epochs_run": int(np.mean([i["epochs_run"] for i in infos])), "best_val_nll": nll(softmax(Lv), yval), "members_val_nll": [i["best_val_nll"] for i in infos]}
    return logits, info

def logreg_fit(Xtr, ytr, Xval, yval, K, **kw):
    lr = LogisticRegression(C=0.5, max_iter=300, n_jobs=8).fit(Xtr, ytr)
    return (lambda X: lr.decision_function(X)), {}

def softmax(L, T=1.0):
    L = L / T; L = L - L.max(1, keepdims=True); E = np.exp(L); return E / E.sum(1, keepdims=True)

def fit_temperature(L, y):
    best_T, best = 1.0, 1e9
    for T in np.exp(np.linspace(np.log(0.3), np.log(10), 60)):
        v = nll(softmax(L, T), y)
        if v < best: best, best_T = v, T
    return float(best_T)

# ---------------- conformal ----------------
def nlp(P, y): return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))
def nlp_sets(P, q): return -np.log(np.clip(P, 1e-12, 1)) <= q

def aps(P, y, rng):
    o = np.argsort(-P, axis=1); Ps = np.take_along_axis(P, o, 1); cs = np.cumsum(Ps, 1)
    rank = np.argmax(o == y[:, None], axis=1); u = rng.random(len(y))
    return cs[np.arange(len(y)), rank] - u * Ps[np.arange(len(y)), rank]

def aps_sets(P, q, rng):
    o = np.argsort(-P, axis=1); Ps = np.take_along_axis(P, o, 1); cs = np.cumsum(Ps, 1)
    u = rng.random((len(P), 1)); inc = (cs - u * Ps) <= q
    sets = np.zeros_like(P, bool); np.put_along_axis(sets, o, inc, 1); return sets

def quantile(scores, alpha, w=None):
    n = len(scores)
    if n == 0: return float("inf")
    if w is None:
        lvl = np.ceil((n + 1) * (1 - alpha)) / n
        return float(np.quantile(scores, lvl, method="higher")) if lvl <= 1 else float("inf")
    w = np.asarray(w, float); w = w / (w.sum() + w.mean())
    o = np.argsort(scores); cs = np.cumsum(w[o]); k = np.searchsorted(cs, 1 - alpha)
    return float(scores[o][k]) if k < n else float("inf")

def cov_size(sets, y):
    if len(y) == 0: return None, None
    return float(sets[np.arange(len(y)), y].mean()), float(sets.sum(1).mean())

# ---------------- metrics ----------------
def aurc(P, y):
    conf = P.max(1); err = (P.argmax(1) != y).astype(float); o = np.argsort(-conf)
    return float((np.cumsum(err[o]) / np.arange(1, len(o) + 1)).mean())

def metrics(P, y):
    if len(y) == 0: return {"n": 0}
    top5 = (np.argsort(-P, axis=1)[:, :5] == y[:, None]).any(1).mean()
    return {"n": int(len(y)), "acc": float((P.argmax(1) == y).mean()), "top5": float(top5), "nll": nll(P, y), "ece": ece(P, y),
            "brier": brier(P, y), "aurc": aurc(P, y), "mean_conf": float(P.max(1).mean())}

def thr_for_risk(conf, err, risk):
    """Largest coverage threshold on confidence such that risk among kept <= target (computed on calibration)."""
    o = np.argsort(-conf); er = np.cumsum(err[o]) / np.arange(1, len(o) + 1)
    ok = np.where(er <= risk)[0]
    return float(conf[o][ok.max()]) if len(ok) else float("inf")

def selective_eval(conf, err, thr):
    keep = conf >= thr
    return {"coverage": float(keep.mean()), "risk": float(err[keep].mean()) if keep.any() else None, "n_kept": int(keep.sum())}

# ---------------- data ----------------
def load_features(args):
    df, meta, feat = load()
    if args.feat == "harmony":
        h = pd.read_parquet(HARMONY)
        hf = [c for c in h.columns if c.startswith("X_")]
        df = df[meta].merge(h, on=["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"], how="inner").reset_index(drop=True)
        X = df[hf].to_numpy(np.float32); feat = hf
        return df, feat, X, True
    if args.norm == "negcon": X = plate_negcon_normalize(df, feat)
    elif args.norm == "allwell": X = plate_all_normalize(df, feat)
    elif args.norm == "none": X = np.clip(df[feat].to_numpy(np.float32), -1e6, 1e6)
    else: X = None
    return df, feat, X, False

# ---------------- fold ----------------
def run_fold(df, feat, Xn, pre_reduced, ho, seed, args):
    rng = np.random.default_rng(seed); t1 = time.time()
    comp, moa = labels(df); src = df.Metadata_Source.to_numpy(); plate = df.Metadata_Plate.to_numpy()
    is_dmso = comp == "DMSO"; ctype = df.Metadata_control_type.fillna("trt").to_numpy()
    is_pos = np.isin(ctype, ["poscon_cp", "poscon_diverse", "poscon_orf"]); is_trt = ~is_pos & ~is_dmso
    use = np.ones(len(df), bool) if args.sources is None else np.isin(src, args.sources)

    tr_src = (src != ho) & use
    tr_plates = np.unique(plate[tr_src]); rng.shuffle(tr_plates)
    n_id = max(6, int(0.2 * len(tr_plates)))
    id_plates, train_plates = tr_plates[:n_id], tr_plates[n_id:]
    cal_plates, idt_plates = id_plates[: n_id // 2], id_plates[n_id // 2:]
    m_train = np.isin(plate, train_plates); m_cal = np.isin(plate, cal_plates); m_idt = np.isin(plate, idt_plates); m_ood = (src == ho) & use

    X = Xn if Xn is not None else global_normalize(df, feat, m_train)
    if pre_reduced:
        Z = X.astype(np.float32)
    else:
        sel = select_features(X[m_train], feat, seed=seed)
        pca = PCA(n_components=args.pcs, random_state=seed).fit(X[m_train][:, sel])
        Z = pca.transform(X[:, sel]).astype(np.float32)
    mu, sd = Z[m_train].mean(0), Z[m_train].std(0) + 1e-6; Z = (Z - mu) / sd

    # phenotype score (DMSO-anchored per plate, label-free)
    pheno = np.zeros(len(Z), np.float32)
    for p in np.unique(plate):
        idx = np.where(plate == p)[0]; ref = idx[is_dmso[idx]]
        if len(ref) < 8: ref = idx
        m_, s_ = Z[ref].mean(0), Z[ref].std(0) + 1e-6
        pheno[idx] = (((Z[idx] - m_) / s_) ** 2).mean(1)
    dmso_thr = float(np.percentile(pheno[m_train & is_dmso], args.active_pct))
    tdf = pd.DataFrame({"comp": comp[m_train], "pheno": pheno[m_train]})
    med = tdf[tdf.comp != "DMSO"].groupby("comp").pheno.median()
    active = set(med.index[med > dmso_thr]) if not args.no_active_merge else set(med.index)
    lab = np.where(np.isin(comp, list(active)), comp, "inactive")
    classes = np.array(sorted(set(lab))); cidx = {c: i for i, c in enumerate(classes)}; y = np.array([cidx[c] for c in lab]); K = len(classes)
    inactive_id = cidx["inactive"]

    if args.sphere == "lab":  # whiten every lab with its own DMSO wells (shrinkage covariance); label-free.
        # Applied after the activity labels are fixed, so the class set is identical to the unsphered pipeline.
        for s_ in np.unique(src):
            ms = src == s_; R = Z[ms & is_dmso]
            S = np.cov(R.T); S = 0.9 * S + 0.1 * np.trace(S) / len(S) * np.eye(len(S))
            ev, V = np.linalg.eigh(S); W = V @ np.diag(ev ** -0.5) @ V.T
            Z[ms] = ((Z[ms] - R.mean(0)) @ W).astype(np.float32)
        mu2, sd2 = Z[m_train].mean(0), Z[m_train].std(0) + 1e-6; Z = (Z - mu2) / sd2
    if args.align != "none":
        # control-anchored alignment: per principal component, an affine map (ridge-shrunk towards identity) that moves the
        # deployed lab's control-group means (each positive-control compound, DMSO) onto the training labs' means.
        # Uses only control identities recorded in the plate map; no treatment-well label.
        grp = np.where(is_dmso, "DMSO", np.where(is_pos, comp, ""))
        gnames = [g for g in np.unique(grp[is_pos | is_dmso])]
        muS = np.array([Z[m_train & (grp == g)].mean(0) for g in gnames])
        units = [m_ood] if args.align == "lab" else [m_ood & (plate == p) for p in np.unique(plate[m_ood])]
        lam = args.align_lam; r_align = []
        for mu_ in units:
            ok = [i for i, g in enumerate(gnames) if (mu_ & (grp == g)).sum() >= 2]
            muT = np.array([Z[mu_ & (grp == gnames[i])].mean(0) for i in ok]); tS = muS[ok]
            w = np.array([(mu_ & (grp == gnames[i])).sum() for i in ok], float); w = w / w.sum() * len(ok)
            # minimise sum_g w_g (a*muT + b - muS)^2 + lam*((a-1)^2 + b^2) per component (closed form 2x2 solve)
            Sw, Sx, Sxx = w.sum(), (w[:, None] * muT).sum(0), (w[:, None] * muT ** 2).sum(0)
            Sy, Sxy = (w[:, None] * tS).sum(0), (w[:, None] * muT * tS).sum(0)
            A11, A12, A22 = Sxx + lam, Sx, Sw + lam; b1, b2 = Sxy + lam, Sy
            det = A11 * A22 - A12 ** 2; a = (b1 * A22 - A12 * b2) / det; b = (A11 * b2 - A12 * b1) / det
            Z[mu_] = (Z[mu_] * a + b).astype(np.float32); r_align.append([float(np.median(a)), float(np.median(np.abs(b)))])
    if args.clf == "mlp":
        gtr = pd.factorize(plate[m_train])[0]
        fit = (lambda *a_, **k_: ens_fit(*a_, n_ens=args.ens, aug=args.aug, gtr=gtr, **k_)) if args.ens > 1 else (lambda *a_, **k_: mlp_fit(*a_, aug=args.aug, gtr=gtr, **k_))
    else:
        fit = logreg_fit
    logit_fn, info = fit(Z[m_train], y[m_train], Z[m_cal], y[m_cal], K, hidden=args.hidden, seed=seed)
    L = {"cal": logit_fn(Z[m_cal]), "idt": logit_fn(Z[m_idt]), "ood": logit_fn(Z[m_ood])}
    T = fit_temperature(L["cal"], y[m_cal])
    P = {"raw": {k: softmax(v) for k, v in L.items()}, "ts": {k: softmax(v, T) for k, v in L.items()}}
    Y = {"cal": y[m_cal], "idt": y[m_idt], "ood": y[m_ood]}
    M = {k: {"trt": is_trt[m], "pos": is_pos[m], "dmso": is_dmso[m]} for k, m in [("cal", m_cal), ("idt", m_idt), ("ood", m_ood)]}
    for k in M: M[k]["act"] = M[k]["trt"] & (Y[k] != inactive_id); M[k]["inact_trt"] = M[k]["trt"] & (Y[k] == inactive_id)
    PL = {"cal": plate[m_cal], "idt": plate[m_idt], "ood": plate[m_ood]}
    PH = {"cal": pheno[m_cal], "idt": pheno[m_idt], "ood": pheno[m_ood]}

    r = {"holdout": ho, "seed": seed, "K": K, "n_active_compounds": len(active), "active_compounds": sorted(active), "dmso_thr": dmso_thr,
         "temperature": T, "clf_info": info, "device": DEV,
         "n": {"train": int(m_train.sum()), "cal": int(m_cal.sum()), "idtest": int(m_idt.sum()), "ood": int(m_ood.sum()),
               "cal_plates": int(len(cal_plates)), "idt_plates": int(len(idt_plates)), "ood_plates": int(len(np.unique(PL["ood"])))},
         "metrics": {}}
    if args.align != "none": r["align_median_scale_shift"] = r_align
    for pk in ["raw", "ts"]:
        r["metrics"][pk] = {f"{split}_{grp}": metrics(P[pk][split][M[split][grp]], Y[split][M[split][grp]])
                            for split in ["idt", "ood"] for grp in ["trt", "act", "pos", "dmso", "inact_trt"]}
    r["inactive_recall"] = {s: float((P["raw"][s][M[s]["dmso"]].argmax(1) == inactive_id).mean()) for s in ["idt", "ood"]}
    r["false_naming_rate"] = {s: float((P["raw"][s][M[s]["inact_trt"]].argmax(1) != inactive_id).mean()) for s in ["idt", "ood"]}

    # DMSO-only domain classifier -> weights (label-free)
    Zd = np.vstack([Z[m_cal][M["cal"]["dmso"]], Z[m_ood][M["ood"]["dmso"]]]); yd = np.r_[np.zeros(M["cal"]["dmso"].sum()), np.ones(M["ood"]["dmso"].sum())]
    dc = LogisticRegression(C=0.05, max_iter=2000).fit(Zd, yd)
    pr = dc.predict_proba(Z[m_cal])[:, 1]; w_cal = np.clip(pr / (1 - pr + 1e-6), 1e-3, 20)
    r["dmso_domain_acc"] = float(dc.score(Zd, yd))

    # canary statistics per plate (ID plates define the in-regime range; OOD plates are gated)
    def plate_stats(split, pk="raw"):
        out = {}
        Pp, yy, pl = P[pk][split], Y[split], PL[split]
        for p in np.unique(pl):
            m = pl == p; pos = m & M[split]["pos"]; trt = m & M[split]["trt"]; act = m & M[split]["act"]; dm = m & M[split]["dmso"]
            out[str(p)] = {"n_pos": int(pos.sum()), "pos_acc": float((Pp[pos].argmax(1) == yy[pos]).mean()) if pos.any() else None,
                           "pos_nlp": float(nlp(Pp[pos], yy[pos]).mean()) if pos.any() else None,
                           "trt_acc": float((Pp[trt].argmax(1) == yy[trt]).mean()) if trt.any() else None,
                           "act_acc": float((Pp[act].argmax(1) == yy[act]).mean()) if act.any() else None,
                           "dmso_pheno_med": float(np.median(PH[split][dm])) if dm.any() else None,
                           "trt_pheno_med": float(np.median(PH[split][trt])) if trt.any() else None,
                           "dmso_inactive_recall": float((Pp[dm].argmax(1) == inactive_id).mean()) if dm.any() else None}
        return out
    ps_id = plate_stats("idt"); ps_cal = plate_stats("cal"); ps_ood = plate_stats("ood")
    id_pos_acc = np.array([v["pos_acc"] for v in list(ps_id.values()) + list(ps_cal.values()) if v["pos_acc"] is not None])
    gate_thr = float(id_pos_acc.mean() - 2 * id_pos_acc.std()) if len(id_pos_acc) > 1 else 0.0
    r["canary"] = {"gate_thr_pos_acc": gate_thr, "id_pos_acc_mean": float(id_pos_acc.mean()), "id_pos_acc_sd": float(id_pos_acc.std()),
                   "ood_plates": ps_ood, "id_plates": ps_id, "cal_plates": ps_cal}
    ood_flag = {p: (v["pos_acc"] is not None and v["pos_acc"] < gate_thr) for p, v in ps_ood.items()}
    flag_mask = np.array([ood_flag[str(p)] for p in PL["ood"]])
    r["canary"]["ood_frac_plates_flagged"] = float(np.mean(list(ood_flag.values())))

    # conformal
    conf = {}
    for pk in ["raw", "ts"]:
        for sc in ["nlp", "aps"]:
            Pc, Pi, Po = P[pk]["cal"], P[pk]["idt"], P[pk]["ood"]
            if sc == "nlp":
                s_cal, s_ood = nlp(Pc, Y["cal"]), nlp(Po, Y["ood"])
                sets_i = lambda q: nlp_sets(Pi, q); sets_o = lambda q: nlp_sets(Po, q)
            else:
                s_cal = aps(Pc, Y["cal"], np.random.default_rng(seed)); s_ood = aps(Po, Y["ood"], np.random.default_rng(seed + 1))
                sets_i = lambda q: aps_sets(Pi, q, np.random.default_rng(seed + 2)); sets_o = lambda q: aps_sets(Po, q, np.random.default_rng(seed + 3))
            ct, cp_, oa, ot, op = M["cal"]["trt"], M["cal"]["pos"], M["ood"]["act"], M["ood"]["trt"], M["ood"]["pos"]
            for a in ALPHAS:
                e = {}
                q_std = quantile(s_cal[ct], a); e["q_std"] = q_std
                Si = sets_i(q_std); e["id_trt_cov"], e["id_trt_size"] = cov_size(Si[M["idt"]["trt"]], Y["idt"][M["idt"]["trt"]])
                e["id_act_cov"], e["id_act_size"] = cov_size(Si[M["idt"]["act"]], Y["idt"][M["idt"]["act"]])
                variants = {"std": q_std,
                            "oracle": quantile(s_ood[ot], a),
                            "v1_poscon": quantile(s_ood[op], a)}
                c_pos_id = float((s_cal[cp_] <= q_std).mean()); e["c_pos_id"] = c_pos_id; e["c_pos_ood_at_qstd"] = float((s_ood[op] <= q_std).mean())
                variants["v2_match"] = float(np.quantile(s_ood[op], min(1.0, c_pos_id), method="higher")) if op.any() else q_std
                variants["v3_dmsoW"] = quantile(s_cal[ct], a, w=w_cal[ct])
                variants["oracle_act"] = quantile(s_ood[oa], a)
                for name, q in variants.items():
                    So = sets_o(q); e[f"ood_trt_cov_{name}"], e[f"ood_trt_size_{name}"] = cov_size(So[ot], Y["ood"][ot])
                    e[f"ood_act_cov_{name}"], e[f"ood_act_size_{name}"] = cov_size(So[oa], Y["ood"][oa]); e[f"q_{name}"] = q
                # v5 activity-split Mondrian: deployed-plate controls split along the activity axis
                looks_act = PH["ood"] > dmso_thr; od = M["ood"]["dmso"]
                q_pos = quantile(s_ood[op], a); q_dm = quantile(s_ood[od], a)
                q_well = np.where(looks_act, q_pos, q_dm)
                So = sets_o(np.inf) if False else None
                if sc == "nlp":
                    So = -np.log(np.clip(Po, 1e-12, 1)) <= q_well[:, None]
                else:
                    o_ = np.argsort(-Po, axis=1); Ps_ = np.take_along_axis(Po, o_, 1); cs_ = np.cumsum(Ps_, 1)
                    u_ = np.random.default_rng(seed + 4).random((len(Po), 1)); inc_ = (cs_ - u_ * Ps_) <= q_well[:, None]
                    So = np.zeros_like(Po, bool); np.put_along_axis(So, o_, inc_, 1)
                e["ood_trt_cov_v5_actmond"], e["ood_trt_size_v5_actmond"] = cov_size(So[ot], Y["ood"][ot])
                e["ood_act_cov_v5_actmond"], e["ood_act_size_v5_actmond"] = cov_size(So[oa], Y["ood"][oa]); e["q_v5_pos"] = q_pos; e["q_v5_dmso"] = q_dm
                e["ood_frac_looks_active_trt"] = float(looks_act[ot].mean())
                # ID analogue of v5 for reference (ID plates' own controls)
                looks_act_i = PH["idt"] > dmso_thr
                s_idt = nlp(Pi, Y["idt"]) if sc == "nlp" else aps(Pi, Y["idt"], np.random.default_rng(seed + 5))
                q_well_i = np.where(looks_act_i, quantile(s_idt[M["idt"]["pos"]], a), quantile(s_idt[M["idt"]["dmso"]], a))
                Si5 = (-np.log(np.clip(Pi, 1e-12, 1)) <= q_well_i[:, None]) if sc == "nlp" else aps_sets(Pi, np.inf, np.random.default_rng(0))
                if sc == "nlp":
                    e["id_trt_cov_v5"], e["id_trt_size_v5"] = cov_size(Si5[M["idt"]["trt"]], Y["idt"][M["idt"]["trt"]])
                    e["id_act_cov_v5"], e["id_act_size_v5"] = cov_size(Si5[M["idt"]["act"]], Y["idt"][M["idt"]["act"]])
                # v6 plate-standardised CP: score minus the plate's own positive-control conformal quantile; the threshold on this
                # standardised score is calibrated on CAL active-compound wells (each against its own CAL plate's controls)
                if sc == "nlp":
                    def plate_q(s_, pos_, pl_):
                        out = np.full(len(s_), np.inf)
                        for p_ in np.unique(pl_):
                            m_ = pl_ == p_; out[m_] = quantile(s_[m_ & pos_], a)
                        return out
                    t6 = quantile((s_cal - plate_q(s_cal, cp_, PL["cal"]))[M["cal"]["act"]], a); e["t_v6"] = t6
                    for sp_, P_, s_, pos_ in [("ood", Po, s_ood, op), ("id", Pi, nlp(Pi, Y["idt"]), M["idt"]["pos"])]:
                        k_ = "ood" if sp_ == "ood" else "idt"
                        S6 = -np.log(np.clip(P_, 1e-12, 1)) <= (plate_q(s_, pos_, PL[k_]) + t6)[:, None]
                        e[f"{sp_}_trt_cov_v6_platestd"], e[f"{sp_}_trt_size_v6_platestd"] = cov_size(S6[M[k_]["trt"]], Y[k_][M[k_]["trt"]])
                        e[f"{sp_}_act_cov_v6_platestd"], e[f"{sp_}_act_size_v6_platestd"] = cov_size(S6[M[k_]["act"]], Y[k_][M[k_]["act"]])
                # v4 gated: v2 on unflagged plates, std on flagged plates
                So = np.where(flag_mask[:, None], sets_o(q_std), sets_o(variants["v2_match"]))
                e["ood_trt_cov_v4_gated"], e["ood_trt_size_v4_gated"] = cov_size(So[ot], Y["ood"][ot])
                e["ood_act_cov_v4_gated"], e["ood_act_size_v4_gated"] = cov_size(So[oa], Y["ood"][oa])
                # coverage on unflagged vs flagged plates under std and v2
                for nm, q in [("std", q_std), ("v2_match", variants["v2_match"])]:
                    S_ = sets_o(q)
                    for fl, lab_ in [(False, "unflagged"), (True, "flagged")]:
                        m_ = ot & (flag_mask == fl)
                        e[f"ood_trt_cov_{nm}_{lab_}"], e[f"ood_trt_size_{nm}_{lab_}"] = cov_size(S_[m_], Y["ood"][m_])
                # per-plate std coverage (for canary correlation)
                S_ = sets_o(q_std)
                e["per_plate_std_trt_cov"] = {str(p): float(S_[(PL["ood"] == p) & ot][np.arange(((PL["ood"] == p) & ot).sum()), Y["ood"][(PL["ood"] == p) & ot]].mean()) for p in np.unique(PL["ood"])}
                conf[f"{pk}_{sc}_a{a}"] = e
    r["conformal"] = conf

    # selective prediction among NAMED calls (pred != inactive)
    sel = {}
    for pk in ["raw", "ts"]:
        def named_stats(split):
            Pp, yy, m = P[pk][split], Y[split], M[split]["trt"]
            pred = Pp.argmax(1); named = m & (pred != inactive_id)
            return Pp.max(1)[named], (pred != yy)[named], named
        c_cal, e_cal, _ = named_stats("cal"); c_id, e_id, n_id_ = named_stats("idt"); c_ood, e_ood, n_ood_ = named_stats("ood")
        c_pos_ood = P[pk]["ood"].max(1)[M["ood"]["pos"]]; e_pos_ood = (P[pk]["ood"].argmax(1) != Y["ood"])[M["ood"]["pos"]]
        c_pos_cal = P[pk]["cal"].max(1)[M["cal"]["pos"]]; e_pos_cal = (P[pk]["cal"].argmax(1) != Y["cal"])[M["cal"]["pos"]]
        s = {"frac_named_id": float(n_id_.sum() / M["idt"]["trt"].sum()), "frac_named_ood": float(n_ood_.sum() / M["ood"]["trt"].sum()),
             "risk_named_id_nothr": float(e_id.mean()), "risk_named_ood_nothr": float(e_ood.mean()),
             "aurc_named_id": None, "aurc_named_ood": None}
        for risk in [0.1, 0.2]:
            thr = thr_for_risk(c_cal, e_cal, risk)
            s[f"thr_id_r{risk}"] = thr; s[f"id_r{risk}"] = selective_eval(c_id, e_id, thr); s[f"ood_r{risk}"] = selective_eval(c_ood, e_ood, thr)
            # canary re-tuning: shift threshold so that OOD poscon risk at thr matches ID poscon risk at thr
            r_pos_cal = selective_eval(c_pos_cal, e_pos_cal, thr)["risk"]
            thr2 = thr_for_risk(c_pos_ood, e_pos_ood, r_pos_cal if r_pos_cal is not None else risk)
            s[f"thr_canary_r{risk}"] = thr2; s[f"ood_canary_r{risk}"] = selective_eval(c_ood, e_ood, thr2)
            thr3 = thr_for_risk(c_ood, e_ood, risk); s[f"ood_oracle_r{risk}"] = selective_eval(c_ood, e_ood, thr3)
        sel[pk] = s
    r["selective_named"] = sel
    r["elapsed_s"] = round(time.time() - t1, 1)
    if getattr(args, "save_wells", False):
        np.savez_compressed(f"{args.outdir}/{ho}_s{seed}{args.tag}_wells.npz", classes=classes,
                            idt_P=P["raw"]["idt"].astype(np.float16), idt_y=Y["idt"], idt_plate=PL["idt"], idt_pheno=PH["idt"], idt_pos=M["idt"]["pos"], idt_dmso=M["idt"]["dmso"],
                            ood_P=P["raw"]["ood"].astype(np.float16), ood_y=Y["ood"], ood_plate=PL["ood"], ood_pheno=PH["ood"], ood_pos=M["ood"]["pos"], ood_dmso=M["ood"]["dmso"],
                            cal_P=P["raw"]["cal"].astype(np.float16), cal_y=Y["cal"], cal_plate=PL["cal"], cal_pheno=PH["cal"], cal_pos=M["cal"]["pos"], cal_dmso=M["cal"]["dmso"],
                            T=T, dmso_thr=dmso_thr, flag_mask=flag_mask)
    return r

def main():
    args = parse(); t0 = time.time()
    df, feat, Xn, pre = load_features(args)
    sources = args.holdout or sorted(df.Metadata_Source.unique())
    if args.sources: sources = [s for s in sources if s in args.sources]
    os.makedirs(args.outdir, exist_ok=True)
    print(f"prepared in {time.time()-t0:.0f}s; device {DEV}; feat {args.feat}/{args.norm}; folds {sources} seeds {args.seeds}", flush=True)
    for ho in sources:
        for seed in args.seeds:
            out = f"{args.outdir}/{ho}_s{seed}{args.tag}.json"
            if os.path.exists(out): print("skip", out); continue
            r = run_fold(df, feat, Xn, pre, ho, seed, args); r["config"] = vars(args)
            json.dump(r, open(out, "w"), indent=1, default=float)
            m = r["metrics"]["raw"]; c = r["conformal"]["raw_nlp_a0.1"]; s = r["selective_named"]["raw"]
            print(f"[{ho} s{seed}] K={r['K']} T={r['temperature']:.2f} ep={r['clf_info'].get('epochs_run')} | act acc ID {m['idt_act']['acc']:.3f} ECE {m['idt_act']['ece']:.3f} / OOD {m['ood_act']['acc']:.3f} ECE {m['ood_act']['ece']:.3f} | trt acc ID {m['idt_trt']['acc']:.3f} OOD {m['ood_trt']['acc']:.3f} | "
                  f"CP trt@0.1 ID {c['id_trt_cov']:.3f}/{c['id_trt_size']:.1f} OOD std {c['ood_trt_cov_std']:.3f}/{c['ood_trt_size_std']:.1f} v2 {c['ood_trt_cov_v2_match']:.3f}/{c['ood_trt_size_v2_match']:.1f} v4 {c['ood_trt_cov_v4_gated']:.3f}/{c['ood_trt_size_v4_gated']:.1f} orc {c['ood_trt_cov_oracle']:.3f}/{c['ood_trt_size_oracle']:.1f} flagged {r['canary']['ood_frac_plates_flagged']:.2f} | "
                  f"named risk@r10 ID {s['id_r0.1']['risk']} OOD {s['ood_r0.1']['risk']} canary {s['ood_canary_r0.1']['risk']} | {r['elapsed_s']}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
