"""Shared data utilities for the TARGET2 trust experiments.

Design (deployment-realistic, inductive):
  * Every plate is normalised ONLY with its own DMSO negative-control wells (robust z-score:
    median / 1.4826*MAD). This is label-free and available on any new plate.
  * Feature selection + PCA are fit on TRAINING sources only; applied to held-out sources.
  * Leave-source-out (LSO): the held-out source is never seen by normalisation-fitting,
    feature selection, PCA, classifier or calibration — except via per-plate controls
    (which are present on every deployed plate by assay design).
"""
import json, os
import numpy as np
import pandas as pd
ROOT = os.environ.get("TRUST_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = f"{ROOT}/data/target2_wells.parquet"
DMSO = "JCP2022_033924"

def load():
    df = pd.read_parquet(DATA)
    meta = [c for c in df.columns if c.startswith("Metadata_")]
    feat = [c for c in df.columns if not c.startswith("Metadata_")]
    return df, meta, feat

def plate_negcon_normalize(df, feat, min_dmso=8):
    """Robust z-score per plate using that plate's DMSO wells. Returns float32 array."""
    X = df[feat].to_numpy(dtype=np.float32)
    out = np.empty_like(X)
    plates = df["Metadata_Plate"].to_numpy()
    is_dmso = (df["Metadata_JCP2022"] == DMSO).to_numpy()
    for p in np.unique(plates):
        idx = np.where(plates == p)[0]
        ref = idx[is_dmso[idx]]
        if len(ref) < min_dmso:   # fall back to all wells of the plate
            ref = idx
        med = np.median(X[ref], axis=0)
        mad = np.median(np.abs(X[ref] - med), axis=0) * 1.4826
        mad[mad < 1e-6] = 1e-6
        out[idx] = (X[idx] - med) / mad
    out = np.clip(out, -20, 20)
    return out

def plate_all_normalize(df, feat):
    """Robust z-score per plate using ALL wells of the plate (no controls needed)."""
    X = df[feat].to_numpy(dtype=np.float32)
    out = np.empty_like(X)
    plates = df["Metadata_Plate"].to_numpy()
    for p in np.unique(plates):
        idx = np.where(plates == p)[0]
        med = np.median(X[idx], axis=0)
        mad = np.median(np.abs(X[idx] - med), axis=0) * 1.4826
        mad[mad < 1e-6] = 1e-6
        out[idx] = (X[idx] - med) / mad
    return np.clip(out, -20, 20)

def global_normalize(df, feat, train_mask):
    """Robust z-score with statistics from training rows only (no plate awareness)."""
    X = df[feat].to_numpy(dtype=np.float32)
    med = np.median(X[train_mask], axis=0)
    mad = np.median(np.abs(X[train_mask] - med), axis=0) * 1.4826
    mad[mad < 1e-6] = 1e-6
    return np.clip((X - med) / mad, -20, 20)

def select_features(Xtr, feat, var_thresh=1e-3, corr_thresh=0.9, max_feats=1200, seed=0):
    """Variance filter + greedy correlation pruning, fit on training rows only.
    Returns column indices."""
    rng = np.random.default_rng(seed)
    v = Xtr.var(axis=0)
    keep = np.where(v > var_thresh)[0]
    # subsample rows for correlation estimate
    sub = Xtr[rng.choice(len(Xtr), size=min(6000, len(Xtr)), replace=False)][:, keep]
    C = np.corrcoef(sub.T)
    C = np.nan_to_num(C)
    order = np.argsort(-v[keep])  # highest variance first
    chosen = []
    taken = np.zeros(len(keep), bool)
    for i in order:
        if taken[i]:
            continue
        chosen.append(keep[i])
        taken |= np.abs(C[i]) > corr_thresh
        if len(chosen) >= max_feats:
            break
    return np.array(sorted(chosen))

def labels(df):
    """Label vectors. compound: InChIKey of every non-DMSO well. moa: first-listed MoA."""
    comp = df["Metadata_InChIKey"].astype(str).to_numpy()
    comp = np.where(df["Metadata_JCP2022"].to_numpy() == DMSO, "DMSO", comp)
    moa = df["Metadata_moa"].astype(object).to_numpy()
    moa = np.array([m.split("|")[0].strip() if isinstance(m, str) else None for m in moa], dtype=object)
    return comp, moa

def ece(probs, y, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)

def nll(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())

def brier(probs, y):
    oh = np.zeros_like(probs); oh[np.arange(len(y)), y] = 1
    return float(((probs - oh) ** 2).sum(1).mean())
