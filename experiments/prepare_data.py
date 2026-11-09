#!/usr/bin/env python3
"""Assemble JUMP-CP TARGET2 well-level profiles into one labelled table.

Inputs (all downloaded from public sources, see data/README.md):
  data/target2/<plate>.parquet         CellProfiler well-level profiles (cpg0016-jump)
  data/meta/well.csv.gz                (Source, Plate, Well) -> JCP2022 id
  data/meta/compound.csv.gz            JCP2022 -> InChIKey
  data/meta/plate.csv.gz               plate -> source/batch
  data/meta/JUMP-Target-2_compound_metadata.tsv   InChIKey -> name, control_type, targets
  data/meta/repurposing_info.tsv       Drug Repurposing Hub MoA annotations

Output:
  data/target2_wells.parquet  : metadata columns (Metadata_*) + feature columns
  data/target2_summary.json   : counts used in the paper
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
ROOT = os.environ.get("TRUST_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D = f"{ROOT}/data"

def main():
    files = sorted(glob.glob(f"{D}/target2/*.parquet"))
    plates = pd.read_csv(f"{D}/meta/plate.csv.gz")
    plates = plates[plates.Metadata_PlateType == "TARGET2"]
    assert len(files) == len(plates), f"expected {len(plates)} plates, found {len(files)}"

    wells = pd.read_csv(f"{D}/meta/well.csv.gz")
    comp = pd.read_csv(f"{D}/meta/compound.csv.gz")[["Metadata_JCP2022", "Metadata_InChIKey"]]
    tgt = pd.read_csv(f"{D}/meta/JUMP-Target-2_compound_metadata.tsv", sep="\t")
    tgt["Metadata_InChIKey"] = tgt["InChIKey"]
    tgt = tgt[["Metadata_InChIKey", "pert_iname", "pert_type", "control_type", "target_list", "pubchem_cid"]]
    tgt = tgt.drop_duplicates("Metadata_InChIKey")

    rep = pd.read_csv(f"{D}/meta/repurposing_info.tsv", sep="\t", low_memory=False)
    rep["key14"] = rep["InChIKey"].astype(str).str[:14]
    rep_by_key = rep.dropna(subset=["moa"]).drop_duplicates("key14").set_index("key14")["moa"]
    rep_by_name = rep.dropna(subset=["moa"]).drop_duplicates("pert_iname").set_index("pert_iname")["moa"]

    dfs = []
    for f in files:
        d = pd.read_parquet(f)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    feat_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    print("raw wells", df.shape, "features", len(feat_cols))

    df = df.merge(plates[["Metadata_Source", "Metadata_Plate", "Metadata_Batch"]], on=["Metadata_Source", "Metadata_Plate"], how="left")
    df = df.merge(wells, on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"], how="left")
    df = df.merge(comp, on="Metadata_JCP2022", how="left")
    df = df.merge(tgt, on="Metadata_InChIKey", how="left")

    # JUMP control ids: JCP2022_033924 = DMSO (negcon). Mark anything with no compound mapping.
    is_dmso = df["Metadata_JCP2022"].eq("JCP2022_033924")
    df.loc[is_dmso, "pert_iname"] = "DMSO"
    df.loc[is_dmso, "pert_type"] = "control"
    df.loc[is_dmso, "control_type"] = "negcon"

    key14 = df["Metadata_InChIKey"].astype(str).str[:14]
    df["moa"] = key14.map(rep_by_key)
    df["moa"] = df["moa"].fillna(df["pert_iname"].map(rep_by_name))

    # rename label columns into Metadata_ namespace
    ren = {"pert_iname": "Metadata_pert_iname", "pert_type": "Metadata_pert_type", "control_type": "Metadata_control_type",
           "target_list": "Metadata_target_list", "pubchem_cid": "Metadata_pubchem_cid", "moa": "Metadata_moa"}
    df = df.rename(columns=ren)
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]

    # drop features with any NaN / inf / zero variance across the whole table
    X = df[feat_cols].to_numpy(dtype=np.float32)
    bad = ~np.isfinite(X).all(axis=0)
    bad |= (np.nanstd(X, axis=0) == 0)
    # drop known non-morphology bookkeeping features
    for i, c in enumerate(feat_cols):
        if c.startswith(("Image_", "Metadata_")) or "_Location_" in c or "_Number_Object_Number" in c or "_Parent_" in c or "_Children_" in c:
            bad[i] = True
    keep = [c for c, b in zip(feat_cols, bad) if not b]
    print("dropped features", int(bad.sum()), "kept", len(keep))
    out = df[meta_cols + keep]
    out.to_parquet(f"{D}/target2_wells.parquet", index=False)

    s = {
        "n_wells": int(len(out)),
        "n_plates": int(out.Metadata_Plate.nunique()),
        "n_sources": int(out.Metadata_Source.nunique()),
        "n_batches": int(out.Metadata_Batch.nunique()),
        "n_features_raw": len(feat_cols),
        "n_features_kept": len(keep),
        "wells_per_plate": out.groupby("Metadata_Plate").size().describe().to_dict(),
        "pert_type_counts": out.Metadata_pert_type.value_counts(dropna=False).to_dict(),
        "control_type_counts": out.Metadata_control_type.value_counts(dropna=False).to_dict(),
        "n_trt_compounds": int(out.loc[out.Metadata_pert_type == "trt", "Metadata_InChIKey"].nunique()),
        "n_control_compounds": int(out.loc[(out.Metadata_pert_type == "control") & (out.Metadata_control_type != "negcon"), "Metadata_InChIKey"].nunique()),
        "trt_with_moa": int(out.loc[out.Metadata_pert_type == "trt"].drop_duplicates("Metadata_InChIKey")["Metadata_moa"].notna().sum()),
        "unmapped_wells": int(out.Metadata_pert_type.isna().sum()),
        "plates_per_source": out.groupby("Metadata_Source").Metadata_Plate.nunique().to_dict(),
    }
    json.dump(s, open(f"{D}/target2_summary.json", "w"), indent=2, default=str)
    print(json.dumps(s, indent=2, default=str))

if __name__ == "__main__":
    main()
