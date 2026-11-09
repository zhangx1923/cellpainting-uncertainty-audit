#!/bin/bash
# Download the public JUMP-CP data used in the paper into ./data (about 6 GB).
#   scripts/download_data.sh            # metadata + 141 TARGET2 plate profiles
#   scripts/download_data.sh --harmony  # additionally the consortium Harmony profiles (optional baseline)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/meta data/target2
J=https://raw.githubusercontent.com/jump-cellpainting/datasets/main/metadata
for f in plate.csv.gz well.csv.gz compound.csv.gz; do curl -fsSL -o data/meta/$f $J/$f; done
T=https://raw.githubusercontent.com/jump-cellpainting/JUMP-Target/master
for f in JUMP-Target-2_compound_metadata.tsv JUMP-Target-2_compound_platemap.tsv; do curl -fsSL -o data/meta/$f $T/$f; done
curl -fsSL -o data/meta/repurposing_info.tsv https://raw.githubusercontent.com/broadinstitute/lincs-cell-painting/master/metadata/moa/repurposing_info.tsv
# TARGET2 plate list and profile URLs (cpg0016-jump, public Cell Painting Gallery bucket)
python3 - <<'PY'
import pandas as pd
p = pd.read_csv("data/meta/plate.csv.gz"); p = p[p.Metadata_PlateType == "TARGET2"]
p.to_csv("data/meta/target2_plates.csv", index=False)
with open("data/target2_urls.txt", "w") as f:
    for r in p.itertuples():
        f.write(f"https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/{r.Metadata_Source}/workspace/profiles/{r.Metadata_Batch}/{r.Metadata_Plate}/{r.Metadata_Plate}.parquet\n")
print(len(p), "TARGET2 plates")
PY
xargs -P 8 -I{} sh -c 'f=data/target2/$(basename {}); [ -s "$f" ] || curl -fsSL -o "$f" {}' < data/target2_urls.txt
echo "plates downloaded: $(ls data/target2/*.parquet | wc -l) (expected 141)"
if [ "${1:-}" = "--harmony" ]; then
  # Consortium Harmony-corrected profiles (cpg0016-jump-assembled, COMPOUND v1.0). This path was valid on 2026-10-22;
  # if it has moved, find the current location in the jump-cellpainting/datasets manifests.
  H=https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/profiles_assembled/COMPOUND/v1.0/profiles_var_mad_int_featselect_harmony.parquet
  mkdir -p data/assembled; curl -fSL -o data/assembled/profiles_var_mad_int_featselect_harmony.parquet $H
  python3 - <<'PY'
import pandas as pd, pyarrow.parquet as pq
t2 = list(set(pd.read_csv("data/meta/target2_plates.csv").Metadata_Plate))
pq.read_table("data/assembled/profiles_var_mad_int_featselect_harmony.parquet", filters=[("Metadata_Plate", "in", t2)]).to_pandas().to_parquet("data/target2_harmony.parquet", index=False)
PY
fi
python3 experiments/prepare_data.py
