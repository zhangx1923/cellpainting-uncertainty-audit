#!/bin/bash
# Re-run every experiment and regenerate all tables, figures and numbers of the paper.
# Needs the data from scripts/download_data.sh (use --harmony for the Harmony baseline) and one CUDA GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
R=experiments/results; S=$R/summary
H9="source_10 source_11 source_2 source_3 source_5 source_6 source_7 source_8 source_9"   # the nine labs covered by consortium Harmony profiles
# 1. main leave-one-laboratory-out sweep (11 labs x 3 seeds) and the 64-component rerun
python3 experiments/loso.py --seeds 0 1 2 --outdir $R/loso_v5 --save-wells
python3 experiments/loso.py --pcs 64 --seeds 0 1 2 --outdir $R/loso_pcs64 --tag _pcs64 --save-wells
# 2. model-side repairs: seed-0 screen of all remedies, then 3 seeds for the default and the selected remedy
python3 experiments/loso.py --seeds 0 1 2 --outdir $R/strat --tag _base
python3 experiments/loso.py --seeds 0 1 2 --outdir $R/strat --tag _aug3ens5 --aug 0.3 --ens 5
python3 experiments/loso.py --seeds 0 --outdir $R/strat --tag _ens5 --ens 5
python3 experiments/loso.py --seeds 0 --outdir $R/strat --tag _aug3 --aug 0.3
python3 experiments/loso.py --seeds 0 --outdir $R/strat --tag _aug5 --aug 0.5
python3 experiments/loso.py --seeds 0 --outdir $R/strat --tag _sphere --sphere lab
python3 experiments/loso.py --seeds 0 --outdir $R/strat --tag _alignlab --align lab
# 3. ablations and baselines
python3 experiments/loso.py --clf logreg --seeds 0 --outdir $R/ablation --tag _logreg
python3 experiments/loso.py --pcs 64 --seeds 0 --outdir $R/ablation --tag _pcs64
python3 experiments/loso.py --active-pct 99 --seeds 0 --outdir $R/ablation --tag _pct99
if [ -f data/target2_harmony.parquet ]; then
  python3 experiments/loso.py --feat harmony --sources $H9 --seeds 0 1 2 --outdir $R/harmony --tag _harmony
  for n in negcon allwell global; do python3 experiments/loso.py --norm $n --sources $H9 --seeds 0 1 2 --outdir $R/harmony --tag _${n}9; done
fi
# 4. aggregation, statistics, figures, tables, numbers
python3 experiments/aggregate.py --dir $R/loso_v5 --out $S/main
python3 experiments/aggregate.py --dir $R/loso_v5 --out $S/main_ts --pk ts
python3 experiments/aggregate.py --dir $R/loso_pcs64 --tag _pcs64 --out $S/pcs64full
for t in logreg pcs64 pct99; do python3 experiments/aggregate.py --dir $R/ablation --tag _$t --out $S/$t; done
if [ -d $R/harmony ]; then for t in harmony negcon9 allwell9 global9; do python3 experiments/aggregate.py --dir $R/harmony --tag _$t --out $S/$t; done; fi
python3 experiments/analysis_revision.py
python3 experiments/analysis_revision.py --wells $R/loso_pcs64 --summary $S/pcs64full --out $S/revision_pcs64
python3 experiments/analysis_anchor_gap.py
python3 experiments/screen_compare.py --dir $R/strat --seeds 0 --out $S/repair
python3 experiments/screen_compare.py --dir $R/strat --tags _base _aug3ens5 --out $S/repair3
mkdir -p paper/figures paper/tables
python3 experiments/figures.py; python3 experiments/make_tables.py; python3 experiments/make_numbers.py
echo "done: paper/figures, paper/tables, paper/numbers.tex"
