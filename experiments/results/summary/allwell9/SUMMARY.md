# Summary (experiments/results/harmony, tag='_allwell9', probs=raw, score=nlp)

Folds: 9 sources × 1 seeds = 9 runs

## Table 1 — ID vs OOD metrics per held-out lab (mean over seeds)

| lab | plates | K | acc trt ID→OOD | acc active ID→OOD | ECE active ID→OOD | NLL active ID→OOD | AURC active ID→OOD | false-naming ID→OOD | poscon acc ID→OOD | flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| source_10 | 6 | 87 | 0.903→0.856 | 0.602→0.396 | 0.263→0.352 | 2.32→3.71 | 0.171→0.627 | 0.003→0.000 | 0.769→0.716 | 0.17 |
| source_11 | 7 | 86 | 0.950→0.844 | 0.800→0.350 | 0.066→0.446 | 0.81→3.60 | 0.081→0.714 | 0.004→0.004 | 0.938→0.602 | 0.86 |
| source_2 | 10 | 85 | 0.971→0.914 | 0.877→0.679 | 0.037→0.150 | 0.38→1.87 | 0.025→0.184 | 0.001→0.017 | 0.943→0.784 | 1.00 |
| source_3 | 25 | 83 | 0.905→0.865 | 0.586→0.406 | 0.274→0.372 | 2.34→2.99 | 0.162→0.602 | 0.002→0.000 | 0.772→0.709 | 0.08 |
| source_5 | 24 | 84 | 0.963→0.902 | 0.847→0.570 | 0.055→0.185 | 0.64→2.12 | 0.065→0.400 | 0.003→0.001 | 0.941→0.760 | 0.00 |
| source_6 | 21 | 83 | 0.949→0.918 | 0.786→0.640 | 0.065→0.160 | 0.82→1.67 | 0.091→0.214 | 0.004→0.002 | 0.905→0.860 | 0.10 |
| source_7 | 7 | 86 | 0.956→0.807 | 0.824→0.206 | 0.049→0.421 | 0.71→6.40 | 0.068→0.814 | 0.004→0.009 | 0.926→0.490 | 1.00 |
| source_8 | 4 | 85 | 0.961→0.901 | 0.838→0.592 | 0.071→0.174 | 0.64→1.66 | 0.060→0.216 | 0.002→0.006 | 0.926→0.883 | 0.00 |
| source_9 | 9 | 90 | 0.951→0.756 | 0.809→0.000 | 0.107→0.995 | 0.89→13.95 | 0.062→1.000 | 0.004→0.002 | 0.918→0.333 | 1.00 |
| **mean** | | | 0.945→0.863 | 0.774→0.427 | 0.110→0.362 | 1.06→4.22 | 0.087→0.530 | 0.003→0.005 | 0.893→0.682 | 0.47 |

## Table 2trt/aps — conformal coverage / set size on OOD treatment wells (aps score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.958 / 5.1 | 0.899±0.070 / 5.3 | — | 0.974±0.011 / 48.1 | 0.959±0.021 / 40.4 | 0.898±0.070 / 5.1 | 0.907±0.073 / 12.3 | 0.948±0.003 / 25.7 | 7/9 | 5/9 |
| 0.1 | 0.912 / 2.1 | 0.842±0.070 / 2.1 | — | 0.944±0.019 / 26.6 | 0.912±0.041 / 17.0 | 0.847±0.071 / 2.2 | 0.849±0.074 / 4.1 | 0.894±0.005 / 9.5 | 7/9 | 6/9 |
| 0.2 | 0.809 / 1.1 | 0.740±0.060 / 1.1 | — | 0.873±0.042 / 11.7 | 0.824±0.069 / 8.5 | 0.747±0.065 / 1.2 | 0.749±0.075 / 1.3 | 0.787±0.008 / 3.1 | 7/9 | 6/9 |

## Table 2trt/nlp — conformal coverage / set size on OOD treatment wells (nlp score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.947 / 2.1 | 0.890±0.075 / 2.8 | — | 0.974±0.009 / 36.1 | 0.944±0.023 / 15.5 | 0.895±0.058 / 1.3 | 0.901±0.062 / 2.0 | 0.951±0.001 / 16.7 | 6/9 | 6/9 |
| 0.1 | 0.879 / 0.9 | 0.817±0.065 / 0.9 | — | 0.948±0.014 / 16.7 | 0.902±0.028 / 8.5 | 0.855±0.052 / 1.0 | 0.843±0.062 / 1.0 | 0.901±0.001 / 7.3 | 8/9 | 5/9 |
| 0.2 | 0.793 / 0.8 | 0.715±0.111 / 0.8 | — | 0.902±0.023 / 8.4 | 0.853±0.047 / 6.7 | 0.794±0.063 / 0.9 | 0.774±0.070 / 0.9 | 0.801±0.001 / 2.7 | 6/9 | 4/9 |

## Table 2act/aps — conformal coverage / set size on OOD active-compound treatment wells (aps score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.891 / 4.4 | 0.658±0.266 / 5.2 | — | 0.898±0.036 / 33.6 | 0.861±0.044 / 27.5 | 0.660±0.264 / 5.0 | 0.690±0.275 / 8.5 | 0.813±0.027 / 20.0 | 9/9 | 9/9 |
| 0.1 | 0.832 / 2.4 | 0.590±0.250 / 2.9 | — | 0.814±0.033 / 20.4 | 0.749±0.042 / 14.6 | 0.597±0.249 / 3.0 | 0.607±0.246 / 3.7 | 0.692±0.081 / 10.1 | 9/9 | 9/9 |
| 0.2 | 0.730 / 1.4 | 0.495±0.219 / 1.7 | — | 0.681±0.050 / 11.4 | 0.621±0.056 / 9.2 | 0.500±0.216 / 1.9 | 0.505±0.214 / 1.9 | 0.536±0.163 / 3.8 | 9/9 | 9/9 |

## Table 2act/nlp — conformal coverage / set size on OOD active-compound treatment wells (nlp score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.802 / 2.7 | 0.550±0.304 / 4.5 | — | 0.889±0.038 / 28.2 | 0.764±0.090 / 16.1 | 0.561±0.236 / 2.0 | 0.599±0.239 / 3.7 | 0.791±0.009 / 17.0 | 9/9 | 9/9 |
| 0.1 | 0.676 / 0.8 | 0.328±0.218 / 0.8 | — | 0.777±0.064 / 16.5 | 0.602±0.087 / 9.3 | 0.413±0.207 / 1.0 | 0.403±0.200 / 1.0 | 0.587±0.026 / 8.0 | 9/9 | 9/9 |
| 0.2 | 0.561 / 0.6 | 0.217±0.150 / 0.5 | — | 0.600±0.083 / 9.4 | 0.456±0.110 / 6.9 | 0.265±0.165 / 0.6 | 0.281±0.153 / 0.6 | 0.276±0.100 / 2.4 | 9/9 | 9/9 |

## Table 3 — selective prediction among NAMED calls (prediction ≠ inactive), threshold chosen on ID calibration for target risk

| target risk | ID risk / coverage | OOD risk / coverage (ID threshold) | OOD canary-retuned risk / coverage | OOD oracle risk / coverage | folds with OOD risk > target+0.05 |
|---|---|---|---|---|---|
| 0.1 | 0.108 / 0.988 | 0.400 / 0.936 | 0.383 / 0.320 | 0.097 / 0.286 | 9/9 |
| 0.2 | 0.108 / 0.988 | 0.400 / 0.936 | 0.383 / 0.320 | 0.199 / 0.585 | 6/9 |

Unthresholded risk among named calls: ID 0.115, OOD 0.411; fraction of treatment wells named: ID 0.203, OOD 0.148.

## Table 4 — plate-level canary analysis (OOD plates, std CP at α=0.1)

n plates = 113; Spearman(poscon acc, std CP trt coverage) = 0.548 (p=3.5e-10); Spearman(poscon acc, trt acc) = 0.896 (p=6.6e-41); Spearman(poscon acc, active acc) = 0.878 (p=2.1e-37)
flagged plates: 37/113; mean std-CP coverage flagged 0.785 vs unflagged 0.847; mean active acc flagged 0.320 vs unflagged 0.537