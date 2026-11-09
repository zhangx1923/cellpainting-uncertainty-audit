# Summary (experiments/results/harmony, tag='_harmony', probs=raw, score=nlp)

Folds: 9 sources × 3 seeds = 27 runs

## Table 1 — ID vs OOD metrics per held-out lab (mean over seeds)

| lab | plates | K | acc trt ID→OOD | acc active ID→OOD | ECE active ID→OOD | NLL active ID→OOD | AURC active ID→OOD | false-naming ID→OOD | poscon acc ID→OOD | flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| source_10 | 6 | 107 | 0.892→0.843 | 0.662→0.493 | 0.217→0.325 | 1.86→2.56 | 0.118→0.427 | 0.008→0.005 | 0.790→0.717 | 0.22 |
| source_11 | 7 | 106 | 0.874→0.828 | 0.607→0.433 | 0.255→0.385 | 2.35→2.88 | 0.177→0.466 | 0.010→0.002 | 0.745→0.625 | 0.24 |
| source_2 | 10 | 105 | 0.922→0.912 | 0.748→0.741 | 0.151→0.126 | 1.23→0.96 | 0.094→0.106 | 0.003→0.016 | 0.867→0.852 | 0.17 |
| source_3 | 25 | 101 | 0.873→0.856 | 0.567→0.517 | 0.310→0.307 | 2.71→2.25 | 0.169→0.318 | 0.003→0.007 | 0.741→0.745 | 0.00 |
| source_5 | 24 | 104 | 0.870→0.888 | 0.566→0.620 | 0.316→0.224 | 2.90→1.75 | 0.219→0.205 | 0.005→0.003 | 0.730→0.847 | 0.00 |
| source_6 | 21 | 101 | 0.874→0.892 | 0.576→0.623 | 0.293→0.229 | 2.47→1.66 | 0.209→0.242 | 0.007→0.001 | 0.763→0.838 | 0.11 |
| source_7 | 7 | 107 | 0.924→0.731 | 0.763→0.203 | 0.148→0.634 | 1.33→5.54 | 0.082→0.824 | 0.006→0.037 | 0.872→0.469 | 1.00 |
| source_8 | 4 | 104 | 0.957→0.899 | 0.862→0.680 | 0.049→0.151 | 0.46→1.28 | 0.035→0.148 | 0.004→0.010 | 0.927→0.896 | 0.08 |
| source_9 | 9 | 118 | 0.925→0.645 | 0.801→0.000 | 0.102→0.968 | 0.80→15.58 | 0.075→1.000 | 0.011→0.017 | 0.908→0.264 | 1.00 |
| **mean** | | | 0.901→0.833 | 0.684→0.479 | 0.205→0.372 | 1.79→3.83 | 0.131→0.415 | 0.006→0.011 | 0.816→0.695 | 0.31 |

## Table 2trt/aps — conformal coverage / set size on OOD treatment wells (aps score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.925 / 5.6 | 0.872±0.098 / 3.8 | — | 0.971±0.010 / 32.9 | 0.958±0.017 / 28.8 | 0.873±0.101 / 3.4 | 0.899±0.107 / 8.9 | 0.950±0.002 / 22.8 | 20/27 | 10/27 |
| 0.1 | 0.876 / 2.9 | 0.820±0.092 / 1.9 | — | 0.934±0.018 / 22.7 | 0.908±0.043 / 20.1 | 0.821±0.094 / 1.7 | 0.843±0.103 / 3.8 | 0.898±0.005 / 13.8 | 21/27 | 14/27 |
| 0.2 | 0.781 / 1.4 | 0.728±0.081 / 1.1 | — | 0.861±0.041 / 14.6 | 0.817±0.076 / 13.1 | 0.729±0.085 / 1.1 | 0.745±0.102 / 1.5 | 0.798±0.008 / 6.7 | 21/27 | 13/27 |

## Table 2trt/nlp — conformal coverage / set size on OOD treatment wells (nlp score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.903 / 3.2 | 0.853±0.113 / 2.8 | — | 0.971±0.011 / 26.7 | 0.951±0.026 / 23.3 | 0.868±0.103 / 1.3 | 0.884±0.112 / 4.3 | 0.951±0.001 / 18.6 | 20/27 | 14/27 |
| 0.1 | 0.820 / 1.0 | 0.780±0.134 / 1.0 | — | 0.935±0.016 / 18.8 | 0.902±0.051 / 16.7 | 0.822±0.102 / 1.0 | 0.820±0.126 / 1.4 | 0.901±0.001 / 11.5 | 20/27 | 15/27 |
| 0.2 | 0.695 / 0.7 | 0.643±0.157 / 0.7 | — | 0.879±0.034 / 12.9 | 0.851±0.059 / 12.1 | 0.727±0.120 / 0.8 | 0.735±0.163 / 0.9 | 0.801±0.001 / 6.2 | 20/27 | 11/27 |

## Table 2act/aps — conformal coverage / set size on OOD active-compound treatment wells (aps score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.806 / 5.0 | 0.657±0.270 / 2.4 | — | 0.910±0.033 / 23.1 | 0.877±0.043 / 20.7 | 0.657±0.275 / 2.2 | 0.719±0.290 / 4.0 | 0.853±0.021 / 17.2 | 27/27 | 23/27 |
| 0.1 | 0.732 / 2.9 | 0.583±0.249 / 1.7 | — | 0.824±0.039 / 17.5 | 0.782±0.063 / 16.1 | 0.582±0.251 / 1.6 | 0.626±0.261 / 2.2 | 0.739±0.048 / 11.9 | 27/27 | 27/27 |
| 0.2 | 0.629 / 1.5 | 0.490±0.217 / 1.2 | — | 0.698±0.045 / 12.4 | 0.643±0.068 / 11.5 | 0.489±0.218 / 1.2 | 0.513±0.223 / 1.3 | 0.578±0.102 / 6.6 | 27/27 | 27/27 |

## Table 2act/nlp — conformal coverage / set size on OOD active-compound treatment wells (nlp score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.731 / 3.0 | 0.561±0.312 / 2.4 | — | 0.904±0.041 / 22.0 | 0.841±0.082 / 19.5 | 0.590±0.288 / 1.6 | 0.656±0.303 / 3.0 | 0.839±0.012 / 16.5 | 22/27 | 25/27 |
| 0.1 | 0.648 / 1.0 | 0.453±0.280 / 1.1 | — | 0.789±0.058 / 16.4 | 0.701±0.128 / 14.9 | 0.475±0.254 / 1.0 | 0.519±0.259 / 1.4 | 0.680±0.023 / 11.3 | 26/27 | 27/27 |
| 0.2 | 0.533 / 0.6 | 0.302±0.185 / 0.6 | — | 0.629±0.085 / 11.8 | 0.572±0.120 / 11.4 | 0.348±0.203 / 0.7 | 0.398±0.213 / 0.8 | 0.436±0.061 / 6.0 | 27/27 | 27/27 |

## Table 3 — selective prediction among NAMED calls (prediction ≠ inactive), threshold chosen on ID calibration for target risk

| target risk | ID risk / coverage | OOD risk / coverage (ID threshold) | OOD canary-retuned risk / coverage | OOD oracle risk / coverage | folds with OOD risk > target+0.05 |
|---|---|---|---|---|---|
| 0.1 | 0.097 / 0.994 | 0.336 / 0.988 | 0.109 / 0.405 | 0.094 / 0.493 | 26/27 |
| 0.2 | 0.097 / 0.996 | 0.338 / 0.993 | 0.109 / 0.407 | 0.192 / 0.724 | 10/27 |

Unthresholded risk among named calls: ID 0.100, OOD 0.341; fraction of treatment wells named: ID 0.228, OOD 0.187.

## Table 4 — plate-level canary analysis (OOD plates, std CP at α=0.1)

n plates = 339; Spearman(poscon acc, std CP trt coverage) = 0.580 (p=8.3e-32); Spearman(poscon acc, trt acc) = 0.853 (p=3.3e-97); Spearman(poscon acc, active acc) = 0.846 (p=6.2e-94)
flagged plates: 70/339; mean std-CP coverage flagged 0.679 vs unflagged 0.840; mean active acc flagged 0.239 vs unflagged 0.590