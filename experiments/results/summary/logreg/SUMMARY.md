# Summary (results/ablation, tag='_logreg', probs=raw, score=nlp)

Folds: 11 sources × 1 seeds = 11 runs

## Table 1 — ID vs OOD metrics per held-out lab (mean over seeds)

| lab | plates | K | acc trt ID→OOD | acc active ID→OOD | ECE active ID→OOD | NLL active ID→OOD | AURC active ID→OOD | false-naming ID→OOD | poscon acc ID→OOD | flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| source_10 | 6 | 90 | 0.871→0.818 | 0.514→0.292 | 0.376→0.516 | 3.53→6.28 | 0.253→0.639 | 0.015→0.014 | 0.716→0.610 | 0.00 |
| source_11 | 7 | 90 | 0.867→0.822 | 0.522→0.421 | 0.361→0.400 | 3.58→4.76 | 0.255→0.405 | 0.020→0.047 | 0.730→0.619 | 0.00 |
| source_13 | 6 | 90 | 0.870→0.771 | 0.524→0.143 | 0.369→0.705 | 3.51→7.43 | 0.255→0.844 | 0.017→0.021 | 0.719→0.482 | 0.00 |
| source_2 | 10 | 88 | 0.887→0.838 | 0.588→0.538 | 0.303→0.333 | 3.01→4.03 | 0.175→0.296 | 0.019→0.070 | 0.746→0.691 | 0.00 |
| source_3 | 25 | 82 | 0.936→0.869 | 0.741→0.496 | 0.157→0.358 | 1.52→4.04 | 0.118→0.304 | 0.009→0.026 | 0.886→0.687 | 0.04 |
| source_4 | 22 | 92 | 0.848→0.766 | 0.422→0.337 | 0.462→0.491 | 4.22→7.37 | 0.308→0.588 | 0.010→0.092 | 0.630→0.601 | 0.14 |
| source_5 | 24 | 88 | 0.897→0.868 | 0.608→0.497 | 0.314→0.362 | 2.76→4.57 | 0.156→0.370 | 0.012→0.016 | 0.781→0.683 | 0.00 |
| source_6 | 21 | 85 | 0.858→0.859 | 0.401→0.494 | 0.491→0.392 | 4.42→4.89 | 0.316→0.336 | 0.008→0.035 | 0.623→0.722 | 0.00 |
| source_7 | 7 | 88 | 0.869→0.798 | 0.524→0.334 | 0.357→0.470 | 3.49→6.38 | 0.252→0.509 | 0.023→0.057 | 0.728→0.562 | 0.00 |
| source_8 | 4 | 87 | 0.865→0.871 | 0.503→0.533 | 0.359→0.333 | 3.49→4.46 | 0.290→0.317 | 0.024→0.025 | 0.730→0.750 | 0.00 |
| source_9 | 9 | 91 | 0.945→0.751 | 0.790→0.000 | 0.126→0.992 | 1.31→12.72 | 0.089→1.000 | 0.005→0.004 | 0.888→0.331 | 1.00 |
| **mean** | | | 0.883→0.821 | 0.558→0.371 | 0.334→0.487 | 3.17→6.08 | 0.224→0.510 | 0.015→0.037 | 0.743→0.613 | 0.11 |

## Table 2trt/aps — conformal coverage / set size on OOD treatment wells (aps score); mean over folds (±sd over folds)

| α | ID cov/size | std | v1 poscon | v2 match | v3 DMSO-w | v4 gated | v5 act-Mondrian | oracle | folds std under | folds v5 under |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.943 / 16.2 | 0.895±0.060 / 13.6 | 0.978±0.006 / 75.2 | 0.956±0.009 / 57.1 | 0.916±0.062 / 31.2 | 0.933±0.065 / 48.7 | — | 0.951±0.001 / 52.1 | 8/11 | —/11 |
| 0.1 | 0.903 / 4.1 | 0.855±0.065 / 3.1 | 0.959±0.009 / 62.1 | 0.909±0.026 / 21.5 | 0.880±0.066 / 10.6 | 0.884±0.065 / 14.0 | — | 0.899±0.002 / 16.4 | 6/11 | —/11 |
| 0.2 | 0.809 / 1.2 | 0.754±0.055 / 1.1 | 0.917±0.019 / 24.7 | 0.824±0.054 / 10.1 | 0.800±0.065 / 1.5 | 0.794±0.067 / 3.9 | — | 0.793±0.007 / 2.7 | 5/11 | —/11 |

## Table 2trt/nlp — conformal coverage / set size on OOD treatment wells (nlp score); mean over folds (±sd over folds)

| α | ID cov/size | std | v1 poscon | v2 match | v3 DMSO-w | v4 gated | v5 act-Mondrian | oracle | folds std under | folds v5 under |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.943 / 13.1 | 0.903±0.066 / 11.5 | 0.981±0.007 / 75.5 | 0.954±0.010 / 43.0 | 0.918±0.062 / 21.3 | 0.932±0.061 / 34.7 | — | 0.951±0.001 / 41.2 | 5/11 | —/11 |
| 0.1 | 0.902 / 2.3 | 0.861±0.066 / 2.1 | 0.958±0.009 / 54.1 | 0.901±0.024 / 13.4 | 0.878±0.060 / 4.0 | 0.881±0.048 / 6.5 | — | 0.901±0.001 / 11.4 | 5/11 | —/11 |
| 0.2 | 0.779 / 0.8 | 0.732±0.062 / 0.8 | 0.915±0.019 / 14.5 | 0.824±0.033 / 5.3 | 0.791±0.062 / 1.0 | 0.807±0.037 / 1.0 | — | 0.801±0.001 / 2.3 | 10/11 | —/11 |

## Table 2act/aps — conformal coverage / set size on OOD active-compound treatment wells (aps score); mean over folds (±sd over folds)

| α | ID cov/size | std | v1 poscon | v2 match | v3 DMSO-w | v4 gated | v5 act-Mondrian | oracle | folds std under | folds v5 under |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.771 / 8.8 | 0.607±0.214 / 4.3 | 0.916±0.017 / 42.8 | 0.829±0.040 / 25.2 | 0.677±0.236 / 10.0 | 0.742±0.247 / 17.1 | — | 0.806±0.025 / 22.7 | 11/11 | —/11 |
| 0.1 | 0.655 / 3.3 | 0.508±0.187 / 2.1 | 0.839±0.027 / 26.8 | 0.663±0.085 / 12.3 | 0.571±0.203 / 4.1 | 0.583±0.202 / 5.2 | — | 0.624±0.036 / 9.5 | 11/11 | —/11 |
| 0.2 | 0.532 / 1.2 | 0.397±0.166 / 1.3 | 0.686±0.055 / 12.2 | 0.513±0.110 / 7.8 | 0.438±0.166 / 1.5 | 0.445±0.170 / 1.9 | — | 0.436±0.108 / 2.8 | 11/11 | —/11 |

## Table 2act/nlp — conformal coverage / set size on OOD active-compound treatment wells (nlp score); mean over folds (±sd over folds)

| α | ID cov/size | std | v1 poscon | v2 match | v3 DMSO-w | v4 gated | v5 act-Mondrian | oracle | folds std under | folds v5 under |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.763 / 7.9 | 0.633±0.238 / 5.3 | 0.924±0.030 / 43.6 | 0.820±0.046 / 22.4 | 0.683±0.240 / 8.8 | 0.733±0.246 / 14.5 | — | 0.808±0.025 / 21.4 | 11/11 | —/11 |
| 0.1 | 0.628 / 2.1 | 0.504±0.199 / 2.2 | 0.838±0.028 / 24.7 | 0.626±0.088 / 10.0 | 0.546±0.209 / 3.0 | 0.550±0.194 / 3.5 | — | 0.620±0.041 / 8.6 | 11/11 | —/11 |
| 0.2 | 0.479 / 0.8 | 0.299±0.149 / 0.7 | 0.674±0.058 / 10.6 | 0.434±0.090 / 5.4 | 0.370±0.157 / 1.0 | 0.382±0.150 / 1.1 | — | 0.373±0.100 / 2.3 | 11/11 | —/11 |

## Table 3 — selective prediction among NAMED calls (prediction ≠ inactive), threshold chosen on ID calibration for target risk

| target risk | ID risk / coverage | OOD risk / coverage (ID threshold) | OOD canary-retuned risk / coverage | OOD oracle risk / coverage | folds with OOD risk > target+0.05 |
|---|---|---|---|---|---|
| 0.1 | 0.116 / 0.597 | 0.482 / 0.429 | 0.321 / 0.056 | 0.033 / 0.021 | 11/11 |
| 0.2 | 0.211 / 0.874 | 0.566 / 0.792 | 0.457 / 0.486 | 0.116 / 0.079 | 11/11 |

Unthresholded risk among named calls: ID 0.273, OOD 0.625; fraction of treatment wells named: ID 0.183, OOD 0.212.

## Table 4 — plate-level canary analysis (OOD plates, std CP at α=0.1)

n plates = 141; Spearman(poscon acc, std CP trt coverage) = 0.376 (p=4.3e-06); Spearman(poscon acc, trt acc) = 0.746 (p=2.7e-26); Spearman(poscon acc, active acc) = 0.784 (p=1.5e-30)
flagged plates: 13/141; mean std-CP coverage flagged 0.743 vs unflagged 0.864; mean active acc flagged 0.060 vs unflagged 0.443