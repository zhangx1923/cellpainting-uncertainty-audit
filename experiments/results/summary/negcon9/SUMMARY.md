# Summary (experiments/results/harmony, tag='_negcon9', probs=raw, score=nlp)

Folds: 9 sources × 3 seeds = 27 runs

## Table 1 — ID vs OOD metrics per held-out lab (mean over seeds)

| lab | plates | K | acc trt ID→OOD | acc active ID→OOD | ECE active ID→OOD | NLL active ID→OOD | AURC active ID→OOD | false-naming ID→OOD | poscon acc ID→OOD | flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| source_10 | 6 | 92 | 0.915→0.834 | 0.673→0.341 | 0.206→0.394 | 1.86→3.44 | 0.139→0.623 | 0.003→0.000 | 0.814→0.653 | 0.67 |
| source_11 | 7 | 90 | 0.899→0.824 | 0.601→0.289 | 0.248→0.519 | 2.18→4.09 | 0.191→0.712 | 0.004→0.003 | 0.767→0.588 | 0.43 |
| source_2 | 10 | 91 | 0.934→0.889 | 0.741→0.610 | 0.140→0.189 | 1.29→1.74 | 0.123→0.209 | 0.002→0.021 | 0.866→0.774 | 0.40 |
| source_3 | 25 | 89 | 0.893→0.857 | 0.564→0.410 | 0.307→0.400 | 2.58→3.57 | 0.210→0.576 | 0.002→0.001 | 0.742→0.691 | 0.13 |
| source_5 | 24 | 87 | 0.896→0.884 | 0.565→0.509 | 0.283→0.237 | 2.46→2.14 | 0.263→0.404 | 0.003→0.001 | 0.756→0.781 | 0.00 |
| source_6 | 21 | 87 | 0.896→0.906 | 0.568→0.614 | 0.262→0.185 | 2.33→1.82 | 0.214→0.263 | 0.003→0.003 | 0.759→0.829 | 0.03 |
| source_7 | 7 | 90 | 0.936→0.789 | 0.749→0.184 | 0.138→0.533 | 1.25→8.94 | 0.100→0.765 | 0.003→0.015 | 0.874→0.483 | 1.00 |
| source_8 | 4 | 91 | 0.961→0.899 | 0.848→0.613 | 0.056→0.176 | 0.54→1.63 | 0.045→0.187 | 0.002→0.007 | 0.937→0.824 | 0.25 |
| source_9 | 9 | 97 | 0.955→0.730 | 0.842→0.000 | 0.068→0.997 | 0.65→12.74 | 0.051→1.000 | 0.004→0.001 | 0.929→0.333 | 1.00 |
| **mean** | | | 0.921→0.846 | 0.683→0.397 | 0.190→0.403 | 1.68→4.46 | 0.148→0.526 | 0.003→0.006 | 0.827→0.662 | 0.43 |

## Table 2trt/aps — conformal coverage / set size on OOD treatment wells (aps score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.935 / 6.7 | 0.875±0.078 / 4.6 | — | 0.977±0.008 / 52.5 | 0.959±0.019 / 40.8 | 0.883±0.072 / 4.5 | 0.896±0.085 / 15.3 | 0.950±0.003 / 31.9 | 22/27 | 13/27 |
| 0.1 | 0.884 / 2.9 | 0.822±0.075 / 1.9 | — | 0.947±0.015 / 32.2 | 0.918±0.039 / 24.8 | 0.833±0.070 / 2.0 | 0.845±0.084 / 5.5 | 0.899±0.005 / 15.3 | 21/27 | 15/27 |
| 0.2 | 0.788 / 1.3 | 0.728±0.066 / 1.1 | — | 0.884±0.031 / 17.7 | 0.834±0.072 / 14.6 | 0.737±0.063 / 1.1 | 0.748±0.083 / 1.6 | 0.799±0.008 / 4.3 | 21/27 | 17/27 |

## Table 2trt/nlp — conformal coverage / set size on OOD treatment wells (nlp score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.912 / 2.7 | 0.859±0.084 / 2.6 | — | 0.977±0.010 / 40.2 | 0.956±0.018 / 24.2 | 0.874±0.065 / 1.2 | 0.886±0.081 / 3.9 | 0.951±0.001 / 22.1 | 20/27 | 15/27 |
| 0.1 | 0.820 / 0.9 | 0.796±0.071 / 0.9 | — | 0.949±0.014 / 21.8 | 0.910±0.030 / 16.4 | 0.831±0.050 / 1.0 | 0.834±0.074 / 1.3 | 0.901±0.001 / 11.4 | 21/27 | 19/27 |
| 0.2 | 0.709 / 0.7 | 0.697±0.092 / 0.8 | — | 0.900±0.013 / 14.9 | 0.864±0.039 / 12.1 | 0.754±0.050 / 0.8 | 0.769±0.087 / 0.9 | 0.801±0.001 / 3.8 | 20/27 | 12/27 |

## Table 2act/aps — conformal coverage / set size on OOD active-compound treatment wells (aps score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.815 / 5.9 | 0.597±0.264 / 3.3 | — | 0.908±0.032 / 33.6 | 0.853±0.050 / 26.1 | 0.610±0.256 / 3.4 | 0.656±0.281 / 7.1 | 0.816±0.021 / 21.0 | 27/27 | 27/27 |
| 0.1 | 0.739 / 3.1 | 0.527±0.244 / 2.1 | — | 0.810±0.046 / 21.5 | 0.755±0.066 / 18.0 | 0.538±0.242 / 2.2 | 0.572±0.252 / 3.1 | 0.685±0.067 / 11.8 | 27/27 | 27/27 |
| 0.2 | 0.636 / 1.5 | 0.440±0.212 / 1.4 | — | 0.676±0.056 / 13.4 | 0.610±0.071 / 11.7 | 0.447±0.207 / 1.5 | 0.466±0.213 / 1.6 | 0.513±0.140 / 4.6 | 27/27 | 27/27 |

## Table 2act/nlp — conformal coverage / set size on OOD active-compound treatment wells (nlp score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.704 / 2.8 | 0.460±0.313 / 2.7 | — | 0.905±0.040 / 29.8 | 0.822±0.076 / 21.4 | 0.505±0.241 / 1.7 | 0.564±0.294 / 3.3 | 0.804±0.011 / 18.4 | 24/27 | 27/27 |
| 0.1 | 0.584 / 0.8 | 0.305±0.221 / 0.7 | — | 0.794±0.059 / 18.6 | 0.647±0.115 / 14.3 | 0.351±0.190 / 0.9 | 0.409±0.234 / 1.3 | 0.608±0.024 / 10.2 | 27/27 | 27/27 |
| 0.2 | 0.485 / 0.6 | 0.212±0.154 / 0.5 | — | 0.607±0.058 / 12.2 | 0.501±0.105 / 10.2 | 0.234±0.151 / 0.5 | 0.308±0.184 / 0.8 | 0.300±0.084 / 3.6 | 27/27 | 27/27 |

## Table 3 — selective prediction among NAMED calls (prediction ≠ inactive), threshold chosen on ID calibration for target risk

| target risk | ID risk / coverage | OOD risk / coverage (ID threshold) | OOD canary-retuned risk / coverage | OOD oracle risk / coverage | folds with OOD risk > target+0.05 |
|---|---|---|---|---|---|
| 0.1 | 0.106 / 0.987 | 0.431 / 0.940 | 0.258 / 0.298 | 0.085 / 0.276 | 27/27 |
| 0.2 | 0.108 / 0.991 | 0.433 / 0.947 | 0.261 / 0.303 | 0.190 / 0.506 | 24/27 |

Unthresholded risk among named calls: ID 0.113, OOD 0.439; fraction of treatment wells named: ID 0.190, OOD 0.152.

## Table 4 — plate-level canary analysis (OOD plates, std CP at α=0.1)

n plates = 339; Spearman(poscon acc, std CP trt coverage) = 0.637 (p=5.1e-40); Spearman(poscon acc, trt acc) = 0.890 (p=5.4e-117); Spearman(poscon acc, active acc) = 0.875 (p=4.4e-108)
flagged plates: 96/339; mean std-CP coverage flagged 0.747 vs unflagged 0.837; mean active acc flagged 0.232 vs unflagged 0.517