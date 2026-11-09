# Summary (experiments/results/harmony, tag='_global9', probs=raw, score=nlp)

Folds: 9 sources × 1 seeds = 9 runs

## Table 1 — ID vs OOD metrics per held-out lab (mean over seeds)

| lab | plates | K | acc trt ID→OOD | acc active ID→OOD | ECE active ID→OOD | NLL active ID→OOD | AURC active ID→OOD | false-naming ID→OOD | poscon acc ID→OOD | flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| source_10 | 6 | 84 | 0.832→0.834 | 0.535→0.290 | 0.287→0.516 | 3.13→6.51 | 0.216→0.628 | 0.079→0.003 | 0.740→0.602 | 0.50 |
| source_11 | 7 | 84 | 0.934→0.816 | 0.731→0.200 | 0.110→0.593 | 1.36→5.64 | 0.193→0.824 | 0.005→0.000 | 0.901→0.528 | 1.00 |
| source_2 | 10 | 84 | 0.966→0.799 | 0.862→0.287 | 0.056→0.560 | 0.50→5.93 | 0.046→0.714 | 0.002→0.051 | 0.940→0.505 | 1.00 |
| source_3 | 25 | 81 | 0.901→0.791 | 0.554→0.388 | 0.267→0.234 | 2.52→2.23 | 0.246→0.482 | 0.002→0.096 | 0.777→0.701 | 0.08 |
| source_5 | 24 | 84 | 0.943→0.835 | 0.766→0.285 | 0.089→0.431 | 1.34→3.43 | 0.138→0.630 | 0.006→0.004 | 0.915→0.603 | 0.17 |
| source_6 | 21 | 80 | 0.926→0.825 | 0.663→0.193 | 0.184→0.663 | 1.70→6.46 | 0.288→0.886 | 0.002→0.000 | 0.838→0.603 | 0.10 |
| source_7 | 7 | 82 | 0.937→0.777 | 0.742→0.005 | 0.063→0.972 | 1.10→17.95 | 0.143→1.000 | 0.007→0.001 | 0.918→0.400 | 1.00 |
| source_8 | 4 | 84 | 0.943→0.838 | 0.770→0.314 | 0.064→0.457 | 1.03→4.09 | 0.132→0.671 | 0.007→0.007 | 0.895→0.621 | 0.25 |
| source_9 | 9 | 90 | 0.929→0.751 | 0.730→0.000 | 0.149→0.969 | 1.36→19.23 | 0.124→1.000 | 0.008→0.009 | 0.900→0.329 | 1.00 |
| **mean** | | | 0.923→0.807 | 0.706→0.218 | 0.141→0.599 | 1.56→7.94 | 0.170→0.759 | 0.013→0.019 | 0.869→0.544 | 0.57 |

## Table 2trt/aps — conformal coverage / set size on OOD treatment wells (aps score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.944 / 3.2 | 0.837±0.074 / 2.2 | — | 0.978±0.011 / 30.1 | 0.971±0.013 / 28.1 | 0.838±0.075 / 2.1 | 0.870±0.090 / 4.4 | 0.950±0.002 / 19.7 | 8/9 | 5/9 |
| 0.1 | 0.898 / 1.8 | 0.782±0.072 / 1.5 | — | 0.956±0.017 / 21.4 | 0.939±0.026 / 19.4 | 0.781±0.072 / 1.4 | 0.823±0.092 / 2.0 | 0.897±0.004 / 11.8 | 8/9 | 5/9 |
| 0.2 | 0.793 / 1.1 | 0.688±0.068 / 1.1 | — | 0.911±0.026 / 15.4 | 0.879±0.040 / 11.6 | 0.687±0.066 / 1.0 | 0.743±0.099 / 1.3 | 0.795±0.007 / 3.1 | 8/9 | 5/9 |

## Table 2trt/nlp — conformal coverage / set size on OOD treatment wells (nlp score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.938 / 1.7 | 0.845±0.078 / 1.6 | — | 0.982±0.015 / 30.1 | 0.959±0.018 / 21.6 | 0.855±0.079 / 1.6 | 0.871±0.083 / 2.3 | 0.951±0.001 / 17.8 | 7/9 | 7/9 |
| 0.1 | 0.880 / 0.9 | 0.761±0.083 / 0.9 | — | 0.957±0.015 / 19.7 | 0.910±0.025 / 13.0 | 0.789±0.058 / 1.0 | 0.815±0.078 / 1.2 | 0.901±0.001 / 10.3 | 9/9 | 6/9 |
| 0.2 | 0.793 / 0.8 | 0.671±0.206 / 0.8 | — | 0.914±0.021 / 13.9 | 0.851±0.064 / 8.7 | 0.717±0.104 / 0.8 | 0.749±0.105 / 0.9 | 0.801±0.001 / 2.8 | 5/9 | 5/9 |

## Table 2act/aps — conformal coverage / set size on OOD active-compound treatment wells (aps score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.848 / 4.4 | 0.420±0.281 / 3.1 | — | 0.902±0.050 / 37.3 | 0.875±0.055 / 34.5 | 0.422±0.281 / 3.1 | 0.521±0.327 / 8.2 | 0.788±0.010 / 23.9 | 9/9 | 9/9 |
| 0.1 | 0.792 / 2.5 | 0.356±0.239 / 2.1 | — | 0.814±0.070 / 26.8 | 0.758±0.076 / 23.1 | 0.353±0.236 / 2.1 | 0.443±0.281 / 4.2 | 0.598±0.044 / 13.0 | 9/9 | 9/9 |
| 0.2 | 0.691 / 1.4 | 0.279±0.186 / 1.4 | — | 0.656±0.069 / 16.9 | 0.571±0.072 / 12.8 | 0.279±0.185 / 1.4 | 0.345±0.218 / 2.1 | 0.358±0.140 / 3.8 | 9/9 | 9/9 |

## Table 2act/nlp — conformal coverage / set size on OOD active-compound treatment wells (nlp score); mean over folds (±sd over folds)

| α | ID cov/size | std | TS? | v1 poscon | v2 match | v3 DMSO-w | v4 gated | oracle | folds with std cov < 1−α−0.02 | folds with v4 cov < 1−α−0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.771 / 2.9 | 0.380±0.299 / 2.9 | — | 0.921±0.068 / 37.9 | 0.819±0.080 / 25.8 | 0.405±0.307 / 3.0 | 0.491±0.315 / 4.8 | 0.785±0.007 / 21.5 | 9/9 | 9/9 |
| 0.1 | 0.616 / 0.8 | 0.175±0.118 / 0.8 | — | 0.811±0.070 / 23.9 | 0.617±0.101 / 14.5 | 0.209±0.155 / 0.9 | 0.318±0.225 / 1.7 | 0.573±0.026 / 11.8 | 9/9 | 9/9 |
| 0.2 | 0.449 / 0.5 | 0.104±0.074 / 0.5 | — | 0.634±0.077 / 15.1 | 0.418±0.117 / 9.6 | 0.123±0.083 / 0.6 | 0.191±0.134 / 0.9 | 0.194±0.099 / 2.7 | 9/9 | 9/9 |

## Table 3 — selective prediction among NAMED calls (prediction ≠ inactive), threshold chosen on ID calibration for target risk

| target risk | ID risk / coverage | OOD risk / coverage (ID threshold) | OOD canary-retuned risk / coverage | OOD oracle risk / coverage | folds with OOD risk > target+0.05 |
|---|---|---|---|---|---|
| 0.1 | 0.164 / 0.932 | 0.595 / 0.831 | 0.223 / 0.075 | 0.075 / 0.043 | 9/9 |
| 0.2 | 0.189 / 0.997 | 0.640 / 0.993 | 0.257 / 0.097 | 0.194 / 0.135 | 9/9 |

Unthresholded risk among named calls: ID 0.191, OOD 0.642; fraction of treatment wells named: ID 0.200, OOD 0.123.

## Table 4 — plate-level canary analysis (OOD plates, std CP at α=0.1)

n plates = 113; Spearman(poscon acc, std CP trt coverage) = 0.245 (p=8.8e-03); Spearman(poscon acc, trt acc) = 0.755 (p=4.9e-22); Spearman(poscon acc, active acc) = 0.734 (p=2.2e-20)
flagged plates: 45/113; mean std-CP coverage flagged 0.726 vs unflagged 0.767; mean active acc flagged 0.158 vs unflagged 0.306