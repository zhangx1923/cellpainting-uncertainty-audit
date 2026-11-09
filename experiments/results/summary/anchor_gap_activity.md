# Anchor-gap and activity-split analysis (from per-well outputs, 33 runs, alpha=0.1, -log p score)

## Anchor vs target score-distribution distance (Kolmogorov) and realised coverage gaps, mean over runs

ks_pos_vs_active                0.319
ks_pos_vs_looksactive           0.184
ks_dmso_vs_looksinactive        0.080
ks_pos_vs_alltrt                0.239
cov_v1_active                   0.784
gap_v1_active                  -0.116
cov_posanchor_looksactive       0.858
gap_posanchor_looksactive      -0.042
cov_dmsoanchor_looksinactive    0.857
gap_dmsoanchor_looksinactive   -0.043
bound_ok_v1                     1.000
bound_ok_looksactive            1.000
bound_ok_looksinactive          1.000

bound holds: v1 active 33/33, looks-active 33/33, looks-inactive 33/33

## Activity split under shift (fraction of wells with phenotype score > training DMSO 95th pct)

       seed  frac_active_wells_look_active  frac_inactive_wells_look_active  frac_dmso_look_active  frac_pos_look_active    n_act
split                                                                                                                            
idt     1.0                          0.813                            0.117                  0.049                 0.586  903.121
ood     1.0                          0.828                            0.122                  0.046                 0.607  946.242

per lab (OOD) fraction of training-active compound wells that look active:
holdout
source_10    0.790
source_11    0.968
source_13    0.678
source_2     0.942
source_3     0.973
source_4     0.799
source_5     0.940
source_6     0.993
source_7     0.985
source_8     0.988
source_9     0.052