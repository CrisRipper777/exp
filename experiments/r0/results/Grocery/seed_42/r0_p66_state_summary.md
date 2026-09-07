# R0-P6.6: Functional Teacher Necessity & Receiver-State Audit

_Sections A-C state observed numbers; 'Observed facts' and 'Exploratory interpretation' are separated at the end._

- n_relations=3998 | n_receivers=2602 | dataset=Grocery seed=42 | confidence-recompute invariant max dev=4.77e-07

## A. Exact utility regret (evaluated on EXACT Q of the chosen action)

| subset | est | acc | mean reg | med reg | p90 | p95 | harmful | retention |
|---|---|---|---|---|---|---|---|---|
| all | first | 0.955 | 0.00169 | 0.00000 | 0.00000 | 0.00000 | 0.013 | 0.985 |
| all | second | 0.892 | 0.00433 | 0.00000 | 0.00001 | 0.00553 | 0.000 | 0.961 |
| all | hybrid | 0.889 | 0.00413 | 0.00000 | 0.00001 | 0.00226 | 0.014 | 0.963 |
| receiver_correct_full_True | first | 0.955 | 0.00055 | 0.00000 | 0.00000 | 0.00000 | 0.009 | 0.993 |
| receiver_correct_full_True | second | 0.868 | 0.00511 | 0.00000 | 0.00031 | 0.01055 | 0.000 | 0.940 |
| receiver_correct_full_True | hybrid | 0.874 | 0.00219 | 0.00000 | 0.00006 | 0.00261 | 0.010 | 0.974 |
| receiver_correct_full_False | first | 0.957 | 0.00652 | 0.00000 | 0.00000 | 0.00000 | 0.028 | 0.971 |
| receiver_correct_full_False | second | 0.993 | 0.00103 | 0.00000 | 0.00000 | 0.00000 | 0.000 | 0.996 |
| receiver_correct_full_False | hybrid | 0.950 | 0.01234 | 0.00000 | 0.00000 | 0.00000 | 0.033 | 0.946 |
| absS_norm_gt_0.05 | first | 0.922 | 0.00501 | 0.00000 | 0.00000 | 0.00128 | 0.030 | 0.965 |
| absS_norm_gt_0.05 | second | 0.725 | 0.01274 | 0.00000 | 0.01600 | 0.06536 | 0.001 | 0.912 |
| absS_norm_gt_0.05 | hybrid | 0.720 | 0.01332 | 0.00000 | 0.00844 | 0.04304 | 0.034 | 0.908 |
| absS_norm_gt_0.10 | first | 0.890 | 0.00647 | 0.00000 | 0.00002 | 0.00650 | 0.038 | 0.959 |
| absS_norm_gt_0.10 | second | 0.612 | 0.01862 | 0.00000 | 0.04359 | 0.11044 | 0.002 | 0.883 |
| absS_norm_gt_0.10 | hybrid | 0.570 | 0.02418 | 0.00000 | 0.03143 | 0.09279 | 0.049 | 0.848 |
| S_exact_gt_0 | first | 0.949 | 0.00152 | 0.00000 | 0.00000 | 0.00000 | 0.015 | 0.988 |
| S_exact_gt_0 | second | 0.934 | 0.00320 | 0.00000 | 0.00000 | 0.00043 | 0.001 | 0.975 |
| S_exact_gt_0 | hybrid | 0.884 | 0.00702 | 0.00000 | 0.00005 | 0.00595 | 0.020 | 0.945 |
| S_exact_lt_0 | first | 0.965 | 0.00186 | 0.00000 | 0.00000 | 0.00000 | 0.011 | 0.981 |
| S_exact_lt_0 | second | 0.858 | 0.00540 | 0.00000 | 0.00084 | 0.01516 | 0.000 | 0.946 |
| S_exact_lt_0 | hybrid | 0.897 | 0.00152 | 0.00000 | 0.00000 | 0.00101 | 0.009 | 0.985 |

- mean Q_star=0.1122 | frac rows Q_star=0 (no useful action): 0.097

## B. Receiver uncertainty quartiles (rank quartiles over CSV receivers)

### confidence (Q1 smallest -> Q4 largest)

| quartile | n_recv | n_rel | recv corr | acc f/s/h | mean reg f/s/h | harmful f/s/h | mean\|Sn\| | ΔRegret (f−s) |
|---|---|---|---|---|---|---|---|---|
| Q1 | 651 | 972 | 0.513 | 0.938/0.977/0.934 | 0.00420/0.00336/0.01171 | 0.024/0.000/0.029 | 0.0497 | +0.00084 |
| Q2 | 650 | 961 | 0.789 | 0.955/0.950/0.922 | 0.00080/0.00355/0.00343 | 0.015/0.000/0.017 | 0.0485 | -0.00274 |
| Q3 | 651 | 995 | 0.928 | 0.962/0.858/0.880 | 0.00191/0.00762/0.00156 | 0.012/0.001/0.010 | 0.0539 | -0.00571 |
| Q4 | 650 | 1070 | 0.977 | 0.965/0.793/0.825 | 0.00000/0.00286/0.00025 | 0.002/0.000/0.002 | 0.0548 | -0.00286 |

### entropy_norm (Q1 smallest -> Q4 largest)

| quartile | n_recv | n_rel | recv corr | acc f/s/h | mean reg f/s/h | harmful f/s/h | mean\|Sn\| | ΔRegret (f−s) |
|---|---|---|---|---|---|---|---|---|
| Q1 | 651 | 1071 | 0.975 | 0.965/0.793/0.824 | 0.00000/0.00283/0.00025 | 0.002/0.000/0.002 | 0.0548 | -0.00282 |
| Q2 | 650 | 998 | 0.929 | 0.962/0.864/0.882 | 0.00189/0.00688/0.00155 | 0.011/0.001/0.010 | 0.0535 | -0.00499 |
| Q3 | 651 | 970 | 0.768 | 0.948/0.943/0.922 | 0.00126/0.00472/0.00456 | 0.018/0.000/0.021 | 0.0517 | -0.00346 |
| Q4 | 650 | 959 | 0.534 | 0.945/0.980/0.935 | 0.00380/0.00297/0.01070 | 0.022/0.000/0.025 | 0.0468 | +0.00083 |

### margin (Q1 smallest -> Q4 largest)

| quartile | n_recv | n_rel | recv corr | acc f/s/h | mean reg f/s/h | harmful f/s/h | mean\|Sn\| | ΔRegret (f−s) |
|---|---|---|---|---|---|---|---|---|
| Q1 | 651 | 977 | 0.522 | 0.939/0.979/0.939 | 0.00423/0.00371/0.01159 | 0.026/0.000/0.029 | 0.0499 | +0.00052 |
| Q2 | 650 | 954 | 0.778 | 0.956/0.953/0.921 | 0.00075/0.00305/0.00347 | 0.013/0.000/0.017 | 0.0479 | -0.00230 |
| Q3 | 651 | 996 | 0.931 | 0.963/0.853/0.876 | 0.00191/0.00754/0.00160 | 0.012/0.001/0.010 | 0.0549 | -0.00563 |
| Q4 | 650 | 1071 | 0.975 | 0.964/0.795/0.826 | 0.00000/0.00305/0.00026 | 0.002/0.000/0.002 | 0.0542 | -0.00305 |

## C. Receiver-cluster bootstrap (10,000 resamples; cluster = receiver_i)

| state | n_recv | ΔRegret f−s obs | 95% CI | ΔActionAcc s−f obs | 95% CI |
|---|---|---|---|---|---|
| overall | 2602 | -0.00264 | [-0.00396, -0.00130] | -0.06353 | [-0.07552, -0.05152] |
| correct_False | 516 | +0.00549 | [+0.00151, +0.01058] | +0.03675 | [+0.02187, +0.05298] |
| conf_Q1 | 651 | +0.00084 | [-0.00248, +0.00447] | +0.03909 | [+0.02222, +0.05711] |
| entropy_Q4 | 650 | +0.00083 | [-0.00251, +0.00441] | +0.03545 | [+0.01833, +0.05268] |
| margin_Q1 | 651 | +0.00052 | [-0.00287, +0.00410] | +0.03992 | [+0.02273, +0.05765] |

## Figures

- fig_p66_1_regret_comparison.png
- fig_p66_2_confidence_vs_regret.png
- fig_p66_3_entropy_vs_regret.png
- fig_p66_4_margin_vs_regret.png
- fig_p66_5_second_minus_first_by_state.png

## Observed facts (auto-derived, no interpretation)

- A(all): mean regret first=0.00169 / second=0.00433 / hybrid=0.00413 vs mean Q_star=0.1122; harmful rate first=0.013 / second=0.000 / hybrid=0.014; utility retention 0.985/0.961/0.963
- A(receiver_correct_full_False): lowest mean regret = second (0.00103); second harmful rate 0.000
- A(absS_norm_gt_0.10): lowest mean regret = first (0.00647); second harmful rate 0.002
- A(S_exact_lt_0): lowest mean regret = hybrid (0.00152); second harmful rate 0.000
- C(overall): DeltaRegret obs=-0.00264 95% CI=[-0.00396, -0.00130] | DeltaActionAcc obs=-0.06353 95% CI=[-0.07552, -0.05152]
- C(correct_False): DeltaRegret obs=+0.00549 95% CI=[+0.00151, +0.01058] | DeltaActionAcc obs=+0.03675 95% CI=[+0.02187, +0.05298]
- C(conf_Q1): DeltaRegret obs=+0.00084 95% CI=[-0.00248, +0.00447] | DeltaActionAcc obs=+0.03909 95% CI=[+0.02222, +0.05711]
- C(entropy_Q4): DeltaRegret obs=+0.00083 95% CI=[-0.00251, +0.00441] | DeltaActionAcc obs=+0.03545 95% CI=[+0.01833, +0.05268]
- C(margin_Q1): DeltaRegret obs=+0.00052 95% CI=[-0.00287, +0.00410] | DeltaActionAcc obs=+0.03992 95% CI=[+0.02273, +0.05765]

## Exploratory interpretation (labeled; single dataset/seed 42, one probe; correlational)

- Estimator-action mistakes are almost always near-tie decisions: first-order's mean regret is ~1.5% of mean Q_star and its p95 regret rounds to 0; a replacement action teacher must beat an ~0.002 nats regret baseline to be necessary.
- Full second-order is never harmful in any reported subset (harmful rate 0.000), consistent with its curvature penalties suppressing negative-gain picks, but on strong-interaction rows it loses 2.9x the utility of first-order (|S_norm|>0.10: 0.0186 vs 0.0065), i.e. its conservatism costs on genuinely joint-benefit relations.
- Receiver-state stratification reproduces the P6.5 correct=False result at the label-free level: second-order's ActionAcc advantage is concentrated and CI-excluding-0 among low-confidence / high-entropy / small-margin receivers, while it is a clear disadvantage on confident receivers; regret-level CIs for the quartile states straddle 0, so the utility-level advantage is established only for the incorrect-receiver state (+0.0015..+0.0106).
- No new method is proposed or validated here; nothing above should be read as claiming a routing/teacher method works.
