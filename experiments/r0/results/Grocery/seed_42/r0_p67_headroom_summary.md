# R0-P6.7: Oracle Routing Headroom & Policy Deployability Audit

_Sections A-G state observed numbers; 'Observed facts' and 'Exploratory interpretation' are separated at the end. No training; no router; validation labels only (oracle diagnostics); frozen Grocery seed-42 probe._

- relation CSV (sampled): n=3998 | full val edge set: n=27524 edges into 3415 receivers | all four CSV invariants re-asserted on the full set (Check-4 dev 4.44e-15, linearity 2.67e-15, overlap best-action agree 1.0000/3998)

## A. Routing opportunity vs JOINT (sampled relations)

- D_exact = Q_star − Q_joint ≥ 0 (CE units): mean 0.0821, median 5.24e-06, p90 1.69e-01, p95 4.08e-01, p99 1.58e+00, max 5.790
- 48.5% of relations have D=0 (JOINT already the oracle-best action); mean Q_star = 0.112 (90.3% relations have a useful action)
- concentration of ΣD_exact over top-k% relations (ranked desc): top1% → 32.1%, top5% → 69.6%, top10% → 86.0%, top20% → 97.2%, top50% → 100.0%
- retention of first-order per-edge decisions: 97.9% (Σ realized 321.42 / Σ opportunity 328.18); second-order: 94.7%
- best_exact action distribution (sampled): null 387, text 668, visual 1022, joint 1921

## B. On-error conditional regrets (rows where the estimator action ≠ oracle best)

| subset | est | n_errors | mean | median | p90 | p95 | p99 | max | n(regret>0) |
|---|---|---|---|---|---|---|---|---|---|
| all | first | 178 | 3.80e-02 | 1.9e-03 | 9.2e-02 | 2.1e-01 | 4.7e-01 | 1.12e+00 | 163 |
| all | second | 432 | 4.01e-02 | 4.2e-03 | 1.1e-01 | 2.2e-01 | 5.3e-01 | 5.81e-01 | 416 |
| receiver correct | first | 145 | 1.23e-02 | 8.0e-04 | 3.6e-02 | 8.4e-02 | 1.2e-01 | 1.90e-01 | 130 |
| receiver correct | second | 427 | 3.87e-02 | 3.9e-03 | 1.1e-01 | 1.9e-01 | 5.3e-01 | 5.81e-01 | 411 |
| receiver incorrect | first | 33 | 1.51e-01 | 4.6e-02 | 3.5e-01 | 5.4e-01 | 1.0e+00 | 1.12e+00 | 33 |
| receiver incorrect | second | 5 | 1.57e-01 | 9.4e-02 | 3.5e-01 | 3.6e-01 | 3.7e-01 | 3.78e-01 | 5 |
| |S_norm|>0.05 | first | 92 | 6.43e-02 | 6.2e-03 | 1.8e-01 | 3.0e-01 | 8.2e-01 | 1.12e+00 | 87 |
| |S_norm|>0.05 | second | 324 | 4.64e-02 | 7.3e-03 | 1.3e-01 | 2.4e-01 | 5.5e-01 | 5.81e-01 | 318 |
| |S_norm|>0.10 | first | 63 | 5.88e-02 | 5.4e-03 | 1.2e-01 | 2.8e-01 | 6.6e-01 | 1.12e+00 | 59 |
| |S_norm|>0.10 | second | 222 | 4.80e-02 | 8.7e-03 | 1.3e-01 | 2.4e-01 | 5.3e-01 | 5.81e-01 | 218 |
| S_exact>0 | first | 96 | 3.01e-02 | 3.9e-03 | 8.8e-02 | 1.9e-01 | 2.9e-01 | 3.72e-01 | 95 |
| S_exact>0 | second | 126 | 4.82e-02 | 5.5e-03 | 1.4e-01 | 3.0e-01 | 5.3e-01 | 5.48e-01 | 124 |
| S_exact<0 | first | 72 | 5.37e-02 | 1.1e-03 | 9.6e-02 | 2.7e-01 | 8.8e-01 | 1.12e+00 | 68 |
| S_exact<0 | second | 296 | 3.80e-02 | 4.4e-03 | 1.1e-01 | 1.7e-01 | 4.6e-01 | 5.81e-01 | 292 |

## C. Full-receiver simultaneous policies (all val incoming relations, one-shot at Joint reference)

| policy | CE | ΔCE vs Joint | val acc | macro-F1 | P(improved) | P(worsened) | P(unchanged) |
|---|---|---|---|---|---|---|---|
| Joint-All | 0.7994 | +0.0000 | 79.21% | 0.6929 | 0.000 | 0.000 | 1.000 |
| Text-All | 1.0504 | +0.2510 | 69.28% | 0.5800 | 0.190 | 0.810 | 0.000 |
| Visual-All | 0.9163 | +0.1169 | 75.49% | 0.6482 | 0.284 | 0.716 | 0.000 |
| Null-All | 1.0510 | +0.2517 | 69.93% | 0.5838 | 0.165 | 0.835 | 0.000 |
| Exact-OS | 0.5311 | -0.2683 | 84.74% | 0.7583 | 0.736 | 0.018 | 0.246 |
| First-OS | 0.5346 | -0.2648 | 84.51% | 0.7614 | 0.705 | 0.029 | 0.266 |
| Second-OS | 0.5453 | -0.2541 | 84.74% | 0.7579 | 0.668 | 0.197 | 0.136 |
- full-edge action distributions (n=27524): exact null/text/visual/joint = 3211/4245/7071/12997 | first = 2648/4190/7271/13415 | second = 2938/4554/7427/12605

## D. Local-to-global compositionality (per receiver, over full val edges)

| policy | n | Pearson | Spearman | ΣG_global/ΣG_local | receivers w/ G_local>0 | sign-agree among those | frac G_global<0 |
|---|---|---|---|---|---|---|---|
| Exact-OneShot | 3415 | 0.962 | 0.977 | 0.818 | 2558 | 0.980 | 0.018 |
| FirstOrder-OneShot | 3415 | 0.962 | 0.980 | 0.820 | 2453 | 0.979 | 0.029 |
| Second-OneShot | 3415 | 0.955 | 0.895 | 0.787 | 2453 | 0.896 | 0.197 |

## E. Selective routing curves (top-k% of eligible edges, one-shot)

- eligible: exact D>0 on 13807 edges (50.2%); first-order D_pred>0 on 14109 edges (51.3%)
- Joint-All baseline: CE 0.7994, acc 79.21%

| estimator | frac | CE | acc | macro-F1 |
|---|---|---|---|---|
| exact | 0.00 | 0.7994 | 79.21% | 0.6929 |
| exact | 0.01 | 0.7124 | 79.77% | 0.6984 |
| exact | 0.05 | 0.6248 | 81.90% | 0.7209 |
| exact | 0.10 | 0.5831 | 83.19% | 0.7368 |
| exact | 0.20 | 0.5471 | 84.33% | 0.7569 |
| exact | 0.30 | 0.5355 | 84.74% | 0.7649 |
| exact | 0.50 | 0.5313 | 84.80% | 0.7613 |
| exact | 1.00 | 0.5311 | 84.74% | 0.7583 |
| first | 0.00 | 0.7994 | 79.21% | 0.6929 |
| first | 0.01 | 0.7150 | 79.68% | 0.6971 |
| first | 0.05 | 0.6285 | 81.70% | 0.7190 |
| first | 0.10 | 0.5872 | 82.99% | 0.7347 |
| first | 0.20 | 0.5508 | 84.13% | 0.7532 |
| first | 0.30 | 0.5391 | 84.51% | 0.7615 |
| first | 0.50 | 0.5348 | 84.57% | 0.7643 |
| first | 1.00 | 0.5346 | 84.51% | 0.7614 |

## F. Four-action argmax vs two independent sign gates (first-order Q1)

- gate mapping (Q1_text>0, Q1_visual>0) == argmax over [0, Q1_text, Q1_visual, Q1_joint]: mismatches 0/3998 (CSV) and 0/27524 (full val edge set)
- max |Q1_joint − (Q1_text + Q1_visual)| = 2.22e-15 (CSV, fp32 storage)

## Figures

- fig_p67_1_Dexact_distribution.png
- fig_p67_2_gain_concentration.png
- fig_p67_3_policy_metrics.png
- fig_p67_4_local_vs_global.png
- fig_p67_5_selective_curves.png
- fig_p67_6_onerror_regret.png

## Observed facts (auto-derived, no interpretation)

- A: mean D_exact 0.0821; 48.5% D=0; top1% holds 32.1% of ΣD; top5% holds 69.6% of ΣD; top10% holds 86.0% of ΣD; top20% holds 97.2% of ΣD; top50% holds 100.0% of ΣD
- A: retention first 0.979, second 0.947
- B(on-error): all/first mean=3.80e-02 n=178 | all/second mean=4.01e-02 n=432; receiver correct/first mean=1.23e-02 n=145 | receiver correct/second mean=3.87e-02 n=427; receiver incorrect/first mean=1.51e-01 n=33 | receiver incorrect/second mean=1.57e-01 n=5; |S_norm|>0.05/first mean=6.43e-02 n=92 | |S_norm|>0.05/second mean=4.64e-02 n=324; |S_norm|>0.10/first mean=5.88e-02 n=63 | |S_norm|>0.10/second mean=4.80e-02 n=222; S_exact>0/first mean=3.01e-02 n=96 | S_exact>0/second mean=4.82e-02 n=126; S_exact<0/first mean=5.37e-02 n=72 | S_exact<0/second mean=3.80e-02 n=296
- C: one-shot exact/first/second gain over Joint-All: ΔCE -0.268/-0.265/-0.254; acc 79.21% -> 84.74%/84.51%/84.74%; P(improved) 0.736/0.705/0.668
- D: ΣG_global/ΣG_local exact 0.818, first 0.820; Pearson 0.962/0.962; Spearman 0.977/0.980
- E: routing 100% of eligible exact/first reaches CE 0.5311/0.5346, acc 84.74%/84.51%
- F: 0/27524 gate-vs-argmax mismatches on the full val edge set (0/3998 CSV)
- action distributions full-edge: exact best_exact null/text/visual/joint = 3211/4245/7071/12997

## Exploratory interpretation (labeled; single dataset/seed, one frozen probe)

- Per-edge exact utility, applied one-shot to all incoming relations of every validation receiver, shows a large downstream headroom over sending the full text+visual message on every edge: mean val CE drops 0.799→0.531 and accuracy rises 79.2%→84.7% while ~47% of edges are left untouched; receivers improve on 73.6% of nodes and worsen on 1.8%. The opportunity is heavily concentrated: top 1% of relations by D_exact hold 32% of the total, top 10% hold 86%, and routing only 20% of eligible edges already captures ~5.1pp of the ~5.5pp accuracy gain (10% of eligible → +4.0pp).
- First-order estimates capture most of the oracle headroom in the one-shot setting (val acc 84.5% vs oracle 84.7%; retention ~0.98 per-edge on sampled relations), so the headroom is not contingent on exact counterfactuals being available at deploy time — but the first-order per-edge decisions still presuppose knowing the receiver's task outcome (soft labels) at reference state.
- Realized global gains are smaller than the per-edge sums (Σ-ratio ~0.80): per-receiver CE is not additive in per-edge utility, and compositionality is imperfect even though the intervention is exactly linear in the representation — the residual is the classifier nonlinearity over simultaneously moved representations.
- These are oracle/surrogate measurements on one dataset and one frozen checkpoint; no routing mechanism, teacher or policy is proposed or validated here, and no claim is made that such gains transfer to deployed systems (labels-at-reference is not available at inference).
