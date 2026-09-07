# R0 Interaction Diagnostics Summary (observed data only)

- dataset=Grocery | train_seed=42 | analysis_seed=12345 | sampled relations=3998

## A. Interaction statistics
- mean(S_exact)=+0.00982 | median(S_exact)=-0.00000 | mean(|S_exact|)=0.02531 | median(|S_exact|)=0.00099
- mean(S_norm)=+0.01037 | median(S_norm)=-0.00089 | mean(|S_norm|)=0.05182 | median(|S_norm|)=0.02421
- P(|S_norm|>0.05)=0.295 | P(|S_norm|>0.10)=0.143 | P(|S_norm|>0.20)=0.047
- P(S>0)=0.474 | P(S<0)=0.522 | |S| q10=5.93e-06 q50=9.87e-04

## B. Optimal action distribution (exact)
- NULL: 387 (9.7%) | TEXT: 668 (16.7%) | VISUAL: 1022 (25.6%) | JOINT: 1921 (48.0%)
- per-degree-bin shares: bin0: NULL=0.08, TEXT=0.21, VISUAL=0.24, JOINT=0.48; bin1: NULL=0.09, TEXT=0.18, VISUAL=0.25, JOINT=0.48; bin2: NULL=0.09, TEXT=0.17, VISUAL=0.25, JOINT=0.50; bin3: NULL=0.11, TEXT=0.14, VISUAL=0.28, JOINT=0.47
- by receiver-correct-full: False: NULL=0.24, TEXT=0.31, VISUAL=0.25, JOINT=0.20; True: NULL=0.06, TEXT=0.13, VISUAL=0.26, JOINT=0.55

## C. Semantic proxy
- Spearman(sim_text, Q_text)=+0.071 | Pearson=+0.010
- Spearman(sim_visual, Q_visual)=+0.011 | Pearson=-0.069
- Spearman(sim_text * sim_visual, S_exact)=+0.016 (exploratory only)

## D. Approximation quality (exact vs first/second order)
- text:   Spearman 1st=+0.967 2nd=+0.974 | Kendall 1st=+0.942 2nd=+0.968
- visual: Spearman 1st=+0.965 2nd=+0.926 | Kendall 1st=+0.927 2nd=+0.907
- joint:  Spearman 1st=+0.967 2nd=+0.893 | Kendall 1st=+0.928 2nd=+0.872
- pooled: Spearman 1st=+0.967 2nd=+0.927 | Kendall 1st=+0.933 2nd=+0.913

## E. Action prediction
- ActionAcc(first vs exact)=0.955 | ActionAcc(second vs exact)=0.892 | most-common-exact prior=0.480
- confusion (rows=exact, cols=second): [[379, 1, 4, 3], [5, 649, 0, 14], [25, 2, 939, 56], [28, 144, 150, 1599]]

## F. Interaction sign (second vs exact)
- Spearman(S_exact, S_second)=+0.922 | Pearson=+0.750
- SignAcc over |S_exact|>eps_s samples (eps_s=5.93e-06, n=3598)=0.967

_This summary states observed numbers only; no hypothesis conclusion is drawn here._
