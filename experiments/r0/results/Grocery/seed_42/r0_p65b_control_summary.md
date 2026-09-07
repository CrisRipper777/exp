# R0-P6.5B: Real Pair vs Shuffled Pair Control (observed data only)

- dataset=Grocery | matched pairs=2535 | eps_s=4.22e-06
- control: same receiver i; real (T from j, V from j) vs shuffled (T from j, V from k, k!=j); k chosen to match message/edge norm (median |log alpha_ratio|=0.378, median |log normV ratio|=0.449)

## Interaction magnitude (real vs shuffled)
- |S_exact| mean: real=0.01127 vs shuffle=0.00798 (median 0.00055 vs 0.00061) | wilcoxon p=3.43e-01
- |S_norm| mean: real=0.04378 vs shuffle=0.04201 | P(|S_norm|>0.1): real=0.117 vs shuffle=0.104
- |S_second| mean: real=0.01325 vs shuffle=0.00948 (median 0.00069 vs 0.00070) | wilcoxon p=5.39e-01

## Sign systematics (S2 vs S_exact)
- real:    P(S>0)=0.464 | mean(S)=+0.00434 | sign-acc(>eps, n=2281)=0.981 | Spearman(S,S2)=+0.966
- shuffle: P(S>0)=0.499 | mean(S)=+0.00304 | sign-acc(>eps, n=2289)=0.979 | Spearman(S,S2)=+0.968

## Task effect (best achievable gain on the receiver)
- Q_best mean: real=+0.07531 vs shuffle=+0.07556 (median +0.00753 vs +0.00749) | wilcoxon p=9.78e-01
- P(Q_best_real > Q_best_shuffle)=0.478 (incl. ties=0.527)

## Far-tail comparison (|S| thresholds, McNemar on discordant pairs)
- |S|>0.01: P_real=0.166 vs P_shuf=0.158 (real-only=147, shuf-only=125, p=2.03e-01) | |S|>0.02: P_real=0.107 vs P_shuf=0.095 (real-only=125, shuf-only=96, p=5.94e-02) | |S|>0.05: P_real=0.047 vs P_shuf=0.036 (real-only=79, shuf-only=52, p=2.27e-02) | |S|>0.1: P_real=0.020 vs P_shuf=0.013 (real-only=40, shuf-only=21, p=2.04e-02)
- top-10% |S| rows (n=254): real mean=0.0879 vs shuffle mean=0.0558
- matched corr(|S_real|, |S_shuffle|)=0.376

## Robustness: better-matched half (combined mismatch <= median)
- n=1268 | |S_exact| mean real=0.00729 vs shuffle=0.00750 (p=7.76e-01) | |S_second| mean real=0.00875 vs shuffle=0.00915 (p=3.53e-01) | Q_best mean real=+0.06692 vs shuffle=+0.06678 (p=8.45e-01)

_Observed numbers only._
