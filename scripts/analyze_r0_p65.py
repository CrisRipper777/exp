"""R0-P6.5 analysis: Hybrid Functional Teacher (A) + Real-vs-Shuffled control (B).

Pure post-processing; never re-runs a model, never modifies the input CSVs.

A (reads r0_relation_diagnostics.csv):
    Q_text_h   = Q_text_1        (CSV col, first order)
    Q_visual_h = Q_visual_1
    Q_joint_h  = Q_joint_1 + S_second     (first-order main effects +
                                           second-order cross interaction)
    best_hybrid = argmax([0, Q_text_h, Q_visual_h, Q_joint_h])
  Reports joint-level and pooled Spearman/Kendall vs exact, ActionAcc of
  first / full-second / hybrid, hybrid confusion, and per-subset ActionAcc
  (|S_norm|>0.05, |S_norm|>0.10, S>0, S<0, receiver correct/incorrect).

B (reads the shuffle-pairs CSV from run_r0_p65b_shuffle_control.py):
  Matched (same receiver) real pair (T_j, V_j) vs shuffled pair (T_j, V_k):
    |S_exact|, |S_norm| (and P(|S_norm|>0.1)), |S_second|,
    sign systematics (P(S>0), sign(S2)==sign(S) accuracy, Spearman(S,S2)),
    task effect (best achievable gain Q_best).

Outputs (in --out):
    r0_p65_hybrid_metrics.json / r0_p65_hybrid_summary.md
    r0_p65b_control_metrics.json / r0_p65b_control_summary.md

Usage (any numpy/scipy env):
    python scripts/analyze_r0_p65.py \
        --hybrid-csv experiments/r0/results/Grocery/seed_42/r0_relation_diagnostics.csv \
        --pairs-csv  experiments/r0/results/Grocery/seed_42/r0_p65b_shuffle_pairs.csv \
        --out        experiments/r0/results/Grocery/seed_42
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ACTION_NAMES = ["NULL", "TEXT", "VISUAL", "JOINT"]


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hybrid-csv", type=Path, required=True)
    p.add_argument("--pairs-csv", type=Path, required=False)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


# ------------------------------------------------------------------ helpers
def load_csv(path: Path, int_cols, bool_cols=(), meta_cols=("dataset",)):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows
    raw = {k: [r[k] for r in rows] for k in rows[0].keys()}
    cols = {}
    for k in rows[0].keys():
        if k in bool_cols:
            cols[k] = np.array([v == "True" for v in raw[k]])
        elif k in int_cols:
            cols[k] = np.array(raw[k], dtype=np.float64).astype(np.int64)
        elif k not in meta_cols:
            cols[k] = np.array(raw[k], dtype=np.float64)
    return {"cols": cols, "meta": {k: raw[k][0] for k in meta_cols}, "n": len(rows)}


def _corr(fn, a, b):
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    res = fn(a, b)
    return float(res.statistic if hasattr(res, "statistic") else res[0])


def _sp(a, b): return _corr(stats.spearmanr, a, b)
def _pe(a, b): return _corr(stats.pearsonr, a, b)
def _ke(a, b): return _corr(stats.kendalltau, a, b)


def _wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    if float(np.count_nonzero(d)) < 10:
        return float("nan")
    try:
        res = stats.wilcoxon(x, y)
        return float(res.pvalue if hasattr(res, "pvalue") else res[1])
    except ValueError:
        return float("nan")


def _q(vals):
    return {f"q{q}": float(np.quantile(vals, q / 100)) for q in (10, 25, 50, 75, 90)}


# ------------------------------------------------------------ A: hybrid
def analyze_hybrid(csv_path: Path) -> dict:
    data = load_csv(
        csv_path,
        int_cols=("train_seed", "analysis_seed", "receiver_i", "neighbor_j",
                  "edge_id", "label_i", "label_j", "degree_i", "degree_j",
                  "receiver_degree_bin", "best_exact", "best_first", "best_second"),
        bool_cols=("receiver_correct_full",),
    )
    c, meta, n = data["cols"], data["meta"], data["n"]
    q = c

    Qj_h = q["Q_joint_1"] + q["S_second"]
    dev_lin = np.abs(q["Q_joint_1"] - (q["Q_text_1"] + q["Q_visual_1"])).max()
    assert dev_lin < 1e-4, f"Q_joint_1 != Q_text_1+Q_visual_1 in CSV (max dev {dev_lin})"

    stack_h = np.stack([np.zeros(n), q["Q_text_1"], q["Q_visual_1"], Qj_h], axis=1)
    best_hybrid = stack_h.argmax(axis=1)
    be = q["best_exact"]

    m = {"meta": meta, "n_relations": n,
         "identity_maxdev_Qjoint1_vs_sum": float(dev_lin)}
    m["joint"] = {
        "spearman_exact_vs_hybrid": _sp(q["Q_joint"], Qj_h),
        "kendall_exact_vs_hybrid": _ke(q["Q_joint"], Qj_h),
        "spearman_exact_vs_first": _sp(q["Q_joint"], q["Q_joint_1"]),
        "spearman_exact_vs_second": _sp(q["Q_joint"], q["Q_joint_2"]),
    }
    pooled_exact = np.concatenate([q["Q_text"], q["Q_visual"], q["Q_joint"]])
    pooled_h = np.concatenate([q["Q_text_1"], q["Q_visual_1"], Qj_h])
    pooled_1 = np.concatenate([q["Q_text_1"], q["Q_visual_1"], q["Q_joint_1"]])
    pooled_2 = np.concatenate([q["Q_text_2"], q["Q_visual_2"], q["Q_joint_2"]])
    m["pooled"] = {
        "spearman_exact_vs_hybrid": _sp(pooled_exact, pooled_h),
        "kendall_exact_vs_hybrid": _ke(pooled_exact, pooled_h),
        "spearman_exact_vs_first": _sp(pooled_exact, pooled_1),
        "spearman_exact_vs_second": _sp(pooled_exact, pooled_2),
    }

    acc = {
        "first": float(np.mean(q["best_first"] == be)),
        "second": float(np.mean(q["best_second"] == be)),
        "hybrid": float(np.mean(best_hybrid == be)),
        "most_common_exact_prior": float(np.bincount(be, minlength=4).max() / n),
    }
    m["action_acc"] = acc

    conf = np.zeros((4, 4), dtype=np.int64)
    for i, j in zip(be, best_hybrid):
        conf[int(i), int(j)] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        row_norm = conf / conf.sum(axis=1, keepdims=True)
    m["confusion_hybrid_vs_exact"] = conf.tolist()
    m["confusion_hybrid_vs_exact_row_norm"] = row_norm.tolist()

    sn_abs = np.abs(q["S_norm"])
    subsets = {
        "all": np.ones(n, dtype=bool),
        "absS_norm_gt_0.05": sn_abs > 0.05,
        "absS_norm_gt_0.10": sn_abs > 0.10,
        "S_exact_gt_0": q["S_exact"] > 0,
        "S_exact_lt_0": q["S_exact"] < 0,
        "receiver_correct_full_True": q["receiver_correct_full"],
        "receiver_correct_full_False": ~q["receiver_correct_full"],
    }
    m["subsets"] = {}
    for name, mask in subsets.items():
        nn = int(mask.sum())
        m["subsets"][name] = {
            "n": nn,
            "action_acc_first": float(np.mean(q["best_first"][mask] == be[mask])),
            "action_acc_second": float(np.mean(q["best_second"][mask] == be[mask])),
            "action_acc_hybrid": float(np.mean(best_hybrid[mask] == be[mask])),
        }
    return m


# ------------------------------------------------------------ B: control
def analyze_control(pairs_path: Path) -> dict:
    data = load_csv(
        pairs_path,
        int_cols=("receiver_i", "neighbor_j", "neighbor_k",
                  "edge_id_j", "edge_id_k", "label_i"),
        bool_cols=("receiver_correct_full",),
    )
    c, meta, n = data["cols"], data["meta"], data["n"]
    S_r, S_sh = c["S_exact_real"], c["S_exact_sh"]
    N_r, N_sh = c["S_norm_real"], c["S_norm_sh"]
    S2_r, S2_sh = c["S_second_real"], c["S_second_sh"]
    Qb_r, Qb_sh = c["Q_best_real"], c["Q_best_sh"]

    # pairing quality (log-ratio mismatch)
    with np.errstate(divide="ignore"):
        lr_alpha = np.abs(np.log(c["alpha_j"] / c["alpha_k"]))
        lr_norm = np.abs(np.log(c["norm_deltaV_j"] / c["norm_deltaV_k"]))
    combined = lr_alpha + lr_norm
    eps_s = max(float(np.quantile(np.abs(S_r), 0.10)), 1e-6)

    def _sign_stats(s, s2):
        eff = np.abs(s) > eps_s
        return {
            "P_S_gt_0": float(np.mean(s > 0)),
            "P_S_lt_0": float(np.mean(s < 0)),
            "mean_S": float(np.mean(s)),
            "sign_acc_S2_vs_S_gt_eps": float(np.mean(np.sign(s2[eff]) == np.sign(s[eff])))
            if int(eff.sum()) > 0 else float("nan"),
            "n_gt_eps": int(eff.sum()),
            "spearman_S_S2": _sp(s, s2),
        }

    m = {"meta": meta, "n_pairs": n,
         "eps_s": eps_s,
         "pairing": {
             "median_log_alpha_ratio_abs": float(np.median(lr_alpha)),
             "q90_log_alpha_ratio_abs": float(np.quantile(lr_alpha, 0.9)),
             "median_log_normV_ratio_abs": float(np.median(lr_norm)),
             "q90_log_normV_ratio_abs": float(np.quantile(lr_norm, 0.9)),
         }}
    m["interaction_magnitude"] = {
        "absS_exact_real": {"mean": float(np.abs(S_r).mean()), "median": float(np.median(np.abs(S_r))), **{"_q": _q(np.abs(S_r))}},
        "absS_exact_shuffle": {"mean": float(np.abs(S_sh).mean()), "median": float(np.median(np.abs(S_sh))), **{"_q": _q(np.abs(S_sh))}},
        "wilcoxon_p_absS": _wilcoxon(np.abs(S_r), np.abs(S_sh)),
        "absS_norm_real": {"mean": float(np.abs(N_r).mean()), "median": float(np.median(np.abs(N_r)))},
        "absS_norm_shuffle": {"mean": float(np.abs(N_sh).mean()), "median": float(np.median(np.abs(N_sh)))},
        "P_absS_norm_gt0.1_real": float(np.mean(np.abs(N_r) > 0.1)),
        "P_absS_norm_gt0.1_shuffle": float(np.mean(np.abs(N_sh) > 0.1)),
        "absS_second_real": {"mean": float(np.abs(S2_r).mean()), "median": float(np.median(np.abs(S2_r)))},
        "absS_second_shuffle": {"mean": float(np.abs(S2_sh).mean()), "median": float(np.median(np.abs(S2_sh)))},
        "wilcoxon_p_absS_second": _wilcoxon(np.abs(S2_r), np.abs(S2_sh)),
    }
    # far-tail comparison: real-only vs shuffle-only discordant pairs (McNemar)
    tails = {}
    from math import comb
    for th in (0.01, 0.02, 0.05, 0.10):
        r_only = int(np.sum((np.abs(S_r) > th) & ~(np.abs(S_sh) > th)))
        s_only = int(np.sum(~(np.abs(S_r) > th) & (np.abs(S_sh) > th)))
        dsc = r_only + s_only
        p_mc = float("nan")
        if dsc >= 10:
            k = min(r_only, s_only)
            p_mc = 2.0 * sum(comb(dsc, i) for i in range(0, k + 1)) / (2.0 ** dsc)
            p_mc = min(p_mc, 1.0)
        tails[f"gt_{th:g}"] = {
            "P_real": float(np.mean(np.abs(S_r) > th)),
            "P_shuffle": float(np.mean(np.abs(S_sh) > th)),
            "real_only": r_only, "shuffle_only": s_only, "mcnemar_p": p_mc,
        }
    mx = np.maximum(np.abs(S_r), np.abs(S_sh))
    top = mx >= float(np.quantile(mx, 0.9))
    tails["_top10pct_rows"] = {
        "n": int(top.sum()),
        "absS_exact_real_mean": float(np.abs(S_r[top]).mean()),
        "absS_exact_shuffle_mean": float(np.abs(S_sh[top]).mean()),
    }
    tails["matched_corr_absS_real_shuffle"] = float(np.corrcoef(np.abs(S_r), np.abs(S_sh))[0, 1])
    m["far_tail"] = tails

    m["sign_systematics"] = {
        "real": _sign_stats(S_r, S2_r),
        "shuffle": _sign_stats(S_sh, S2_sh),
    }
    m["task_effect"] = {
        "Q_best_real": {"mean": float(Qb_r.mean()), "median": float(np.median(Qb_r))},
        "Q_best_shuffle": {"mean": float(Qb_sh.mean()), "median": float(np.median(Qb_sh))},
        "wilcoxon_p": _wilcoxon(Qb_r, Qb_sh),
        "P_Qbest_real_gt_shuffle": float(np.mean(Qb_r > Qb_sh)),
        "P_Qbest_real_gt_shuffle_ties": float(np.mean(Qb_r >= Qb_sh)),
    }
    # robustness: better-matched half (combined mismatch <= median)
    keep = combined <= np.median(combined)
    nk = int(keep.sum())
    if nk >= 20:
        m["better_matched_half"] = {
            "n": nk,
            "absS_exact_real_mean": float(np.abs(S_r[keep]).mean()),
            "absS_exact_shuffle_mean": float(np.abs(S_sh[keep]).mean()),
            "wilcoxon_p_absS": _wilcoxon(np.abs(S_r[keep]), np.abs(S_sh[keep])),
            "absS_second_real_mean": float(np.abs(S2_r[keep]).mean()),
            "absS_second_shuffle_mean": float(np.abs(S2_sh[keep]).mean()),
            "wilcoxon_p_absS_second": _wilcoxon(np.abs(S2_r[keep]), np.abs(S2_sh[keep])),
            "Q_best_real_mean": float(Qb_r[keep].mean()),
            "Q_best_shuffle_mean": float(Qb_sh[keep].mean()),
            "wilcoxon_p_Qbest": _wilcoxon(Qb_r[keep], Qb_sh[keep]),
        }
    return m


# ---------------------------------------------------------------- md writers
def _fmt(x: float, w: int = 3) -> str:
    return "nan" if np.isnan(x) else f"{x:+.{w}f}"


def write_hybrid_md(out: Path, m: dict) -> None:
    s = m["subsets"]
    lines = [
        "# R0-P6.5A: Hybrid Functional Teacher (observed data only)",
        "",
        f"- dataset={m['meta']['dataset']} | n_relations={m['n_relations']}",
        f"- hybrid definition: Q_text_h=Q_text_1, Q_visual_h=Q_visual_1, "
        f"Q_joint_h=Q_joint_1+S_second | "
        f"max|Q_joint_1-(Q_text_1+Q_visual_1)|={m['identity_maxdev_Qjoint1_vs_sum']:.2e}",
        "",
        "## Joint utility ranking (exact Q_joint vs estimates)",
        "- " + " | ".join(f"{k}={v:+.3f}" for k, v in m["joint"].items()),
        "",
        "## Pooled ranking (exact vs estimates over text/visual/joint)",
        "- " + " | ".join(f"{k}={v:+.3f}" for k, v in m["pooled"].items()),
        "",
        "## Action accuracy vs best_exact",
        "- first=" + f"{m['action_acc']['first']:.3f} | "
        f"second={m['action_acc']['second']:.3f} | "
        f"hybrid={m['action_acc']['hybrid']:.3f} | "
        f"prior={m['action_acc']['most_common_exact_prior']:.3f}",
        "",
        "## Subset action accuracy",
        "",
        "| subset | n | first | second | hybrid |",
        "|---|---|---|---|---|",
    ]
    for k, v in s.items():
        lines.append(f"| {k} | {v['n']} | {v['action_acc_first']:.3f} | "
                     f"{v['action_acc_second']:.3f} | {v['action_acc_hybrid']:.3f} |")
    lines += [
        "",
        "## Confusion (rows=exact, cols=hybrid)",
        "- " + str(m["confusion_hybrid_vs_exact"]),
        "",
        "_Observed numbers only._",
    ]
    out.write_text("\n".join(lines) + "\n")


def write_control_md(out: Path, m: dict) -> None:
    im, sg, te = m["interaction_magnitude"], m["sign_systematics"], m["task_effect"]
    lines = [
        "# R0-P6.5B: Real Pair vs Shuffled Pair Control (observed data only)",
        "",
        f"- dataset={m['meta']['dataset']} | matched pairs={m['n_pairs']} | "
        f"eps_s={m['eps_s']:.2e}",
        "- control: same receiver i; real (T from j, V from j) vs shuffled "
        "(T from j, V from k, k!=j); k chosen to match message/edge norm "
        f"(median |log alpha_ratio|={m['pairing']['median_log_alpha_ratio_abs']:.3f}, "
        f"median |log normV ratio|={m['pairing']['median_log_normV_ratio_abs']:.3f})",
        "",
        "## Interaction magnitude (real vs shuffled)",
        "- |S_exact| mean: "
        f"real={im['absS_exact_real']['mean']:.5f} vs shuffle={im['absS_exact_shuffle']['mean']:.5f} "
        f"(median {im['absS_exact_real']['median']:.5f} vs "
        f"{im['absS_exact_shuffle']['median']:.5f}) | wilcoxon p={im['wilcoxon_p_absS']:.2e}",
        "- |S_norm| mean: "
        f"real={im['absS_norm_real']['mean']:.5f} vs shuffle={im['absS_norm_shuffle']['mean']:.5f} | "
        f"P(|S_norm|>0.1): real={im['P_absS_norm_gt0.1_real']:.3f} vs "
        f"shuffle={im['P_absS_norm_gt0.1_shuffle']:.3f}",
        "- |S_second| mean: "
        f"real={im['absS_second_real']['mean']:.5f} vs shuffle={im['absS_second_shuffle']['mean']:.5f} "
        f"(median {im['absS_second_real']['median']:.5f} vs "
        f"{im['absS_second_shuffle']['median']:.5f}) | wilcoxon p={im['wilcoxon_p_absS_second']:.2e}",
        "",
        "## Sign systematics (S2 vs S_exact)",
        "- real:    " + f"P(S>0)={sg['real']['P_S_gt_0']:.3f} | mean(S)={sg['real']['mean_S']:+.5f} | "
        f"sign-acc(>eps, n={sg['real']['n_gt_eps']})={sg['real']['sign_acc_S2_vs_S_gt_eps']:.3f} | "
        f"Spearman(S,S2)={sg['real']['spearman_S_S2']:+.3f}",
        "- shuffle: " + f"P(S>0)={sg['shuffle']['P_S_gt_0']:.3f} | mean(S)={sg['shuffle']['mean_S']:+.5f} | "
        f"sign-acc(>eps, n={sg['shuffle']['n_gt_eps']})={sg['shuffle']['sign_acc_S2_vs_S_gt_eps']:.3f} | "
        f"Spearman(S,S2)={sg['shuffle']['spearman_S_S2']:+.3f}",
        "",
        "## Task effect (best achievable gain on the receiver)",
        "- Q_best mean: "
        f"real={te['Q_best_real']['mean']:+.5f} vs shuffle={te['Q_best_shuffle']['mean']:+.5f} "
        f"(median {te['Q_best_real']['median']:+.5f} vs "
        f"{te['Q_best_shuffle']['median']:+.5f}) | wilcoxon p={te['wilcoxon_p']:.2e}",
        f"- P(Q_best_real > Q_best_shuffle)={te['P_Qbest_real_gt_shuffle']:.3f} "
        f"(incl. ties={te['P_Qbest_real_gt_shuffle_ties']:.3f})",
    ]
    ft = m["far_tail"]
    lines += [
        "",
        "## Far-tail comparison (|S| thresholds, McNemar on discordant pairs)",
        "- " + " | ".join(
            f"|S|>{th}: P_real={ft[k]['P_real']:.3f} vs P_shuf={ft[k]['P_shuffle']:.3f} "
            f"(real-only={ft[k]['real_only']}, shuf-only={ft[k]['shuffle_only']}, "
            f"p={ft[k]['mcnemar_p']:.2e})"
            for k, th in (("gt_0.01", 0.01), ("gt_0.02", 0.02),
                          ("gt_0.05", 0.05), ("gt_0.1", 0.10))),
        f"- top-10% |S| rows (n={ft['_top10pct_rows']['n']}): "
        f"real mean={ft['_top10pct_rows']['absS_exact_real_mean']:.4f} vs "
        f"shuffle mean={ft['_top10pct_rows']['absS_exact_shuffle_mean']:.4f}",
        f"- matched corr(|S_real|, |S_shuffle|)={ft['matched_corr_absS_real_shuffle']:.3f}",
    ]
    if "better_matched_half" in m:
        b = m["better_matched_half"]
        lines += [
            "",
            "## Robustness: better-matched half (combined mismatch <= median)",
            f"- n={b['n']} | |S_exact| mean real={b['absS_exact_real_mean']:.5f} vs "
            f"shuffle={b['absS_exact_shuffle_mean']:.5f} (p={b['wilcoxon_p_absS']:.2e}) | "
            f"|S_second| mean real={b['absS_second_real_mean']:.5f} vs "
            f"shuffle={b['absS_second_shuffle_mean']:.5f} (p={b['wilcoxon_p_absS_second']:.2e}) | "
            f"Q_best mean real={b['Q_best_real_mean']:+.5f} vs "
            f"shuffle={b['Q_best_shuffle_mean']:+.5f} (p={b['wilcoxon_p_Qbest']:.2e})",
        ]
    lines += ["", "_Observed numbers only._"]
    out.write_text("\n".join(lines) + "\n")


def main(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)

    mh = analyze_hybrid(args.hybrid_csv)
    (args.out / "r0_p65_hybrid_metrics.json").write_text(
        json.dumps(mh, indent=2, allow_nan=True))
    write_hybrid_md(args.out / "r0_p65_hybrid_summary.md", mh)
    print("[A] r0_p65_hybrid_metrics.json + r0_p65_hybrid_summary.md")
    print(f"    action acc: first={mh['action_acc']['first']:.3f} "
          f"second={mh['action_acc']['second']:.3f} "
          f"hybrid={mh['action_acc']['hybrid']:.3f}")

    if args.pairs_csv is not None and args.pairs_csv.exists():
        mc = analyze_control(args.pairs_csv)
        (args.out / "r0_p65b_control_metrics.json").write_text(
            json.dumps(mc, indent=2, allow_nan=True))
        write_control_md(args.out / "r0_p65b_control_summary.md", mc)
        im = mc["interaction_magnitude"]
        print("[B] r0_p65b_control_metrics.json + r0_p65b_control_summary.md")
        print(f"    mean|S_exact| real={im['absS_exact_real']['mean']:.5f} "
              f"shuffle={im['absS_exact_shuffle']['mean']:.5f}")
        print(f"    mean|S_second| real={im['absS_second_real']['mean']:.5f} "
              f"shuffle={im['absS_second_shuffle']['mean']:.5f}")
    else:
        print("[B] --pairs-csv missing or nonexistent; skipping control analysis")


if __name__ == "__main__":
    main(_parse_args())
