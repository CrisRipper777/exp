"""R0 analysis + figures (FIRM-MAG plan §11-§14).

Reads one r0_relation_diagnostics.csv (produced by run_r0_counterfactual.py),
never re-runs a model. Emits:

  r0_metrics.json   - A..F statistics (plan §11)
  r0_summary.md     - observed-facts-only summary (plan §14)
  figures/fig1..fig6.png (plan §12/§13)

Usage:
    python scripts/analyze_r0.py \
        --csv  experiments/r0/results/Grocery/seed_42/r0_relation_diagnostics.csv \
        --out  experiments/r0/results/Grocery/seed_42
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ACTION_NAMES = ["NULL", "TEXT", "VISUAL", "JOINT"]

_INT_COLS = [
    "receiver_i", "neighbor_j", "edge_id", "label_i", "label_j",
    "degree_i", "degree_j", "receiver_degree_bin",
    "best_exact", "best_first", "best_second",
]


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_csv(path: Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "empty CSV"
    fieldnames = list(rows[0].keys())
    raw = {k: [r[k] for r in rows] for k in fieldnames}

    cols = {}
    for k in fieldnames:
        if k == "receiver_correct_full":
            cols[k] = np.array([v == "True" for v in raw[k]])
        elif k in _INT_COLS:
            cols[k] = np.array(raw[k], dtype=np.float64).astype(np.int64)
        elif k != "dataset":  # all remaining columns are float64 measurements
            cols[k] = np.array(raw[k], dtype=np.float64)
    meta = {
        "dataset": raw["dataset"][0],
        "train_seed": str(int(float(raw["train_seed"][0]))),
        "analysis_seed": str(int(float(raw["analysis_seed"][0]))),
    }
    return {"cols": cols, "meta": meta, "n": len(rows)}


def _corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    """scipy>=1.9 returns SignificanceResult (.statistic); older returns tuple."""
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return float("nan")
    res = fn(a, b)
    v = res.statistic if hasattr(res, "statistic") else res[0]
    return float(v)


def _sp(a, b): return _corr(stats.spearmanr, a, b)
def _pe(a, b): return _corr(stats.pearsonr, a, b)
def _ke(a, b): return _corr(stats.kendalltau, a, b)


def analyze(args) -> None:
    data = load_csv(args.csv)
    c = data["cols"]
    n = data["n"]
    meta = data["meta"]
    m: dict = {"meta": meta, "n_relations": n}

    # ---------------- A. interaction statistics (plan §11.1) ----------------
    S, Sn = c["S_exact"], c["S_norm"]
    absS = np.abs(S)
    m["A_interaction"] = {
        "mean_S": float(S.mean()),
        "median_S": float(np.median(S)),
        "mean_abs_S": float(absS.mean()),
        "median_abs_S": float(np.median(absS)),
        "mean_S_norm": float(Sn.mean()),
        "median_S_norm": float(np.median(Sn)),
        "mean_abs_S_norm": float(np.abs(Sn).mean()),
        "median_abs_S_norm": float(np.median(np.abs(Sn))),
        "P_absS_norm_gt_0.05": float(np.mean(np.abs(Sn) > 0.05)),
        "P_absS_norm_gt_0.10": float(np.mean(np.abs(Sn) > 0.10)),
        "P_absS_norm_gt_0.20": float(np.mean(np.abs(Sn) > 0.20)),
        "P_S_gt_0": float(np.mean(S > 0)),
        "P_S_lt_0": float(np.mean(S < 0)),
        "quantiles_absS": {
            "q05": float(np.quantile(absS, 0.05)),
            "q10": float(np.quantile(absS, 0.10)),
            "q25": float(np.quantile(absS, 0.25)),
            "q50": float(np.quantile(absS, 0.50)),
        },
    }

    # ---------------- B. optimal action distribution (plan §11.2) ----------
    be = c["best_exact"]
    counts = [int((be == k).sum()) for k in range(4)]
    dist = {ACTION_NAMES[k]: {"count": counts[k], "pct": counts[k] / n} for k in range(4)}

    def _share(mask: np.ndarray) -> dict:
        cnt = [int((be[mask] == k).sum()) for k in range(4)]
        tot = max(int(mask.sum()), 1)
        return {ACTION_NAMES[k]: cnt[k] / tot for k in range(4)}

    grouped_deg = {
        int(b): _share(c["receiver_degree_bin"] == b) for b in range(4)
        if int((c["receiver_degree_bin"] == b).sum()) > 0
    }
    grouped_corr = {
        str(corr): _share(c["receiver_correct_full"] == corr) for corr in (True, False)
    }
    class_action = {
        int(cl): _share(c["label_i"] == cl) for cl in np.unique(c["label_i"])
    }
    m["B_actions"] = {
        "distribution": dist,
        "by_receiver_degree_bin": grouped_deg,
        "by_receiver_correct_full": grouped_corr,
        "classes_present": len(class_action),
        "per_class_action_share": class_action,
    }

    # ---------------- C. semantic proxy (plan §11.3) -----------------------
    st, sv, qt, qv = c["sim_text"], c["sim_visual"], c["Q_text"], c["Q_visual"]
    m["C_semantic_proxy"] = {
        "spearman_simT_Qtext": _sp(st, qt),
        "spearman_simV_Qvisual": _sp(sv, qv),
        "pearson_simT_Qtext": _pe(st, qt),
        "pearson_simV_Qvisual": _pe(sv, qv),
        "spearman_simT_simV_prod_S": _sp(st * sv, S),
        "spearman_simT_Qjoint": _sp(st, c["Q_joint"]),
        "spearman_simV_Qjoint": _sp(sv, c["Q_joint"]),
    }

    # ---------------- D. approximation quality (plan §11.4) ----------------
    approx = {}
    for name, exact, e1, e2 in (
        ("text", qt, c["Q_text_1"], c["Q_text_2"]),
        ("visual", qv, c["Q_visual_1"], c["Q_visual_2"]),
        ("joint", c["Q_joint"], c["Q_joint_1"], c["Q_joint_2"]),
    ):
        approx[name] = {
            "spearman_exact_vs_first": _sp(exact, e1),
            "spearman_exact_vs_second": _sp(exact, e2),
            "kendall_exact_vs_first": _ke(exact, e1),
            "kendall_exact_vs_second": _ke(exact, e2),
        }
    pooled_exact = np.concatenate([qt, qv, c["Q_joint"]])
    pooled_1 = np.concatenate([c["Q_text_1"], c["Q_visual_1"], c["Q_joint_1"]])
    pooled_2 = np.concatenate([c["Q_text_2"], c["Q_visual_2"], c["Q_joint_2"]])
    approx["pooled"] = {
        "spearman_exact_vs_first": _sp(pooled_exact, pooled_1),
        "spearman_exact_vs_second": _sp(pooled_exact, pooled_2),
        "kendall_exact_vs_first": _ke(pooled_exact, pooled_1),
        "kendall_exact_vs_second": _ke(pooled_exact, pooled_2),
    }
    m["D_approximation"] = approx

    # ---------------- E. action prediction (plan §11.5) --------------------
    b1, b2 = c["best_first"], c["best_second"]
    conf = np.zeros((4, 4), dtype=np.int64)
    for i, j in zip(be, b2):
        conf[int(i), int(j)] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        conf_row_norm = conf / conf.sum(axis=1, keepdims=True)
    m["E_action"] = {
        "action_acc_first_vs_exact": float(np.mean(b1 == be)),
        "action_acc_second_vs_exact": float(np.mean(b2 == be)),
        "most_common_exact_prior": max(counts) / n,
        "confusion_second_vs_exact": conf.tolist(),
        "confusion_second_vs_exact_row_norm": conf_row_norm.tolist(),
    }

    # ---------------- F. interaction sign (plan §11.6) ---------------------
    S2 = c["S_second"]
    eps_s = float(np.quantile(absS, 0.10))
    eps_s = max(eps_s, 1e-6)  # guard against an all-zero |S| column
    eff = absS > eps_s
    m["F_interaction"] = {
        "spearman_S_S2": _sp(S, S2),
        "pearson_S_S2": _pe(S, S2),
        "eps_s": eps_s,
        "n_effective_absS_gt_eps": int(eff.sum()),
        "sign_acc_effective": float(np.mean(np.sign(S[eff]) == np.sign(S2[eff])))
        if int(eff.sum()) > 0 else float("nan"),
        "sign_acc_all": float(np.mean(np.sign(S) == np.sign(S2)))
        if bool(np.any(S != 0)) else float("nan"),
    }

    # -------- figures (plan §12/§13) ---------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[warn] matplotlib unavailable: {exc}; skipping figures")
        write_outputs(args.out, m)
        return

    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # fig1: additivity test Q_T+Q_V vs Q_TV
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    hb = ax.hexbin(qt + qv, c["Q_joint"], gridsize=60, cmap="viridis", bins="log")
    lim = [float(min(qt.min() + qv.min(), c["Q_joint"].min())),
           float(max(qt.max() + qv.max(), c["Q_joint"].max()))]
    ax.plot(lim, lim, "r--", lw=1.2, label="y = x (additive)")
    ax.set_xlabel("$Q_T + Q_V$ (exact)")
    ax.set_ylabel("$Q_{TV}$ (exact)")
    ax.set_title(f"{meta['dataset']}: additivity test (n={n})")
    fig.colorbar(hb, ax=ax, label="count")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_additive_vs_joint.png", dpi=150)
    plt.close(fig)

    # fig2: S_norm distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(Sn, bins=100, range=(-1.0, 1.0), color="#4c72b0", alpha=0.9)
    for th in (0.05, 0.10, 0.20):
        ax.axvline(th, color="gray", ls=":", lw=1)
        ax.axvline(-th, color="gray", ls=":", lw=1)
    ax.set_xlabel("$S_{norm}$")
    ax.set_ylabel("count")
    p10 = m["A_interaction"]["P_absS_norm_gt_0.10"]
    ax.set_title(f"{meta['dataset']}: normalized interaction (P(|S_norm|>0.1)={p10:.3f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_snorm_distribution.png", dpi=150)
    plt.close(fig)

    # fig3: action distribution
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    fracs = [dist[k]["pct"] for k in ACTION_NAMES]
    bars = ax.bar(ACTION_NAMES, fracs, color=["#8c8c8c", "#4c72b0", "#dd8452", "#55a868"])
    ax.bar_label(bars, fmt="%.1f%%")
    ax.set_ylabel("fraction of sampled relations")
    ax.set_ylim(0, max(fracs) * 1.25)
    ax.set_title(f"{meta['dataset']}: optimal propagation action (exact)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_action_distribution.png", dpi=150)
    plt.close(fig)

    # fig4: exact vs second-order interaction
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.hexbin(S, S2, gridsize=60, cmap="viridis", bins="log")
    lim = [float(min(S.min(), S2.min())), float(max(S.max(), S2.max()))]
    ax.plot(lim, lim, "r--", lw=1.2, label="y = x")
    ax.set_xlabel("$S_{exact}$")
    ax.set_ylabel("$S_{second}$")
    fF = m["F_interaction"]
    ax.set_title(f"exact vs 2nd-order interaction\n"
                 f"Spearman={fF['spearman_S_S2']:.3f} | "
                 f"SignAcc(|S|>eps)={fF['sign_acc_effective']:.3f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_exact_vs_second_interaction.png", dpi=150)
    plt.close(fig)

    # fig5: action confusion (second vs exact)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(np.nan_to_num(conf_row_norm, nan=0.0), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), ACTION_NAMES)
    ax.set_yticks(range(4), ACTION_NAMES)
    ax.set_xlabel("best_second")
    ax.set_ylabel("best_exact")
    for i in range(4):
        for j in range(4):
            val = conf[i, j]
            share = conf_row_norm[i, j]
            color = "white" if (np.isfinite(share) and share > 0.5) else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color)
    ax.set_title(f"action confusion (2nd vs exact)\n"
                 f"ActionAcc_2={m['E_action']['action_acc_second_vs_exact']:.3f}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_action_confusion.png", dpi=150)
    plt.close(fig)

    # fig6: semantic proxy vs functional effect
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hexbin(st, qt, gridsize=50, cmap="viridis", bins="log")
    axes[0].set_xlabel("sim_text")
    axes[0].set_ylabel("$Q_T$")
    fC = m["C_semantic_proxy"]
    axes[0].set_title(f"Spearman={fC['spearman_simT_Qtext']:.3f}")
    axes[1].hexbin(sv, qv, gridsize=50, cmap="viridis", bins="log")
    axes[1].set_xlabel("sim_visual")
    axes[1].set_ylabel("$Q_V$")
    axes[1].set_title(f"Spearman={fC['spearman_simV_Qvisual']:.3f}")
    fig.suptitle(f"{meta['dataset']}: semantic proxy vs functional effect")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_proxy_correlations.png", dpi=150)
    plt.close(fig)

    print("[figures] saved:", ", ".join(p.name for p in sorted(fig_dir.glob("fig*.png"))))
    write_outputs(args.out, m)


def write_outputs(out_dir: Path, m: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "r0_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, allow_nan=True)
    _write_summary_md(out_dir, m)
    print("[out] r0_metrics.json + r0_summary.md written to", out_dir)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _write_summary_md(out_dir: Path, m: dict) -> None:
    A, B, C = m["A_interaction"], m["B_actions"], m["C_semantic_proxy"]
    D, E, F = m["D_approximation"], m["E_action"], m["F_interaction"]
    meta = m["meta"]
    md = [
        "# R0 Interaction Diagnostics Summary (observed data only)",
        "",
        f"- dataset={meta['dataset']} | train_seed={meta['train_seed']} | "
        f"analysis_seed={meta['analysis_seed']} | sampled relations={m['n_relations']}",
        "",
        "## A. Interaction statistics",
        f"- mean(S_exact)={A['mean_S']:+.5f} | median(S_exact)={A['median_S']:+.5f} | "
        f"mean(|S_exact|)={A['mean_abs_S']:.5f} | median(|S_exact|)={A['median_abs_S']:.5f}",
        f"- mean(S_norm)={A['mean_S_norm']:+.5f} | median(S_norm)={A['median_S_norm']:+.5f} | "
        f"mean(|S_norm|)={A['mean_abs_S_norm']:.5f} | median(|S_norm|)={A['median_abs_S_norm']:.5f}",
        f"- P(|S_norm|>0.05)={A['P_absS_norm_gt_0.05']:.3f} | "
        f"P(|S_norm|>0.10)={A['P_absS_norm_gt_0.10']:.3f} | "
        f"P(|S_norm|>0.20)={A['P_absS_norm_gt_0.20']:.3f}",
        f"- P(S>0)={A['P_S_gt_0']:.3f} | P(S<0)={A['P_S_lt_0']:.3f} | "
        f"|S| q10={A['quantiles_absS']['q10']:.2e} q50={A['quantiles_absS']['q50']:.2e}",
        "",
        "## B. Optimal action distribution (exact)",
        "- " + " | ".join(f"{k}: {v['count']} ({_pct(v['pct'])})"
                          for k, v in B["distribution"].items()),
        "- per-degree-bin shares: " + "; ".join(
            f"bin{k}: " + ", ".join(f"{a}={v[a]:.2f}" for a in ACTION_NAMES)
            for k, v in sorted(B["by_receiver_degree_bin"].items())),
        "- by receiver-correct-full: "
        + "; ".join(f"{k}: " + ", ".join(f"{a}={v[a]:.2f}" for a in ACTION_NAMES)
                    for k, v in sorted(B["by_receiver_correct_full"].items())),
        "",
        "## C. Semantic proxy",
        f"- Spearman(sim_text, Q_text)={C['spearman_simT_Qtext']:+.3f} | "
        f"Pearson={C['pearson_simT_Qtext']:+.3f}",
        f"- Spearman(sim_visual, Q_visual)={C['spearman_simV_Qvisual']:+.3f} | "
        f"Pearson={C['pearson_simV_Qvisual']:+.3f}",
        f"- Spearman(sim_text * sim_visual, S_exact)={C['spearman_simT_simV_prod_S']:+.3f} "
        f"(exploratory only)",
        "",
        "## D. Approximation quality (exact vs first/second order)",
        "- text:   " + f"Spearman 1st={D['text']['spearman_exact_vs_first']:+.3f} "
        f"2nd={D['text']['spearman_exact_vs_second']:+.3f} | "
        f"Kendall 1st={D['text']['kendall_exact_vs_first']:+.3f} "
        f"2nd={D['text']['kendall_exact_vs_second']:+.3f}",
        "- visual: " + f"Spearman 1st={D['visual']['spearman_exact_vs_first']:+.3f} "
        f"2nd={D['visual']['spearman_exact_vs_second']:+.3f} | "
        f"Kendall 1st={D['visual']['kendall_exact_vs_first']:+.3f} "
        f"2nd={D['visual']['kendall_exact_vs_second']:+.3f}",
        "- joint:  " + f"Spearman 1st={D['joint']['spearman_exact_vs_first']:+.3f} "
        f"2nd={D['joint']['spearman_exact_vs_second']:+.3f} | "
        f"Kendall 1st={D['joint']['kendall_exact_vs_first']:+.3f} "
        f"2nd={D['joint']['kendall_exact_vs_second']:+.3f}",
        "- pooled: " + f"Spearman 1st={D['pooled']['spearman_exact_vs_first']:+.3f} "
        f"2nd={D['pooled']['spearman_exact_vs_second']:+.3f} | "
        f"Kendall 1st={D['pooled']['kendall_exact_vs_first']:+.3f} "
        f"2nd={D['pooled']['kendall_exact_vs_second']:+.3f}",
        "",
        "## E. Action prediction",
        f"- ActionAcc(first vs exact)={E['action_acc_first_vs_exact']:.3f} | "
        f"ActionAcc(second vs exact)={E['action_acc_second_vs_exact']:.3f} | "
        f"most-common-exact prior={E['most_common_exact_prior']:.3f}",
        f"- confusion (rows=exact, cols=second): {E['confusion_second_vs_exact']}",
        "",
        "## F. Interaction sign (second vs exact)",
        f"- Spearman(S_exact, S_second)={F['spearman_S_S2']:+.3f} | "
        f"Pearson={F['pearson_S_S2']:+.3f}",
        f"- SignAcc over |S_exact|>eps_s samples "
        f"(eps_s={F['eps_s']:.2e}, n={F['n_effective_absS_gt_eps']})"
        f"={F['sign_acc_effective']:.3f}",
        "",
        "_This summary states observed numbers only; no hypothesis conclusion "
        "is drawn here._",
    ]
    with (out_dir / "r0_summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    analyze(_parse_args())
