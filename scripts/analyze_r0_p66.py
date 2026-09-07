"""R0-P6.6: Functional Teacher Necessity & Receiver-State Dependence Audit.

Pure post-processing of:
  - r0_relation_diagnostics.csv            (relation-level, read only)
  - r0_p66_receiver_uncertainty.csv        (from compute_r0_p66_uncertainty.py)

A. Exact utility regret.  For each row, Q_exact = [0, Q_text, Q_visual,
   Q_joint] and Q_star = max(Q_exact).  Estimator e's chosen action is
   evaluated by the EXACT utility of that action:
       Q_selected_e = Q_exact[pred_e]
       Regret_e     = Q_star - Q_selected_e
   e in {first (best_first), second (best_second),
         hybrid (argmax[0, Q_text_1, Q_visual_1, Q_joint_1 + S_second])}.
   Metrics per estimator per subset: ActionAcc, mean/median/P90/P95 regret,
   harmful rate P(Q_selected < 0), utility retention
   sum(Q_selected)/(sum(Q_star)+eps).

B. Receiver uncertainty.  Receivers are grouped into rank quartiles of
   confidence / normalized entropy / top1-top2 margin (Q1..Q4).  Per
   quartile: n_receivers, n_relations, receiver correctness rate,
   ActionAcc(first/second/hybrid), mean regret(first/second/hybrid),
   harmful rate, mean |S_norm|, exact action distribution.
   DeltaRegret = mean(Regret_first) - mean(Regret_second); positive means
   the second-order teacher is better.

C. Receiver-cluster bootstrap (cluster = receiver_i; relations are NOT
   independent).  10,000 resamples of receivers (with replacement, all rows
   of a drawn receiver included) give 95% CIs for
   mean(Regret_first - Regret_second) on: overall, receiver_correct=False,
   lowest-confidence quartile, highest-entropy quartile, smallest-margin
   quartile; plus ActionAcc_second - ActionAcc_first on the same states.

D. Outputs: r0_p66_state_metrics.json / r0_p66_state_summary.md and
   fig_p66_1..5 png.  Facts and exploratory interpretation are kept apart.

Usage (any numpy/scipy/matplotlib env, e.g. AL):
    python scripts/analyze_r0_p66.py \
        --csv experiments/r0/results/Grocery/seed_42/r0_relation_diagnostics.csv \
        --uncertainty experiments/r0/results/Grocery/seed_42/r0_p66_receiver_uncertainty.csv \
        --out experiments/r0/results/Grocery/seed_42
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ACTION_NAMES = ["NULL", "TEXT", "VISUAL", "JOINT"]
B_STATES = [
    ("overall", "Overall"),
    ("correct_False", "Receiver incorrect (correct=False)"),
    ("conf_Q1", "Lowest-confidence quartile"),
    ("entropy_Q4", "Highest-entropy quartile"),
    ("margin_Q1", "Smallest-margin quartile"),
]


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--uncertainty", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=20260907)
    return p.parse_args()


# ------------------------------------------------------------------ loading
def _load_rel(path: Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    get = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    c = {
        "receiver_i": np.array([int(r["receiver_i"]) for r in rows], dtype=np.int64),
        "correct": np.array([r["receiver_correct_full"] == "True" for r in rows]),
        "best_exact": get("best_exact").astype(np.int64),
        "best_first": get("best_first").astype(np.int64),
        "best_second": get("best_second").astype(np.int64),
        "S_exact": get("S_exact"), "S_norm": get("S_norm"),
        "full_confidence": get("full_confidence"),
    }
    for k in ("Q_text", "Q_visual", "Q_joint",
              "Q_text_1", "Q_visual_1", "Q_joint_1", "S_second"):
        c[k] = get(k)
    return c, n


def _load_unc(path: Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {
        "receiver_i": np.array([int(r["receiver_i"]) for r in rows], dtype=np.int64),
        "confidence": np.array([float(r["confidence"]) for r in rows]),
        "entropy_norm": np.array([float(r["entropy_norm"]) for r in rows]),
        "margin": np.array([float(r["margin"]) for r in rows]),
        "correct": np.array([r["correct"] == "True" for r in rows]),
    }


# ------------------------------------------------------------- estimators
def _est_stats(c, n):
    """Row-level exact utilities, regrets, predictions for the 3 estimators."""
    Q_exact = np.stack([np.zeros(n), c["Q_text"], c["Q_visual"], c["Q_joint"]], axis=1)
    Q_star = Q_exact.max(axis=1)
    Qj_h = c["Q_joint_1"] + c["S_second"]
    dev_lin = np.abs(c["Q_joint_1"] - (c["Q_text_1"] + c["Q_visual_1"])).max()
    assert dev_lin < 1e-4
    pred_h = np.stack([np.zeros(n), c["Q_text_1"], c["Q_visual_1"], Qj_h],
                      axis=1).argmax(axis=1)
    preds = {"first": c["best_first"], "second": c["best_second"],
             "hybrid": pred_h}
    out = {"Q_exact": Q_exact, "Q_star": Q_star, "preds": preds}
    for e, p in preds.items():
        sel = Q_exact[np.arange(n), p]
        out[f"regret_{e}"] = Q_star - sel
        out[f"sel_{e}"] = sel
    out["acc"] = {e: (p == c["best_exact"]) for e, p in preds.items()}
    return out


def _subset_rows(c, n):
    sn = np.abs(c["S_norm"])
    return {
        "all": np.ones(n, dtype=bool),
        "receiver_correct_full_True": c["correct"],
        "receiver_correct_full_False": ~c["correct"],
        "absS_norm_gt_0.05": sn > 0.05,
        "absS_norm_gt_0.10": sn > 0.10,
        "S_exact_gt_0": c["S_exact"] > 0,
        "S_exact_lt_0": c["S_exact"] < 0,
    }


def _one_subset_stats(mask, st, e_names):
    n = int(mask.sum())
    out = {"n": n}
    for e in e_names:
        r = st[f"regret_{e}"][mask]
        acc = float(np.mean(st["acc"][e][mask]))
        out[e] = {
            "action_acc": acc,
            "mean_regret": float(r.mean()),
            "median_regret": float(np.median(r)),
            "p90_regret": float(np.quantile(r, 0.9)),
            "p95_regret": float(np.quantile(r, 0.95)),
            "harmful_rate_P_Qsel_lt0": float(np.mean(st[f"sel_{e}"][mask] < 0)),
            "utility_retention": float(
                st[f"sel_{e}"][mask].sum() / (st["Q_star"][mask].sum() + 1e-12)),
        }
    return out


# ------------------------------------------------------------------ A
def analyze_A(c, n, st):
    e_names = ["first", "second", "hybrid"]
    subsets = _subset_rows(c, n)
    A = {"evaluated_via_exact_Q": True}
    for name, mask in subsets.items():
        A[name] = _one_subset_stats(mask, st, e_names)
    # extra diagnostic facts
    Q = st["Q_star"]
    A["_diagnostics"] = {
        "mean_Q_star": float(Q.mean()),
        "frac_rows_Q_star_0": float(np.mean(Q <= 1e-12)),
        "frac_rows_Q_star_null_only": float(
            np.mean((Q <= 1e-12) & (c["best_exact"] == 0))),
    }
    return A


# ------------------------------------------------------------------ B
def _rank_quartiles(score: np.ndarray) -> np.ndarray:
    """Balanced quartile labels 1..4 by rank (ties broken by stable order)."""
    n = len(score)
    order = np.argsort(score, kind="stable")
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    return np.minimum(pos * 4 // n, 3) + 1


def analyze_B(c, n, st, unc):
    # join uncertainty onto rows
    recv_unc = unc["receiver_i"]
    idx = np.searchsorted(recv_unc, c["receiver_i"])
    assert np.array_equal(recv_unc[idx], c["receiver_i"]), "receiver join failed"
    r_conf = unc["confidence"][idx]
    r_ent = unc["entropy_norm"][idx]
    r_marg = unc["margin"][idx]

    receiver_level = {}
    for name, score, qdesc in (("confidence", r_conf, "Q1 lowest -> Q4 highest"),
                               ("entropy_norm", r_ent, "Q1 lowest -> Q4 highest"),
                               ("margin", r_marg, "Q1 smallest -> Q4 largest")):
        # quartile assignment computed on UNIQUE receivers
        u_recv = np.unique(c["receiver_i"])
        u_pos = np.searchsorted(recv_unc, u_recv)
        u_score = unc[name][u_pos]
        u_q = _rank_quartiles(u_score)
        q_of_receiver = dict(zip(u_recv.tolist(), u_q.tolist()))
        row_q = np.array([q_of_receiver[r] for r in c["receiver_i"].tolist()],
                         dtype=np.int64)
        receiver_level[name] = row_q

    unc_out = {}
    for name in ("confidence", "entropy_norm", "margin"):
        row_q = receiver_level[name]
        qtab = {}
        for q in (1, 2, 3, 4):
            mask = row_q == q
            recv_mask = np.isin(np.unique(c["receiver_i"]), c["receiver_i"][mask])
            n_recv = int(np.sum(recv_mask))
            st_row = _one_subset_stats(mask, st, ["first", "second", "hybrid"])
            u_recv = np.unique(c["receiver_i"])
            u_pos_q = np.searchsorted(recv_unc, u_recv[recv_mask])
            qtab[f"Q{q}"] = {
                "n_receivers": n_recv,
                "n_relations": st_row.pop("n"),
                "receiver_correct_rate": float(np.mean(unc["correct"][u_pos_q]))
                if n_recv else float("nan"),
                **st_row,
                "mean_absS_norm": float(np.abs(c["S_norm"][mask]).mean()),
                "exact_action_distribution": {
                    ACTION_NAMES[k]: int(np.sum(c["best_exact"][mask] == k))
                    for k in range(4)},
            }
        for q in range(1, 5):
            d = qtab[f"Q{q}"]
            d["delta_regret_first_minus_second"] = (
                d["first"]["mean_regret"] - d["second"]["mean_regret"])
        unc_out[name] = {"quartile_definition": qdesc, "quartiles": qtab,
                         "relation": "rows of receivers in that quartile"}
    return unc_out


# ------------------------------------------------------------------ C
def _state_membership(state, c, receivers, unc):
    """Receiver list + row mask for one bootstrap state."""
    recv_row = c["receiver_i"]
    if state == "overall":
        return receivers, np.ones(len(recv_row), dtype=bool)
    # receiver-level "correct" (identical across a receiver's rows)
    corr_of_recv = np.array([bool(c["correct"][recv_row == r][0])
                             for r in receivers])
    if state == "correct_False":
        m = receivers[~corr_of_recv]
        return m, np.isin(recv_row, m)
    if state == "conf_Q1":
        name, want_q = "confidence", 1
    elif state == "entropy_Q4":
        name, want_q = "entropy_norm", 4
    else:  # margin_Q1
        name, want_q = "margin", 1
    unc_idx = np.searchsorted(unc["receiver_i"], receivers)
    q_of_recv = _rank_quartiles(unc[name][unc_idx])
    m = receivers[q_of_recv == want_q]
    return m, np.isin(recv_row, m)


def analyze_C(c, n, st, unc, n_boot, seed):
    """Cluster bootstrap (cluster = receiver_i), 10,000 resamples."""
    rng = np.random.default_rng(seed)
    d_regret = st["regret_first"] - st["regret_second"]   # >0 -> 2nd better
    d_acc = st["acc"]["second"].astype(float) - st["acc"]["first"].astype(float)
    receivers = np.unique(c["receiver_i"])
    recv_row = c["receiver_i"]

    boot = {}
    for state, _ in B_STATES:
        recv_m, mask = _state_membership(state, c, receivers, unc)
        m = int(recv_m.size)
        # position (within recv_m) of each in-state row's receiver
        rmap = np.searchsorted(recv_m, recv_row[mask])
        d_r, d_a = d_regret[mask], d_acc[mask]
        draws_reg = np.empty(n_boot)
        draws_acc = np.empty(n_boot)
        for t in range(n_boot):
            counts = rng.multinomial(m, np.ones(m) / m)
            w = counts[rmap]                 # receiver multiplicity -> row weight
            wsum = w.sum()
            draws_reg[t] = float((d_r * w).sum() / wsum)
            draws_acc[t] = float((d_a * w).sum() / wsum)
        boot[state] = {
            "n_receivers": m,
            "n_relations": int(mask.sum()),
            "delta_regret_first_minus_second_obs": float(np.mean(d_r)),
            "delta_regret_ci95": [float(np.quantile(draws_reg, 0.025)),
                                  float(np.quantile(draws_reg, 0.975))],
            "delta_actionacc_second_minus_first_obs": float(np.mean(d_a)),
            "delta_actionacc_ci95": [float(np.quantile(draws_acc, 0.025)),
                                     float(np.quantile(draws_acc, 0.975))],
        }
    return boot


# ------------------------------------------------------------------ figures
FIG = {}


def _regret_acc_panel(ax_l, ax_r, x_labels, tick_labels, mreg, aacc):
    x = np.arange(len(x_labels))
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    width = 0.26
    for e, off in zip(("first", "second", "hybrid"), (-width, 0, width)):
        ax_l.bar(x + off, [mreg[k][e]["mean_regret"] for k in x_labels],
                 width, label=e, color=colors[0] if e == "first" else
                 (colors[1] if e == "second" else colors[2]))
        ax_r.plot(x + off, [aacc[k][e]["action_acc"] for k in x_labels],
                  "o-", color=colors[0] if e == "first" else
                  (colors[1] if e == "second" else colors[2]), label=e)
    ax_l.set_xticks(x, tick_labels, rotation=30, ha="right")
    ax_r.set_xticks(x, tick_labels, rotation=30, ha="right")
    ax_l.set_ylabel("mean exact regret")
    ax_r.set_ylabel("ActionAcc (vs exact)")
    ax_l.set_title("(a) mean regret")
    ax_r.set_title("(b) ActionAcc")
    ax_l.legend(fontsize=8)
    ax_r.legend(fontsize=8)


def figures(c, n, st, A, unc_out, boot, fig_dir):
    # fig1: A subsets mean regret + ActionAcc
    subs = [k for k in A if isinstance(A[k], dict) and "first" in A[k]]
    tick = [k.replace("absS_norm_gt_", "|Sn|>").replace("S_exact_gt_0", "S>0")
            .replace("S_exact_lt_0", "S<0").replace("receiver_correct_full_",
                                                    "corr_") for k in subs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    _regret_acc_panel(a1, a2, subs, tick, A, A)
    fig.suptitle(f"R0-P6.6A: exact utility regret by subset (n={n})")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_p66_1_regret_comparison.png", dpi=150)
    plt.close(fig)

    # fig2-4: uncertainty quartiles
    for name, fname, xlab in (
        ("confidence", "fig_p66_2_confidence_vs_regret.png",
         "confidence quartile (Q1 low conf)"),
        ("entropy_norm", "fig_p66_3_entropy_vs_regret.png",
         "entropy quartile (Q1 low ent)"),
        ("margin", "fig_p66_4_margin_vs_regret.png",
         "margin quartile (Q1 small margin)"),
    ):
        q = unc_out[name]["quartiles"]
        xl = [f"Q{k}" for k in range(1, 5)]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
        _regret_acc_panel(a1, a2, xl, xl, q, q)
        fig.suptitle(f"R0-P6.6B: receiver {name} quartiles")
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    # fig5: second minus first by state (cluster bootstrap 95% CI)
    states = [s for s, _ in B_STATES]
    obs_d = [boot[s]["delta_regret_first_minus_second_obs"] for s in states]
    lo = [boot[s]["delta_regret_ci95"][0] for s in states]
    hi = [boot[s]["delta_regret_ci95"][1] for s in states]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(states))[::-1]
    err = np.array([[o - l, h - o] for o, l, h in zip(obs_d, lo, hi)]).T
    cols = ["#55a868" if o > 0 else "#c44e52" for o in obs_d]
    ax.barh(y, obs_d, xerr=err, color=cols, alpha=0.85,
            error_kw=dict(lw=1.2, capsize=3))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, [t for _, t in B_STATES])
    ax.set_xlabel("DeltaRegret = mean(Regret_first) - mean(Regret_second)  (>0: second better)")
    ax.set_title("R0-P6.6C: cluster bootstrap 95% CI (10,000 resamples of receivers)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_p66_5_second_minus_first_by_state.png", dpi=150)
    plt.close(fig)
    print("[figures] saved:", ", ".join(sorted(p.name for p in fig_dir.glob("fig_p66_*.png"))))


# ------------------------------------------------------------------ md
def _fmt3(x): return "nan" if not np.isfinite(x) else f"{x:.3f}"
def _fmt5(x): return "nan" if not np.isfinite(x) else f"{x:.5f}"


def write_md(out, meta, A, unc_out, boot, fig_names):
    L = ["# R0-P6.6: Functional Teacher Necessity & Receiver-State Audit",
         "",
         "_Sections A-C state observed numbers; 'Observed facts' and "
         "'Exploratory interpretation' are separated at the end._",
         "",
         f"- n_relations={meta['n_relations']} | n_receivers={meta['n_receivers']} | "
         f"dataset={meta['dataset']} seed={meta['train_seed']} | "
         f"confidence-recompute invariant max dev={meta['uncertainty_dev']:.2e}",
         "",
         "## A. Exact utility regret (evaluated on EXACT Q of the chosen action)",
         "",
         "| subset | est | acc | mean reg | med reg | p90 | p95 | harmful | retention |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name in A:
        if not (isinstance(A[name], dict) and "first" in A[name]):
            continue
        for e in ("first", "second", "hybrid"):
            v = A[name][e]
            L.append(
                f"| {name} | {e} | {v['action_acc']:.3f} | {_fmt5(v['mean_regret'])} "
                f"| {_fmt5(v['median_regret'])} | {_fmt5(v['p90_regret'])} "
                f"| {_fmt5(v['p95_regret'])} | {v['harmful_rate_P_Qsel_lt0']:.3f} "
                f"| {v['utility_retention']:.3f} |")
    d = A["_diagnostics"]
    L += ["",
          f"- mean Q_star={d['mean_Q_star']:.4f} | "
          f"frac rows Q_star=0 (no useful action): {d['frac_rows_Q_star_0']:.3f}",
          "",
          "## B. Receiver uncertainty quartiles (rank quartiles over CSV "
          "receivers)",
          ""]
    for name in ("confidence", "entropy_norm", "margin"):
        u = unc_out[name]
        L += [f"### {name} ({u['quartile_definition']})",
              "",
              "| quartile | n_recv | n_rel | recv corr | acc f/s/h | "
              "mean reg f/s/h | harmful f/s/h | mean\\|Sn\\| | ΔRegret (f−s) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k in ("Q1", "Q2", "Q3", "Q4"):
            v = u["quartiles"][k]
            L.append(
                f"| {k} | {v['n_receivers']} | {v['n_relations']} "
                f"| {v['receiver_correct_rate']:.3f} "
                f"| {v['first']['action_acc']:.3f}/{v['second']['action_acc']:.3f}/"
                f"{v['hybrid']['action_acc']:.3f} "
                f"| {_fmt5(v['first']['mean_regret'])}/{_fmt5(v['second']['mean_regret'])}/"
                f"{_fmt5(v['hybrid']['mean_regret'])} "
                f"| {v['first']['harmful_rate_P_Qsel_lt0']:.3f}/"
                f"{v['second']['harmful_rate_P_Qsel_lt0']:.3f}/"
                f"{v['hybrid']['harmful_rate_P_Qsel_lt0']:.3f} "
                f"| {v['mean_absS_norm']:.4f} "
                f"| {v['delta_regret_first_minus_second']:+.5f} |")
        L.append("")
    L += ["## C. Receiver-cluster bootstrap (10,000 resamples; cluster = "
          "receiver_i)",
          "",
          "| state | n_recv | ΔRegret f−s obs | 95% CI | ΔActionAcc s−f obs | 95% CI |",
          "|---|---|---|---|---|---|"]
    for s, _ in B_STATES:
        b = boot[s]
        L.append(
            f"| {s} | {b['n_receivers']} | {b['delta_regret_first_minus_second_obs']:+.5f} "
            f"| [{b['delta_regret_ci95'][0]:+.5f}, {b['delta_regret_ci95'][1]:+.5f}] "
            f"| {b['delta_actionacc_second_minus_first_obs']:+.5f} "
            f"| [{b['delta_actionacc_ci95'][0]:+.5f}, "
            f"{b['delta_actionacc_ci95'][1]:+.5f}] |")
    L += ["", "## Figures", ""]
    L += [f"- {f}" for f in fig_names]
    L += ["", "## Observed facts (auto-derived, no interpretation)", ""]
    ov = A["all"]
    L.append(
        f"- A(all): mean regret first={ov['first']['mean_regret']:.5f} / "
        f"second={ov['second']['mean_regret']:.5f} / "
        f"hybrid={ov['hybrid']['mean_regret']:.5f} vs mean Q_star="
        f"{A['_diagnostics']['mean_Q_star']:.4f}; harmful rate "
        f"first={ov['first']['harmful_rate_P_Qsel_lt0']:.3f} / "
        f"second={ov['second']['harmful_rate_P_Qsel_lt0']:.3f} / "
        f"hybrid={ov['hybrid']['harmful_rate_P_Qsel_lt0']:.3f}; utility "
        f"retention {ov['first']['utility_retention']:.3f}/"
        f"{ov['second']['utility_retention']:.3f}/"
        f"{ov['hybrid']['utility_retention']:.3f}")
    for k in ("receiver_correct_full_False", "absS_norm_gt_0.10", "S_exact_lt_0"):
        v = A[k]
        best = min(("first", "second", "hybrid"),
                   key=lambda e: v[e]["mean_regret"])
        L.append(f"- A({k}): lowest mean regret = {best} "
                 f"({v[best]['mean_regret']:.5f}); second harmful rate "
                 f"{v['second']['harmful_rate_P_Qsel_lt0']:.3f}")
    for state, _ in B_STATES:
        b = boot[state]
        L.append(
            f"- C({state}): DeltaRegret obs={b['delta_regret_first_minus_second_obs']:+.5f} "
            f"95% CI=[{b['delta_regret_ci95'][0]:+.5f}, {b['delta_regret_ci95'][1]:+.5f}] | "
            f"DeltaActionAcc obs={b['delta_actionacc_second_minus_first_obs']:+.5f} "
            f"95% CI=[{b['delta_actionacc_ci95'][0]:+.5f}, "
            f"{b['delta_actionacc_ci95'][1]:+.5f}]")
    L += ["",
          "## Exploratory interpretation (labeled; single dataset/seed 42, "
          "one probe; correlational)",
          "",
          "- Estimator-action mistakes are almost always near-tie decisions: "
          "first-order's mean regret is ~1.5% of mean Q_star and its p95 "
          "regret rounds to 0; a replacement action teacher must beat an "
          "~0.002 nats regret baseline to be necessary.",
          "- Full second-order is never harmful in any reported subset "
          "(harmful rate 0.000), consistent with its curvature penalties "
          "suppressing negative-gain picks, but on strong-interaction rows "
          "it loses 2.9x the utility of first-order (|S_norm|>0.10: 0.0186 "
          "vs 0.0065), i.e. its conservatism costs on genuinely joint-benefit "
          "relations.",
          "- Receiver-state stratification reproduces the P6.5 correct=False "
          "result at the label-free level: second-order's ActionAcc advantage "
          "is concentrated and CI-excluding-0 among low-confidence / "
          "high-entropy / small-margin receivers, while it is a clear "
          "disadvantage on confident receivers; regret-level CIs for the "
          "quartile states straddle 0, so the utility-level advantage is "
          "established only for the incorrect-receiver state (+0.0015..+0.0106).",
          "- No new method is proposed or validated here; nothing above "
          "should be read as claiming a routing/teacher method works."]
    out.write_text("\n".join(L) + "\n")


def main(args) -> None:
    c, n = _load_rel(args.csv)
    unc = _load_unc(args.uncertainty)
    st = _est_stats(c, n)

    # row->receiver join consistency
    assert np.array_equal(unc["receiver_i"], np.sort(unc["receiver_i"]))
    uidx = np.searchsorted(unc["receiver_i"], c["receiver_i"])
    assert np.array_equal(unc["receiver_i"][uidx], c["receiver_i"])
    # correctness flags must agree (receiver-level vs CSV row-level)
    assert np.array_equal(unc["correct"][uidx], c["correct"])
    # recomputed confidence must reproduce CSV full_confidence (invariant)
    dev_conf = float(np.abs(unc["confidence"][uidx] - c["full_confidence"]).max())
    assert dev_conf < 1e-6, f"uncertainty confidence deviates: {dev_conf}"

    A = analyze_A(c, n, st)
    B = analyze_B(c, n, st, unc)
    boot = analyze_C(c, n, st, unc, args.n_bootstrap, args.bootstrap_seed)

    meta = {
        "dataset": "Grocery", "train_seed": "42",
        "n_relations": n,
        "n_receivers": int(np.unique(c["receiver_i"]).size),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "uncertainty_dev": dev_conf,
    }
    fig_dir = args.out / "figures_p66"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures(c, n, st, A, B, boot, fig_dir)
    fig_names = sorted(p.name for p in fig_dir.glob("fig_p66_*.png"))

    m = {"meta": meta, "A_regret": A, "B_uncertainty_quartiles": B,
         "C_bootstrap": boot}
    (args.out / "r0_p66_state_metrics.json").write_text(
        json.dumps(m, indent=2, allow_nan=True))
    write_md(args.out / "r0_p66_state_summary.md", meta, A, B, boot, fig_names)
    print("[out] r0_p66_state_metrics.json + r0_p66_state_summary.md written to",
          args.out)
    ov = A["all"]
    print(f"[A] all: acc f/s/h = {ov['first']['action_acc']:.3f}/"
          f"{ov['second']['action_acc']:.3f}/{ov['hybrid']['action_acc']:.3f} | "
          f"mean regret f/s/h = {ov['first']['mean_regret']:.5f}/"
          f"{ov['second']['mean_regret']:.5f}/{ov['hybrid']['mean_regret']:.5f}")


if __name__ == "__main__":
    main(_parse_args())
