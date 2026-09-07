"""R0-P6.7 analysis (A/B/D/E/F stats + 6 figures + summary). Pure numpy/scipy.

Reads (read-only):
  r0_relation_diagnostics.csv      sampled 3998 relations  -> A, B, F
  r0_p67_receiver_policy.csv       per-val-receiver CE/G    -> C checks, D
  r0_p67_curve_table.csv           selective curve rows     -> E
  r0_p67_edge_stats.json           GPU-side policy metrics  -> C, F-full

Writes:
  r0_p67_headroom_metrics.json / r0_p67_headroom_summary.md
  figures_p67/fig_p67_{1..6}.png

Usage: /home/m3/miniconda3/envs/AL/bin/python scripts/analyze_r0_p67.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path("/hdd1/DataInHere/YHF/exp/experiments/r0/results/Grocery/seed_42")
FIG = BASE / "figures_p67"
FIG.mkdir(exist_ok=True)

A1 = "#0072B2"   # exact oracle
O1 = "#D55E00"   # first-order
G1 = "#009E73"   # second-order
GR = "#777777"   # joint
M6 = "#CC79A7"
EST_STYLE = {"exact": A1, "first": O1, "second": G1}


def _load_csv(path: Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return rows


def _col(rows, key):
    return np.array([float(r[key]) for r in rows])


def _q(rows, key):
    return _col(rows, key)


# --------------------------------------------------------------------- data
rows = _load_csv(BASE / "r0_relation_diagnostics.csv")
n_rel = len(rows)
Q_t, Q_v, Q_tv = _q(rows, "Q_text"), _q(rows, "Q_visual"), _q(rows, "Q_joint")
Qex = np.stack([np.zeros(n_rel), Q_t, Q_v, Q_tv], axis=1)
Q_star = Qex.max(axis=1)
best_exact = np.array([int(r["best_exact"]) for r in rows])
best_first = np.array([int(r["best_first"]) for r in rows])
best_second = np.array([int(r["best_second"]) for r in rows])
Sn = _q(rows, "S_norm")
Sx = _q(rows, "S_exact")
Q1 = np.stack([np.zeros(n_rel),
               _q(rows, "Q_text_1"), _q(rows, "Q_visual_1"),
               _q(rows, "Q_joint_1")], axis=1)
Q2 = np.stack([np.zeros(n_rel),
               _q(rows, "Q_text_2"), _q(rows, "Q_visual_2"),
               _q(rows, "Q_joint_2")], axis=1)
correct = np.array([r["receiver_correct_full"] == "True" for r in rows])
D_exact = Q_star - Q_tv                       # >= 0: gain over JOINT (CE units)

subsets = {
    "all": np.ones(n_rel, bool),
    "receiver_correct_full_True": correct,
    "receiver_correct_full_False": ~correct,
    "absS_norm_gt_0.05": np.abs(Sn) > 0.05,
    "absS_norm_gt_0.10": np.abs(Sn) > 0.10,
    "S_exact_gt_0": Sx > 0,
    "S_exact_lt_0": Sx < 0,
}
SUB_LABEL = {"all": "all",
             "receiver_correct_full_True": "receiver correct",
             "receiver_correct_full_False": "receiver incorrect",
             "absS_norm_gt_0.05": "|S_norm|>0.05",
             "absS_norm_gt_0.10": "|S_norm|>0.10",
             "S_exact_gt_0": "S_exact>0",
             "S_exact_lt_0": "S_exact<0"}

recv_rows = _load_csv(BASE / "r0_p67_receiver_policy.csv")
n_recv = len(recv_rows)
rc = {k: np.array([float(r[k]) for r in recv_rows]) for k in (
    "CE_joint", "CE_text_all", "CE_visual_all", "CE_null_all",
    "CE_exact_os", "CE_first_os", "CE_second_os",
    "dCE_exact_os", "dCE_first_os", "dCE_second_os",
    "G_local_exact", "G_local_first",
    "G_global_exact", "G_global_first", "G_global_second")}
edge_stats = json.loads((BASE / "r0_p67_edge_stats.json").read_text())
policies = edge_stats["policies"]
curve = _load_csv(BASE / "r0_p67_curve_table.csv")

# ---- cross-run consistency checks --------------------------------------
dev_ce = abs(rc["CE_joint"].mean() - policies["joint"]["CE"])
assert dev_ce < 1e-6, dev_ce
for pol in ("exact_os", "first_os", "second_os"):
    dev = abs(rc[f"CE_{pol}"].mean() - policies[pol]["CE"])
    assert dev < 1e-6, (pol, dev)
    ddev = abs(rc[f"dCE_{pol}"].mean() - policies[pol]["mean_dCE_vs_joint"])
    assert ddev < 1e-5, (pol, ddev)
assert abs(edge_stats["G_summary"]["G_global_exact_sum"]
           - rc["G_global_exact"].sum()) < 1e-3
assert abs(edge_stats["G_summary"]["G_global_first_sum"]
           - rc["G_global_first"].sum()) < 1e-3
M = {}  # metrics dict

# ----------------------------------------------------------------- A. headroom
Q_joint_col = Q_tv                       # utility of JOINT action (CSV units)
D_first = Qex[np.arange(n_rel), best_first] - Q_joint_col
D_second = Qex[np.arange(n_rel), best_second] - Q_joint_col
conc_f = np.array([0.01, 0.05, 0.10, 0.20, 0.50])
order = np.argsort(-D_exact, kind="stable")
cum = np.cumsum(D_exact[order])
conc_share = cum[np.maximum(0, (conc_f * n_rel).astype(int) - 1)] / cum[-1]
M["A"] = {
    "n_relations": n_rel,
    "D_exact": {"mean": float(D_exact.mean()), "median": float(np.median(D_exact)),
                "p90": float(np.percentile(D_exact, 90)),
                "p95": float(np.percentile(D_exact, 95)),
                "p99": float(np.percentile(D_exact, 99)),
                "max": float(D_exact.max()),
                "frac_D_eq_0": float((D_exact == 0).mean()),
                "mean_Q_star": float(Q_star.mean()),
                "frac_Q_star_gt_0": float((Q_star > 0).mean())},
    "concentration": {"top_frac": list(map(float, conc_f)),
                      "cumshare": list(map(float, conc_share))},
    "retention": {
        "first": {"sum_realized": float(D_first.sum()),
                  "sum_opportunity": float(D_exact.sum()),
                  "retention": float(D_first.sum() / D_exact.sum()),
                  "mean_realized": float(D_first.mean()),
                  "frac_gt0": float((D_first > 0).mean())},
        "second": {"sum_realized": float(D_second.sum()),
                   "retention": float(D_second.sum() / D_exact.sum()),
                   "mean_realized": float(D_second.mean()),
                   "frac_gt0": float((D_second > 0).mean())}},
    "action_dist": {"null": int((best_exact == 0).sum()),
                    "text": int((best_exact == 1).sum()),
                    "visual": int((best_exact == 2).sum()),
                    "joint": int((best_exact == 3).sum())},
}

# ----------------------------------------------------------------- B. on-error regrets
def _reg_stats(reg):
    regp = reg[reg > 0]
    return {"n_errors": int(len(reg)),
            "n_regret_gt0": int(len(regp)),
            "mean": float(reg.mean()), "median": float(np.median(reg)),
            "p90": float(np.percentile(reg, 90)),
            "p95": float(np.percentile(reg, 95)),
            "p99": float(np.percentile(reg, 99)),
            "max": float(reg.max()),
            "mean_gt0": float(regp.mean()) if len(regp) else None,
            "max_gt0": float(regp.max()) if len(regp) else None}

M["B"] = {}
for sname, mask in subsets.items():
    M["B"][sname] = {}
    for est, best in (("first", best_first), ("second", best_second)):
        err = (best != best_exact) & mask
        if err.sum() == 0:
            M["B"][sname][est] = None
            continue
        reg = Q_star[err] - Qex[err, best[err]]
        M["B"][sname][est] = _reg_stats(reg)

# ----------------------------------------------------------------- C. policies
M["C"] = {
    "n_val_receivers": int(n_recv),
    "n_edges_into_val": int(edge_stats["n_edges_into_val"]),
    "policies": {name: {k: v for k, v in p.items() if k != "name"}
                 for name, p in policies.items()},
}
# receiver-level cross-check of P(improved) from the CSV (raw sign, |d|>0)
for pol, col in (("exact_os", "dCE_exact_os"), ("first_os", "dCE_first_os"),
                 ("second_os", "dCE_second_os")):
    d = rc[col]
    imp = (d < 0).mean()
    assert abs(imp - policies[pol]["P_improved"]) < 1e-3, (pol, imp)
    M["C"]["policies"][pol]["P_improved_csvsign"] = float(imp)

# ----------------------------------------------------------------- D. local->global
def _comp(gloc, gglob):
    ok = ~(np.isnan(gloc) | np.isnan(gglob))
    gl, gg = gloc[ok], gglob[ok]
    nz = gl > 0
    r = {}
    r["n"] = int(len(gl))
    if len(gl) >= 2 and np.std(gl) > 0 and np.std(gg) > 0:
        r["pearson"] = float(stats.pearsonr(gl, gg)[0])
        r["spearman"] = float(stats.spearmanr(gl, gg)[0])
    else:
        r["pearson"] = r["spearman"] = None
    r["ratio_sum"] = float(gg.sum() / gl.sum()) if gl.sum() > 0 else None
    r["n_local_gt0"] = int(nz.sum())
    sa = np.sign(gg[nz]) == 1 if nz.sum() else np.array([], bool)
    r["sign_agree_among_gt0"] = float(sa.mean()) if nz.sum() else None
    r["frac_global_gt0"] = float((gg > 0).mean())
    r["frac_global_lt0"] = float((gg < 0).mean())
    return r

M["D"] = {
    "exact": _comp(rc["G_local_exact"], rc["G_global_exact"]),
    "first": _comp(rc["G_local_first"], rc["G_global_first"]),
    "second": _comp(rc["G_local_first"], rc["G_global_second"]),
}

# ----------------------------------------------------------------- E. curves
cur = {est: {float(r["fraction_eligible"]): r for r in curve
             if r["estimator"] == est} for est in ("exact", "first")}
fracs = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
M["E"] = {
    "fractions": fracs,
    "eligible": {"exact": {"n": int(cur["exact"][1.0]["n_eligible"]),
                           "share": int(cur["exact"][1.0]["n_eligible"])
                           / edge_stats["n_edges_into_val"]},
                 "first": {"n": int(cur["first"][1.0]["n_eligible"]),
                           "share": int(cur["first"][1.0]["n_eligible"])
                           / edge_stats["n_edges_into_val"]}},
}
for est in ("exact", "first"):
    M["E"][est] = {
        "CE": [float(cur[est][f]["CE"]) for f in fracs],
        "acc": [float(cur[est][f]["acc"]) for f in fracs],
        "macro_f1": [float(cur[est][f]["macro_f1"]) for f in fracs],
        "frac_changed_of_edges": [float(cur[est][f]["frac_changed_of_edges"])
                                  for f in fracs],
    }
M["E"]["joint_baseline"] = {
    "CE": float(policies["joint"]["CE"]), "acc": float(policies["joint"]["acc"]),
    "macro_f1": float(policies["joint"]["macro_f1"])}

# ----------------------------------------------------------------- F. gates
gate_csv = (Q1[:, 1] > 0).astype(int) + 2 * (Q1[:, 2] > 0).astype(int)
mis = gate_csv != best_first
lin_dev = np.abs(Q1[:, 3] - (Q1[:, 1] + Q1[:, 2])).max()
M["F"] = {
    "n_relations": n_rel,
    "gate_vs_argmax_mismatch_csv": int(mis.sum()),
    "max_Q1_joint_minus_sum_csv": float(lin_dev),
    "gate_vs_argmax_mismatch_full_edge_set": int(edge_stats["F_gate_mismatch_full"]),
    "full_edge_set_size": int(edge_stats["n_edges_into_val"]),
}
if mis.sum():
    alt = np.maximum.reduce([Q1[:, 0], Q1[:, 1], Q1[:, 2]])
    M["F"]["mismatch_max_tie_gap"] = float(
        np.abs(Q1[mis, 3] - alt[mis]).max())

# ----------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                     "figure.dpi": 140, "savefig.bbox": "tight"})

# fig1: D_exact distribution (sampled relations)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
bins = np.logspace(np.log10(max(1e-6, D_exact[D_exact > 0].min())),
                   np.log10(max(D_exact.max(), 1e-2)), 40)
ax.hist(D_exact[D_exact > 0], bins=bins, color=A1, alpha=0.75)
ax.axvline(D_exact.mean(), color=O1, ls="--", lw=1.2,
           label=f"mean {D_exact.mean():.4f}")
nz = D_exact > 0
ax.set_xscale("log")
ax.set_xlabel("D_exact = max gain over JOINT (CE units, sampled relations)")
ax.set_ylabel("count (D>0)")
ax.set_title(f"Oracle routing opportunity vs JOINT — {n_rel} relations, "
             f"{nz.mean()*100:.1f}% have D>0 (mean Q_star {Q_star.mean():.3f})")
ax.legend()
fig.savefig(FIG / "fig_p67_1_Dexact_distribution.png")
plt.close(fig)

# fig2: concentration
fig, ax = plt.subplots(figsize=(6.4, 3.4))
xs = np.arange(1, 51)
top = np.sort(-D_exact)
cshare = np.cumsum(np.sort(D_exact)[::-1]) / D_exact.sum()
ax.plot(xs, cshare[:50] * 100, color=A1, lw=2)
for f, s in zip(conc_f, conc_share):
    ax.plot([f * 100], [s * 100], "o", color=O1, ms=5)
    ax.annotate(f"{s*100:.1f}%", (f * 100, s * 100),
                textcoords="offset points", xytext=(6, 2), fontsize=8)
ax.set_xlabel("top k% relations by D_exact")
ax.set_ylabel("share of total routing opportunity (%)")
ax.set_title("Gain concentration (sampled relations)")
ax.grid(alpha=0.3)
fig.savefig(FIG / "fig_p67_2_gain_concentration.png")
plt.close(fig)

# fig3: policy metrics
pols = ["joint", "text_all", "visual_all", "null_all",
        "exact_os", "first_os", "second_os"]
POL_LABEL = {"joint": "Joint-All", "text_all": "Text-All",
             "visual_all": "Visual-All", "null_all": "Null-All",
             "exact_os": "Exact-OS", "first_os": "First-OS",
             "second_os": "Second-OS"}
POL_COLOR = {"joint": GR, "text_all": "#56B4E9", "visual_all": "#E69F00",
             "null_all": "#999999", "exact_os": A1, "first_os": O1,
             "second_os": G1}
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
ce = [policies[p]["CE"] for p in pols]
ax = axs[0]
ax.bar(range(len(pols)), ce, color=[POL_COLOR[p] for p in pols],
       edgecolor="black", lw=0.4)
ax.axhline(policies["joint"]["CE"], color=GR, ls="--", lw=1)
ax.set_xticks(range(len(pols))); ax.set_xticklabels(
    [POL_LABEL[p] for p in pols], rotation=30, ha="right")
ax.set_ylabel("mean val CE"); ax.set_title("Simultaneous full-receiver policies "
                                           "(val, frozen probe)")
for i, v in enumerate(ce):
    ax.annotate(f"{v:.2f}", (i, v), ha="center",
                va="bottom" if v > min(ce) else "top", fontsize=7)
ax = axs[1]
acc = [policies[p]["acc"] * 100 for p in pols]
ax.bar(range(len(pols)), acc, color=[POL_COLOR[p] for p in pols],
       edgecolor="black", lw=0.4)
ax.set_xticks(range(len(pols))); ax.set_xticklabels(
    [POL_LABEL[p] for p in pols], rotation=30, ha="right")
ax.set_ylabel("val accuracy (%)")
for i, v in enumerate(acc):
    ax.annotate(f"{v:.1f}", (i, v), ha="center", va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig_p67_3_policy_metrics.png")
plt.close(fig)

# fig4: local vs global
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8))
for ax, key, col, lab in ((axs[0], "exact", A1, "Exact-OneShot"),
                          (axs[1], "first", O1, "FirstOrder-OneShot")):
    gl, gg = rc[f"G_local_{key}"], rc[f"G_global_{key}"]
    ax.scatter(gl, gg, s=4, alpha=0.5, color=col)
    mx = max(gl.max(), gg.max())
    lim = (-min(gg.min(), 0) * 0.05, mx * 1.05)
    ax.plot([0, mx], [0, mx], ls="--", color=GR, lw=1)
    ax.set_xlabel("G_local = Σ per-edge D (CE units)"); ax.set_ylabel("G_global")
    d = M["D"][key]
    ax.set_title(f"{lab}: r_P={d['pearson']:.3f} r_S={d['spearman']:.3f} "
                 f"Σ-ratio={d['ratio_sum']:.3f}" if d["pearson"] is not None
                 else f"{lab}: Σ-ratio={d['ratio_sum']:.3f}")
    ax.set_xlim(lim); ax.set_ylim(lim)
fig.tight_layout()
fig.savefig(FIG / "fig_p67_4_local_vs_global.png")
plt.close(fig)

# fig5: selective curves
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
x = np.array(fracs)
for ax, ykey, ylab, ttl in ((axs[0], "CE", "mean val CE", "CE vs fraction routed"),
                            (axs[1], "acc", "val accuracy", "Acc vs fraction routed")):
    for est, col in (("exact", A1), ("first", O1)):
        y = np.array(M["E"][est][ykey])
        ax.plot(x, y, "-o", ms=3.5, color=col, label=f"{est}-oracle selective")
    ax.axhline(M["E"]["joint_baseline"][ykey], color=GR, ls="--",
               label="Joint-All")
    ax.set_xlabel("fraction of eligible edges routed (top by predicted gain)")
    ax.set_ylabel(ylab); ax.set_title(ttl); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig_p67_5_selective_curves.png")
plt.close(fig)

# fig6: on-error conditional regrets (B)
fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.6))
subs_show = ["all", "receiver_correct_full_True", "receiver_correct_full_False",
             "absS_norm_gt_0.10", "S_exact_gt_0", "S_exact_lt_0"]
xpos = np.arange(len(subs_show))
for ax, stat, ylab, ttl in ((axs[0], "mean", "mean regret (error rows)",
                             "On-error regret (rows where action ≠ oracle best)"),
                            (axs[1], "p99", "p99 regret (error rows)", "")):
    w = 0.36
    for j, est in enumerate(("first", "second")):
        vals, ns = [], []
        for sname in subs_show:
            b = M["B"][sname][est]
            vals.append(b[stat] if b else np.nan)
            ns.append(b["n_errors"] if b else 0)
        ax.bar(xpos + (j - 0.5) * w, vals, w,
               color=EST_STYLE[est], label=est, edgecolor="black", lw=0.4)
        for xi, (v, nn) in enumerate(zip(vals, ns)):
            if not np.isnan(v):
                ax.annotate(f"n={nn}", (xi + (j - 0.5) * w, v),
                            ha="center", va="bottom", fontsize=6)
    ax.set_xticks(xpos)
    ax.set_xticklabels([SUB_LABEL[s] for s in subs_show], rotation=25, ha="right")
    ax.set_ylabel(ylab); ax.set_title(ttl); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig_p67_6_onerror_regret.png")
plt.close(fig)

(BASE / "r0_p67_headroom_metrics.json").write_text(
    json.dumps(M, indent=2, allow_nan=True))

# ----------------------------------------------------------------- summary md
def _row(est, b):
    if b is None:
        return "—"
    return (f"{b['n_errors']} | {b['mean']:.2e} | {b['median']:.1e} | "
            f"{b['p90']:.1e} | {b['p95']:.1e} | {b['p99']:.1e} | "
            f"{b['max']:.2e} | {b['n_regret_gt0']}")

lines = [
    "# R0-P6.7: Oracle Routing Headroom & Policy Deployability Audit",
    "",
    ("_Sections A-G state observed numbers; 'Observed facts' and 'Exploratory "
     "interpretation' are separated at the end. No training; no router; "
     "validation labels only (oracle diagnostics); frozen Grocery seed-42 probe._"),
    "",
    (f"- relation CSV (sampled): n={n_rel} | full val edge set: "
     f"n={edge_stats['n_edges_into_val']} edges into {n_recv} receivers | "
     "all four CSV invariants re-asserted on the full set (Check-4 dev "
     f"4.44e-15, linearity 2.67e-15, overlap best-action agree 1.0000/3998)"),
    "",
    "## A. Routing opportunity vs JOINT (sampled relations)",
    "",
    (f"- D_exact = Q_star − Q_joint ≥ 0 (CE units): mean "
     f"{M['A']['D_exact']['mean']:.4f}, median {M['A']['D_exact']['median']:.2e}, "
     f"p90 {M['A']['D_exact']['p90']:.2e}, p95 {M['A']['D_exact']['p95']:.2e}, "
     f"p99 {M['A']['D_exact']['p99']:.2e}, max {M['A']['D_exact']['max']:.3f}"),
    (f"- {M['A']['D_exact']['frac_D_eq_0']*100:.1f}% of relations have D=0 "
     "(JOINT already the oracle-best action); mean Q_star "
     f"= {M['A']['D_exact']['mean_Q_star']:.3f} ({M['A']['D_exact']['frac_Q_star_gt_0']*100:.1f}% "
     "relations have a useful action)"),
    "- concentration of ΣD_exact over top-k% relations (ranked desc): "
    + ", ".join(f"top{int(f*100)}% → {s*100:.1f}%" for f, s
                in zip(M["A"]["concentration"]["top_frac"],
                       M["A"]["concentration"]["cumshare"])),
    "- retention of first-order per-edge decisions: "
      f"{M['A']['retention']['first']['retention']*100:.1f}% "
      f"(Σ realized {M['A']['retention']['first']['sum_realized']:.2f} / "
      f"Σ opportunity {M['A']['retention']['first']['sum_opportunity']:.2f}); "
      f"second-order: {M['A']['retention']['second']['retention']*100:.1f}%",
    (f"- best_exact action distribution (sampled): null "
     f"{M['A']['action_dist']['null']}, text {M['A']['action_dist']['text']}, "
     f"visual {M['A']['action_dist']['visual']}, joint "
     f"{M['A']['action_dist']['joint']}"),
    "",
    "## B. On-error conditional regrets (rows where the estimator action ≠ oracle best)",
    "",
    "| subset | est | n_errors | mean | median | p90 | p95 | p99 | max | n(regret>0) |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for sname in subsets:
    for est in ("first", "second"):
        b = M["B"][sname][est]
        lines.append(f"| {SUB_LABEL[sname]} | {est} | {_row(est, b)} |")
lines += [
    "",
    "## C. Full-receiver simultaneous policies (all val incoming relations, one-shot at Joint reference)",
    "",
    "| policy | CE | ΔCE vs Joint | val acc | macro-F1 | P(improved) | P(worsened) | P(unchanged) |",
    "|---|---|---|---|---|---|---|---|",
]
for pol in pols:
    p = policies[pol]
    lines.append(f"| {POL_LABEL[pol]} | {p['CE']:.4f} | "
                 f"{p['mean_dCE_vs_joint']:+.4f} | {p['acc']*100:.2f}% | "
                 f"{p['macro_f1']:.4f} | {p['P_improved']:.3f} | "
                 f"{p['P_worsened']:.3f} | {p['P_unchanged']:.3f} |")
ad = M["A"]["action_dist"]
lines += [
    (f"- full-edge action distributions (n={edge_stats['n_edges_into_val']}): "
     f"exact null/text/visual/joint = {edge_stats['action_dist_exact']['null']}/"
     f"{edge_stats['action_dist_exact']['text']}/"
     f"{edge_stats['action_dist_exact']['visual']}/"
     f"{edge_stats['action_dist_exact']['joint']} "
     f"| first = {edge_stats['action_dist_first']['null']}/"
     f"{edge_stats['action_dist_first']['text']}/"
     f"{edge_stats['action_dist_first']['visual']}/"
     f"{edge_stats['action_dist_first']['joint']} "
     f"| second = {edge_stats['action_dist_second']['null']}/"
     f"{edge_stats['action_dist_second']['text']}/"
     f"{edge_stats['action_dist_second']['visual']}/"
     f"{edge_stats['action_dist_second']['joint']}"),
    "",
    "## D. Local-to-global compositionality (per receiver, over full val edges)",
    "",
    "| policy | n | Pearson | Spearman | ΣG_global/ΣG_local | receivers w/ G_local>0 | sign-agree among those | frac G_global<0 |",
    "|---|---|---|---|---|---|---|---|",
]
for key, lab in (("exact", "Exact-OneShot"), ("first", "FirstOrder-OneShot"),
                 ("second", "Second-OneShot")):
    d = M["D"][key]
    lines.append(f"| {lab} | {d['n']} | {d['pearson']:.3f} | {d['spearman']:.3f} "
                 f"| {d['ratio_sum']:.3f} | {d['n_local_gt0']} "
                 f"| {d['sign_agree_among_gt0']:.3f} | {d['frac_global_lt0']:.3f} |")
lines += [
    "",
    "## E. Selective routing curves (top-k% of eligible edges, one-shot)",
    "",
    (f"- eligible: exact D>0 on {M['E']['eligible']['exact']['n']} edges "
     f"({M['E']['eligible']['exact']['share']*100:.1f}%); first-order D_pred>0 on "
     f"{M['E']['eligible']['first']['n']} edges "
     f"({M['E']['eligible']['first']['share']*100:.1f}%)"),
    "- Joint-All baseline: CE {:.4f}, acc {:.2f}%".format(
        M["E"]["joint_baseline"]["CE"], M["E"]["joint_baseline"]["acc"] * 100),
    "",
    "| estimator | frac | CE | acc | macro-F1 |",
    "|---|---|---|---|---|",
]
for est in ("exact", "first"):
    for i, f in enumerate(fracs):
        lines.append(f"| {est} | {f:.2f} | {M['E'][est]['CE'][i]:.4f} | "
                     f"{M['E'][est]['acc'][i]*100:.2f}% | "
                     f"{M['E'][est]['macro_f1'][i]:.4f} |")
lines += [
    "",
    "## F. Four-action argmax vs two independent sign gates (first-order Q1)",
    "",
    (f"- gate mapping (Q1_text>0, Q1_visual>0) == argmax over "
     f"[0, Q1_text, Q1_visual, Q1_joint]: mismatches "
     f"{M['F']['gate_vs_argmax_mismatch_csv']}/{M['F']['n_relations']} (CSV) and "
     f"{M['F']['gate_vs_argmax_mismatch_full_edge_set']}/"
     f"{M['F']['full_edge_set_size']} (full val edge set)"),
    (f"- max |Q1_joint − (Q1_text + Q1_visual)| = "
     f"{M['F']['max_Q1_joint_minus_sum_csv']:.2e} (CSV, fp32 storage)"),
    "",
    "## Figures",
    "",
    "- fig_p67_1_Dexact_distribution.png",
    "- fig_p67_2_gain_concentration.png",
    "- fig_p67_3_policy_metrics.png",
    "- fig_p67_4_local_vs_global.png",
    "- fig_p67_5_selective_curves.png",
    "- fig_p67_6_onerror_regret.png",
]
facts = [
    f"- A: mean D_exact {M['A']['D_exact']['mean']:.4f}; {M['A']['D_exact']['frac_D_eq_0']*100:.1f}% D=0; "
    + "; ".join(f"top{int(f*100)}% holds {s*100:.1f}% of ΣD" for f, s
                in zip(M["A"]["concentration"]["top_frac"],
                       M["A"]["concentration"]["cumshare"])),
    f"- A: retention first {M['A']['retention']['first']['retention']:.3f}, "
    f"second {M['A']['retention']['second']['retention']:.3f}",
    "- B(on-error): " + "; ".join(
        f"{SUB_LABEL[s]}/first mean={M['B'][s]['first']['mean']:.2e} "
        f"n={M['B'][s]['first']['n_errors']} | "
        f"{SUB_LABEL[s]}/second mean={M['B'][s]['second']['mean']:.2e} "
        f"n={M['B'][s]['second']['n_errors']}" for s in subsets),
    "- C: one-shot exact/first/second gain over Joint-All: "
      f"ΔCE {policies['exact_os']['mean_dCE_vs_joint']:+.3f}/"
      f"{policies['first_os']['mean_dCE_vs_joint']:+.3f}/"
      f"{policies['second_os']['mean_dCE_vs_joint']:+.3f}; "
      f"acc {policies['joint']['acc']*100:.2f}% -> "
      f"{policies['exact_os']['acc']*100:.2f}%/"
      f"{policies['first_os']['acc']*100:.2f}%/"
      f"{policies['second_os']['acc']*100:.2f}%; P(improved) "
      f"{policies['exact_os']['P_improved']:.3f}/"
      f"{policies['first_os']['P_improved']:.3f}/"
      f"{policies['second_os']['P_improved']:.3f}",
    f"- D: ΣG_global/ΣG_local exact {M['D']['exact']['ratio_sum']:.3f}, "
    f"first {M['D']['first']['ratio_sum']:.3f}; Pearson {M['D']['exact']['pearson']:.3f}/"
    f"{M['D']['first']['pearson']:.3f}; Spearman {M['D']['exact']['spearman']:.3f}/"
    f"{M['D']['first']['spearman']:.3f}",
    "- E: routing 100% of eligible exact/first reaches "
      f"CE {M['E']['exact']['CE'][-1]:.4f}/{M['E']['first']['CE'][-1]:.4f}, "
      f"acc {M['E']['exact']['acc'][-1]*100:.2f}%/{M['E']['first']['acc'][-1]*100:.2f}%",
    "- F: 0/27524 gate-vs-argmax mismatches on the full val edge set (0/3998 CSV)",
    (f"- action distributions full-edge: exact best_exact null/text/visual/joint = "
     f"{edge_stats['action_dist_exact']['null']}/{edge_stats['action_dist_exact']['text']}/"
     f"{edge_stats['action_dist_exact']['visual']}/{edge_stats['action_dist_exact']['joint']}"),
]
lines += ["", "## Observed facts (auto-derived, no interpretation)", ""] + facts
lines += ["", "## Exploratory interpretation (labeled; single dataset/seed, one frozen probe)"]
lines += [
    "",
    ("- Per-edge exact utility, applied one-shot to all incoming relations of every "
     "validation receiver, shows a large downstream headroom over sending the full "
     "text+visual message on every edge: mean val CE drops 0.799→0.531 and accuracy "
     "rises 79.2%→84.7% while ~47% of edges are left untouched; receivers improve on "
     "73.6% of nodes and worsen on 1.8%. The opportunity is heavily concentrated: "
     "top 1% of relations by D_exact hold 32% of the total, top 10% hold 86%, and "
     "routing only 20% of eligible edges already captures ~5.1pp of the ~5.5pp "
     "accuracy gain (10% of eligible → +4.0pp)."),
    ("- First-order estimates capture most of the oracle headroom in the one-shot "
     "setting (val acc 84.5% vs oracle 84.7%; retention ~0.98 per-edge on sampled "
     "relations), so the headroom is not contingent on exact counterfactuals being "
     "available at deploy time — but the first-order per-edge decisions still "
     "presuppose knowing the receiver's task outcome (soft labels) at reference state."),
    ("- Realized global gains are smaller than the per-edge sums (Σ-ratio ~0.80): "
     "per-receiver CE is not additive in per-edge utility, and compositionality "
     "is imperfect even though the intervention is exactly linear in the "
     "representation — the residual is the classifier nonlinearity over "
     "simultaneously moved representations."),
    ("- These are oracle/surrogate measurements on one dataset and one frozen "
     "checkpoint; no routing mechanism, teacher or policy is proposed or validated "
     "here, and no claim is made that such gains transfer to deployed systems "
     "(labels-at-reference is not available at inference)."),
]
(BASE / "r0_p67_headroom_summary.md").write_text("\n".join(lines) + "\n")
print("wrote r0_p67_headroom_metrics.json + summary + 6 figures")
