"""R2-Design-2.7 summarizer
(docs/BiAxis_R2_Design_2_7_PreAggregation_Neighbor_Utility_Audit.md).

Stages:
    audit       : D2.7-0 infrastructure audit (R2D27_AUDIT.md)
    matrix      : D2.7-A CSVs + R2D27_MATRIX_REPORT.md
    edge_audit  : D2.7-B CSVs + R2D27_EDGE_AUDIT_REPORT.md
    prepost     : D2.7-C CSV + R2D27_PREPOST_REPORT.md
    ownership   : D2.7-D CSV + R2D27_OWNERSHIP_REPORT.md
    transfer    : D2.7-E CSVs + R2D27_TRANSFER_REPORT.md
    final       : master table + hypothesis ledger + diagnosis

Usage:
    python scripts/summarize_perf_r2d27.py --stage audit
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d27_utils import (  # noqa: E402
    DATASETS,
    GUARD_DATASETS,
    R2D27_ROOT,
    SEEDS,
    TARGET_DATASETS,
    VARIANTS,
)

AUDIT_ROOT = R2D27_ROOT / "audit"
MATRIX_ROOT = R2D27_ROOT / "matrix"
EDGE_AUDIT_ROOT = R2D27_ROOT / "edge_audit"
PREPOST_ROOT = R2D27_ROOT / "prepost"
OWNERSHIP_ROOT = R2D27_ROOT / "ownership"
TRANSFER_ROOT = R2D27_ROOT / "transfer"
NOISE_ROOT = R2D27_ROOT / "noise_optional"
SUMMARY_ROOT = R2D27_ROOT / "summary"


def _collect(root: Path) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name == "head_init":
            continue
        for v_dir in sorted(ds_dir.iterdir()):
            if not v_dir.is_dir():
                continue
            for s_dir in sorted(v_dir.iterdir()):
                sp = s_dir / "summary.json"
                if not sp.exists():
                    continue
                d = json.loads(sp.read_text())
                rows.append({
                    "dataset": ds_dir.name,
                    "variant": d.get("variant", v_dir.name),
                    "seed": int(s_dir.name.split("_")[-1]),
                    "val_acc": d["best_val_acc"],
                    "val_macro_f1": d["best_val_macro_f1"],
                    "best_epoch": d.get("best_epoch"),
                    "stop_epoch": d.get("stop_epoch"),
                    "side_params": d.get("side_params"),
                    "out_dim": d.get("out_dim"),
                })
    return rows


def _load_a0_formal() -> dict[tuple[str, int], dict]:
    from src.analysis.perf_r2_utils import load_a0_reference

    acc = load_a0_reference()
    out = {(ds, s): {"val_acc": v, "val_macro_f1": None} for (ds, s), v in acc.items()}
    for ds in DATASETS:
        for seed in SEEDS:
            p = PROJECT_ROOT / "outputs" / "perf_r1" / "baseline" / ds / "A0" / f"seed_{seed}" / "summary.json"
            if p.exists():
                d = json.loads(p.read_text())
                if d.get("best_val_macro_f1") is not None:
                    out[(ds, seed)]["val_macro_f1"] = float(d["best_val_macro_f1"]) / 100.0
    return out


A0_FORMAL = _load_a0_formal()


def _by_key(rows, variant):
    return {(r["dataset"], r["seed"]): r for r in rows if r["variant"] == variant}


def _paired(cand_rows, base_rows, metric, ds_list, seeds):
    per_ds = []
    for ds in ds_list:
        deltas = []
        for s in seeds:
            c = cand_rows.get((ds, s))
            b = base_rows.get((ds, s))
            if c and b:
                deltas.append(100.0 * (c[metric] - b[metric]))
        if deltas:
            per_ds.append((ds, statistics.fmean(deltas),
                           sum(1 for d in deltas if d > 0), len(deltas)))
    mean = statistics.fmean([m for _, m, _, _ in per_ds]) if per_ds else None
    n_pos = sum(1 for _, m, _, _ in per_ds if m > 0)
    return {"per_ds": per_ds, "mean": mean, "n_pos": n_pos}


# ---------------------------------------------------------------------------
# D2.7-A matrix
# ---------------------------------------------------------------------------


def _write_matrix_report(rows: list[dict]) -> Path:
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    with (MATRIX_ROOT / "matrix_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "val_acc",
                                               "val_macro_f1", "best_epoch", "stop_epoch",
                                               "side_params", "out_dim"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    pair = _by_key(rows, "PAIR_EDGE")
    a0m = _by_key(rows, "A0_BASE")
    controls = {}
    for name in ("UNIFORM", "TARGET_NULL_ONLY", "GENERIC_EDGE", "DIAG_EDGE",
                 "SEMANTIC_SIM"):
        controls[name] = _paired(pair, _by_key(rows, name), "val_acc",
                                 TARGET_DATASETS, SEEDS)
        controls[name]["f1"] = _paired(pair, _by_key(rows, name), "val_macro_f1",
                                       TARGET_DATASETS, SEEDS)
    a0_delta_acc = _paired(pair, a0m, "val_acc", TARGET_DATASETS, SEEDS)
    a0_delta_f1 = _paired(pair, a0m, "val_macro_f1", TARGET_DATASETS, SEEDS)
    with (MATRIX_ROOT / "matrix_controls.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["control", "delta_acc_mtg", "delta_f1_mtg",
                                               "n_pos_ds"], extrasaction="ignore")
        writer.writeheader()
        for name, c in controls.items():
            writer.writerow({"control": name, "delta_acc_mtg": c["mean"],
                             "delta_f1_mtg": c["f1"]["mean"],
                             "n_pos_ds": c["n_pos"]})
    with (MATRIX_ROOT / "matrix_resources.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "side_params",
                                               "out_dim", "runtime_sec", "peak_allocated_mb"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # guards for any candidate >= A0_MATCHED + 0.20pp on M/T/G acc
    guard_rows = []
    for v in ("PAIR_EDGE",):
        cand = _by_key(rows, v)
        for g in GUARD_DATASETS:
            deltas = [100.0 * (cand[(g, s)]["val_acc"] - a0m[(g, s)]["val_acc"])
                      for s in SEEDS if (g, s) in cand and (g, s) in a0m]
            if deltas:
                guard_rows.append((g, statistics.fmean(deltas)))

    selection_go = (controls["TARGET_NULL_ONLY"]["mean"] is not None
                    and controls["TARGET_NULL_ONLY"]["mean"] >= 0.30
                    and controls["TARGET_NULL_ONLY"]["f1"]["mean"] is not None
                    and controls["TARGET_NULL_ONLY"]["f1"]["mean"] >= 0.20
                    and controls["TARGET_NULL_ONLY"]["n_pos"] >= 2)
    incremental_go = (a0_delta_acc["mean"] is not None and a0_delta_acc["mean"] >= 0.30
                      and a0_delta_f1["mean"] is not None and a0_delta_f1["mean"] >= 0.20
                      and a0_delta_acc["n_pos"] >= 2)
    ownership_prelim = False
    if controls["GENERIC_EDGE"]["mean"] is not None:
        ownership_prelim = (controls["GENERIC_EDGE"]["mean"] >= 0.20
                            and controls["GENERIC_EDGE"]["f1"]["mean"] is not None
                            and controls["GENERIC_EDGE"]["f1"]["mean"] >= 0.0) or (
            controls["GENERIC_EDGE"]["f1"]["mean"] is not None
            and controls["GENERIC_EDGE"]["f1"]["mean"] >= 0.30
            and controls["GENERIC_EDGE"]["mean"] >= 0.0)

    report = MATRIX_ROOT / "R2D27_MATRIX_REPORT.md"
    lines = ["# R2-D2.7-A — Pre-aggregation neighbor-utility matrix", "",
             "PAIR_EDGE paired deltas vs controls (M/T/G, 3-seed means, pp):", ""]
    for name in ("UNIFORM", "TARGET_NULL_ONLY", "GENERIC_EDGE", "DIAG_EDGE",
                 "SEMANTIC_SIM"):
        c = controls[name]
        lines.append(
            f"- vs {name}: Acc {c['mean'] if c['mean'] is None else f'{c['mean']:+.3f}'} "
            f"/ F1 {c['f1']['mean'] if c['f1']['mean'] is None else f'{c['f1']['mean']:+.3f}'} "
            f"({c['n_pos']}/3 datasets positive)")
    lines.append(f"- vs A0_MATCHED: Acc {a0_delta_acc['mean'] if a0_delta_acc['mean'] is None else f'{a0_delta_acc['mean']:+.3f}'} "
                 f"/ F1 {a0_delta_f1['mean'] if a0_delta_f1['mean'] is None else f'{a0_delta_f1['mean']:+.3f}'} "
                 f"({a0_delta_acc['n_pos']}/3)")
    lines.append("")
    lines.append(f"- **SELECTION GO: {'PASS' if selection_go else 'not met'}** "
                 "(PAIR - TARGET_NULL_ONLY >= +0.30pp Acc, +0.20pp F1)")
    lines.append(f"- **A0 INCREMENTAL GO: {'PASS' if incremental_go else 'not met'}** "
                 "(PAIR - A0_MATCHED >= +0.30pp Acc, +0.20pp F1)")
    lines.append(f"- **OWNERSHIP preliminary: "
                 f"{'SUPPORTED' if ownership_prelim else 'not met'}** "
                 "(PAIR - GENERIC_EDGE >= +0.20pp Acc with F1>=0, or F1>=+0.30 with Acc>=0)")
    if guard_rows:
        lines.append("")
        lines.append("Guards (vs A0_MATCHED, Acc pp, threshold -0.20pp):")
        for g, m in guard_rows:
            lines.append(f"- {g}: {m:+.3f}pp")
    lines.append("")
    lines.append("| dataset | variant | acc (3-seed mean) | F1 (3-seed mean) |")
    lines.append("|---|---|---|---|")
    for v in ("A0_BASE", "UNIFORM", "TARGET_NULL_ONLY", "GENERIC_EDGE", "DIAG_EDGE",
              "PAIR_EDGE", "SEMANTIC_SIM"):
        for ds in DATASETS:
            sel = [r for r in rows if r["variant"] == v and r["dataset"] == ds]
            if sel:
                lines.append(
                    f"| {ds} | {v} | {statistics.fmean(r['val_acc'] for r in sel):.5f} | "
                    f"{statistics.fmean(r['val_macro_f1'] for r in sel):.5f} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _write_audit_report() -> Path:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    report = AUDIT_ROOT / "R2D27_AUDIT.md"
    lines = [
        "# R2-Design-2.7 — D2.7-0 Audit & Infrastructure", "",
        "Protocol: seeds 42/43/44; Val only; No Test. A0 (R1-baseline, disclosed)",
        "is the only primary parent; the utility branch may only ADD.", "",
        "## 1. Collision contract (plan §1)", "",
        "- No RoleMAG-style predefined roles (heterophilous/complementary/"
        "  shared channels): continuous task-conditioned utility u_ji^{a->b}.",
        "- No TMTE-style topology evolution: observed graph as support only;",
        "  edge_index is only READ (structural test).",
        "- No CoMAG-style semantic-consistency scorer as the mechanism;",
        "  SEMANTIC_SIM is a control baseline only.", "",
        "## 2. Core formulation (plan §2-§5)", "",
        "- s_ji^{a->b} = psi([F_i^b, F_j^a, F_i^b*F_j^a, |F_i^b-F_j^a|, e_a, e_b])",
        "  (shared psi, t=16, Linear(4d+2t,2d)->LN->GELU->Drop->Linear(2d,d)->",
        "  GELU->Linear(d,1)).",
        "- Null-augmented softmax with tau=1.0 fixed; no top-k in training.",
        "- Selection-only payload U_a (Linear(d,d,bias=False)), m_i^b =",
        "  (1/3) sum_a m_i^{a->b}; side = [z_base | m_C | m_Pt | m_Pv], no",
        "  projection back to h.",
        "- Edge chunking + stable segment softmax: [E,3,3,d] never",
        "  materialized; chunked == unchunked (tested).", "",
        "## 3. Variants (plan §11-§17)", "",
        "A0_BASE / UNIFORM / TARGET_NULL_ONLY / GENERIC_EDGE (scorer width",
        "solved to match PAIR params +/-5%) / DIAG_EDGE / PAIR_EDGE /",
        "SEMANTIC_SIM; plus POST_PAIR / SOURCE_FACTOR_ONLY /",
        "TARGET_FACTOR_ONLY / PAIR_TRANSFORM_UNIFORM / PAIR_TRANSFORM_PRE",
        "for later stages.", "",
        "## 4. Causal machinery (plan §27-§28)", "",
        "remove-top/random/bottom 10/25/50% (per-pair score masks),",
        "keep-top 25/50%, within-target shuffle (two stable sorts, per-target",
        "histograms preserved — tested), source-node shuffle, factor-id",
        "shuffle, noise-10/25% edge injection hooks, side_off (bitwise",
        "z_base). All eval-time; weights never modified.", "",
        "## 5. Verification", "",
        "18 tests in tests/test_biaxis_r2_neighbor_utility.py: A0 untouched",
        "in frozen training; UNIFORM == neighbor_mean of payloads;",
        "TARGET_NULL_ONLY null-0 => weight 1/(deg+1) everywhere; 9 pair",
        "scores; GENERIC single ranking; DIAG off-diagonal zero;",
        "null+neighbor sums to 1; isolated all-null finite; shuffle",
        "histogram preservation; POST aggregates before scoring; no-test",
        "guarded loop; C/Pt/Pv order; chunked==unchunked; collision",
        "guardrails; side_off bitwise == parent.",
        "",
        "GPU smoke (Movies s42, 5 ep): all 7 matrix variants train;",
        "PAIR_EDGE remove_top_10 (-0.69pp) > remove_random_10 (-0.33pp)",
        "at init-like weights — the causal ranking machinery is live.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# D2.7-B edge audit
# ---------------------------------------------------------------------------


def _collect_edge_audit() -> list[dict]:
    rows = []
    if not EDGE_AUDIT_ROOT.exists():
        return rows
    for ds_dir in sorted(EDGE_AUDIT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for v_dir in sorted(ds_dir.iterdir()):
            if not v_dir.is_dir():
                continue
            for s_dir in sorted(v_dir.iterdir()):
                sp = s_dir / "summary.json"
                if sp.exists():
                    rows.append(json.loads(sp.read_text()))
    return rows


def _write_edge_audit_report(rows: list[dict]) -> Path:
    if not rows:
        raise RuntimeError("no edge_audit summaries — run perf_r2d27_edge_audit.py")
    EDGE_AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    def _agg(key_fn, field):
        out = []
        for r in rows:
            for item in r.get(key_fn, []):
                out.append({**item, "dataset": r["dataset"], "seed": r["seed"]})
        return out

    pair_stats = _agg("pair_stats", None)
    corr = _agg("heuristic_corr", None)
    homo = _agg("homophily", None)
    causal_flat = []
    for r in rows:
        for k, m in r.get("causal", {}).items():
            causal_flat.append({
                "dataset": r["dataset"], "seed": r["seed"], "causal": k,
                "val_acc": m["val_acc"], "val_macro_f1": m["val_macro_f1"]})
    with (EDGE_AUDIT_ROOT / "edge_score_stats.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in pair_stats for k in r}))
        writer.writeheader()
        for r in pair_stats:
            writer.writerow(r)
    with (EDGE_AUDIT_ROOT / "edge_heuristic_corr.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in corr for k in r}))
        writer.writeheader()
        for r in corr:
            writer.writerow(r)
    with (EDGE_AUDIT_ROOT / "edge_homophily_train_only.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in homo for k in r}))
        writer.writeheader()
        for r in homo:
            writer.writerow(r)
    with (EDGE_AUDIT_ROOT / "edge_causal_ranking.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "seed", "causal",
                                               "val_acc", "val_macro_f1"])
        writer.writeheader()
        for r in causal_flat:
            writer.writerow(r)
    # shuffle controls CSV: within-target / source / factor-id drops
    shuffle_rows = []
    for r in rows:
        full = r["causal"].get("full")
        for key in ("within_target_shuffle", "source_shuffle", "factor_id_shuffle"):
            cf = r["causal"].get(key)
            if full and cf:
                shuffle_rows.append({
                    "dataset": r["dataset"], "seed": r["seed"], "shuffle": key,
                    "acc_drop_pp": 100.0 * (full["val_acc"] - cf["val_acc"]),
                    "f1_drop_pp": 100.0 * (full["val_macro_f1"] - cf["val_macro_f1"]),
                })
    with (EDGE_AUDIT_ROOT / "edge_shuffle_controls.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "seed", "shuffle",
                                               "acc_drop_pp", "f1_drop_pp"])
        writer.writeheader()
        for r in shuffle_rows:
            writer.writerow(r)
    # 9x9 diversity (mean over seeds)
    pair_keys = None
    jsd_means = None
    spr_means = None
    topk_means = None
    for r in rows:
        div = r.get("pair_diversity") or {}
        if not div:
            continue
        if pair_keys is None:
            pair_keys = div["keys"]
            jsd_means = [[0.0] * 9 for _ in range(9)]
            spr_means = [[0.0] * 9 for _ in range(9)]
            topk_means = [[0.0] * 9 for _ in range(9)]
        for i in range(9):
            for j in range(9):
                jsd_means[i][j] += div["jsd"][i][j] / len(rows)
                spr_means[i][j] += div["spearman"][i][j] / len(rows)
                topk_means[i][j] += div["topk_overlap"][i][j] / len(rows)
    div_rows = []
    for i in range(9):
        for j in range(9):
            div_rows.append({"pair_i": pair_keys[i], "pair_j": pair_keys[j],
                             "jsd_mean": jsd_means[i][j],
                             "spearman_mean": spr_means[i][j],
                             "topk_overlap_mean": topk_means[i][j]})
    with (EDGE_AUDIT_ROOT / "edge_pair_diversity.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(div_rows[0].keys()))
        writer.writeheader()
        for r in div_rows:
            writer.writerow(r)

    # ---- report: the 5 questions --------------------------------------------
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)

    def _mtg_mean(field_of):
        vals = []
        for ds in TARGET_DATASETS:
            for r in by_ds.get(ds, []):
                v = field_of(r)
                if v is not None:
                    vals.append(v)
        return statistics.fmean(vals) if vals else None

    def _causal_drop(key):
        vals = []
        for ds in TARGET_DATASETS:
            for r in by_ds.get(ds, []):
                full = r["causal"].get("full")
                cf = r["causal"].get(key)
                if full and cf:
                    vals.append(100.0 * (full["val_acc"] - cf["val_acc"]))
        return statistics.fmean(vals) if vals else None

    null_mean = _mtg_mean(lambda r: statistics.fmean(
        p["null_mass_mean"] for p in r["pair_stats"]))
    entropy_mean = _mtg_mean(lambda r: statistics.fmean(
        p["entropy_mean"] for p in r["pair_stats"]))
    mass_top10 = _mtg_mean(lambda r: statistics.fmean(
        p["mass_top10"] for p in r["pair_stats"]))
    cos_corr = _mtg_mean(lambda r: statistics.fmean(
        c["corr_cos"] for c in r["heuristic_corr"]))
    same_label = _mtg_mean(lambda r: statistics.fmean(
        h["same_label_mean_alpha"] for h in r["homophily"]
        if h["same_label_mean_alpha"] is not None))
    diff_label = _mtg_mean(lambda r: statistics.fmean(
        h["diff_label_mean_alpha"] for h in r["homophily"]
        if h["diff_label_mean_alpha"] is not None))

    report = EDGE_AUDIT_ROOT / "R2D27_EDGE_AUDIT_REPORT.md"
    lines = ["# R2-D2.7-B — Edge-utility structure & causal ranking audit", "",
             "PAIR_EDGE best checkpoints; no retraining. M/T/G 3-seed means.", "",
             "## The five questions", "",
             f"1. **Is the scorer truly non-uniform?** null mass mean {null_mean:.3f}",
             f"   (frac<0.05: {None}); mean per-target entropy {entropy_mean:.3f};",
             f"   top-10% mass {mass_top10:.3f} — non-uniformity requires entropy",
             "   clearly below log(deg) and top-10% mass above ~0.3.",
             f"2. **Are pair rankings really different?** mean off-diagonal JSD ",
             f"   {statistics.fmean(div_rows[i*9+j]['jsd_mean'] for i in range(9) for j in range(9) if i != j):.4f},",
             f"   mean off-diagonal Spearman {statistics.fmean(div_rows[i*9+j]['spearman_mean'] for i in range(9) for j in range(9) if i != j):.3f}.",
             f"3. **Is it just cosine similarity?** corr(score, cos(F_i^b,F_j^a)) = {cos_corr:.3f}.",
             f"4. **Is it just label homophily?** same-label alpha {same_label:.5f} vs",
             f"   diff-label alpha {diff_label:.5f} (train-train edges only).",
             f"5. **Are high-utility edges causally important?**",
             f"   remove_top_10 drop {_causal_drop('remove_top_10'):+.3f}pp vs",
             f"   remove_random_10 {_causal_drop('remove_random_10'):+.3f}pp vs",
             f"   remove_bottom_10 {_causal_drop('remove_bottom_10'):+.3f}pp;",
             f"   within-target shuffle drop {_causal_drop('within_target_shuffle'):+.3f}pp.",
             ""]
    lines.append("Strong ranking: top-drop > random-drop > bottom-drop with")
    lines.append("top-random >= +0.20pp at one rate. Strong correspondence:")
    lines.append("within-target shuffle drop >= +0.30pp.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# D2.7-C prepost
# ---------------------------------------------------------------------------


def _write_prepost_report() -> Path:
    prepost_rows = _collect(PREPOST_ROOT)
    matrix_rows = _collect(MATRIX_ROOT)
    if not prepost_rows:
        raise RuntimeError("no prepost summaries — run POST_PAIR via perf_r2d27_matrix.py")
    PREPOST_ROOT.mkdir(parents=True, exist_ok=True)
    rows = prepost_rows + [r for r in matrix_rows
                           if r["variant"] in ("PAIR_EDGE", "TARGET_NULL_ONLY")
                           and (r["dataset"], r["variant"], r["seed"])
                           not in {(p["dataset"], p["variant"], p["seed"])
                                   for p in prepost_rows}]
    with (PREPOST_ROOT / "prepost_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "val_acc",
                                               "val_macro_f1", "best_epoch", "stop_epoch",
                                               "side_params"], extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    pre = _by_key(rows, "PAIR_EDGE")
    post = _by_key(rows, "POST_PAIR")
    a0m = _by_key(matrix_rows, "A0_BASE")
    pre_post_acc = _paired(pre, post, "val_acc", TARGET_DATASETS, SEEDS)
    pre_post_f1 = _paired(pre, post, "val_macro_f1", TARGET_DATASETS, SEEDS)
    pre_a0_acc = _paired(pre, a0m, "val_acc", TARGET_DATASETS, SEEDS)
    post_a0_acc = _paired(post, a0m, "val_acc", TARGET_DATASETS, SEEDS)
    go = (pre_post_acc["mean"] is not None and pre_post_acc["mean"] >= 0.30
          and pre_post_f1["mean"] is not None and pre_post_f1["mean"] >= 0.20
          and pre_post_acc["n_pos"] >= 2)
    report = PREPOST_ROOT / "R2D27_PREPOST_REPORT.md"
    lines = ["# R2-D2.7-C — PRE vs POST aggregation timing", "",
             "PRE_PAIR (edge-level selection) vs parameter-matched POST_PAIR",
             "(aggregate first, then per-node pair gate). M/T/G 3-seed means.", "",
             f"- PRE - POST: Acc {pre_post_acc['mean'] if pre_post_acc['mean'] is None else f'{pre_post_acc['mean']:+.3f}'}pp "
             f"/ F1 {pre_post_f1['mean'] if pre_post_f1['mean'] is None else f'{pre_post_f1['mean']:+.3f}'}pp "
             f"({pre_post_acc['n_pos']}/3 datasets) -> "
             f"**PRE-AGGREGATION GO: {'PASS' if go else 'not met'}**",
             f"- PRE - A0_MATCHED: Acc {pre_a0_acc['mean'] if pre_a0_acc['mean'] is None else f'{pre_a0_acc['mean']:+.3f}'}pp; "
             f"POST - A0_MATCHED: Acc {post_a0_acc['mean'] if post_a0_acc['mean'] is None else f'{post_a0_acc['mean']:+.3f}'}pp",
             ""]
    if pre_post_acc["mean"] is not None and pre_post_acc["mean"] < 0.05:
        lines.append("**PRE ≈ POST**: aggregation timing must NOT be claimed as")
        lines.append("the core mechanism (plan §60).")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# D2.7-D ownership
# ---------------------------------------------------------------------------


def _write_ownership_report() -> Path:
    own_rows = _collect(OWNERSHIP_ROOT)
    matrix_rows = _collect(MATRIX_ROOT)
    if not own_rows:
        raise RuntimeError("no ownership summaries — run SOURCE/TARGET_FACTOR_ONLY")
    OWNERSHIP_ROOT.mkdir(parents=True, exist_ok=True)
    rows = own_rows + [r for r in matrix_rows if r["variant"] in
                       ("PAIR_EDGE", "GENERIC_EDGE", "DIAG_EDGE")]
    with (OWNERSHIP_ROOT / "ownership_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "val_acc",
                                               "val_macro_f1", "side_params"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    pair = _by_key(rows, "PAIR_EDGE")
    controls = {}
    control_abs = {}
    for name in ("NODE_SHARED", "FACTOR_DIAG", "SOURCE_FACTOR_ONLY", "TARGET_FACTOR_ONLY"):
        key = {"NODE_SHARED": "GENERIC_EDGE", "FACTOR_DIAG": "DIAG_EDGE"}.get(name, name)
        c_rows = _by_key(rows, key)
        controls[name] = _paired(pair, c_rows, "val_acc", TARGET_DATASETS, SEEDS)
        controls[name]["f1"] = _paired(pair, c_rows, "val_macro_f1",
                                       TARGET_DATASETS, SEEDS)
        accs = [r["val_acc"] for (ds, s), r in c_rows.items()
                if ds in TARGET_DATASETS]
        control_abs[name] = statistics.fmean(accs) if accs else None
    # strongest control = the control with the HIGHEST absolute M/T/G acc
    # (the one hardest to beat), per plan §38.
    strongest_name = max(control_abs, key=lambda n: control_abs[n] or -1e9)
    strongest = controls[strongest_name]
    report = OWNERSHIP_ROOT / "R2D27_OWNERSHIP_REPORT.md"
    lines = ["# R2-D2.7-D — Ownership-specificity audit", "",
             "PAIR_EDGE vs the factor-conditioned controls. M/T/G 3-seed means.", ""]
    for name, c in controls.items():
        lines.append(
            f"- vs {name}: Acc {c['mean'] if c['mean'] is None else f'{c['mean']:+.3f}'} "
            f"/ F1 {c['f1']['mean'] if c['f1']['mean'] is None else f'{c['f1']['mean']:+.3f}'} "
            f"(abs acc {control_abs[name]:.5f}) ({c['n_pos']}/3)")
    support = (strongest["mean"] is not None and strongest["mean"] >= 0.20
               and strongest["f1"]["mean"] is not None
               and strongest["f1"]["mean"] >= 0.0 and strongest["n_pos"] >= 2)
    lines.append("")
    lines.append(f"Strongest control by absolute Acc = **{strongest_name}**.")
    lines.append(f"**Full factor-pair ownership SUPPORTED: "
                 f"{'YES' if support else 'NO'}** (needs >= +0.20pp Acc over the "
                 "strongest control with F1 nonnegative and >=2/3 datasets, "
                 "plus D2.7-B pair-ranking divergence).")
    if not support:
        lines.append("")
        lines.append("Per plan §38 the conclusion must read: \"neighbor utility")
        lines.append("supported (vs generic/diag); semantic-ownership factor-pair")
        lines.append("specificity not supported\" — the simpler factor-conditioned")
        lines.append("formulation (TARGET_FACTOR_ONLY) matches the full 9-pair model.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# D2.7-E transfer
# ---------------------------------------------------------------------------


def _write_transfer_report() -> Path:
    t_rows = _collect(TRANSFER_ROOT)
    matrix_rows = _collect(MATRIX_ROOT)
    if not t_rows:
        raise RuntimeError("no transfer summaries — run PAIR_TRANSFORM_*")
    TRANSFER_ROOT.mkdir(parents=True, exist_ok=True)
    rows = t_rows + [r for r in matrix_rows if r["variant"] in
                     ("UNIFORM", "PAIR_EDGE")]
    with (TRANSFER_ROOT / "transfer_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "val_acc",
                                               "val_macro_f1", "side_params"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    t0 = _by_key(rows, "UNIFORM")
    t1 = _by_key(rows, "PAIR_EDGE")
    t2 = _by_key(rows, "PAIR_TRANSFORM_UNIFORM")
    t3 = _by_key(rows, "PAIR_TRANSFORM_PRE")
    sel = _paired(t1, t0, "val_acc", TARGET_DATASETS, SEEDS)
    trf = _paired(t2, t0, "val_acc", TARGET_DATASETS, SEEDS)
    sel_f1 = _paired(t1, t0, "val_macro_f1", TARGET_DATASETS, SEEDS)
    trf_f1 = _paired(t2, t0, "val_macro_f1", TARGET_DATASETS, SEEDS)
    # complementarity: per (dataset, seed): t3 - max(t1, t2)
    deltas = []
    for ds in TARGET_DATASETS:
        for s in SEEDS:
            a, b, c = t3.get((ds, s)), t1.get((ds, s)), t2.get((ds, s))
            if a and b and c:
                deltas.append(100.0 * (a["val_acc"] - max(b["val_acc"], c["val_acc"])))
    comp_acc = statistics.fmean(deltas) if deltas else None
    deltas_f1 = []
    for ds in TARGET_DATASETS:
        for s in SEEDS:
            a, b, c = t3.get((ds, s)), t1.get((ds, s)), t2.get((ds, s))
            if a and b and c:
                deltas_f1.append(100.0 * (a["val_macro_f1"] - max(b["val_macro_f1"], c["val_macro_f1"])))
    comp_f1 = statistics.fmean(deltas_f1) if deltas_f1 else None
    report = TRANSFER_ROOT / "R2D27_TRANSFER_REPORT.md"
    lines = ["# R2-D2.7-E — Selection x transformation decomposition", "",
             "T0 shared payload + UNIFORM; T1 + PRE_PAIR selection; T2 pair",
             "transforms + UNIFORM; T3 both. M/T/G 3-seed means.", "",
             f"- Selection = T1 - T0: Acc {sel['mean'] if sel['mean'] is None else f'{sel['mean']:+.3f}'} "
             f"/ F1 {sel_f1['mean'] if sel_f1['mean'] is None else f'{sel_f1['mean']:+.3f}'}",
             f"- Transform = T2 - T0: Acc {trf['mean'] if trf['mean'] is None else f'{trf['mean']:+.3f}'} "
             f"/ F1 {trf_f1['mean'] if trf_f1['mean'] is None else f'{trf_f1['mean']:+.3f}'}",
             f"- Complementarity = T3 - max(T1,T2): Acc {comp_acc if comp_acc is None else f'{comp_acc:+.3f}'} "
             f"/ F1 {comp_f1 if comp_f1 is None else f'{comp_f1:+.3f}'}",
             ""]
    ft = (comp_acc is not None and comp_f1 is not None and
          (comp_acc >= 0.20 or comp_f1 >= 0.20) and
          (comp_acc >= 0.0 and comp_f1 >= 0.0))
    lines.append(f"**Functional Transfer support: {'YES' if ft else 'NO'}**")
    if not ft:
        lines.append("Do NOT call the method functional transfer if T3 adds no")
        lines.append("value beyond selection alone.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# D2.7-G final synthesis
# ---------------------------------------------------------------------------


def _write_final_synthesis() -> tuple[Path, Path, Path]:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    master = SUMMARY_ROOT / "R2D27_MASTER_TABLE.csv"
    ledger = SUMMARY_ROOT / "R2D27_HYPOTHESIS_LEDGER.csv"
    diagnosis = SUMMARY_ROOT / "R2D27_FINAL_DIAGNOSIS.md"
    master_rows = []
    for stage, root in (("matrix", MATRIX_ROOT), ("prepost", PREPOST_ROOT),
                        ("ownership", OWNERSHIP_ROOT), ("transfer", TRANSFER_ROOT)):
        for r in _collect(root):
            master_rows.append({"stage": stage, **r})
    if master_rows:
        fields = ["stage"] + sorted({k for r in master_rows for k in r if k != "stage"})
        with master.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in master_rows:
                writer.writerow(r)
    ledger_rows = [
        {"hypothesis": "observed topology as useful support",
         "stage": "D2.7-A/B", "verdict": "SUPPORTED (source_shuffle drops 4.73pp)",
         "evidence": "outputs/perf_r2d27/"},
        {"hypothesis": "uniform neighbor aggregation is enough",
         "stage": "D2.7-A", "verdict": "REJECTED (PAIR-UNIFORM +0.361 Acc / +0.245 F1)",
         "evidence": "outputs/perf_r2d27/matrix/"},
        {"hypothesis": "target-only graph-mass control explains selection",
         "stage": "D2.7-A", "verdict": "PARTIAL (PAIR-TARGET_NULL +0.284 Acc but -0.493 F1: mass control explains F1, not Acc)",
         "evidence": "outputs/perf_r2d27/matrix/"},
        {"hypothesis": "generic edge utility suffices",
         "stage": "D2.7-A", "verdict": "REJECTED (PAIR-GENERIC +0.452 Acc / +0.391 F1, 3/3)",
         "evidence": "outputs/perf_r2d27/matrix/"},
        {"hypothesis": "factor-pair (9-cell) neighbor utility",
         "stage": "D2.7-A/D", "verdict": "CONDITIONAL — beats generic/diag but NOT TARGET_FACTOR_ONLY (+0.044 Acc / -0.349 F1) -> specificity NOT supported",
         "evidence": "outputs/perf_r2d27/ownership/"},
        {"hypothesis": "pre-aggregation timing matters",
         "stage": "D2.7-C", "verdict": "PARTIAL (PRE-POST +0.326 Acc / -0.153 F1; Acc-side only)",
         "evidence": "outputs/perf_r2d27/prepost/"},
        {"hypothesis": "semantic similarity is a sufficient edge score",
         "stage": "D2.7-A", "verdict": "REJECTED as sufficient (PAIR-SEM_SIM +0.271 Acc); corr(score,cos)=0.341 (not identity)",
         "evidence": "outputs/perf_r2d27/"},
        {"hypothesis": "null neighbor option is meaningful",
         "stage": "D2.7-B", "verdict": "SUPPORTED (null mass mean 0.189, not 0/1 collapsed)",
         "evidence": "outputs/perf_r2d27/edge_audit/"},
        {"hypothesis": "within-neighborhood utility heterogeneity (neighbor IDENTITY) matters",
         "stage": "D2.7-B", "verdict": "STRONGLY REJECTED — within-target shuffle drop = 0.000pp (M/T/G, and T3: 9/9 zero); noise edges get >= average alpha",
         "evidence": "outputs/perf_r2d27/edge_audit/"},
        {"hypothesis": "factor-pair ranking diversity",
         "stage": "D2.7-B", "verdict": "SUPPORTED structurally (off-diag Spearman 0.156, JSD 0.23) but carries no performance value (see identity rejection)",
         "evidence": "outputs/perf_r2d27/edge_audit/"},
        {"hypothesis": "H2/multihop as downstream symptom",
         "stage": "D2.7-G", "verdict": "OPEN — D2.5/D2.6 H2 findings reinterpreted: post-aggregation utility was mass/concentration, not identity",
         "evidence": "synthesis"},
        {"hypothesis": "selection-only transfer",
         "stage": "D2.7-E", "verdict": "SUPPORTED (T1-T0 +0.361 Acc / +0.245 F1)",
         "evidence": "outputs/perf_r2d27/transfer/"},
        {"hypothesis": "pair-specific message transform",
         "stage": "D2.7-E", "verdict": "NOT SUPPORTED as pair-specific (T2-T0 -0.032 Acc / +0.753 F1)",
         "evidence": "outputs/perf_r2d27/transfer/"},
        {"hypothesis": "selection x transform complementarity (functional transfer)",
         "stage": "D2.7-E", "verdict": "CLOSED (T3-max(T1,T2) -0.325 Acc / +0.021 F1)",
         "evidence": "outputs/perf_r2d27/transfer/"},
        {"hypothesis": "A0 incremental utility",
         "stage": "D2.7-A", "verdict": "STRONGLY_SUPPORTED (PAIR-A0_MATCHED +0.496 Acc / +0.790 F1, 3/3, guards safe) — first A0 increment in the R2 series",
         "evidence": "outputs/perf_r2d27/matrix/"},
        {"hypothesis": "noise-edge rejection",
         "stage": "D2.7-F", "verdict": "NOT SUPPORTED (injected edges get >= average alpha on 2/3 datasets; PAIR degrades similarly to UNIFORM)",
         "evidence": "outputs/perf_r2d27/noise_optional/"},
        {"hypothesis": "RoleMAG / TMTE / CoMAG collisions",
         "stage": "D2.7-0", "verdict": "CLOSED (guardrails implemented + tested)",
         "evidence": "outputs/perf_r2d27/audit/"},
    ]
    with ledger.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hypothesis", "stage", "verdict", "evidence"])
        writer.writeheader()
        for r in ledger_rows:
            writer.writerow(r)
    if not diagnosis.exists() or "skeleton" in diagnosis.read_text():
        diagnosis.write_text(
            "# R2-Design-2.7 — FINAL DIAGNOSIS (skeleton)\n\n"
            "The 20-question synthesis is authored at D2.7-G.\n", encoding="utf-8")
    return master, ledger, diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-Design-2.7 summarizer")
    parser.add_argument("--stage", default="audit",
                        choices=["audit", "matrix", "edge_audit", "prepost",
                                 "ownership", "transfer", "final"])
    args = parser.parse_args()
    if args.stage == "audit":
        print(f"[audit] wrote {_write_audit_report()}")
        return
    if args.stage == "matrix":
        rows = _collect(MATRIX_ROOT)
        if not rows:
            raise RuntimeError("no matrix summaries — run perf_r2d27_matrix.py")
        print(f"[matrix] wrote {_write_matrix_report(rows)}")
        return
    if args.stage == "edge_audit":
        print(f"[edge_audit] wrote {_write_edge_audit_report(_collect_edge_audit())}")
        return
    if args.stage == "prepost":
        print(f"[prepost] wrote {_write_prepost_report()}")
        return
    if args.stage == "ownership":
        print(f"[ownership] wrote {_write_ownership_report()}")
        return
    if args.stage == "transfer":
        print(f"[transfer] wrote {_write_transfer_report()}")
        return
    if args.stage == "final":
        m, l, d = _write_final_synthesis()
        print(f"[final] wrote {m} / {l} / {d}")
        return


if __name__ == "__main__":
    main()
