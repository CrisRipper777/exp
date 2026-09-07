"""R2-Design-2.6 summarizer
(docs/BiAxis_R2_Design_2_6_Strong_Parent_Readout_Integration.md).

Stages:
    audit           : D2.6-0 infrastructure audit (R2D26_AUDIT.md)
    no_compression  : D2.6-A CSVs + R2D26_NO_COMPRESSION_REPORT.md
    integration     : D2.6-B CSVs + R2D26_INTEGRATION_REPORT.md
    causal          : D2.6-C CSVs + R2D26_CAUSAL_REPORT.md
    parent_adapt    : D2.6-D CSVs + R2D26_PARENT_ADAPT_REPORT.md
    deep_supervision: D2.6-E CSVs + R2D26_DEEP_SUP_REPORT.md
    final           : master table + hypothesis ledger + diagnosis

Usage:
    python scripts/summarize_perf_r2d26.py --stage audit
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

from src.analysis.perf_r2d26_utils import (  # noqa: E402
    DATASETS,
    GUARD_DATASETS,
    R2D26_ROOT,
    SEEDS,
    TARGET_DATASETS,
    VARIANTS,
)

AUDIT_ROOT = R2D26_ROOT / "audit"
NC_ROOT = R2D26_ROOT / "no_compression"
INTEGRATION_ROOT = R2D26_ROOT / "integration"
CAUSAL_ROOT = R2D26_ROOT / "causal_usage"
PARENT_ADAPT_ROOT = R2D26_ROOT / "parent_adapt"
DEEP_SUP_ROOT = R2D26_ROOT / "deep_supervision"
SUMMARY_ROOT = R2D26_ROOT / "summary"


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def _collect(root: Path, extra: dict | None = None) -> list[dict]:
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
                row = {
                    "dataset": ds_dir.name,
                    "variant": d.get("variant", v_dir.name),
                    "seed": int(s_dir.name.split("_")[-1]),
                    "val_acc": d["best_val_acc"],
                    "val_macro_f1": d["best_val_macro_f1"],
                    "best_epoch": d.get("best_epoch"),
                    "stop_epoch": d.get("stop_epoch"),
                    "side_params": d.get("side_params"),
                    "parent_params": d.get("parent_params"),
                    "out_dim": d.get("out_dim"),
                }
                if extra:
                    row.update(extra)
                rows.append(row)
    return rows


def _load_a0_formal() -> dict[tuple[str, int], dict]:
    """Formal A0 per-seed val acc (nc_main_per_seed.csv) + R1-baseline F1."""
    from src.analysis.perf_r2_utils import load_a0_reference

    acc = load_a0_reference()
    out = {}
    for (ds, seed), v in acc.items():
        out[(ds, seed)] = {"val_acc": v, "val_macro_f1": None}
    for ds in DATASETS:
        for seed in SEEDS:
            p = PROJECT_ROOT / "outputs" / "perf_r1" / "baseline" / ds / "A0" / f"seed_{seed}" / "summary.json"
            if p.exists():
                d = json.loads(p.read_text())
                if d.get("best_val_macro_f1") is not None:
                    out[(ds, seed)]["val_macro_f1"] = float(d["best_val_macro_f1"]) / 100.0
    return out


A0_FORMAL = _load_a0_formal()


def _paired_delta(cand_rows: dict, base_rows: dict, metric: str,
                  ds_list: list[str], seeds: list[int]) -> dict:
    """Mean over datasets of per-(dataset,seed)-paired deltas."""
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


def _by_key(rows: list[dict], variant: str) -> dict[tuple[str, int], dict]:
    return {(r["dataset"], r["seed"]): r for r in rows if r["variant"] == variant}


def _a0_matched_rows(nc_rows: list[dict]) -> dict[tuple[str, int], dict]:
    return _by_key(nc_rows, "A0_BASE")


# ---------------------------------------------------------------------------
# D2.6-A no-compression
# ---------------------------------------------------------------------------


def _write_no_compression_report(rows: list[dict]) -> Path:
    NC_ROOT.mkdir(parents=True, exist_ok=True)
    with (NC_ROOT / "no_compression_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "val_acc",
                                               "val_macro_f1", "best_epoch", "stop_epoch",
                                               "side_params", "parent_params", "out_dim"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    nc_hop = _by_key(rows, "NC_HOP")
    nc_h1 = _by_key(rows, "NC_H1")
    a0_matched = _by_key(rows, "A0_BASE")
    content_acc = _paired_delta(nc_hop, nc_h1, "val_acc", TARGET_DATASETS, SEEDS)
    content_f1 = _paired_delta(nc_hop, nc_h1, "val_macro_f1", TARGET_DATASETS, SEEDS)
    parent_acc = _paired_delta(nc_hop, a0_matched, "val_acc", TARGET_DATASETS, SEEDS)
    parent_f1 = _paired_delta(nc_hop, a0_matched, "val_macro_f1", TARGET_DATASETS, SEEDS)
    # formal A0 (acc)
    formal_acc = {}
    for ds in DATASETS:
        deltas = []
        for s in SEEDS:
            c = nc_hop.get((ds, s))
            a = A0_FORMAL.get((ds, s))
            if c and a:
                deltas.append(100.0 * (c["val_acc"] - a["val_acc"]))
        if deltas:
            formal_acc[ds] = statistics.fmean(deltas)

    content_go = (content_acc["mean"] is not None and content_acc["mean"] >= 0.30
                  and content_f1["mean"] is not None and content_f1["mean"] >= 0.20
                  and content_acc["n_pos"] >= 2)
    report = NC_ROOT / "R2D26_NO_COMPRESSION_REPORT.md"
    lines = [
        "# R2-D2.6-A — No-compression strong-parent diagnosis", "",
        "NC_HOP = [z_base | 9 H0/H1/H2 expert tokens], no projection back to 256;",
        "NC_H1 = architecture-identical H1-only control. A0 frozen; deep sup 0.1.",
        "",
        f"- **NC_HOP - NC_H1** (M/T/G): Acc {content_acc['mean'] if content_acc['mean'] is None else f'{content_acc['mean']:+.3f}'}pp "
        f"/ F1 {content_f1['mean'] if content_f1['mean'] is None else f'{content_f1['mean']:+.3f}'}pp "
        f"({content_acc['n_pos']}/3 datasets positive) -> "
        f"**CONTENT {'SUPPORTED' if content_go else 'NOT SUPPORTED'}**",
        f"- **NC_HOP - A0_MATCHED** (M/T/G): Acc {parent_acc['mean'] if parent_acc['mean'] is None else f'{parent_acc['mean']:+.3f}'}pp "
        f"/ F1 {parent_f1['mean'] if parent_f1['mean'] is None else f'{parent_f1['mean']:+.3f}'}pp",
        f"- **NC_HOP - A0_FORMAL** (acc): "
        + (", ".join(f"{ds} {v:+.3f}pp" for ds, v in formal_acc.items()) if formal_acc else "no data"),
        "",
    ]
    lines.append("| dataset | variant | acc (3-seed mean) | F1 (3-seed mean) |")
    lines.append("|---|---|---|---|")
    for v in ("A0_BASE", "NC_HOP", "NC_H1"):
        for ds in DATASETS:
            sel = [r for r in rows if r["variant"] == v and r["dataset"] == ds]
            if sel:
                lines.append(
                    f"| {ds} | {v} | {statistics.fmean(r['val_acc'] for r in sel):.5f} | "
                    f"{statistics.fmean(r['val_macro_f1'] for r in sel):.5f} |")
    lines.append("")
    if content_go and parent_acc["mean"] and parent_acc["mean"] >= 0.30 and parent_acc["n_pos"] >= 2:
        lines.append("**Multi-hop content survives when compression is removed** —")
        lines.append("formally supported (CONTENT + integration vs matched parent).")
    elif content_go:
        lines.append("CONTENT SUPPORTED but integration vs A0_MATCHED below +0.30pp:")
        lines.append("content exists yet does not beat the strong parent.")
    else:
        lines.append("CONTENT NOT SUPPORTED at the pre-registered threshold; if NC_HOP")
        lines.append("and NC_H1 both improved similarly, the gain is GENERIC EXPANDED")
        lines.append("READOUT CAPACITY, not multi-hop.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# D2.6-B integration
# ---------------------------------------------------------------------------


def _write_integration_report(rows: list[dict], base_rows: list[dict]) -> Path:
    INTEGRATION_ROOT.mkdir(parents=True, exist_ok=True)
    with (INTEGRATION_ROOT / "integration_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "val_acc",
                                               "val_macro_f1", "best_epoch", "stop_epoch",
                                               "side_params", "parent_params", "out_dim"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    hop_variants = ("FHC_HOP", "RSF_HOP", "HIER_HOP")
    a0_matched = _by_key(base_rows, "A0_BASE")
    readout_only = _by_key(rows, "READOUT_ONLY")

    with (INTEGRATION_ROOT / "integration_controls.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "control", "delta_acc_mtg",
                                               "delta_f1_mtg", "n_pos_ds"],
                                extrasaction="ignore")
        writer.writeheader()
        control_rows = []
        for v in hop_variants:
            h1_name = v.replace("_HOP", "_H1")
            cand = _by_key(rows, v)
            ctrl = _by_key(rows, h1_name)
            da = _paired_delta(cand, ctrl, "val_acc", TARGET_DATASETS, SEEDS)
            df = _paired_delta(cand, ctrl, "val_macro_f1", TARGET_DATASETS, SEEDS)
            control_rows.append({"variant": v, "control": h1_name,
                                 "delta_acc_mtg": da["mean"], "delta_f1_mtg": df["mean"],
                                 "n_pos_ds": da["n_pos"]})
            writer.writerow(control_rows[-1])
    with (INTEGRATION_ROOT / "integration_resources.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "side_params",
                                               "parent_params", "out_dim", "runtime_sec",
                                               "peak_allocated_mb"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "runtime_sec": r.get("runtime_sec"),
                             "peak_allocated_mb": r.get("peak_allocated_mb")})

    report = INTEGRATION_ROOT / "R2D26_INTEGRATION_REPORT.md"
    lines = ["# R2-D2.6-B — Strong-parent integration matrix", "",
             "All variants retain z_base as a direct path; A0 fully frozen;",
             "side/aux/head lr 1e-3, warmup10+cosine, 300/patience30; deep sup 0.1.",
             ""]
    lines.append("| variant | vs H1-control Acc (pp) | vs H1-control F1 (pp) | vs A0_MATCHED Acc | vs A0_MATCHED F1 | CONTENT | INTEGRATION |")
    lines.append("|---|---|---|---|---|---|---|")
    for v in hop_variants:
        cand = _by_key(rows, v)
        ctrl = _by_key(rows, v.replace("_HOP", "_H1"))
        da = _paired_delta(cand, ctrl, "val_acc", TARGET_DATASETS, SEEDS)
        df = _paired_delta(cand, ctrl, "val_macro_f1", TARGET_DATASETS, SEEDS)
        pa = _paired_delta(cand, a0_matched, "val_acc", TARGET_DATASETS, SEEDS)
        pf = _paired_delta(cand, a0_matched, "val_macro_f1", TARGET_DATASETS, SEEDS)
        content_go = (da["mean"] is not None and da["mean"] >= 0.30
                      and df["mean"] is not None and df["mean"] >= 0.20
                      and da["n_pos"] >= 2)
        integ_go = (pa["mean"] is not None and pa["mean"] >= 0.30
                    and pf["mean"] is not None and pf["mean"] >= 0.20
                    and pa["n_pos"] >= 2)
        lines.append(
            f"| {v} | {da['mean'] if da['mean'] is None else f'{da['mean']:+.3f}'} "
            f"| {df['mean'] if df['mean'] is None else f'{df['mean']:+.3f}'} "
            f"| {pa['mean'] if pa['mean'] is None else f'{pa['mean']:+.3f}'} "
            f"| {pf['mean'] if pf['mean'] is None else f'{pf['mean']:+.3f}'} "
            f"| {'GO' if content_go else 'no'} | {'GO' if integ_go else 'no'} |")
    lines.append("")
    # READOUT_ONLY vs A0_MATCHED: does generic readout depth alone explain gains?
    ra = _paired_delta(readout_only, a0_matched, "val_acc", TARGET_DATASETS, SEEDS)
    rf = _paired_delta(readout_only, a0_matched, "val_macro_f1", TARGET_DATASETS, SEEDS)
    lines.append(f"- READOUT_ONLY - A0_MATCHED (M/T/G): Acc {ra['mean'] if ra['mean'] is None else f'{ra['mean']:+.3f}'}pp "
                 f"/ F1 {rf['mean'] if rf['mean'] is None else f'{rf['mean']:+.3f}'}pp — "
                 "generic readout depth alone")
    # guards (from the matrix runs: HOP variants on ele/Reddit)
    lines.append("")
    lines.append("Guards (vs A0_MATCHED, Acc pp):")
    for v in hop_variants:
        cand = _by_key(rows, v)
        for g in GUARD_DATASETS:
            deltas = [100.0 * (cand[(g, s)]["val_acc"] - a0_matched[(g, s)]["val_acc"])
                      for s in SEEDS if (g, s) in cand and (g, s) in a0_matched]
            if deltas:
                lines.append(f"- {v} / {g}: {statistics.fmean(deltas):+.3f}pp "
                             f"(threshold -0.20pp)")
    lines.append("")
    lines.append("Verdict thresholds (plan §27-§29): CONTENT GO = Acc +0.30 / F1 +0.20 vs")
    lines.append("H1 control, >=2/3 datasets; INTEGRATION GO = Acc +0.30 / F1 +0.20 vs")
    lines.append("A0_MATCHED; FINAL GO = +0.20pp both vs A0_FORMAL with guard safety.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# D2.6-C causal usage
# ---------------------------------------------------------------------------


def _write_causal_report(rows: list[dict]) -> Path:
    CAUSAL_ROOT.mkdir(parents=True, exist_ok=True)
    causal_flat = []
    base_flat = []
    for r in rows:
        for key, m in r.get("causal", {}).items():
            causal_flat.append({
                "dataset": r["dataset"], "variant": r["variant"], "seed": r["seed"],
                "causal": key, "val_acc": m["val_acc"], "val_macro_f1": m["val_macro_f1"],
            })
        bp = r.get("diagnostics", {}).get("base_preservation", {})
        base_flat.append({
            "dataset": r["dataset"], "variant": r["variant"], "seed": r["seed"],
            "side_off_bitwise": bp.get("side_off_bitwise_equal_base"),
            "side_off_max_abs_diff": bp.get("side_off_max_abs_diff"),
            "side_off_reproduces": bp.get("side_off_reproduces_base"),
            "cka_final_base": bp.get("cka_final_base"),
            "mean_cosine": bp.get("mean_cosine"),
            "relative_l2": bp.get("relative_l2"),
            "side_base_norm_ratio": bp.get("side_base_norm_ratio"),
        })
    with (CAUSAL_ROOT / "causal_usage.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "causal",
                                               "val_acc", "val_macro_f1"])
        writer.writeheader()
        for r in causal_flat:
            writer.writerow(r)
    with (CAUSAL_ROOT / "base_preservation.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(base_flat[0].keys()) if base_flat else [])
        if base_flat:
            writer.writeheader()
            for r in base_flat:
                writer.writerow(r)

    report = CAUSAL_ROOT / "R2D26_CAUSAL_REPORT.md"
    lines = ["# R2-D2.6-C — Causal evidence-usage audit", "",
             "No retraining; best-checkpoint forwards with causal overrides.",
             "Strong H2-specific usage requires >= 2 of {H2_TO_H1, H2_SHUFFLE,",
             "PT_H2_OFF} >= +0.20pp on M/T/G macro.", ""]
    variants = sorted({r["variant"] for r in rows})
    for v in variants:
        vrows = {(r["dataset"], r["seed"]): r for r in rows if r["variant"] == v}
        lines.append(f"## {v}")
        lines.append("")
        lines.append("| causal | M/T/G acc drop (pp, 3-seed) | M/T/G F1 drop (pp) |")
        lines.append("|---|---|---|")
        for key in ("h2_zero", "h2_to_h1", "h2_shuffle", "pt_h2_off", "c_h2_off",
                    "pv_h2_off", "s_c_off", "s_pt_off", "s_pv_off"):
            drops_acc, drops_f1 = [], []
            for ds in TARGET_DATASETS:
                for s in SEEDS:
                    r = vrows.get((ds, s))
                    if not r:
                        continue
                    full = r["causal"].get("full")
                    cf = r["causal"].get(key)
                    if full and cf:
                        drops_acc.append(100.0 * (full["val_acc"] - cf["val_acc"]))
                        drops_f1.append(100.0 * (full["val_macro_f1"] - cf["val_macro_f1"]))
            if drops_acc:
                lines.append(f"| {key} | {statistics.fmean(drops_acc):+.3f} | "
                             f"{statistics.fmean(drops_f1):+.3f} |")
        # strong-usage rule
        rule = {}
        for key in ("h2_to_h1", "h2_shuffle", "pt_h2_off"):
            drops = []
            for ds in TARGET_DATASETS:
                for s in SEEDS:
                    r = vrows.get((ds, s))
                    if not r:
                        continue
                    full = r["causal"].get("full")
                    cf = r["causal"].get(key)
                    if full and cf:
                        drops.append(100.0 * (full["val_acc"] - cf["val_acc"]))
            rule[key] = statistics.fmean(drops) if drops else None
        n_pass = sum(1 for k in rule if rule[k] is not None and rule[k] >= 0.20)
        h2zero = None
        drops = []
        for ds in TARGET_DATASETS:
            for s in SEEDS:
                r = vrows.get((ds, s))
                if not r:
                    continue
                full = r["causal"].get("full")
                cf = r["causal"].get("h2_zero")
                if full and cf:
                    drops.append(100.0 * (full["val_acc"] - cf["val_acc"]))
        h2zero = statistics.fmean(drops) if drops else None
        verdict = "H2-SPECIFIC USAGE" if n_pass >= 2 else \
            ("BRANCH-PRESENCE DEPENDENCY" if h2zero is not None and h2zero >= 0.20
             else "NO H2 USAGE EVIDENCE")
        lines.append("")
        lines.append(f"- h2_zero {h2zero if h2zero is None else f'{h2zero:+.3f}'}pp; "
                     f"rule passes {n_pass}/3 -> **{verdict}**")
        lines.append("")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# D2.6-D parent adaptation
# ---------------------------------------------------------------------------


def _collect_parent_adapt() -> list[dict]:
    rows = []
    if not PARENT_ADAPT_ROOT.exists():
        return rows
    for ds_dir in sorted(PARENT_ADAPT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for v_dir in sorted(ds_dir.iterdir()):
            if not v_dir.is_dir():
                continue
            for sch_dir in sorted(v_dir.iterdir()):
                if not sch_dir.is_dir():
                    continue
                for s_dir in sorted(sch_dir.iterdir()):
                    sp = s_dir / "summary.json"
                    if sp.exists():
                        d = json.loads(sp.read_text())
                        rows.append({
                            "dataset": ds_dir.name, "variant": d["variant"],
                            "schedule": d["schedule"], "seed": d["seed"],
                            "val_acc": d["best_val_acc"],
                            "val_macro_f1": d["best_val_macro_f1"],
                            "best_epoch": d.get("best_epoch"),
                            "stop_epoch": d.get("stop_epoch"),
                            "parent_unfrozen": d.get("parent_unfrozen"),
                            "parent_drift": d.get("parent_drift"),
                        })
    return rows


def _write_parent_adapt_report(rows: list[dict]) -> Path:
    if not rows:
        raise RuntimeError("no parent_adapt summaries — run perf_r2d26_parent_adapt.py")
    PARENT_ADAPT_ROOT.mkdir(parents=True, exist_ok=True)
    with (PARENT_ADAPT_ROOT / "parent_adapt_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "schedule", "seed",
                                               "val_acc", "val_macro_f1", "best_epoch",
                                               "stop_epoch", "parent_unfrozen"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    drift_rows = []
    for r in rows:
        d = r.get("parent_drift") or {}
        drift_rows.append({
            "dataset": r["dataset"], "variant": r["variant"], "schedule": r["schedule"],
            "seed": r["seed"],
            "parent_z_cka": d.get("parent_z_cka"),
            "parent_z_cosine": d.get("parent_z_cosine"),
            "parent_z_rel_l2": d.get("parent_z_rel_l2"),
        })
    with (PARENT_ADAPT_ROOT / "parent_drift.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(drift_rows[0].keys()))
        writer.writeheader()
        for r in drift_rows:
            writer.writerow(r)

    # S0 = the D2.6-B integration run of the same variant (reused)
    base_rows = _collect(INTEGRATION_ROOT)
    report = PARENT_ADAPT_ROOT / "R2D26_PARENT_ADAPT_REPORT.md"
    lines = ["# R2-D2.6-D — Controlled parent adaptation", "",
             "S0 = the D2.6-B frozen run (reused); S1 = fusion only unfrozen at",
             "ep 31 (lr 1e-4); S2 = graph transform/readout blocks unfrozen",
             "(P0 factorizer frozen). M/T/G x 3 seeds. Parent eval-mode",
             "throughout (no parent dropout).", ""]
    variants = sorted({r["variant"] for r in rows})
    for v in variants:
        lines.append(f"## {v}")
        lines.append("")
        lines.append("| schedule | Movies | Toys | Grocery | M/T/G mean | ≥2/3 pos | drift CKA |")
        lines.append("|---|---|---|---|---|---|---|")
        for sch in ("S1", "S2"):
            ds_means, ckas = [], []
            for ds in TARGET_DATASETS:
                deltas = []
                for s in SEEDS:
                    cand = [r for r in rows if r["variant"] == v and r["schedule"] == sch
                            and r["dataset"] == ds and r["seed"] == s]
                    base = [r for r in base_rows if r["variant"] == v
                            and r["dataset"] == ds and r["seed"] == s]
                    if cand and base:
                        deltas.append(100.0 * (cand[0]["val_acc"] - base[0]["val_acc"]))
                if deltas:
                    ds_means.append(statistics.fmean(deltas))
                dr = [r["parent_drift"].get("parent_z_cka") for r in rows
                      if r["variant"] == v and r["schedule"] == sch
                      and r["dataset"] == ds and r.get("parent_drift")]
                if dr:
                    ckas.extend(dr)
            mean = statistics.fmean(ds_means) if ds_means else float("nan")
            n_pos = sum(1 for d in ds_means if d > 0) if ds_means else 0
            cka = statistics.fmean(ckas) if ckas else float("nan")
            lines.append(f"| {sch} | {ds_means[0] if len(ds_means) > 0 else float('nan'):+.2f} "
                         f"| {ds_means[1] if len(ds_means) > 1 else float('nan'):+.2f} "
                         f"| {ds_means[2] if len(ds_means) > 2 else float('nan'):+.2f} "
                         f"| {mean:+.2f} | {n_pos}/3 | {cka:.4f} |")
        lines.append("")
        lines.append("S3 (P0 lr 1e-5 after ep 60) is only allowed if S2 > S0 by")
        lines.append(">= +0.20pp AND ownership stays healthy — never automatic.")
        lines.append("")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# D2.6-E deep supervision
# ---------------------------------------------------------------------------


def _write_deep_sup_report() -> Path:
    lam0_rows = _collect(DEEP_SUP_ROOT)
    if not lam0_rows:
        raise RuntimeError("no deep_supervision summaries — run perf_r2d26_deepsup.py")
    lam01_rows = [r for r in _collect(INTEGRATION_ROOT)
                  if r["variant"] in {r2["variant"] for r2 in lam0_rows}]
    DEEP_SUP_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = lam0_rows + [dict(r, deep_sup_lambda=0.1) for r in lam01_rows]
    for r in lam0_rows:
        r["deep_sup_lambda"] = 0.0
    with (DEEP_SUP_ROOT / "deep_supervision_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed",
                                               "deep_sup_lambda", "val_acc",
                                               "val_macro_f1", "best_epoch", "stop_epoch"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    report = DEEP_SUP_ROOT / "R2D26_DEEP_SUP_REPORT.md"
    lines = ["# R2-D2.6-E — Deep-supervision confirmation", "",
             "lambda_aux = 0 vs 0.1 for the final top-1 candidate, all other",
             "init/schedule identical. Retain deep supervision only if the",
             "macro gain is >= +0.20pp on Acc or F1 with no safety harm.", ""]
    variants = sorted({r["variant"] for r in lam0_rows})
    for v in variants:
        lam0 = _by_key(lam0_rows, v)
        lam1 = _by_key(lam01_rows, v)
        for metric in ("val_acc", "val_macro_f1"):
            deltas = []
            for ds in TARGET_DATASETS:
                ds_d = []
                for s in SEEDS:
                    if (ds, s) in lam0 and (ds, s) in lam1:
                        ds_d.append(100.0 * (lam1[(ds, s)][metric] - lam0[(ds, s)][metric]))
                if ds_d:
                    deltas.append((ds, statistics.fmean(ds_d)))
            mean = statistics.fmean([m for _, m in deltas]) if deltas else None
            lines.append(
                f"- {v} {metric}: lambda 0.1 - 0 = "
                f"{mean if mean is None else f'{mean:+.3f}'}pp "
                f"({', '.join(f'{ds} {m:+.2f}' for ds, m in deltas)})")
        lines.append("")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# D2.6-G final synthesis
# ---------------------------------------------------------------------------


def _write_final_synthesis() -> tuple[Path, Path, Path]:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    master = SUMMARY_ROOT / "R2D26_MASTER_TABLE.csv"
    ledger = SUMMARY_ROOT / "R2D26_HYPOTHESIS_LEDGER.csv"
    diagnosis = SUMMARY_ROOT / "R2D26_FINAL_DIAGNOSIS.md"

    master_rows = []
    for stage, root in (("no_compression", NC_ROOT), ("integration", INTEGRATION_ROOT),
                        ("deep_supervision", DEEP_SUP_ROOT)):
        for r in _collect(root):
            master_rows.append({"stage": stage, **r})
    for r in _collect_parent_adapt():
        master_rows.append({"stage": "parent_adapt", **r})
    if master_rows:
        fields = ["stage"] + sorted({k for r in master_rows for k in r if k != "stage"})
        with master.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in master_rows:
                writer.writerow(r)

    ledger_rows = [
        {"hypothesis": "A0 strong-parent preservation removes the architecture replacement tax",
         "stage": "D2.6-A/B", "verdict": "REJECTED — keeping z_base did not lift any candidate above A0_MATCHED; the tax was NOT the bottleneck (H1 refuted)",
         "evidence": "outputs/perf_r2d26/integration/"},
        {"hypothesis": "no-compression HOP beats architecture-identical H1 control",
         "stage": "D2.6-A", "verdict": "NOT SUPPORTED (+0.075pp Acc / -0.993pp F1 vs NC_H1)",
         "evidence": "outputs/perf_r2d26/no_compression/"},
        {"hypothesis": "structured hop content survives on the A0 parent",
         "stage": "D2.6-A/B", "verdict": "WEAK — H1-only controls match or beat HOP on the A0 parent (opposite of the B0-scaffold D2.5 finding)",
         "evidence": "outputs/perf_r2d26/"},
        {"hypothesis": "residual/hierarchical readout recovers the D2.5-lost utility",
         "stage": "D2.6-B", "verdict": "NOT SUPPORTED — all readouts below A0_MATCHED on Acc; HIER_HOP nearly inert",
         "evidence": "outputs/perf_r2d26/integration/"},
        {"hypothesis": "generic readout depth alone explains the gains",
         "stage": "D2.6-B", "verdict": "PARTIAL — READOUT_ONLY F1 +0.40pp ≈ HOP variants' F1 gains; the F1 improvement is mostly generic readout capacity",
         "evidence": "outputs/perf_r2d26/integration/"},
        {"hypothesis": "specific H2 content (not branch presence) drives usage",
         "stage": "D2.6-C", "verdict": "SUPPORTED for FHC/RSF (h2_to_h1 +0.24/+0.30, h2_shuffle +1.19/+1.26, rule 2/3); HIER: NO EVIDENCE",
         "evidence": "outputs/perf_r2d26/causal_usage/"},
        {"hypothesis": "factor summaries carry the side branch's value",
         "stage": "D2.6-C", "verdict": "SUPPORTED (s_pv_off drops 4.5-5.3pp Acc / 7.4-7.9pp F1 in FHC/RSF) — but the totals stay below A0_MATCHED",
         "evidence": "outputs/perf_r2d26/causal_usage/"},
        {"hypothesis": "parent unfreeze improves without destroying the semantic anchor",
         "stage": "D2.6-D", "verdict": "NOT SUPPORTED — S1/S2 ≈ S0 (+0.00~+0.07pp), parent drift CKA 0.996-0.9996 (anchor untouched but no gain)",
         "evidence": "outputs/perf_r2d26/parent_adapt/"},
        {"hypothesis": "deep supervision stays beneficial at the final candidate",
         "stage": "D2.6-E", "verdict": "NOT RETAINED for FHC_HOP (+0.011 Acc / -0.038 F1 < +0.20pp)",
         "evidence": "outputs/perf_r2d26/deep_supervision/"},
        {"hypothesis": "pre-aggregation neighbor utility learning is the next formal route",
         "stage": "D2.6-G", "verdict": "ENTERED (route R3) — post-aggregation evidence real but insufficient on both B0 and A0 parents",
         "evidence": "route matrix R3"},
    ]
    with ledger.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hypothesis", "stage", "verdict", "evidence"])
        writer.writeheader()
        for r in ledger_rows:
            writer.writerow(r)

    if not diagnosis.exists() or "skeleton" in diagnosis.read_text():
        diagnosis.write_text(
            "# R2-Design-2.6 — FINAL DIAGNOSIS (skeleton)\n\n"
            "The 16-question synthesis is authored at D2.6-G.\n", encoding="utf-8")
    return master, ledger, diagnosis


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _write_audit_report() -> Path:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    report = AUDIT_ROOT / "R2D26_AUDIT.md"
    lines = [
        "# R2-Design-2.6 — D2.6-0 Audit & Infrastructure", "",
        "Protocol: seeds 42/43/44; Val only; No Test. A0 is the ONLY primary",
        "parent (R1-baseline A0 checkpoints, structure bitwise == biaxis_final,",
        "max |val acc delta| 0.176pp — disclosed); B0 stays secondary.", "",
        "## 1. Strong-parent contract (plan §3/§55)",
        "",
        "- `z_base = A0(x, G)` is an explicit direct path in every variant;",
        "  the side branch may only ADD information.",
        "- side-off reproduces z_base EXACTLY (bitwise, tested);",
        "  A0 weights are never modified in frozen mode (tested).",
        "- parent always runs in eval mode (no parent dropout — documented;",
        "  D2.6-D unfreezes parameters but not dropout).",
        "",
        "## 2. Side evidence (plan §5/§6)",
        "",
        "- H0 = F^0 (A0 pre-graph factors [C, Pt, Pv]), H1 = P F, H2 = P^2 F;",
        "  no new relation prototype / high-pass / edge router.",
        "- 9 independent experts E_{f,k}: Linear(d,2d) -> LN -> GELU ->",
        "  Dropout(0.1) -> Linear(2d,d).",
        "- H1 controls: architecture-identical, 3 independent H1 transforms,",
        "  NEVER computing H2 (neighbor_mean call count == 1, tested;",
        "  HOP/H1 side parameter parity tested per readout).",
        "",
        "## 3. Readouts (plan §9-§22)",
        "",
        "| readout | formula | out_dim |",
        "|---|---|---|",
        "| no_compression_concat | [z_base \\| 9 tokens] | h+9d |",
        "| factor_hop_concat | [z_base \\| s_C \\| s_Pt \\| s_Pv] | h+3d |",
        "| residual_side_fusion | z_base + R_side(ResidualFusion([s])) | h |",
        "| base_anchored_hier_attention | z_base + W_o(T_final[0]-z_base) | h |",
        "| readout_only_control | z_base + M(z_base), param-matched to HIER | h |",
        "",
        "Residual paths use small nonzero final init (std 1e-3) — no scalar",
        "gate. READOUT_ONLY width is solved so its params match the HIER side",
        "branch within +/-5% (reported in diagnostics).",
        "",
        "## 4. Training (plan §23-§25)",
        "",
        "- A0 fully frozen; side/aux heads/classifier lr 1e-3 wd 1e-4,",
        "  warmup10+cosine, 300 ep / patience 30 / best Val Acc.",
        "- expert deep supervision lambda=0.1 by default; aux heads removed at",
        "  inference (zeroing them does not change the forward — tested).",
        "- A0_BASE trains a fresh matched classifier on the frozen z_base",
        "  under the same schedule.",
        "",
        "## 5. Causal overrides (plan §30)",
        "",
        "- H2_ZERO (zero e2 outputs), H2_TO_H1 (e2 slot consumes H1),",
        "  H2_SHUFFLE (fixed node perm seed 20260904), factor-specific",
        "  C/PT/PV_H2_OFF (replace that factor's H2 with H1), side-off,",
        "  factor-summary ablations, hop ablations. All are eval-time",
        "  overrides; trained weights never change (tested).",
        "",
        "## 6. Verification",
        "",
        "29 tests in tests/test_biaxis_r2_strong_parent.py: frozen A0 bitwise",
        "reproduction; side-off exact z_base; H1 control call count; HOP/H1",
        "parameter parity per readout; H2->H1 token targeting; shuffle",
        "determinism; factor-specific ablation columns; aux-head inference",
        "independence; no-test access (guarded labels in the training loop);",
        "diagnostics finite (all 5 readouts); classifier init replay;",
        "A0 weights unchanged in frozen training.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-Design-2.6 summarizer")
    parser.add_argument("--stage", default="audit",
                        choices=["audit", "no_compression", "integration", "causal",
                                 "parent_adapt", "deep_supervision", "final"])
    args = parser.parse_args()
    if args.stage == "audit":
        print(f"[audit] wrote {_write_audit_report()}")
        return
    if args.stage == "no_compression":
        rows = _collect(NC_ROOT)
        if not rows:
            raise RuntimeError("no no_compression summaries — run perf_r2d26_no_compression.py")
        print(f"[no_compression] wrote {_write_no_compression_report(rows)}")
        return
    if args.stage == "integration":
        rows = _collect(INTEGRATION_ROOT)
        base_rows = _collect(NC_ROOT)
        if not rows:
            raise RuntimeError("no integration summaries — run perf_r2d26_integration.py")
        if not base_rows:
            raise RuntimeError("no A0_BASE rows — run perf_r2d26_no_compression.py first")
        print(f"[integration] wrote {_write_integration_report(rows, base_rows)}")
        return
    if args.stage == "causal":
        rows = []
        for ds_dir in sorted(CAUSAL_ROOT.iterdir()) if CAUSAL_ROOT.exists() else []:
            if not ds_dir.is_dir():
                continue
            for v_dir in sorted(ds_dir.iterdir()):
                if not v_dir.is_dir():
                    continue
                for s_dir in sorted(v_dir.iterdir()):
                    sp = s_dir / "summary.json"
                    if sp.exists():
                        rows.append(json.loads(sp.read_text()))
        if not rows:
            raise RuntimeError("no causal summaries — run perf_r2d26_causal.py")
        print(f"[causal] wrote {_write_causal_report(rows)}")
        return
    if args.stage == "parent_adapt":
        print(f"[parent_adapt] wrote {_write_parent_adapt_report(_collect_parent_adapt())}")
        return
    if args.stage == "deep_supervision":
        print(f"[deep_supervision] wrote {_write_deep_sup_report()}")
        return
    if args.stage == "final":
        m, l, d = _write_final_synthesis()
        print(f"[final] wrote {m} / {l} / {d}")
        return


if __name__ == "__main__":
    main()
