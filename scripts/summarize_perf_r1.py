"""R1-A paired summarizer (plan §39 Prompt 5 / §40): each variant minus the
baseline on Val Acc (primary) and Val Macro-F1 (secondary), plus mechanism
tables from the best-checkpoint diagnostics.

Supports the review option B control: --variants A1R1,A1R2,A1R3,A1R4
compares the regularized A1 variants against the A0 baseline (and A1).

Discipline:
    - reads ONLY val metrics (results.json val_acc + log-parsed best Val F1);
      TEST numbers are never read for any selection or judgment.
    - GO gates are pre-registered: §3.1 (single seed) / §3.2 (multi seed);
      each variant is judged independently against the baseline.
    - multi-seed format (plan §40): mean +- population std, paired delta
      mean/std, positive seed count.

Usage:
    python scripts/summarize_perf_r1.py --seeds 42,43,44
    python scripts/summarize_perf_r1.py --seeds 42 --variants A1,A1R1,A1R2,A1R3,A1R4
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

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
WEAK = {"Movies", "Toys", "Grocery"}
GUARDS = {"ele-fashion", "Reddit-S"}
FACTOR_NAMES = ["C", "Pt", "Pv"]
MODE_ROOTS = {
    "A0": "baseline", "A1": "reliability", "A2": "relation_calibration",
    "BL": "routing", "BR": "routing", "BLR": "routing", "C1SG": "multihop",
}
for _v in ("A1R1", "A1R2", "A1R3", "A1R4"):
    MODE_ROOTS[_v] = "reliability"
ROOT = PROJECT_ROOT / "outputs" / "perf_r1"
OUT = ROOT / "summary"

GO_MEAN_DELTA = 0.30
GO_WEAK_FRAC = 0.20
GO_GUARD_MIN = -0.20


def _load_summary(dataset: str, mode: str, seed: int) -> dict | None:
    path = ROOT / MODE_ROOTS[mode] / dataset / mode / f"seed_{seed}" / "summary.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_diag(dataset: str, mode: str, seed: int) -> dict:
    """Read the diagnostics payload straight from disk (fresher than the
    summary's embedded copy — re-analyzed checkpoints don't rewrite
    summary.json)."""
    path = ROOT / MODE_ROOTS[mode] / dataset / mode / f"seed_{seed}" / "r1_diagnostics.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _val_acc(summary: dict) -> float | None:
    results = summary.get("results") or {}
    return (results.get("val_acc") or {}).get("mean")


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def _pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _paired_table(seeds: list[int], baseline: str, variants: list[str]):
    """Per-variant paired deltas vs the baseline + per-dataset aggregates."""
    rows: list[dict] = []
    aggs: dict[str, dict[str, dict]] = {}
    for variant in variants:
        aggs[variant] = {}
        for dataset in DATASETS:
            acc_deltas, f1_deltas, pos = [], [], 0
            a0_accs, a1_accs = [], []
            for seed in seeds:
                s0 = _load_summary(dataset, baseline, seed)
                s1 = _load_summary(dataset, variant, seed)
                if s0 is None or s1 is None:
                    continue
                a0 = _val_acc(s0)
                a1 = _val_acc(s1)
                f0 = s0.get("best_val_macro_f1")
                f1 = s1.get("best_val_macro_f1")
                if a0 is None or a1 is None:
                    continue
                d_acc = 100.0 * (a1 - a0)
                # train.log Val F1 is ALREADY in percentage units (0-100): the
                # pp delta is the raw difference, no x100.
                d_f1 = (f1 - f0) if f0 is not None and f1 is not None else None
                pos += int(d_acc > 0)
                acc_deltas.append(d_acc)
                a0_accs.append(100.0 * a0)
                a1_accs.append(100.0 * a1)
                if d_f1 is not None:
                    f1_deltas.append(d_f1)
                rows.append({
                    "variant": variant, "dataset": dataset, "seed": seed,
                    "A0_val_acc": round(100.0 * a0, 4), "val_acc": round(100.0 * a1, 4),
                    "delta_val_acc_pp": round(d_acc, 4),
                    "delta_val_f1_pp": round(d_f1, 4) if d_f1 is not None else None,
                })
            if acc_deltas:
                aggs[variant][dataset] = {
                    "mean_delta": _mean(acc_deltas),
                    "std_delta": _pstdev(acc_deltas),
                    "positive_seeds": f"{pos}/{len(acc_deltas)}",
                    "a0_mean": _mean(a0_accs),
                    "a1_mean": _mean(a1_accs),
                    "f1_mean_delta": _mean(f1_deltas) if f1_deltas else float("nan"),
                }
    return rows, aggs


def _judge_c1sg(agg: dict[str, dict]) -> dict:
    """R1-C dedicated GO (user ruling): mean(DeltaMovies, DeltaToys) >=
    +0.40 pp with BOTH positive, and Grocery / ele-fashion / Reddit-S all
    >= -0.20 pp."""
    if "Movies" not in agg or "Toys" not in agg:
        return {"verdict": "INCOMPLETE", "checks": {}}
    d_m = agg["Movies"]["mean_delta"]
    d_t = agg["Toys"]["mean_delta"]
    others = [agg[d]["mean_delta"] for d in ("Grocery", "ele-fashion", "Reddit-S") if d in agg]
    mean_mt = 0.5 * (d_m + d_t)
    ok = mean_mt >= 0.40 and d_m > 0 and d_t > 0 and all(o >= -0.20 for o in others)
    return {
        "verdict": "GO" if ok else "NO-GO",
        "checks": {
            "mean_delta_MT_pp": round(mean_mt, 3),
            "movies_delta_pp": round(d_m, 3),
            "toys_delta_pp": round(d_t, 3),
            "others_min_delta_pp": round(min(others), 3) if others else None,
            "verdict": "GO" if ok else "NO-GO",
        },
    }


def _judge(agg: dict[str, dict], seed_rows: list[dict], num_seeds: int) -> dict:
    weak = {d: agg[d] for d in WEAK if d in agg}
    guards = {d: agg[d] for d in GUARDS if d in agg}
    if not weak:
        return {"verdict": "INCOMPLETE", "checks": {}}
    mean_weak = _mean([v["mean_delta"] for v in weak.values()])
    guard_ok = all(v["mean_delta"] >= GO_GUARD_MIN for v in guards.values())
    f1_weak = _mean([v["f1_mean_delta"] for v in weak.values()])
    f1_all = _mean([v["f1_mean_delta"] for v in agg.values()])
    checks = {
        "mean_delta_MTG_pp": round(mean_weak, 3),
        "guards_above_-0.20": bool(guard_ok),
        "f1_secondary": {"mean_delta_MTG": round(f1_weak, 3), "mean_delta_all": round(f1_all, 3)},
    }

    if num_seeds <= 1:
        # Plan §3.1 seed42 screening GO.
        n_above = sum(1 for v in weak.values() if v["mean_delta"] > GO_WEAK_FRAC)
        checks["weak_datasets_above_0.20"] = n_above
        verdict = "GO" if (mean_weak >= GO_MEAN_DELTA and n_above >= 2 and guard_ok) else "NO-GO"
        checks["verdict"] = verdict
        return checks

    # Plan §3.2 multi-seed formal GO.
    n_weak_pos = sum(1 for v in weak.values() if v["mean_delta"] > 0)
    weak_seed_rows = [r for r in seed_rows if r["dataset"] in WEAK]
    pos_frac = (
        sum(1 for r in weak_seed_rows if r["delta_val_acc_pp"] > 0) / len(weak_seed_rows)
        if weak_seed_rows else 0.0
    )
    checks["weak_datasets_positive"] = n_weak_pos
    checks["positive_seed_frac_weak"] = round(pos_frac, 3)

    if mean_weak >= 0.50 and guard_ok:
        strong = True
    elif sum(1 for v in weak.values() if v["mean_delta"] >= 0.50) >= 2 and guard_ok:
        strong = True
    else:
        strong = False

    if strong:
        verdict = "STRONG-GO"
    elif mean_weak >= GO_MEAN_DELTA and n_weak_pos >= 2:
        verdict = "GO"
    elif GO_MEAN_DELTA > mean_weak >= 0.15 and pos_frac >= 2.0 / 3.0:
        verdict = "BORDERLINE"
    else:
        verdict = "NO-GO"
    checks["verdict"] = verdict
    return checks


def _fmt(vals: list[float]) -> str:
    return f"{_mean(vals):.4f}" if vals else "—"


def _eta_table(seeds: list[int], variants: list[str]) -> list[str]:
    lines = ["| dataset | mode | factor | eta mean | eta cv | frac<0.5 | frac>1.5 | neighbor_std | neighbor_cv | corr(eta,cos) |"]
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in DATASETS:
        for variant in variants:
            if variant == "A0":
                continue
            for fname in FACTOR_NAMES:
                rels, nbs, corrs = [], [], []
                for seed in seeds:
                    if _load_summary(dataset, variant, seed) is None:
                        continue
                    rel = _load_diag(dataset, variant, seed).get("reliability")
                    if not rel:
                        continue
                    idx = FACTOR_NAMES.index(fname)
                    rels.append(rel["eta"][f"F{idx + 1}"])
                    nbs.append(rel["neighbor"][f"F{idx + 1}"])
                    corrs.append(rel["corr_eta_cos"][f"F{idx + 1}"])
                if not rels:
                    continue
                lines.append(
                    f"| {dataset} | {variant} | {fname} | {_mean([e['mean'] for e in rels]):.4f} | "
                    f"{_mean([e['cv'] for e in rels]):.4f} | "
                    f"{_mean([e['frac_lt_0.5'] for e in rels]):.4f} | "
                    f"{_mean([e['frac_gt_1.5'] for e in rels]):.4f} | "
                    f"{_mean([n['neighbor_std_mean'] for n in nbs]):.4f} | "
                    f"{_mean([n['neighbor_cv_mean'] for n in nbs]):.4f} | "
                    f"{_mean(corrs):.4f} |"
                )
    return lines


def _mechanism_table(seeds: list[int], baseline: str, variants: list[str]) -> list[str]:
    """D_ctx, context_change (mean over cells), weighted coherence sim_range."""
    lines = ["| dataset | mode | D_ctx^C | D_ctx^Pt | D_ctx^Pv | mean Δg | sim_range C | sim_range Pt | sim_range Pv |"]
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    modes = [baseline] + [v for v in variants if v != baseline]
    for dataset in DATASETS:
        for mode in modes:
            dctx = {f: [] for f in FACTOR_NAMES}
            dg, sim_range = [], {f: [] for f in FACTOR_NAMES}
            for seed in seeds:
                if _load_summary(dataset, mode, seed) is None:
                    continue
                diag = _load_diag(dataset, mode, seed)
                if diag.get("reliability"):
                    cc = diag.get("context_change") or {}
                    vals = [c["mean_all"] for c in cc.values() if c.get("mean_all") is not None]
                    if vals:
                        dg.append(_mean(vals))
                    sim = diag["reliability"]["weighted_semantic_coherence"]
                    for fi, fname in enumerate(FACTOR_NAMES):
                        row = sim[fi]
                        sim_range[fname].append(max(row) - min(row))
                dd = diag.get("d_ctx")
                if dd:
                    for fname in FACTOR_NAMES:
                        v = dd.get(fname)
                        if v is not None:
                            dctx[fname].append(v)
            cells = [_fmt(dctx[f]) for f in FACTOR_NAMES]
            lines.append(
                f"| {dataset} | {mode} | {cells[0]} | {cells[1]} | {cells[2]} | "
                f"{_fmt(dg)} | {_fmt(sim_range['C'])} | {_fmt(sim_range['Pt'])} | {_fmt(sim_range['Pv'])} |"
            )
    return lines


def _probe_table(seeds: list[int], baseline: str, variants: list[str]) -> list[str]:
    """Delta_relctx = probe(f|g_all) - probe(f|g_bar), per dataset/mode/factor."""
    lines = ["| dataset | mode | Δ_relctx C | Δ_relctx Pt | Δ_relctx Pv |"]
    lines.append("|---|---|---:|---:|---:|")
    modes = [baseline] + [v for v in variants if v != baseline]
    for dataset in DATASETS:
        for mode in modes:
            deltas: dict[str, list[float]] = {f: [] for f in FACTOR_NAMES}
            for seed in seeds:
                rows = _read_csv_rows(
                    ROOT / MODE_ROOTS[mode] / dataset / mode / f"seed_{seed}" / "context_probes.csv"
                )
                for fname in FACTOR_NAMES:
                    fr = [r for r in rows if r["factor"] == fname]
                    g_all = [float(r["val_acc"]) for r in fr if r["variant"] == "f|g_all"]
                    g_bar = [float(r["val_acc"]) for r in fr if r["variant"] == "f|g_bar"]
                    if g_all and g_bar:
                        deltas[fname].append(100.0 * (g_all[0] - g_bar[0]))
            lines.append(
                f"| {dataset} | {mode} | {_fmt(deltas['C'])} | {_fmt(deltas['Pt'])} | {_fmt(deltas['Pv'])} |"
            )
    return lines


def _cf_table(seeds: list[int], baseline: str, variants: list[str]) -> list[str]:
    lines = [
        "| dataset | mode | CF0 | CF1 uniform | CF2 avail | Δ_select (CF0−CF1) | "
        "Δ_sem (CF0−CF2) | Δ_local (CF0−CF4) | Δ_demand (CF0−CF5) |",
    ]
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    modes = [baseline] + [v for v in variants if v != baseline and v != "C1SG"]
    for dataset in DATASETS:
        for mode in modes:
            by_cf: dict[str, list[float]] = {
                "CF0_current": [], "CF1_uniform": [], "CF2_availability": [],
                "CF4_no_local": [], "CF5_mean_beta": [],
            }
            for seed in seeds:
                rows = _read_csv_rows(
                    ROOT / MODE_ROOTS[mode] / dataset / mode / f"seed_{seed}" / "routing_counterfactual.csv"
                )
                for row in rows:
                    if row["cf"] in by_cf:
                        by_cf[row["cf"]].append(100.0 * float(row["val_acc"]))
            v = {k: _mean(vals) for k, vals in by_cf.items()}
            lines.append(
                f"| {dataset} | {mode} | {v['CF0_current']:.4f} | {v['CF1_uniform']:.4f} | "
                f"{v['CF2_availability']:.4f} | {v['CF0_current'] - v['CF1_uniform']:+.2f} | "
                f"{v['CF0_current'] - v['CF2_availability']:+.2f} | "
                f"{v['CF0_current'] - v['CF4_no_local']:+.2f} | "
                f"{v['CF0_current'] - v['CF5_mean_beta']:+.2f} |"
            )
    return lines


def _calibration_table(seeds: list[int], variants: list[str]) -> list[str]:
    """A2 edge-level mechanism statistics: JS/KL vs r^str, pairwise factor
    JS, semantic coherence range, entropy / K_eff (user-specified list)."""
    lines = [
        "| dataset | factor | JS(r^f,r^str) | KL(r^f‖r^str) | KL(r^str‖r^f) | "
        "coher range | entropy | K_eff |",
    ]
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    pair_lines = ["| dataset | JS(r^C,r^Pt) | JS(r^C,r^Pv) | JS(r^Pt,r^Pv) |", "|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for variant in variants:
            if variant != "A2":
                continue
            stats: list[dict] = []
            for seed in seeds:
                if _load_summary(dataset, variant, seed) is None:
                    continue
                cal = _load_diag(dataset, variant, seed).get("calibration")
                if cal:
                    stats.append(cal)
            if not stats:
                continue
            for f, fname in enumerate(FACTOR_NAMES):
                lines.append(
                    f"| {dataset} | {fname} | {_mean([s['js_str'][f] for s in stats]):.4f} | "
                    f"{_mean([s['kl_f2str'][f] for s in stats]):.4f} | "
                    f"{_mean([s['kl_str2f'][f] for s in stats]):.4f} | "
                    f"{_mean([s['semantic_coherence_range'][f] for s in stats]):.4f} | "
                    f"{_mean([s['entropy'][f] for s in stats]):.4f} | "
                    f"{_mean([s['k_eff'][f] for s in stats]):.4f} |"
                )
            jp = [s["js_pairwise"] for s in stats]
            pair_lines.append(
                f"| {dataset} | {_mean([s['C_Pt'] for s in jp]):.4f} | "
                f"{_mean([s['C_Pv'] for s in jp]):.4f} | {_mean([s['Pt_Pv'] for s in jp]):.4f} |"
            )
    return lines + [""] + pair_lines


def _hop_table(seeds: list[int], variants: list[str]) -> list[str]:
    """C1SG hop mechanism (user ruling): lam distribution per factor,
    |lam|<0.05 fraction, correction/base norm ratio, cos(F1,F2)."""
    lines = [
        "| dataset | factor | λ mean | λ std | mean\\|λ\\| | λ p50 | "
        "frac \\|λ\\|<0.05 | corr/base ratio | cos(F1,F2) |",
    ]
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in DATASETS:
        for variant in variants:
            if variant != "C1SG":
                continue
            for fname in FACTOR_NAMES:
                rows: list[dict] = []
                for seed in seeds:
                    if _load_summary(dataset, variant, seed) is None:
                        continue
                    hop = _load_diag(dataset, variant, seed).get("hop") or {}
                    if hop.get(fname):
                        rows.append(hop[fname])
                if not rows:
                    continue
                lines.append(
                    f"| {dataset} | {fname} | {_mean([r['lam_mean'] for r in rows]):.4f} | "
                    f"{_mean([r['lam_std'] for r in rows]):.4f} | {_mean([r['lam_abs_mean'] for r in rows]):.4f} | "
                    f"{_mean([r['lam_p50'] for r in rows]):.4f} | {_mean([r['frac_abs_lt_0.05'] for r in rows]):.4f} | "
                    f"{_mean([r['correction_base_ratio'] for r in rows]):.4f} | {_mean([r['cos_F1F2'] for r in rows]):.4f} |"
                )
    return lines


def _hop_cf_table(seeds: list[int], variants: list[str]) -> list[str]:
    """C1SG same-checkpoint learned-lambda vs lambda=0 (user ruling)."""
    lines = ["| dataset | CF8 hop-on | CF9 hop-off | Δ (on−off) pp |"]
    lines.append("|---|---:|---:|---:|")
    for dataset in DATASETS:
        for variant in variants:
            if variant != "C1SG":
                continue
            on, off = [], []
            for seed in seeds:
                rows = _read_csv_rows(
                    ROOT / MODE_ROOTS[variant] / dataset / variant / f"seed_{seed}" / "hop_counterfactual.csv"
                )
                for row in rows:
                    if row["cf"] == "CF8_hop_on":
                        on.append(100.0 * float(row["val_acc"]))
                    elif row["cf"] == "CF9_hop_off":
                        off.append(100.0 * float(row["val_acc"]))
            if on and off:
                lines.append(
                    f"| {dataset} | {_mean(on):.4f} | {_mean(off):.4f} | {_mean(on) - _mean(off):+.2f} |"
                )
    return lines


def _j_table(seeds: list[int], variants: list[str]) -> list[str]:
    """BLR J-series (user ruling): J0 full / J1 local off / J2 relation off /
    J3 both off, frozen same-checkpoint, Val Acc %."""
    lines = [
        "| dataset | J0 | J1 (δ_local=0) | J2 (δ_rel=0) | J3 (both 0) | "
        "J0−J1 pp | J0−J2 pp | J0−J3 pp |",
    ]
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in DATASETS:
        for variant in variants:
            if variant != "BLR":
                continue
            by_j: dict[str, list[float]] = {"J0_full": [], "J1_local_off": [], "J2_relation_off": [], "J3_both_off": []}
            for seed in seeds:
                rows = _read_csv_rows(
                    ROOT / MODE_ROOTS[variant] / dataset / variant / f"seed_{seed}" / "j_counterfactual.csv"
                )
                for row in rows:
                    if row["j"] in by_j:
                        by_j[row["j"]].append(100.0 * float(row["val_acc"]))
            v = {k: _mean(vals) for k, vals in by_j.items()}
            lines.append(
                f"| {dataset} | {v['J0_full']:.4f} | {v['J1_local_off']:.4f} | "
                f"{v['J2_relation_off']:.4f} | {v['J3_both_off']:.4f} | "
                f"{v['J0_full'] - v['J1_local_off']:+.2f} | "
                f"{v['J0_full'] - v['J2_relation_off']:+.2f} | "
                f"{v['J0_full'] - v['J3_both_off']:+.2f} |"
            )
    return lines


def _router_table(seeds: list[int], variants: list[str]) -> list[str]:
    """BLR margin / common-shift / centered-residual stats (user ruling)."""
    lines = [
        "| dataset | factor | M^LG mean | M^LG std | M^LG p50 | "
        "δ̄^R mean | δ̄^R std | δ^{R,c} std |",
    ]
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for dataset in DATASETS:
        for variant in variants:
            if variant != "BLR":
                continue
            for fname in FACTOR_NAMES:
                rows: list[dict] = []
                for seed in seeds:
                    path = ROOT / MODE_ROOTS[variant] / dataset / variant / f"seed_{seed}" / "router_stats.json"
                    if not path.exists():
                        continue
                    with path.open(encoding="utf-8") as f:
                        stats = json.load(f)
                    rows.append(stats[fname])
                if not rows:
                    continue
                lines.append(
                    f"| {dataset} | {fname} | {_mean([r['margin_mean'] for r in rows]):.4f} | "
                    f"{_mean([r['margin_std'] for r in rows]):.4f} | {_mean([r['margin_p50'] for r in rows]):.4f} | "
                    f"{_mean([r['shift_mean'] for r in rows]):.4f} | {_mean([r['shift_std'] for r in rows]):.4f} | "
                    f"{_mean([r['centered_std'] for r in rows]):.4f} |"
                )
    return lines


def _local_table(seeds: list[int], variants: list[str]) -> list[str]:
    """BL/BLR dynamic-Local mechanism (user §10): delta_if0 stats,
    std_i(s_if0), std_i(beta), and the frozen Delta_dyn-local."""
    lines = [
        "| dataset | mode | factor | δ mean | δ std | δ p50 | std_i(s_if0) | "
        "std_i(β) | Δ_dyn-local pp |",
    ]
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for dataset in DATASETS:
        for variant in variants:
            if variant not in ("BL", "BLR"):
                continue
            stats_by_factor: dict[str, list[dict]] = {f: [] for f in FACTOR_NAMES}
            dyn: list[float] = []
            for seed in seeds:
                if _load_summary(dataset, variant, seed) is None:
                    continue
                path = ROOT / MODE_ROOTS[variant] / dataset / variant / f"seed_{seed}" / "local_stats.json"
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as f:
                    stats = json.load(f)
                for fname in FACTOR_NAMES:
                    stats_by_factor[fname].append(stats[fname])
                rows = _read_csv_rows(
                    ROOT / MODE_ROOTS[variant] / dataset / variant / f"seed_{seed}" / "routing_counterfactual.csv"
                )
                c0 = [float(r["val_acc"]) for r in rows if r["cf"] == "CF0_current"]
                c7 = [float(r["val_acc"]) for r in rows if r["cf"] == "CF7_dynlocal_off"]
                if c0 and c7:
                    dyn.append(100.0 * (c0[0] - c7[0]))
            for fname in FACTOR_NAMES:
                rows = stats_by_factor[fname]
                if not rows:
                    continue
                lines.append(
                    f"| {dataset} | {variant} | {fname} | {_mean([r['delta_mean'] for r in rows]):.4f} | "
                    f"{_mean([r['delta_std'] for r in rows]):.4f} | {_mean([r['delta_p50'] for r in rows]):.4f} | "
                    f"{_mean([r['s_local_std'] for r in rows]):.4f} | {_mean([r['beta_std'] for r in rows]):.4f} | "
                    f"{_fmt(dyn)} |"
                )
    return lines


def _cost_table(seeds: list[int], baseline: str, variants: list[str]) -> list[str]:
    lines = ["| dataset | mode | params | epoch s | peak GB | diag peak MB |"]
    lines.append("|---|---|---:|---:|---:|---:|")
    modes = [baseline] + [v for v in variants if v != baseline]
    for dataset in DATASETS:
        for mode in modes:
            p, et, pk, dk = [], [], [], []
            for seed in seeds:
                s = _load_summary(dataset, mode, seed)
                if not s:
                    continue
                p.append(s.get("params") or 0)
                if s.get("epoch_time_sec"):
                    et.append(s["epoch_time_sec"])
                if s.get("train_peak_gpu_mb"):
                    pk.append(s["train_peak_gpu_mb"] / 1024.0)
                diag = _load_diag(dataset, mode, seed)
                if diag.get("diag_peak_allocated_mb"):
                    dk.append(diag["diag_peak_allocated_mb"])
                elif (s.get("diagnostics") or {}).get("diag_peak_allocated_mb"):
                    dk.append(s["diagnostics"]["diag_peak_allocated_mb"])
            lines.append(
                f"| {dataset} | {mode} | {_mean(p):.0f} | {_fmt(et)} | {_fmt(pk)} | {_fmt(dk)} |"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="R1-A paired summarizer")
    parser.add_argument("--seeds", default="42", help="comma-separated seeds")
    parser.add_argument("--baseline", default="A0", help="baseline mode label")
    parser.add_argument("--variants", default="A1", help="comma-separated variant labels")
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    baseline = args.baseline
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows, aggs = _paired_table(seeds, baseline, variants)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "R1_A_paired_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    judges = {}
    for variant in variants:
        if variant == "C1SG":
            judges[variant] = _judge_c1sg(aggs.get(variant, {}))
        else:
            vrows = [r for r in rows if r["variant"] == variant]
            judges[variant] = _judge(aggs.get(variant, {}), vrows, len(seeds))

    gate = "§3.1 seed42 screening GO" if len(seeds) <= 1 else "§3.2 multi-seed formal GO"
    generic = [v for v in variants if v != "C1SG"]
    lines = [f"# R1-A REPORT — baseline {baseline}, variants {variants}", ""]
    lines.append(f"> seeds = {seeds}；paired variant−{baseline}；只依据 Validation，未读 Test。")
    lines.append("")
    if generic:
        lines.append(f"## 0. GO 判定（预注册 {gate}，逐 variant）")
        lines.append("")
        lines.append("| variant | verdict | mean ΔValAcc(M/T/G) pp | weak positive | guards OK | ΔValF1 M/T/G |")
        lines.append("|---|---|---:|---|---|---:|")
        for variant in generic:
            j = judges[variant]
            weak_pos = j.get("weak_datasets_above_0.20", j.get("weak_datasets_positive"))
            lines.append(
                f"| {variant} | **{j['verdict']}** | {j['mean_delta_MTG_pp']:+.3f} | "
                f"{weak_pos}/3 | {j['guards_above_-0.20']} | {j['f1_secondary']['mean_delta_MTG']:+.3f} |"
            )
        lines.append("")
    if "C1SG" in variants:
        j = judges["C1SG"]
        c = j["checks"]
        lines.append("## 0b. C1SG 判定（R1-C 专用 GO：mean(ΔMovies,ΔToys) ≥ +0.40 且两者为正，其余三集 ≥ −0.20）")
        lines.append("")
        lines.append(
            f"- **{c['verdict']}**  mean(ΔMovies,ΔToys) = {c['mean_delta_MT_pp']:+.3f} pp "
            f"（Movies {c['movies_delta_pp']:+.3f} / Toys {c['toys_delta_pp']:+.3f}；"
            f"其余最低 {c['others_min_delta_pp']:+.3f}）"
        )
        lines.append("")
    lines.append("## 1. Paired 主表（Val Acc %，variant−A0 pp）")
    lines.append("")
    lines.append("| dataset | variant | A0 mean | variant mean | Δ mean | Δ std | pos seeds | Δ F1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for variant in variants:
        for dataset in DATASETS:
            v = aggs.get(variant, {}).get(dataset)
            if not v:
                continue
            lines.append(
                f"| {dataset} | {variant} | {v['a0_mean']:.4f} | {v['a1_mean']:.4f} | {v['mean_delta']:+.3f} | "
                f"{v['std_delta']:.3f} | {v['positive_seeds']} | {v['f1_mean_delta']:+.3f} |"
            )
    lines.append("")
    lines.append("## 2. eta 统计（best checkpoint，3-seed 均值）")
    lines.append("")
    lines.extend(_eta_table(seeds, variants))
    lines.append("")
    lines.append("## 3. Context 机制（D_ctx / Δg / weighted coherence）")
    lines.append("")
    lines.extend(_mechanism_table(seeds, baseline, variants))
    lines.append("")
    lines.append("## 4. Δ_relctx fixed Ridge probe（pp = probe(f|g_all) − probe(f|g_bar)）")
    lines.append("")
    lines.extend(_probe_table(seeds, baseline, variants))
    lines.append("")
    if "A2" in variants:
        lines.append("## 4b. A2 relation calibration 机制（r^f vs r^str）")
        lines.append("")
        lines.extend(_calibration_table(seeds, variants))
        lines.append("")
    if any(v in ("BL", "BLR") for v in variants):
        lines.append("## 4c. BL dynamic-Local 机制（δ_if0 / Δ_dyn-local）")
        lines.append("")
        lines.extend(_local_table(seeds, variants))
        lines.append("")
    if "BLR" in variants:
        lines.append("## 4d. BLR J-series 冻结反事实（同 checkpoint）")
        lines.append("")
        lines.extend(_j_table(seeds, variants))
        lines.append("")
        lines.append("## 4e. BLR Local-vs-Graph margin 与 relation 残差分解")
        lines.append("")
        lines.extend(_router_table(seeds, variants))
        lines.append("")
    if "C1SG" in variants:
        lines.append("## 4f. C1SG hop 机制（λ / correction ratio / cos(F1,F2)）")
        lines.append("")
        lines.extend(_hop_table(seeds, variants))
        lines.append("")
        lines.append("## 4g. C1SG learned-λ vs λ=0（同 checkpoint，冻结权重）")
        lines.append("")
        lines.extend(_hop_cf_table(seeds, variants))
        lines.append("")
    lines.append("## 5. Γ counterfactual（冻结权重，Val Acc %）")
    lines.append("")
    lines.extend(_cf_table(seeds, baseline, variants))
    lines.append("")
    lines.append("## 6. 成本")
    lines.append("")
    lines.extend(_cost_table(seeds, baseline, variants))
    lines.append("")

    fname = "R1_A_REPORT.md" if variants == ["A1"] else f"R1_A_{'-'.join(variants)}_REPORT.md"
    (OUT / fname).write_text("\n".join(lines), encoding="utf-8")
    for variant in variants:
        j = judges[variant]
        if variant == "C1SG":
            print(f"[summarize] {variant}: verdict={j['verdict']} "
                  f"mean_MT={j['checks']['mean_delta_MT_pp']:+.3f}")
        else:
            print(f"[summarize] {variant}: verdict={j['verdict']} "
                  f"mean_MTG={j['mean_delta_MTG_pp']:+.3f}")
    print(f"[summarize] saved -> {OUT / fname}")


if __name__ == "__main__":
    main()
