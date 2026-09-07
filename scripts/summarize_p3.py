"""P3 aggregation + report (plan §11/§20/§37/§38).

Stages:
    operator   outputs/p3/operator   O0/OF/OR/OADD/OFR x seeds
               -> p3_operator_results.csv / deltas / mechanism +
                  P3_OPERATOR_REPORT.md
    lowrank    outputs/p3/lowrank    LR-ADD/LR-INT x seeds, references
               O0/OADD/OFR loaded from outputs/p3/operator (never re-run)
               -> p3_lowrank_*.csv + P3_LOWRANK_REPORT.md

Statistical protocol (plan §3):
    - mean ± population std (ddof=0) over the seeds present
    - paired-seed deltas: Delta_s = Metric(variant, s) - Metric(reference, s)
      reported as mean, population std, positive-seed count x/n (pp for acc)
    - single-seed differences < 0.2pp are never interpreted alone
    - decision metric: validation Accuracy; Macro-F1 guards minority classes

Usage:
    python scripts/summarize_p3.py --stage operator
    python scripts/summarize_p3.py --stage lowrank
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
P3_ROOT = PROJECT_ROOT / "outputs" / "p3"

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
OPERATOR_MODES = ["O0", "OF", "OR", "OADD", "OFR"]
OPERATOR_DELTAS = [("OF", "O0"), ("OR", "O0"), ("OADD", "O0"), ("OFR", "OADD"), ("OFR", "O0")]
LOWRANK_MODES = ["O0", "OADD", "OFR", "LR-ADD", "LR-INT"]
# plan §20: the key comparison is LR-INT - LR-ADD; LR-INT vs OFR shows how
# much of the full pair capacity the rank-16 interaction recovers.
LOWRANK_DELTAS = [
    ("LR-INT", "LR-ADD"),
    ("LR-INT", "OFR"),
    ("LR-ADD", "OADD"),
    ("LR-INT", "O0"),
    ("LR-ADD", "O0"),
]
# review §20: the basis operator forms the capacity curve
# O0 -> B4 (65.6K) -> OADD (115K) -> B8 (131.2K) -> B16 (262.3K) -> OFR (311K).
BASIS_MODES = ["O0", "Basis4", "OADD", "Basis8", "Basis16", "OFR"]
BASIS_DELTAS = [
    ("Basis4", "O0"),
    ("Basis8", "Basis4"),
    ("Basis16", "Basis8"),
    ("Basis8", "OADD"),
    ("Basis16", "OFR"),
]

METRIC_LABELS = {
    "val_acc": "Val Acc",
    "best_val_macro_f1": "Val Macro-F1",
    "test_acc": "Test Acc",
    "test_macro_f1": "Test Macro-F1",
}


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return f"{value:.{digits}f}"


def _mean_std(values: list[float], digits: int = 4) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return _fmt(values[0], digits)
    return f"{statistics.mean(values):.{digits}f}±{statistics.pstdev(values):.{digits}f}"


def _metric(results: dict | None, key: str) -> float | None:
    if not results or key not in results:
        return None
    entry = results[key]
    if isinstance(entry, dict):
        return entry.get("mean")
    return entry


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_summaries(stage_root: Path) -> list[dict]:
    if not stage_root.exists():
        return []
    summaries = []
    for path in sorted(stage_root.glob("*/**/summary.json")):
        with path.open(encoding="utf-8") as f:
            summaries.append(json.load(f))
    return summaries


def _mode_groups(summaries: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for s in summaries:
        grouped.setdefault((s["dataset"], s["mode"]), []).append(s)
    return grouped


def _seeded_values(grouped: dict, dataset: str, mode: str, metric_key: str) -> dict[int, float]:
    """{seed: metric} for one (dataset, mode) cell."""
    out: dict[int, float] = {}
    for s in grouped.get((dataset, mode), []):
        if metric_key == "best_val_macro_f1":
            value = s.get("best_val_macro_f1")
        else:
            value = _metric(s.get("results"), metric_key)
        if value is not None:
            out[int(s["seed"])] = float(value)
    return out


def _paired_delta(
    seeded_a: dict[int, float],
    seeded_b: dict[int, float],
) -> dict[str, float] | None:
    """Paired-seed delta a - b over the shared seed set (plan §3.2)."""
    seeds = sorted(set(seeded_a) & set(seeded_b))
    if not seeds:
        return None
    deltas = [seeded_a[s] - seeded_b[s] for s in seeds]
    return {
        "n": len(deltas),
        "mean": statistics.mean(deltas),
        "std": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
        "positive": sum(1 for d in deltas if d > 0),
    }


def _results_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for s in summaries:
        rows.append({
            "dataset": s["dataset"],
            "mode": s["mode"],
            "seed": s["seed"],
            "best_val_acc": _fmt(_metric(s.get("results"), "val_acc")),
            "best_val_macro_f1": _fmt(s.get("best_val_macro_f1")),
            "test_acc": _fmt(_metric(s.get("results"), "test_acc")),
            "test_macro_f1": _fmt(_metric(s.get("results"), "test_macro_f1")),
            "params": s.get("params", ""),
            "epoch_time_sec": _fmt(s.get("epoch_time_sec"), 2),
            "train_peak_gpu_mb": s.get("train_peak_gpu_mb", ""),
        })
    return rows


def _mechanism_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for s in summaries:
        diag = s.get("diagnostics") or {}
        plan = diag.get("plan", {})
        op = diag.get("operator", {})
        row: dict = {
            "dataset": s["dataset"],
            "mode": s["mode"],
            "seed": s["seed"],
            "K_eff": _fmt(diag.get("relation", {}).get("effective_num")),
            "S_R": _fmt(diag.get("relation", {}).get("specialization")),
            "w0_norm": _fmt(op.get("w0_norm")),
            "pair_strength": _fmt(op.get("pair_strength"), 6),
            "message_dev": _fmt(op.get("message_deviation_usage_weighted"), 6),
            "extra_residual_params": op.get("extra_residual_params", ""),
        }
        for name in ("C", "Pt", "Pv"):
            stats = plan.get(name, {})
            row[f"null_{name}"] = _fmt(stats.get("null_mean"))
            row[f"graph_{name}"] = _fmt(stats.get("graph_mass_mean"))
        residual = op.get("residual_norms", {})
        if residual.get("factor"):
            for i, value in enumerate(residual["factor"]):
                row[f"rA_{i}"] = _fmt(value, 5)
        if residual.get("relation"):
            for i, value in enumerate(residual["relation"]):
                row[f"rB_{i}"] = _fmt(value, 5)
        if residual.get("pair"):
            for fi, row_vals in enumerate(residual["pair"]):
                for ki, value in enumerate(row_vals):
                    row[f"rC_{fi}_{ki}"] = _fmt(value, 5)
        rows.append(row)
    return rows


def _report_tables(lines: list[str], grouped: dict, modes: list[str]) -> None:
    for metric_key, label in METRIC_LABELS.items():
        lines.append(f"## {label} (mean±std over seeds)")
        lines.append("")
        lines.append("| Dataset | " + " | ".join(modes) + " |")
        lines.append("|---|" + "---:|" * len(modes))
        for dataset in DATASETS:
            cells = []
            for mode in modes:
                seeded = _seeded_values(grouped, dataset, mode, metric_key)
                cells.append(_mean_std(list(seeded.values())))
            lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
        lines.append("")


def _report_deltas(
    lines: list[str], grouped: dict, deltas: list[tuple[str, str]], metric_key: str, title: str
) -> None:
    lines.append(f"## Paired-seed deltas ({title}, percentage points; mean Δ / std Δ / positive seeds)")
    lines.append("")
    header = "| Dataset | " + " | ".join(f"{a}−{b}" for a, b in deltas) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(deltas))
    for dataset in DATASETS:
        cells = []
        for variant, reference in deltas:
            delta = _paired_delta(
                _seeded_values(grouped, dataset, variant, metric_key),
                _seeded_values(grouped, dataset, reference, metric_key),
            )
            if delta is None:
                cells.append("")
            else:
                cells.append(f"{100 * delta['mean']:+.4f}±{100 * delta['std']:.4f} ({delta['positive']}/{delta['n']})")
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    lines.append("")


def _report_mechanism(lines: list[str], grouped: dict, modes: list[str]) -> None:
    lines.append("## Mechanism (mean over seeds, per dataset x mode)")
    lines.append("")
    lines.append("| Dataset | Mode | K_eff | S_R | pair_strength | message_dev | extra_params |")
    lines.append("|---|" + "---:|" * 6)
    for dataset in DATASETS:
        for mode in modes:
            runs = grouped.get((dataset, mode), [])
            if not runs:
                continue
            diags = [r.get("diagnostics") or {} for r in runs]
            rel = [d.get("relation", {}) for d in diags]
            ops = [d.get("operator", {}) for d in diags]
            k_eff = [v for v in (r.get("effective_num") for r in rel) if v is not None]
            s_r = [v for v in (r.get("specialization") for r in rel) if v is not None]
            pair = [v for v in (o.get("pair_strength") for o in ops) if v is not None]
            dev = [v for v in (o.get("message_deviation_usage_weighted") for o in ops) if v is not None]
            lines.append(
                f"| {dataset} | {mode} | {_mean_std(k_eff, 3)} | {_mean_std(s_r, 3)} | "
                f"{_mean_std(pair, 5)} | {_mean_std(dev, 5)} | {ops[0].get('extra_residual_params', '') if ops else ''} |"
            )
    lines.append("")


def _write_operator_report(summaries: list[dict]) -> None:
    grouped = _mode_groups(summaries)
    lines: list[str] = []
    lines.append("# P3-A Operator Report — Factor–Relation-specific Transformation")
    lines.append("")
    lines.append(
        f"> Runs: {len(summaries)} (5 datasets x 5 variants x seeds). Protocol: full-graph NC, "
        f"p2.mode=null_softmax (eps=0.2), p2.deterministic=false, zero residual init, "
        f"no operator regularizer. mean ± population std; decision metric: Val Acc."
    )
    lines.append("")

    _report_tables(lines, grouped, OPERATOR_MODES)
    _report_deltas(lines, grouped, OPERATOR_DELTAS, "val_acc", "Val Acc")
    _report_deltas(lines, grouped, OPERATOR_DELTAS, "test_acc", "Test Acc")
    _report_mechanism(lines, grouped, OPERATOR_MODES)

    lines.append("## GO / NO-GO checklist (plan §12, fill after analysis)")
    lines.append("")
    lines.append("Strong Pair-specific Signal (→ P3-B interaction branch):")
    lines.append("- [ ] OFR−OADD ≥ +0.30pp mean Val gain on ≥2/5 datasets")
    lines.append("- [ ] OR ≥2 datasets with ≥ +0.20pp and 3/3 seeds same sign")
    lines.append("- [ ] no systematic Macro-F1 regression")
    lines.append("")
    lines.append("Main-effect GO (→ low-rank additive):")
    lines.append("- [ ] OADD > O0 stable across datasets/seeds, OFR ≈ OADD")
    lines.append("")
    lines.append("Single-axis GO: OF > O0 clearly, others no further value → Factor Operator")
    lines.append("- [ ] or: OR > O0 clearly → Relation Operator")
    lines.append("")
    lines.append("Borderline: mean +0.15~0.30pp, 2/3 seeds same sign → add seeds 45/46 (NOT deterministic mode)")
    lines.append("")
    lines.append("NO-GO: no stable multi-seed gain over O0 → keep shared W0")
    lines.append("")
    lines.append("- Decision: **PENDING** (fill after analysis)")
    lines.append("")
    (P3_ROOT / "operator" / "P3_OPERATOR_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_lowrank_report(lowrank_summaries: list[dict], operator_summaries: list[dict]) -> None:
    grouped = _mode_groups(lowrank_summaries + operator_summaries)
    lines: list[str] = []
    lines.append("# P3-B Low-rank Report — Parameter-Matched Factor–Relation Operator")
    lines.append("")
    lines.append(
        f"> Low-rank runs: {len(lowrank_summaries)} (5 datasets x 2 modes x seeds). "
        f"O0/OADD/OFR references from P3-A (outputs/p3/operator, never re-run). "
        f"rank=16, U/V Xavier, a/b zero init; LR-ADD and LR-INT share the EXACT same "
        f"parameter set — only the explicit a_f*b_k interaction differs (plan §17)."
    )
    lines.append("")

    _report_tables(lines, grouped, LOWRANK_MODES)
    _report_deltas(lines, grouped, LOWRANK_DELTAS, "val_acc", "Val Acc")
    _report_deltas(lines, grouped, LOWRANK_DELTAS, "test_acc", "Test Acc")
    _report_mechanism(lines, grouped, LOWRANK_MODES)

    lines.append("## GO checklist (plan §21, fill after analysis)")
    lines.append("")
    lines.append("Interaction GO (LR-INT > LR-ADD):")
    lines.append("- [ ] mean positive on ≥3/5 datasets")
    lines.append("- [ ] ≥2 datasets with ≥ +0.20~0.30pp OR 3/3 seeds same sign")
    lines.append("- [ ] no systematic F1 regression")
    lines.append("")
    lines.append("Additive GO (LR-ADD ≈ LR-INT, both > O0): keep LR-ADD, drop interaction")
    lines.append("")
    lines.append("Full-only GO (OFR clearly > LR variants): one rank sensitivity (r=8/16/32), then decide")
    lines.append("")
    lines.append("- Decision: **PENDING** (fill after analysis)")
    lines.append("")
    (P3_ROOT / "lowrank" / "P3_LOWRANK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_basis_report(basis_summaries: list[dict], operator_summaries: list[dict]) -> None:
    grouped = _mode_groups(basis_summaries + operator_summaries)
    lines: list[str] = []
    lines.append("# P3 Basis Report — Basis-decomposed Cell Operator (review §20)")
    lines.append("")
    lines.append(
        f"> Basis runs: {len(basis_summaries)} (5 datasets x B∈{{4,8,16}} x 3 seeds). "
        f"O0/OADD/OFR references from P3-A (never re-run). T_fk = W0 + sum_b c_fkb V_b, "
        f"full-matrix bases (Xavier) + per-cell coefficients (zero init). Extra params: "
        f"B4 65.6K / B8 131.2K / B16 262.3K vs OADD 115K / OFR 311K / LR 4.2K."
    )
    lines.append("")

    _report_tables(lines, grouped, BASIS_MODES)
    _report_deltas(lines, grouped, BASIS_DELTAS, "val_acc", "Val Acc")
    _report_deltas(lines, grouped, BASIS_DELTAS, "test_acc", "Test Acc")
    _report_mechanism(lines, grouped, BASIS_MODES)

    lines.append("## Review §20 question checklist (fill after analysis)")
    lines.append("")
    lines.append("Does the basis family recover the full-operator gains?")
    lines.append("- [ ] Basis4 > O0 (matches/adds params at low cost)")
    lines.append("- [ ] Basis8 >= OADD (same-params comparison, 115K vs 131K)")
    lines.append("- [ ] Basis16 approaches OFR (262K vs 311K)")
    lines.append("")
    lines.append("If monotone recovery: P3-B failure = shared low-rank subspace too restrictive.")
    lines.append("If flat ~O0: cell transformation genuinely needs the unconstrained full residual (capacity itself).")
    lines.append("")
    lines.append("- Decision: **PENDING** (fill after analysis)")
    lines.append("")
    (P3_ROOT / "basis" / "P3_BASIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 aggregation + report")
    parser.add_argument("--stage", default="operator", choices=["operator", "lowrank", "basis"])
    args = parser.parse_args()
    stage = args.stage

    if stage == "operator":
        summaries = _load_summaries(P3_ROOT / "operator")
        if not summaries:
            print("[summarize] no summary.json under outputs/p3/operator", flush=True)
            return
        stage_root = P3_ROOT / "operator"
        stage_root.mkdir(parents=True, exist_ok=True)
        _write_csv(stage_root / "p3_operator_results.csv", _results_rows(summaries))
        _write_csv(stage_root / "p3_operator_mechanism.csv", _mechanism_rows(summaries))
        grouped = _mode_groups(summaries)
        delta_rows = []
        for dataset in DATASETS:
            for variant, reference in OPERATOR_DELTAS:
                for metric_key in ("val_acc", "test_acc"):
                    delta = _paired_delta(
                        _seeded_values(grouped, dataset, variant, metric_key),
                        _seeded_values(grouped, dataset, reference, metric_key),
                    )
                    if delta is None:
                        continue
                    delta_rows.append({
                        "dataset": dataset,
                        "comparison": f"{variant}-{reference}",
                        "metric": metric_key,
                        "mean_pp": f"{100 * delta['mean']:+.6f}",
                        "std_pp": f"{100 * delta['std']:.6f}",
                        "positive_seeds": delta["positive"],
                        "n_seeds": delta["n"],
                    })
        _write_csv(stage_root / "p3_operator_deltas.csv", delta_rows)
        _write_operator_report(summaries)
        print(f"[summarize] {len(summaries)} runs -> {stage_root}", flush=True)
    elif stage == "lowrank":
        lowrank_summaries = _load_summaries(P3_ROOT / "lowrank")
        operator_summaries = _load_summaries(P3_ROOT / "operator")
        if not lowrank_summaries:
            print("[summarize] no summary.json under outputs/p3/lowrank", flush=True)
            return
        stage_root = P3_ROOT / "lowrank"
        stage_root.mkdir(parents=True, exist_ok=True)
        _write_csv(stage_root / "p3_lowrank_results.csv", _results_rows(lowrank_summaries))
        _write_csv(stage_root / "p3_lowrank_mechanism.csv", _mechanism_rows(lowrank_summaries))
        grouped = _mode_groups(lowrank_summaries + operator_summaries)
        delta_rows = []
        for dataset in DATASETS:
            for variant, reference in LOWRANK_DELTAS:
                for metric_key in ("val_acc", "test_acc"):
                    delta = _paired_delta(
                        _seeded_values(grouped, dataset, variant, metric_key),
                        _seeded_values(grouped, dataset, reference, metric_key),
                    )
                    if delta is None:
                        continue
                    delta_rows.append({
                        "dataset": dataset,
                        "comparison": f"{variant}-{reference}",
                        "metric": metric_key,
                        "mean_pp": f"{100 * delta['mean']:+.6f}",
                        "std_pp": f"{100 * delta['std']:.6f}",
                        "positive_seeds": delta["positive"],
                        "n_seeds": delta["n"],
                    })
        _write_csv(stage_root / "p3_lowrank_deltas.csv", delta_rows)
        _write_lowrank_report(lowrank_summaries, operator_summaries)
        print(f"[summarize] {len(lowrank_summaries)} runs -> {stage_root}", flush=True)
    else:
        basis_summaries = _load_summaries(P3_ROOT / "basis")
        operator_summaries = _load_summaries(P3_ROOT / "operator")
        if not basis_summaries:
            print("[summarize] no summary.json under outputs/p3/basis", flush=True)
            return
        stage_root = P3_ROOT / "basis"
        stage_root.mkdir(parents=True, exist_ok=True)
        _write_csv(stage_root / "p3_basis_results.csv", _results_rows(basis_summaries))
        _write_csv(stage_root / "p3_basis_mechanism.csv", _mechanism_rows(basis_summaries))
        grouped = _mode_groups(basis_summaries + operator_summaries)
        delta_rows = []
        for dataset in DATASETS:
            for variant, reference in BASIS_DELTAS:
                for metric_key in ("val_acc", "test_acc"):
                    delta = _paired_delta(
                        _seeded_values(grouped, dataset, variant, metric_key),
                        _seeded_values(grouped, dataset, reference, metric_key),
                    )
                    if delta is None:
                        continue
                    delta_rows.append({
                        "dataset": dataset,
                        "comparison": f"{variant}-{reference}",
                        "metric": metric_key,
                        "mean_pp": f"{100 * delta['mean']:+.6f}",
                        "std_pp": f"{100 * delta['std']:.6f}",
                        "positive_seeds": delta["positive"],
                        "n_seeds": delta["n"],
                    })
        _write_csv(stage_root / "p3_basis_deltas.csv", delta_rows)
        _write_basis_report(basis_summaries, operator_summaries)
        print(f"[summarize] {len(basis_summaries)} runs -> {stage_root}", flush=True)


if __name__ == "__main__":
    main()
