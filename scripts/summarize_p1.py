"""P1 aggregation + report (plan §28 / §38 / §31).

Reads outputs/p1/<stage>/<dataset>/<variant>/(seed_<s>/)summary.json and writes:

    p1_<stage>_results.csv        Dataset | Variant | Seed | Best Val Acc | Test Acc | Test F1 | Params | Epoch Time | Peak Mem
    p1_<stage>_interaction.csv    Dataset | Metric | dF | dR | dFR | F1R1-F1R0 | F1R1-F0R1
    p1_<stage>_mechanism.csv      relation occupancy / K_eff / beta / alpha entropy / JS / usage matrix
    P1_SCREEN_REPORT.md (screen) / P1_REPORT.md (confirm) / P1_BUDGET_ABLATION_REPORT.md

Interaction effects (plan §17):
    dF  = P(F1R0) - P(F0R0)
    dR  = P(F0R1) - P(F0R0)
    dFR = P(F1R1) - P(F1R0) - P(F0R1) + P(F0R0)

Model decisions use VALIDATION accuracy; test metrics are frozen-checkpoint
confirmations only. Usage matrices are NOT averaged across seeds (relation
prototype permutation, plan §38).

Usage:
    python scripts/summarize_p1.py --stage screen
    python scripts/summarize_p1.py --stage confirm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
P1_ROOT = PROJECT_ROOT / "outputs" / "p1"
BASELINE_CSV = PROJECT_ROOT / "outputs" / "baseline_nc" / "nc_baseline_table.csv"

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
FACTORIAL_VARIANTS = ["F0R0", "F1R0", "F0R1", "F1R1"]
BUDGET_VARIANTS = ["B0", "B1", "B2"]
METRICS = ["val_acc", "test_acc", "test_macro_f1"]
REFERENCE_MODELS = ["gcn", "mmgcn", "dip"]


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


def _load_run_summaries(stage: str) -> list[dict]:
    stage_root = P1_ROOT / stage
    summaries: list[dict] = []
    if not stage_root.exists():
        return summaries
    for path in sorted(stage_root.glob("*/**/summary.json")):
        with path.open(encoding="utf-8") as f:
            summary = json.load(f)
        summaries.append(summary)
    return summaries


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
    # Union of keys: variants can differ in columns (q vs C/Pt/Pv, K=1 vs K=4).
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------


def _results_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        rows.append({
            "dataset": summary["dataset"],
            "variant": summary["variant"],
            "seed": summary["seed"],
            "best_val_acc": _fmt(_metric(summary.get("results"), "val_acc")),
            "test_acc": _fmt(_metric(summary.get("results"), "test_acc")),
            "test_macro_f1": _fmt(_metric(summary.get("results"), "test_macro_f1")),
            "params": summary.get("params", ""),
            "epoch_time_sec": _fmt(summary.get("epoch_time_sec"), 2),
            "train_peak_gpu_mb": summary.get("train_peak_gpu_mb", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# Interaction CSV
# ---------------------------------------------------------------------------


def _grouped_metrics(summaries: list[dict]) -> dict[tuple[str, str, str], float]:
    """{(dataset, variant, seed): metric_value} for val_acc."""
    out = {}
    for summary in summaries:
        value = _metric(summary.get("results"), "val_acc")
        if value is None:
            continue
        out[(summary["dataset"], summary["variant"], summary["seed"])] = value
    return out


def _interaction_rows(summaries: list[dict], stage: str) -> list[dict]:
    rows: list[dict] = []
    grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    for summary in summaries:
        values = {key: _metric(summary.get("results"), key) for key in METRICS}
        if any(v is None for v in values.values()):
            continue
        grouped[(summary["dataset"], summary["variant"], summary["seed"])] = values

    seeds = sorted({key[2] for key in grouped})
    for dataset in DATASETS:
        for metric in METRICS:
            per_seed = []
            for seed in seeds:
                p = {variant: grouped.get((dataset, variant, seed), {}).get(metric) for variant in FACTORIAL_VARIANTS}
                if any(v is None for v in p.values()):
                    continue
                per_seed.append({
                    "dF": p["F1R0"] - p["F0R0"],
                    "dR": p["F0R1"] - p["F0R0"],
                    "dFR": p["F1R1"] - p["F1R0"] - p["F0R1"] + p["F0R0"],
                    "F1R1-F1R0": p["F1R1"] - p["F1R0"],
                    "F1R1-F0R1": p["F1R1"] - p["F0R1"],
                })
            if not per_seed:
                continue
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "dF": _mean_std([item["dF"] for item in per_seed]),
                "dR": _mean_std([item["dR"] for item in per_seed]),
                "dFR": _mean_std([item["dFR"] for item in per_seed]),
                "F1R1-F1R0": _mean_std([item["F1R1-F1R0"] for item in per_seed]),
                "F1R1-F0R1": _mean_std([item["F1R1-F0R1"] for item in per_seed]),
            })
    return rows


# ---------------------------------------------------------------------------
# Mechanism CSV
# ---------------------------------------------------------------------------


def _mechanism_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        diag = summary.get("diagnostics")
        if not diag:
            continue
        relation = diag.get("relation", {})
        budget = diag.get("budget", {})
        alpha_ent = diag.get("alpha_entropy", {})
        js = diag.get("alpha_js", {})
        num_relations = len(relation.get("occ", []))
        edge_entropy = relation.get("mean_edge_entropy")
        row: dict = {
            "dataset": summary["dataset"],
            "variant": summary["variant"],
            "seed": summary["seed"],
            "rel_effective_num": _fmt(relation.get("effective_num")),
            "rel_edge_entropy": _fmt(edge_entropy),
            # Edge-level relation specialization S_R = 1 - H_R / log K
            # (review §12: K_eff excludes global occupancy collapse only;
            # S_R detects per-edge uniform collapse H_R -> log K).
            "rel_specialization": (
                _fmt(1.0 - edge_entropy / (math.log(num_relations) if num_relations > 1 else 1.0))
                if edge_entropy is not None and num_relations > 1
                else ""
            ),
        }
        for idx, occ in enumerate(relation.get("occ", [])):
            row[f"rel_occ_{idx}"] = _fmt(occ)
        for name in ("C", "Pt", "Pv", "q"):
            if name in budget:
                row[f"beta_{name}"] = _fmt(budget[name].get("mean"))
                row[f"beta_{name}_low_frac"] = _fmt(budget[name].get("low_frac"))
                row[f"beta_{name}_high_frac"] = _fmt(budget[name].get("high_frac"))
        for name, value in alpha_ent.items():
            row[f"alpha_ent_{name}"] = _fmt(value)
        for key, value in js.items():
            row[f"js_{key}"] = _fmt(value)
        um = diag.get("usage_matrix")
        if um and um.get("values"):
            for factor, values in zip(um["factors"], um["values"]):
                for rel_idx, value in enumerate(values):
                    row[f"usage_{factor}_R{rel_idx + 1}"] = _fmt(value)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Baseline reference
# ---------------------------------------------------------------------------


def _load_baseline_reference() -> dict[tuple[str, str], str]:
    """{(dataset, model): 'acc±std'} from the frozen NC baseline table."""
    ref: dict[tuple[str, str], str] = {}
    if not BASELINE_CSV.exists():
        return ref
    with BASELINE_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ref[(row["dataset"], row["model"])] = row["test_acc"]
    return ref


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _report_title(stage: str) -> str:
    return {
        "screen": "P1 Screen Report — Semantic Factor × Structural Relation",
        "confirm": "P1 Confirm Report — Semantic Factor × Structural Relation",
        "budget_ablation": "P1 Budget Ablation Report — B0/B1/B2",
    }[stage]


def _write_report(stage: str, summaries: list[dict], interaction_rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append(f"# {_report_title(stage)}")
    lines.append("")
    lines.append("> Auto-generated from `outputs/p1/<stage>/**/summary.json`.")
    lines.append("")
    lines.append(f"- Runs found: {len(summaries)}")
    lines.append("- Decision metric: **validation Accuracy** (test is frozen-checkpoint confirmation only).")
    lines.append("")

    report_variants = BUDGET_VARIANTS if stage == "budget_ablation" else FACTORIAL_VARIANTS
    grouped: dict[tuple[str, str], list[dict]] = {}
    for summary in summaries:
        grouped.setdefault((summary["dataset"], summary["variant"]), []).append(summary)

    # Results table (val + test, mean±std over seeds when present).
    lines.append("## Results")
    lines.append("")
    header = "| Dataset | Variant | Best Val Acc | Test Acc | Test F1 |"
    lines.append(header)
    lines.append("|---|" + "---:|" * 4)
    for dataset in DATASETS:
        for variant in report_variants:
            runs = grouped.get((dataset, variant), [])
            if not runs:
                continue
            val = [_metric(run.get("results"), "val_acc") for run in runs]
            test_acc = [_metric(run.get("results"), "test_acc") for run in runs]
            test_f1 = [_metric(run.get("results"), "test_macro_f1") for run in runs]
            lines.append(
                f"| {dataset} | {variant} | {_mean_std([v for v in val if v is not None])} | "
                f"{_mean_std([v for v in test_acc if v is not None])} | "
                f"{_mean_std([v for v in test_f1 if v is not None])} |"
            )
    lines.append("")

    # B0-vs-B2 comparison (budget ablation only; review §5 supplement).
    if stage == "budget_ablation":
        lines.append("## B0 vs B2 per seed")
        lines.append("")
        lines.append("| Dataset | Seed | B0 val | B2 val | B2−B0 val | B0 test | B2 test |")
        lines.append("|---|" + "---:|" * 6)
        for dataset in DATASETS:
            for seed in sorted({s["seed"] for s in summaries}):
                b0 = next((s for s in grouped.get((dataset, "B0"), []) if s["seed"] == seed), None)
                b2 = next((s for s in grouped.get((dataset, "B2"), []) if s["seed"] == seed), None)
                if b0 is None or b2 is None:
                    continue
                b0_val = _metric(b0.get("results"), "val_acc")
                b2_val = _metric(b2.get("results"), "val_acc")
                delta = (b2_val - b0_val) if b0_val is not None and b2_val is not None else None
                lines.append(
                    f"| {dataset} | {seed} | {_fmt(b0_val)} | {_fmt(b2_val)} | {_fmt(delta)} | "
                    f"{_fmt(_metric(b0.get('results'), 'test_acc'))} | "
                    f"{_fmt(_metric(b2.get('results'), 'test_acc'))} |"
                )
        lines.append("")

    # Interaction table (factorial stages only; val acc rows, decision metric).
    if stage != "budget_ablation":
        lines.append("## Interaction Effects (validation Accuracy)")
        lines.append("")
        lines.append("| Dataset | ΔF | ΔR | ΔFR | F1R1−F1R0 | F1R1−F0R1 |")
        lines.append("|---|" + "---:|" * 5)
        for row in interaction_rows:
            if row["metric"] == "val_acc":
                lines.append(
                    f"| {row['dataset']} | {row['dF']} | {row['dR']} | {row['dFR']} | "
                    f"{row['F1R1-F1R0']} | {row['F1R1-F0R1']} |"
                )
        lines.append("")

    # Mechanism summary (all stages).
    lines.append("## Mechanism")
    lines.append("")
    lines.append("| Dataset | Variant | Seed | K_eff | H_R | S_R | beta_C | beta_Pt | beta_Pv | JS C/Pt | JS C/Pv | JS Pt/Pv |")
    lines.append("|---|" + "---:|" * 12)
    for summary in summaries:
        diag = summary.get("diagnostics") or {}
        relation = diag.get("relation", {})
        budget = diag.get("budget", {})
        js = diag.get("alpha_js", {})
        num_relations = len(relation.get("occ", []))
        h_r = relation.get("mean_edge_entropy")
        s_r = 1.0 - h_r / math.log(num_relations) if h_r is not None and num_relations > 1 else None
        lines.append(
            f"| {summary['dataset']} | {summary['variant']} | {summary['seed']} | "
            f"{_fmt(relation.get('effective_num'))} | {_fmt(h_r)} | {_fmt(s_r)} | "
            f"{_fmt(budget.get('C', {}).get('mean'))} | {_fmt(budget.get('Pt', {}).get('mean'))} | "
            f"{_fmt(budget.get('Pv', {}).get('mean'))} | "
            f"{_fmt(js.get('C_Pt'))} | {_fmt(js.get('C_Pv'))} | {_fmt(js.get('Pt_Pv'))} |"
        )
    lines.append("")
    lines.append("> S_R = 1 − H_R/ln(K): edge-level relation specialization (review §12). S_R ≈ 0 means per-edge r is near uniform — K_eff alone cannot detect this. Usage matrices are saved per run in `usage_matrix.csv` and are NOT averaged across seeds (relation prototype permutation, plan §38).")
    lines.append("")

    # Reference lines (factorial stages only).
    ref = _load_baseline_reference()
    if ref and stage != "budget_ablation":
        lines.append("## Reference lines (frozen NC baselines, test Acc ± std)")
        lines.append("")
        lines.append("| Dataset | GCN | MMGCN | DiP |")
        lines.append("|---|" + "---:|" * 3)
        for dataset in DATASETS:
            lines.append(
                f"| {dataset} | {ref.get((dataset, 'gcn'), '')} | {ref.get((dataset, 'mmgcn'), '')} | "
                f"{ref.get((dataset, 'dip'), '')} |"
            )
        lines.append("")
        lines.append("> P1 mechanism GO does not require beating DiP (plan §32): scientific line F1R1 > F1R0/F0R1; health line ≈ GCN/MMGCN region; final target is P2/P3.")
        lines.append("")

    # GO criteria (§29) computed evidence (factorial stages only).
    if stage != "budget_ablation":
        lines.append("## GO / REVISE / NO-GO checklist (plan §29)")
        lines.append("")
        val_rows = {row["dataset"]: row for row in interaction_rows if row["metric"] == "val_acc"}
        dfr_positive = sum(1 for row in val_rows.values() if row["dFR"] and float(row["dFR"].split("±")[0]) > 0)
        f1r1_gt_f1r0 = sum(1 for row in val_rows.values() if row["F1R1-F1R0"] and float(row["F1R1-F1R0"].split("±")[0]) > 0)
        f1r1_gt_f0r1 = sum(1 for row in val_rows.values() if row["F1R1-F0R1"] and float(row["F1R1-F0R1"].split("±")[0]) > 0)
        num_ds = len(val_rows)
        lines.append(f"- [ ] F1R1 > F1R0 (val) on ≥3/5 datasets — {f1r1_gt_f1r0}/{num_ds} datasets")
        lines.append(f"- [ ] F1R1 > F0R1 (val) on ≥3/5 datasets — {f1r1_gt_f0r1}/{num_ds} datasets")
        lines.append(f"- [ ] ΔFR > 0 (val) on ≥3/5 datasets — {dfr_positive}/{num_ds} datasets")
        collapse = [s for s in summaries if s["variant"] in ("F1R1", "B0", "B1", "B2") and s.get("diagnostics")]
        if collapse:
            k_effs = [s["diagnostics"]["relation"]["effective_num"] for s in collapse]
            occs = [max(s["diagnostics"]["relation"].get("occ", [0])) for s in collapse]
            lines.append(f"- [ ] No relation collapse — K_eff range {min(k_effs):.2f}-{max(k_effs):.2f} (≈1 fails), max occ {max(occs):.2f} (>0.85 fails)")
        else:
            lines.append("- [ ] No relation collapse — no F1R1 diagnostics found")
        lines.append("- [ ] Node-wise factor-relation JS nonzero with clear differences — see mechanism table")
        lines.append("- [ ] F1R1 reaches the GCN/MMGCN healthy region — see reference lines")
        lines.append("")
        lines.append("- Decision: **PENDING** (fill after analysis)")
        lines.append("")

    report_name = {
        "screen": "P1_SCREEN_REPORT.md",
        "confirm": "P1_REPORT.md",
        "budget_ablation": "P1_BUDGET_ABLATION_REPORT.md",
    }[stage]
    (P1_ROOT / stage / report_name).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 aggregation + report")
    parser.add_argument("--stage", default="screen", choices=["screen", "confirm", "budget_ablation"])
    args = parser.parse_args()
    stage = args.stage

    summaries = _load_run_summaries(stage)
    if not summaries:
        print(f"[summarize] no summary.json found under outputs/p1/{stage}", flush=True)
        return

    stage_root = P1_ROOT / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    _write_csv(stage_root / f"p1_{stage}_results.csv", _results_rows(summaries))
    interaction_rows = _interaction_rows(summaries, stage)
    _write_csv(stage_root / f"p1_{stage}_interaction.csv", interaction_rows)
    _write_csv(stage_root / f"p1_{stage}_mechanism.csv", _mechanism_rows(summaries))
    _write_report(stage, summaries, interaction_rows)
    print(
        f"[summarize] {len(summaries)} runs | interaction rows={len(interaction_rows)} | "
        f"-> {stage_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
