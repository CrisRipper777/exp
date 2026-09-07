"""Audit and summarize the R5-0 validation-only screen."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs" / "r5_0_context_localization"
FORMAL_ROOT = OUT_ROOT / "formal"
MASTER_PATH = OUT_ROOT / "summary" / "R5_0_MASTER_TABLE.csv"
DATASETS = ["Movies", "Toys", "Grocery"]
SEEDS = [42, 43]
VARIANTS = ["M0-MEAN", "M1-MEAN-LN", "S0-SYM", "M2-MEAN-G4", "S1-SYM-G4"]


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_baselines() -> dict[tuple[str, str, int], dict[str, float]]:
    rows = _read_csv(PROJECT_ROOT / "outputs/r2d29/g0_reference/runs.csv")
    baselines = {}
    for row in rows:
        model = str(row["model"]).lower()
        if model not in {"biaxis_final", "dip"} or row["dataset"] not in DATASETS or int(row["seed"]) not in SEEDS:
            continue
        baselines[("A0" if model == "biaxis_final" else "DiP", row["dataset"], int(row["seed"]))] = {
            "acc": float(row["val_acc"]),
            "f1": float(row["val_f1"]),
        }
    return baselines


def _audit_r4_sources() -> dict:
    """Recalculate the specified R4 CSVs without trusting report aggregates."""
    audit_rows: list[dict] = []

    def add_source(label: str, path: Path, selector) -> None:
        selected = [row for row in _read_csv(path) if selector(row)]
        acc = [float(row["val_acc"]) * (100.0 if float(row["val_acc"]) <= 1.0 else 1.0) for row in selected]
        f1_key = "best_val_macro_f1" if selected and "best_val_macro_f1" in selected[0] else "val_f1"
        f1 = [float(row[f1_key]) * (100.0 if float(row[f1_key]) <= 1.0 else 1.0) for row in selected]
        acc_mean, acc_std = _mean_std(acc)
        f1_mean, f1_std = _mean_std(f1)
        audit_rows.append({
            "source": label,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "n": len(selected),
            "acc_mean_pct": acc_mean,
            "acc_std_pct": acc_std,
            "f1_mean_pct": f1_mean,
            "f1_std_pct": f1_std,
        })

    g0 = PROJECT_ROOT / "outputs/r2d29/g0_reference/runs.csv"
    add_source("A0", g0, lambda row: row["model"] == "biaxis_final" and row["dataset"] in DATASETS and int(row["seed"]) in SEEDS)
    add_source("DiP", g0, lambda row: row["model"] == "dip" and row["dataset"] in DATASETS and int(row["seed"]) in SEEDS)
    source_specs = [
        ("R4-parent2-C0", "outputs/r4_scope_v2b/parent2/runs.csv", "C0"),
        ("R4-parent2-C1", "outputs/r4_scope_v2b/parent2/runs.csv", "C1"),
        ("R4-parent2-C2", "outputs/r4_scope_v2b/parent2/runs.csv", "C2"),
        ("R4-parent2-control-C1-BETA1", "outputs/r4_scope_v2b/parent2_control/runs.csv", "C1-BETA1"),
        ("R4-parent2-control-C2-BETA1", "outputs/r4_scope_v2b/parent2_control/runs.csv", "C2-BETA1"),
        ("R4-repair2-IND", "outputs/r4_scope_v2b/repair2/runs.csv", "IND"),
        ("R4-repair2-FULL", "outputs/r4_scope_v2b/repair2/runs.csv", "FULL"),
        ("R4-repair2-ORB", "outputs/r4_scope_v2b/repair2/runs.csv", "ORB"),
        ("R4-diag-parent2-control-C2-BETA1", "outputs/r4_scope_v2b_diag/parent2_control/runs.csv", "C2-BETA1"),
    ]
    for label, relative_path, variant in source_specs:
        path = PROJECT_ROOT / relative_path
        add_source(label, path, lambda row, variant=variant: row["variant"] == variant and row["dataset"] in DATASETS and int(row["seed"]) in SEEDS)

    audit_dir = OUT_ROOT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        audit_dir / "R4_A_RECALC.csv", audit_rows,
        ["source", "path", "n", "acc_mean_pct", "acc_std_pct", "f1_mean_pct", "f1_std_pct"],
    )
    lookup = {row["source"]: row for row in audit_rows}
    discrepancy = {
        "report_path": "docs/R4-1b_R4-2_results_report.md",
        "report_claim": {"variant": "C2 beta=1 control", "acc_mean_pct": 72.547, "f1_mean_pct": 66.068},
        "canonical_csv": lookup["R4-parent2-control-C2-BETA1"],
        "diagnostic_rerun_csv": lookup["R4-diag-parent2-control-C2-BETA1"],
        "explanation": (
            "The report values round the separate r4_scope_v2b_diag rerun "
            "(72.546931/66.067683), not the canonical parent2_control/runs.csv "
            "specified for this audit (72.504713/65.890467). This is a source "
            "mismatch between two six-run CSVs, not an arithmetic rounding error."
        ),
        "audit_rule": "Use the explicitly specified canonical CSV for R4-A comparisons; retain the diagnostic rerun as provenance only.",
    }
    (audit_dir / "R4_A_DISCREPANCY.json").write_text(
        json.dumps(discrepancy, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"rows": audit_rows, "discrepancy": discrepancy}


def main() -> None:
    rows = _read_csv(MASTER_PATH)
    if len(rows) != 30:
        raise SystemExit(f"expected 30 formal rows, found {len(rows)} in {MASTER_PATH}")
    for row in rows:
        row["best_val_acc_pct"] = float(row["best_val_acc_pct"])
        row["best_val_f1_pct"] = float(row["best_val_f1_pct"])
        row["seed"] = int(row["seed"])

    r4_audit = _audit_r4_sources()

    means: list[dict] = []
    by_variant: dict[str, list[dict]] = defaultdict(list)
    by_key = {}
    for row in rows:
        by_variant[row["variant"]].append(row)
        by_key[(row["variant"], row["dataset"], row["seed"])] = row
    for variant in VARIANTS:
        subset = by_variant[variant]
        acc_mean, acc_std = _mean_std([row["best_val_acc_pct"] for row in subset])
        f1_mean, f1_std = _mean_std([row["best_val_f1_pct"] for row in subset])
        means.append({
            "variant": variant,
            "n": len(subset),
            "acc_mean_pct": acc_mean,
            "acc_std_pct": acc_std,
            "f1_mean_pct": f1_mean,
            "f1_std_pct": f1_std,
        })
        for dataset in DATASETS:
            drows = [row for row in subset if row["dataset"] == dataset]
            da, dstd = _mean_std([row["best_val_acc_pct"] for row in drows])
            df, fstd = _mean_std([row["best_val_f1_pct"] for row in drows])
            means.append({
                "variant": f"{variant}:{dataset}", "n": len(drows),
                "acc_mean_pct": da, "acc_std_pct": dstd,
                "f1_mean_pct": df, "f1_std_pct": fstd,
            })
    _write_csv(
        OUT_ROOT / "summary" / "R5_0_MEANS.csv", means,
        ["variant", "n", "acc_mean_pct", "acc_std_pct", "f1_mean_pct", "f1_std_pct"],
    )

    baselines = _load_baselines()
    paired: list[dict] = []
    for variant in VARIANTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                current = by_key[(variant, dataset, seed)]
                m0 = by_key[("M0-MEAN", dataset, seed)]
                a0 = baselines[("A0", dataset, seed)]
                dip = baselines[("DiP", dataset, seed)]
                paired.append({
                    "variant": variant, "dataset": dataset, "seed": seed,
                    "acc_pct": current["best_val_acc_pct"], "f1_pct": current["best_val_f1_pct"],
                    "delta_acc_vs_M0": current["best_val_acc_pct"] - m0["best_val_acc_pct"],
                    "delta_f1_vs_M0": current["best_val_f1_pct"] - m0["best_val_f1_pct"],
                    "delta_acc_vs_A0": current["best_val_acc_pct"] - a0["acc"],
                    "delta_f1_vs_A0": current["best_val_f1_pct"] - a0["f1"],
                    "delta_acc_vs_DiP": current["best_val_acc_pct"] - dip["acc"],
                    "delta_f1_vs_DiP": current["best_val_f1_pct"] - dip["f1"],
                })
    _write_csv(
        OUT_ROOT / "summary" / "R5_0_PAIRED_DELTAS.csv", paired,
        [
            "variant", "dataset", "seed", "acc_pct", "f1_pct",
            "delta_acc_vs_M0", "delta_f1_vs_M0", "delta_acc_vs_A0", "delta_f1_vs_A0",
            "delta_acc_vs_DiP", "delta_f1_vs_DiP",
        ],
    )

    diagnostic_keys = [
        "local_message_ratio", "global_local_ratio", "global_scale",
        "slot_entropy", "effective_active_slots", "active_slots",
        "slot_usage_0", "slot_usage_1", "slot_usage_2", "slot_usage_3",
    ]
    diagnostics: list[dict] = []
    for variant in VARIANTS:
        drows = by_variant[variant]
        out = {"variant": variant, "n": len(drows)}
        for metric in diagnostic_keys:
            values = []
            for row in drows:
                keys = [key for key in row if key.startswith("scope_") and key.endswith(f"_{metric}")]
                for key in keys:
                    if row.get(key, "") not in ("", None):
                        values.append(float(row[key]))
            out[metric] = statistics.mean(values) if values else ""
        diagnostics.append(out)
    _write_csv(
        OUT_ROOT / "summary" / "R5_0_DIAGNOSTICS.csv", diagnostics,
        ["variant", "n", *diagnostic_keys],
    )

    def comparison(left: str, right: str) -> dict:
        lhs = by_variant[left]
        rhs = by_variant[right]
        lhs_acc = statistics.mean(row["best_val_acc_pct"] for row in lhs)
        rhs_acc = statistics.mean(row["best_val_acc_pct"] for row in rhs)
        lhs_f1 = statistics.mean(row["best_val_f1_pct"] for row in lhs)
        rhs_f1 = statistics.mean(row["best_val_f1_pct"] for row in rhs)
        deltas = [
            by_key[(left, dataset, seed)]["best_val_acc_pct"]
            - by_key[(right, dataset, seed)]["best_val_acc_pct"]
            for dataset in DATASETS for seed in SEEDS
        ]
        dataset_deltas = {
            dataset: statistics.mean(
                by_key[(left, dataset, seed)]["best_val_acc_pct"]
                - by_key[(right, dataset, seed)]["best_val_acc_pct"]
                for seed in SEEDS
            )
            for dataset in DATASETS
        }
        return {
            "left": left, "right": right,
            "delta_acc_mean_pct": lhs_acc - rhs_acc,
            "delta_f1_mean_pct": lhs_f1 - rhs_f1,
            "delta_acc_by_dataset_pct": dataset_deltas,
            "paired_acc_nonnegative": sum(delta >= 0 for delta in deltas),
            "paired_acc_deltas_pct": deltas,
        }

    comparisons = {
        "normalization": comparison("M1-MEAN-LN", "M0-MEAN"),
        "global_mean": comparison("M2-MEAN-G4", "M0-MEAN"),
        "global_sym": comparison("S1-SYM-G4", "S0-SYM"),
        "combined": comparison("S1-SYM-G4", "M0-MEAN"),
    }

    # This is an audit-screen verdict, not a claim of universal superiority.
    # GREEN requires positive overall Acc/F1, positive Grocery Acc, and at
    # least four of six paired Acc deltas non-negative.
    for item in comparisons.values():
        item["green"] = bool(
            item["delta_acc_mean_pct"] > 0
            and item["delta_f1_mean_pct"] > 0
            and item["delta_acc_by_dataset_pct"]["Grocery"] > 0
            and item["paired_acc_nonnegative"] >= 4
        )
    if comparisons["combined"]["green"]:
        recommendation = "PROCEED_TO_OSLM"
    elif not comparisons["global_mean"]["green"] and not comparisons["global_sym"]["green"]:
        recommendation = "GLOBAL_HYPOTHESIS_NOT_SUPPORTED"
    else:
        recommendation = "REVISE_LOCAL_CORE"

    git_commit = rows[0].get("git_commit")
    decision = {
        "protocol": {
            "formal_runs": len(rows), "datasets": DATASETS, "seeds": SEEDS,
            "epochs": 300, "patience": 30, "evaluate_test": False,
            "checkpoint": "best validation accuracy", "means": "six dataset-seed runs",
            "acc_f1_units": "percent",
        },
        "git_commit": git_commit,
        "r4_audit": r4_audit,
        "means": means,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "recommendation": recommendation,
    }
    (OUT_ROOT / "summary" / "R5_0_DECISION.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"recommendation": recommendation, "comparisons": comparisons}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
