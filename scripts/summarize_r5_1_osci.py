"""Summarize the validation-only R5-1 OSCI matrix and apply its preregistered rules."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs" / "r5_1_osci"
FORMAL_ROOT = OUT_ROOT / "formal"
DATASETS = ["Movies", "Toys", "Grocery"]
SEEDS = [42, 43, 44]
VARIANTS = ["C0-DIAG", "C1-DUP", "C2-SRC-STATIC", "C3-OSCI"]
BASELINE_PATH = PROJECT_ROOT / "outputs/r2d29/g0_reference/runs.csv"


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
    }


def _load_master() -> list[dict]:
    path = OUT_ROOT / "summary/R5_1_MASTER_TABLE.csv"
    rows = _read_csv(path)
    expected = {(variant, dataset, seed) for variant in VARIANTS for dataset in DATASETS for seed in SEEDS}
    actual = {(row["variant"], row["dataset"], int(row["seed"])) for row in rows}
    if len(rows) != 36 or actual != expected:
        raise SystemExit(f"expected exactly 36 complete validation rows, got {len(rows)}")
    for row in rows:
        if str(row.get("evaluate_test", "False")).lower() not in {"false", "0"}:
            raise SystemExit("R5-1 summary found a row with evaluate_test=true")
        row["seed"] = int(row["seed"])
        row["best_val_acc_pct"] = float(row["best_val_acc_pct"])
        row["best_val_f1_pct"] = float(row["best_val_f1_pct"])
        if not math.isfinite(row["best_val_acc_pct"]) or not math.isfinite(row["best_val_f1_pct"]):
            raise SystemExit("non-finite validation metric in R5-1 master table")
    return rows


def _load_baselines() -> dict[tuple[str, str, int], dict[str, float]]:
    rows = _read_csv(BASELINE_PATH)
    result: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in rows:
        if row["model"] not in {"biaxis_final", "dip"}:
            continue
        if row["dataset"] not in DATASETS or int(row["seed"]) not in SEEDS:
            continue
        label = "A0" if row["model"] == "biaxis_final" else "DiP"
        result[(label, row["dataset"], int(row["seed"]))] = {
            "acc": float(row["val_acc"]), "f1": float(row["val_f1"]),
        }
    expected = {(label, dataset, seed) for label in ("A0", "DiP") for dataset in DATASETS for seed in SEEDS}
    if set(result) != expected:
        raise SystemExit("A0/DiP unified-protocol validation baselines are incomplete")
    return result


def _comparison(
    left: str,
    right: str,
    by_key: dict[tuple[str, str, int], dict],
) -> dict:
    acc_deltas = [
        by_key[(left, dataset, seed)]["best_val_acc_pct"]
        - by_key[(right, dataset, seed)]["best_val_acc_pct"]
        for dataset in DATASETS for seed in SEEDS
    ]
    f1_deltas = [
        by_key[(left, dataset, seed)]["best_val_f1_pct"]
        - by_key[(right, dataset, seed)]["best_val_f1_pct"]
        for dataset in DATASETS for seed in SEEDS
    ]
    return {
        "left": left,
        "right": right,
        "acc_delta_pp": _stats(acc_deltas),
        "f1_delta_pp": _stats(f1_deltas),
        "acc_nonnegative_count": sum(delta >= 0.0 for delta in acc_deltas),
        "acc_positive_count": sum(delta > 0.0 for delta in acc_deltas),
        "f1_nonnegative_count": sum(delta >= 0.0 for delta in f1_deltas),
        "f1_positive_count": sum(delta > 0.0 for delta in f1_deltas),
        "acc_by_dataset": {
            dataset: _stats([
                by_key[(left, dataset, seed)]["best_val_acc_pct"]
                - by_key[(right, dataset, seed)]["best_val_acc_pct"]
                for seed in SEEDS
            ])
            for dataset in DATASETS
        },
        "f1_by_dataset": {
            dataset: _stats([
                by_key[(left, dataset, seed)]["best_val_f1_pct"]
                - by_key[(right, dataset, seed)]["best_val_f1_pct"]
                for seed in SEEDS
            ])
            for dataset in DATASETS
        },
    }


def _comparison_to_baseline(
    variant: str,
    label: str,
    master_by_key: dict[tuple[str, str, int], dict],
    baselines: dict[tuple[str, str, int], dict[str, float]],
) -> dict:
    acc = [
        master_by_key[(variant, dataset, seed)]["best_val_acc_pct"]
        - baselines[(label, dataset, seed)]["acc"]
        for dataset in DATASETS for seed in SEEDS
    ]
    f1 = [
        master_by_key[(variant, dataset, seed)]["best_val_f1_pct"]
        - baselines[(label, dataset, seed)]["f1"]
        for dataset in DATASETS for seed in SEEDS
    ]
    return {
        "left": variant, "right": label,
        "acc_delta_pp": _stats(acc), "f1_delta_pp": _stats(f1),
        "acc_nonnegative_count": sum(delta >= 0.0 for delta in acc),
        "f1_nonnegative_count": sum(delta >= 0.0 for delta in f1),
        "acc_by_dataset": {
            dataset: _stats([
                master_by_key[(variant, dataset, seed)]["best_val_acc_pct"]
                - baselines[(label, dataset, seed)]["acc"]
                for seed in SEEDS
            ])
            for dataset in DATASETS
        },
        "f1_by_dataset": {
            dataset: _stats([
                master_by_key[(variant, dataset, seed)]["best_val_f1_pct"]
                - baselines[(label, dataset, seed)]["f1"]
                for seed in SEEDS
            ])
            for dataset in DATASETS
        },
    }


def _passes_threshold(comparison: dict, acc_threshold: float, f1_threshold: float) -> bool:
    acc = comparison["acc_delta_pp"]["mean"]
    f1 = comparison["f1_delta_pp"]["mean"]
    return bool(
        (acc >= acc_threshold and f1 >= 0.0)
        or (f1 >= f1_threshold and acc >= 0.0)
    )


def _write_paired_deltas(rows: list[dict], baselines: dict[tuple[str, str, int], dict[str, float]]) -> None:
    by_key = {(row["variant"], row["dataset"], row["seed"]): row for row in rows}
    output = []
    references = VARIANTS + ["A0", "DiP"]
    for variant in VARIANTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                current = by_key[(variant, dataset, seed)]
                item = {
                    "variant": variant, "dataset": dataset, "seed": seed,
                    "acc_pct": current["best_val_acc_pct"],
                    "f1_pct": current["best_val_f1_pct"],
                }
                for reference in references:
                    if reference in VARIANTS:
                        ref_acc = by_key[(reference, dataset, seed)]["best_val_acc_pct"]
                        ref_f1 = by_key[(reference, dataset, seed)]["best_val_f1_pct"]
                    else:
                        ref_acc = baselines[(reference, dataset, seed)]["acc"]
                        ref_f1 = baselines[(reference, dataset, seed)]["f1"]
                    item[f"delta_acc_vs_{reference}_pp"] = current["best_val_acc_pct"] - ref_acc
                    item[f"delta_f1_vs_{reference}_pp"] = current["best_val_f1_pct"] - ref_f1
                output.append(item)
    fields = ["variant", "dataset", "seed", "acc_pct", "f1_pct"]
    for reference in references:
        fields.extend([f"delta_acc_vs_{reference}_pp", f"delta_f1_vs_{reference}_pp"])
    _write_csv(OUT_ROOT / "summary/R5_1_PAIRED_DELTAS.csv", output, fields)


def _write_diagnostics(rows: list[dict]) -> None:
    pattern_rows = []
    aggregate: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for key, raw_value in row.items():
            if not key.startswith("scope_") or "cross_" not in key or raw_value in (None, ""):
                continue
            pieces = key.split("_")
            # scope_l{layer}_{factor}_{metric pieces...}
            layer = pieces[1][1:]
            factor = pieces[2]
            metric = "_".join(pieces[3:])
            value = float(raw_value)
            record = {
                "record_type": "best_checkpoint",
                "variant": row["variant"], "dataset": row["dataset"], "seed": row["seed"],
                "layer": layer, "factor": factor, "metric": metric, "value": value,
            }
            pattern_rows.append(record)
            aggregate[(row["variant"], f"l{layer}_{factor}", metric)].append(value)
    for (variant, location, metric), values in sorted(aggregate.items()):
        layer, factor = location.split("_", 1)
        pattern_rows.append({
            "record_type": "variant_overall_mean",
            "variant": variant, "dataset": "ALL", "seed": "ALL",
            "layer": layer[1:], "factor": factor, "metric": metric,
            "value": statistics.mean(values),
        })
    _write_csv(
        OUT_ROOT / "summary/R5_1_DIAGNOSTICS.csv", pattern_rows,
        ["record_type", "variant", "dataset", "seed", "layer", "factor", "metric", "value"],
    )


def main() -> None:
    rows = _load_master()
    baselines = _load_baselines()
    by_key = {(row["variant"], row["dataset"], row["seed"]): row for row in rows}
    _write_paired_deltas(rows, baselines)
    _write_diagnostics(rows)

    comparisons = {}
    for left, right in (("C3-OSCI", "C0-DIAG"), ("C3-OSCI", "C1-DUP"), ("C3-OSCI", "C2-SRC-STATIC"), ("C2-SRC-STATIC", "C0-DIAG"), ("C2-SRC-STATIC", "C1-DUP"), ("C2-SRC-STATIC", "C3-OSCI"), ("C1-DUP", "C0-DIAG"), ("C1-DUP", "C3-OSCI")):
        comparisons[f"{left}_vs_{right}"] = _comparison(left, right, by_key)
    for variant in VARIANTS:
        comparisons[f"{variant}_vs_A0"] = _comparison_to_baseline(variant, "A0", by_key, baselines)
        comparisons[f"{variant}_vs_DiP"] = _comparison_to_baseline(variant, "DiP", by_key, baselines)

    c3_c0 = comparisons["C3-OSCI_vs_C0-DIAG"]
    c3_c1 = comparisons["C3-OSCI_vs_C1-DUP"]
    c3_c2 = comparisons["C3-OSCI_vs_C2-SRC-STATIC"]
    c2_c0 = comparisons["C2-SRC-STATIC_vs_C0-DIAG"]
    c2_c1 = comparisons["C2-SRC-STATIC_vs_C1-DUP"]
    c2_c3 = comparisons["C2-SRC-STATIC_vs_C3-OSCI"]
    c3_vs_c0_pass = _passes_threshold(c3_c0, 0.30, 0.40)
    c3_vs_c1_pass = _passes_threshold(c3_c1, 0.15, 0.25)
    c3_vs_c2_pass = _passes_threshold(c3_c2, 0.15, 0.25)
    c3_acc_coverage = c3_c0["acc_nonnegative_count"] >= 6
    c3_dataset_floor = min(
        c3_c0["acc_by_dataset"][dataset]["mean"] for dataset in DATASETS
    ) >= -0.35
    osci_supported = bool(
        c3_vs_c0_pass and c3_vs_c1_pass and c3_vs_c2_pass
        and c3_acc_coverage and c3_dataset_floor
    )
    c2_beats_c3 = bool(
        (c2_c3["acc_delta_pp"]["mean"] > 0.0 and c2_c3["f1_delta_pp"]["mean"] >= 0.0)
        or (c2_c3["f1_delta_pp"]["mean"] > 0.0 and c2_c3["acc_delta_pp"]["mean"] >= 0.0)
    )
    static_supported = bool(
        c2_beats_c3 and _passes_threshold(c2_c0, 0.30, 0.40)
        and _passes_threshold(c2_c1, 0.15, 0.25)
    )
    c1_c3 = comparisons["C1-DUP_vs_C3-OSCI"]
    generic_capacity_signal = bool(
        c1_c3["acc_delta_pp"]["mean"] >= 0.0
        and c1_c3["f1_delta_pp"]["mean"] >= 0.0
    )
    all_not_superior_c0 = all(
        comparisons[f"{variant}_vs_C0-DIAG"]["acc_delta_pp"]["mean"] <= 0.0
        and comparisons[f"{variant}_vs_C0-DIAG"]["f1_delta_pp"]["mean"] <= 0.0
        for variant in ("C1-DUP", "C2-SRC-STATIC", "C3-OSCI")
    )

    if osci_supported:
        recommendation = "PROCEED_WITH_OSCI_INTERVENTION_AUDIT"
    elif static_supported:
        recommendation = "PROCEED_WITH_STATIC_SOURCE_INTEGRATION"
    elif generic_capacity_signal:
        recommendation = "STOP_CROSS_FACTOR_CLAIM_GENERIC_CAPACITY"
    elif all_not_superior_c0:
        recommendation = "STOP_SOURCE_INTEGRATION"
    else:
        recommendation = "AMBIGUOUS_NO_GO"

    decision = {
        "protocol": {
            "formal_runs": len(rows), "variants": VARIANTS,
            "datasets": DATASETS, "seeds": SEEDS,
            "epochs": 300, "patience": 30,
            "evaluate_test": False,
            "checkpoint": "best validation accuracy",
            "macro_f1_checkpoint": "same best validation accuracy checkpoint",
            "training": "full-graph NC",
            "cross_scale_init": 0.05,
        },
        "baseline_source": str(BASELINE_PATH.relative_to(PROJECT_ROOT)),
        "baseline_protocol": "existing unified validation-only A0/DiP rows; no retraining and no Test access",
        "means": {
            variant: {
                "overall": {
                    "acc_pct": _stats([by_key[(variant, dataset, seed)]["best_val_acc_pct"] for dataset in DATASETS for seed in SEEDS]),
                    "f1_pct": _stats([by_key[(variant, dataset, seed)]["best_val_f1_pct"] for dataset in DATASETS for seed in SEEDS]),
                },
                "by_dataset": {
                    dataset: {
                        "acc_pct": _stats([by_key[(variant, dataset, seed)]["best_val_acc_pct"] for seed in SEEDS]),
                        "f1_pct": _stats([by_key[(variant, dataset, seed)]["best_val_f1_pct"] for seed in SEEDS]),
                    }
                    for dataset in DATASETS
                },
            }
            for variant in VARIANTS
        },
        "paired_comparisons": comparisons,
        "decision_rules": {
            "C3_VS_C0": {
                "value": c3_vs_c0_pass,
                "acc_or_f1_threshold": "Acc >= +0.30pp OR F1 >= +0.40pp; other metric >= 0",
                "acc_nonnegative_at_least_6_of_9": c3_acc_coverage,
                "dataset_acc_floor_at_least_-0.35pp": c3_dataset_floor,
            },
            "C3_VS_C1": {"value": c3_vs_c1_pass, "threshold": "Acc >= +0.15pp OR F1 >= +0.25pp; other metric >= 0"},
            "C3_VS_C2": {"value": c3_vs_c2_pass, "threshold": "Acc >= +0.15pp OR F1 >= +0.25pp; other metric >= 0"},
            "OSCI_SUPPORTED": {"value": osci_supported},
            "C2_BEATS_C3_AND_PASSES_C0_C1": {"value": static_supported, "c2_beats_c3": c2_beats_c3},
            "STATIC_SOURCE_INTEGRATION_SUPPORTED": {"value": static_supported},
            "C1_GENERIC_CAPACITY_SIGNAL": {"value": generic_capacity_signal},
            "ALL_MODELS_NOT_SUPERIOR_TO_C0": {"value": all_not_superior_c0},
        },
        "interventions_triggered": bool(osci_supported),
        "interventions_condition": "Only run the nine C3 best-checkpoint interventions when OSCI_SUPPORTED=true",
        "recommendation": recommendation,
    }
    path = OUT_ROOT / "summary/R5_1_DECISION.json"
    path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "OSCI_SUPPORTED": osci_supported,
        "STATIC_SOURCE_INTEGRATION_SUPPORTED": static_supported,
        "C1_GENERIC_CAPACITY_SIGNAL": generic_capacity_signal,
        "recommendation": recommendation,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
