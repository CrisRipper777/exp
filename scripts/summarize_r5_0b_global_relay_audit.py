"""Compute R5-0b paired tables and the pre-registered route decision."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs" / "r5_0b_global_relay_audit"
DATASETS = ["Movies", "Toys", "Grocery"]
SEEDS = [42, 43, 44]
VARIANTS = ["B0-MEAN", "B1-SYM", "B2-FSEP4", "B3-FSEP1", "B4-UNIFORM4"]


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
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
    }


def _load_baselines() -> dict[tuple[str, str, int], dict[str, float]]:
    rows = _read_csv(PROJECT_ROOT / "outputs/r2d29/g0_reference/runs.csv")
    result = {}
    for row in rows:
        if row["model"] not in {"biaxis_final", "dip"}:
            continue
        if row["dataset"] not in DATASETS or int(row["seed"]) not in SEEDS:
            continue
        label = "A0" if row["model"] == "biaxis_final" else "DiP"
        result[(label, row["dataset"], int(row["seed"]))] = {
            "acc": float(row["val_acc"]), "f1": float(row["val_f1"]),
        }
    return result


def main() -> None:
    master = _read_csv(OUT_ROOT / "summary/R5_0B_MASTER_TABLE.csv")
    if len(master) != 45:
        raise SystemExit(f"expected 45 formal rows, got {len(master)}")
    failure_dir = OUT_ROOT / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    (failure_dir / "formal_failures.jsonl").touch()
    for row in master:
        row["seed"] = int(row["seed"])
        row["best_val_acc_pct"] = float(row["best_val_acc_pct"])
        row["best_val_f1_pct"] = float(row["best_val_f1_pct"])
    by_key = {(row["variant"], row["dataset"], row["seed"]): row for row in master}
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in master:
        by_variant[row["variant"]].append(row)

    mean_rows = []
    for variant in VARIANTS:
        for scope, subset in [("overall", by_variant[variant])] + [
            (dataset, [row for row in by_variant[variant] if row["dataset"] == dataset])
            for dataset in DATASETS
        ]:
            acc = _stats([row["best_val_acc_pct"] for row in subset])
            f1 = _stats([row["best_val_f1_pct"] for row in subset])
            mean_rows.append({
                "variant": variant, "scope": scope, "n": acc["n"],
                "acc_mean_pct": acc["mean"], "acc_std_pct": acc["std"], "acc_median_pct": acc["median"],
                "f1_mean_pct": f1["mean"], "f1_std_pct": f1["std"], "f1_median_pct": f1["median"],
            })
    _write_csv(
        OUT_ROOT / "summary/R5_0B_MEANS.csv", mean_rows,
        ["variant", "scope", "n", "acc_mean_pct", "acc_std_pct", "acc_median_pct", "f1_mean_pct", "f1_std_pct", "f1_median_pct"],
    )

    baselines = _load_baselines()
    paired = []
    for variant in VARIANTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                current = by_key[(variant, dataset, seed)]
                row = {
                    "variant": variant, "dataset": dataset, "seed": seed,
                    "acc_pct": current["best_val_acc_pct"], "f1_pct": current["best_val_f1_pct"],
                }
                for control in VARIANTS:
                    ref = by_key[(control, dataset, seed)]
                    row[f"delta_acc_vs_{control}"] = current["best_val_acc_pct"] - ref["best_val_acc_pct"]
                    row[f"delta_f1_vs_{control}"] = current["best_val_f1_pct"] - ref["best_val_f1_pct"]
                for baseline in ("A0", "DiP"):
                    ref = baselines[(baseline, dataset, seed)]
                    row[f"delta_acc_vs_{baseline}"] = current["best_val_acc_pct"] - ref["acc"]
                    row[f"delta_f1_vs_{baseline}"] = current["best_val_f1_pct"] - ref["f1"]
                paired.append(row)
    pair_fields = ["variant", "dataset", "seed", "acc_pct", "f1_pct"]
    for control in VARIANTS:
        pair_fields.extend([f"delta_acc_vs_{control}", f"delta_f1_vs_{control}"])
    pair_fields.extend(["delta_acc_vs_A0", "delta_f1_vs_A0", "delta_acc_vs_DiP", "delta_f1_vs_DiP"])
    _write_csv(OUT_ROOT / "summary/R5_0B_PAIRED_DELTAS.csv", paired, pair_fields)

    def compare(left: str, right: str) -> dict:
        deltas_acc = [
            by_key[(left, dataset, seed)]["best_val_acc_pct"] - by_key[(right, dataset, seed)]["best_val_acc_pct"]
            for dataset in DATASETS for seed in SEEDS
        ]
        deltas_f1 = [
            by_key[(left, dataset, seed)]["best_val_f1_pct"] - by_key[(right, dataset, seed)]["best_val_f1_pct"]
            for dataset in DATASETS for seed in SEEDS
        ]
        return {
            "left": left, "right": right,
            "acc": _stats(deltas_acc), "f1": _stats(deltas_f1),
            "acc_nonnegative_count": sum(delta >= 0 for delta in deltas_acc),
            "acc_positive_count": sum(delta > 0 for delta in deltas_acc),
            "f1_nonnegative_count": sum(delta >= 0 for delta in deltas_f1),
            "f1_positive_count": sum(delta > 0 for delta in deltas_f1),
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

    comparisons = {
        "B2_vs_B0": compare("B2-FSEP4", "B0-MEAN"),
        "B2_vs_B1": compare("B2-FSEP4", "B1-SYM"),
        "B2_vs_B3": compare("B2-FSEP4", "B3-FSEP1"),
        "B2_vs_B4": compare("B2-FSEP4", "B4-UNIFORM4"),
        "B3_vs_B1": compare("B3-FSEP1", "B1-SYM"),
        "B4_vs_B1": compare("B4-UNIFORM4", "B1-SYM"),
    }

    interventions = _read_csv(OUT_ROOT / "summary/R5_0B_INTERVENTIONS.csv")
    for row in interventions:
        row["val_acc_pct"] = float(row["val_acc_pct"])
        row["val_f1_pct"] = float(row["val_f1_pct"])
        row["delta_acc_vs_I0_pp"] = float(row["delta_acc_vs_I0_pp"])
        row["delta_f1_vs_I0_pp"] = float(row["delta_f1_vs_I0_pp"])
    intervention_summary = {}
    for name in [row["intervention"] for row in interventions if row["intervention"] != "I0-NORMAL"]:
        subset = [row for row in interventions if row["intervention"] == name]
        acc = _stats([row["delta_acc_vs_I0_pp"] for row in subset])
        f1 = _stats([row["delta_f1_vs_I0_pp"] for row in subset])
        intervention_summary[name] = {
            "acc_delta": acc, "f1_delta": f1,
            "acc_negative_count": sum(row["delta_acc_vs_I0_pp"] < 0 for row in subset),
            "f1_negative_count": sum(row["delta_f1_vs_I0_pp"] < 0 for row in subset),
            "acc_by_dataset": {
                dataset: _stats([
                    row["delta_acc_vs_I0_pp"] for row in subset if row["dataset"] == dataset
                ])
                for dataset in DATASETS
            },
            "f1_by_dataset": {
                dataset: _stats([
                    row["delta_f1_vs_I0_pp"] for row in subset if row["dataset"] == dataset
                ])
                for dataset in DATASETS
            },
        }
    _write_csv(
        OUT_ROOT / "summary/R5_0B_INTERVENTION_MEANS.csv",
        [
            {"intervention": name, "acc_delta_mean_pp": values["acc_delta"]["mean"], "acc_delta_std_pp": values["acc_delta"]["std"], "acc_delta_median_pp": values["acc_delta"]["median"], "f1_delta_mean_pp": values["f1_delta"]["mean"], "f1_delta_std_pp": values["f1_delta"]["std"], "f1_delta_median_pp": values["f1_delta"]["median"], "acc_negative_count": values["acc_negative_count"], "f1_negative_count": values["f1_negative_count"]}
            for name, values in intervention_summary.items()
        ],
        ["intervention", "acc_delta_mean_pp", "acc_delta_std_pp", "acc_delta_median_pp", "f1_delta_mean_pp", "f1_delta_std_pp", "f1_delta_median_pp", "acc_negative_count", "f1_negative_count"],
    )

    b2_b1 = comparisons["B2_vs_B1"]
    global_robust = bool(
        b2_b1["acc"]["mean"] >= 0.20
        and b2_b1["f1"]["mean"] >= 0.20
        and b2_b1["acc_nonnegative_count"] >= 5
        and sum(b2_b1["acc_by_dataset"][dataset]["mean"] > 0 for dataset in DATASETS) >= 2
        and min(b2_b1["acc_by_dataset"][dataset]["mean"] for dataset in DATASETS) >= -0.30
    )

    b2_control = "B3-FSEP1" if statistics.mean(row["best_val_acc_pct"] for row in by_variant["B3-FSEP1"]) >= statistics.mean(row["best_val_acc_pct"] for row in by_variant["B4-UNIFORM4"]) else "B4-UNIFORM4"
    selected = comparisons["B2_vs_B3"] if b2_control == "B3-FSEP1" else comparisons["B2_vs_B4"]
    path_a = bool(selected["acc"]["mean"] >= 0.15 and selected["f1"]["mean"] >= 0 and selected["acc_positive_count"] >= 5)
    path_b = bool(selected["f1"]["mean"] >= 0.25 and selected["acc"]["mean"] >= 0 and selected["f1_positive_count"] >= 5)
    intervention_signal = any(
        intervention_summary[name]["acc_delta"]["mean"] <= -0.15
        or intervention_summary[name]["f1_delta"]["mean"] <= -0.25
        for name in ("I5-UNIFORM-ASSIGN", "I6-SHUFFLE-COLLECT", "I7-SHUFFLE-DISPATCH")
    )
    multislot_supported = bool((path_a or path_b) and intervention_signal)
    simpler_positive = (
        comparisons["B3_vs_B1"]["acc"]["mean"] > 0
        or comparisons["B4_vs_B1"]["acc"]["mean"] > 0
    )
    summary_sufficient = bool(
        global_robust
        and not multislot_supported
        and min(abs(comparisons["B2_vs_B3"]["acc"]["mean"]), abs(comparisons["B2_vs_B4"]["acc"]["mean"])) <= 0.10
        and simpler_positive
    )

    factor_contribution = {}
    for factor, name in (("C", "I2-ZERO-C"), ("Pt", "I3-ZERO-PT"), ("Pv", "I4-ZERO-PV")):
        values = intervention_summary[name]
        factor_contribution[factor] = {
            "intervention": name,
            "acc_delta": values["acc_delta"], "f1_delta": values["f1_delta"],
            "acc_negative_count": values["acc_negative_count"],
            "f1_negative_count": values["f1_negative_count"],
            "datasets_with_negative_acc_mean": sum(values["acc_by_dataset"][dataset]["mean"] < 0 for dataset in DATASETS),
            "supported_by_acc": bool(values["acc_negative_count"] >= 5 or sum(values["acc_by_dataset"][dataset]["mean"] < 0 for dataset in DATASETS) >= 2),
        }

    if not global_robust:
        recommendation = "STOP_GLOBAL_ROUTE"
    elif multislot_supported:
        recommendation = "PROCEED_TO_MULTI_SLOT_OSLM"
    elif summary_sufficient:
        recommendation = "PROCEED_TO_SUMMARY_NULL_DISPATCH"
    else:
        recommendation = "AMBIGUOUS_REPEAT_REQUIRED"

    geometry_path = OUT_ROOT / "summary/R5_0B_SLOT_GEOMETRY.csv"
    geometry_rows = _read_csv(geometry_path)
    geometry_summary = {}
    for variant in ("B2-FSEP4", "B3-FSEP1", "B4-UNIFORM4"):
        subset = [row for row in geometry_rows if row["variant"] == variant]
        geometry_summary[variant] = {
            metric: statistics.mean(float(row[metric]) for row in subset)
            for metric in (
                "assignment_entropy_mean", "normalized_entropy_mean", "assignment_std_mean",
                "assignment_stable_rank", "slot_cosine_mean_offdiag", "slot_distance_mean_offdiag",
                "global_output_variance", "global_local_norm_ratio", "global_scale", "actual_injection_ratio",
            )
        }

    implementation_commits = sorted({row["implementation_commit"] for row in master})
    decision = {
        "protocol": {
            "formal_runs": len(master), "datasets": DATASETS, "seeds": SEEDS,
            "epochs": 300, "patience": 30, "evaluate_test": False,
            "checkpoint": "best validation accuracy", "acc_f1_units": "percent",
        },
        "implementation_commits": implementation_commits,
        "means": mean_rows,
        "paired_comparisons": comparisons,
        "intervention_summary": intervention_summary,
        "factor_contribution": factor_contribution,
        "slot_geometry_summary": geometry_summary,
        "decision_rules": {
            "GLOBAL_ROBUST": {
                "value": global_robust,
                "requirements": "B2-B1 Acc/F1 >= 0.20pp, >=5/9 nonnegative Acc, >=2 dataset Acc means positive, worst dataset >= -0.30pp",
            },
            "MULTISLOT_ROUTING_SUPPORTED": {
                "value": multislot_supported,
                "selected_control": b2_control,
                "path_a": path_a, "path_b": path_b, "intervention_signal": intervention_signal,
            },
            "GLOBAL_SUMMARY_SUFFICIENT": {
                "value": summary_sufficient,
                "simpler_control_positive": simpler_positive,
            },
        },
        "recommendation": recommendation,
    }
    (OUT_ROOT / "summary/R5_0B_DECISION.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "GLOBAL_ROBUST": global_robust,
        "MULTISLOT_ROUTING_SUPPORTED": multislot_supported,
        "GLOBAL_SUMMARY_SUFFICIENT": summary_sufficient,
        "recommendation": recommendation,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
