"""Main-story ablation aggregation + report (plan §24 Prompt 8).

Sources (all matched seeds 42/43/44):
    Full                    outputs/final_nc_benchmark/main (biaxis_final)
    no_factor_axis          outputs/final_nc_ablation/main_story
    no_relation_axis        outputs/final_nc_ablation/main_story
    no_adaptive_allocation  outputs/final_nc_ablation/main_story
    shared_operator         P3 O0 (outputs/p3/operator/<ds>/O0)
    no_cell_correction      P3 OADD (outputs/p3/operator/<ds>/OADD)

Outputs (outputs/final_nc_ablation/tables/):
    NC_ABLATION_PER_SEED.csv / NC_ABLATION_MAIN.csv / NC_ABLATION_REPORT.md

Report: Test Acc / Test Macro-F1 / Val Acc / Params; paired-seed deltas
Full − ablation (mean / population std / positive seeds). Paper claims
used as variant names (no P0/P1/P2/P3 module naming).
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "outputs" / "final_nc_ablation" / "tables"
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]

# paper name -> (root, model dir name, result subpath)
SOURCES = {
    "Full": (PROJECT_ROOT / "outputs" / "final_nc_benchmark" / "main", "biaxis_final", "hydra/results.json"),
    "w/o Semantic Factor Axis": (PROJECT_ROOT / "outputs" / "final_nc_ablation" / "main_story", "no_factor_axis", "hydra/results.json"),
    "w/o Structural Relation Axis": (PROJECT_ROOT / "outputs" / "final_nc_ablation" / "main_story", "no_relation_axis", "hydra/results.json"),
    "w/o Adaptive Allocation": (PROJECT_ROOT / "outputs" / "final_nc_ablation" / "main_story", "no_adaptive_allocation", "hydra/results.json"),
    "w/o Hierarchical Operator": (PROJECT_ROOT / "outputs" / "p3" / "operator", "O0", "hydra/results.json"),
    "w/o Cell-specific Correction": (PROJECT_ROOT / "outputs" / "p3" / "operator", "OADD", "hydra/results.json"),
}
ORDER = ["Full", "w/o Semantic Factor Axis", "w/o Structural Relation Axis",
         "w/o Adaptive Allocation", "w/o Hierarchical Operator", "w/o Cell-specific Correction"]


def _load(variant: str) -> dict:
    root, model_dir, subpath = SOURCES[variant]
    out: dict = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            res = root / dataset / model_dir / f"seed_{seed}" / subpath
            if not res.exists():
                continue
            with res.open(encoding="utf-8") as f:
                r = json.load(f)
            params = None
            info = root / dataset / model_dir / f"seed_{seed}" / "run_info.json"
            if info.exists():
                params = json.load(open(info))["params"]
            out[(dataset, seed)] = {
                "val_acc": float(r["val_acc"]["mean"]),
                "test_acc": float(r["test_acc"]["mean"]),
                "test_f1": float(r["test_macro_f1"]["mean"]),
                "params": params,
            }
    return out


def _mean_std(vals: list[float]) -> str:
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f}±{statistics.pstdev(vals):.4f}"


def main() -> None:
    runs = {variant: _load(variant) for variant in ORDER}
    TABLES.mkdir(parents=True, exist_ok=True)

    # per-seed CSV
    per_seed_rows = []
    for variant in ORDER:
        for (dataset, seed), r in sorted(runs[variant].items()):
            per_seed_rows.append({
                "dataset": dataset, "variant": variant, "seed": seed,
                "val_acc": f"{r['val_acc']:.6f}", "test_acc": f"{r['test_acc']:.6f}",
                "test_macro_f1": f"{r['test_f1']:.6f}", "params": r["params"] or "",
            })
    with (TABLES / "NC_ABLATION_PER_SEED.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "val_acc", "test_acc", "test_macro_f1", "params"])
        w.writeheader()
        w.writerows(per_seed_rows)

    def agg(variant, dataset, key):
        return [runs[variant][(dataset, s)][key] for s in SEEDS if (dataset, s) in runs[variant]]

    # main CSV
    main_rows = []
    for dataset in DATASETS:
        for variant in ORDER:
            main_rows.append({
                "dataset": dataset, "variant": variant,
                "val_acc": _mean_std(agg(variant, dataset, "val_acc")),
                "test_acc": _mean_std(agg(variant, dataset, "test_acc")),
                "test_macro_f1": _mean_std(agg(variant, dataset, "test_f1")),
                "params": runs[variant].get((dataset, 42), {}).get("params") or "",
            })
    with (TABLES / "NC_ABLATION_MAIN.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "variant", "val_acc", "test_acc", "test_macro_f1", "params"])
        w.writeheader()
        w.writerows(main_rows)

    # report
    lines: list[str] = []
    lines.append("# Bi-Axis Final — Main Story Ablation Report")
    lines.append("")
    lines.append(
        f"> 6 variants x 5 datasets x seeds 42/43/44. Full = biaxis_final benchmark runs; "
        f"w/o Hierarchical Operator = P3 O0; w/o Cell-specific Correction = P3 OADD "
        f"(matched seeds). mean ± population std; paper-claim naming (no stage names)."
    )
    lines.append("")

    for key, title in (("test_acc", "Test Accuracy"), ("test_f1", "Test Macro-F1"), ("val_acc", "Val Accuracy (decision)")):
        lines.append(f"## {title} (mean±population std)")
        lines.append("")
        lines.append("| Dataset | " + " | ".join(ORDER) + " |")
        lines.append("|---|" + "---:|" * len(ORDER))
        for dataset in DATASETS:
            cells = [_mean_std(agg(v, dataset, key)) for v in ORDER]
            lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Params (model+head, Movies)")
    lines.append("")
    lines.append("| Variant | Params |")
    lines.append("|---|--:|")
    for variant in ORDER:
        p = runs[variant].get(("Movies", 42), {}).get("params")
        lines.append(f"| {variant} | {p:,} |" if p else f"| {variant} | |")
    lines.append("")

    lines.append("## Paired-seed deltas (Test Acc, pp; Full − ablation)")
    lines.append("")
    lines.append("| Dataset | " + " | ".join(ORDER[1:]) + " |")
    lines.append("|---|" + "---|" * (len(ORDER) - 1))
    for dataset in DATASETS:
        cells = []
        for variant in ORDER[1:]:
            seeds = [s for s in SEEDS if (dataset, s) in runs["Full"] and (dataset, s) in runs[variant]]
            deltas = [100 * (runs["Full"][(dataset, s)]["test_acc"] - runs[variant][(dataset, s)]["test_acc"]) for s in seeds]
            if not deltas:
                cells.append("")
                continue
            pos = sum(1 for d in deltas if d > 0)
            cells.append(f"{statistics.mean(deltas):+.2f}±{statistics.pstdev(deltas):.2f} ({pos}/{len(seeds)})")
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Paired-seed deltas (Val Acc, pp; Full − ablation)")
    lines.append("")
    lines.append("| Dataset | " + " | ".join(ORDER[1:]) + " |")
    lines.append("|---|" + "---|" * (len(ORDER) - 1))
    for dataset in DATASETS:
        cells = []
        for variant in ORDER[1:]:
            seeds = [s for s in SEEDS if (dataset, s) in runs["Full"] and (dataset, s) in runs[variant]]
            deltas = [100 * (runs["Full"][(dataset, s)]["val_acc"] - runs[variant][(dataset, s)]["val_acc"]) for s in seeds]
            if not deltas:
                cells.append("")
                continue
            pos = sum(1 for d in deltas if d > 0)
            cells.append(f"{statistics.mean(deltas):+.2f}±{statistics.pstdev(deltas):.2f} ({pos}/{len(seeds)})")
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Claims answered (fill after analysis)")
    lines.append("")
    lines.append("- [ ] Semantic Factor Axis contributes (Full vs w/o Semantic Factor Axis)")
    lines.append("- [ ] Structural Relation Axis contributes (Full vs w/o Structural Relation Axis)")
    lines.append("- [ ] Adaptive Allocation contributes (Full vs w/o Adaptive Allocation)")
    lines.append("- [ ] Hierarchical Operator contributes (Full vs w/o Hierarchical Operator)")
    lines.append("- [ ] Cell-specific Correction contributes (Full vs w/o Cell-specific Correction)")
    lines.append("")
    lines.append("- Decision: **PENDING** (fill after analysis)")
    lines.append("")
    (TABLES / "NC_ABLATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[summarize] {sum(len(r) for r in runs.values())} runs -> {TABLES}", flush=True)


if __name__ == "__main__":
    main()
