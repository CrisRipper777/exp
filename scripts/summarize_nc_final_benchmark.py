"""Final NC benchmark aggregation + main report (plan §18/§20 Prompt 4).

Reads:
    - 8 baselines from outputs/baseline_nc (audit-passed, never re-run)
    - biaxis_final from outputs/final_nc_benchmark/main

Writes to outputs/final_nc_benchmark/tables/:
    nc_main_per_seed.csv    per (dataset, model, seed) val/test acc + test F1
    nc_main_table_acc.csv   Test Acc mean±population std per model x dataset
    nc_main_table_f1.csv    Test Macro-F1 mean±population std
    nc_main_rank.csv        ranks + best-baseline deltas + paired deltas
    NC_MAIN_REPORT.md       full markdown report

Statistics: mean ± population std (ddof=0) over seeds 42/43/44; paired-seed
deltas vs DiP and vs the per-dataset strongest baseline (matched seeds).
Rank computed on TEST accuracy for the paper table (val reported for
reference; architecture is already frozen).

Usage:
    python scripts/summarize_nc_final_benchmark.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "outputs" / "baseline_nc"
FINAL_ROOT = PROJECT_ROOT / "outputs" / "final_nc_benchmark" / "main"
TABLES = PROJECT_ROOT / "outputs" / "final_nc_benchmark" / "tables"

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
BASELINE_MODELS = ["mlp", "gcn", "sage", "mmgcn", "mgat", "dmgc", "dgf", "dip", "lgmrec"]
MODELS = BASELINE_MODELS + ["biaxis_final"]
SEEDS = [42, 43, 44]


def _load_runs(root: Path, model: str, result_subpath: str = "hydra/results.json",
               log_subpath: str = "hydra/main.log") -> dict:
    """{(dataset, seed): {'val_acc','test_acc','test_f1','params'}}"""
    out: dict = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            res_path = root / dataset / model / f"seed_{seed}" / result_subpath
            if not res_path.exists():
                continue
            with res_path.open(encoding="utf-8") as f:
                r = json.load(f)
            params = None
            log_path = root / dataset / model / f"seed_{seed}" / log_subpath
            if log_path.exists():
                match = re.search(r"model\+head params=(\d+)", log_path.read_text(encoding="utf-8"))
                if match:
                    params = int(match.group(1))
            out[(dataset, seed)] = {
                "val_acc": float(r["val_acc"]["mean"]),
                "test_acc": float(r["test_acc"]["mean"]),
                "test_f1": float(r["test_macro_f1"]["mean"]),
                "params": params,
            }
    return out


def _mean_std(values: list[float]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f}±{statistics.pstdev(values):.4f}"


def _fmt_rank_mark(rank: int) -> str:
    return {1: "①", 2: "②"}.get(rank, "")


def main() -> None:
    runs: dict[str, dict] = {}
    for model in BASELINE_MODELS:
        runs[model] = _load_runs(BASELINE_ROOT, model)
    runs["biaxis_final"] = _load_runs(FINAL_ROOT, "biaxis_final")

    TABLES.mkdir(parents=True, exist_ok=True)

    # ---- per-seed CSV ------------------------------------------------------
    per_seed_rows = []
    for model in MODELS:
        for (dataset, seed), r in sorted(runs[model].items()):
            per_seed_rows.append({
                "dataset": dataset, "model": model, "seed": seed,
                "val_acc": f"{r['val_acc']:.6f}", "test_acc": f"{r['test_acc']:.6f}",
                "test_macro_f1": f"{r['test_f1']:.6f}", "params": r["params"] or "",
            })
    with (TABLES / "nc_main_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "seed", "val_acc", "test_acc", "test_macro_f1", "params"])
        writer.writeheader()
        writer.writerows(per_seed_rows)

    # ---- aggregates ---------------------------------------------------------
    def agg(dataset: str, model: str, key: str) -> list[float]:
        return [runs[model][(dataset, s)][key] for s in SEEDS if (dataset, s) in runs[model]]

    agg_cells: dict = {}
    for dataset in DATASETS:
        for model in MODELS:
            agg_cells[(dataset, model)] = {
                "val": agg(dataset, model, "val_acc"),
                "test": agg(dataset, model, "test_acc"),
                "f1": agg(dataset, model, "test_f1"),
            }

    # per-dataset ranks — computed INDEPENDENTLY per metric (user decision
    # 2026-09-03): Test Acc ranks and Test F1 ranks are separate.
    ranks: dict = {}
    ranks_f1: dict = {}
    best_baseline: dict = {}
    for dataset in DATASETS:
        means = {m: statistics.mean(agg_cells[(dataset, m)]["test"]) for m in MODELS if agg_cells[(dataset, m)]["test"]}
        order = sorted(means, key=lambda m: -means[m])
        ranks[dataset] = {m: order.index(m) + 1 for m in order}
        f1_means = {m: statistics.mean(agg_cells[(dataset, m)]["f1"]) for m in MODELS if agg_cells[(dataset, m)]["f1"]}
        f1_order = sorted(f1_means, key=lambda m: -f1_means[m])
        ranks_f1[dataset] = {m: f1_order.index(m) + 1 for m in f1_order}
        baseline_means = {m: means[m] for m in BASELINE_MODELS if m in means}
        best_baseline[dataset] = max(baseline_means, key=baseline_means.get)

    # paired deltas (matched seeds) — reference only, no structure selection
    paired_rows = []
    for dataset in DATASETS:
        ours = runs["biaxis_final"]
        for ref_name, ref_runs in (("dip", runs["dip"]), (f"best_{best_baseline[dataset]}", runs[best_baseline[dataset]])):
            deltas = [ours[(dataset, s)]["test_acc"] - ref_runs[(dataset, s)]["test_acc"]
                      for s in SEEDS if (dataset, s) in ours and (dataset, s) in ref_runs]
            if not deltas:
                continue
            paired_rows.append({
                "dataset": dataset,
                "comparison": f"biaxis_final-{ref_name}",
                "metric": "test_acc",
                "mean_pp": f"{100 * statistics.mean(deltas):+.4f}",
                "std_pp": f"{100 * statistics.pstdev(deltas):.4f}",
                "positive_seeds": sum(1 for d in deltas if d > 0),
                "n_seeds": len(deltas),
            })
    with (TABLES / "nc_main_rank.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "comparison", "metric", "mean_pp", "std_pp", "positive_seeds", "n_seeds"])
        writer.writeheader()
        writer.writerows(paired_rows)

    # ---- main tables CSV -----------------------------------------------------
    def _table_csv(name: str, key: str) -> None:
        rows = []
        for dataset in DATASETS:
            row = {"dataset": dataset}
            for model in MODELS:
                row[model] = _mean_std(agg_cells[(dataset, model)][key]) if agg_cells[(dataset, model)][key] else ""
            rows.append(row)
        with (TABLES / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset"] + MODELS)
            writer.writeheader()
            writer.writerows(rows)

    _table_csv("nc_main_table_acc.csv", "test")
    _table_csv("nc_main_table_f1.csv", "f1")

    # ---- markdown report ----------------------------------------------------
    lines: list[str] = []
    lines.append("# Bi-Axis Final — NC Main Benchmark Report")
    lines.append("")
    lines.append(
        f"> {len(MODELS)} models ({len(BASELINE_MODELS)} baselines + biaxis_final) x 5 datasets x "
        f"seeds 42/43/44. Protocol: full-graph NC, 300 epochs, patience 30, AdamW, val-Acc "
        f"checkpoint, test once. mean ± population std. Provenance audit: "
        f"docs/Final_NC_Benchmark_Audit.md (baselines never re-run; final model runs "
        f"fresh via model=biaxis_final; lgmrec ported from OpenMAG)."
    )
    lines.append("")

    def _md_table(key: str, title: str, rank_dict: dict, mark: bool = True) -> None:
        lines.append(f"## {title}")
        lines.append("")
        header = "| Model | " + " | ".join(DATASETS) + " | Avg Rank |"
        lines.append(header)
        lines.append("|---|" + "---:|" * (len(DATASETS) + 1))
        for model in MODELS:
            cells = []
            rank_vals = []
            for dataset in DATASETS:
                vals = agg_cells[(dataset, model)][key]
                if not vals:
                    cells.append("")
                    continue
                cell_mark = _fmt_rank_mark(rank_dict[dataset].get(model, 99)) if mark else ""
                cells.append(f"{_mean_std(vals)}{cell_mark}")
                rank_vals.append(rank_dict[dataset].get(model, len(MODELS)))
            avg_rank = f"{statistics.mean(rank_vals):.2f}" if rank_vals else ""
            lines.append(f"| {model} | " + " | ".join(cells) + f" | {avg_rank} |")
        lines.append("")

    lines.append("> ① best, ② second best per dataset — ranks computed independently per metric.")
    lines.append("")
    _md_table("test", "Table 1 — Test Accuracy (mean±population std)", ranks)
    _md_table("f1", "Table 2 — Test Macro-F1 (mean±population std)", ranks_f1)

    lines.append("## Per-dataset summary")
    lines.append("")
    lines.append("| Dataset | best baseline | ours | ours − best (pp) | ours rank |")
    lines.append("|---|---:|---:|---:|---:|")
    for dataset in DATASETS:
        bb = best_baseline[dataset]
        ours_vals = agg_cells[(dataset, "biaxis_final")]["test"]
        bb_vals = agg_cells[(dataset, bb)]["test"]
        delta = 100 * (statistics.mean(ours_vals) - statistics.mean(bb_vals)) if ours_vals and bb_vals else float("nan")
        lines.append(
            f"| {dataset} | {bb} ({_mean_std(bb_vals)}) | {_mean_std(ours_vals)} | "
            f"{delta:+.2f} | {ranks[dataset].get('biaxis_final', '-')}/{len(MODELS)} |"
        )
    lines.append("")

    rank_vals = [ranks[d].get("biaxis_final", len(MODELS)) for d in DATASETS]
    rank_f1_vals = [ranks_f1[d].get("biaxis_final", len(MODELS)) for d in DATASETS]
    top1 = sum(1 for d in DATASETS if ranks[d].get("biaxis_final") == 1)
    top2 = sum(1 for d in DATASETS if ranks[d].get("biaxis_final", 99) <= 2)
    top1_f1 = sum(1 for d in DATASETS if ranks_f1[d].get("biaxis_final") == 1)
    top2_f1 = sum(1 for d in DATASETS if ranks_f1[d].get("biaxis_final", 99) <= 2)
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Average rank (Test Acc): **{statistics.mean(rank_vals):.2f}** / {len(MODELS)} — Top-1: {top1}/5, Top-2: {top2}/5")
    lines.append(f"- Average rank (Test Macro-F1): **{statistics.mean(rank_f1_vals):.2f}** / {len(MODELS)} — Top-1: {top1_f1}/5, Top-2: {top2_f1}/5")
    lines.append("")

    lines.append("## Paired-seed deltas (Test Acc, pp; reference only — no structure selection)")
    lines.append("")
    lines.append("| Dataset | comparison | mean Δ | std Δ | positive seeds |")
    lines.append("|---|" + "---:|" * 3 + "---:|")
    for row in paired_rows:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {row['mean_pp']} | {row['std_pp']} | "
            f"{row['positive_seeds']}/{row['n_seeds']} |"
        )
    lines.append("")

    lines.append("## Val Acc (decision protocol reference; mean±population std)")
    lines.append("")
    _md_table("val", "Val Accuracy", ranks, mark=False)

    lines.append("## Params (model+head, per run log)")
    lines.append("")
    lines.append("| Model | Movies | Grocery | ele-fashion |")
    lines.append("|---|---:|---:|---:|")
    for model in MODELS:
        cells = []
        for dataset in ("Movies", "Grocery", "ele-fashion"):
            p = runs[model].get((dataset, 42), {}).get("params")
            cells.append(f"{p:,}" if p else "")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("> Params vary with num_classes per dataset (heads). biaxis_final: M1/M2/Γ/operator + head.")
    lines.append("")

    (TABLES / "NC_MAIN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[summarize] {sum(len(r) for r in runs.values())} runs -> {TABLES}", flush=True)


if __name__ == "__main__":
    main()
