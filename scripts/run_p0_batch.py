"""P0 batch driver + aggregation.

Stages (plan §17) — one job per DATASET, all seeds handled inside each job:
    d1: screening  Movies-NC, ele-fashion-NC
    d2: NC confirm  Movies/Toys/Grocery/ele-fashion/Reddit-S
    d3: LP confirm  sports-copurchase/cloth-copurchase
    all: d1+d2+d3 (d1 jobs deduplicated)

Each job runs seeds 42/43/44 sequentially (MAGB split files are per-seed,
so every seed needs its own training pass). Jobs run in parallel across
GPUs (one worker per GPU). Aggregates into outputs/p0/p0_nc_summary.csv,
p0_lp_summary.csv, p0_conflict_summary.csv, p0_report.md plus an
old-vs-new protocol comparison against outputs/p0/old_protocol/ (when
present). A failed job never overwrites existing results.

Usage:
    python scripts/run_p0_batch.py --stage d1
    python scripts/run_p0_batch.py --stage d2 --gpus 0,1
    python scripts/run_p0_batch.py --stage all --include-test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs" / "p0"
OLD_PROTOCOL_ROOT = OUT_ROOT / "old_protocol"

NC_DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
LP_DATASETS = ["sports-copurchase", "cloth-copurchase"]
SEEDS = [42, 43, 44]
SEEDS_ARG = ",".join(str(seed) for seed in SEEDS)


def _jobs_for_stage(stage: str) -> list[tuple[str, str]]:
    if stage == "d1":
        return [("Movies", "nc"), ("ele-fashion", "nc")]
    if stage == "d2":
        return [(dataset, "nc") for dataset in NC_DATASETS]
    if stage == "d3":
        return [(dataset, "lp") for dataset in LP_DATASETS]
    if stage == "all":
        return _jobs_for_stage("d2") + _jobs_for_stage("d3")
    raise ValueError(f"unknown stage {stage!r}")


def _run_job(job: tuple[str, str], gpu_id: int, include_test: bool, force: bool, variant: str | None, model_overrides: str) -> None:
    dataset, task = job
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_p0.py"),
        "--dataset", dataset,
        "--task", task,
        "--seeds", SEEDS_ARG,
        "--device", "cuda:0",
        f"--out-root={OUT_ROOT}",
    ]
    if variant:
        cmd += ["--variant", variant]
    if model_overrides:
        cmd += ["--model-overrides", model_overrides]
    if include_test:
        cmd.append("--include-test")
    if force:
        cmd.append("--force")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    tag = f"[{gpu_id}] {dataset} {task}"
    if proc.returncode != 0:
        print(f"{tag} FAILED rc={proc.returncode}", flush=True)
        print(proc.stdout[-4000:], flush=True)
        print(proc.stderr[-4000:], flush=True)
    else:
        print(f"{tag} OK", flush=True)


def _load_summary(dataset: str, task: str, seed: int, variant: str | None = None) -> dict | None:
    seed_dir = f"seed_{seed}" + (f"_{variant}" if variant else "")
    path = OUT_ROOT / dataset / seed_dir / "summary.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fmt(value, digits: int = 4):
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return f"{value:.{digits}f}"


def _mean_std_str(values: list[float], digits: int = 4) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return _fmt(values[0], digits)
    return f"{statistics.mean(values):.{digits}f}±{statistics.pstdev(values):.{digits}f}"


def _rows_for_nc(summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        probe = {(row["factor"], row["mode"]): row for row in summary["probe"]}
        delta = {}
        for factor in ("C", "Pt", "Pv"):
            delta[factor] = probe[(factor, "graph")]["val_acc"] - probe[(factor, "local")]["val_acc"]
        rows.append({
            "dataset": summary["dataset"],
            "seed": summary["seed"],
            "common_sim": summary["factor_sanity"]["common_sim"],
            "private_sim": summary["factor_sanity"]["private_sim"],
            "effrank_c": summary["factor_sanity"]["effrank_c"],
            "effrank_pt": summary["factor_sanity"]["effrank_pt"],
            "effrank_pv": summary["factor_sanity"]["effrank_pv"],
            "rho_C_Pt": summary["edge_statistics"]["rho_C_Pt"],
            "rho_C_Pv": summary["edge_statistics"]["rho_C_Pv"],
            "rho_Pt_Pv": summary["edge_statistics"]["rho_Pt_Pv"],
            "jaccard_top10_C_Pt": summary["edge_statistics"]["jaccard_top10_C_Pt"],
            "jaccard_top10_C_Pv": summary["edge_statistics"]["jaccard_top10_C_Pv"],
            "jaccard_top10_Pt_Pv": summary["edge_statistics"]["jaccard_top10_Pt_Pv"],
            "jaccard_top20_C_Pt": summary["edge_statistics"]["jaccard_top20_C_Pt"],
            "jaccard_top20_Pt_Pv": summary["edge_statistics"]["jaccard_top20_Pt_Pv"],
            "mean_abs_gap_C_Pt": summary["edge_statistics"]["mean_abs_gap_C_Pt"],
            "mean_abs_gap_C_Pv": summary["edge_statistics"]["mean_abs_gap_C_Pv"],
            "delta_acc_C": delta["C"],
            "delta_acc_Pt": delta["Pt"],
            "delta_acc_Pv": delta["Pv"],
            "acc_local_C": probe[("C", "local")]["val_acc"],
            "acc_graph_C": probe[("C", "graph")]["val_acc"],
            "acc_local_Pt": probe[("Pt", "local")]["val_acc"],
            "acc_graph_Pt": probe[("Pt", "graph")]["val_acc"],
            "acc_local_Pv": probe[("Pv", "local")]["val_acc"],
            "acc_graph_Pv": probe[("Pv", "graph")]["val_acc"],
            "conflict_C_Pt": summary["conflict"]["conflict_C_Pt"],
            "conflict_C_Pv": summary["conflict"]["conflict_C_Pv"],
            "conflict_Pt_Pv": summary["conflict"]["conflict_Pt_Pv"],
            "corr_spearman_delta_C_Pt": summary["conflict"]["corr_spearman_delta_C_Pt"],
            "corr_spearman_delta_C_Pv": summary["conflict"]["corr_spearman_delta_C_Pv"],
            "pattern_all_help": summary["conflict"]["pattern_all_help"],
            "pattern_all_hurt": summary["conflict"]["pattern_all_hurt"],
            "pattern_mixed": summary["conflict"]["pattern_mixed"],
            "fused_val_acc": (summary.get("fused_train_results") or {}).get("val_acc", {}).get("mean"),
            "fused_test_acc": (summary.get("fused_train_results") or {}).get("test_acc", {}).get("mean"),
        })
    return rows


def _rows_for_lp(summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        probe = {(row["factor"], row["mode"]): row for row in summary["probe"]}
        delta = {}
        for factor in ("C", "Pt", "Pv"):
            delta[factor] = probe[(factor, "graph")]["val_mrr"] - probe[(factor, "local")]["val_mrr"]
        rows.append({
            "dataset": summary["dataset"],
            "seed": summary["seed"],
            "common_sim": summary["factor_sanity"]["common_sim"],
            "private_sim": summary["factor_sanity"]["private_sim"],
            "effrank_c": summary["factor_sanity"]["effrank_c"],
            "effrank_pt": summary["factor_sanity"]["effrank_pt"],
            "effrank_pv": summary["factor_sanity"]["effrank_pv"],
            "rho_C_Pt": summary["edge_statistics"]["rho_C_Pt"],
            "rho_C_Pv": summary["edge_statistics"]["rho_C_Pv"],
            "rho_Pt_Pv": summary["edge_statistics"]["rho_Pt_Pv"],
            "jaccard_top10_C_Pt": summary["edge_statistics"]["jaccard_top10_C_Pt"],
            "jaccard_top10_Pt_Pv": summary["edge_statistics"]["jaccard_top10_Pt_Pv"],
            "jaccard_top20_C_Pt": summary["edge_statistics"]["jaccard_top20_C_Pt"],
            "jaccard_top20_Pt_Pv": summary["edge_statistics"]["jaccard_top20_Pt_Pv"],
            "delta_mrr_C": delta["C"],
            "delta_mrr_Pt": delta["Pt"],
            "delta_mrr_Pv": delta["Pv"],
            "mrr_local_C": probe[("C", "local")]["val_mrr"],
            "mrr_graph_C": probe[("C", "graph")]["val_mrr"],
            "mrr_local_Pt": probe[("Pt", "local")]["val_mrr"],
            "mrr_graph_Pt": probe[("Pt", "graph")]["val_mrr"],
            "mrr_local_Pv": probe[("Pv", "local")]["val_mrr"],
            "mrr_graph_Pv": probe[("Pv", "graph")]["val_mrr"],
            "conflict_C_Pt": summary["conflict"]["conflict_C_Pt"],
            "conflict_C_Pv": summary["conflict"]["conflict_C_Pv"],
            "conflict_Pt_Pv": summary["conflict"]["conflict_Pt_Pv"],
            "corr_spearman_delta_C_Pt": summary["conflict"]["corr_spearman_delta_C_Pt"],
            "corr_spearman_delta_C_Pv": summary["conflict"]["corr_spearman_delta_C_Pv"],
            "pattern_all_help": summary["conflict"]["pattern_all_help"],
            "pattern_all_hurt": summary["conflict"]["pattern_all_hurt"],
            "pattern_mixed": summary["conflict"]["pattern_mixed"],
            "fused_val_mrr": (summary.get("fused_train_results") or {}).get("val_mrr", {}).get("mean"),
            "fused_test_mrr": (summary.get("fused_train_results") or {}).get("test_mrr", {}).get("mean"),
        })
    return rows


def _aggregate(variant: str | None = None) -> None:
    nc_jobs = [(d, "nc", s) for d in NC_DATASETS for s in SEEDS]
    lp_jobs = [(d, "lp", s) for d in LP_DATASETS for s in SEEDS]
    nc_summaries = [s for job in nc_jobs if (s := _load_summary(*job, variant=variant)) is not None]
    lp_summaries = [s for job in lp_jobs if (s := _load_summary(*job, variant=variant)) is not None]

    nc_rows = _rows_for_nc(nc_summaries)
    lp_rows = _rows_for_lp(lp_summaries)
    _write_csv(OUT_ROOT / "p0_nc_summary.csv", nc_rows)
    _write_csv(OUT_ROOT / "p0_lp_summary.csv", lp_rows)
    _write_protocol_comparison(nc_rows, lp_rows)

    conflict_rows = []
    for row in nc_rows:
        conflict_rows.append({"task": "nc", **{k: v for k, v in row.items() if k in (
            "dataset", "seed", "conflict_C_Pt", "conflict_C_Pv", "conflict_Pt_Pv",
            "corr_spearman_delta_C_Pt", "corr_spearman_delta_C_Pv",
            "pattern_all_help", "pattern_all_hurt", "pattern_mixed",
        )}})
    for row in lp_rows:
        conflict_rows.append({"task": "lp", **{k: v for k, v in row.items() if k in (
            "dataset", "seed", "conflict_C_Pt", "conflict_C_Pv", "conflict_Pt_Pv",
            "corr_spearman_delta_C_Pt", "corr_spearman_delta_C_Pv",
            "pattern_all_help", "pattern_all_hurt", "pattern_mixed",
        )}})
    _write_csv(OUT_ROOT / "p0_conflict_summary.csv", conflict_rows)

    _write_report(nc_rows, lp_rows)
    print(f"[aggregate] {len(nc_rows)} NC rows, {len(lp_rows)} LP rows", flush=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


COMPARE_KEYS = (
    "common_sim",
    "private_sim",
    "effrank_c",
    "effrank_pt",
    "effrank_pv",
    "rho_C_Pt",
    "rho_C_Pv",
    "jaccard_top20_C_Pt",
    "conflict_C_Pt",
    "conflict_C_Pv",
)


def _write_protocol_comparison(nc_rows: list[dict], lp_rows: list[dict]) -> None:
    """Old (pre-overhaul) vs new (RPTA-style) protocol comparison per dataset x seed."""
    rows: list[dict] = []
    for task, task_rows, datasets in (
        ("nc", nc_rows, NC_DATASETS),
        ("lp", lp_rows, LP_DATASETS),
    ):
        for dataset in datasets:
            for seed in SEEDS:
                old_path = OLD_PROTOCOL_ROOT / dataset / f"seed_{seed}" / "summary.json"
                if not old_path.exists():
                    continue
                with old_path.open(encoding="utf-8") as f:
                    old = json.load(f)
                new = _load_summary(dataset, task, seed)
                if new is None:
                    continue
                old_rows = _rows_for_nc([old]) if task == "nc" else _rows_for_lp([old])
                new_rows = _rows_for_nc([new]) if task == "nc" else _rows_for_lp([new])
                row = {"task": task, "dataset": dataset, "seed": seed}
                for key in COMPARE_KEYS:
                    if key in old_rows[0] and key in new_rows[0]:
                        row[f"old_{key}"] = old_rows[0][key]
                        row[f"new_{key}"] = new_rows[0][key]
                old_fused = old_rows[0].get("fused_test_acc" if task == "nc" else "fused_test_mrr")
                new_fused = new_rows[0].get("fused_test_acc" if task == "nc" else "fused_test_mrr")
                row[f"old_fused_test_{task}"] = old_fused
                row[f"new_fused_test_{task}"] = new_fused
                rows.append(row)
    if rows:
        _write_csv(OUT_ROOT / "p0_protocol_comparison.csv", rows)
        print(f"[compare] old-vs-new protocol rows: {len(rows)}", flush=True)


def _by_dataset(rows: list[dict], key: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row[key] is not None and row[key] == row[key]:
            grouped.setdefault(row["dataset"], []).append(row[key])
    return grouped


def _write_report(nc_rows: list[dict], lp_rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# P0 Report — Factor-Dependent Neighborhood Utility")
    lines.append("")
    lines.append("> Auto-generated from `outputs/p0/*/seed_*/summary.json` (mean±std over seeds).")
    lines.append("")

    lines.append("## NC")
    lines.append("")
    lines.append("| Dataset | Common Sim | Private Sim | rho C/T | rho C/V | Jaccard20 C/T | ΔAcc C | ΔAcc Pt | ΔAcc Pv | Conflict C/T | Conflict C/V |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in NC_DATASETS:
        rows = [r for r in nc_rows if r["dataset"] == dataset]
        if not rows:
            continue
        lines.append(
            f"| {dataset} | {_mean_std_str([r['common_sim'] for r in rows])} | "
            f"{_mean_std_str([r['private_sim'] for r in rows])} | "
            f"{_mean_std_str([r['rho_C_Pt'] for r in rows])} | {_mean_std_str([r['rho_C_Pv'] for r in rows])} | "
            f"{_mean_std_str([r['jaccard_top20_C_Pt'] for r in rows])} | "
            f"{_mean_std_str([r['delta_acc_C'] for r in rows])} | {_mean_std_str([r['delta_acc_Pt'] for r in rows])} | "
            f"{_mean_std_str([r['delta_acc_Pv'] for r in rows])} | "
            f"{_mean_std_str([r['conflict_C_Pt'] for r in rows])} | {_mean_std_str([r['conflict_C_Pv'] for r in rows])} |"
        )
    lines.append("")

    lines.append("## LP")
    lines.append("")
    lines.append("| Dataset | rho C/T | rho C/V | ΔMRR C | ΔMRR Pt | ΔMRR Pv | RR Conflict C/T | RR Conflict C/V |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in LP_DATASETS:
        rows = [r for r in lp_rows if r["dataset"] == dataset]
        if not rows:
            continue
        lines.append(
            f"| {dataset} | {_mean_std_str([r['rho_C_Pt'] for r in rows])} | "
            f"{_mean_std_str([r['rho_C_Pv'] for r in rows])} | "
            f"{_mean_std_str([r['delta_mrr_C'] for r in rows])} | {_mean_std_str([r['delta_mrr_Pt'] for r in rows])} | "
            f"{_mean_std_str([r['delta_mrr_Pv'] for r in rows])} | "
            f"{_mean_std_str([r['conflict_C_Pt'] for r in rows])} | {_mean_std_str([r['conflict_C_Pv'] for r in rows])} |"
        )
    lines.append("")

    lines.append("## GO / NO-GO checklist (plan §19)")
    lines.append("")
    lines.append("- Factorization: S_C > S_P and no rank collapse —")
    lines.append("- Edge ranking: Spearman(C,T) < 0.7 or Spearman(C,V) < 0.7 or Top20 Jaccard < 0.6 —")
    lines.append("- Propagation utility: ConflictRate(C,T) > 15% or ConflictRate(C,V) > 15% on ≥2 NC datasets —")
    lines.append("- Cross-task support: factor-wise ΔMRR inconsistency + non-trivial RR conflict on ≥1 LP dataset —")
    lines.append("- Decision: PENDING (fill after analysis)")
    lines.append("")

    (OUT_ROOT / "p0_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 batch driver")
    parser.add_argument("--stage", required=True, choices=["d1", "d2", "d3", "all"])
    parser.add_argument("--gpus", default="0,1", help="comma-separated GPU ids")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--variant", default=None, help="output subdir suffix (variant lambda runs)")
    parser.add_argument(
        "--model-overrides",
        default="",
        help="comma-separated model config overrides passed to every job",
    )
    args = parser.parse_args()

    gpus = [int(gpu.strip()) for gpu in args.gpus.split(",") if gpu.strip()]
    jobs = _jobs_for_stage(args.stage)
    print(f"[batch] stage={args.stage} jobs={len(jobs)} gpus={gpus} variant={args.variant}", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
        for job in jobs:
            gpu_id = next(gpu_iter)
            futures[executor.submit(_run_job, job, gpu_id, args.include_test, args.force, args.variant, args.model_overrides)] = job
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)

    if not args.no_aggregate:
        _aggregate(args.variant)
    print("[batch] done", flush=True)


if __name__ == "__main__":
    main()
