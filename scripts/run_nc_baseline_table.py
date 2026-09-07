"""NC baseline reference table under the unified RPTA-style protocol.

5 datasets x 8 models (mlp/gcn/sage/mmgcn/mgat/dmgc/dgf/dip) x 3 seeds,
each as one src.main invocation (MAGB split files are per-seed). Records
val/test Accuracy and Macro-F1; aggregates per dataset x model with
mean ± population std over seeds.

Outputs:
    outputs/baseline_nc/<dataset>/<model>/seed_<seed>/hydra/results.json
    outputs/baseline_nc/nc_baseline_table.csv
    outputs/baseline_nc/nc_baseline_table_per_seed.csv

Usage:
    python scripts/run_nc_baseline_table.py --gpus 0,1
    python scripts/run_nc_baseline_table.py --gpus 0,1 --datasets Movies,Toys
    python scripts/run_nc_baseline_table.py --models mlp,gcn
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
OUT_ROOT = PROJECT_ROOT / "outputs" / "baseline_nc"

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
MODELS = ["mlp", "gcn", "sage", "mmgcn", "mgat", "dmgc", "dgf", "dip"]
SEEDS = [42, 43, 44]


def _run(gpu_id: int, dataset: str, model: str, seed: int) -> None:
    outdir = OUT_ROOT / dataset / model / f"seed_{seed}"
    results_json = outdir / "hydra" / "results.json"
    if results_json.exists():
        print(f"[skip] {dataset} {model} seed={seed}", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=nc",
        f"model={model}",
        "num_runs=1",
        f"seed={seed}",
        "device=cuda:0",
        f"hydra.run.dir={outdir / 'hydra'}",
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[FAIL] [{gpu_id}] {dataset} {model} seed={seed} rc={proc.returncode}", flush=True)
        print(proc.stdout[-2000:], flush=True)
        print(proc.stderr[-2000:], flush=True)
    else:
        print(f"[ok] [{gpu_id}] {dataset} {model} seed={seed}", flush=True)


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}"


def _aggregate() -> None:
    table_rows: list[dict] = []
    per_seed_rows: list[dict] = []
    for dataset in DATASETS:
        for model in MODELS:
            per_seed: dict[int, dict] = {}
            for seed in SEEDS:
                results_json = OUT_ROOT / dataset / model / f"seed_{seed}" / "hydra" / "results.json"
                if not results_json.exists():
                    continue
                with results_json.open(encoding="utf-8") as f:
                    per_seed[seed] = json.load(f)
            if not per_seed:
                continue
            for seed, result in sorted(per_seed.items()):
                per_seed_rows.append({
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "val_acc": result["val_acc"]["mean"],
                    "test_acc": result["test_acc"]["mean"],
                    "test_macro_f1": result["test_macro_f1"]["mean"],
                })

            def agg(key: str) -> tuple[float, float]:
                values = [result[key]["mean"] for result in per_seed.values()]
                if len(values) == 1:
                    return values[0], 0.0
                return statistics.mean(values), statistics.pstdev(values)

            val_mean, val_std = agg("val_acc")
            test_mean, test_std = agg("test_acc")
            f1_mean, f1_std = agg("test_macro_f1")
            table_rows.append({
                "dataset": dataset,
                "model": model,
                "val_acc": _fmt(val_mean, val_std),
                "test_acc": _fmt(test_mean, test_std),
                "test_macro_f1": _fmt(f1_mean, f1_std),
                "num_seeds": len(per_seed),
            })

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "nc_baseline_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "model", "val_acc", "test_acc", "test_macro_f1", "num_seeds"]
        )
        writer.writeheader()
        writer.writerows(table_rows)
    with (OUT_ROOT / "nc_baseline_table_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "model", "seed", "val_acc", "test_acc", "test_macro_f1"]
        )
        writer.writeheader()
        writer.writerows(per_seed_rows)
    print(f"[aggregate] {len(table_rows)} dataset-model rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="NC baseline reference table")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None, help="comma-separated subset")
    parser.add_argument("--models", default=None, help="comma-separated subset")
    args = parser.parse_args()

    gpus = [int(gpu.strip()) for gpu in args.gpus.split(",") if gpu.strip()]
    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",") if d]
    models = MODELS if not args.models else [m for m in args.models.split(",") if m]
    # Model-major order mixes fast/slow datasets across GPUs.
    jobs = [(dataset, model, seed) for model in models for dataset in datasets for seed in SEEDS]
    print(f"[table] {len(jobs)} jobs on gpus={gpus}", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
        for dataset, model, seed in jobs:
            gpu_id = next(gpu_iter)
            futures[executor.submit(_run, gpu_id, dataset, model, seed)] = (dataset, model, seed)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)

    _aggregate()
    print("[table] done", flush=True)


if __name__ == "__main__":
    main()
