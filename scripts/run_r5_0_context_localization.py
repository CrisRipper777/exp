"""Run the pre-registered R5-0 local/global context localization screen.

The formal screen is validation-only and writes only to
``outputs/r5_0_context_localization``.  One process is assigned to each GPU;
jobs on the same GPU are serialized so ele-fashion cannot contend for memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = ["Movies", "Toys", "Grocery"]
SEEDS = [42, 43]
VARIANTS = {
    "M0-MEAN": [
        "model.scope.context_core=oacc",
        "model.scope.graph_mode=local",
        "model.scope.local_aggregation=mean",
        "model.scope.post_context_norm=false",
        "model.scope.num_blocks=2",
        "model.scope.beta_mode=one",
        "model.scope.reconciliation.mode=none",
        "model.scope.use_source_projection=false",
    ],
    "M1-MEAN-LN": [
        "model.scope.context_core=oacc",
        "model.scope.graph_mode=local",
        "model.scope.local_aggregation=mean",
        "model.scope.post_context_norm=true",
        "model.scope.num_blocks=2",
        "model.scope.beta_mode=one",
        "model.scope.reconciliation.mode=none",
        "model.scope.use_source_projection=false",
    ],
    "S0-SYM": [
        "model.scope.context_core=oacc",
        "model.scope.graph_mode=local",
        "model.scope.local_aggregation=sym_norm",
        "model.scope.post_context_norm=false",
        "model.scope.num_blocks=2",
        "model.scope.beta_mode=one",
        "model.scope.reconciliation.mode=none",
        "model.scope.use_source_projection=false",
    ],
    "M2-MEAN-G4": [
        "model.scope.context_core=oacc",
        "model.scope.graph_mode=local_global",
        "model.scope.local_aggregation=mean",
        "model.scope.post_context_norm=false",
        "model.scope.num_blocks=2",
        "model.scope.beta_mode=one",
        "model.scope.reconciliation.mode=none",
        "model.scope.use_source_projection=false",
        "model.scope.global_num_slots=4",
        "model.scope.global_scale_init=0.05",
    ],
    "S1-SYM-G4": [
        "model.scope.context_core=oacc",
        "model.scope.graph_mode=local_global",
        "model.scope.local_aggregation=sym_norm",
        "model.scope.post_context_norm=false",
        "model.scope.num_blocks=2",
        "model.scope.beta_mode=one",
        "model.scope.reconciliation.mode=none",
        "model.scope.use_source_projection=false",
        "model.scope.global_num_slots=4",
        "model.scope.global_scale_init=0.05",
    ],
}
OUT_ROOT = PROJECT_ROOT / "outputs" / "r5_0_context_localization"


def _poll_peak_mem(gpu_id: int, stop: threading.Event, holder: dict[str, int]) -> None:
    while not stop.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi", "--id", str(gpu_id),
                    "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                ], capture_output=True, text=True, timeout=5,
            )
            values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            if values:
                holder["peak"] = max(holder.get("peak", 0), max(values))
        except Exception:
            pass
        stop.wait(2.0)


def _best_history_row(history_path: Path) -> dict | None:
    if not history_path.exists():
        return None
    with history_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return max(rows, key=lambda row: float(row["val_acc"]))


def _run_one(
    variant: str,
    dataset: str,
    seed: int,
    gpu: int,
    epochs: int,
    patience: int,
    relative_root: str,
    force: bool,
) -> None:
    outdir = OUT_ROOT / relative_root / variant / dataset / f"seed_{seed}"
    result_path = outdir / "hydra" / "results.json"
    tag = f"[{gpu}] {relative_root}/{variant} {dataset} s{seed}"
    if result_path.exists() and not force:
        print(f"{tag} SKIP", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable, "-m", "src.main",
        f"dataset={dataset}", "task=nc", "model=biaxis_scope_v2",
        "num_runs=1", f"seed={seed}", "device=cuda:0",
        "task.evaluate_test=false", f"task.patience={patience}",
        f"task.epochs={epochs}",
        f"task.history_path={outdir / 'history.csv'}",
        f"task.save_ckpt_path={outdir / 'best.pt'}",
        f"hydra.run.dir={outdir / 'hydra'}",
        *VARIANTS[variant],
    ]
    log_path = outdir / "train.log"
    memory: dict[str, int] = {}
    stop = threading.Event()
    watcher = threading.Thread(target=_poll_peak_mem, args=(gpu, stop, memory), daemon=True)
    watcher.start()
    print(f"{tag} TRAIN", flush=True)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    stop.set()
    watcher.join(timeout=5)
    runtime = time.monotonic() - start
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    params_match = re.search(r"model\+head params=(\d+)", log_text)
    info = {
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "gpu": gpu,
        "epochs": epochs,
        "patience": patience,
        "runtime_sec": round(runtime, 1),
        "peak_gpu_mb": memory.get("peak"),
        "returncode": process.returncode,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
    }
    if params_match:
        info["params"] = int(params_match.group(1))
    (outdir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if process.returncode:
        failure = {
            **info,
            "log_tail": log_text[-4000:],
        }
        failure_path = OUT_ROOT / "audit" / "failures.jsonl"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        print(f"{tag} FAILED rc={process.returncode}; see {log_path}", flush=True)
    else:
        print(
            f"{tag} OK runtime={runtime:.0f}s peak={memory.get('peak')}MB",
            flush=True,
        )


def _write_master(relative_root: str, variants: list[str], datasets: list[str], seeds: list[int]) -> None:
    rows: list[dict] = []
    for variant in variants:
        for dataset in datasets:
            for seed in seeds:
                outdir = OUT_ROOT / relative_root / variant / dataset / f"seed_{seed}"
                result_path = outdir / "hydra" / "results.json"
                info_path = outdir / "run_info.json"
                best = _best_history_row(outdir / "history.csv")
                if not result_path.exists() or best is None or not info_path.exists():
                    continue
                info = json.loads(info_path.read_text(encoding="utf-8"))
                row = {
                    "variant": variant,
                    "dataset": dataset,
                    "seed": seed,
                    "best_epoch": int(best["epoch"]),
                    "best_val_acc_pct": 100.0 * float(best["val_acc"]),
                    "best_val_f1_pct": 100.0 * float(best["val_macro_f1"]),
                    "params": info.get("params"),
                    "peak_gpu_mb": info.get("peak_gpu_mb"),
                    "runtime_sec": info.get("runtime_sec"),
                    "git_commit": info.get("git_commit"),
                }
                for key, value in best.items():
                    if key.startswith("scope_"):
                        row[key] = value
                rows.append(row)
    summary_dir = OUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / "R5_0_MASTER_TABLE.csv"
    fields = [
        "variant", "dataset", "seed", "best_epoch", "best_val_acc_pct",
        "best_val_f1_pct", "params", "peak_gpu_mb", "runtime_sec", "git_commit",
    ]
    extra = sorted({key for row in rows for key in row if key.startswith("scope_")})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] wrote {path} ({len(rows)} rows)", flush=True)


def _run_jobs(
    jobs: list[tuple[str, str, int]],
    gpus: list[int],
    epochs: int,
    patience: int,
    relative_root: str,
    force: bool,
) -> None:
    queues = [jobs[index::len(gpus)] for index in range(len(gpus))]

    def worker(gpu: int, queue: list[tuple[str, str, int]]) -> None:
        for variant, dataset, seed in queue:
            _run_one(variant, dataset, seed, gpu, epochs, patience, relative_root, force)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, queue) for gpu, queue in zip(gpus, queues)]
        for future in futures:
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--root", default="formal")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run Movies/1 epoch and ele-fashion/2 epoch S1 smoke tests")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise SystemExit("at least one GPU is required")
    if args.smoke:
        _run_jobs(
            [("S1-SYM-G4", "Movies", 42)], gpus, 1, 1,
            "smoke_movies_1ep", args.force,
        )
        # Move the ele-fashion smoke to its own output namespace by running it
        # explicitly; it must be two epochs and must not be confused with the
        # one-epoch Movies check.
        _run_jobs(
            [("S1-SYM-G4", "ele-fashion", 42)], gpus, 2, 2,
            "smoke_ele_2ep", args.force,
        )
        return
    datasets = [value for value in args.datasets.split(",") if value in DATASETS]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    variants = [value for value in args.variants.split(",") if value in VARIANTS]
    jobs = [(variant, dataset, seed) for variant in variants for dataset in datasets for seed in seeds]
    print(f"[driver] formal jobs={len(jobs)} gpus={gpus}; one serialized queue/GPU", flush=True)
    _run_jobs(jobs, gpus, args.epochs, args.patience, args.root, args.force)
    _write_master(args.root, variants, datasets, seeds)


if __name__ == "__main__":
    main()
