"""Final NC benchmark driver (plan §18 Prompt 2).

Runs ONLY the frozen final model (model=biaxis_final) under the unified NC
protocol: 5 datasets x seeds 42/43/44 = 15 runs into
outputs/final_nc_benchmark/main/<dataset>/biaxis_final/seed_<s>/.

The 8 baselines are NEVER re-run: the provenance audit
(docs/Final_NC_Benchmark_Audit.md) passed all 120 existing runs in
outputs/baseline_nc (identical task protocol, matched model presets,
seed-aligned splits) — the summarizer reads them directly.

Discipline: no structural overrides of the frozen model; deterministic
mode stays off; resume/skip on results.json (--force overrides).

Usage:
    python scripts/run_nc_final_benchmark.py --gpus 0,1
    python scripts/run_nc_final_benchmark.py --datasets Movies --seeds 42  # smoke
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_p3_lowrank_screen import LARGE_DATASETS, _WeightedSemaphore  # noqa: E402
from run_p1_screen import _poll_peak_mem  # noqa: E402

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
OUT_ROOT = PROJECT_ROOT / "outputs" / "final_nc_benchmark" / "main"


def _run_job(
    dataset: str,
    seed: int,
    gpu_id: int,
    force: bool,
    gpu_locks: dict[int, _WeightedSemaphore],
    slots_per_gpu: int,
) -> None:
    lock = gpu_locks.get(gpu_id)
    weight = int(slots_per_gpu) if dataset in LARGE_DATASETS else 1
    if lock is not None:
        lock.acquire(weight)
    try:
        _run_job_locked(dataset, seed, gpu_id, force)
    finally:
        if lock is not None:
            lock.release(weight)


def _run_job_locked(dataset: str, seed: int, gpu_id: int, force: bool) -> None:
    outdir = OUT_ROOT / dataset / "biaxis_final" / f"seed_{seed}"
    tag = f"[{gpu_id}] {dataset} seed={seed}"
    results_json = outdir / "hydra" / "results.json"
    if results_json.exists() and not force:
        print(f"{tag} SKIP (results exist)", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, "-m", "src.main",
        f"dataset={dataset}", "task=nc", "model=biaxis_final",
        "num_runs=1", f"seed={seed}", "device=cuda:0",
        f"hydra.run.dir={outdir / 'hydra'}",
    ]
    mem_holder: dict[str, int] = {}
    stop_event = threading.Event()
    mem_thread = threading.Thread(target=_poll_peak_mem, args=(gpu_id, stop_event, mem_holder))
    mem_thread.start()
    log_path = outdir / "train.log"
    print(f"{tag} TRAIN", flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    runtime_sec = time.monotonic() - started
    stop_event.set()
    mem_thread.join(timeout=5)
    if proc.returncode != 0:
        print(f"{tag} TRAIN FAILED rc={proc.returncode}", flush=True)
        print(log_path.read_text(encoding="utf-8")[-3000:], flush=True)
        return
    # record params + peak mem in a sidecar for the summarizer
    text = log_path.read_text(encoding="utf-8")
    params = None
    match = re.search(r"model\+head params=(\d+)", text)
    if match:
        params = int(match.group(1))
    import json
    with (outdir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset, "model": "biaxis_final", "seed": seed,
            "params": params,
            "runtime_sec": round(runtime_sec, 1),
            "train_peak_gpu_mb": mem_holder.get("peak"),
        }, f, indent=2)
    print(f"{tag} OK ({runtime_sec:.0f}s, params={params})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final NC benchmark driver (biaxis_final only)")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    args = parser.parse_args()

    datasets = DATASETS
    if args.datasets:
        datasets = [d for d in args.datasets.split(",") if d in DATASETS]
    seeds = SEEDS
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s]
    gpus = [int(g) for g in args.gpus.split(",") if g]
    slots = max(1, int(args.slots_per_gpu))
    gpu_locks = {g: _WeightedSemaphore(slots) for g in gpus}

    jobs = sorted([(d, s) for d in datasets for s in seeds], key=lambda j: j[0] != "ele-fashion")
    print(f"[driver] {len(jobs)} jobs gpus={gpus} slots/gpu={slots}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus) * slots) as executor:
        futures = {}
        gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
        for dataset, seed in jobs:
            futures[executor.submit(_run_job, dataset, seed, next(gpu_iter), args.force, gpu_locks, slots)] = (dataset, seed)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
