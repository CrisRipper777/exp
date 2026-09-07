"""RPTA migration performance probe (user request 2026-09-03).

Runs model=rpta under the unified NC protocol with the per-dataset FROZEN
presets from RPTA/configs/rpta_final_nc_v1.yaml (dataset-specific capacity/
optimisation selection is part of the frozen RPTA config, kept as-is for a
faithful performance probe). Outputs to outputs/rpta_probe/.

Usage:
    python scripts/run_rpta_probe.py --gpus 0,1
    python scripts/run_rpta_probe.py --datasets Movies --seeds 42 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import os
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
OUT_ROOT = PROJECT_ROOT / "outputs" / "rpta_probe"

# Frozen per-dataset presets (RPTA/configs/rpta_final_nc_v1.yaml).
# Gate priors are NOT dataset-tuned: the frozen core profile fixes
# common_late_max_gate=0.0 / outer_relation_prior=0.50 internally.
PRESETS = {
    "Movies": dict(hidden_dim=256, factor_dim=128, dropout=0.6, lr=0.00075,
                   lambda_decomposition=0.01, lambda_factor_route_task=0.3,
                   lambda_decoupled_route_task=0.05),
    "Grocery": dict(hidden_dim=192, factor_dim=96, dropout=0.5, lr=0.001,
                    lambda_decomposition=0.005, lambda_factor_route_task=0.4,
                    lambda_decoupled_route_task=0.0666666666667),
    "Toys": dict(hidden_dim=128, factor_dim=64, dropout=0.5, lr=0.00125,
                 lambda_decomposition=0.005, lambda_factor_route_task=0.4,
                 lambda_decoupled_route_task=0.0666666666667),
    "ele-fashion": dict(hidden_dim=192, factor_dim=96, dropout=0.5, lr=0.001,
                        lambda_decomposition=0.005, lambda_factor_route_task=0.3,
                        lambda_decoupled_route_task=0.05),
    "Reddit-S": dict(hidden_dim=256, factor_dim=128, dropout=0.5, lr=0.001,
                     lambda_decomposition=0.005, lambda_factor_route_task=0.1,
                     lambda_decoupled_route_task=0.02),
}
# shared across datasets in the frozen config
for preset in PRESETS.values():
    preset.update(weight_decay=0.0001, lambda_common_node=0.1,
                  lambda_prototype_balance=0.1, lambda_private_sharpness=0.005)


def _run_job(dataset, seed, gpu_id, force, epochs, gpu_locks, slots_per_gpu, out_root):
    lock = gpu_locks.get(gpu_id)
    weight = int(slots_per_gpu) if dataset in LARGE_DATASETS else 1
    if lock is not None:
        lock.acquire(weight)
    try:
        _run_job_locked(dataset, seed, gpu_id, force, epochs, out_root)
    finally:
        if lock is not None:
            lock.release(weight)


def _run_job_locked(dataset, seed, gpu_id, force, epochs, out_root):
    outdir = Path(out_root) / dataset / f"seed_{seed}"
    tag = f"[{gpu_id}] {dataset} seed={seed}"
    results_json = outdir / "hydra" / "results.json"
    if results_json.exists() and not force:
        print(f"{tag} SKIP (results exist)", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    preset = PRESETS[dataset]
    overrides = [f"model.{k}={v}" for k, v in preset.items()]
    cmd = [
        sys.executable, "-m", "src.main",
        f"dataset={dataset}", "task=nc", "model=rpta",
        "num_runs=1", f"seed={seed}", "device=cuda:0",
        f"hydra.run.dir={outdir / 'hydra'}",
    ] + overrides
    if epochs is not None:
        cmd.append(f"task.epochs={int(epochs)}")
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
    with (outdir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "model": "rpta", "seed": seed,
                   "runtime_sec": round(runtime_sec, 1),
                   "train_peak_gpu_mb": mem_holder.get("peak"),
                   "preset": preset}, f, indent=2)
    print(f"{tag} OK ({runtime_sec:.0f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="RPTA migration performance probe")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()

    datasets = [d for d in (args.datasets or ",".join(DATASETS)).split(",") if d in DATASETS]
    seeds = [int(s) for s in (args.seeds or ",".join(map(str, SEEDS))).split(",") if s]
    gpus = [int(g) for g in args.gpus.split(",") if g]
    slots = max(1, int(args.slots_per_gpu))
    gpu_locks = {g: _WeightedSemaphore(slots) for g in gpus}
    out_root = args.out_root or str(OUT_ROOT)

    jobs = sorted([(d, s) for d in datasets for s in seeds], key=lambda j: j[0] != "ele-fashion")
    print(f"[driver] {len(jobs)} jobs gpus={gpus} slots/gpu={slots}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus) * slots) as executor:
        futures = {}
        gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
        for dataset, seed in jobs:
            futures[executor.submit(_run_job, dataset, seed, next(gpu_iter),
                                    args.force, args.epochs, gpu_locks, slots, out_root)] = (dataset, seed)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
