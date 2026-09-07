"""R2D29 G0 driver: rebuild the current-commit NC Validation reference.

Runs the 10 formal benchmark models on 5 NC datasets x seeds 42/43/44
(=150 runs) under the unified NC protocol, Val-only selection. Outputs go
to outputs/r2d29/g0_reference/main/<dataset>/<model>/seed_<s>/ with the
hydra results.json + train.log + run_info.json sidecar
(params / peak GPU mem / runtime / git commit).

Discipline (plan §5): no baseline hyperparameter changes, no Test use for
selection, failed runs are recorded (failures.jsonl) and never silently
skipped. Resume/skip on existing results.json (--force overrides).

Usage:
    python scripts/perf_r2d29_g0_reference.py --gpus 0,1
    python scripts/perf_r2d29_g0_reference.py --datasets Movies --models mlp --seeds 42  # smoke
"""

from __future__ import annotations

import argparse
import json
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

from run_p3_operator_screen import LARGE_DATASETS, _WeightedSemaphore  # noqa: E402
from run_p1_screen import _poll_peak_mem  # noqa: E402

MODELS = ["mlp", "gcn", "sage", "mmgcn", "mgat", "dmgc", "dgf", "dip", "lgmrec", "biaxis_final"]
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
OUT_ROOT = PROJECT_ROOT / "outputs" / "r2d29" / "g0_reference"


def _resolve_python() -> str:
    """The driver may itself be launched with a python that lacks hydra
    (e.g. from a bare shell); prefer the project conda env yhf_env."""
    try:
        import hydra  # noqa: F401
        return sys.executable
    except ImportError:
        pass
    for candidate in (
        Path.home() / "miniconda3" / "envs" / "yhf_env" / "bin" / "python",
        Path.home() / "anaconda3" / "envs" / "yhf_env" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_job(
    dataset: str,
    model: str,
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
        _run_job_locked(dataset, model, seed, gpu_id, force)
    finally:
        if lock is not None:
            lock.release(weight)


def _run_job_locked(dataset: str, model: str, seed: int, gpu_id: int, force: bool) -> None:
    outdir = OUT_ROOT / "main" / dataset / model / f"seed_{seed}"
    tag = f"[{gpu_id}] {dataset} {model} seed={seed}"
    results_json = outdir / "hydra" / "results.json"
    if results_json.exists() and not force:
        print(f"{tag} SKIP (results exist)", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        _resolve_python(), "-m", "src.main",
        f"dataset={dataset}", "task=nc", f"model={model}",
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
        print(log_path.read_text(encoding="utf-8")[-2500:], flush=True)
        with (OUT_ROOT / "failures.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "dataset": dataset, "model": model, "seed": seed, "gpu": gpu_id,
                "returncode": proc.returncode,
                "runtime_sec": round(runtime_sec, 1),
                "log_tail": log_path.read_text(encoding="utf-8")[-1200:],
            }) + "\n")
        return
    text = log_path.read_text(encoding="utf-8")
    params = None
    match = re.search(r"model\+head params=(\d+)", text)
    if match:
        params = int(match.group(1))
    with (outdir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset, "model": model, "seed": seed,
            "params": params,
            "runtime_sec": round(runtime_sec, 1),
            "train_peak_gpu_mb": mem_holder.get("peak"),
            "git_commit": _git_commit(),
        }, f, indent=2)
    print(f"{tag} OK ({runtime_sec:.0f}s, params={params})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2D29 G0 reference driver (10 models x 5 datasets x 3 seeds)")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    args = parser.parse_args()

    datasets = DATASETS
    if args.datasets:
        datasets = [d for d in args.datasets.split(",") if d in DATASETS]
    models = MODELS
    if args.models:
        models = [m for m in args.models.split(",") if m in MODELS]
    seeds = SEEDS
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s]
    gpus = [int(g) for g in args.gpus.split(",") if g]
    slots = max(1, int(args.slots_per_gpu))
    gpu_locks = {g: _WeightedSemaphore(slots) for g in gpus}

    jobs = sorted(
        [(d, m, s) for d in datasets for m in models for s in seeds],
        key=lambda j: j[0] != "ele-fashion",
    )
    print(f"[driver] {len(jobs)} jobs gpus={gpus} slots/gpu={slots} commit={_git_commit()[:8]}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus) * slots) as executor:
        futures = {}
        gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
        for dataset, model, seed in jobs:
            futures[executor.submit(
                _run_job, dataset, model, seed, next(gpu_iter), args.force, gpu_locks, slots
            )] = (dataset, model, seed)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
