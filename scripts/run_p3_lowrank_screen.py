"""P3-B experiment driver: parameter-matched low-rank operator study
(plan §38 Prompt 7, only after P3-A GO).

    LR-ADD : c_fk = a_f + b_k
    LR-INT : c_fk = a_f + b_k + a_f * b_k

5 NC datasets x 2 modes x seeds 42/43/44 = 30 runs. O0/OADD/OFR references
come from the already-finished P3-A study (outputs/p3/operator), never
re-run. Same resume/skip + per-GPU slot discipline as the operator driver.

Usage:
    python scripts/run_p3_lowrank_screen.py --gpus 0,1
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

from run_p1_screen import _parse_train_log, _poll_peak_mem  # noqa: E402
from run_p3_operator_screen import LARGE_DATASETS, _WeightedSemaphore  # noqa: E402

NC_DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
CONFIRM_SEEDS = [42, 43, 44]

MODE_OVERRIDES: dict[str, list[str]] = {
    "LR-ADD": ["model.p3.operator_mode=lowrank_add"],
    "LR-INT": ["model.p3.operator_mode=lowrank_interaction"],
}


def _parse_best_val_f1(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    best_acc, best_f1 = -1.0, None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = re.search(r"Val Acc ([\d.]+) \| Val F1 ([\d.]+)", line)
        if match:
            acc = float(match.group(1))
            if acc > best_acc:
                best_acc, best_f1 = acc, float(match.group(2))
    return best_f1


def _run_dir(out_root: str, dataset: str, mode: str, seed: int, num_seeds: int) -> Path:
    base = Path(out_root) / dataset / mode
    if num_seeds > 1:
        return base / f"seed_{seed}"
    return base


def _run_job(
    dataset: str,
    mode: str,
    seed: int,
    gpu_id: int,
    out_root: str,
    force: bool,
    epochs: int | None,
    no_diagnostics: bool,
    gpu_locks: dict[int, _WeightedSemaphore],
    num_seeds: int,
    slots_per_gpu: int = 1,
) -> None:
    lock = gpu_locks.get(gpu_id)
    weight = int(slots_per_gpu) if dataset in LARGE_DATASETS else 1
    if lock is not None:
        lock.acquire(weight)
    try:
        _run_job_locked(dataset, mode, seed, gpu_id, out_root, force, epochs, no_diagnostics, num_seeds)
    finally:
        if lock is not None:
            lock.release(weight)


def _run_job_locked(
    dataset: str,
    mode: str,
    seed: int,
    gpu_id: int,
    out_root: str,
    force: bool,
    epochs: int | None,
    no_diagnostics: bool,
    num_seeds: int,
) -> None:
    overrides = list(MODE_OVERRIDES[mode])
    outdir = _run_dir(out_root, dataset, mode, seed, num_seeds)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"[{gpu_id}] {dataset} {mode} seed={seed}"
    summary_path = outdir / "summary.json"
    if summary_path.exists() and not force:
        print(f"{tag} SKIP (summary exists)", flush=True)
        return

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    train_cmd = [
        sys.executable, "-m", "src.main",
        f"dataset={dataset}", "task=nc", "model=biaxis_p3",
        "num_runs=1", f"seed={seed}", "device=cuda:0",
        f"task.save_ckpt_path={outdir / 'model.pt'}",
        f"hydra.run.dir={outdir / 'hydra'}",
    ] + overrides
    if epochs is not None:
        train_cmd.append(f"task.epochs={int(epochs)}")

    mem_holder: dict[str, int] = {}
    stop_event = threading.Event()
    mem_thread = threading.Thread(target=_poll_peak_mem, args=(gpu_id, stop_event, mem_holder))
    mem_thread.start()
    train_log = outdir / "train.log"
    print(f"{tag} TRAIN", flush=True)
    started = time.monotonic()
    with train_log.open("w", encoding="utf-8") as log:
        proc = subprocess.run(train_cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    runtime_sec = time.monotonic() - started
    stop_event.set()
    mem_thread.join(timeout=5)
    peak_gpu_mb = mem_holder.get("peak")

    if proc.returncode != 0:
        print(f"{tag} TRAIN FAILED rc={proc.returncode}", flush=True)
        print(train_log.read_text(encoding="utf-8")[-3000:], flush=True)
        return

    diag = None
    if not no_diagnostics and (outdir / "model.pt").exists():
        diag_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "analyze_p3_checkpoint.py"),
            "--dataset", dataset, "--task", "nc", "--seed", str(seed),
            "--ckpt", str(outdir / "model.pt"),
            "--out", str(outdir), "--device", "cuda:0",
            "--model-overrides", ",".join(overrides),
        ]
        print(f"{tag} DIAG", flush=True)
        diag_log = outdir / "diag.log"
        with diag_log.open("w", encoding="utf-8") as log:
            diag_proc = subprocess.run(diag_cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if diag_proc.returncode != 0:
            print(f"{tag} DIAG FAILED rc={diag_proc.returncode}", flush=True)
            print(diag_log.read_text(encoding="utf-8")[-3000:], flush=True)
        else:
            with (outdir / "diagnostics.json").open(encoding="utf-8") as f:
                diag = json.load(f)
            diag_text = diag_log.read_text(encoding="utf-8")
            mem_match = re.search(r"peak_allocated_mb=([0-9.]+)", diag_text)
            diag["diag_peak_allocated_mb"] = float(mem_match.group(1)) if mem_match else None

    results = None
    results_json = outdir / "hydra" / "results.json"
    if results_json.exists():
        with results_json.open(encoding="utf-8") as f:
            results = json.load(f)
    log_info = _parse_train_log(train_log)
    summary = {
        "dataset": dataset,
        "mode": mode,
        "seed": seed,
        "results": results,
        "best_val_macro_f1": _parse_best_val_f1(train_log),
        "params": log_info["params"],
        "runtime_sec": round(runtime_sec, 1),
        "epoch_time_sec": round(runtime_sec / log_info["epochs_run"], 2) if log_info["epochs_run"] else None,
        "epochs_run": log_info["epochs_run"],
        "train_peak_gpu_mb": peak_gpu_mb,
        "diagnostics": diag,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    best_val = (results or {}).get("val_acc", {}).get("mean") if results else None
    print(f"{tag} OK (best val acc={best_val}, {runtime_sec:.0f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="P3-B low-rank driver (30-run study)")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--modes", default=None, help="comma-separated subset of LR-ADD/LR-INT")
    parser.add_argument("--seeds", default=None, help="comma-separated seed override")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--out-root", default=None, help="override output root (smoke experiments)")
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument(
        "--mode-def",
        action="append",
        default=[],
        help="register a custom mode: label=comma,separated,hydra,overrides (repeatable)"
        "; e.g. LR-INT-r8=model.p3.operator_mode=lowrank_interaction,model.p3.lowrank_rank=8",
    )
    args = parser.parse_args()

    for mode_def in args.mode_def:
        label, _, overrides = mode_def.partition("=")
        MODE_OVERRIDES[label.strip()] = [o.strip() for o in overrides.split(",") if o.strip()]

    datasets = NC_DATASETS
    if args.datasets:
        requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
        datasets = [d for d in NC_DATASETS if d in requested]
    modes = list(MODE_OVERRIDES)
    if args.modes:
        modes = [item.strip() for item in args.modes.split(",") if item.strip()]
        unknown = [m for m in modes if m not in MODE_OVERRIDES]
        if unknown:
            parser.error(f"unknown modes: {unknown}")
    seeds = CONFIRM_SEEDS
    if args.seeds:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    gpus = [int(gpu.strip()) for gpu in args.gpus.split(",") if gpu.strip()]
    out_root = args.out_root or "outputs/p3/lowrank"

    order = {"ele-fashion": 0, "Reddit-S": 1, "Grocery": 2, "Toys": 3, "Movies": 4}
    jobs = sorted([(d, m, s) for d in datasets for m in modes for s in seeds], key=lambda j: order.get(j[0], 9))
    pinned: dict[str, int] = {}
    gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
    slots = max(1, int(args.slots_per_gpu))
    gpu_locks = {gpu_id: _WeightedSemaphore(slots) for gpu_id in gpus}
    print(
        f"[driver] jobs={len(jobs)} gpus={gpus} slots/gpu={slots} out={out_root}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=len(gpus) * slots) as executor:
        futures = {}
        for job in jobs:
            dataset, mode, seed = job
            gpu_id = pinned.get(dataset, next(gpu_iter))
            futures[executor.submit(
                _run_job, dataset, mode, seed, gpu_id, out_root, args.force,
                args.epochs, args.no_diagnostics, gpu_locks, len(seeds), slots
            )] = job
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)

    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
