"""R2-Design-1 experiment driver (plan §25-§31).

Trains the four R2 variants (B0/F/S/J — ONE implementation, config toggles)
via the FROZEN nc.py trainer, then runs best-checkpoint mechanism
diagnostics and saves a per-run summary with the §34 fields.

Stage usage:
    screen   : --datasets Movies,Toys,Grocery --variants B0 --seeds 42    (D1-1)
               --variants F / S                                          (D1-2 / D1-3)
               --variants J (only after the pre-registered gates)        (D1-4)
    guards   : --datasets ele-fashion,Reddit-S --seeds 42                 (D1-5A)
    confirm  : --datasets Movies,Toys,Grocery,ele-fashion,Reddit-S
               --seeds 42,43,44 (seed42 rows are reused when present)    (D1-5B)

Protocol is hard-coded (plan §20/§33): evaluate_test=false, 300 epochs,
patience 30, best by Val Acc, AdamW lr=1e-3 wd=1e-4, full graph. Val only —
this driver NEVER touches test.

One job = (dataset, variant, seed): train via src.main -> analyze via
scripts/analyze_perf_r2_checkpoint.py -> summary.json. Resume/skip on
existing summary.json (--force overrides). Per-GPU weighted semaphores keep
ele-fashion on a full card (P3/R1 driver policy).

Usage:
    python scripts/run_perf_r2d1.py --gpus 0,1 --datasets Movies --variants B0 \
        --seeds 42 --epochs 5 --no-diagnostics                            # smoke
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_p1_screen import _parse_train_log, _poll_peak_mem  # noqa: E402
from src.analysis.perf_r2_utils import VARIANT_ROOTS, VARIANT_YAMLS, VARIANTS  # noqa: E402

NC_DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]

# Datasets whose training peak claims a full card (P3/R1 driver policy).
LARGE_DATASETS = {"ele-fashion"}


def _parse_history(history_path: Path) -> dict:
    """Best-epoch bookkeeping from the per-epoch history CSV (§34):
    best_epoch / stop_epoch / best_val_acc / best_val_macro_f1 /
    train_acc_at_best / train_loss_at_best."""
    out: dict = {
        "best_epoch": None,
        "stop_epoch": None,
        "best_val_acc": None,
        "best_val_macro_f1": None,
        "train_acc_at_best": None,
        "train_loss_at_best": None,
    }
    if not history_path.exists():
        return out
    best_val = -1.0
    with history_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epoch = int(row["epoch"])
            out["stop_epoch"] = epoch
            val_acc = float(row["val_acc"])
            if val_acc > best_val:
                best_val = val_acc
                out["best_epoch"] = epoch
                out["best_val_acc"] = val_acc
                out["best_val_macro_f1"] = float(row["val_macro_f1"])
                out["train_acc_at_best"] = float(row["train_acc"])
                out["train_loss_at_best"] = float(row["train_total_loss"])
    return out


class _WeightedSemaphore:
    """Per-GPU slot pool: large jobs take all slots (full card), small jobs
    take 1 and pack together."""

    def __init__(self, value: int) -> None:
        self._cond = threading.Condition()
        self._value = int(value)

    def acquire(self, n: int = 1) -> None:
        with self._cond:
            while self._value < n:
                self._cond.wait()
            self._value -= n

    def release(self, n: int = 1) -> None:
        with self._cond:
            self._value += n
            self._cond.notify_all()


def _run_job(
    dataset: str,
    variant: str,
    seed: int,
    gpu_id: int,
    force: bool,
    epochs: int | None,
    no_diagnostics: bool,
    gpu_locks: dict[int, _WeightedSemaphore],
    slots_per_gpu: int = 1,
    out_root: Path | None = None,
) -> None:
    lock = gpu_locks.get(gpu_id)
    weight = int(slots_per_gpu) if dataset in LARGE_DATASETS else 1
    if lock is not None:
        lock.acquire(weight)
    try:
        _run_job_locked(dataset, variant, seed, gpu_id, force, epochs, no_diagnostics, out_root)
    finally:
        if lock is not None:
            lock.release(weight)


def _run_job_locked(
    dataset: str,
    variant: str,
    seed: int,
    gpu_id: int,
    force: bool,
    epochs: int | None,
    no_diagnostics: bool,
    out_root: Path | None = None,
) -> None:
    # Seed subdirs always (seed42 screen merges with 43/44 later, plan §31).
    base = out_root if out_root is not None else PROJECT_ROOT / "outputs" / "perf_r2d1" / VARIANT_ROOTS[variant]
    outdir = base / dataset / variant / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"[{gpu_id}] {dataset} {variant} seed={seed}"
    summary_path = outdir / "summary.json"
    if summary_path.exists() and not force:
        print(f"{tag} SKIP (summary exists)", flush=True)
        return

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    train_cmd = [
        sys.executable, "-m", "src.main",
        f"dataset={dataset}", "task=nc", f"model={VARIANT_YAMLS[variant]}",
        "num_runs=1", f"seed={seed}", "device=cuda:0",
        # R2-Design-1 protocol (plan §20/§33): Val only, NEVER test.
        "task.evaluate_test=false",
        f"task.save_ckpt_path={outdir / 'model.pt'}",
        f"task.history_path={outdir / 'history.csv'}",
        f"hydra.run.dir={outdir / 'hydra'}",
    ]
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
            str(PROJECT_ROOT / "scripts" / "analyze_perf_r2_checkpoint.py"),
            "--dataset", dataset, "--seed", str(seed), "--variant", variant,
            "--ckpt", str(outdir / "model.pt"),
            "--out", str(outdir), "--device", "cuda:0",
        ]
        if out_root is not None:
            # out_root is the per-variant base (e.g. outputs/perf_r2d15/b0_confirm)
            diag_cmd += ["--ckpt-root", str(out_root)]
        print(f"{tag} DIAG", flush=True)
        diag_log = outdir / "diag.log"
        with diag_log.open("w", encoding="utf-8") as log:
            diag_proc = subprocess.run(diag_cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if diag_proc.returncode != 0:
            print(f"{tag} DIAG FAILED rc={diag_proc.returncode}", flush=True)
            print(diag_log.read_text(encoding="utf-8")[-3000:], flush=True)
        else:
            with (outdir / "r2_diagnostics.json").open(encoding="utf-8") as f:
                diag = json.load(f)
            diag_text = diag_log.read_text(encoding="utf-8")
            mem_match = re.search(r"peak_allocated_mb=([0-9.]+)", diag_text)
            diag["diag_peak_allocated_mb"] = float(mem_match.group(1)) if mem_match else None

    log_info = _parse_train_log(train_log)
    history_info = _parse_history(outdir / "history.csv")
    summary = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "parameter_count": log_info["params"],
        "peak_gpu_mb": peak_gpu_mb,
        "runtime_sec": round(runtime_sec, 1),
        "epoch_time_sec": round(runtime_sec / log_info["epochs_run"], 2) if log_info["epochs_run"] else None,
        "epochs_run": log_info["epochs_run"],
        "best_epoch": history_info["best_epoch"],
        "stop_epoch": history_info["stop_epoch"],
        "best_val_acc": history_info["best_val_acc"],
        "best_val_macro_f1": history_info["best_val_macro_f1"],
        "train_acc_at_best": history_info["train_acc_at_best"],
        "train_loss_at_best": history_info["train_loss_at_best"],
        "diagnostics": diag,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"{tag} OK (best val acc={summary['best_val_acc']}, "
        f"ep={summary['best_epoch']}/{summary['stop_epoch']}, {runtime_sec:.0f}s)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-Design-1 experiment driver (Val only)")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None, help="comma-separated subset of B0/F/S/J")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds (default 42)")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument(
        "--slots-per-gpu",
        type=int,
        default=1,
        help="concurrent jobs per GPU: large datasets (ele-fashion) take ALL "
        "slots (full card), small datasets take 1 and pack together",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="override output root (default outputs/perf_r2d1/<variant_root>)",
    )
    args = parser.parse_args()

    datasets = NC_DATASETS
    if args.datasets:
        requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
        datasets = [d for d in NC_DATASETS if d in requested]
    variants = list(VARIANTS)
    if args.variants:
        variants = [item.strip() for item in args.variants.split(",") if item.strip()]
        unknown = [v for v in variants if v not in VARIANTS]
        if unknown:
            parser.error(f"unknown variants: {unknown}")
    seeds = [42]
    if args.seeds:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    gpus = [int(gpu.strip()) for gpu in args.gpus.split(",") if gpu.strip()]

    order = {"ele-fashion": 0, "Reddit-S": 1, "Grocery": 2, "Toys": 3, "Movies": 4}
    jobs = sorted([(d, v, s) for d in datasets for v in variants for s in seeds], key=lambda j: order.get(j[0], 9))
    gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
    slots = max(1, int(args.slots_per_gpu))
    gpu_locks = {gpu_id: _WeightedSemaphore(slots) for gpu_id in gpus}
    out_root = Path(args.out_root) if args.out_root else None
    print(
        f"[driver] jobs={len(jobs)} gpus={gpus} slots/gpu={slots} "
        f"out={out_root or 'outputs/perf_r2d1/{b0,functional,semantic,joint}'} "
        f"evaluate_test=false",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=len(gpus) * slots) as executor:
        futures = {}
        for job in jobs:
            dataset, variant, seed = job
            gpu_id = next(gpu_iter)
            futures[executor.submit(
                _run_job, dataset, variant, seed, gpu_id, args.force,
                args.epochs, args.no_diagnostics, gpu_locks, slots, out_root
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
