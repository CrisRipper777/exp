"""R1.5 experiment driver (plan §8/§12/§15): train biaxis_final variants
with evaluate_test=false + per-epoch history, save best checkpoints.

Families (outputs/perf_r15/<family>/<dataset>/<variant>/seed_<s>/):
    anchor    fresh current-code A0 (memory_checkpoint=true)  [R15-0]
    opt       LR x weight-decay screen                         [R15-2]
    capacity  hidden/factor capacity screen                    [R15-3]
    objective conditional objective/dropout/schedule contrasts [R15-4]

Every job: train via src.main (300ep/patience30 default) -> parse the
per-epoch history CSV into summary stats -> summary.json. Resume/skip on
existing summary.json (--force overrides). Per-GPU weighted semaphores keep
ele-fashion on a full card.

Usage:
    python scripts/run_perf_r15_screen.py --family anchor --gpus 0,1
    python scripts/run_perf_r15_screen.py --family opt --variants lr3e-4_wd0,lr3e-3_wd0 \
        --datasets Movies,Toys,Grocery --seeds 42
"""

from __future__ import annotations

import argparse
import csv
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

from run_p1_screen import _parse_train_log, _poll_peak_mem  # noqa: E402

NC_DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]

# Variant label -> hydra overrides (model=biaxis_final is fixed).
VARIANTS: dict[str, list[str]] = {
    "A0": [],
    # R15-2 LR x WD grid (baseline 1e-3/1e-4 is A0; plan §14).
    "lr3e-4_wd0": ["model.lr=0.0003", "model.weight_decay=0.0"],
    "lr3e-4_wd1e-4": ["model.lr=0.0003", "model.weight_decay=0.0001"],
    "lr3e-4_wd1e-3": ["model.lr=0.0003", "model.weight_decay=0.001"],
    "lr1e-3_wd0": ["model.lr=0.001", "model.weight_decay=0.0"],
    "lr1e-3_wd1e-3": ["model.lr=0.001", "model.weight_decay=0.001"],
    "lr3e-3_wd0": ["model.lr=0.003", "model.weight_decay=0.0"],
    "lr3e-3_wd1e-4": ["model.lr=0.003", "model.weight_decay=0.0001"],
    "lr3e-3_wd1e-3": ["model.lr=0.003", "model.weight_decay=0.001"],
    # R15-4 conditional objective/schedule contrasts (plan §R15-4, driven by
    # the R15-1 evidence: aux weak (R_aux/CE=0.088), no CE-aux conflict,
    # mild late plateau).
    "OBJ1_lc05": ["model.lambda_common=0.05"],
    "OBJ2_lc05_lr01": ["model.lambda_common=0.05", "model.lambda_recon=0.1"],
    "SCH1_cos": ["task.scheduler=warmup_cosine"],
    # R15-3 capacity candidates (plan §17).
    "C1_h384_f128": ["model.hidden_dim=384"],
    "C2_h512_f128": ["model.hidden_dim=512"],
    "C3_h384_f160": ["model.hidden_dim=384", "model.factor_dim=160"],
    "C4_h384_f192": ["model.hidden_dim=384", "model.factor_dim=192"],
}

LARGE_DATASETS = {"ele-fashion"}

# Per-family default variants (explicit --variants always wins).
FAMILY_VARIANTS: dict[str, list[str]] = {
    "anchor": ["A0"],
    "opt": ["lr3e-4_wd0", "lr3e-4_wd1e-4", "lr3e-4_wd1e-3",
            "lr1e-3_wd0", "lr1e-3_wd1e-3",
            "lr3e-3_wd0", "lr3e-3_wd1e-4", "lr3e-3_wd1e-3"],
    "capacity": ["C1_h384_f128", "C2_h512_f128", "C3_h384_f160", "C4_h384_f192"],
    "objective": ["OBJ1_lc05", "OBJ2_lc05_lr01", "SCH1_cos"],
}


class _WeightedSemaphore:
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


def _history_stats(history_path: Path) -> dict | None:
    if not history_path.exists():
        return None
    rows = []
    with history_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return None
    val = [float(r["val_acc"]) for r in rows]
    train = [float(r["train_acc"]) for r in rows]
    best_i = max(range(len(val)), key=lambda i: val[i])
    stop_epoch = int(rows[-1]["epoch"])
    best_epoch = int(rows[best_i]["epoch"])
    # last-20 val slope (pp/epoch, least squares on the final <=20 points)
    tail = val[-20:]
    xs = list(range(len(tail)))
    n = len(tail)
    slope = 0.0
    if n >= 2:
        mx, my = sum(xs) / n, sum(tail) / n
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, tail)) / max(
            sum((x - mx) ** 2 for x in xs), 1e-12
        )
    gap = train[best_i] - val[best_i]
    return {
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "best_train_acc": train[best_i],
        "best_val_acc": val[best_i],
        "train_val_gap_at_best": gap,
        "last20_val_slope_pp_per_epoch": 100.0 * slope,
        "hit_max_epoch": stop_epoch >= 300,
        "early_plateau": best_epoch <= int(0.4 * stop_epoch),
        "overfit_evidence": gap > 0.20,
        "final_lr": float(rows[-1]["lr"]),
    }


def _run_job(dataset, variant, seed, gpu_id, family, force, epochs,
             gpu_locks, slots_per_gpu):
    lock = gpu_locks.get(gpu_id)
    weight = int(slots_per_gpu) if dataset in LARGE_DATASETS else 1
    if lock is not None:
        lock.acquire(weight)
    try:
        _run_job_locked(dataset, variant, seed, gpu_id, family, force, epochs)
    finally:
        if lock is not None:
            lock.release(weight)


def _run_job_locked(dataset, variant, seed, gpu_id, family, force, epochs):
    overrides = list(VARIANTS[variant])
    outdir = PROJECT_ROOT / "outputs" / "perf_r15" / family / dataset / variant / f"seed_{seed}"
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
        f"dataset={dataset}", "task=nc", "model=biaxis_final",
        "num_runs=1", f"seed={seed}", "device=cuda:0",
        "task.evaluate_test=false",
        f"task.history_path={outdir / 'history.csv'}",
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

    results = None
    results_json = outdir / "hydra" / "results.json"
    if results_json.exists():
        with results_json.open(encoding="utf-8") as f:
            results = json.load(f)
    log_info = _parse_train_log(train_log)
    summary = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "results": results,
        "history": _history_stats(outdir / "history.csv"),
        "params": log_info["params"],
        "runtime_sec": round(runtime_sec, 1),
        "epoch_time_sec": round(runtime_sec / log_info["epochs_run"], 2) if log_info["epochs_run"] else None,
        "epochs_run": log_info["epochs_run"],
        "train_peak_gpu_mb": peak_gpu_mb,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    best_val = (results or {}).get("val_acc", {}).get("mean") if results else None
    print(f"{tag} OK (best val acc={best_val}, {runtime_sec:.0f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R1.5 screen driver (val-only, history)")
    parser.add_argument("--family", required=True, choices=["anchor", "opt", "capacity", "objective"])
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None, help="comma-separated subset of registered variants")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds (default 42)")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    args = parser.parse_args()

    datasets = NC_DATASETS
    if args.datasets:
        requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
        datasets = [d for d in NC_DATASETS if d in requested]
    variants = list(FAMILY_VARIANTS[args.family])
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in variants if v not in VARIANTS]
        if unknown:
            parser.error(f"unknown variants: {unknown}")
    if not variants:
        parser.error(f"family={args.family} has no default variants; pass --variants")
    seeds = [42]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

    order = {"ele-fashion": 0, "Reddit-S": 1, "Grocery": 2, "Toys": 3, "Movies": 4}
    jobs = sorted([(d, v, s) for d in datasets for v in variants for s in seeds],
                  key=lambda j: order.get(j[0], 9))
    pinned: dict[str, int] = {}
    gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
    slots = max(1, int(args.slots_per_gpu))
    gpu_locks = {g: _WeightedSemaphore(slots) for g in gpus}
    print(f"[driver] family={args.family} jobs={len(jobs)} gpus={gpus} slots/gpu={slots}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus) * slots) as executor:
        futures = {}
        for job in jobs:
            dataset, variant, seed = job
            gpu_id = pinned.get(dataset, next(gpu_iter))
            futures[executor.submit(
                _run_job, dataset, variant, seed, gpu_id, args.family,
                args.force, args.epochs, gpu_locks, slots
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
