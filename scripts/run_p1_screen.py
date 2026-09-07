"""P1 experiment driver: train biaxis_p1 variants via the FROZEN nc.py
trainer, then run best-checkpoint mechanism diagnostics (plan §25-§31, §40).

Stages:
    screen           5 NC datasets x 4 variants x seed 42        (20 runs)
    confirm          5 NC datasets x 4 variants x seeds 42/43/44 (60 runs)
    budget_ablation  Movies/Grocery/ele-fashion x B0/B1/B2 x seed 42 (9 runs)

One job = one (dataset, variant, seed):
    1. train:  python -m src.main dataset=... task=nc model=biaxis_p1 ...
               (existing frozen trainer; checkpoint via task.save_ckpt_path)
    2. analyze: scripts/analyze_p1_checkpoint.py -> diagnostics.json
    3. write summary.json (results + params + runtime + peak mem + diagnostics)

Resume/skip: a completed run (summary.json present) is skipped unless --force.
A failed run never overwrites existing results. Jobs run in parallel across
the given GPUs (ele-fashion is pinned to the LAST gpu by default — its full-
graph forward peaks >10GB).

Usage:
    python scripts/run_p1_screen.py --stage screen --gpus 0,1
    python scripts/run_p1_screen.py --stage screen --datasets Movies --epochs 5
    python scripts/run_p1_confirm.py --gpus 0,1
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

NC_DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SCREEN_SEEDS = [42]
CONFIRM_SEEDS = [42, 43, 44]

# Variant hydra overrides (plan §22). B0/B1/B2 are the F1R1 budget ablation
# (§30): B0 beta=1, B1 shared beta, B2 factor-specific (P1 default).
FACTORIAL_VARIANTS = {
    "F0R0": ["model.p1.factor_aware=false", "model.p1.num_relations=1"],
    "F1R0": ["model.p1.factor_aware=true", "model.p1.num_relations=1"],
    "F0R1": ["model.p1.factor_aware=false", "model.p1.num_relations=4"],
    "F1R1": ["model.p1.factor_aware=true", "model.p1.num_relations=4"],
}
BUDGET_VARIANTS = {
    "B0": ["model.p1.factor_aware=true", "model.p1.num_relations=4", "model.p1.use_graph_budget=false"],
    "B1": ["model.p1.factor_aware=true", "model.p1.num_relations=4", "model.p1.budget_shared=true"],
    "B2": ["model.p1.factor_aware=true", "model.p1.num_relations=4"],
}


def _stage_config(stage: str) -> tuple[list[str], list[str], list[int], str]:
    if stage == "screen":
        return NC_DATASETS, list(FACTORIAL_VARIANTS.keys()), SCREEN_SEEDS, "outputs/p1/screen"
    if stage == "confirm":
        return NC_DATASETS, list(FACTORIAL_VARIANTS.keys()), CONFIRM_SEEDS, "outputs/p1/confirm"
    if stage == "budget_ablation":
        return ["Movies", "Grocery", "ele-fashion"], list(BUDGET_VARIANTS.keys()), SCREEN_SEEDS, "outputs/p1/budget_ablation"
    raise ValueError(f"unknown stage {stage!r}")


def _variant_overrides(stage: str, variant: str) -> list[str]:
    table = BUDGET_VARIANTS if stage == "budget_ablation" else FACTORIAL_VARIANTS
    return table[variant]


def _run_dir(out_root: str, dataset: str, variant: str, seed: int, num_seeds: int) -> Path:
    base = Path(out_root) / dataset / variant
    if num_seeds > 1:
        return base / f"seed_{seed}"
    return base


def _poll_peak_mem(gpu_id: int, stop_event: threading.Event, holder: dict) -> None:
    """Poll GPU memory.used during training into ``holder["peak"]``
    (approximate: shared-GPU caveat, best-effort)."""
    peak = 0
    while not stop_event.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu_id)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                peak = max(peak, int(out.stdout.strip().splitlines()[0]))
        except Exception:  # noqa: BLE001 — polling is best-effort
            pass
        stop_event.wait(2.0)
    holder["peak"] = peak


def _parse_train_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    params = None
    match = re.search(r"model\+head params=(\d+)", text)
    if match:
        params = int(match.group(1))
    epochs = [int(value) for value in re.findall(r"Epoch (\d+)", text)]
    return {"params": params, "epochs_run": max(epochs) if epochs else None}


def _run_job(
    dataset: str,
    variant: str,
    seed: int,
    gpu_id: int,
    stage: str,
    out_root: str,
    force: bool,
    epochs: int | None,
    no_diagnostics: bool,
    extra_overrides: list[str] | None = None,
    gpu_locks: dict[int, threading.Semaphore] | None = None,
    num_seeds: int = 1,
) -> None:
    # Per-GPU mutual exclusion (review §21): the executor may schedule a job
    # whose pre-assigned GPU is still busy; the semaphore forces one process
    # per physical GPU so peak-memory/runtime numbers stay meaningful.
    lock = (gpu_locks or {}).get(gpu_id)
    if lock is not None:
        lock.acquire()
    try:
        _run_job_locked(
            dataset, variant, seed, gpu_id, stage, out_root, force, epochs, no_diagnostics,
            extra_overrides, num_seeds,
        )
    finally:
        if lock is not None:
            lock.release()


def _run_job_locked(
    dataset: str,
    variant: str,
    seed: int,
    gpu_id: int,
    stage: str,
    out_root: str,
    force: bool,
    epochs: int | None,
    no_diagnostics: bool,
    extra_overrides: list[str] | None = None,
    num_seeds: int = 1,
) -> None:
    """Job body; the caller holds the per-GPU semaphore."""
    overrides = _variant_overrides(stage, variant) + list(extra_overrides or [])
    outdir = _run_dir(out_root, dataset, variant, seed, num_seeds)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"[{gpu_id}] {dataset} {variant} seed={seed}"
    summary_path = outdir / "summary.json"
    if summary_path.exists() and not force:
        print(f"{tag} SKIP (summary exists, --force to rerun)", flush=True)
        return

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # --- 1. training (frozen nc.py trainer) --------------------------------
    train_cmd = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=nc",
        "model=biaxis_p1",
        "num_runs=1",
        f"seed={seed}",
        "device=cuda:0",
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
    print(f"{tag} TRAIN {' '.join(train_cmd[2:])}", flush=True)
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

    # --- 2. mechanism diagnostics on the best checkpoint -------------------
    diag = None
    if not no_diagnostics and (outdir / "model.pt").exists():
        diag_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "analyze_p1_checkpoint.py"),
            "--dataset", dataset,
            "--task", "nc",
            "--seed", str(seed),
            "--ckpt", str(outdir / "model.pt"),
            "--out", str(outdir),
            "--device", "cuda:0",
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
            diag_text = diag_log.read_text(encoding="utf-8")
            mem_match = re.search(r"peak_allocated_mb=([0-9.]+)", diag_text)
            with (outdir / "diagnostics.json").open(encoding="utf-8") as f:
                diag = json.load(f)
            diag["diag_peak_allocated_mb"] = float(mem_match.group(1)) if mem_match else None

    # --- 3. summary ----------------------------------------------------------
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
        "stage": stage,
        "results": results,
        "params": log_info["params"],
        "runtime_sec": round(runtime_sec, 1),
        "epoch_time_sec": round(runtime_sec / log_info["epochs_run"], 2) if log_info["epochs_run"] else None,
        "epochs_run": log_info["epochs_run"],
        "train_peak_gpu_mb": peak_gpu_mb,
        "diagnostics": diag,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    best_val = _metric_mean(results, "val_acc")
    print(f"{tag} OK (best val acc={best_val}, {runtime_sec:.0f}s)", flush=True)


def _metric_mean(results: dict | None, key: str):
    if not results:
        return None
    entry = results.get(key)
    if isinstance(entry, dict):
        return entry.get("mean")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 biaxis experiment driver (screen / confirm / budget_ablation)")
    parser.add_argument("--stage", default="screen", choices=["screen", "confirm", "budget_ablation"])
    parser.add_argument("--datasets", default=None, help="comma-separated subset of datasets")
    parser.add_argument("--variants", default=None, help="comma-separated subset of variants")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true", help="rerun completed runs")
    parser.add_argument("--epochs", type=int, default=None, help="override train epochs (smoke only)")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument(
        "--out-root",
        default=None,
        help="override output root (diagnostic/revise experiments, e.g. outputs/p1/revise_b)",
    )
    parser.add_argument(
        "--model-overrides",
        default="",
        help="extra comma-separated overrides appended to every job (revise experiments)",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="comma-separated seed override (supplementary experiments, e.g. 43,44)",
    )
    args = parser.parse_args()

    datasets_all, variants, seeds, out_root = _stage_config(args.stage)
    if args.out_root:
        out_root = args.out_root
    if args.seeds:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    extra_overrides = [item.strip() for item in args.model_overrides.split(",") if item.strip()]
    datasets = datasets_all
    if args.datasets:
        requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
        datasets = [d for d in datasets_all if d in requested]
    if args.variants:
        requested_variants = [item.strip() for item in args.variants.split(",") if item.strip()]
        variants = [v for v in variants if v in requested_variants]
    gpus = [int(gpu.strip()) for gpu in args.gpus.split(",") if gpu.strip()]

    # Heavy graphs first so the long pole starts immediately.
    order = {"ele-fashion": 0, "Reddit-S": 1, "Grocery": 2, "Toys": 3, "Movies": 4}
    jobs = sorted(
        [(d, v, s) for d in datasets for v in variants for s in seeds],
        key=lambda job: order.get(job[0], 9),
    )

    # No dataset pinning: ele-fashion peaks ~13GB, both 24GB GPUs are free.
    # (Pin manually via a driver edit only if a GPU is shared with other work.)
    pinned: dict[str, int] = {}
    gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
    print(
        f"[driver] stage={args.stage} jobs={len(jobs)} gpus={gpus} "
        f"out={out_root} epochs={args.epochs} force={args.force}",
        flush=True,
    )
    gpu_locks = {gpu_id: threading.Semaphore(1) for gpu_id in gpus}
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for job in jobs:
            dataset, variant, seed = job
            gpu_id = pinned.get(dataset, next(gpu_iter))
            futures[executor.submit(
                _run_job, dataset, variant, seed, gpu_id, args.stage, out_root, args.force,
                args.epochs, args.no_diagnostics, extra_overrides, gpu_locks, len(seeds)
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
