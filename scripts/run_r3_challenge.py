"""R3-1 Challenge Set driver (plan §20-§22, extended per user: 5 datasets
+ test reporting).

Runs the 7 R3 variants V0-V6 on Movies/Toys/Grocery/ele-fashion/Reddit-S
x seeds 42/43/44 (=105 runs) under the unified NC protocol. Model
selection stays VAL-ONLY (early stopping on val accuracy); task.
evaluate_test=true additionally reports test metrics from the best-VAL
checkpoint (the standard benchmark protocol — test is evaluated once after
the checkpoint is frozen, never used for selection). Every variant is the
single biaxis_r3 code path + config switches (plan §16.1).

Outputs go to outputs/r3/r3_1_challenge/<variant>/<dataset>/seed_<s>/ with
hydra results.json + history.csv + train.log + run_info.json sidecar
(params / peak GPU mem / runtime / git commit). Resume/skip on existing
COMPLETE results.json (contains test_acc; --force overrides); failed runs
are recorded (failures.jsonl) and never silently skipped. ele-fashion gets
exclusive GPU slots (memory).

Usage:
    python scripts/run_r3_challenge.py --gpus 0,1
    python scripts/run_r3_challenge.py --variants V0 --datasets Movies --seeds 42  # smoke
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

# plan §21: each variant is a set of model.transition.* overrides on top of
# configs/model/biaxis_r3.yaml (transition_mode=basis, cross_factor=true,
# use_dual_space=true, use_same_node_context=true,
# preserve_source_channels=true, multi_scale=concat).
VARIANTS: dict[str, list[str]] = {
    "V0": [
        "model.transition.cross_factor=false",
        "model.transition.transition_mode=diagonal",
        "model.transition.use_same_node_context=false",
        "model.transition.multi_scale=last",
    ],
    "V1": [
        "model.transition.transition_mode=static",
        "model.transition.use_same_node_context=false",
        "model.transition.multi_scale=last",
    ],
    "V2": [
        "model.transition.transition_mode=basis",
        "model.transition.use_dual_space=false",
        "model.transition.use_same_node_context=false",
        "model.transition.preserve_source_channels=false",
        "model.transition.multi_scale=last",
    ],
    "V3": [
        "model.transition.transition_mode=basis",
        "model.transition.use_same_node_context=false",
        "model.transition.multi_scale=last",
    ],
    "V4": [
        "model.transition.transition_mode=basis",
        "model.transition.multi_scale=last",
    ],
    "V5": [
        # FULL: biaxis_r3.yaml defaults (basis + dual + context + preserve + concat)
    ],
    "V6": [
        "model.transition.transition_mode=film",
    ],
}

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
OUT_ROOT = PROJECT_ROOT / "outputs" / "r3" / "r3_1_challenge"


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
    variant: str,
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
        _run_job_locked(variant, dataset, seed, gpu_id, force)
    finally:
        if lock is not None:
            lock.release(weight)


def _run_job_locked(variant: str, dataset: str, seed: int, gpu_id: int, force: bool) -> None:
    outdir = OUT_ROOT / variant / dataset / f"seed_{seed}"
    tag = f"[{gpu_id}] {variant} {dataset} seed={seed}"
    results_json = outdir / "hydra" / "results.json"
    if results_json.exists() and not force:
        # skip only COMPLETE runs (evaluate_test=true must have test metrics;
        # the earlier val-only batch is silently re-run in place)
        try:
            with results_json.open(encoding="utf-8") as f:
                payload = json.load(f)
            if "test_acc" in payload:
                print(f"{tag} SKIP (results exist)", flush=True)
                return
        except Exception:  # noqa: BLE001
            pass
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    history_path = str(outdir / "history.csv")
    cmd = [
        _resolve_python(), "-m", "src.main",
        f"dataset={dataset}", "task=nc", "model=biaxis_r3",
        "num_runs=1", f"seed={seed}", "device=cuda:0",
        "task.evaluate_test=true",
        f"task.history_path={history_path}",
        *VARIANTS[variant],
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
        with (OUT_ROOT / "failures.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "variant": variant, "dataset": dataset, "seed": seed, "gpu": gpu_id,
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
            "variant": variant, "dataset": dataset, "seed": seed,
            "params": params,
            "runtime_sec": round(runtime_sec, 1),
            "train_peak_gpu_mb": mem_holder.get("peak"),
            "git_commit": _git_commit(),
            "cmd": " ".join(cmd),
        }, f, indent=2)
    print(f"{tag} OK ({runtime_sec:.0f}s, params={params})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R3-1 challenge driver (V0-V6 x M/T/G x 3 seeds, val-only)")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    args = parser.parse_args()

    variants = VARIANTS
    if args.variants:
        variants = {v: VARIANTS[v] for v in args.variants.split(",") if v in VARIANTS}
    datasets = DATASETS
    if args.datasets:
        datasets = [d for d in args.datasets.split(",") if d in DATASETS]
    seeds = SEEDS
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s]
    gpus = [int(g) for g in args.gpus.split(",") if g]
    slots = max(1, int(args.slots_per_gpu))
    gpu_locks = {g: _WeightedSemaphore(slots) for g in gpus}

    # ele-fashion first (slowest, exclusive GPU slots), small datasets after
    jobs = sorted(
        [(v, d, s) for v in variants for d in datasets for s in seeds],
        key=lambda j: j[1] != "ele-fashion",
    )
    print(
        f"[driver] {len(jobs)} jobs gpus={gpus} slots/gpu={slots} commit={_git_commit()[:8]}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=len(gpus) * slots) as executor:
        futures = {}
        gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
        for variant, dataset, seed in jobs:
            futures[executor.submit(
                _run_job, variant, dataset, seed, next(gpu_iter), args.force, gpu_locks, slots
            )] = (variant, dataset, seed)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
