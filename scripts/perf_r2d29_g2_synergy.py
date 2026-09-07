"""R2D29 G2 driver: Full System Synergy Matrix (plan §7).

2x2x2x2 factorial over R (router) x S (source) x W (writeback) x F (fusion)
= 16 variants, fixed backbone_mode=a0_augment + num_blocks=1. Each variant
runs 5 NC datasets x seeds 42/43/44 = 240 runs into
outputs/r2d29/g2_synergy/main/<dataset>/<cell>/seed_<s>/.

Discipline (plan §7.2): the full factorial MUST be executed — no early stop
because a main effect looks weak; Val-only; no single-seed filtering.

Phases:
    --cells        the 240-run factorial (default)
    --matched      MEAN_DUP matched controls for the named S1 cells
                   (plan §7.5; --matched-cells R1S1W1F1,... default: none)

Usage:
    python scripts/perf_r2d29_g2_synergy.py --gpus 0,1
    python scripts/perf_r2d29_g2_synergy.py --matched --matched-cells R1S1W1F1,R0S1W1F1
    python scripts/perf_r2d29_g2_synergy.py --datasets Movies --cells R0S0W0F0 --seeds 42  # smoke
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
from src.analysis.perf_r2d29_utils import (  # noqa: E402
    DATASETS,
    G2_CELLS,
    G2_FIXED,
    G2_MATCHED_CONTROLS,
    G2_ROOT,
    SEEDS,
)

OUT_ROOT = G2_ROOT


def _resolve_python() -> str:
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
    cell: str,
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
        _run_job_locked(dataset, cell, seed, gpu_id, force)
    finally:
        if lock is not None:
            lock.release(weight)


def _run_job_locked(dataset: str, cell: str, seed: int, gpu_id: int, force: bool) -> None:
    outdir = OUT_ROOT / "main" / dataset / cell / f"seed_{seed}"
    tag = f"[{gpu_id}] {dataset} {cell} seed={seed}"
    results_json = outdir / "hydra" / "results.json"
    if results_json.exists() and not force:
        print(f"{tag} SKIP (results exist)", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    overrides = {**G2_FIXED, **G2_CELLS.get(cell, G2_MATCHED_CONTROLS.get(cell, {}))}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        _resolve_python(), "-m", "src.main",
        f"dataset={dataset}", "task=nc", "model=biaxis_cort",
        "num_runs=1", f"seed={seed}", "device=cuda:0",
        f"hydra.run.dir={outdir / 'hydra'}",
    ]
    for key, value in overrides.items():
        if isinstance(value, bool):
            cmd.append(f"model.cort.{key}={'true' if value else 'false'}")
        else:
            cmd.append(f"model.cort.{key}={value}")
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
        tail = log_path.read_text(encoding="utf-8")[-2500:]
        print(tail, flush=True)
        with (OUT_ROOT / "failures.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "dataset": dataset, "cell": cell, "seed": seed, "gpu": gpu_id,
                "returncode": proc.returncode,
                "runtime_sec": round(runtime_sec, 1),
                "log_tail": tail,
            }) + "\n")
        return
    text = log_path.read_text(encoding="utf-8")
    params = None
    match = re.search(r"model\+head params=(\d+)", text)
    if match:
        params = int(match.group(1))
    with (outdir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset, "cell": cell, "seed": seed,
            "params": params,
            "runtime_sec": round(runtime_sec, 1),
            "train_peak_gpu_mb": mem_holder.get("peak"),
            "git_commit": _git_commit(),
        }, f, indent=2)
    print(f"{tag} OK ({runtime_sec:.0f}s, params={params})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2D29 G2 synergy-matrix driver")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--cells", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--matched-cells", default=None)
    args = parser.parse_args()

    datasets = DATASETS
    if args.datasets:
        datasets = [d for d in args.datasets.split(",") if d in DATASETS]
    if args.matched:
        if args.matched_cells:
            # accept either the base cell (R1S1W1F0) or the full
            # matched-control name (R1S1W1F0+MEAN_DUP)
            requested = args.matched_cells.split(",")
            cells = {
                mc: cfg for mc, cfg in G2_MATCHED_CONTROLS.items()
                if mc in requested or mc.split("+")[0] in requested
            }
        else:
            cells = dict(G2_MATCHED_CONTROLS)
    else:
        cells = dict(G2_CELLS)
        if args.cells:
            cells = {c: G2_CELLS[c] for c in args.cells.split(",") if c in G2_CELLS}
    seeds = SEEDS
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s]
    gpus = [int(g) for g in args.gpus.split(",") if g]
    slots = max(1, int(args.slots_per_gpu))
    gpu_locks = {g: _WeightedSemaphore(slots) for g in gpus}

    jobs = sorted(
        [(d, c, s) for d in datasets for c in cells for s in seeds],
        key=lambda j: j[0] != "ele-fashion",
    )
    print(f"[driver] {len(jobs)} jobs ({len(cells)} cells) gpus={gpus} "
          f"slots/gpu={slots} commit={_git_commit()[:8]}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus) * slots) as executor:
        futures = {}
        gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
        for dataset, cell, seed in jobs:
            futures[executor.submit(
                _run_job, dataset, cell, seed, next(gpu_iter), args.force, gpu_locks, slots
            )] = (dataset, cell, seed)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
