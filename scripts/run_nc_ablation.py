"""Paper-facing main-story ablation driver (plan §23/§24).

Runs model=biaxis_ablation with ablation.mode=<mode> into
outputs/final_nc_ablation/main_story/<dataset>/<mode>/seed_<s>/.

New runs: no_factor_axis / no_relation_axis / no_adaptive_allocation
(5 datasets x 3 modes x 3 seeds = 45). full_reference reuses the final
benchmark runs; shared_operator / no_cell_correction reuse P3 O0 / OADD
matched seeds (summarize side).

Resume/skip on results.json; per-GPU weighted slots (ele-fashion full card);
no deterministic mode; frozen files untouched.

Usage:
    python scripts/run_nc_ablation.py --gpus 0,1
    python scripts/run_nc_ablation.py --datasets Movies --modes no_factor_axis \
        --seeds 42 --epochs 5 --out-root outputs/final_nc_ablation/smoke
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

from run_p3_lowrank_screen import LARGE_DATASETS, _WeightedSemaphore  # noqa: E402
from run_p1_screen import _poll_peak_mem  # noqa: E402

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
MODES = ["no_factor_axis", "no_relation_axis", "no_adaptive_allocation"]
SEEDS = [42, 43, 44]
OUT_ROOT = PROJECT_ROOT / "outputs" / "final_nc_ablation" / "main_story"


def _run_job(dataset, mode, seed, gpu_id, force, epochs, gpu_locks, slots_per_gpu, out_root):
    lock = gpu_locks.get(gpu_id)
    weight = int(slots_per_gpu) if dataset in LARGE_DATASETS else 1
    if lock is not None:
        lock.acquire(weight)
    try:
        _run_job_locked(dataset, mode, seed, gpu_id, force, epochs, out_root)
    finally:
        if lock is not None:
            lock.release(weight)


def _run_job_locked(dataset, mode, seed, gpu_id, force, epochs, out_root):
    outdir = Path(out_root) / dataset / mode / f"seed_{seed}"
    tag = f"[{gpu_id}] {dataset} {mode} seed={seed}"
    results_json = outdir / "hydra" / "results.json"
    if results_json.exists() and not force:
        print(f"{tag} SKIP (results exist)", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, "-m", "src.main",
        f"dataset={dataset}", "task=nc", "model=biaxis_ablation",
        f"model.ablation.mode={mode}",
        "num_runs=1", f"seed={seed}", "device=cuda:0",
        f"hydra.run.dir={outdir / 'hydra'}",
    ]
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
    text = log_path.read_text(encoding="utf-8")
    params = None
    match = re.search(r"model\+head params=(\d+)", text)
    if match:
        params = int(match.group(1))
    with (outdir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset, "model": "biaxis_ablation", "mode": mode, "seed": seed,
            "params": params, "runtime_sec": round(runtime_sec, 1),
            "train_peak_gpu_mb": mem_holder.get("peak"),
        }, f, indent=2)
    print(f"{tag} OK ({runtime_sec:.0f}s, params={params})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Main-story ablation driver (45 runs)")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--modes", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()

    datasets = [d for d in (args.datasets or ",".join(DATASETS)).split(",") if d in DATASETS]
    modes = [m for m in (args.modes or ",".join(MODES)).split(",") if m in MODES]
    seeds = [int(s) for s in (args.seeds or ",".join(map(str, SEEDS))).split(",") if s]
    gpus = [int(g) for g in args.gpus.split(",") if g]
    slots = max(1, int(args.slots_per_gpu))
    gpu_locks = {g: _WeightedSemaphore(slots) for g in gpus}
    out_root = args.out_root or str(OUT_ROOT)

    jobs = sorted([(d, m, s) for d in datasets for m in modes for s in seeds],
                  key=lambda j: j[0] != "ele-fashion")
    print(f"[driver] {len(jobs)} jobs gpus={gpus} slots/gpu={slots} out={out_root}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus) * slots) as executor:
        futures = {}
        gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
        for dataset, mode, seed in jobs:
            futures[executor.submit(_run_job, dataset, mode, seed, next(gpu_iter),
                                    args.force, args.epochs, gpu_locks, slots, out_root)] = (dataset, mode, seed)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
