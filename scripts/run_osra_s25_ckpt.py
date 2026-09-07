"""OSRA S2.5 Stage-A prep driver: re-run Q4 (L3 parent + flat access with
Null) on Movies/Toys/Grocery x seeds 42/43/44 (=9 runs) WITH
task.save_ckpt_path so the causal usage audit can intervene on the exact
best checkpoints. Val-only, same frozen protocol as S2.

Outputs go to outputs/osra_next/q4_ckpt/<dataset>/seed_<s>/ (hydra +
history.csv + train.log + run_info.json + checkpoint .pt). Resume/skip on
existing results.json + checkpoint (--force overrides).

Usage:
    python scripts/run_osra_s25_ckpt.py --gpus 0,1
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

from run_p3_operator_screen import _WeightedSemaphore  # noqa: E402
from run_p1_screen import _poll_peak_mem  # noqa: E402

DATASETS = ["Movies", "Toys", "Grocery"]
SEEDS = [42, 43, 44]
OUT_ROOT = PROJECT_ROOT / "outputs" / "osra_next" / "q4_ckpt"
# the S2 Q4 variant overrides (L3 parent + flat access + Null)
Q4_OVERRIDES = [
    "model.osra.local_mode=gatv2",
    "model.osra.cross_access=true",
    "model.osra.access_mode=query",
    "model.osra.query_mode=target",
    "model.osra.use_null=true",
]


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


def _run_job_locked(dataset: str, seed: int, gpu_id: int, force: bool) -> None:
    outdir = OUT_ROOT / dataset / f"seed_{seed}"
    tag = f"[{gpu_id}] Q4-ckpt {dataset} seed={seed}"
    results_json = outdir / "hydra" / "results.json"
    ckpt = outdir / "checkpoint.pt"
    if results_json.exists() and ckpt.exists() and not force:
        print(f"{tag} SKIP (results+ckpt exist)", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        _resolve_python(), "-m", "src.main",
        f"dataset={dataset}", "task=nc", "model=osra",
        "num_runs=1", f"seed={seed}", "device=cuda:0",
        "task.evaluate_test=false",
        f"task.history_path={outdir / 'history.csv'}",
        f"task.save_ckpt_path={outdir / 'checkpoint.pt'}",
        *Q4_OVERRIDES,
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
                "dataset": dataset, "seed": seed, "gpu": gpu_id,
                "returncode": proc.returncode, "runtime_sec": round(runtime_sec, 1),
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
            "dataset": dataset, "seed": seed, "params": params,
            "runtime_sec": round(runtime_sec, 1),
            "train_peak_gpu_mb": mem_holder.get("peak"),
            "git_commit": _git_commit(),
            "cmd": " ".join(cmd),
        }, f, indent=2)
    print(f"{tag} OK ({runtime_sec:.0f}s, params={params})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="OSRA S2.5 Q4 checkpoint re-runs (9 runs)")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    gpu_locks = {g: _WeightedSemaphore(2) for g in gpus}
    jobs = sorted([(d, s) for d in DATASETS for s in SEEDS])
    print(f"[driver] {len(jobs)} jobs gpus={gpus} commit={_git_commit()[:8]}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus) * 2) as executor:
        futures = {}
        gpu_iter = iter(gpus * (len(jobs) // len(gpus) + 1))
        for dataset, seed in jobs:
            futures[executor.submit(_run_job_locked, dataset, seed, next(gpu_iter), args.force)] = (dataset, seed)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
