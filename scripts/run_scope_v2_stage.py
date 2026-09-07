"""Run the validation-only R4 SCOPE-MAG v2 stages.

The driver is intentionally small and resumable.  It launches one full-graph
job per GPU at a time; this is conservative for ele-fashion, whose node-side
activations are much larger than the other NC datasets.

Examples:
    python scripts/run_scope_v2_stage.py --stage parent --gpus 0,1
    python scripts/run_scope_v2_stage.py --stage repair --gpus 0,1
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

DATASETS = ["Movies", "Toys", "Grocery"]
SEEDS = [42, 43]
VARIANTS = {
    "parent": {
        "IND-L": ["model.scope.graph_mode=local", "model.scope.reconciliation.mode=none"],
        "IND-LG": ["model.scope.graph_mode=local_global", "model.scope.reconciliation.mode=none"],
    },
    "repair": {
        "IND": ["model.scope.graph_mode=local", "model.scope.reconciliation.mode=none"],
        "FULL": ["model.scope.graph_mode=local", "model.scope.reconciliation.mode=full"],
        "ORB": ["model.scope.graph_mode=local", "model.scope.reconciliation.mode=orb"],
    },
    "parent2": {
        # C0: protocol-corrected original parent, patience=30.
        "C0": [
            "model.scope.context_core=osc",
            "model.scope.graph_mode=local_global",
            "model.scope.reconciliation.mode=none",
            "model.scope.num_blocks=2",
        ],
        # C1/C2: OACC with one/two anchored blocks and node x factor beta.
        "C1": [
            "model.scope.context_core=oacc",
            "model.scope.graph_mode=local",
            "model.scope.reconciliation.mode=none",
            "model.scope.num_blocks=1",
            "model.scope.beta_mode=adaptive",
        ],
        "C2": [
            "model.scope.context_core=oacc",
            "model.scope.graph_mode=local",
            "model.scope.reconciliation.mode=none",
            "model.scope.num_blocks=2",
            "model.scope.beta_mode=adaptive",
        ],
    },
    "parent2_control": {
        "C1-BETA1": [
            "model.scope.context_core=oacc",
            "model.scope.graph_mode=local",
            "model.scope.reconciliation.mode=none",
            "model.scope.num_blocks=1",
            "model.scope.beta_mode=one",
        ],
        "C2-BETA1": [
            "model.scope.context_core=oacc",
            "model.scope.graph_mode=local",
            "model.scope.reconciliation.mode=none",
            "model.scope.num_blocks=2",
            "model.scope.beta_mode=one",
        ],
    },
    "repair2": {
        # Winner of R4-1b: two-block OACC with the beta=1 control.  All three
        # variants use this identical ContextCore; only post-context
        # reconciliation changes.
        "IND": [
            "model.scope.context_core=oacc",
            "model.scope.graph_mode=local",
            "model.scope.num_blocks=2",
            "model.scope.beta_mode=one",
            "model.scope.reconciliation.mode=none",
        ],
        "FULL": [
            "model.scope.context_core=oacc",
            "model.scope.graph_mode=local",
            "model.scope.num_blocks=2",
            "model.scope.beta_mode=one",
            "model.scope.reconciliation.mode=full",
        ],
        "ORB": [
            "model.scope.context_core=oacc",
            "model.scope.graph_mode=local",
            "model.scope.num_blocks=2",
            "model.scope.beta_mode=one",
            "model.scope.reconciliation.mode=orb",
        ],
    },
}
OUT_ROOT = PROJECT_ROOT / "outputs" / "r4_scope_v2"


def _poll_peak_mem(gpu_id: int, stop: threading.Event, holder: dict[str, int]) -> None:
    """Best-effort peak memory poll without importing the larger R3 driver."""
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--id", str(gpu_id), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            value = int(out.stdout.strip().splitlines()[0])
            holder["peak"] = max(value, holder.get("peak", 0))
        except Exception:
            pass
        stop.wait(2.0)


def _run_one(stage: str, variant: str, dataset: str, seed: int, gpu: int, force: bool) -> None:
    outdir = OUT_ROOT / stage / variant / dataset / f"seed_{seed}"
    result_path = outdir / "hydra" / "results.json"
    tag = f"[{gpu}] {stage}/{variant} {dataset} s{seed}"
    if result_path.exists() and not force:
        print(f"{tag} SKIP", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable, "-m", "src.main",
        f"dataset={dataset}", "task=nc", "model=biaxis_scope_v2",
        "num_runs=1", f"seed={seed}", "device=cuda:0",
        "task.evaluate_test=false", "task.patience=30", "task.epochs=300",
        f"task.history_path={outdir / 'history.csv'}",
        f"task.save_ckpt_path={outdir / 'best.pt'}",
        f"hydra.run.dir={outdir / 'hydra'}",
        *VARIANTS[stage][variant],
    ]
    log_path = outdir / "train.log"
    mem: dict[str, int] = {}
    stop = threading.Event()
    watcher = threading.Thread(target=_poll_peak_mem, args=(gpu, stop, mem), daemon=True)
    watcher.start()
    print(f"{tag} TRAIN", flush=True)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    stop.set()
    watcher.join(timeout=5)
    runtime = time.monotonic() - start
    info = {
        "stage": stage, "variant": variant, "dataset": dataset, "seed": seed,
        "gpu": gpu, "runtime_sec": round(runtime, 1), "peak_gpu_mb": mem.get("peak"),
        "returncode": proc.returncode,
    }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"model\+head params=(\d+)", text)
    if match:
        info["params"] = int(match.group(1))
    (outdir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if proc.returncode:
        print(f"{tag} FAILED rc={proc.returncode}\n{text[-2500:]}", flush=True)
    else:
        print(f"{tag} OK runtime={runtime:.0f}s peak={mem.get('peak')}", flush=True)


def _summarize(stage: str, variants: list[str]) -> None:
    path = OUT_ROOT / stage / "runs.csv"
    rows = []
    for variant in variants:
        for dataset in DATASETS:
            for seed in SEEDS:
                outdir = OUT_ROOT / stage / variant / dataset / f"seed_{seed}"
                result = outdir / "hydra" / "results.json"
                if not result.exists():
                    continue
                try:
                    payload = json.loads(result.read_text(encoding="utf-8"))
                    val = payload["val_acc"]["mean"]
                    info = json.loads((outdir / "run_info.json").read_text(encoding="utf-8"))
                    best_epoch = None
                    best_f1 = None
                    history = outdir / "history.csv"
                    if history.exists():
                        with history.open(encoding="utf-8") as hf:
                            hist_rows = list(csv.DictReader(hf))
                        if hist_rows:
                            best_row = max(hist_rows, key=lambda row: float(row["val_acc"]))
                            best_epoch = int(best_row["epoch"])
                            best_f1 = float(best_row["val_macro_f1"])
                    rows.append({"stage": stage, "variant": variant, "dataset": dataset,
                                 "seed": seed, "val_acc": val, "best_epoch": best_epoch,
                                 "best_val_macro_f1": best_f1,
                                 "params": info.get("params"), "peak_gpu_mb": info.get("peak_gpu_mb"),
                                 "runtime_sec": info.get("runtime_sec")})
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    continue
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "variant", "dataset", "seed", "val_acc", "best_epoch", "best_val_macro_f1", "params", "peak_gpu_mb", "runtime_sec"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] wrote {path} ({len(rows)} rows)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--variants", default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--root", default="r4_scope_v2", help="subdirectory under outputs/")
    args = parser.parse_args()
    global OUT_ROOT
    OUT_ROOT = PROJECT_ROOT / "outputs" / args.root
    datasets = [x for x in args.datasets.split(",") if x in DATASETS]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    variants = [x for x in (args.variants.split(",") if args.variants else VARIANTS[args.stage]) if x in VARIANTS[args.stage]]
    gpus = [int(x) for x in args.gpus.split(",") if x]
    jobs = [(v, d, s) for v in variants for d in datasets for s in seeds]
    print(f"[driver] stage={args.stage} jobs={len(jobs)} gpus={gpus}; one job/GPU", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(_run_one, args.stage, v, d, s, gpus[i % len(gpus)], args.force)
                   for i, (v, d, s) in enumerate(jobs)]
        for future in as_completed(futures):
            future.result()
    _summarize(args.stage, variants)


if __name__ == "__main__":
    main()
