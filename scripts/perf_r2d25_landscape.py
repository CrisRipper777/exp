"""R2-D2.5-A: formal alpha_Pt objective landscape
(docs/BiAxis_R2_Design_2_5_Structured_Capacity_Utilization_Audit.md).

For each (dataset, seed): load the B0 best checkpoint, freeze the parent,
precompute the fixed-alpha pipeline (v(alpha) = v1 + alpha*(v2 - v1) on the
Pt factor, alpha_C = alpha_Pv = 0), and train a FRESH classifier per
alpha_Pt in {0, 0.25, 0.5, 0.75, 1.0} — exact same classifier init across
alpha values (per dataset/seed), 300 ep / patience 30 / best Val Acc,
AdamW lr1e-3 wd1e-4. Val only.

Per alpha record: best Train CE / Train Acc, best Val CE / Val Acc /
Macro-F1, best epoch. At alpha in {0, 0.25, 0.5} with the trained head
FIXED compute diagnostic-only dTrainCE/dalpha_Pt and dValCE/dalpha_Pt
(never used to update parameters).

Outputs:
    outputs/perf_r2d25/landscape/<dataset>/seed_<s>/summary.json
    outputs/perf_r2d25/landscape/alpha_landscape.csv
    outputs/perf_r2d25/landscape/alpha_gradients.csv
    (report: scripts/summarize_perf_r2d25.py --stage landscape)

Usage:
    python scripts/perf_r2d25_landscape.py --gpus 0,1            # all
    python scripts/perf_r2d25_landscape.py --datasets Movies --seeds 42 --gpu 0 --epochs 5  # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d25_utils import (  # noqa: E402
    ALPHA_GRAD_VALUES,
    ALPHA_PT_VALUES,
    DATASETS,
    R2D25_ROOT,
    SEEDS,
)

LANDSCAPE_ROOT = R2D25_ROOT / "landscape"


class _Semaphore:
    def __init__(self, value: int) -> None:
        self._cond = threading.Condition()
        self._value = int(value)

    def acquire(self) -> None:
        with self._cond:
            while self._value < 1:
                self._cond.wait()
            self._value -= 1

    def release(self) -> None:
        with self._cond:
            self._value += 1
            self._cond.notify_all()


def _run_one(dataset: str, seed: int, gpu: int, force: bool, epochs: int | None) -> None:
    outdir = LANDSCAPE_ROOT / dataset / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"[{gpu}] {dataset} seed={seed}"
    if (outdir / "summary.json").exists() and not force:
        print(f"{tag} SKIP", flush=True)
        return
    code = f"""
import json, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d25_utils import (
    ALPHA_PT_VALUES, ALPHA_GRAD_VALUES, alpha_ce_gradients,
    build_fixed_alpha_pipeline, load_or_make_head_init,
    load_r2d25_b0_setup, train_head_on_frozen_z,
)
from src.analysis.perf_r2d15_utils import val_metrics_with_head

device = torch.device("cuda:0")
dataset, seed = "{dataset}", {seed}
epochs_override = {epochs if epochs is not None else 'None'}
outdir = Path(r"{outdir}")

setup = load_r2d25_b0_setup(dataset, seed, device)
model = setup.model.eval()
for p in model.parameters():
    p.requires_grad_(False)
x = setup.data.x.to(device)
ei = setup.data.edge_index.to(device)
pipeline = build_fixed_alpha_pipeline(setup, x, ei)

head_init_path = outdir / "head_init.pt"
head_init_path.parent.mkdir(parents=True, exist_ok=True)
num_classes = int(setup.data.num_classes)
t0 = time.monotonic()
rows = []
for alpha_pt in ALPHA_PT_VALUES:
    head = load_or_make_head_init(head_init_path, model.out_dim, num_classes, device)
    z = pipeline.z_at(alpha_pt)
    res = train_head_on_frozen_z(
        z, head, setup.data, device,
        epochs=(300 if epochs_override is None else epochs_override),
    )
    del z
    row = {{
        "dataset": dataset, "seed": seed, "alpha_pt": alpha_pt,
        **{{k: v for k, v in res.items() if k != "stop_epoch"}},
    }}
    # diagnostic-only CE gradients w.r.t. alpha_Pt (trained head fixed)
    if alpha_pt in ALPHA_GRAD_VALUES:
        row.update(alpha_ce_gradients(pipeline, head, setup.data, alpha_pt))
    rows.append(row)
    print(f"[run] {{dataset}} s{{seed}} a={{alpha_pt:g}} "
          f"acc={{res['best_val_acc']:.5f}} f1={{res['best_val_macro_f1']:.5f}} "
          f"ep={{res['best_epoch']}}/{{res['stop_epoch']}}", flush=True)
runtime_sec = time.monotonic() - t0
summary = {{
    "dataset": dataset, "seed": seed, "runtime_sec": round(runtime_sec, 1),
    "rows": rows,
    "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
}}
with (outdir / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)
print(f"[done] {{dataset}} s{{seed}} runtime={{runtime_sec:.0f}}s", flush=True)
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = outdir / "run.log"
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"{tag} FAILED rc={proc.returncode}", flush=True)
        print(log.read_text(encoding="utf-8")[-3000:], flush=True)
        return
    print(f"{tag} OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-D2.5-A alpha_Pt landscape")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    args = parser.parse_args()
    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    seeds = SEEDS if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, s) for d in datasets for s in seeds]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out=outputs/perf_r2d25/landscape", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (d, s) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(_run_one, d, s, gpu, args.force, args.epochs)] = (d, s)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
