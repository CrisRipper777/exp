"""R2-D2.6-D: controlled parent adaptation
(docs/BiAxis_R2_Design_2_6_Strong_Parent_Readout_Integration.md §34-§39).

Top-2 D2.6-B candidates x schedules:
    S0 FROZEN           (== D2.6-B setting, reused from integration)
    S1 READOUT_ADAPT    ep 1-30 frozen; ep 31+ A0 fusion only, lr 1e-4
    S2 GRAPH_READOUT    ep 1-30 frozen; ep 31+ A0 graph transform/readout
                        blocks (P0 factorizer frozen), lr 1e-4
M/T/G x seeds 42/43/44 (first round is NOT a seed-42 screen). Exact same
A0 checkpoint / side init / classifier init across schedules. Parent runs
in eval mode throughout (no parent dropout — documented).

Outputs: outputs/perf_r2d26/parent_adapt/<ds>/<variant>/<schedule>/seed_<s>/
    {summary.json, history.csv, best.pt, run.log}

Usage:
    python scripts/perf_r2d26_parent_adapt.py --variants FHC_HOP,RSF_HOP --gpus 0,1
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

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d26_utils import (  # noqa: E402
    R2D26_ROOT,
    TARGET_DATASETS,
    load_a0_parent,
    load_or_make_head_init,
    parent_drift_metrics,
    train_parent_adapt,
)

PARENT_ADAPT_ROOT = R2D26_ROOT / "parent_adapt"
HEAD_INIT_ROOT = R2D26_ROOT / "head_init"
SCHEDULES = ("S1", "S2")


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


def run_worker(dataset: str, variant: str, schedule: str, seed: int,
               outdir: Path, epochs: int | None, force: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} {variant} {schedule} s{seed}] SKIP", flush=True)
        return
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    setup = load_a0_parent(dataset, seed, device)
    from perf_r2d26_integration import resolve_cfg

    cfg = resolve_cfg(dataset, seed, variant)
    info = {
        "input_dim": setup.data.input_dim, "num_nodes": setup.data.num_nodes,
        "num_classes": setup.data.num_classes,
        "text_dim": int(setup.data.x_t.shape[1]), "visual_dim": int(setup.data.x_i.shape[1]),
    }
    from src.models.biaxis_r2_strong_parent import Model

    model = Model(cfg, info, setup.parent).to(device)
    head = load_or_make_head_init(
        HEAD_INIT_ROOT / f"{dataset}_seed{seed}_d{model.out_dim}.pt",
        model.out_dim, int(setup.data.num_classes), device)
    data = setup.data
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    total_epochs = 300 if epochs is None else int(epochs)
    t0 = time.monotonic()
    history_path = outdir / "history.csv"
    history_file = history_path.open("w", encoding="utf-8", newline="")
    history_writer = csv.DictWriter(
        history_file, fieldnames=["epoch", "train_ce", "val_acc", "parent_unfrozen"])
    history_writer.writeheader()
    res = train_parent_adapt(
        data, model, head, device, schedule=schedule, total_epochs=total_epochs,
        deep_sup_lambda=0.1, history_callback=history_writer.writerow,
    )
    history_file.close()
    drift = parent_drift_metrics(setup, model, x, ei)
    torch.save({"head_state": head.state_dict(), "model_state": model.state_dict(),
                "parent_state": model.parent.state_dict()},
               outdir / "best.pt")
    summary = {
        "dataset": dataset, "variant": variant, "schedule": schedule, "seed": seed,
        "best_val_acc": res["best_val_acc"],
        "best_val_macro_f1": res["best_val_macro_f1"],
        "per_class_f1": res["per_class_f1"],
        "best_epoch": res["best_epoch"], "stop_epoch": res["stop_epoch"],
        "parent_unfrozen": res["parent_unfrozen"],
        "parent_drift": drift,
        "side_params": int(model.side_parameter_count),
        "runtime_sec": round(time.monotonic() - t0, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] {dataset} {variant} {schedule} s{seed} "
          f"best_acc={res['best_val_acc']:.5f} f1={res['best_val_macro_f1']:.5f} "
          f"ep={res['best_epoch']}/{res['stop_epoch']} ({summary['runtime_sec']:.0f}s)",
          flush=True)


def _run_one(dataset, variant, schedule, seed, gpu, force, epochs):
    outdir = PARENT_ADAPT_ROOT / dataset / variant / schedule / f"seed_{seed}"
    tag = f"[{gpu}] {dataset} {variant} {schedule} seed={seed}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--dataset", dataset, "--variant", variant, "--schedule", schedule,
        "--seed", str(seed), "--outdir", str(outdir),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(int(epochs))]
    if force:
        cmd += ["--force"]
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "run.log"
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"{tag} FAILED rc={proc.returncode}", flush=True)
        print(log.read_text(encoding="utf-8")[-3000:], flush=True)
        return
    print(f"{tag} OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-D2.6-D controlled parent adaptation")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None, help="top-2 HOP variants")
    parser.add_argument("--schedules", default=None, help="S1,S2")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--schedule", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.dataset, args.variant, args.schedule, args.seed,
                   Path(args.outdir), args.epochs, args.force)
        return

    datasets = TARGET_DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    variants = ["FHC_HOP", "RSF_HOP"] if not args.variants \
        else [v for v in args.variants.split(",")]
    schedules = list(SCHEDULES) if not args.schedules \
        else [s for s in args.schedules.split(",")]
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, v, sched, s) for d in datasets for v in variants
            for sched in schedules for s in seeds]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out=outputs/perf_r2d26/parent_adapt",
          flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (d, v, sched, s) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(_run_one, d, v, sched, s, gpu, args.force,
                                    args.epochs)] = (d, v, sched, s)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
