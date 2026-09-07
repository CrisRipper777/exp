"""R2-D2.7-A: pre-aggregation neighbor-utility matrix
(docs/BiAxis_R2_Design_2_7_PreAggregation_Neighbor_Utility_Audit.md §18).

Variants: A0_BASE / UNIFORM / TARGET_NULL_ONLY / GENERIC_EDGE /
DIAG_EDGE / PAIR_EDGE / SEMANTIC_SIM (later stages reuse this driver
with POST_PAIR / SOURCE_FACTOR_ONLY / TARGET_FACTOR_ONLY /
PAIR_TRANSFORM_* and other out-roots).

A0 fully frozen; side/scorer/payload/classifier lr 1e-3 wd 1e-4,
warmup10+cosine, 300 ep / patience 30 / best Val Acc / grad clip 1.0.
No aux loss, no edge supervision. Val only — NEVER test.

Outputs: outputs/perf_r2d27/matrix/<ds>/<variant>/seed_<s>/
    {summary.json, history.csv, best.pt, run.log}

Usage:
    python scripts/perf_r2d27_matrix.py --gpus 0,1
    python scripts/perf_r2d27_matrix.py --datasets Movies --variants PAIR_EDGE,UNIFORM --seeds 42 --epochs 5  # smoke
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

from src.analysis.perf_r2d27_utils import (  # noqa: E402
    DATASETS,
    R2D27_ROOT,
    VARIANTS,
    load_a0_parent,
    load_or_make_head_init,
    train_utility_model,
)

MATRIX_ROOT = R2D27_ROOT / "matrix"
HEAD_INIT_ROOT = R2D27_ROOT / "head_init"


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


def resolve_cfg(dataset: str, seed: int, variant: str):
    from hydra import compose, initialize_config_dir

    mode = VARIANTS[variant]
    overrides = [
        f"dataset={dataset}", "task=nc", "model=biaxis_r2_neighbor_utility",
        f"model.mode={mode}", f"seed={int(seed)}",
    ]
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        return compose(config_name="config", overrides=overrides)


def run_worker(dataset: str, variant: str, seed: int, outdir: Path,
               epochs: int | None, force: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} {variant} s{seed}] SKIP", flush=True)
        return
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    setup = load_a0_parent(dataset, seed, device)
    cfg = resolve_cfg(dataset, seed, variant)
    info = {
        "input_dim": setup.data.input_dim, "num_nodes": setup.data.num_nodes,
        "num_classes": setup.data.num_classes,
        "text_dim": int(setup.data.x_t.shape[1]), "visual_dim": int(setup.data.x_i.shape[1]),
    }
    data = setup.data
    total_epochs = 300 if epochs is None else int(epochs)
    t0 = time.monotonic()

    if variant == "A0_BASE":
        from scripts.perf_r2d26_integration import _train_a0_base_head

        x = data.x.to(device)
        ei = data.edge_index.to(device)
        num_nodes = int(x.size(0))
        with torch.no_grad():
            factors, _ = setup.parent._encode(x)
            f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
            graph_out = setup.parent._graph_update(f_block, ei, num_nodes)
            f_tilde = graph_out["f_tilde"]
            z_base = setup.parent.fusion(
                torch.cat([f_tilde[:, 0], f_tilde[:, 1], f_tilde[:, 2]], dim=-1))
        head = load_or_make_head_init(
            HEAD_INIT_ROOT / f"{dataset}_seed{seed}_d{setup.parent.hidden_dim}.pt",
            setup.parent.hidden_dim, int(data.num_classes), device)
        res = _train_a0_base_head(z_base, head, data, device, total_epochs, 30)
        summary = {
            "dataset": dataset, "variant": variant, "seed": seed, **res,
            "side_params": 0, "out_dim": setup.parent.hidden_dim,
            "runtime_sec": round(time.monotonic() - t0, 1),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
            "ablations": None,
        }
        with (outdir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[run] {dataset} {variant} s{seed} best_acc={res['best_val_acc']:.5f} "
              f"f1={res['best_val_macro_f1']:.5f} ep={res['best_epoch']}/{res['stop_epoch']}",
              flush=True)
        return

    from src.models.biaxis_r2_neighbor_utility import Model

    model = Model(cfg, info, setup.parent).to(device)
    head = load_or_make_head_init(
        HEAD_INIT_ROOT / f"{dataset}_seed{seed}_d{model.out_dim}.pt",
        model.out_dim, int(data.num_classes), device)
    history_path = outdir / "history.csv"
    history_file = history_path.open("w", encoding="utf-8", newline="")
    history_writer = csv.DictWriter(
        history_file, fieldnames=["epoch", "lr", "train_ce", "val_acc"])
    history_writer.writeheader()
    res = train_utility_model(
        data, model, head, device, total_epochs=total_epochs,
        history_callback=history_writer.writerow)
    history_file.close()

    x = data.x.to(device)
    ei = data.edge_index.to(device)
    abl = None
    if variant == "PAIR_EDGE":
        from src.analysis.perf_r2d27_utils import causal_metrics

        abl = causal_metrics(model, head, x, ei, data, device,
                             causal_keys=("full", "within_target_shuffle",
                                          "remove_top_10", "remove_random_10"))
    torch.save({"head_state": head.state_dict(), "model_state": model.state_dict()},
               outdir / "best.pt")
    summary = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "mode": model.mode,
        "best_val_acc": res["best_val_acc"],
        "best_val_macro_f1": res["best_val_macro_f1"],
        "per_class_f1": res["per_class_f1"],
        "best_epoch": res["best_epoch"], "stop_epoch": res["stop_epoch"],
        "side_params": int(model.side_parameter_count),
        "out_dim": int(model.out_dim),
        "ablations": abl,
        "runtime_sec": round(time.monotonic() - t0, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] {dataset} {variant} s{seed} best_acc={res['best_val_acc']:.5f} "
          f"f1={res['best_val_macro_f1']:.5f} ep={res['best_epoch']}/{res['stop_epoch']} "
          f"side_params={model.side_parameter_count} ({summary['runtime_sec']:.0f}s)",
          flush=True)


def _run_one(dataset, variant, seed, gpu, force, epochs, out_root):
    outdir = out_root / dataset / variant / f"seed_{seed}"
    tag = f"[{gpu}] {dataset} {variant} seed={seed}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--dataset", dataset, "--variant", variant, "--seed", str(seed),
        "--outdir", str(outdir), "--out-root", str(out_root),
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="R2-D2.7 neighbor-utility matrix")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args(argv)

    out_root = Path(args.out_root) if args.out_root else MATRIX_ROOT
    if args.worker:
        run_worker(args.dataset, args.variant, args.seed, Path(args.outdir),
                   args.epochs, args.force)
        return

    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    variants = list(VARIANTS) if not args.variants else [v for v in args.variants.split(",")]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {unknown}")
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, v, s) for d in datasets for v in variants for s in seeds]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out={out_root}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (d, v, s) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(_run_one, d, v, s, gpu, args.force, args.epochs,
                                    out_root)] = (d, v, s)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
