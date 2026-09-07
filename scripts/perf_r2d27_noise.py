"""R2-D2.7-F: random-edge stress test (optional, plan §46/§63).

Only for a top PRE candidate achieving A0 incremental GO or within
+0.10pp of it (entered: PAIR_EDGE A0 incremental PASS at +0.496pp).

Evaluation ONLY (no retraining): inject random edges equal to 10%/25%
of the original edge count (fixed seed). The A0 parent path always
consumes the ORIGINAL graph; only the utility side branch sees noise.

Compare A0 / UNIFORM / PAIR_EDGE degradation and report injected-edge
mean utility + top-10/25% occupancy.

Usage:
    python scripts/perf_r2d27_noise.py --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d27_utils import (  # noqa: E402
    DATASETS,
    R2D27_ROOT,
    SEEDS,
    load_a0_parent,
)

NOISE_ROOT = R2D27_ROOT / "noise_optional"
MATRIX_ROOT = R2D27_ROOT / "matrix"
TARGET_DATASETS = ["Movies", "Toys", "Grocery"]
GUARD_DATASETS = ["ele-fashion", "Reddit-S"]
NOISE_PCTS = (10, 25)


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


def run_worker(dataset: str, seed: int, outdir: Path, force: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} s{seed}] SKIP", flush=True)
        return
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    setup = load_a0_parent(dataset, seed, device)
    from scripts.perf_r2d27_matrix import resolve_cfg
    from src.models.biaxis_r2_neighbor_utility import Model
    from src.analysis.perf_r2d15_utils import val_metrics_with_head

    x = setup.data.x.to(device)
    ei = setup.data.edge_index.to(device)
    num_nodes = int(x.size(0))
    num_classes = int(setup.data.num_classes)

    results = {}
    # A0 clean reference
    with torch.no_grad():
        factors, _ = setup.parent._encode(x)
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        graph_out = setup.parent._graph_update(f_block, ei, num_nodes)
        f_tilde = graph_out["f_tilde"]
        z_a0 = setup.parent.fusion(
            torch.cat([f_tilde[:, 0], f_tilde[:, 1], f_tilde[:, 2]], dim=-1))
    from src.analysis.perf_r2d27_utils import load_or_make_head_init

    head_init = R2D27_ROOT / "head_init"
    head_a0 = load_or_make_head_init(
        head_init / f"{dataset}_seed{seed}_d{setup.parent.hidden_dim}.pt",
        setup.parent.hidden_dim, num_classes, device)
    from scripts.perf_r2d26_integration import _train_a0_base_head

    _train_a0_base_head(z_a0, head_a0, setup.data, device, 300, 30)
    a0_clean = val_metrics_with_head(head_a0, z_a0, setup.data, device)

    for variant in ("UNIFORM", "PAIR_EDGE"):
        cfg = resolve_cfg(dataset, seed, variant)
        info = {"input_dim": setup.data.input_dim, "num_nodes": num_nodes,
                "num_classes": num_classes,
                "text_dim": int(setup.data.x_t.shape[1]),
                "visual_dim": int(setup.data.x_i.shape[1])}
        ckpt = torch.load(MATRIX_ROOT / dataset / variant / f"seed_{seed}" / "best.pt",
                          map_location="cpu", weights_only=False)
        model = Model(cfg, info, setup.parent).to(device)
        model.load_state_dict(ckpt["model_state"])
        head = torch.nn.Linear(model.out_dim, num_classes).to(device)
        head.load_state_dict(ckpt["head_state"])
        model.eval()
        head.eval()
        for pct in NOISE_PCTS:
            with torch.no_grad():
                z_clean, _, _, _, _ = model(x, ei, causal="full")
                z_noisy, _, _, _, _ = model(x, ei, causal=f"noise_{pct}")
            m_clean = val_metrics_with_head(head, z_clean, setup.data, device)
            m_noisy = val_metrics_with_head(head, z_noisy, setup.data, device)
            results[f"{variant}_noise{pct}"] = {
                "clean_acc": m_clean["val_acc"], "noisy_acc": m_noisy["val_acc"],
                "clean_f1": m_clean["val_macro_f1"], "noisy_f1": m_noisy["val_macro_f1"],
                "acc_drop_pp": 100.0 * (m_clean["val_acc"] - m_noisy["val_acc"]),
                "f1_drop_pp": 100.0 * (m_clean["val_macro_f1"] - m_noisy["val_macro_f1"]),
            }
            if variant == "PAIR_EDGE":
                util = model.injected_edge_utility(x, ei, float(pct))
                results[f"PAIR_EDGE_noise{pct}"].update(util)
    summary = {
        "dataset": dataset, "seed": seed,
        "a0_clean_acc": a0_clean["val_acc"], "a0_clean_f1": a0_clean["val_macro_f1"],
        "results": results,
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] {dataset} s{seed} noise done", flush=True)


def _run_one(dataset, seed, gpu, force):
    outdir = NOISE_ROOT / dataset / f"seed_{seed}"
    tag = f"[{gpu}] {dataset} seed={seed}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--dataset", dataset, "--seed", str(seed),
        "--outdir", str(outdir),
    ]
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
    parser = argparse.ArgumentParser(description="R2-D2.7-F random-edge stress test")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.dataset, args.seed, Path(args.outdir), args.force)
        return

    datasets = TARGET_DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    seeds = SEEDS if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, s) for d in datasets for s in seeds]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out=outputs/perf_r2d27/noise_optional",
          flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (d, s) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(_run_one, d, s, gpu, args.force)] = (d, s)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
