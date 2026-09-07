"""R2-D2.6-C: causal evidence-usage audit
(docs/BiAxis_R2_Design_2_6_Strong_Parent_Readout_Integration.md §30-§33).

No retraining. Loads the D2.6-B best checkpoint (side branch + head),
rebuilds the frozen A0 parent, and evaluates at the best checkpoint:

    FULL / H2_ZERO / H2_TO_H1 / H2_SHUFFLE(seed=20260904)
    PT_H2_OFF / C_H2_OFF / PV_H2_OFF
    side-off (base preservation), factor-summary ablations, hop ablations

Plus: base-preservation geometry (CKA / cosine / rel L2 / side-base norm
ratio), attention matrices, gradient sensitivity to factor summaries.

Usage:
    python scripts/perf_r2d26_causal.py --gpus 0,1 --variants FHC_HOP,RSF_HOP
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

from src.analysis.perf_r2d26_utils import (  # noqa: E402
    DATASETS,
    R2D26_ROOT,
    VARIANTS,
    causal_metrics,
    load_a0_parent,
)

CAUSAL_ROOT = R2D26_ROOT / "causal_usage"
INTEGRATION_ROOT = R2D26_ROOT / "integration"

CAUSAL_KEYS = ("full", "h2_zero", "h2_to_h1", "h2_shuffle",
               "pt_h2_off", "c_h2_off", "pv_h2_off",
               "s_c_off", "s_pt_off", "s_pv_off", "side_off")


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


def run_worker(dataset: str, variant: str, seed: int, outdir: Path,
               force: bool, ckpt_root: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} {variant} s{seed}] SKIP", flush=True)
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
    ckpt_path = ckpt_root / dataset / variant / f"seed_{seed}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from src.models.biaxis_r2_strong_parent import Model

    model = Model(cfg, info, setup.parent).to(device)
    model.load_state_dict(ckpt["model_state"])
    head = torch.nn.Linear(model.out_dim, int(setup.data.num_classes)).to(device)
    head.load_state_dict(ckpt["head_state"])
    model.eval()
    head.eval()

    x = setup.data.x.to(device)
    ei = setup.data.edge_index.to(device)
    # side_off returns the h-dim z_base: only residual readouts (out_dim==h)
    # can evaluate it through the candidate head.
    keys = [k for k in CAUSAL_KEYS
            if k != "side_off" or model.out_dim == model.hidden_dim]
    causal = causal_metrics(model, head, x, ei, setup.data, device,
                            causal_keys=tuple(keys))
    with torch.no_grad():
        diag = model.compute_diagnostics(x, ei)
        sens = model.gradient_sensitivity(x, ei)
    summary = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "readout_type": model.readout_type, "token_source": model.token_source,
        "causal": causal,
        "diagnostics": diag,
        "gradient_sensitivity": sens,
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] {dataset} {variant} s{seed} causal done", flush=True)


def _run_one(dataset, variant, seed, gpu, force, ckpt_root):
    outdir = CAUSAL_ROOT / dataset / variant / f"seed_{seed}"
    tag = f"[{gpu}] {dataset} {variant} seed={seed}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--dataset", dataset, "--variant", variant, "--seed", str(seed),
        "--outdir", str(outdir), "--ckpt-root", str(ckpt_root),
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
    parser = argparse.ArgumentParser(description="R2-D2.6-C causal usage audit")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None, help="comma-separated HOP variants")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ckpt-root", default=str(INTEGRATION_ROOT))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.dataset, args.variant, args.seed, Path(args.outdir),
                   args.force, Path(args.ckpt_root))
        return

    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    variants = ["FHC_HOP", "RSF_HOP", "HIER_HOP"] if not args.variants \
        else [v for v in args.variants.split(",")]
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, v, s) for d in datasets for v in variants for s in seeds]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out=outputs/perf_r2d26/causal_usage",
          flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (d, v, s) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(_run_one, d, v, s, gpu, args.force,
                                    Path(args.ckpt_root))] = (d, v, s)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
