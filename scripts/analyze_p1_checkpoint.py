"""P1 best-checkpoint mechanism diagnostics (plan §38).

Loads a saved biaxis_p1 checkpoint (from ``task.save_ckpt_path``), reloads the
dataset, and runs ``Model.compute_p1_diagnostics`` on the full graph. Outputs
``diagnostics.json`` + ``usage_matrix.csv`` into the run directory.

Diagnostics NEVER use test labels and do not modify model state. The model
config overrides must match the ones used at training time (variant switches).

Usage:
    python scripts/analyze_p1_checkpoint.py \
        --dataset Movies --task nc --seed 42 \
        --ckpt outputs/p1/screen/Movies/F1R1/model.pt \
        --out outputs/p1/screen/Movies/F1R1 \
        --device cuda:1 \
        --model-overrides model.p1.factor_aware=true,model.p1.num_relations=4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_cfg(dataset: str, task: str, seed: int, model_overrides: list[str] | None):
    # seed is included for strict config reproducibility (review §22); P1
    # diagnostics never use labels/splits so results are unaffected, but the
    # composed config must match the training run exactly.
    overrides = [f"dataset={dataset}", f"task={task}", f"seed={int(seed)}", "model=biaxis_p1"] + list(
        model_overrides or []
    )
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="config", overrides=overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 checkpoint mechanism diagnostics")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", default="nc", choices=["nc", "lp"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt", required=True, help="path to model.pt saved by the task runner")
    parser.add_argument("--out", required=True, help="output directory (run dir)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-overrides",
        default="",
        help="comma-separated hydra overrides matching the training run",
    )
    args = parser.parse_args()

    model_overrides = [item.strip() for item in args.model_overrides.split(",") if item.strip()] or None
    cfg = _resolve_cfg(args.dataset, args.task, args.seed, model_overrides)

    from src.data import load_mag_data
    from src.models.biaxis_p1 import Model

    data = load_mag_data(cfg, args.task, args.seed)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Model(cfg, ckpt["data_info"])
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device)
    model = model.to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    diag = model.compute_p1_diagnostics(x, edge_index)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    um = diag["usage_matrix"]
    with (outdir / "usage_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["factor"] + um["relations"])
        for name, row in zip(um["factors"], um["values"]):
            writer.writerow([name] + [f"{value:.6f}" for value in row])

    budget = diag["budget"]
    print(
        f"[diag] {args.dataset} seed={args.seed} "
        f"K_eff={diag['relation']['effective_num']:.3f} "
        f"edge_ent={diag['relation']['mean_edge_entropy']:.3f}",
        flush=True,
    )
    for name, stats in budget.items():
        print(
            f"[diag] beta_{name}: mean={stats['mean']:.3f} "
            f"p10={stats['p10']:.3f} p90={stats['p90']:.3f} "
            f"low={stats['low_frac']:.3f} high={stats['high_frac']:.3f}",
            flush=True,
        )
    for key, value in diag["alpha_entropy"].items():
        print(f"[diag] alpha_ent_{key}={value:.3f}", flush=True)
    for key, value in diag["alpha_js"].items():
        print(f"[diag] alpha_js_{key}={value:.4f}", flush=True)
    if x.is_cuda:
        peak = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"[mem] peak_allocated_mb={peak:.1f}", flush=True)
    print(f"[diag] saved -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
