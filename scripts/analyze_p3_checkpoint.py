"""P3 best-checkpoint mechanism diagnostics (plan §35 Prompt 4).

Loads a biaxis_p3 checkpoint, reruns the dataset, computes
``Model.compute_p3_diagnostics`` on the full graph (P2 plan diagnostics +
operator diagnostics). Outputs:

    diagnostics.json
    transport_plan_summary.csv   (per-factor null/graph mass + plan entropy)
    conditional_usage_matrix.csv (factor x relation, graph-normalized)
    operator_residuals.csv       (per-cell residual norms + usage)

Diagnostics NEVER use labels; the config overrides must match training.
The full Gamma plan is NOT saved (debug only).

Usage:
    python scripts/analyze_p3_checkpoint.py --dataset Movies --seed 42 \
        --ckpt outputs/p3/operator/Movies/O0/seed_42/model.pt \
        --out outputs/p3/operator/Movies/O0/seed_42 --device cuda:0 \
        --model-overrides model.p3.operator_mode=shared
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
    overrides = [f"dataset={dataset}", f"task={task}", f"seed={int(seed)}", "model=biaxis_p3"] + list(
        model_overrides or []
    )
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="config", overrides=overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 checkpoint mechanism diagnostics")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", default="nc")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-overrides", default="")
    args = parser.parse_args()

    model_overrides = [item.strip() for item in args.model_overrides.split(",") if item.strip()] or None
    cfg = _resolve_cfg(args.dataset, args.task, args.seed, model_overrides)

    from src.data import load_mag_data
    from src.models.biaxis_p3 import Model

    data = load_mag_data(cfg, args.task, args.seed)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Model(cfg, ckpt["data_info"])
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device)
    model = model.to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    diag = model.compute_p3_diagnostics(x, edge_index)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    with (outdir / "transport_plan_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["factor", "null_mean", "null_p10", "null_p50", "null_p90", "null_high_frac",
             "graph_mass_mean", "plan_entropy", "alpha_entropy"]
        )
        for name, stats in diag["plan"].items():
            writer.writerow([
                name,
                f"{stats['null_mean']:.6f}", f"{stats['null_p10']:.6f}",
                f"{stats['null_p50']:.6f}", f"{stats['null_p90']:.6f}",
                f"{stats['null_high_frac']:.6f}", f"{stats['graph_mass_mean']:.6f}",
                f"{diag['plan_entropy'][name]:.6f}", f"{diag['alpha_entropy'][name]:.6f}",
            ])

    um = diag["usage_matrix"]
    with (outdir / "conditional_usage_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["factor"] + um["relations"])
        for name, row in zip(um["factors"], um["values"]):
            writer.writerow([name] + [f"{value:.6f}" for value in row])

    op = diag.get("operator", {})
    factors = um["factors"]
    relations = um["relations"]
    with (outdir / "operator_residuals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "residual_norm_rel", "usage"])
        if op.get("residual_norms", {}).get("pair"):
            for fi, row in enumerate(op["residual_norms"]["pair"]):
                for ki, value in enumerate(row):
                    writer.writerow([
                        f"{factors[fi]}-{relations[ki]}",
                        f"{value:.6f}",
                        f"{op['usage'][fi][ki]:.6f}",
                    ])
        else:
            for fi, factor in enumerate(factors):
                for ki, rel in enumerate(relations):
                    writer.writerow([f"{factor}-{rel}", "", f"{op['usage'][fi][ki]:.6f}"])

    print(
        f"[diag] {args.dataset} seed={args.seed} mode={op.get('mode')} "
        f"K_eff={diag['relation']['effective_num']:.3f} S_R={diag['relation']['specialization']:.3f}",
        flush=True,
    )
    for name, stats in diag["plan"].items():
        print(
            f"[diag] {name}: null={stats['null_mean']:.3f} graph={stats['graph_mass_mean']:.3f} "
            f"plan_ent={diag['plan_entropy'][name]:.3f}",
            flush=True,
        )
    if op:
        print(
            f"[diag] operator: mode={op['mode']} w0_norm={op['w0_norm']:.3f} "
            f"pair_strength={op['pair_strength']:.5f} "
            f"msg_dev={op['message_deviation_usage_weighted']:.5f} "
            f"extra_params={op['extra_residual_params']}",
            flush=True,
        )
    if x.is_cuda:
        print(f"[mem] peak_allocated_mb={torch.cuda.max_memory_allocated(device) / 1e6:.1f}", flush=True)
    print(f"[diag] saved -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
