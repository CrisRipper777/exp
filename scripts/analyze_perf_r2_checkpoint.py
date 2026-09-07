"""R2-Design-1 best-checkpoint mechanism diagnostics (plan §18/§34).

For one (dataset, seed, variant) checkpoint, runs the model-internal
Model.compute_r2_diagnostics and writes r2_diagnostics.json:

    semantic  : common gate weight stats + per-factor semantic residual ratio
    functional: 3x3 gate matrix stats / 3x3 contribution matrix /
                rho_base / rho_func / base & functional residual ratios
    p0        : P0 ownership health (common_sim / private_sim / C-P overlap /
                factor norms) — verifies refinement did not break the
                factorizer (plan §18.1)

Prints the analysis peak allocation. NEVER reads test.

Usage:
    python scripts/analyze_perf_r2_checkpoint.py --dataset Movies --seed 42 \
        --variant B0 --ckpt outputs/perf_r2d1/b0/Movies/B0/seed_42/model.pt \
        --out outputs/perf_r2d1/b0/Movies/B0/seed_42 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2_utils import VARIANTS, assert_no_test_access, load_r2_setup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="R2 checkpoint mechanism diagnostics")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ckpt-root", default=None,
                        help="checkpoint tree root (default outputs/perf_r2d1/<variant_root>)")
    args = parser.parse_args()

    device = torch.device(args.device)
    setup = load_r2_setup(args.dataset, args.seed, args.variant, device, root=args.ckpt_root)
    assert_no_test_access(setup.data)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    x = setup.data.x.to(device)
    edge_index = setup.data.edge_index.to(device)
    diag = setup.model.compute_r2_diagnostics(x, edge_index)
    with (outdir / "r2_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    if diag.get("semantic"):
        w_t = diag["semantic"]["w_t"]
        print(
            f"[diag] {args.dataset} s{args.seed} {args.variant} "
            f"w_t mean={w_t['mean']:.4f} std={w_t['std']:.4f} "
            f"frac<.05={w_t['frac_lt_05']:.3f} frac>.95={w_t['frac_gt_95']:.3f}",
            flush=True,
        )
    if diag.get("functional"):
        gm = diag["functional"]["gate_matrix"]
        print(
            f"[diag] {args.dataset} s{args.seed} {args.variant} "
            f"gate mean=\n"
            + "\n".join("    " + " ".join(f"{v:6.4f}" for v in row) for row in gm["mean"]),
            flush=True,
        )
        print(
            f"[diag] {args.dataset} s{args.seed} {args.variant} "
            f"contrib=\n"
            + "\n".join(
                "    " + " ".join(f"{v:6.4f}" for v in row)
                for row in diag["functional"]["contribution_matrix"]["values"]
            ),
            flush=True,
        )
        print(
            f"[diag] rho_base={['%.4f' % v for v in diag['rho_base']]} "
            f"rho_func={['%.4f' % v for v in diag['functional']['rho_func']]}",
            flush=True,
        )
    p0 = diag["p0"]
    print(
        f"[p0] common_sim={p0['p0_common_sim']:.4f} private_sim={p0['p0_private_sim']:.4f} "
        f"cp_overlap=({p0['p0_cp_overlap_t']:.4f},{p0['p0_cp_overlap_v']:.4f}) "
        f"norms=(c {p0['p0_c_norm']:.2f}, pt {p0['p0_pt_norm']:.2f}, pv {p0['p0_pv_norm']:.2f})",
        flush=True,
    )

    if x.is_cuda:
        print(f"[mem] peak_allocated_mb={torch.cuda.max_memory_allocated(device) / 1e6:.1f}", flush=True)
    print(f"[r2-diag] saved -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
