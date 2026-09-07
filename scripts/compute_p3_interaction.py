"""Post-hoc double-centered interaction diagnostic (review §12).

Reconstructs the EFFECTIVE cell operators T_fk from existing best
checkpoints (CPU only — no data loading, no re-training) and computes the
double-centered interaction:

    I_fk = T_fk - Tbar_f· - Tbar_·k + Tbar_··

reported as ||I_fk||_F / ||W0||_F per cell plus the usage-weighted strength
sum_fk u_fk ||I_fk||_F / ||W0||_F, where u_fk = mean_i Gamma_ifk is read from
the EXISTING diagnostics.json (usage_matrix.values, saved by the analyzer).

This is defined on the final effective operators, so it is invariant to how
A/B/C absorb each other — strictly more rigorous than ||C_fk|| (review §12).

Supported checkpoints: full residual (A/B/C), low-rank (U/V/a/b), basis
(V/c). OADD is a built-in sanity: its T_fk is exactly additive, so I_fk ≈ 0
up to float-mean ulp noise.

Outputs per run: interaction.json + rows appended to interaction_summary.csv
under the stage root given by --root.

Usage:
    python scripts/compute_p3_interaction.py --root outputs/p3/operator --modes OFR,OADD
    python scripts/compute_p3_interaction.py --root outputs/p3/lowrank
    python scripts/compute_p3_interaction.py --root outputs/p3/basis
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-8


def _effective_cells(state: dict, mode_label: str, num_factors: int = 3, num_relations: int = 4) -> torch.Tensor:
    """Effective T_fk [F, K, d, d] from a checkpoint state dict.

    mode_label is the directory label (O0/OF/OR/OADD/OFR/LR-ADD/LR-INT/
    Basis4/...); the state dict itself determines which residuals exist.
    """
    w0 = state["graph_w0.weight"]  # [d, d]
    dim = int(w0.size(0))
    t = w0.clone().unsqueeze(0).unsqueeze(0).expand(num_factors, num_relations, dim, dim).clone()

    if "operator.A" in state:
        t = t + state["operator.A"].unsqueeze(1)  # [F, 1, d, d]
    if "operator.B" in state:
        t = t + state["operator.B"].unsqueeze(0)  # [1, K, d, d]
    if "operator.C" in state:
        t = t + state["operator.C"]

    if "operator.U" in state:  # low-rank: U diag(c_fk) V^T
        u, v = state["operator.U"], state["operator.V"]  # [d, r], [d, r]
        a, b = state["operator.a"], state["operator.b"]  # [F, r], [K, r]
        c = a.unsqueeze(1) + b.unsqueeze(0)
        if mode_label.startswith("LR-INT"):
            c = c + a.unsqueeze(1) * b.unsqueeze(0)
        t = t + torch.einsum("dr,fkr,er->fkde", u, c, v)

    if "operator.V" in state and "operator.c" in state:  # basis
        t = t + torch.einsum("bde,fkb->fkde", state["operator.V"], state["operator.c"])

    return t


def _interaction(t_cells: torch.Tensor, w0_norm: float, usage: torch.Tensor) -> dict:
    tbar_f = t_cells.mean(dim=1, keepdim=True)
    tbar_k = t_cells.mean(dim=0, keepdim=True)
    tbar_all = t_cells.mean(dim=(0, 1), keepdim=True)
    i_cells = t_cells - tbar_f - tbar_k + tbar_all
    i_norms = i_cells.norm(p="fro", dim=(2, 3)) / (w0_norm + EPS)  # [F, K]
    raw = (usage * i_norms).sum()
    total = usage.sum()
    return {
        "norms": [[float(v) for v in row] for row in i_norms.tolist()],
        # review-2 §2: raw = total interaction exposure; normalized =
        # usage-weighted AVERAGE (comparable across datasets).
        "usage_weighted_strength": float(raw.item()),
        "usage_weighted_average": float((raw / (total + EPS)).item()),
    }


def _run_dir_info(run_dir: Path) -> dict | None:
    """Extract mode label + summary + diagnostics usage for one run dir."""
    summary_path = run_dir / "summary.json"
    model_path = run_dir / "model.pt"
    diag_path = run_dir / "diagnostics.json"
    if not summary_path.exists() or not model_path.exists():
        return None
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    usage = None
    if diag_path.exists():
        with diag_path.open(encoding="utf-8") as f:
            diag = json.load(f)
        um = diag.get("usage_matrix") or {}
        values = um.get("values")
        if values:
            usage = torch.tensor([[float(v) for v in row] for row in values])
    return {
        "dataset": summary["dataset"],
        "mode": summary["mode"],
        "seed": summary["seed"],
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc double-centered interaction diagnostic")
    parser.add_argument("--root", required=True, help="stage root, e.g. outputs/p3/operator")
    parser.add_argument("--modes", default=None, help="comma-separated mode labels to restrict (default: all)")
    args = parser.parse_args()

    root = Path(args.root)
    modes = None
    if args.modes:
        modes = {item.strip() for item in args.modes.split(",") if item.strip()}

    rows: list[dict] = []
    for run_dir in sorted(root.glob("*/**/seed_*")):
        info = _run_dir_info(run_dir)
        if info is None:
            continue
        if modes is not None and info["mode"] not in modes:
            continue
        if info["usage"] is None:
            print(f"[skip] {run_dir}: no usage_matrix in diagnostics.json", flush=True)
            continue
        ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
        state = ckpt["model_state"]
        t_cells = _effective_cells(state, info["mode"])
        w0_norm = float(state["graph_w0.weight"].norm(p="fro").item())
        result = _interaction(t_cells, w0_norm, info["usage"])
        with (run_dir / "interaction.json").open("w", encoding="utf-8") as f:
            json.dump({
                "dataset": info["dataset"],
                "mode": info["mode"],
                "seed": info["seed"],
                **result,
            }, f, indent=2)
        rows.append({
            "dataset": info["dataset"],
            "mode": info["mode"],
            "seed": info["seed"],
            "interaction_strength": f"{result['usage_weighted_strength']:.6f}",
            "interaction_average": f"{result['usage_weighted_average']:.6f}",
        })
        print(
            f"[interaction] {info['dataset']} {info['mode']} seed={info['seed']} "
            f"strength={result['usage_weighted_strength']:.6f} "
            f"average={result['usage_weighted_average']:.6f}",
            flush=True,
        )

    if rows:
        csv_path = root / "interaction_summary.csv"
        fieldnames = ["dataset", "mode", "seed", "interaction_strength", "interaction_average"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[interaction] {len(rows)} runs -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
