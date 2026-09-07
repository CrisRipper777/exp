"""R0-D6: multi-hop potential probe (plan §22-§25/§36).

No model modification: row-normalized message-direction diffusion
Z_{l+1} = D^{-1} A Z_l (chunked scatter aggregation, same direction as the
model's relation contexts) applied to the FROZEN embeddings:

    A. Z = z_local   B. Z = z_final

Fixed Ridge probes on Z0, [Z0|Z1], [Z0|Z1|Z2], [Z0|Z1|Z2|Z3]
(fit TRAIN / eval VAL). Plus hop-wise cosine convergence / variance
(oversmoothing statistics).

Delta_hop2 = Probe([Z0|Z1|Z2]) - Probe(Z0)
Delta_hop3 = Probe([Z0|Z1|Z2|Z3]) - Probe(Z0)

NOTE: this measures frozen-representation multi-hop INFORMATION POTENTIAL,
not multi-hop model performance. No test access.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r0_utils import (  # noqa: E402
    DATASETS,
    SEEDS,
    chunked_mean_cos,
    extract_forward,
    load_setup,
    ridge_probe,
    write_csv,
)

OUT_ROOT = PROJECT_ROOT / "outputs" / "perf_r0" / "hop"
HOP_VARIANTS = ["Z0", "Z0|Z1", "Z0|Z1|Z2", "Z0|Z1|Z2|Z3"]


def _diffuse(z: torch.Tensor, edge_index: torch.Tensor, deg: torch.Tensor,
             steps: int = 3, chunk: int = 500_000) -> list[torch.Tensor]:
    """Row-normalized message-direction diffusion (chunked scatter_add)."""
    src, dst = edge_index[0], edge_index[1]
    n = int(z.size(0))
    trajectory = [z]
    current = z
    for _ in range(steps):
        acc = torch.zeros_like(current)
        for start in range(0, src.size(0), chunk):
            acc.index_add_(0, dst[start : start + chunk], current[src[start : start + chunk]])
        current = acc / deg.clamp_min(1.0).unsqueeze(-1)
        trajectory.append(current)
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="R0-D6 multi-hop potential probe")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    datasets = [d for d in (args.datasets or ",".join(DATASETS)).split(",") if d in DATASETS]
    seeds = [int(s) for s in (args.seeds or ",".join(map(str, SEEDS))).split(",") if s]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    probe_rows: list[dict] = []
    smooth_rows: list[dict] = []

    for di, dataset in enumerate(datasets):
        device = torch.device(f"cuda:{gpus[di % len(gpus)]}")
        for seed in seeds:
            setup = load_setup(dataset, seed, device)
            fex = extract_forward(setup)
            deg = fex["deg"]
            edge_index = fex["edge_index"]
            for traj_name, z0 in (("z_local", fex["z_local"]), ("z_final", fex["z_final"])):
                traj = _diffuse(z0, edge_index, deg, steps=3)
                # oversmoothing statistics
                smooth_row = {"dataset": dataset, "seed": seed, "trajectory": traj_name}
                for l in range(3):
                    smooth_row[f"cos_Z{l}_Z{l+1}"] = chunked_mean_cos(traj[l], traj[l + 1])
                    smooth_row[f"var_Z{l}"] = float(traj[l].var(dim=-1).mean().item())
                smooth_rows.append(smooth_row)
                for vname in HOP_VARIANTS:
                    parts = vname.split("|")
                    feat = torch.cat([traj[int(p[1])] for p in parts], dim=-1)
                    probe = ridge_probe(feat, setup)
                    probe_rows.append({
                        "dataset": dataset, "seed": seed, "trajectory": traj_name,
                        "variant": vname, "val_acc": probe["val_acc"], "val_macro_f1": probe["val_macro_f1"],
                    })
                    print(
                        f"[hop] {dataset:12s} s{seed} {traj_name:7s} {vname:10s} val={probe['val_acc']:.4f}",
                        flush=True,
                    )
                del traj
                torch.cuda.empty_cache()
            del fex
            torch.cuda.empty_cache()

    write_csv(OUT_ROOT / "hop_probe_per_seed.csv", probe_rows)
    write_csv(OUT_ROOT / "hop_smoothing_stats.csv", smooth_rows)

    summary_rows: list[dict] = []
    for dataset in datasets:
        for traj_name in ("z_local", "z_final"):
            for vname in HOP_VARIANTS:
                vals = [r["val_acc"] for r in probe_rows
                        if r["dataset"] == dataset and r["trajectory"] == traj_name and r["variant"] == vname]
                if vals:
                    summary_rows.append({
                        "dataset": dataset, "trajectory": traj_name, "variant": vname,
                        "val_acc": statistics.mean(vals),
                        "val_acc_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    })
    write_csv(OUT_ROOT / "hop_probe_summary.csv", summary_rows)

    lines = ["# R0-HOP-REPORT — Multi-hop Potential (frozen representation probe)", ""]
    lines.append("> Row-normalized message-direction diffusion P=D^{-1}A；固定 Ridge probe（train fit / val eval）。")
    lines.append("")
    for traj_name in ("z_local", "z_final"):
        lines.append(f"## trajectory: {traj_name}")
        lines.append("")
        lines.append("| dataset | Probe(Z0) | Δ_hop1 [Z0|Z1] | Δ_hop2 [Z0|Z1|Z2] | Δ_hop3 [Z0..Z3] |")
        lines.append("|---|---:|---:|---:|---:|")
        for dataset in datasets:
            def v(vname):
                rows = [r for r in summary_rows if r["dataset"] == dataset
                        and r["trajectory"] == traj_name and r["variant"] == vname]
                return rows[0] if rows else None
            z0 = v("Z0")
            h1, h2, h3 = v("Z0|Z1"), v("Z0|Z1|Z2"), v("Z0|Z1|Z2|Z3")
            if z0 and h1 and h2 and h3:
                lines.append(
                    f"| {dataset} | {z0['val_acc']:.4f} | {100*(h1['val_acc']-z0['val_acc']):+.2f} | "
                    f"{100*(h2['val_acc']-z0['val_acc']):+.2f} | {100*(h3['val_acc']-z0['val_acc']):+.2f} |"
                )
        lines.append("")
    (OUT_ROOT / "R0_HOP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[hop] done -> {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
