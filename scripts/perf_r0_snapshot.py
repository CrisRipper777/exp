"""R0-D1: frozen baseline snapshot (plan §4/§32 Prompt 2).

For each OFR checkpoint (5 datasets x seeds 42/43/44): reload, verify the
current-Gamma val reproduction, re-aggregate existing P2/P3 diagnostics, and
add the three new relation metrics (Conf_R, H_R^norm, within-seed prototype
separation) plus factor label-free statistics. No test access. Big tensors
are chunked/streamed; nothing large is persisted.

Outputs: outputs/perf_r0/baseline_snapshot/{per_seed_snapshot.csv,
dataset_summary.csv, R0_BASELINE_SNAPSHOT.md}
"""

from __future__ import annotations

import argparse
import json
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
    FACTOR_NAMES,
    _sanities,
    chunked_mean_cos,
    chunked_mean_var,
    chunked_pairwise_overlap,
    extract_forward,
    load_setup,
    write_csv,
)

OUT_ROOT = PROJECT_ROOT / "outputs" / "perf_r0" / "baseline_snapshot"


def _prototype_separation(model) -> dict[str, float]:
    proto = model.relation_prototypes.prototypes.detach()  # [K, dim]
    proto = torch.nn.functional.normalize(proto, dim=-1)
    cos = proto @ proto.t()
    k = int(cos.size(0))
    off = [float(cos[i, j].item()) for i in range(k) for j in range(i + 1, k)]
    dists = [1.0 - c for c in off]
    return {
        "proto_mean_offdiag_cos": float(statistics.mean(off)) if off else float("nan"),
        "proto_max_offdiag_cos": float(max(off)) if off else float("nan"),
        "proto_min_pairwise_dist": float(min(dists)) if dists else float("nan"),
    }


def _relation_assignment_metrics(graph_out: dict) -> dict[str, float]:
    r = graph_out["r"]  # [E, K]
    k = int(r.size(1))
    max_conf = float(r.max(dim=-1).values.mean().item())
    log_r = torch.log(r + 1e-8)
    ent = -(r * log_r).sum(dim=-1)
    h_norm = float(ent.mean().item()) / float(torch.log(torch.tensor(float(k))).item())
    return {"edge_max_confidence": max_conf, "norm_relation_entropy": h_norm}


def _factor_stats(factors: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    out["common_alignment_ct_cv"] = chunked_mean_cos(factors["c_t"], factors["c_v"])
    out["private_sim_pt_pv"] = chunked_mean_cos(factors["p_t"], factors["p_v"])
    for (a_name, a), (b_name, b) in [
        (("C", factors["c"]), ("Pt", factors["p_t"])),
        (("C", factors["c"]), ("Pv", factors["p_v"])),
        (("Pt", factors["p_t"]), ("Pv", factors["p_v"])),
    ]:
        ov = chunked_pairwise_overlap(a, b)
        out[f"overlap_{a_name}_{b_name}_cos"] = ov["mean_cos"]
        out[f"overlap_{a_name}_{b_name}_xcov"] = ov["mean_abs_xcov"]
    for name, tensor in (("C", factors["c"]), ("Pt", factors["p_t"]), ("Pv", factors["p_v"])):
        mean_norm = float(tensor.norm(dim=-1).mean().item())
        var = chunked_mean_var(tensor)[1]
        out[f"norm_{name}"] = mean_norm
        out[f"var_{name}"] = var
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="R0-D1 frozen baseline snapshot")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    datasets = [d for d in (args.datasets or ",".join(DATASETS)).split(",") if d in DATASETS]
    seeds = [int(s) for s in (args.seeds or ",".join(map(str, SEEDS))).split(",") if s]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for di, dataset in enumerate(datasets):
        device = torch.device(f"cuda:{gpus[di % len(gpus)]}")
        for seed in seeds:
            setup = load_setup(dataset, seed, device)
            x = setup.data.x.to(device)
            edge_index = setup.data.edge_index.to(device)
            fex = extract_forward(setup)
            sanities = _sanities(fex["graph_out"], fex["deg"])
            diag = setup.model.compute_p3_diagnostics(x, edge_index)
            from src.analysis.perf_r0_utils import val_metrics_with_head

            val_repro = val_metrics_with_head(setup, fex["z_final"])
            row = {
                "dataset": dataset,
                "seed": seed,
                "val_acc_recomputed": val_repro["val_acc"],
                **sanities,
                **_relation_assignment_metrics(fex["graph_out"]),
                **_prototype_separation(setup.model),
                **_factor_stats(fex["factors"]),
                "K_eff": diag["relation"]["effective_num"],
                "S_R": diag["relation"]["specialization"],
            }
            for fname in FACTOR_NAMES:
                plan = diag["plan"].get(fname, {})
                row[f"null_{fname}"] = plan.get("null_mean")
                row[f"graph_{fname}"] = plan.get("graph_mass_mean")
                row[f"plan_ent_{fname}"] = diag["plan_entropy"].get(fname)
            um = diag["usage_matrix"]
            for fname, vals in zip(um["factors"], um["values"]):
                for rname, value in zip(um["relations"], vals):
                    row[f"usage_{fname}_{rname}"] = value
            op = diag["operator"]
            row["pair_strength"] = op.get("pair_strength")
            row["message_dev"] = op.get("message_deviation_usage_weighted")
            row["interaction_strength"] = op.get("interaction", {}).get("usage_weighted_strength")
            row["interaction_average"] = op.get("interaction", {}).get("usage_weighted_average")
            rows.append(row)
            print(
                f"[snapshot] {dataset:12s} s{seed} val={val_repro['val_acc']:.4f} "
                f"K_eff={diag['relation']['effective_num']:.2f} S_R={diag['relation']['specialization']:.3f} "
                f"conf={row['edge_max_confidence']:.3f} H_norm={row['norm_relation_entropy']:.3f}",
                flush=True,
            )
            del fex, diag
            torch.cuda.empty_cache()

    write_csv(OUT_ROOT / "per_seed_snapshot.csv", rows)

    # dataset summary: mean over seeds for numeric fields
    numeric_keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    summary_rows: list[dict] = []
    for dataset in datasets:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        srow: dict = {"dataset": dataset, "n_seeds": len(ds_rows)}
        for key in numeric_keys:
            vals = [r[key] for r in ds_rows if r[key] is not None]
            if vals:
                srow[f"{key}_mean"] = statistics.mean(vals)
                srow[f"{key}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append(srow)
    write_csv(OUT_ROOT / "dataset_summary.csv", summary_rows)

    lines = ["# R0-BASELINE-SNAPSHOT — Frozen OFR baseline", ""]
    lines.append("> 每 seed 的 val 复现、新 relation 指标、prototype 分离、factor label-free 统计；")
    lines.append("> dataset_summary.csv 为 seed 均值。详见 per_seed_snapshot.csv。")
    lines.append("")
    lines.append("| dataset | Conf_R | H_R^norm | proto mean cos | proto min dist | C↔Pt cos | Pt↔Pv cos |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for srow in summary_rows:
        lines.append(
            f"| {srow['dataset']} | {srow.get('edge_max_confidence_mean', 0):.3f} | "
            f"{srow.get('norm_relation_entropy_mean', 0):.3f} | "
            f"{srow.get('proto_mean_offdiag_cos_mean', 0):.3f} | "
            f"{srow.get('proto_min_pairwise_dist_mean', 0):.3f} | "
            f"{srow.get('overlap_C_Pt_cos_mean', 0):.3f} | "
            f"{srow.get('overlap_Pt_Pv_cos_mean', 0):.3f} |"
        )
    lines.append("")
    (OUT_ROOT / "R0_BASELINE_SNAPSHOT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[snapshot] done -> {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
