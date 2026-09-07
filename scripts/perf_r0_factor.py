"""R0-D2: semantic factor quality diagnostic (plan §5/§33 Prompt 3).

For each OFR checkpoint: label-free factor statistics + the FIXED linear
probe (StandardScaler + Ridge(alpha=1.0), fit TRAIN, eval VAL) over:
h_t, h_v, [h_t|h_v], C, Pt, Pv, [C|Pt|Pv], z_local, C', Pt', Pv',
[C'|Pt'|Pv'], z_final.

Core quantities:
    Delta_fact  = Probe([C|Pt|Pv]) - Probe([h_t|h_v])   (factorization gap)
    Delta_graph = Probe(z_final)    - Probe(z_local)    (graph gain)
    per-factor  Probe(f') - Probe(f)

No test access; no probe hyperparameter tuning. Frozen model untouched.
Outputs: outputs/perf_r0/factor/{factor_stats_per_seed.csv,
factor_probe_per_seed.csv, factor_probe_summary.csv, R0_FACTOR_REPORT.md}
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
    chunked_pairwise_overlap,
    extract_forward,
    load_setup,
    ridge_probe,
    write_csv,
)

OUT_ROOT = PROJECT_ROOT / "outputs" / "perf_r0" / "factor"

REPRESENTATIONS = [
    "h_t", "h_v", "h_t|h_v",
    "C", "Pt", "Pv", "C|Pt|Pv",
    "z_local",
    "C'", "Pt'", "Pv'", "C'|Pt'|Pv'",
    "z_final",
]


def _representation_tensor(fex: dict, name: str) -> torch.Tensor:
    factors, f_tilde = fex["factors"], fex["f_tilde"]
    if name == "h_t":
        return factors["h_t"]
    if name == "h_v":
        return factors["h_v"]
    if name == "h_t|h_v":
        return torch.cat([factors["h_t"], factors["h_v"]], dim=-1)
    if name == "C":
        return factors["c"]
    if name == "Pt":
        return factors["p_t"]
    if name == "Pv":
        return factors["p_v"]
    if name == "C|Pt|Pv":
        return torch.cat([factors["c"], factors["p_t"], factors["p_v"]], dim=-1)
    if name == "z_local":
        return fex["z_local"]
    if name == "C'":
        return f_tilde[:, 0]
    if name == "Pt'":
        return f_tilde[:, 1]
    if name == "Pv'":
        return f_tilde[:, 2]
    if name == "C'|Pt'|Pv'":
        return f_tilde.reshape(f_tilde.size(0), -1)
    if name == "z_final":
        return fex["z_final"]
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="R0-D2 factor quality diagnostic")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    datasets = [d for d in (args.datasets or ",".join(DATASETS)).split(",") if d in DATASETS]
    seeds = [int(s) for s in (args.seeds or ",".join(map(str, SEEDS))).split(",") if s]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    probe_rows: list[dict] = []
    stat_rows: list[dict] = []
    for di, dataset in enumerate(datasets):
        device = torch.device(f"cuda:{gpus[di % len(gpus)]}")
        for seed in seeds:
            setup = load_setup(dataset, seed, device)
            fex = extract_forward(setup)
            factors, f_tilde = fex["factors"], fex["f_tilde"]
            stat_row = {
                "dataset": dataset, "seed": seed,
                "common_alignment_ct_cv": chunked_mean_cos(factors["c_t"], factors["c_v"]),
                "private_sim_pt_pv": chunked_mean_cos(factors["p_t"], factors["p_v"]),
            }
            for (a_name, a), (b_name, b) in [
                (("C", factors["c"]), ("Pt", factors["p_t"])),
                (("C", factors["c"]), ("Pv", factors["p_v"])),
                (("Pt", factors["p_t"]), ("Pv", factors["p_v"])),
            ]:
                ov = chunked_pairwise_overlap(a, b)
                stat_row[f"overlap_{a_name}_{b_name}_cos"] = ov["mean_cos"]
                stat_row[f"overlap_{a_name}_{b_name}_xcov"] = ov["mean_abs_xcov"]
            stat_rows.append(stat_row)

            for name in REPRESENTATIONS:
                tensor = _representation_tensor(fex, name)
                probe = ridge_probe(tensor, setup)
                probe_rows.append({
                    "dataset": dataset, "seed": seed, "representation": name,
                    "val_acc": probe["val_acc"], "val_macro_f1": probe["val_macro_f1"],
                })
                print(f"[factor] {dataset:12s} s{seed} {name:10s} val={probe['val_acc']:.4f}", flush=True)
                del tensor
                torch.cuda.empty_cache()
            del fex
            torch.cuda.empty_cache()

    write_csv(OUT_ROOT / "factor_stats_per_seed.csv", stat_rows)
    write_csv(OUT_ROOT / "factor_probe_per_seed.csv", probe_rows)

    # summary over seeds
    summary_rows: list[dict] = []
    for dataset in datasets:
        for name in REPRESENTATIONS:
            vals = [r["val_acc"] for r in probe_rows if r["dataset"] == dataset and r["representation"] == name]
            f1s = [r["val_macro_f1"] for r in probe_rows if r["dataset"] == dataset and r["representation"] == name]
            summary_rows.append({
                "dataset": dataset, "representation": name,
                "val_acc": statistics.mean(vals) if vals else float("nan"),
                "val_acc_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "val_macro_f1": statistics.mean(f1s) if f1s else float("nan"),
            })
    write_csv(OUT_ROOT / "factor_probe_summary.csv", summary_rows)

    # report with the two core gaps
    lines = ["# R0-FACTOR-REPORT — Semantic Factor Quality", ""]
    lines.append("> 固定 Ridge probe（StandardScaler + Ridge alpha=1.0），fit TRAIN / eval VAL。")
    lines.append("> Δ_fact = Probe([C|Pt|Pv]) − Probe([h_t|h_v])；Δ_graph = Probe(z_final) − Probe(z_local)。")
    lines.append("")
    lines.append("| dataset | Probe([h_t|h_v]) | Probe([C|Pt|Pv]) | **Δ_fact (pp)** | Probe(z_local) | Probe(z_final) | **Δ_graph (pp)** |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        def v(name):
            rows = [r for r in summary_rows if r["dataset"] == dataset and r["representation"] == name]
            return rows[0] if rows else None
        htv, fac, zl, zf = v("h_t|h_v"), v("C|Pt|Pv"), v("z_local"), v("z_final")
        def fmt(row, key="val_acc"):
            return f"{row[key]:.4f}±{row['val_acc_std']:.4f}" if row else ""
        if htv and fac and zl and zf:
            lines.append(
                f"| {dataset} | {fmt(htv)} | {fmt(fac)} | "
                f"{100*(fac['val_acc'] - htv['val_acc']):+.2f} | {fmt(zl)} | {fmt(zf)} | "
                f"{100*(zf['val_acc'] - zl['val_acc']):+.2f} |"
            )
    lines.append("")
    lines.append("| dataset | Probe(C) → Probe(C') | Probe(Pt) → Probe(Pt') | Probe(Pv) → Probe(Pv') |")
    lines.append("|---|---:|---:|---:|")
    for dataset in datasets:
        def v(name):
            rows = [r for r in summary_rows if r["dataset"] == dataset and r["representation"] == name]
            return rows[0] if rows else None
        c, cp, pt, ptp, pv, pvp = v("C"), v("C'"), v("Pt"), v("Pt'"), v("Pv"), v("Pv'")
        if all(x is not None for x in (c, cp, pt, ptp, pv, pvp)):
            lines.append(
                f"| {dataset} | {c['val_acc']:.4f} → {cp['val_acc']:.4f} "
                f"({100*(cp['val_acc']-c['val_acc']):+.2f}pp) | "
                f"{pt['val_acc']:.4f} → {ptp['val_acc']:.4f} ({100*(ptp['val_acc']-pt['val_acc']):+.2f}pp) | "
                f"{pv['val_acc']:.4f} → {pvp['val_acc']:.4f} ({100*(pvp['val_acc']-pv['val_acc']):+.2f}pp) |"
            )
    lines.append("")
    (OUT_ROOT / "R0_FACTOR_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[factor] done -> {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
