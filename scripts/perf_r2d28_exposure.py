"""R2-Design-2.8 v2 D2.8-B: relational exposure decomposition (v2 §7).

Fixes (Rule I): pi = uniform real-neighbor mean, O = U_a, lambda = 1/3.
Only the exposure r varies.

Variants:
    E0 FIXED_FULL (r = 1)          — uniform real-neighbor aggregation
    E1 NODE_EXPOSURE               — one r_i shared across factors
    E2 TARGET_FACTOR_EXPOSURE      — three r_i^b
    E3 SOURCE_FACTOR_EXPOSURE      — three r_i^a
    E4 PAIR_EXPOSURE               — nine r_i^{a->b}

All r are shared-predictor outputs (sigmoid), capacity-matched against E4.
Reference rows A0_MATCHED / TARGET_NULL_ONLY_D27 / PAIR_EDGE_D27 are read
from the existing D2.7 CSVs at summarize time (no retraining).

Protocol: A0 frozen; lr 1e-3 AdamW wd 1e-4; warmup10+cosine; 300 epochs;
patience 30; best Val Acc; No Test.

Outputs: outputs/perf_r2d28/exposure/
    exposure_results.csv, exposure_stats.csv, R2D28_EXPOSURE_REPORT.md
    per-run <ds>/<variant>/seed_<s>/{summary.json, history.csv, best.pt, run.log}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d27_utils import (  # noqa: E402
    DATASETS,
    R2D27_ROOT,
    load_a0_parent,
    load_or_make_head_init,
)
from src.analysis.perf_r2d28_utils import (  # noqa: E402
    EXPOSURE_VARIANTS,
    HEAD_INIT_ROOT,
    R2D28_ROOT,
    build_model,
    launch_jobs,
    resolve_cfg,
    train_relfunc_model,
)

EXPOSURE_ROOT = R2D28_ROOT / "exposure"


def run_worker(dataset: str, variant: str, seed: int, outdir: Path,
               epochs: int | None, force: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} {variant} s{seed}] SKIP", flush=True)
        return
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    setup = load_a0_parent(dataset, seed, device)
    data = setup.data
    cfg = resolve_cfg(dataset, seed, EXPOSURE_VARIANTS[variant])
    model = build_model(cfg, data, setup.parent, device)
    head = load_or_make_head_init(
        HEAD_INIT_ROOT / f"{dataset}_seed{seed}_d{model.out_dim}.pt",
        model.out_dim, int(data.num_classes), device)
    total_epochs = 300 if epochs is None else int(epochs)
    t0 = time.monotonic()

    history_path = outdir / "history.csv"
    history_file = history_path.open("w", encoding="utf-8", newline="")
    history_writer = csv.DictWriter(history_file,
                                    fieldnames=["epoch", "lr", "train_ce", "val_acc"])
    history_writer.writeheader()
    res = train_relfunc_model(
        data, model, head, device, total_epochs=total_epochs,
        history_callback=history_writer.writerow)
    history_file.close()

    # exposure diagnostics at best checkpoint (TRAIN labels only)
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    exp_stats = model.export_exposure_stats(
        x, ei, data.train_idx, data.y[data.train_idx])
    torch.save({"head_state": head.state_dict(), "model_state": model.state_dict()},
               outdir / "best.pt")
    summary = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "exposure_kind": model.exposure_kind,
        "best_val_acc": res["best_val_acc"],
        "best_val_macro_f1": res["best_val_macro_f1"],
        "per_class_f1": res["per_class_f1"],
        "best_epoch": res["best_epoch"], "stop_epoch": res["stop_epoch"],
        "side_params": int(model.side_parameter_count),
        "trainable_params": int(res["trainable_params"]),
        "out_dim": int(model.out_dim),
        "exposure_stats": exp_stats,
        "runtime_sec": round(time.monotonic() - t0, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[run] {dataset} {variant} s{seed} best_acc={res['best_val_acc']:.5f} "
          f"f1={res['best_val_macro_f1']:.5f} ep={res['best_epoch']}/{res['stop_epoch']} "
          f"params={model.side_parameter_count} ({summary['runtime_sec']:.0f}s)",
          flush=True)


def load_reference_rows() -> list[dict]:
    """A0_MATCHED / TARGET_NULL_ONLY_D27 / PAIR_EDGE_D27 from D2.7 CSVs."""
    rows = []
    import csv as _csv
    matrix_csv = R2D27_ROOT / "matrix" / "matrix_results.csv"
    with matrix_csv.open(encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if r["variant"] in ("A0_BASE", "PAIR_EDGE", "TARGET_NULL_ONLY"):
                name = {"A0_BASE": "A0_MATCHED",
                        "PAIR_EDGE": "PAIR_EDGE_D27",
                        "TARGET_NULL_ONLY": "TARGET_NULL_ONLY_D27"}[r["variant"]]
                rows.append({
                    "dataset": r["dataset"], "variant": name, "seed": int(r["seed"]),
                    "best_val_acc": float(r["val_acc"]),
                    "best_val_macro_f1": float(r["val_macro_f1"]),
                    "reference": True,
                })
    return rows


def summarize() -> None:
    rows = []
    for p in sorted(EXPOSURE_ROOT.rglob("summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append(d)
    rows += load_reference_rows()

    with (EXPOSURE_ROOT / "exposure_results.csv").open("w", encoding="utf-8",
                                                       newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "variant", "seed", "val_acc", "val_macro_f1",
            "best_epoch", "stop_epoch", "side_params", "trainable_params",
            "runtime_sec", "peak_allocated_mb", "reference"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "dataset": r["dataset"], "variant": r["variant"],
                "seed": r["seed"], "val_acc": r.get("best_val_acc"),
                "val_macro_f1": r.get("best_val_macro_f1"),
                "best_epoch": r.get("best_epoch"), "stop_epoch": r.get("stop_epoch"),
                "side_params": r.get("side_params"),
                "trainable_params": r.get("trainable_params"),
                "runtime_sec": r.get("runtime_sec"),
                "peak_allocated_mb": r.get("peak_allocated_mb"),
                "reference": bool(r.get("reference", False)),
            })

    stats_rows = []
    for r in rows:
        if not r.get("exposure_stats") or r["exposure_stats"].get("fixed_full"):
            continue
        es = r["exposure_stats"]
        for key, st in es["per_key"].items():
            stats_rows.append({
                "dataset": r["dataset"], "variant": r["variant"],
                "seed": r["seed"], "r_key": key,
                "mean": st["mean"], "std": st["std"], "q10": st["q10"],
                "q50": st["q50"], "q90": st["q90"],
                "frac_lt_0.1": st["frac_lt_0.1"],
                "frac_gt_0.9": st["frac_gt_0.9"],
                "degree_corr": st["degree_corr"],
                "mean_r_train_class": json.dumps(st["mean_r_train_class"]),
            })
    with (EXPOSURE_ROOT / "exposure_stats.csv").open("w", encoding="utf-8",
                                                     newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        for r in stats_rows:
            w.writerow(r)

    from src.analysis.perf_r2d28_utils import mean_std_pp, paired_delta

    lines = ["# R2D28_EXPOSURE_REPORT — D2.8-B relational exposure",
             "",
             "Protocol: pi uniform / O = U_a / lambda = 1/3 (Rule I); A0 frozen;",
             "3 seeds; Val only; No Test.",
             ""]
    evars = ["E0", "E1", "E2", "E3", "E4"]
    # best exposure vs E0 (GO: >= +0.30pp Acc AND >= +0.20pp F1, M/T/G macro)
    for v in evars[1:]:
        lines.append(
            f"- {v} - E0: Acc {mean_std_pp(rows, v, 'E0', 'best_val_acc')} pp; "
            f"F1 {mean_std_pp(rows, v, 'E0', 'best_val_macro_f1')} pp")
    # pair specificity: E4 vs strongest E1/E2/E3
    best_simple = max(evars[1:4],
                      key=lambda v: paired_delta(rows, v, "E0", "best_val_acc")["mean"]
                      or -1e9)
    lines.append("")
    lines.append(f"- E4 - {best_simple}: Acc "
                 f"{mean_std_pp(rows, 'E4', best_simple, 'best_val_acc')} pp; F1 "
                 f"{mean_std_pp(rows, 'E4', best_simple, 'best_val_macro_f1')} pp")
    lines.append("")
    lines.append("E* selection: apply the v2 §7 thresholds (exposure GO and"
                 " pair-exposure specificity) by hand after inspecting the"
                 " table below; E2≈E4 prefers target-factor exposure.")
    lines.append("")
    lines.append("(R2D28_EXPOSURE_REPORT.md)")
    (EXPOSURE_ROOT / "R2D28_EXPOSURE_REPORT.md").write_text("\n".join(lines),
                                                            encoding="utf-8")
    print("[summarize] done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2D2.8 v2 D2.8-B exposure")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.dataset, args.variant, args.seed, Path(args.outdir),
                   args.epochs, args.force)
        return
    if args.summarize:
        summarize()
        return

    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    variants = list(EXPOSURE_VARIANTS) if not args.variants else \
        [v for v in args.variants.split(",")]
    unknown = [v for v in variants if v not in EXPOSURE_VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {unknown}")
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, v, s) for d in datasets for v in variants for s in seeds]
    launch_jobs(Path(__file__).resolve(), jobs, EXPOSURE_ROOT, gpus, args.force,
                args.epochs)


if __name__ == "__main__":
    main()
