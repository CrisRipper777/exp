"""R2-Design-2.8 v2 D2.8-D: source-channel integration (v2 §9).

Fixes (Rule III + Rule V): the chosen exposure E* and composition C* are
LOADED and FROZEN; operator = U_a; only the source-channel integration
changes.

Variants:
    M0 SOURCE_MEAN             m^b = mean_a m^{a->b}
    M1 SOURCE_SOFTMAX_MIX      lambda = Softmax_a(g(F_i^b, e_a)); simplex
    M2 SOURCE_CONCAT_MLP       per b: Linear(3d,2d)->LN->GELU->Drop->Linear(2d,d)
    M2_MEAN_DUP                same M2 net, mean message in all three slots
    M3 TARGET_QUERY_SOURCE_ATTN 3 source tokens, query F_i^b, 2 Pre-LN blocks
    M3_MEAN_DUP                same M3 net, mean message as all three tokens

Channel support (v2 §9) requires BOTH:
    best M - M0 >= +0.20pp Acc or F1  (other metric nonnegative)
    best M - matched MEAN_DUP >= +0.20pp Acc or F1
— source-channel identity must beat the same readout fed the mean message.

Outputs: outputs/perf_r2d28/channel/
    channel_results.csv, channel_ablation.csv, R2D28_CHANNEL_REPORT.md
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
    load_a0_parent,
    load_or_make_head_init,
)
from src.analysis.perf_r2d28_utils import (  # noqa: E402
    CHANNEL_VARIANTS,
    COMPOSITION_ROOT,
    EXPOSURE_ROOT,
    HEAD_INIT_ROOT,
    R2D28_ROOT,
    build_model,
    channel_prefixes,
    composition_prefixes,
    exposure_prefixes,
    launch_jobs,
    load_frozen_components,
    resolve_cfg,
    train_relfunc_model,
)

CHANNEL_ROOT = R2D28_ROOT / "channel"
DEFAULT_EXPOSURE = "E4"
DEFAULT_COMPOSITION = "C4"

EXPOSURE_KINDS = {"E0": "fixed_full", "E1": "node", "E2": "target",
                  "E3": "source", "E4": "pair"}
COMPOSITION_KINDS = {"C0": "uniform", "C1": "generic", "C2": "target",
                     "C3": "source", "C4": "pair"}


def run_worker(dataset: str, variant: str, seed: int, outdir: Path,
               epochs: int | None, force: bool, exposure_variant: str,
               composition_variant: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} {variant} s{seed}] SKIP", flush=True)
        return
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    setup = load_a0_parent(dataset, seed, device)
    data = setup.data
    overrides = dict(CHANNEL_VARIANTS[variant])
    overrides["exposure"] = EXPOSURE_KINDS[exposure_variant]
    overrides["composition"] = COMPOSITION_KINDS[composition_variant]
    overrides["freeze_exposure"] = True
    overrides["freeze_composition"] = True
    cfg = resolve_cfg(dataset, seed, overrides)
    model = build_model(cfg, data, setup.parent, device)

    # Rule V: load + freeze E* and C* (NO-GO slots with no module params are
    # skipped — e.g. C0 uniform has no composition modules)
    e_ckpt = EXPOSURE_ROOT / dataset / exposure_variant / f"seed_{seed}" / "best.pt"
    c_ckpt = COMPOSITION_ROOT / dataset / composition_variant / f"seed_{seed}" / "best.pt"
    load_info_e = load_frozen_components(model, e_ckpt, exposure_prefixes())
    c_prefixes = (composition_prefixes()
                  if model.composition_kind != "uniform" else [])
    load_info_c = load_frozen_components(model, c_ckpt, c_prefixes) \
        if c_prefixes else {"copied_params": 0}
    model._apply_freezes()

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

    x = data.x.to(device)
    ei = data.edge_index.to(device)
    channel_stats = model.export_channel_stats(x, ei)
    torch.save({"head_state": head.state_dict(), "model_state": model.state_dict()},
               outdir / "best.pt")
    summary = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "exposure_kind": model.exposure_kind,
        "composition_kind": model.composition_kind,
        "channel_kind": model.channel_kind,
        "mean_dup": bool(model.mean_dup),
        "frozen_from": [str(e_ckpt), str(c_ckpt)],
        "frozen_copied_params": load_info_e["copied_params"]
        + load_info_c["copied_params"],
        "best_val_acc": res["best_val_acc"],
        "best_val_macro_f1": res["best_val_macro_f1"],
        "per_class_f1": res["per_class_f1"],
        "best_epoch": res["best_epoch"], "stop_epoch": res["stop_epoch"],
        "side_params": int(model.side_parameter_count),
        "trainable_params": int(res["trainable_params"]),
        "out_dim": int(model.out_dim),
        "channel_stats": channel_stats,
        "runtime_sec": round(time.monotonic() - t0, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[run] {dataset} {variant} s{seed} best_acc={res['best_val_acc']:.5f} "
          f"f1={res['best_val_macro_f1']:.5f} ep={res['best_epoch']}/{res['stop_epoch']} "
          f"params={model.side_parameter_count} ({summary['runtime_sec']:.0f}s)",
          flush=True)


def summarize() -> None:
    rows = []
    for p in sorted(CHANNEL_ROOT.rglob("summary.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))

    with (CHANNEL_ROOT / "channel_results.csv").open("w", encoding="utf-8",
                                                     newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "variant", "seed", "val_acc", "val_macro_f1",
            "best_epoch", "stop_epoch", "side_params", "trainable_params",
            "runtime_sec", "peak_allocated_mb"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "dataset": r["dataset"], "variant": r["variant"],
                "seed": r["seed"], "val_acc": r["best_val_acc"],
                "val_macro_f1": r["best_val_macro_f1"],
                "best_epoch": r["best_epoch"], "stop_epoch": r["stop_epoch"],
                "side_params": r["side_params"],
                "trainable_params": r["trainable_params"],
                "runtime_sec": r["runtime_sec"],
                "peak_allocated_mb": r["peak_allocated_mb"],
            })

    # channel ablations: M vs M0 and M vs MEAN_DUP, and dup vs M0
    from src.analysis.perf_r2d28_utils import mean_std_pp

    ablation_rows = []
    pairs = [("M1", "M0"), ("M2", "M0"), ("M3", "M0"),
             ("M2", "M2_MEAN_DUP"), ("M3", "M3_MEAN_DUP"),
             ("M2_MEAN_DUP", "M0"), ("M3_MEAN_DUP", "M0")]
    lines = ["# R2D28_CHANNEL_REPORT — D2.8-D source-channel integration",
             "",
             "Rule III + Rule V: E* and C* loaded and frozen; only the",
             "source-channel integration varies; lambda is a simplex.",
             ""]
    for cand, base in pairs:
        acc = mean_std_pp(rows, cand, base, "best_val_acc")
        f1 = mean_std_pp(rows, cand, base, "best_val_macro_f1")
        ablation_rows.append({"candidate": cand, "base": base, "metric": "acc",
                              "delta_pp": acc})
        ablation_rows.append({"candidate": cand, "base": base, "metric": "f1",
                              "delta_pp": f1})
        lines.append(f"- {cand} - {base}: Acc {acc} pp; F1 {f1} pp")
    lines += [
        "",
        "M* selection (v2 §9): best M - M0 >= +0.20pp AND best M - matched",
        "MEAN_DUP >= +0.20pp (Acc or F1, other metric nonnegative).",
        "",
        "(R2D28_CHANNEL_REPORT.md)",
    ]
    with (CHANNEL_ROOT / "channel_ablation.csv").open("w", encoding="utf-8",
                                                      newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate", "base", "metric", "delta_pp"])
        w.writeheader()
        for r in ablation_rows:
            w.writerow(r)
    (CHANNEL_ROOT / "R2D28_CHANNEL_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("[summarize] done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2D2.8 v2 D2.8-D channel")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--exposure-variant", default=DEFAULT_EXPOSURE)
    parser.add_argument("--composition-variant", default=DEFAULT_COMPOSITION)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    assert args.exposure_variant in EXPOSURE_KINDS, args.exposure_variant
    assert args.composition_variant in COMPOSITION_KINDS, args.composition_variant

    if args.worker:
        run_worker(args.dataset, args.variant, args.seed, Path(args.outdir),
                   args.epochs, args.force, args.exposure_variant,
                   args.composition_variant)
        return
    if args.summarize:
        summarize()
        return

    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    variants = list(CHANNEL_VARIANTS) if not args.variants else \
        [v for v in args.variants.split(",")]
    unknown = [v for v in variants if v not in CHANNEL_VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {unknown}")
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, v, s) for d in datasets for v in variants for s in seeds]
    launch_jobs(Path(__file__).resolve(), jobs, CHANNEL_ROOT, gpus, args.force,
                args.epochs, extra_flags=[
                    "--exposure-variant", args.exposure_variant,
                    "--composition-variant", args.composition_variant])


if __name__ == "__main__":
    main()
