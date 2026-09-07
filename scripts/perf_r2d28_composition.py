"""R2-Design-2.8 v2 D2.8-C: neighbor-composition decomposition (v2 §8).

Fixes (Rule II + Rule V): the chosen exposure E* is LOADED and FROZEN;
pi = Softmax over REAL neighbors only (no null inside pi); exposure r
multiplies afterwards. O = U_a, lambda = 1/3.

Variants:
    C0 UNIFORM_COMP       pi = 1/deg
    C1 GENERIC_COMP       one pi shared across factors (width-matched)
    C2 TARGET_FACTOR_COMP three pi^b (psi on target-factor features)
    C3 SOURCE_FACTOR_COMP three pi^a
    C4 PAIR_COMP          nine pi^{a->b}

The exposure architecture/params are identical across all C variants
(staged loading from the E* checkpoint); simpler scorers are capacity
matched (+/-5%). Same U_a, same source mean, same classifier init.

Composition support (v2 §8): best C - C0 >= +0.20pp Acc or +0.30pp F1 with
the other metric nonnegative; pair specificity: C4 beats the strongest
C1/C2/C3 by +0.20pp Acc with F1 >= 0. Causal confirmation on the best C:
corrected within-target shuffle, per-target top/random/bottom removal,
source-node shuffle, factor-id shuffle. Composition is not accepted on
performance alone (v2 §8).

Outputs: outputs/perf_r2d28/composition/
    composition_results.csv, composition_causal.csv, R2D28_COMPOSITION_REPORT.md
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
    causal_metrics,
    load_a0_parent,
    load_or_make_head_init,
)
from src.analysis.perf_r2d28_utils import (  # noqa: E402
    COMPOSITION_VARIANTS,
    EXPOSURE_ROOT,
    HEAD_INIT_ROOT,
    R2D28_ROOT,
    build_model,
    composition_prefixes,
    exposure_prefixes,
    launch_jobs,
    load_frozen_components,
    resolve_cfg,
    train_relfunc_model,
)

COMPOSITION_ROOT = R2D28_ROOT / "composition"
DEFAULT_EXPOSURE = "E4"  # replaced by the B-stage verdict at launch time

COMP_CAUSAL_KEYS = (
    "within_target_shuffle",
    "remove_top_per_target_10", "remove_random_per_target_10",
    "remove_bottom_per_target_10",
    "source_shuffle", "factor_id_shuffle",
)


def run_worker(dataset: str, variant: str, seed: int, outdir: Path,
               epochs: int | None, force: bool, exposure_variant: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} {variant} s{seed}] SKIP", flush=True)
        return
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    setup = load_a0_parent(dataset, seed, device)
    data = setup.data
    overrides = dict(COMPOSITION_VARIANTS[variant])
    overrides["exposure"] = EXPOSURE_VARIANTS_CFG[exposure_variant]
    overrides["freeze_exposure"] = True
    cfg = resolve_cfg(dataset, seed, overrides)
    model = build_model(cfg, data, setup.parent, device)

    # Rule V: load + freeze the E* exposure predictor and base payload
    e_ckpt = EXPOSURE_ROOT / dataset / exposure_variant / f"seed_{seed}" / "best.pt"
    load_info = load_frozen_components(model, e_ckpt, exposure_prefixes())
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

    # composition causal audit (learned composition only)
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    causal = None
    comp_stats = None
    if model.composition_kind != "uniform":
        causal = causal_metrics(model, head, x, ei, data, device, COMP_CAUSAL_KEYS)
        comp_stats = model.export_comp_stats(x, ei)
    torch.save({"head_state": head.state_dict(), "model_state": model.state_dict()},
               outdir / "best.pt")
    summary = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "exposure_kind": model.exposure_kind,
        "composition_kind": model.composition_kind,
        "frozen_from": str(e_ckpt),
        "frozen_copied_params": load_info["copied_params"],
        "best_val_acc": res["best_val_acc"],
        "best_val_macro_f1": res["best_val_macro_f1"],
        "per_class_f1": res["per_class_f1"],
        "best_epoch": res["best_epoch"], "stop_epoch": res["stop_epoch"],
        "side_params": int(model.side_parameter_count),
        "trainable_params": int(res["trainable_params"]),
        "out_dim": int(model.out_dim),
        "causal": causal,
        "comp_stats": comp_stats,
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
    for p in sorted(COMPOSITION_ROOT.rglob("summary.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))

    with (COMPOSITION_ROOT / "composition_results.csv").open(
            "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "variant", "seed", "val_acc", "val_macro_f1",
            "best_epoch", "stop_epoch", "side_params", "trainable_params",
            "runtime_sec", "peak_allocated_mb", "frozen_copied_params"])
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
                "frozen_copied_params": r["frozen_copied_params"],
            })

    causal_rows = []
    for r in rows:
        if not r.get("causal"):
            continue
        for key, m in r["causal"].items():
            causal_rows.append({
                "dataset": r["dataset"], "variant": r["variant"],
                "seed": r["seed"], "causal": key,
                "val_acc": m["val_acc"], "val_macro_f1": m["val_macro_f1"],
            })
    with (COMPOSITION_ROOT / "composition_causal.csv").open(
            "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "causal",
                                          "val_acc", "val_macro_f1"])
        w.writeheader()
        for r in causal_rows:
            w.writerow(r)

    from src.analysis.perf_r2d28_utils import mean_std_pp

    lines = ["# R2D28_COMPOSITION_REPORT — D2.8-C neighbor composition",
             "",
             "Rule II + Rule V: E* exposure loaded and frozen; pi over real",
             "neighbors only; O = U_a; lambda = 1/3.",
             ""]
    for v in ("C1", "C2", "C3", "C4"):
        lines.append(
            f"- {v} - C0: Acc {mean_std_pp(rows, v, 'C0', 'best_val_acc')} pp; "
            f"F1 {mean_std_pp(rows, v, 'C0', 'best_val_macro_f1')} pp")
    lines += [
        "",
        "C* selection: v2 §8 thresholds — composition GO (>=+0.20pp Acc or",
        "+0.30pp F1, other metric nonnegative) AND corrected causal evidence",
        "(shuffle/per-target removal/source shuffle/factor-id shuffle).",
        "",
        "(R2D28_COMPOSITION_REPORT.md)",
    ]
    (COMPOSITION_ROOT / "R2D28_COMPOSITION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("[summarize] done", flush=True)


EXPOSURE_VARIANTS_CFG = {
    "E0": "fixed_full", "E1": "node", "E2": "target",
    "E3": "source", "E4": "pair",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="R2D2.8 v2 D2.8-C composition")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--exposure-variant", default=DEFAULT_EXPOSURE,
                        help="E* selected in D2.8-B")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    assert args.exposure_variant in EXPOSURE_VARIANTS_CFG, args.exposure_variant

    if args.worker:
        run_worker(args.dataset, args.variant, args.seed, Path(args.outdir),
                   args.epochs, args.force, args.exposure_variant)
        return
    if args.summarize:
        summarize()
        return

    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    variants = list(COMPOSITION_VARIANTS) if not args.variants else \
        [v for v in args.variants.split(",")]
    unknown = [v for v in variants if v not in COMPOSITION_VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {unknown}")
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, v, s) for d in datasets for v in variants for s in seeds]
    launch_jobs(Path(__file__).resolve(), jobs, COMPOSITION_ROOT, gpus, args.force,
                args.epochs, extra_flags=["--exposure-variant", args.exposure_variant])


if __name__ == "__main__":
    main()
