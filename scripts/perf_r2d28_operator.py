"""R2-Design-2.8 v2 D2.8-E: functional operator capacity (v2 §10).

Fixes (Rule IV + Rule V): the chosen E*, C*, M* are LOADED and FROZEN;
only the operator freedom changes. The PRIMARY diagnostic uses NormMatch
(Ohat(v) = Otilde(v)/||Otilde|| * ||v||) so the operator changes feature
content, not graph amplitude (r remains the only explicit amplitude).

Variants:
    O0 SOURCE_LINEAR          v = U_a F  (no operator)
    O1 STATIC_PAIR_RESIDUAL   v + DeltaT_ab(v), zero-init final (step0 == O0)
    O2 TARGET_FILM            (1+dg_i^ab)*v + beta_i^ab, zero-init final
    O3 EDGE_FILM              edge-conditioned FiLM (chunked), zero-init final
    O4 DYNAMIC_BASIS          K=4 residual bases + softmax router, small-init
    O4_UNIFORM                same bases, q = 1/K
    O4_TARGET                 same bases, target-conditioned router only

Causal overrides at best checkpoint: film_neutralize (O2/O3),
operator_shuffle / router_permute (O3/O4, within-target), router_uniformize
(O4). Operator GO requires performance + dynamic-vs-static specificity +
functional diversity + causal usage (v2 Prompt 6).

Formal matrix: Movies/Toys/Grocery x 3 seeds first; candidates reaching
A0_MATCHED+0.20pp or mechanism GO then run ele-fashion/Reddit-S guards.

Secondary test (v2 §11): if a norm-preserving operator gets GO, one
UNRESTRICTED-vs-norm-preserving comparison (norm_match=false) may run as a
secondary performance variant — never as primary mechanism evidence.

Outputs: outputs/perf_r2d28/operator/
    operator_results.csv, operator_controls.csv, operator_usage.csv,
    operator_causal.csv, R2D28_OPERATOR_REPORT.md
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
    GUARD_DATASETS,
    TARGET_DATASETS,
    causal_metrics,
    load_a0_parent,
    load_or_make_head_init,
)
from src.analysis.perf_r2d28_utils import (  # noqa: E402
    CHANNEL_ROOT,
    COMPOSITION_ROOT,
    EXPOSURE_ROOT,
    HEAD_INIT_ROOT,
    OPERATOR_VARIANTS,
    R2D28_ROOT,
    build_model,
    channel_prefixes,
    composition_prefixes,
    exposure_prefixes,
    launch_jobs,
    load_frozen_components,
    operator_prefixes,
    resolve_cfg,
    train_relfunc_model,
)

OPERATOR_ROOT = R2D28_ROOT / "operator"
DEFAULT_EXPOSURE = "E4"
DEFAULT_COMPOSITION = "C4"
DEFAULT_CHANNEL = "M0"

EXPOSURE_KINDS = {"E0": "fixed_full", "E1": "node", "E2": "target",
                  "E3": "source", "E4": "pair"}
COMPOSITION_KINDS = {"C0": "uniform", "C1": "generic", "C2": "target",
                     "C3": "source", "C4": "pair"}
CHANNEL_KINDS = {"M0": "mean", "M1": "softmax", "M2": "concat", "M3": "attn"}

OPERATOR_CAUSAL = {
    "linear": (),
    "static_pair": (),
    "target_film": ("film_neutralize",),
    "edge_film": ("film_neutralize", "operator_shuffle"),
    "basis": ("router_uniformize", "router_permute"),
}


def run_worker(dataset: str, variant: str, seed: int, outdir: Path,
               epochs: int | None, force: bool, exposure_variant: str,
               composition_variant: str, channel_variant: str,
               norm_match: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} {variant} s{seed}] SKIP", flush=True)
        return
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    setup = load_a0_parent(dataset, seed, device)
    data = setup.data
    overrides = dict(OPERATOR_VARIANTS[variant])
    overrides["exposure"] = EXPOSURE_KINDS[exposure_variant]
    overrides["composition"] = COMPOSITION_KINDS[composition_variant]
    overrides["channel"] = CHANNEL_KINDS[channel_variant]
    overrides["freeze_exposure"] = True
    overrides["freeze_composition"] = True
    overrides["freeze_channel"] = True
    overrides["norm_match"] = bool(norm_match)
    cfg = resolve_cfg(dataset, seed, overrides)
    model = build_model(cfg, data, setup.parent, device)

    # Rule V: load + freeze E*, C*, M* (NO-GO slots with no module params are
    # skipped — e.g. C0 uniform / M0 mean have no modules)
    e_ckpt = EXPOSURE_ROOT / dataset / exposure_variant / f"seed_{seed}" / "best.pt"
    c_ckpt = COMPOSITION_ROOT / dataset / composition_variant / f"seed_{seed}" / "best.pt"
    m_ckpt = CHANNEL_ROOT / dataset / channel_variant / f"seed_{seed}" / "best.pt"
    c_prefixes = (composition_prefixes()
                  if model.composition_kind != "uniform" else [])
    m_prefixes = (channel_prefixes() if model.channel_kind != "mean" else [])
    copied = load_frozen_components(model, e_ckpt, exposure_prefixes())["copied_params"]
    if c_prefixes:
        copied += load_frozen_components(model, c_ckpt, c_prefixes)["copied_params"]
    if m_prefixes:
        copied += load_frozen_components(model, m_ckpt, m_prefixes)["copied_params"]
    load_info = {"copied_params": copied}
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
    causal = causal_metrics(model, head, x, ei, data, device,
                            OPERATOR_CAUSAL[model.operator_kind])
    op_stats = model.export_operator_stats(x, ei)
    torch.save({"head_state": head.state_dict(), "model_state": model.state_dict()},
               outdir / "best.pt")
    summary = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "exposure_kind": model.exposure_kind,
        "composition_kind": model.composition_kind,
        "channel_kind": model.channel_kind,
        "operator_kind": model.operator_kind,
        "norm_match": bool(model.norm_match),
        "uniform_router": bool(getattr(model.operator_net, "uniform_router", False)),
        "target_router": bool(getattr(model.operator_net, "target_router", False)),
        "frozen_from": [str(e_ckpt), str(c_ckpt), str(m_ckpt)],
        "frozen_copied_params": load_info["copied_params"],
        "best_val_acc": res["best_val_acc"],
        "best_val_macro_f1": res["best_val_macro_f1"],
        "per_class_f1": res["per_class_f1"],
        "best_epoch": res["best_epoch"], "stop_epoch": res["stop_epoch"],
        "side_params": int(model.side_parameter_count),
        "trainable_params": int(res["trainable_params"]),
        "out_dim": int(model.out_dim),
        "causal": causal,
        "operator_stats": op_stats,
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
    for p in sorted(OPERATOR_ROOT.rglob("summary.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))

    with (OPERATOR_ROOT / "operator_results.csv").open("w", encoding="utf-8",
                                                       newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "variant", "seed", "val_acc", "val_macro_f1",
            "best_epoch", "stop_epoch", "side_params", "trainable_params",
            "norm_match", "uniform_router", "target_router",
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
                "norm_match": r["norm_match"],
                "uniform_router": r["uniform_router"],
                "target_router": r["target_router"],
                "runtime_sec": r["runtime_sec"],
                "peak_allocated_mb": r["peak_allocated_mb"],
            })

    causal_rows = []
    for r in rows:
        for key, m in (r.get("causal") or {}).items():
            causal_rows.append({
                "dataset": r["dataset"], "variant": r["variant"],
                "seed": r["seed"], "causal": key,
                "val_acc": m["val_acc"], "val_macro_f1": m["val_macro_f1"],
            })
    with (OPERATOR_ROOT / "operator_causal.csv").open("w", encoding="utf-8",
                                                      newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "causal",
                                          "val_acc", "val_macro_f1"])
        w.writeheader()
        for r in causal_rows:
            w.writerow(r)

    usage_rows = []
    for r in rows:
        st = r.get("operator_stats") or {}
        if r.get("operator_kind") in ("target_film", "edge_film"):
            for key in sorted(k for k in st if k.startswith("pair_")):
                usage_rows.append({
                    "dataset": r["dataset"], "variant": r["variant"],
                    "seed": r["seed"], "kind": r["operator_kind"], "pair": key,
                    "gamma_mean": st[key].get("gamma_mean"),
                    "gamma_std": st[key].get("gamma_std"),
                    "beta_mean": st[key].get("beta_mean"),
                    "beta_std": st[key].get("beta_std"),
                    "featurewise_var_mean": st[key].get("featurewise_var_mean"),
                })
        elif r.get("operator_kind") == "basis":
            for key in sorted(k for k in st if k.startswith("pair_")):
                usage_rows.append({
                    "dataset": r["dataset"], "variant": r["variant"],
                    "seed": r["seed"], "kind": "basis", "pair": key,
                    "q_mean": json.dumps(st[key].get("q_mean")),
                    "router_entropy_mean": st[key].get("router_entropy_mean"),
                    "effective_experts": st[key].get("effective_experts"),
                })
    if usage_rows:
        fields = ["dataset", "variant", "seed", "kind", "pair", "gamma_mean",
                  "gamma_std", "beta_mean", "beta_std", "featurewise_var_mean",
                  "q_mean", "router_entropy_mean", "effective_experts"]
        with (OPERATOR_ROOT / "operator_usage.csv").open("w", encoding="utf-8",
                                                         newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in usage_rows:
                w.writerow(r)

    from src.analysis.perf_r2d28_utils import mean_std_pp

    lines = ["# R2D28_OPERATOR_REPORT — D2.8-E functional operator capacity",
             "",
             "Rule IV + Rule V: E*/C*/M* loaded and frozen; primary diagnostic",
             "uses NormMatch; zero/small-init so step 0 == O0.",
             ""]
    for cand, base in (("O1", "O0"), ("O2", "O0"), ("O3", "O0"), ("O4", "O0")):
        lines.append(
            f"- {cand} - O0: Acc {mean_std_pp(rows, cand, 'O0', 'best_val_acc')} pp; "
            f"F1 {mean_std_pp(rows, cand, 'O0', 'best_val_macro_f1')} pp")
    for cand, base in (("O2", "O1"), ("O3", "O1"), ("O4", "O1")):
        lines.append(
            f"- {cand} - O1 (static): Acc "
            f"{mean_std_pp(rows, cand, 'O1', 'best_val_acc')} pp; F1 "
            f"{mean_std_pp(rows, cand, 'O1', 'best_val_macro_f1')} pp")
    for cand, base in (("O3", "O2"), ("O4", "O4_TARGET")):
        lines.append(
            f"- {cand} - {base} (edge vs target conditioning): Acc "
            f"{mean_std_pp(rows, cand, base, 'best_val_acc')} pp; F1 "
            f"{mean_std_pp(rows, cand, base, 'best_val_macro_f1')} pp")
    lines.append(f"- O4 - O4_UNIFORM: Acc "
                 f"{mean_std_pp(rows, 'O4', 'O4_UNIFORM', 'best_val_acc')} pp; F1 "
                 f"{mean_std_pp(rows, 'O4', 'O4_UNIFORM', 'best_val_macro_f1')} pp")
    lines += [
        "",
        "O* selection (v2 §10 + Prompt 6): operator GO (>=+0.30pp Acc or",
        "+0.40pp F1 vs O0, other nonnegative) + dynamic-vs-static specificity",
        "(>=+0.20pp) + functional diversity + positive causal usage.",
        "",
        "(R2D28_OPERATOR_REPORT.md)",
    ]
    (OPERATOR_ROOT / "R2D28_OPERATOR_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")

    controls_rows = []
    for cand, base in (("O4", "O4_UNIFORM"), ("O4", "O4_TARGET")):
        controls_rows.append({
            "candidate": cand, "base": base, "metric": "acc",
            "delta_pp": mean_std_pp(rows, cand, base, "best_val_acc"),
        })
        controls_rows.append({
            "candidate": cand, "base": base, "metric": "f1",
            "delta_pp": mean_std_pp(rows, cand, base, "best_val_macro_f1"),
        })
    with (OPERATOR_ROOT / "operator_controls.csv").open("w", encoding="utf-8",
                                                        newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate", "base", "metric", "delta_pp"])
        w.writeheader()
        for r in controls_rows:
            w.writerow(r)
    print("[summarize] done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2D2.8 v2 D2.8-E operator")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--exposure-variant", default=DEFAULT_EXPOSURE)
    parser.add_argument("--composition-variant", default=DEFAULT_COMPOSITION)
    parser.add_argument("--channel-variant", default=DEFAULT_CHANNEL)
    parser.add_argument("--norm-match", dest="norm_match", action="store_true",
                        default=True, help="primary diagnostic (default on)")
    parser.add_argument("--unrestricted", dest="norm_match", action="store_false",
                        help="secondary unrestricted test (NormMatch off)")
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
    assert args.channel_variant in CHANNEL_KINDS, args.channel_variant

    if args.worker:
        run_worker(args.dataset, args.variant, args.seed, Path(args.outdir),
                   args.epochs, args.force, args.exposure_variant,
                   args.composition_variant, args.channel_variant, args.norm_match)
        return
    if args.summarize:
        summarize()
        return

    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    variants = list(OPERATOR_VARIANTS) if not args.variants else \
        [v for v in args.variants.split(",")]
    unknown = [v for v in variants if v not in OPERATOR_VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {unknown}")
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, v, s) for d in datasets for v in variants for s in seeds]
    launch_jobs(Path(__file__).resolve(), jobs, OPERATOR_ROOT, gpus, args.force,
                args.epochs, extra_flags=[
                    "--exposure-variant", args.exposure_variant,
                    "--composition-variant", args.composition_variant,
                    "--channel-variant", args.channel_variant,
                    "--norm-match" if args.norm_match else "--unrestricted"])


if __name__ == "__main__":
    main()
