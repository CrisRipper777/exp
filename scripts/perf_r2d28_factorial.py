"""R2-Design-2.8 v2 D2.8-F: controlled co-training + factorial matrix (v2 §12).

Only components that received SUPPORTED/STRONGLY_SUPPORTED status in B-E
enter E*/C*/M*/O*; a NO-GO component keeps its baseline version. The
factorial runs are the secondary INTEGRATED (joint co-training) evidence —
the frozen-attribution results from B-E remain the primary evidence
(Rule V).

    F0 E0+C0+M0+O0   (== stage-B E0; reused from the exposure CSV)
    F1 E*+C0+M0+O0
    F2 E0+C*+M0+O0
    F3 E0+C0+M*+O0
    F4 E0+C0+M0+O*
    F5 E*+C*+M0+O0
    F6 E*+C0+M0+O*
    F7 E*+C0+M*+O0
    F8 E0+C*+M0+O*
    F9 E*+C*+M*+O*

Synergy attribution (v2 §12):
    Synergy(E,C) = F5 - max(F1,F2);  Synergy(E,O) = F6 - max(F1,F4)
    Synergy(E,M) = F7 - max(F1,F3);  Synergy(C,O) = F8 - max(F2,F4)
    Synergy_full = F9 - max(F5,F6,F7,F8)
F9 being highest does NOT certify all modules (v2 §12).

Movies/Toys/Grocery x seeds 42/43/44; promising combinations then run
ele-fashion/Reddit-S guards (--guards flag).

Outputs: outputs/perf_r2d28/factorial/
    factorial_results.csv, factorial_attribution.csv, R2D28_FACTORIAL_REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
    load_a0_parent,
    load_or_make_head_init,
)
from src.analysis.perf_r2d28_utils import (  # noqa: E402
    EXPOSURE_ROOT,
    HEAD_INIT_ROOT,
    R2D28_ROOT,
    build_model,
    launch_jobs,
    resolve_cfg,
    train_relfunc_model,
)

FACTORIAL_ROOT = R2D28_ROOT / "factorial"

EXPOSURE_KINDS = {"E0": "fixed_full", "E1": "node", "E2": "target",
                  "E3": "source", "E4": "pair"}
COMPOSITION_KINDS = {"C0": "uniform", "C1": "generic", "C2": "target",
                     "C3": "source", "C4": "pair"}
CHANNEL_KINDS = {"M0": "mean", "M1": "softmax", "M2": "concat", "M3": "attn"}
OPERATOR_KINDS = {"O0": "linear", "O1": "static_pair", "O2": "target_film",
                  "O3": "edge_film", "O4": "basis", "O4_UNIFORM": "basis",
                  "O4_TARGET": "basis"}
OPERATOR_EXTRA = {"O0": {}, "O1": {}, "O2": {}, "O3": {}, "O4": {},
                  "O4_UNIFORM": {"uniform_router": True},
                  "O4_TARGET": {"target_router": True}}


def factorial_spec(e_star: str, c_star: str, m_star: str, o_star: str) -> dict:
    """variant -> (exposure, composition, channel, operator, extra overrides)."""
    extra = OPERATOR_EXTRA[o_star]
    return {
        "F0": ("E0", "C0", "M0", "O0", {}),
        "F1": (e_star, "C0", "M0", "O0", {}),
        "F2": ("E0", c_star, "M0", "O0", {}),
        "F3": ("E0", "C0", m_star, "O0", {}),
        "F4": ("E0", "C0", "M0", o_star, extra),
        "F5": (e_star, c_star, "M0", "O0", {}),
        "F6": (e_star, "C0", "M0", o_star, extra),
        "F7": (e_star, "C0", m_star, "O0", {}),
        "F8": ("E0", c_star, "M0", o_star, extra),
        "F9": (e_star, c_star, m_star, o_star, extra),
    }


def run_worker(dataset: str, variant: str, seed: int, outdir: Path,
               epochs: int | None, force: bool, e_star: str, c_star: str,
               m_star: str, o_star: str, norm_match: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} {variant} s{seed}] SKIP", flush=True)
        return
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    setup = load_a0_parent(dataset, seed, device)
    data = setup.data
    e, c, m, o, extra = factorial_spec(e_star, c_star, m_star, o_star)[variant]
    overrides = {
        "exposure": EXPOSURE_KINDS[e], "composition": COMPOSITION_KINDS[c],
        "channel": CHANNEL_KINDS[m], "operator": OPERATOR_KINDS[o],
        "norm_match": bool(norm_match), **extra,
    }
    cfg = resolve_cfg(dataset, seed, overrides)
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

    torch.save({"head_state": head.state_dict(), "model_state": model.state_dict()},
               outdir / "best.pt")
    summary = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "exposure_kind": model.exposure_kind,
        "composition_kind": model.composition_kind,
        "channel_kind": model.channel_kind,
        "operator_kind": model.operator_kind,
        "norm_match": bool(model.norm_match),
        "best_val_acc": res["best_val_acc"],
        "best_val_macro_f1": res["best_val_macro_f1"],
        "per_class_f1": res["per_class_f1"],
        "best_epoch": res["best_epoch"], "stop_epoch": res["stop_epoch"],
        "side_params": int(model.side_parameter_count),
        "trainable_params": int(res["trainable_params"]),
        "out_dim": int(model.out_dim),
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
    for p in sorted(FACTORIAL_ROOT.rglob("summary.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    # F0 == stage-B E0 (identical config and protocol): reuse
    with (EXPOSURE_ROOT / "exposure_results.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] == "E0" and r["reference"] == "False":
                rows.append({
                    "dataset": r["dataset"], "variant": "F0", "seed": int(r["seed"]),
                    "best_val_acc": float(r["val_acc"]),
                    "best_val_macro_f1": float(r["val_macro_f1"]),
                    "best_epoch": int(r["best_epoch"]),
                    "side_params": int(r["side_params"]),
                    "reused_from_exposure_E0": True,
                })

    with (FACTORIAL_ROOT / "factorial_results.csv").open("w", encoding="utf-8",
                                                         newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "variant", "seed", "val_acc", "val_macro_f1",
            "best_epoch", "side_params", "trainable_params",
            "runtime_sec", "peak_allocated_mb"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "dataset": r["dataset"], "variant": r["variant"],
                "seed": r["seed"], "val_acc": r["best_val_acc"],
                "val_macro_f1": r["best_val_macro_f1"],
                "best_epoch": r.get("best_epoch"),
                "side_params": r.get("side_params"),
                "trainable_params": r.get("trainable_params"),
                "runtime_sec": r.get("runtime_sec"),
                "peak_allocated_mb": r.get("peak_allocated_mb"),
            })

    def _mean(variant, ds_list):
        vals = [r["best_val_acc"] for r in rows
                if r["variant"] == variant and r["dataset"] in ds_list]
        return statistics.fmean(vals) if vals else None

    def _d(variant, ds_list, metric="best_val_acc"):
        vals = [r[metric] for r in rows
                if r["variant"] == variant and r["dataset"] in ds_list]
        return statistics.fmean(vals) if vals else None

    ds_list = TARGET_DATASETS
    pairs = [("F1", "F0"), ("F2", "F0"), ("F3", "F0"), ("F4", "F0"),
             ("F5", "F0"), ("F6", "F0"), ("F7", "F0"), ("F8", "F0"), ("F9", "F0")]
    attrib = []
    for v in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"):
        m_acc = _d(v, ds_list, "best_val_acc")
        m_f1 = _d(v, ds_list, "best_val_macro_f1")
        b_acc = _d("F0", ds_list, "best_val_acc")
        b_f1 = _d("F0", ds_list, "best_val_macro_f1")
        if m_acc is not None and b_acc is not None:
            attrib.append({"variant": v, "delta_acc_pp": 100 * (m_acc - b_acc),
                           "delta_f1_pp": 100 * (m_f1 - b_f1)})
    # synergies
    def _syn(v, others):
        m = _d(v, ds_list)
        base = max((_d(o, ds_list) for o in others if _d(o, ds_list) is not None),
                   default=None)
        return None if (m is None or base is None) else 100 * (m - base)

    synergy = {
        "E_x_C": _syn("F5", ["F1", "F2"]),
        "E_x_O": _syn("F6", ["F1", "F4"]),
        "E_x_M": _syn("F7", ["F1", "F3"]),
        "C_x_O": _syn("F8", ["F2", "F4"]),
        "full": _syn("F9", ["F5", "F6", "F7", "F8"]),
    }
    with (FACTORIAL_ROOT / "factorial_attribution.csv").open(
            "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "delta_acc_pp", "delta_f1_pp"])
        w.writeheader()
        for r in attrib:
            w.writerow(r)
        for k, v in synergy.items():
            w.writerow({"variant": f"synergy_{k}", "delta_acc_pp": v,
                        "delta_f1_pp": None})

    lines = ["# R2D28_FACTORIAL_REPORT — D2.8-F co-training factorial",
             "",
             "Integrated (joint co-training) evidence only; the frozen-",
             "attribution results of B-E remain the primary mechanism",
             "evidence (Rule V).",
             "",
             "| variant | delta Acc (pp) vs F0 | delta F1 (pp) vs F0 |",
             "|---|---|---|"]
    for r in attrib:
        lines.append(f"| {r['variant']} | {r['delta_acc_pp']:+.3f} "
                     f"| {r['delta_f1_pp']:+.3f} |")
    lines += [
        "",
        "| synergy | Acc (pp) |",
        "|---|---|",
    ]
    for k, v in synergy.items():
        lines.append(f"| {k} | {v if v is None else f'{v:+.3f}'} |")
    lines += [
        "",
        "F9 being highest does NOT certify all modules (v2 §12): a component",
        "counts only with its independent B-E support.",
        "",
        "(R2D28_FACTORIAL_REPORT.md)",
    ]
    (FACTORIAL_ROOT / "R2D28_FACTORIAL_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("[summarize] done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2D2.8 v2 D2.8-F factorial")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--e-star", default="E4")
    parser.add_argument("--c-star", default="C0")
    parser.add_argument("--m-star", default="M0")
    parser.add_argument("--o-star", default="O0")
    parser.add_argument("--norm-match", dest="norm_match", action="store_true",
                        default=True)
    parser.add_argument("--unrestricted", dest="norm_match", action="store_false")
    parser.add_argument("--guards", action="store_true",
                        help="run ele-fashion/Reddit-S guards for promising variants")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    assert args.e_star in EXPOSURE_KINDS, args.e_star
    assert args.c_star in COMPOSITION_KINDS, args.c_star
    assert args.m_star in CHANNEL_KINDS, args.m_star
    assert args.o_star in OPERATOR_KINDS, args.o_star

    if args.worker:
        run_worker(args.dataset, args.variant, args.seed, Path(args.outdir),
                   args.epochs, args.force, args.e_star, args.c_star,
                   args.m_star, args.o_star, args.norm_match)
        return
    if args.summarize:
        summarize()
        return

    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    variants = list(factorial_spec(args.e_star, args.c_star, args.m_star,
                                   args.o_star)) if not args.variants else \
        [v for v in args.variants.split(",")]
    # F0 is reused from stage B — never trained here
    variants = [v for v in variants if v != "F0"]
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, v, s) for d in datasets for v in variants for s in seeds]
    launch_jobs(Path(__file__).resolve(), jobs, FACTORIAL_ROOT, gpus, args.force,
                args.epochs, extra_flags=[
                    "--e-star", args.e_star, "--c-star", args.c_star,
                    "--m-star", args.m_star, "--o-star", args.o_star,
                    "--norm-match" if args.norm_match else "--unrestricted"])


if __name__ == "__main__":
    main()
