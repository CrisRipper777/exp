"""R2-Design-2.8 v2 D2.8-G: final strong-parent confirmation (v2 §13).

Runs the final scientifically-supported candidate (joint co-training =
F9 configuration) on all five datasets x seeds 42/43/44, and compares
against reference rows:
    A0_MATCHED                (D2.7 matrix A0_BASE)
    A0_FORMAL                 (perf_r1 baseline A0)
    PAIR_EDGE_D27             (D2.7 matrix PAIR_EDGE)
    TARGET_FACTOR_ONLY_D27    (D2.7 ownership TARGET_FACTOR_ONLY)
    best exposure-only model  (D2.8-B E*)

M/T/G rows already produced by D2.8-F (F9) are reused when present; only
ele-fashion/Reddit-S need new runs in that case.

Incremental GO: Candidate - A0_MATCHED >= +0.40pp Acc AND +0.30pp F1 (M/T/G).
Formal GO:     Candidate - A0_FORMAL  >= +0.20pp Acc, F1 nonnegative.
Guards:        ele/Reddit Acc >= A0_FORMAL - 0.20pp; F1 >= A0_FORMAL - 0.50pp.
No Test.

Optional single controlled parent adaptation (v2 §13) if the candidate is
A0_MATCHED +0.20pp but below FORMAL GO: epochs 1-30 parent frozen,
epoch 31+ unfreeze A0 final fusion + graph operator, parent lr 1e-4,
new branch lr 1e-3, P0 factorizer frozen.

Outputs: outputs/perf_r2d28/confirm/
    confirm_results.csv, confirm_resources.csv, R2D28_CONFIRM_REPORT.md
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
    R2D27_ROOT,
    load_a0_parent,
    load_or_make_head_init,
)
from src.analysis.perf_r2d28_utils import (  # noqa: E402
    EXPOSURE_ROOT,
    FACTORIAL_ROOT,
    HEAD_INIT_ROOT,
    R2D28_ROOT,
    build_model,
    launch_jobs,
    resolve_cfg,
    train_relfunc_model,
)

CONFIRM_ROOT = R2D28_ROOT / "confirm"

EXPOSURE_KINDS = {"E0": "fixed_full", "E1": "node", "E2": "target",
                  "E3": "source", "E4": "pair"}
COMPOSITION_KINDS = {"C0": "uniform", "C1": "generic", "C2": "target",
                     "C3": "source", "C4": "pair"}
CHANNEL_KINDS = {"M0": "mean", "M1": "softmax", "M2": "concat", "M3": "attn"}
OPERATOR_KINDS = {"O0": "linear", "O1": "static_pair", "O2": "target_film",
                  "O3": "edge_film", "O4": "basis"}
OPERATOR_EXTRA = {"O0": {}, "O1": {}, "O2": {}, "O3": {}, "O4": {},
                  "O4_UNIFORM": {"uniform_router": True},
                  "O4_TARGET": {"target_router": True}}


def run_worker(dataset: str, variant: str, seed: int, outdir: Path,
               epochs: int | None, force: bool, e_star: str, c_star: str,
               m_star: str, o_star: str, norm_match: bool,
               adapt: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} {variant} s{seed}] SKIP", flush=True)
        return
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    setup = load_a0_parent(dataset, seed, device)
    data = setup.data
    overrides = {
        "exposure": EXPOSURE_KINDS[e_star], "composition": COMPOSITION_KINDS[c_star],
        "channel": CHANNEL_KINDS[m_star], "operator": OPERATOR_KINDS[o_star],
        "norm_match": bool(norm_match), **OPERATOR_EXTRA[o_star],
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
    if adapt:
        from src.analysis.perf_r2d28_utils import train_relfunc_parent_adapt

        res = train_relfunc_parent_adapt(
            data, model, head, device, total_epochs=total_epochs,
            history_callback=history_writer.writerow)
    else:
        res = train_relfunc_model(
            data, model, head, device, total_epochs=total_epochs,
            history_callback=history_writer.writerow)
    history_file.close()

    ckpt = {"head_state": head.state_dict(), "model_state": model.state_dict()}
    if adapt:
        ckpt["parent_state"] = {k: v.detach().cpu() for k, v
                                in setup.parent.state_dict().items()}
    torch.save(ckpt, outdir / "best.pt")
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


def load_references(e_star: str) -> list[dict]:
    """A0_MATCHED / PAIR_EDGE_D27 / TARGET_FACTOR_ONLY_D27 (D2.7 CSVs),
    A0_FORMAL (R1 baseline), best exposure-only (D2.8-B)."""
    from src.analysis.perf_r2_utils import load_a0_reference

    rows = []
    matrix_csv = R2D27_ROOT / "matrix" / "matrix_results.csv"
    with matrix_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] in ("A0_BASE", "PAIR_EDGE"):
                name = "A0_MATCHED" if r["variant"] == "A0_BASE" else "PAIR_EDGE_D27"
                rows.append({"dataset": r["dataset"], "variant": name,
                             "seed": int(r["seed"]),
                             "best_val_acc": float(r["val_acc"]),
                             "best_val_macro_f1": float(r["val_macro_f1"])})
    ownership_csv = R2D27_ROOT / "ownership" / "ownership_results.csv"
    with ownership_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] == "TARGET_FACTOR_ONLY":
                rows.append({"dataset": r["dataset"],
                             "variant": "TARGET_FACTOR_ONLY_D27",
                             "seed": int(r["seed"]),
                             "best_val_acc": float(r["val_acc"]),
                             "best_val_macro_f1": float(r["val_macro_f1"])})
    a0_formal = load_a0_reference()  # {(ds, seed): acc}
    for ds in DATASETS:
        for seed in (42, 43, 44):
            p = (PROJECT_ROOT / "outputs" / "perf_r1" / "baseline" / ds / "A0"
                 / f"seed_{seed}" / "summary.json")
            f1 = None
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                if d.get("best_val_macro_f1") is not None:
                    f1 = float(d["best_val_macro_f1"]) / 100.0
            rows.append({"dataset": ds, "variant": "A0_FORMAL", "seed": seed,
                         "best_val_acc": a0_formal[(ds, seed)], "best_val_macro_f1": f1})
    # best exposure-only model (D2.8-B)
    with (EXPOSURE_ROOT / "exposure_results.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] == e_star and r["reference"] == "False":
                rows.append({"dataset": r["dataset"],
                             "variant": f"{e_star}_EXPOSURE_ONLY",
                             "seed": int(r["seed"]),
                             "best_val_acc": float(r["val_acc"]),
                             "best_val_macro_f1": float(r["val_macro_f1"])})
    return rows


def summarize(e_star: str) -> None:
    candidate = "FINAL"
    rows = []
    for p in sorted(CONFIRM_ROOT.rglob("summary.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    # reuse F9 M/T/G rows from D2.8-F when the confirm run skipped them
    if FACTORIAL_ROOT.exists():
        for p in sorted(FACTORIAL_ROOT.rglob("summary.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            if d["variant"] in ("F9", "F3"):
                d["variant"] = candidate
                rows.append(d)
    rows += load_references(e_star)

    with (CONFIRM_ROOT / "confirm_results.csv").open("w", encoding="utf-8",
                                                     newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed",
                                          "val_acc", "val_macro_f1"])
        w.writeheader()
        for r in rows:
            w.writerow({"dataset": r["dataset"], "variant": r["variant"],
                        "seed": r["seed"], "val_acc": r.get("best_val_acc"),
                        "val_macro_f1": r.get("best_val_macro_f1")})

    res_rows = []
    for p in sorted(CONFIRM_ROOT.rglob("summary.json")):
        res_rows.append(json.loads(p.read_text(encoding="utf-8")))
    if FACTORIAL_ROOT.exists():
        for p in sorted(FACTORIAL_ROOT.rglob("summary.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            if d["variant"] == "F9" and d["dataset"] in ("Movies", "Toys", "Grocery"):
                res_rows.append(d)
    if res_rows:
        with (CONFIRM_ROOT / "confirm_resources.csv").open(
                "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "dataset", "seed", "side_params", "trainable_params",
                "runtime_sec", "peak_allocated_mb", "best_epoch", "stop_epoch"])
            w.writeheader()
            for r in res_rows:
                w.writerow({
                    "dataset": r["dataset"], "seed": r["seed"],
                    "side_params": r.get("side_params"),
                    "trainable_params": r.get("trainable_params"),
                    "runtime_sec": r.get("runtime_sec"),
                    "peak_allocated_mb": r.get("peak_allocated_mb"),
                    "best_epoch": r.get("best_epoch"),
                    "stop_epoch": r.get("stop_epoch"),
                })

    def _paired(cand, base, metric, ds_list=("Movies", "Toys", "Grocery")):
        cand_r = {(r["dataset"], r["seed"]): r for r in rows
                  if r["variant"] == cand}
        base_r = {(r["dataset"], r["seed"]): r for r in rows
                  if r["variant"] == base}
        deltas, per_ds = [], []
        for ds in ds_list:
            dv = []
            for s in (42, 43, 44):
                c, b = cand_r.get((ds, s)), base_r.get((ds, s))
                if c and b and c.get(metric) is not None and b.get(metric) is not None:
                    dv.append(100 * (c[metric] - b[metric]))
            if dv:
                per_ds.append((ds, statistics.fmean(dv), sum(1 for x in dv if x > 0), len(dv)))
                deltas += dv
        mean = statistics.fmean(deltas) if deltas else None
        n_pos = sum(1 for _, m, _, _ in per_ds if m > 0)
        return {"mean": mean, "per_ds": per_ds, "n_pos": n_pos}

    inc_acc = _paired(candidate, "A0_MATCHED", "best_val_acc")
    inc_f1 = _paired(candidate, "A0_MATCHED", "best_val_macro_f1")
    for_acc = _paired(candidate, "A0_FORMAL", "best_val_acc")
    for_f1 = _paired(candidate, "A0_FORMAL", "best_val_macro_f1")
    inc_go = (inc_acc["mean"] is not None and inc_acc["mean"] >= 0.40
              and inc_f1["mean"] is not None and inc_f1["mean"] >= 0.30)
    for_go = (for_acc["mean"] is not None and for_acc["mean"] >= 0.20
              and (for_f1["mean"] is None or for_f1["mean"] >= 0.0))
    lines = [
        "# R2D28_CONFIRM_REPORT — D2.8-G final confirmation",
        "",
        f"- Candidate - A0_MATCHED (M/T/G): Acc {inc_acc['mean'] if inc_acc['mean'] is None else f'{inc_acc['mean']:+.3f}'}pp; "
        f"F1 {inc_f1['mean'] if inc_f1['mean'] is None else f'{inc_f1['mean']:+.3f}'}pp → "
        f"INCREMENTAL GO: {'PASS' if inc_go else 'FAIL'}",
        f"- Candidate - A0_FORMAL (M/T/G): Acc {for_acc['mean'] if for_acc['mean'] is None else f'{for_acc['mean']:+.3f}'}pp; "
        f"F1 {for_f1['mean'] if for_f1['mean'] is None else f'{for_f1['mean']:+.3f}'}pp → "
        f"FORMAL GO: {'PASS' if for_go else 'FAIL'}",
    ]
    # controlled parent adaptation (v2 §13), if run
    adapt_acc = _paired("ADAPT", "A0_MATCHED", "best_val_acc")
    adapt_f1 = _paired("ADAPT", "A0_MATCHED", "best_val_macro_f1")
    adapt_for = _paired("ADAPT", "A0_FORMAL", "best_val_acc")
    if adapt_acc["mean"] is not None:
        lines += [
            f"- ADAPT - A0_MATCHED (M/T/G): Acc "
            f"{adapt_acc['mean']:+.3f}pp; F1 {adapt_f1['mean']:+.3f}pp",
            f"- ADAPT - A0_FORMAL (M/T/G): Acc {adapt_for['mean']:+.3f}pp",
        ]
    lines += [
        "",
        "Guard verdicts vs A0_FORMAL are computed in the final synthesis"
        " (D2.8-H).",
        "",
        "(R2D28_CONFIRM_REPORT.md)",
    ]
    (CONFIRM_ROOT / "R2D28_CONFIRM_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"[summarize] INCREMENTAL {'PASS' if inc_go else 'FAIL'} / "
          f"FORMAL {'PASS' if for_go else 'FAIL'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2D2.8 v2 D2.8-G confirm")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
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
    parser.add_argument("--adapt", dest="adapt", action="store_true", default=False,
                        help="controlled parent adaptation (v2 §13)")
    parser.add_argument("--no-adapt", dest="adapt", action="store_false")
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
                   args.m_star, args.o_star, args.norm_match, args.adapt)
        return
    if args.summarize:
        summarize(args.e_star)
        return

    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    variant = "ADAPT" if args.adapt else "FINAL"
    jobs = [(d, variant, s) for d in datasets for s in seeds]
    launch_jobs(Path(__file__).resolve(), jobs, CONFIRM_ROOT, gpus, args.force,
                args.epochs, extra_flags=[
                    "--e-star", args.e_star, "--c-star", args.c_star,
                    "--m-star", args.m_star, "--o-star", args.o_star,
                    "--norm-match" if args.norm_match else "--unrestricted",
                    "--adapt" if args.adapt else "--no-adapt"])


if __name__ == "__main__":
    main()
