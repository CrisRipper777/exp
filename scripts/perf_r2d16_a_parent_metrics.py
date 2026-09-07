"""R2-Design-1.6 D1.6-A: A0 metric backfill + parent characterization
(plan §8-§10) + optional strict A0 graph-control diagnostic (plan §11).

NO retraining. Val only.
  - A0 Val Macro-F1: parsed from the FORMAL train.log per-epoch lines
    (best-Val-Acc epoch; the formal runs saved no history.csv / model.pt).
  - A0 per-class F1 / confusion: NOT recoverable from the formal runs ->
    "unavailable"; additionally re-inferred from the R1-A0 checkpoint
    (DISCLOSED provenance, audit §B).
  - B0: summary.json (best acc/F1) + checkpoint re-inference for per-class.
  - Optional graph-control (audit §F says FEASIBLE): A0 full / local-only /
    relation-neutralized (CF1_uniform) via the R1-sanctioned apply_plan —
    strict masks on the trained plan, never approximate.

Usage:
    python scripts/perf_r2d16_a_parent_metrics.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r0_utils import write_csv  # noqa: E402
from src.analysis.perf_r2d16_utils import (  # noqa: E402
    DATASETS,
    R2D16_ROOT,
    SEEDS,
    load_parent_setup,
    assert_no_test_access,
)
from src.analysis.perf_r2d15_utils import val_metrics_with_head  # noqa: E402

A0_FORMAL_ROOT = PROJECT_ROOT / "outputs" / "final_nc_benchmark" / "main"
OUTDIR = R2D16_ROOT / "parent_metrics"


def _parse_a0_log(dataset: str, seed: int) -> dict:
    """Best-Val-Acc epoch bookkeeping from the FORMAL A0 train.log."""
    log = A0_FORMAL_ROOT / dataset / "biaxis_final" / f"seed_{seed}" / "train.log"
    out = {"best_val_acc": None, "best_val_macro_f1": None, "best_epoch": None,
           "stop_epoch": None}
    if not log.exists():
        return out
    best_acc = -1.0
    for line in log.read_text(encoding="utf-8").splitlines():
        m = re.search(r"Epoch (\d+)", line)
        if m:
            out["stop_epoch"] = max(out["stop_epoch"] or 0, int(m.group(1)))
        m = re.search(r"Val Acc ([\d.]+) \| Val F1 ([\d.]+)", line)
        if m:
            acc = float(m.group(1)) / 100.0  # the log prints percentages
            if acc > best_acc:
                best_acc = acc
                out["best_val_acc"] = acc
                out["best_val_macro_f1"] = float(m.group(2)) / 100.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="D1.6-A parent metric backfill")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-graph-control", action="store_true")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    perclass_rows: list[dict] = []
    cf_rows: list[dict] = []

    for ds in DATASETS:
        for seed in SEEDS:
            # ---- A0: formal log (acc/F1) ----
            a0_log = _parse_a0_log(ds, seed)
            # ---- B0: summary ----
            b0_path = PROJECT_ROOT / "outputs" / "perf_r2d15" / "b0_confirm" / ds / "B0" / f"seed_{seed}" / "summary.json"
            b0 = json.load(open(b0_path))
            gap_b0 = None
            if b0.get("train_acc_at_best") is not None:
                gap_b0 = b0["train_acc_at_best"] - b0["best_val_acc"]
            rows.append({
                "dataset": ds, "seed": seed,
                "A0_val_acc": a0_log["best_val_acc"],
                "A0_val_macro_f1": a0_log["best_val_macro_f1"],
                "B0_val_acc": b0["best_val_acc"],
                "B0_val_macro_f1": b0["best_val_macro_f1"],
                "B0_minus_A0_acc": b0["best_val_acc"] - a0_log["best_val_acc"]
                if a0_log["best_val_acc"] is not None else None,
                "B0_minus_A0_f1": b0["best_val_macro_f1"] - a0_log["best_val_macro_f1"]
                if a0_log["best_val_macro_f1"] is not None else None,
                "A0_best_epoch": a0_log["best_epoch"],
                "B0_best_epoch": b0.get("best_epoch"),
                "A0_stop_epoch": a0_log["stop_epoch"],
                "B0_stop_epoch": b0.get("stop_epoch"),
                "B0_train_val_gap": gap_b0,
            })
            # ---- per-class via checkpoint re-inference ----
            for parent in ("A0", "B0"):
                setup = load_parent_setup(parent, ds, seed, device)
                assert_no_test_access(setup.data)
                x = setup.data.x.to(device)
                ei = setup.data.edge_index.to(device)
                from src.analysis.perf_r2d16_utils import extract_parent_states
                states = extract_parent_states(setup, x, ei)
                m = val_metrics_with_head(setup.head, states["z"], setup.data, device)
                perclass_rows.append({
                    "dataset": ds, "seed": seed, "parent": parent,
                    "re_inferred_val_acc": m["val_acc"],
                    "re_inferred_val_macro_f1": m["val_macro_f1"],
                    "per_class_f1": m["per_class_f1"],
                    "confusion": m["confusion"],
                })
                print(f"[perclass] {ds} s{seed} {parent}: re-inf acc={m['val_acc']:.5f} "
                      f"f1={m['val_macro_f1']:.5f}", flush=True)
                del setup, x, ei, states
                torch.cuda.empty_cache()

    write_csv(OUTDIR / "parent_metrics.csv", rows)
    write_csv(OUTDIR / "parent_perclass.csv", perclass_rows)

    # ---- optional strict A0 graph-control (plan §11) ----------------------
    if not args.skip_graph_control:
        from src.analysis.perf_r1_utils import apply_plan, r1_pipeline, val_acc_with_head
        from src.analysis.perf_r1_utils import _cf1_uniform  # relation-neutralized
        for ds in DATASETS:
            seed = 42
            setup = load_parent_setup("A0", ds, seed, device)
            internals = r1_pipeline(setup)
            gamma = internals["graph_out"]["gamma"]
            beta = internals["beta"]
            alpha = internals["alpha"]
            variants = {
                "A0_full": None,
                "A0_local_only": None,
                "A0_relation_neutralized": _cf1_uniform(gamma, beta, alpha, None),
            }
            local_plan = torch.zeros_like(gamma)
            local_plan[..., 0] = 1.0
            isolated = internals["deg"] <= 0
            local_gamma = torch.where(isolated[:, None, None], gamma, local_plan)
            variants["A0_local_only"] = local_gamma
            for name, g in variants.items():
                z = internals["z_final"] if g is None else apply_plan(setup, internals, g)
                acc = val_acc_with_head(setup, z)
                f1 = None
                if name != "A0_full":
                    m = val_metrics_with_head(setup.head, z, setup.data, device)
                    f1 = m["val_macro_f1"]
                else:
                    m = val_metrics_with_head(setup.head, z, setup.data, device)
                    f1 = m["val_macro_f1"]
                cf_rows.append({
                    "dataset": ds, "seed": seed, "counterfactual": name,
                    "val_acc": acc, "val_macro_f1": f1,
                })
                print(f"[graph-control] {ds} s{seed} {name}: acc={acc:.5f} f1={f1:.5f}", flush=True)
                if g is not None:
                    del z, g
                    torch.cuda.empty_cache()
            del internals, setup
            torch.cuda.empty_cache()
        write_csv(OUTDIR / "parent_graph_control.csv", cf_rows)

    # ---- report ----
    lines = [
        "# R2D16_PARENT_REPORT — D1.6-A Parent Metric Backfill",
        "",
        "> No retraining. Val only. A0 Val Macro-F1 parsed from the FORMAL "
        "train.log (best-Val-Acc epoch); A0 per-class re-inferred from the "
        "R1-A0 checkpoint (provenance disclosed, audit §B) — formal-run "
        "per-class is unavailable (no checkpoint).",
        "",
        "## Parent characterization (best-epoch Val metrics)",
        "",
        "| dataset | seed | A0 Acc | A0 F1 | B0 Acc | B0 F1 | B0−A0 Acc (pp) | B0−A0 F1 (pp) | B0 ep | B0 gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['A0_val_acc']:.4f} | "
            f"{row['A0_val_macro_f1']:.4f} | {row['B0_val_acc']:.4f} | "
            f"{row['B0_val_macro_f1']:.4f} | "
            f"{row['B0_minus_A0_acc']*100:+.3f} | {row['B0_minus_A0_f1']*100:+.3f} | "
            f"{row['B0_best_epoch']} | {row['B0_train_val_gap']:.4f} |"
        )
    lines += [
        "",
        "## 3-seed summary",
        "",
        "| dataset | B0−A0 Acc mean (pp) | B0−A0 F1 mean (pp) | B0−A0 Acc pos seeds |",
        "|---|---:|---:|---:|",
    ]
    for ds in DATASETS:
        sub = [r for r in rows if r["dataset"] == ds]
        acc_d = statistics.mean(r["B0_minus_A0_acc"] for r in sub)
        f1_d = statistics.mean(r["B0_minus_A0_f1"] for r in sub)
        pos = sum(1 for r in sub if r["B0_minus_A0_acc"] > 0)
        lines.append(f"| {ds} | {acc_d*100:+.3f} | {f1_d*100:+.3f} | {pos}/3 |")
    lines += [
        "",
        "## A0 graph-control (seed42, strict masks via apply_plan; plan §11)",
        "",
        "| dataset | A0_full | A0_local_only | A0_relation_neutralized |",
        "|---|---:|---:|---:|",
    ]
    if cf_rows:
        for ds in DATASETS:
            sub = {r["counterfactual"]: r for r in cf_rows if r["dataset"] == ds}
            lines.append(
                f"| {ds} | {sub['A0_full']['val_acc']:.4f} "
                f"| {sub['A0_local_only']['val_acc']:.4f} "
                f"| {sub['A0_relation_neutralized']['val_acc']:.4f} |"
            )
    else:
        lines.append("(graph-control skipped)")
    lines += [
        "",
        "## Frozen roles (plan §4)",
        "",
        "- **Parent-P = A0** (performance parent): formal reference for whether a "
        "new mechanism adds value on the full stable model.",
        "- **Parent-C = B0** (clean diagnostic scaffold): approximately A0-level "
        "M/T/G performance WITHOUT the K/Gamma/OFR machinery; answers whether a "
        "mechanism needs the old graph organization.",
        "- No A0-vs-B0 winner gate is re-run here (plan §52).",
    ]
    (OUTDIR / "R2D16_PARENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[parent_metrics] saved -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
