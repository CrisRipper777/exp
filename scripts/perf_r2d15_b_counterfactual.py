"""R2-Design-1.5 D1.5-B: F/S trained-checkpoint counterfactual failure
decomposition + optimization diagnosis (plan §6-§10).

NO retraining. Reads the existing B0/F/S seed42 best checkpoints and runs:
    F: full / func_off / diag_only / offdiag_only / src_C / src_Pt / src_Pv
    S: full / common_only / fixed_common_residual / both_off
(base B0 diagonal path always kept), then computes:

    E_forward   = Acc(full) - Acc(branch_off)
    G_coadapt   = Acc(branch_off) - Acc(B0)
    E_offdiag   = Acc(offdiag_only) - Acc(func_off)
    (S) E_common / E_residual analogous

plus CE-only gradient diagnostics (full vs branch-off, per shared group),
parameter drift vs B0, representation drift vs B0 (Val nodes), per-class F1
and confusion matrices. Post-hoc masking = distribution shift — diagnostic
counterfactuals, NOT retrained causal effects (plan §8.1). Val only.

Usage:
    python scripts/perf_r2d15_b_counterfactual.py --gpu 1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d15_utils import (  # noqa: E402
    FUNC_CFS,
    R2D15_ROOT,
    R2D1_ROOT,
    SEM_CFS,
    TARGET_DATASETS,
    ce_only_gradient_pair,
    forward_cf,
    load_frozen_r2_checkpoint,
    param_drift,
    representation_drift,
    val_metrics_with_head,
)
from src.analysis.perf_r0_utils import write_csv  # noqa: E402

OUTDIR = R2D15_ROOT / "counterfactual"


def _eval_cf(setup, x, ei, sem_cf, func_cf) -> dict:
    z, _ = forward_cf(setup.model, x, ei, sem_cf=sem_cf, func_cf=func_cf)
    metrics = val_metrics_with_head(setup.head, z, setup.data, setup.device)
    del z
    torch.cuda.empty_cache()
    return metrics


def _classify(ds: str, e_forward: float, g_coadapt: float, extra: str = "") -> str:
    """Failure classification (plan §10). |G_coadapt| < 0.10pp counts as ~0."""
    if e_forward > 0 and g_coadapt < -0.20 / 100:
        return f"TYPE-D (optimization masking) {extra}"
    if e_forward < -0.20 / 100 and g_coadapt > -0.10 / 100:
        return f"TYPE-A (forward harm) {extra}"
    if g_coadapt < -0.20 / 100 and e_forward >= -0.20 / 100:
        return f"TYPE-B (co-adaptation harm) {extra}"
    if e_forward < -0.20 / 100 and g_coadapt < -0.20 / 100:
        return f"TYPE-C (both) {extra}"
    if g_coadapt < -0.10 / 100:
        return f"mild co-adaptation + mild forward ({e_forward*100:+.2f}/{g_coadapt*100:+.2f}pp) {extra}"
    return f"no dominant effect ({e_forward*100:+.2f}/{g_coadapt*100:+.2f}pp) {extra}"


def main() -> None:
    parser = argparse.ArgumentParser(description="D1.5-B F/S counterfactual decomposition")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--datasets", default=None)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    f_rows: list[dict] = []
    s_rows: list[dict] = []
    grad_rows: list[dict] = []
    drift_rows: list[dict] = []
    rep_rows: list[dict] = []
    perclass_rows: list[dict] = []
    classifications: dict[str, str] = {}

    datasets = TARGET_DATASETS
    if args.datasets:
        datasets = [d for d in TARGET_DATASETS if d in args.datasets.split(",")]

    for ds in datasets:
        print(f"== {ds} ==", flush=True)
        b0 = load_frozen_r2_checkpoint(ds, 42, "B0", device)
        x = b0.data.x.to(device)
        ei = b0.data.edge_index.to(device)
        val_idx = b0.data.val_idx.to(device)
        z_b0, _ = forward_cf(b0.model, x, ei, None, None)
        m_b0 = val_metrics_with_head(b0.head, z_b0, b0.data, device)

        # ---------------- F counterfactuals (plan §7) ----------------
        f_setup = load_frozen_r2_checkpoint(ds, 42, "F", device)
        f_metrics: dict[str, dict] = {}
        for cf in FUNC_CFS:
            func_cf = None if cf == "full" else cf
            f_metrics[cf] = _eval_cf(f_setup, x, ei, None, func_cf)
            print(f"  F {cf:14s} acc={f_metrics[cf]['val_acc']:.5f} "
                  f"f1={f_metrics[cf]['val_macro_f1']:.5f}", flush=True)

        # consistency: the recomputed full forward must match the training
        # history best val acc (same weights, same computation)
        import json as _json
        _summary = _json.load(open(
            R2D1_ROOT / "functional" / ds / "F" / "seed_42" / "summary.json"
        ))
        assert abs(f_metrics["full"]["val_acc"] - _summary["best_val_acc"]) < 1e-5, (
            f"F/{ds} recomputed {f_metrics['full']['val_acc']} != history {_summary['best_val_acc']}"
        )
        e_forward = f_metrics["full"]["val_acc"] - f_metrics["func_off"]["val_acc"]
        g_coadapt = f_metrics["func_off"]["val_acc"] - m_b0["val_acc"]
        e_offdiag = f_metrics["offdiag_only"]["val_acc"] - f_metrics["func_off"]["val_acc"]
        e_diag = f_metrics["diag_only"]["val_acc"] - f_metrics["func_off"]["val_acc"]
        for cf, metrics in f_metrics.items():
            f_rows.append({
                "dataset": ds, "variant": "F", "cf": cf,
                "val_acc": metrics["val_acc"], "val_macro_f1": metrics["val_macro_f1"],
            })
        f_rows.append({
            "dataset": ds, "variant": "B0", "cf": "reference",
            "val_acc": m_b0["val_acc"], "val_macro_f1": m_b0["val_macro_f1"],
        })
        classifications[f"F/{ds}"] = _classify(ds, e_forward, g_coadapt,
                                               f"(E_offdiag={e_offdiag*100:+.2f}pp, E_diag={e_diag*100:+.2f}pp)")

        # F gradient diagnostics: full vs func_off (CE only)
        grad = ce_only_gradient_pair(
            f_setup.model, f_setup.head, x, ei, f_setup.data, device,
            cf_a=(None, None), cf_b=(None, "func_off"),
        )
        for group, stats in grad.items():
            grad_rows.append({"dataset": ds, "variant": "F", "group": group, **stats})

        # F parameter drift vs B0
        drift_f = param_drift(b0.model, f_setup.model, b0.head, f_setup.head)
        for group, value in drift_f.items():
            drift_rows.append({"dataset": ds, "variant": "F", "group": group, "drift": value})

        # F representation drift: B0 z vs F func_off z (Val nodes)
        z_f_off, _ = forward_cf(f_setup.model, x, ei, None, "func_off")
        rep_f = representation_drift(z_b0, z_f_off, val_idx)
        rep_rows.append({"dataset": ds, "variant": "F", "cf": "func_off", **rep_f})
        del z_f_off
        torch.cuda.empty_cache()

        # per-class for the key F configs
        for tag, metrics in (("F_full", f_metrics["full"]), ("F_func_off", f_metrics["func_off"]), ("B0", m_b0)):
            perclass_rows.append({
                "dataset": ds, "config": tag,
                "per_class_f1": metrics["per_class_f1"],
                "confusion": metrics["confusion"],
            })
        del f_setup
        torch.cuda.empty_cache()

        # ---------------- S counterfactuals (plan §8) ----------------
        s_setup = load_frozen_r2_checkpoint(ds, 42, "S", device)
        s_metrics: dict[str, dict] = {}
        for cf in SEM_CFS:
            sem_cf = None if cf == "full" else cf
            s_metrics[cf] = _eval_cf(s_setup, x, ei, sem_cf, None)
            print(f"  S {cf:26s} acc={s_metrics[cf]['val_acc']:.5f} "
                  f"f1={s_metrics[cf]['val_macro_f1']:.5f}", flush=True)

        _summary_s = _json.load(open(
            R2D1_ROOT / "semantic" / ds / "S" / "seed_42" / "summary.json"
        ))
        assert abs(s_metrics["full"]["val_acc"] - _summary_s["best_val_acc"]) < 1e-5, (
            f"S/{ds} recomputed {s_metrics['full']['val_acc']} != history {_summary_s['best_val_acc']}"
        )
        e_forward_s = s_metrics["full"]["val_acc"] - s_metrics["both_off"]["val_acc"]
        g_coadapt_s = s_metrics["both_off"]["val_acc"] - m_b0["val_acc"]
        e_common = s_metrics["common_only"]["val_acc"] - s_metrics["both_off"]["val_acc"]
        e_residual = s_metrics["full"]["val_acc"] - s_metrics["common_only"]["val_acc"]
        residual_cross = s_metrics["fixed_common_residual"]["val_acc"] - s_metrics["both_off"]["val_acc"]
        for cf, metrics in s_metrics.items():
            s_rows.append({
                "dataset": ds, "variant": "S", "cf": cf,
                "val_acc": metrics["val_acc"], "val_macro_f1": metrics["val_macro_f1"],
            })
        classifications[f"S/{ds}"] = _classify(
            ds, e_forward_s, g_coadapt_s,
            f"(E_common={e_common*100:+.2f}pp, E_residual={e_residual*100:+.2f}pp, "
            f"fixed_common_residual={residual_cross*100:+.2f}pp)",
        )

        grad_s = ce_only_gradient_pair(
            s_setup.model, s_setup.head, x, ei, s_setup.data, device,
            cf_a=(None, None), cf_b=("both_off", None),
        )
        for group, stats in grad_s.items():
            grad_rows.append({"dataset": ds, "variant": "S", "group": group, **stats})

        drift_s = param_drift(b0.model, s_setup.model, b0.head, s_setup.head)
        for group, value in drift_s.items():
            drift_rows.append({"dataset": ds, "variant": "S", "group": group, "drift": value})

        z_s_off, _ = forward_cf(s_setup.model, x, ei, "both_off", None)
        rep_s = representation_drift(z_b0, z_s_off, val_idx)
        rep_rows.append({"dataset": ds, "variant": "S", "cf": "both_off", **rep_s})
        del z_s_off, s_setup
        torch.cuda.empty_cache()

        for tag, metrics in (("S_full", s_metrics["full"]), ("S_both_off", s_metrics["both_off"])):
            perclass_rows.append({
                "dataset": ds, "config": tag,
                "per_class_f1": metrics["per_class_f1"],
                "confusion": metrics["confusion"],
            })
        del z_b0, b0
        torch.cuda.empty_cache()

    write_csv(OUTDIR / "f_counterfactual.csv", f_rows)
    write_csv(OUTDIR / "s_counterfactual.csv", s_rows)
    write_csv(OUTDIR / "gradient_diagnostics.csv", grad_rows)
    write_csv(OUTDIR / "parameter_drift.csv", drift_rows)
    write_csv(OUTDIR / "representation_drift.csv", rep_rows)
    write_csv(OUTDIR / "per_class_metrics.csv", perclass_rows)

    # ---------------- report ----------------
    lines = [
        "# R2D15_COUNTERFACTUAL_REPORT — D1.5-B Failure Decomposition",
        "",
        "> Post-hoc module masking causes distribution shift; these are "
        "diagnostic counterfactuals, NOT retrained causal effects (plan §8.1).",
        "",
        "## Failure classification (plan §10)",
        "",
    ]
    for ds in datasets:
        lines.append(f"- **{ds}**")
        lines.append(f"  - F: {classifications[f'F/{ds}']}")
        lines.append(f"  - S: {classifications[f'S/{ds}']}")
    lines += [
        "",
        "## Key quantities (pp)",
        "",
        "| dataset | E_forward_F | G_coadapt_F | E_offdiag_F | E_forward_S | G_coadapt_S | E_common_S | E_residual_S |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in datasets:
        accs = {}
        for row in f_rows + s_rows:
            if row["dataset"] == ds:
                accs[(row["variant"], row["cf"])] = row["val_acc"]
        b0v = accs[("B0", "reference")]
        e_f = accs[("F", "full")] - accs[("F", "func_off")]
        g_f = accs[("F", "func_off")] - b0v
        o_f = accs[("F", "offdiag_only")] - accs[("F", "func_off")]
        e_s = accs[("S", "full")] - accs[("S", "both_off")]
        g_s = accs[("S", "both_off")] - b0v
        c_s = accs[("S", "common_only")] - accs[("S", "both_off")]
        r_s = accs[("S", "full")] - accs[("S", "common_only")]
        lines.append(
            f"| {ds} | {e_f*100:+.3f} | {g_f*100:+.3f} | {o_f*100:+.3f} "
            f"| {e_s*100:+.3f} | {g_s*100:+.3f} | {c_s*100:+.3f} | {r_s*100:+.3f} |"
        )
    lines += [
        "",
        "Files: f_counterfactual.csv, s_counterfactual.csv, "
        "gradient_diagnostics.csv, parameter_drift.csv, representation_drift.csv, "
        "per_class_metrics.csv.",
    ]
    (OUTDIR / "R2D15_COUNTERFACTUAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[counterfactual] saved -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
