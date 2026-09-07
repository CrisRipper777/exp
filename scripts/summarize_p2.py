"""P2 aggregation + report (plan §47).

Reads outputs/p2/<stage>/<dataset>/<mode>/(seed_<s>/)summary.json, plus the
FROZEN P1 F1R1 confirm results as reference (never re-run), and writes:

    p2_<stage>_results.csv     per-run val/test/params/runtime/peak
    p2_<stage>_mechanism.csv   plan diagnostics per run
    P2_SCREEN_REPORT.md (screen) / P2_REPORT.md (confirm)

Report answers (plan §47): NS vs P1, UOT vs NS, AUOT vs FUOT on low-S_R
graphs, Grocery/ele capacity benefit, Reddit-S weak-constraint regime,
Toys mitigation, plan collapse check. Decision on validation Accuracy.

Usage:
    python scripts/summarize_p2.py --stage screen
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
P2_ROOT = PROJECT_ROOT / "outputs" / "p2"
P1_CONFIRM_CSV = PROJECT_ROOT / "outputs" / "p1" / "confirm" / "p1_confirm_results.csv"

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
MODES = ["null_softmax", "fixed_uot", "adaptive_uot", "composition_uot"]


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return f"{value:.{digits}f}"


def _mean_std(values: list[float], digits: int = 4) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return _fmt(values[0], digits)
    return f"{statistics.mean(values):.{digits}f}±{statistics.pstdev(values):.{digits}f}"


def _metric(results: dict | None, key: str) -> float | None:
    if not results or key not in results:
        return None
    entry = results[key]
    if isinstance(entry, dict):
        return entry.get("mean")
    return entry


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_summaries(stage: str) -> list[dict]:
    stage_root = P2_ROOT / stage
    if not stage_root.exists():
        return []
    summaries = []
    for path in sorted(stage_root.glob("*/**/summary.json")):
        with path.open(encoding="utf-8") as f:
            summaries.append(json.load(f))
    return summaries


def _load_p1_f1r1_reference() -> dict[str, dict[str, str]]:
    """{dataset: {metric: 'mean±std'}} from frozen P1 confirm F1R1 rows.

    Val Macro-F1 is not in P1's results.json; parse it from the P1 train
    logs (Val F1 at the best-val-acc epoch, 2-decimal rounded — same
    convention as the P2 driver).
    """
    ref: dict[str, dict[str, str]] = {}
    if not P1_CONFIRM_CSV.exists():
        return ref
    with P1_CONFIRM_CSV.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["variant"] == "F1R1"]
    for dataset in DATASETS:
        runs = [r for r in rows if r["dataset"] == dataset]
        if not runs:
            continue
        out: dict[str, str] = {}
        for key, digits in (("best_val_acc", 4), ("test_acc", 4), ("test_macro_f1", 4)):
            values = [float(r[key]) for r in runs if r.get(key)]
            out[key] = _mean_std(values, digits)
        val_f1s: list[float] = []
        for run in runs:
            log = P1_CONFIRM_CSV.parent / run["dataset"] / "F1R1" / f"seed_{run['seed']}" / "train.log"
            best_acc, best_f1 = -1.0, None
            if log.exists():
                for line in log.read_text(encoding="utf-8").splitlines():
                    match = re.search(r"Val Acc ([\d.]+) \| Val F1 ([\d.]+)", line)
                    if match:
                        acc = float(match.group(1))
                        if acc > best_acc:
                            best_acc, best_f1 = acc, float(match.group(2))
            if best_f1 is not None:
                val_f1s.append(best_f1)
        if val_f1s:
            out["best_val_macro_f1"] = _mean_std(val_f1s)
        ref[dataset] = out
    return ref


def _results_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for s in summaries:
        rows.append({
            "dataset": s["dataset"],
            "mode": s["mode"],
            "seed": s["seed"],
            "best_val_acc": _fmt(_metric(s.get("results"), "val_acc")),
            "best_val_macro_f1": _fmt(s.get("best_val_macro_f1")),
            "test_acc": _fmt(_metric(s.get("results"), "test_acc")),
            "test_macro_f1": _fmt(_metric(s.get("results"), "test_macro_f1")),
            "params": s.get("params", ""),
            "epoch_time_sec": _fmt(s.get("epoch_time_sec"), 2),
            "train_peak_gpu_mb": s.get("train_peak_gpu_mb", ""),
        })
    return rows


def _mechanism_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for s in summaries:
        diag = s.get("diagnostics") or {}
        plan = diag.get("plan", {})
        row: dict = {
            "dataset": s["dataset"],
            "mode": s["mode"],
            "seed": s["seed"],
            "K_eff": _fmt(diag.get("relation", {}).get("effective_num")),
            "S_R": _fmt(diag.get("relation", {}).get("specialization")),
            "capacity_kl": _fmt(diag.get("capacity_kl")),
            "capacity_l1": _fmt(diag.get("capacity_l1")),
            "rel_conf_mean": _fmt(diag.get("relation_confidence", {}).get("mean")),
            "theta_mean": _fmt(diag.get("theta", {}).get("mean")),
        }
        for name in ("C", "Pt", "Pv"):
            stats = plan.get(name, {})
            row[f"null_{name}"] = _fmt(stats.get("null_mean"))
            row[f"graph_{name}"] = _fmt(stats.get("graph_mass_mean"))
            row[f"plan_ent_{name}"] = _fmt(diag.get("plan_entropy", {}).get(name))
            row[f"alpha_ent_{name}"] = _fmt(diag.get("alpha_entropy", {}).get(name))
        for key, value in diag.get("alpha_js", {}).items():
            row[f"js_{key}"] = _fmt(value)
        rows.append(row)
    return rows


def _mode_groups(summaries: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for s in summaries:
        grouped.setdefault((s["dataset"], s["mode"]), []).append(s)
    return grouped


def _write_report(stage: str, summaries: list[dict], ref: dict[str, dict[str, str]]) -> None:
    lines: list[str] = []
    lines.append("# P2 Screen Report — Null-Augmented Factor–Relation Transport" if stage == "screen" else "# P2 Report")
    lines.append("")
    lines.append(f"> Runs: {len(summaries)}. P1 F1R1 = frozen confirm reference (never re-run). Decision metric: validation Accuracy.")
    lines.append("")
    grouped = _mode_groups(summaries)

    lines.append("## Main table (mean±std over seeds)")
    lines.append("")
    header = "| Dataset | P1 F1R1 | NullSoftmax | Fixed-UOT | Adaptive-UOT | Composition-UOT |"
    lines.append(header)
    lines.append("|---|" + "---:|" * 5)
    for dataset in DATASETS:
        cells = []
        for mode in MODES:
            runs = grouped.get((dataset, mode), [])
            vals = [_metric(r.get("results"), "val_acc") for r in runs]
            vals = [v for v in vals if v is not None]
            cells.append(_mean_std(vals) if vals else "")
        p1 = ref.get(dataset, {}).get("best_val_acc", "")
        lines.append(f"| {dataset} | {p1} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Test acc (same order):")
    lines.append("")
    lines.append("| Dataset | P1 F1R1 | NullSoftmax | Fixed-UOT | Adaptive-UOT | Composition-UOT |")
    lines.append("|---|" + "---:|" * 5)
    for dataset in DATASETS:
        cells = []
        for mode in MODES:
            runs = grouped.get((dataset, mode), [])
            vals = [_metric(r.get("results"), "test_acc") for r in runs]
            vals = [v for v in vals if v is not None]
            cells.append(_mean_std(vals) if vals else "")
        p1 = ref.get(dataset, {}).get("test_acc", "")
        lines.append(f"| {dataset} | {p1} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("Val Macro-F1 (at best-val-acc epoch; P2 parsed from train.log, P1 likewise):")
    lines.append("")
    lines.append("| Dataset | P1 F1R1 | NullSoftmax | Fixed-UOT | Adaptive-UOT | Composition-UOT |")
    lines.append("|---|" + "---:|" * 5)
    for dataset in DATASETS:
        cells = []
        for mode in MODES:
            runs = grouped.get((dataset, mode), [])
            vals = [r.get("best_val_macro_f1") for r in runs]
            vals = [v for v in vals if v is not None]
            cells.append(_mean_std(vals) if vals else "")
        p1 = ref.get(dataset, {}).get("best_val_macro_f1", "")
        lines.append(f"| {dataset} | {p1} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("Test Macro-F1 (frozen-checkpoint confirmation):")
    lines.append("")
    lines.append("| Dataset | P1 F1R1 | NullSoftmax | Fixed-UOT | Adaptive-UOT | Composition-UOT |")
    lines.append("|---|" + "---:|" * 5)
    for dataset in DATASETS:
        cells = []
        for mode in MODES:
            runs = grouped.get((dataset, mode), [])
            vals = [_metric(r.get("results"), "test_macro_f1") for r in runs]
            vals = [v for v in vals if v is not None]
            cells.append(_mean_std(vals) if vals else "")
        p1 = ref.get(dataset, {}).get("test_macro_f1", "")
        lines.append(f"| {dataset} | {p1} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Mechanism (per run)")
    lines.append("")
    lines.append("| Dataset | Mode | Seed | K_eff | S_R | null_C | null_Pt | null_Pv | graph_C | graph_Pt | graph_Pv | JS C/Pt | JS C/Pv | JS Pt/Pv | cap_KL | θ_mean |")
    lines.append("|---|" + "---:|" * 15)
    for s in summaries:
        diag = s.get("diagnostics") or {}
        plan = diag.get("plan", {})
        js = diag.get("alpha_js", {})
        lines.append(
            f"| {s['dataset']} | {s['mode']} | {s['seed']} | "
            f"{_fmt(diag.get('relation', {}).get('effective_num'))} | {_fmt(diag.get('relation', {}).get('specialization'))} | "
            f"{_fmt(plan.get('C', {}).get('null_mean'))} | {_fmt(plan.get('Pt', {}).get('null_mean'))} | {_fmt(plan.get('Pv', {}).get('null_mean'))} | "
            f"{_fmt(plan.get('C', {}).get('graph_mass_mean'))} | {_fmt(plan.get('Pt', {}).get('graph_mass_mean'))} | {_fmt(plan.get('Pv', {}).get('graph_mass_mean'))} | "
            f"{_fmt(js.get('C_Pt'))} | {_fmt(js.get('C_Pv'))} | {_fmt(js.get('Pt_Pv'))} | "
            f"{_fmt(diag.get('capacity_kl'))} | {_fmt(diag.get('theta', {}).get('mean'))} |"
        )
    lines.append("")

    # Hypotheses (plan §35) — computed evidence only.
    lines.append("## Hypotheses (computed, val Acc)")
    lines.append("")
    for dataset in DATASETS:
        ns = grouped.get((dataset, "null_softmax"), [])
        fu = grouped.get((dataset, "fixed_uot"), [])
        au = grouped.get((dataset, "adaptive_uot"), [])
        p1 = ref.get(dataset, {}).get("best_val_acc", "")
        def _m(runs):
            vals = [_metric(r.get("results"), "val_acc") for r in runs]
            vals = [v for v in vals if v is not None]
            return _mean_std(vals) if vals else ""
        lines.append(f"- {dataset}: P1={p1} | NS={_m(ns)} | FUOT={_m(fu)} | AUOT={_m(au)}")
    lines.append("")

    lines.append("## GO / REVISE / NO-GO checklist (plan §36, fill after analysis)")
    lines.append("")
    lines.append("- [ ] One transport variant improves val Acc over P1 F1R1 on ≥3/5 datasets")
    lines.append("- [ ] That variant beats NullSoftmax on ≥2 high-S_R datasets (Grocery/ele-fashion)")
    lines.append("- [ ] No large degradation on Toys/Reddit-S")
    lines.append("- [ ] No full-null / uniform-plan collapse (see null_*/plan entropy)")
    lines.append("- [ ] capacity deviation (KL) decreases vs NullSoftmax")
    lines.append("- [ ] conditional JS stays nonzero on Grocery/ele-fashion")
    lines.append("- [ ] Reddit-S not forced into fake factor-relation differentiation")
    lines.append("")
    lines.append("- Decision: **PENDING** (fill after analysis)")
    lines.append("")

    report_name = "P2_SCREEN_REPORT.md" if stage == "screen" else "P2_REPORT.md"
    (P2_ROOT / stage / report_name).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 aggregation + report")
    parser.add_argument("--stage", default="screen", choices=["screen", "confirm"])
    args = parser.parse_args()
    stage = args.stage

    summaries = _load_summaries(stage)
    if not summaries:
        print(f"[summarize] no summary.json under outputs/p2/{stage}", flush=True)
        return
    stage_root = P2_ROOT / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    _write_csv(stage_root / f"p2_{stage}_results.csv", _results_rows(summaries))
    _write_csv(stage_root / f"p2_{stage}_mechanism.csv", _mechanism_rows(summaries))
    ref = _load_p1_f1r1_reference()
    _write_report(stage, summaries, ref)
    print(f"[summarize] {len(summaries)} runs -> {stage_root}", flush=True)


if __name__ == "__main__":
    main()
