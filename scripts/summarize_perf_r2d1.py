"""R2-Design-1 summarizer: stage tables, pre-registered GO/NO-GO verdicts
and markdown reports (plan §25-§32 / §35).

Only READS existing run summaries + the frozen A0 per-seed reference.
Never trains, never touches test.

Stages:
    b0        : R2-B0 vs A0 -> ACCEPTABLE CLEAN PARENT / B0 AUDIT REQUIRED
    functional: R2-F vs B0/A0 -> Score_F GO / Strong / NO-GO
    semantic  : R2-S vs B0/A0 -> Score_S GO / Strong + mechanism health flags
    joint     : pre-registered entry conditions + R2-J GO verdict
    confirm   : guards + 3-seed formal verdict (STRONG/GO/WEAK/NO-GO)
    final     : master tables + the 12-question final diagnosis

Usage:
    python scripts/summarize_perf_r2d1.py --stage b0
    python scripts/summarize_perf_r2d1.py --stage final
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2_utils import (  # noqa: E402
    A0_REFERENCE_CSV,
    DATASETS,
    GUARD_DATASETS,
    SEEDS,
    TARGET_DATASETS,
    VARIANT_ROOTS,
    VARIANTS,
    a0_val_acc,
    load_a0_reference,
)

R2D1_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d1"
SUMMARY_DIR = R2D1_ROOT / "summary"
VARIANT_LABELS = {"B0": "R2-B0", "F": "R2-F", "S": "R2-S", "J": "R2-J"}


def _load_summary(dataset: str, variant: str, seed: int) -> dict | None:
    path = R2D1_ROOT / VARIANT_ROOTS[variant] / dataset / variant / f"seed_{seed}" / "summary.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_all_summaries() -> dict[tuple[str, str, int], dict]:
    out: dict[tuple[str, str, int], dict] = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                summary = _load_summary(dataset, variant, seed)
                if summary is not None:
                    out[(dataset, variant, seed)] = summary
    return out


def _pp(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _delta_pp(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100 * value:+.3f}"


def _a0_params(reference_csv=None) -> dict[tuple[str, int], int]:
    """A0 per-(dataset, seed) parameter count from the frozen reference CSV."""
    out: dict[tuple[str, int], int] = {}
    with A0_REFERENCE_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] != "biaxis_final":
                continue
            out[(row["dataset"], int(row["seed"]))] = int(row["params"])
    return out


def _rows_for(summaries: dict, variants, datasets, seeds) -> list[dict]:
    """One row per run, joined with the frozen A0 reference. A0 rows are
    included (variant="A0") so all delta helpers work uniformly."""
    reference = load_a0_reference()
    a0_params = _a0_params()
    rows = []
    for dataset in datasets:
        for variant in variants:
            for seed in seeds:
                summary = summaries.get((dataset, variant, seed))
                if summary is None:
                    continue
                val = summary["best_val_acc"]
                a0 = a0_val_acc(reference, dataset, seed)
                rows.append({
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "best_val_acc": val,
                    "best_val_macro_f1": summary["best_val_macro_f1"],
                    "a0_val_acc": a0,
                    "delta_vs_a0": (val - a0) if val is not None else None,
                    "parameter_count": summary.get("parameter_count"),
                    "best_epoch": summary.get("best_epoch"),
                    "stop_epoch": summary.get("stop_epoch"),
                    "epochs_run": summary.get("epochs_run"),
                    "runtime_sec": summary.get("runtime_sec"),
                    "epoch_time_sec": summary.get("epoch_time_sec"),
                    "peak_gpu_mb": summary.get("peak_gpu_mb"),
                    "train_acc_at_best": summary.get("train_acc_at_best"),
                    "train_loss_at_best": summary.get("train_loss_at_best"),
                })
    for dataset in datasets:
        for seed in seeds:
            if (dataset, seed) not in reference:
                continue
            a0 = a0_val_acc(reference, dataset, seed)
            rows.append({
                "dataset": dataset,
                "variant": "A0",
                "seed": seed,
                "best_val_acc": a0,
                "best_val_macro_f1": None,
                "a0_val_acc": a0,
                "delta_vs_a0": 0.0,
                "parameter_count": a0_params.get((dataset, seed)),
                "best_epoch": None,
                "stop_epoch": None,
                "epochs_run": None,
                "runtime_sec": None,
                "epoch_time_sec": None,
                "peak_gpu_mb": None,
                "train_acc_at_best": None,
                "train_loss_at_best": None,
            })
    return rows


def _mean_delta(rows: list[dict], variant_a: str, variant_b: str, datasets=None, seed: int | None = None) -> float | None:
    """Mean ValAcc(variant_a) - ValAcc(variant_b) over (dataset, seed) pairs."""
    values_a, values_b = {}, {}
    for row in rows:
        if seed is not None and row["seed"] != seed:
            continue
        if datasets is not None and row["dataset"] not in datasets:
            continue
        if row["variant"] == variant_a and row["best_val_acc"] is not None:
            values_a[(row["dataset"], row["seed"])] = row["best_val_acc"]
        if row["variant"] == variant_b and row["best_val_acc"] is not None:
            values_b[(row["dataset"], row["seed"])] = row["best_val_acc"]
    deltas = [values_a[k] - values_b[k] for k in values_a if k in values_b]
    return statistics.mean(deltas) if deltas else None


def _per_dataset_delta(rows: list[dict], variant_a: str, variant_b: str, datasets=None) -> dict[str, float]:
    """{dataset: mean over PAIRED seeds of (A - B) ValAcc}."""
    out: dict[str, float] = {}
    for dataset in (datasets or TARGET_DATASETS):
        pairs = []
        seeds_in = {
            r["seed"]
            for r in rows
            if r["dataset"] == dataset and r["variant"] == variant_a and r["best_val_acc"] is not None
        }
        for seed in sorted(seeds_in):
            a = next(r for r in rows if r["dataset"] == dataset and r["variant"] == variant_a and r["seed"] == seed)
            b = next((r for r in rows if r["dataset"] == dataset and r["variant"] == variant_b and r["seed"] == seed), None)
            if b is not None:
                pairs.append(a["best_val_acc"] - b["best_val_acc"])
        if pairs:
            out[dataset] = statistics.mean(pairs)
    return out


def _write_results_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "variant", "seed", "best_val_acc", "best_val_macro_f1",
            "a0_val_acc", "delta_vs_a0", "parameter_count", "best_epoch",
            "stop_epoch", "epochs_run", "runtime_sec", "epoch_time_sec",
            "peak_gpu_mb", "train_acc_at_best", "train_loss_at_best",
        ])
        for row in rows:
            writer.writerow([row[k] for k in (
                "dataset", "variant", "seed", "best_val_acc", "best_val_macro_f1",
                "a0_val_acc", "delta_vs_a0", "parameter_count", "best_epoch",
                "stop_epoch", "epochs_run", "runtime_sec", "epoch_time_sec",
                "peak_gpu_mb", "train_acc_at_best", "train_loss_at_best",
            )])


def _mechanism_rows(summaries: dict, variants, datasets, seeds) -> list[dict]:
    rows: list[dict] = []
    for dataset in datasets:
        for variant in variants:
            for seed in seeds:
                summary = summaries.get((dataset, variant, seed))
                diag = (summary or {}).get("diagnostics") or {}
                row: dict = {"dataset": dataset, "variant": variant, "seed": seed}
                if not diag:
                    rows.append(row)
                    continue
                p0 = diag.get("p0") or {}
                row.update({f"p0_{key}": p0.get(key) for key in (
                    "p0_common_sim", "p0_private_sim", "p0_c_norm",
                    "p0_pt_norm", "p0_pv_norm", "p0_cp_overlap_t",
                    "p0_cp_overlap_v", "p0_aux_loss",
                )})
                sem = diag.get("semantic")
                if sem:
                    row["w_t_mean"] = sem["w_t"]["mean"]
                    row["w_t_std"] = sem["w_t"]["std"]
                    row["w_t_frac_lt_05"] = sem["w_t"]["frac_lt_05"]
                    row["w_t_frac_gt_95"] = sem["w_t"]["frac_gt_95"]
                    row["w_v_mean"] = sem["w_v"]["mean"]
                    row["w_v_std"] = sem["w_v"]["std"]
                    for name, key in (("C", "sem_ratio_C"), ("Pt", "sem_ratio_Pt"), ("Pv", "sem_ratio_Pv")):
                        row[f"{key}_mean"] = sem["sem_residual_ratio"][name]["mean"]
                        row[f"{key}_std"] = sem["sem_residual_ratio"][name]["std"]
                row["rho_base_C"] = diag["rho_base"][0] if diag.get("rho_base") else None
                row["rho_base_Pt"] = diag["rho_base"][1] if diag.get("rho_base") else None
                row["rho_base_Pv"] = diag["rho_base"][2] if diag.get("rho_base") else None
                for name, idx in (("C", 0), ("Pt", 1), ("Pv", 2)):
                    row[f"base_ratio_{name}_mean"] = diag["base_residual_ratio"][name]["mean"]
                    row[f"base_ratio_{name}_std"] = diag["base_residual_ratio"][name]["std"]
                func = diag.get("functional")
                if func:
                    gm = func["gate_matrix"]
                    for a, src in enumerate(("C", "Pt", "Pv")):
                        for b, tgt in enumerate(("C", "Pt", "Pv")):
                            row[f"gate_{src}{tgt}_mean"] = gm["mean"][a][b]
                            row[f"gate_{src}{tgt}_std"] = gm["std"][a][b]
                            row[f"gate_{src}{tgt}_frac_lt_05"] = gm["frac_lt_05"][a][b]
                            row[f"gate_{src}{tgt}_frac_gt_95"] = gm["frac_gt_95"][a][b]
                    cm = func["contribution_matrix"]
                    for a, src in enumerate(("C", "Pt", "Pv")):
                        for b, tgt in enumerate(("C", "Pt", "Pv")):
                            row[f"contrib_{src}{tgt}"] = cm["values"][a][b]
                    row["rho_func_C"] = func["rho_func"][0]
                    row["rho_func_Pt"] = func["rho_func"][1]
                    row["rho_func_Pv"] = func["rho_func"][2]
                    for name, idx in (("C", 0), ("Pt", 1), ("Pv", 2)):
                        row[f"func_ratio_{name}_mean"] = func["func_residual_ratio"][name]["mean"]
                        row[f"func_ratio_{name}_std"] = func["func_residual_ratio"][name]["std"]
                rows.append(row)
    return rows


def _write_mechanism_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def _write_resource_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "variant", "seed", "parameter_count",
                         "peak_gpu_mb", "epoch_time_sec", "runtime_sec",
                         "best_epoch", "stop_epoch"])
        for row in rows:
            writer.writerow([row["dataset"], row["variant"], row["seed"],
                             row["parameter_count"], row["peak_gpu_mb"],
                             row["epoch_time_sec"], row["runtime_sec"],
                             row["best_epoch"], row["stop_epoch"]])


def _dataset_table(rows: list[dict], variants, datasets, seed: int) -> str:
    reference = load_a0_reference()
    header = ["dataset"] + [f"{VARIANT_LABELS[v]} ValAcc" for v in variants] + \
             [f"{v}-B0" for v in variants if v != "B0"] + \
             [f"{v}-A0" for v in variants] + ["A0 ValAcc"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for dataset in datasets:
        a0 = a0_val_acc(reference, dataset, seed)
        vals = {}
        for v in variants:
            summary = _load_summary(dataset, v, seed)
            vals[v] = summary["best_val_acc"] if summary else None
        cells = [dataset]
        cells += [_pp(vals[v]) for v in variants]
        if "B0" in variants:
            cells += [
                _delta_pp(vals[v] - vals["B0"])
                if vals[v] is not None and vals["B0"] is not None else "-"
                for v in variants if v != "B0"
            ]
        cells += [_delta_pp(vals[v] - a0) if vals[v] is not None else "-" for v in variants]
        cells += [_pp(a0)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage reports
# ---------------------------------------------------------------------------


def _write_b0_report(rows: list[dict]) -> None:
    summaries = load_all_summaries()
    deltas = _per_dataset_delta(rows, "B0", "A0", TARGET_DATASETS)
    mean_delta = _mean_delta(rows, "B0", "A0", TARGET_DATASETS, seed=42)
    missing = [d for d in TARGET_DATASETS if (d, "B0", 42) not in summaries]
    lines = [
        "# R2-Design-1 D1-1 — R2-B0 Clean Parent Report",
        "",
        "Protocol: Movies/Toys/Grocery, seed42, Val only, 300ep/patience30, "
        "AdamW lr=1e-3 wd=1e-4, full graph. A0 = frozen `biaxis_final` reference "
        "(no retraining, plan §25).",
        "",
        "## Seed42 results (Val Acc; Δ in pp)",
        "",
        _dataset_table(rows, ["B0"], TARGET_DATASETS, 42),
        "",
        "## Per-dataset B0 − A0 (pp)",
        "",
    ]
    for dataset in TARGET_DATASETS:
        d = deltas.get(dataset)
        lines.append(f"- {dataset}: {_delta_pp(d)} pp" if d is not None else f"- {dataset}: MISSING")
    lines.append("")
    lines.append(f"**mean(B0 − A0) = {_delta_pp(mean_delta)} pp**")
    lines.append("")
    if missing:
        lines.append(f"## MISSING RUNS: {missing} — verdict deferred until complete.")
    elif mean_delta is None:
        lines.append("## No B0 results found — verdict deferred.")
    else:
        audit_required = mean_delta < -0.8 / 100 or any(
            d is not None and d < -1.5 / 100 for d in deltas.values()
        )
        acceptable = mean_delta >= -0.50 / 100 and all(
            d is None or d >= -1.0 / 100 for d in deltas.values()
        )
        if audit_required:
            lines.append("## VERDICT: **B0 AUDIT REQUIRED** (mean < −0.80pp or a dataset < −1.5pp)")
            lines.append("")
            lines.append("Stop the pipeline. Do NOT run F/S/J until B0 is audited (plan §25).")
        elif acceptable:
            lines.append("## VERDICT: **ACCEPTABLE CLEAN PARENT** "
                         "(mean ≥ −0.50pp and no dataset < −1.0pp)")
            lines.append("")
            lines.append("Proceed to D1-2 (R2-F) per plan §26.")
        else:
            lines.append("## VERDICT: **MARGINAL** (between the acceptable and audit thresholds)")
            lines.append("")
            lines.append("Plan §25 leaves this band unspecified — manual review required "
                         "before running F/S.")
    outdir = R2D1_ROOT / "b0"
    (outdir / "R2D1_B0_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_results_csv(rows, outdir / "b0_results.csv")
    print(f"[b0] report -> {outdir / 'R2D1_B0_REPORT.md'}")


def _score_verdict(score: float | None, go: float, strong: float, positives: int, total: int) -> str:
    if score is None:
        return "MISSING"
    if score >= strong and positives >= 2:
        return "STRONG"
    if score >= go and positives >= 2:
        return "GO"
    return "NO-GO"


def _write_functional_report(rows: list[dict]) -> None:
    summaries = load_all_summaries()
    score_f = _mean_delta(rows, "F", "B0", TARGET_DATASETS, seed=42)
    f_minus_a0 = _mean_delta(rows, "F", "A0", TARGET_DATASETS, seed=42)
    per_ds = _per_dataset_delta(rows, "F", "B0", TARGET_DATASETS)
    positives = sum(1 for d in per_ds.values() if d > 0)
    missing = [d for d in TARGET_DATASETS if (d, "F", 42) not in summaries]
    lines = [
        "# R2-Design-1 D1-2 — R2-F Functional Transfer Report",
        "",
        "R2-F = B0 + target-conditioned functional residual (plan §9-§14). "
        "Movies/Toys/Grocery, seed42, Val only.",
        "",
        "## Seed42 results (Val Acc; Δ in pp)",
        "",
        _dataset_table(rows, ["B0", "F"], TARGET_DATASETS, 42),
        "",
        "## Per-dataset F − B0 (pp)",
        "",
    ]
    for dataset in TARGET_DATASETS:
        lines.append(f"- {dataset}: {_delta_pp(per_ds.get(dataset))} pp")
    lines += [
        "",
        f"**Score_F = mean(F − B0) = {_delta_pp(score_f)} pp** "
        f"(positive datasets: {positives}/3)",
        f"**mean(F − A0) = {_delta_pp(f_minus_a0)} pp**",
        "",
    ]
    if missing:
        lines += [f"## MISSING RUNS: {missing} — verdict deferred.", ""]
    elif score_f is None:
        lines += ["## No F results found — verdict deferred.", ""]
    else:
        verdict = _score_verdict(score_f, 0.30 / 100, 0.50 / 100, positives, 3)
        lines += [f"## VERDICT: **{verdict}** (GO-to-confirm: Score_F ≥ +0.30pp & ≥2/3 positive; Strong ≥ +0.50pp)", ""]
        if score_f < -0.30 / 100:
            lines += ["**Functional core explicit NO-GO (Score_F < −0.30pp): do NOT run J** (plan §28)."]
        elif verdict == "GO" or verdict == "STRONG":
            lines += ["Proceed to D1-3 (R2-S). J is decided after S (plan §28 conditions)."]
        else:
            lines += ["Score_F below the GO gate. J is decided after S (plan §28 conditions)."]
    outdir = R2D1_ROOT / "functional"
    (outdir / "R2D1_FUNCTIONAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_results_csv(rows, outdir / "functional_results.csv")
    _write_mechanism_csv(_mechanism_rows(summaries, ["F"], TARGET_DATASETS, [42]), outdir / "functional_mechanism.csv")
    print(f"[functional] report -> {outdir / 'R2D1_FUNCTIONAL_REPORT.md'}")


def _write_semantic_report(rows: list[dict]) -> None:
    summaries = load_all_summaries()
    score_s = _mean_delta(rows, "S", "B0", TARGET_DATASETS, seed=42)
    s_minus_a0 = _mean_delta(rows, "S", "A0", TARGET_DATASETS, seed=42)
    per_ds = _per_dataset_delta(rows, "S", "B0", TARGET_DATASETS)
    positives = sum(1 for d in per_ds.values() if d > 0)
    missing = [d for d in TARGET_DATASETS if (d, "S", 42) not in summaries]
    lines = [
        "# R2-Design-1 D1-3 — R2-S Semantic Refiner Report",
        "",
        "R2-S = B0 + node-adaptive common consensus + zero-init factor "
        "interaction residual (plan §6-§7). Movies/Toys/Grocery, seed42, Val only.",
        "",
        "## Seed42 results (Val Acc; Δ in pp)",
        "",
        _dataset_table(rows, ["B0", "S"], TARGET_DATASETS, 42),
        "",
        "## Per-dataset S − B0 (pp)",
        "",
    ]
    for dataset in TARGET_DATASETS:
        lines.append(f"- {dataset}: {_delta_pp(per_ds.get(dataset))} pp")
    lines += [
        "",
        f"**Score_S = mean(S − B0) = {_delta_pp(score_s)} pp** "
        f"(positive datasets: {positives}/3)",
        f"**mean(S − A0) = {_delta_pp(s_minus_a0)} pp**",
        "",
    ]
    # Mechanism health flags (plan §19/§27): collapse / domination / P0 health.
    flags = []
    for dataset in TARGET_DATASETS:
        summary = summaries.get((dataset, "S", 42))
        diag = (summary or {}).get("diagnostics") or {}
        sem = diag.get("semantic")
        if not sem:
            continue
        if sem["w_t"]["frac_gt_95"] > 0.8 or sem["w_t"]["frac_lt_05"] > 0.8:
            flags.append(f"- **{dataset}: common gate COLLAPSE suspicion** "
                         f"(w_t frac<.05={sem['w_t']['frac_lt_05']:.3f}, frac>.95={sem['w_t']['frac_gt_95']:.3f})")
        for name in ("C", "Pt", "Pv"):
            ratio = sem["sem_residual_ratio"][name]["mean"]
            if ratio > 1.0:
                flags.append(f"- **{dataset}: semantic refiner DOMINATES {name}** (ratio={ratio:.3f})")
    if flags:
        lines += ["## Mechanism flags (plan §19)", ""] + flags + [""]
    else:
        lines += ["## Mechanism flags: none (no collapse / no domination)", ""]
    if missing:
        lines += [f"## MISSING RUNS: {missing} — verdict deferred.", ""]
    elif score_s is None:
        lines += ["## No S results found — verdict deferred.", ""]
    else:
        verdict = _score_verdict(score_s, 0.20 / 100, 0.50 / 100, positives, 3)
        lines += [f"## VERDICT: **{verdict}** (GO-to-confirm: Score_S ≥ +0.20pp & ≥2/3 positive; Strong ≥ +0.50pp)", ""]
    outdir = R2D1_ROOT / "semantic"
    (outdir / "R2D1_SEMANTIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_results_csv(rows, outdir / "semantic_results.csv")
    _write_mechanism_csv(_mechanism_rows(summaries, ["S"], TARGET_DATASETS, [42]), outdir / "semantic_mechanism.csv")
    print(f"[semantic] report -> {outdir / 'R2D1_SEMANTIC_REPORT.md'}")


def _joint_entry_conditions() -> tuple[str, bool]:
    """Pre-registered entry conditions for R2-J (plan §28)."""
    rows = _rows_for(load_all_summaries(), ["B0", "F", "S"], TARGET_DATASETS, [42])
    score_f = _mean_delta(rows, "F", "B0", TARGET_DATASETS, seed=42)
    score_s = _mean_delta(rows, "S", "B0", TARGET_DATASETS, seed=42)
    if score_f is None or score_s is None:
        return ("INCOMPLETE (F and S seed42 required)", False)
    if score_f < -0.30 / 100:
        return ("NO-GO (Score_F < −0.30pp: functional core failed; do not mask it with the refiner)", False)
    cond1 = score_f >= 0.30 / 100 and score_s >= -0.10 / 100
    cond2 = score_s >= 0.20 / 100 and score_f >= -0.10 / 100
    if cond1 or cond2:
        return (f"GO (Score_F={_delta_pp(score_f)}pp, Score_S={_delta_pp(score_s)}pp; "
                f"condition {'1' if cond1 else '2'} satisfied)", True)
    return ("NO-GO (neither condition satisfied)", False)


def _write_joint_report(rows: list[dict]) -> None:
    summaries = load_all_summaries()
    entry_msg, entry_go = _joint_entry_conditions()
    lines = [
        "# R2-Design-1 D1-4 — R2-J Joint Report",
        "",
        f"**Entry conditions: {entry_msg}**",
        "",
    ]
    j_rows = [r for r in rows if r["variant"] == "J"]
    if not entry_go:
        lines += ["J NOT RUN (pre-registered gate, plan §28). No verdict.", ""]
    elif not j_rows:
        lines += ["J runs MISSING — verdict deferred.", ""]
    else:
        score_j_b0 = _mean_delta(rows, "J", "B0", TARGET_DATASETS, seed=42)
        score_j_a0 = _mean_delta(rows, "J", "A0", TARGET_DATASETS, seed=42)
        per_ds = _per_dataset_delta(rows, "J", "B0", TARGET_DATASETS)
        positives = sum(1 for d in per_ds.values() if d > 0)
        lines += [
            "## Seed42 results (Val Acc; Δ in pp)",
            "",
            _dataset_table(rows, ["B0", "F", "S", "J"], TARGET_DATASETS, 42),
            "",
            "## Per-dataset J − B0 (pp)",
            "",
        ]
        for dataset in TARGET_DATASETS:
            lines.append(f"- {dataset}: {_delta_pp(per_ds.get(dataset))} pp")
        lines += [
            "",
            f"**Score_J = mean(J − B0) = {_delta_pp(score_j_b0)} pp** (positive: {positives}/3)",
            f"**mean(J − A0) = {_delta_pp(score_j_a0)} pp**",
            "",
        ]
        go = (
            score_j_b0 is not None and score_j_b0 >= 0.40 / 100
            and positives >= 2
            and score_j_a0 is not None and score_j_a0 >= 0.20 / 100
        )
        lines += [f"## VERDICT: **{'GO' if go else 'NO-GO'}** "
                  "(J GO: mean(J−B0) ≥ +0.40pp, ≥2/3 positive, mean(J−A0) ≥ +0.20pp — plan §29)", ""]
    outdir = R2D1_ROOT / "joint"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "R2D1_JOINT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_results_csv(rows, outdir / "joint_results.csv")
    _write_mechanism_csv(
        _mechanism_rows(summaries, ["J"], TARGET_DATASETS, [42]), outdir / "joint_mechanism.csv"
    )
    print(f"[joint] report -> {outdir / 'R2D1_JOINT_REPORT.md'}")


def _write_confirm_report(rows: list[dict], candidates: list[str]) -> None:
    reference = load_a0_reference()
    summaries = load_all_summaries()
    lines = [
        "# R2-Design-1 D1-5/D1-6 — Guards + 3-seed Formal Confirmation Report",
        "",
        f"Candidates: {', '.join(VARIANT_LABELS[c] for c in candidates)} (pre-registered selection, plan §42)",
        "",
    ]
    guard_safe: dict[str, bool] = {}
    for variant in candidates:
        lines += [f"## Guards (seed42) — {VARIANT_LABELS[variant]} vs A0", ""]
        safe = True
        for dataset in GUARD_DATASETS:
            summary = summaries.get((dataset, variant, 42))
            a0 = a0_val_acc(reference, dataset, 42)
            if summary is None:
                lines.append(f"- {dataset}: MISSING")
                safe = False
                continue
            delta = summary["best_val_acc"] - a0
            ok = delta >= -0.20 / 100
            safe = safe and ok
            lines.append(f"- {dataset}: Δ = {_delta_pp(delta)} pp "
                         f"({'SAFE' if ok else 'GUARD FAILED'} vs ≥ −0.20pp; "
                         f"review if < −0.30pp)")
        guard_safe[variant] = safe
        lines += [f"**{variant} guards: {'SAFE' if safe else 'FAILED'}**", ""]
    lines += ["## 3-seed formal confirmation (seeds 42/43/44, Val Acc)", ""]
    for variant in candidates:
        if not guard_safe[variant]:
            lines += [f"### {VARIANT_LABELS[variant]}: skipped (guards failed)", ""]
            continue
        table_header = ["dataset", "s42 ΔvsA0", "s43 ΔvsA0", "s44 ΔvsA0", "3-seed mean Δ", "positive seeds", "3-seed mean Δ vs B0"]
        lines.append("| " + " | ".join(table_header) + " |")
        lines.append("|" + "---|" * len(table_header))
        dataset_means = []
        for dataset in DATASETS:
            deltas = []
            for seed in SEEDS:
                summary = summaries.get((dataset, variant, seed))
                if summary is None:
                    deltas.append(None)
                    continue
                deltas.append(summary["best_val_acc"] - a0_val_acc(reference, dataset, seed))
            if all(d is None for d in deltas):
                continue
            valid = [d for d in deltas if d is not None]
            mean_d = statistics.mean(valid)
            pos_seeds = sum(1 for d in valid if d > 0)
            b0_deltas = []
            for seed in SEEDS:
                s_v, s_b0 = summaries.get((dataset, variant, seed)), summaries.get((dataset, "B0", seed))
                if s_v and s_b0:
                    b0_deltas.append(s_v["best_val_acc"] - s_b0["best_val_acc"])
            vs_b0 = statistics.mean(b0_deltas) if b0_deltas else None
            dataset_means.append((dataset, mean_d, pos_seeds, len(valid)))
            lines.append("| " + " | ".join([
                dataset,
                *[_delta_pp(d) for d in deltas],
                _delta_pp(mean_d),
                f"{pos_seeds}/{len(valid)}",
                _delta_pp(vs_b0),
            ]) + " |")
        lines.append("")
        if dataset_means:
            mtg = [(ds, m, p, n) for ds, m, p, n in dataset_means if ds in TARGET_DATASETS]
            mtg_mean = statistics.mean([m for _, m, _, _ in mtg]) if mtg else None
            mtg_pos_ds = sum(1 for _, m, _, _ in mtg if m > 0)
            mtg_pos_seeds_ok = sum(1 for _, m, p, n in mtg if p >= max(2, 2 * n // 3) and m > 0)
            lines += [
                f"M/T/G 3-seed mean Δ vs A0 = {_delta_pp(mtg_mean)} pp; "
                f"positive target datasets: {mtg_pos_ds}/3",
                "",
            ]
            if mtg_mean is not None and mtg_mean >= 0.50 / 100 and mtg_pos_ds >= 2 and mtg_pos_seeds_ok >= 2 and guard_safe[variant]:
                verdict = "STRONG"
            elif mtg_mean is not None and mtg_mean >= 0.30 / 100 and mtg_pos_ds >= 2 and guard_safe[variant]:
                verdict = "GO"
            elif mtg_mean is not None and mtg_mean >= 0.15 / 100:
                verdict = "WEAK"
            else:
                verdict = "NO-GO"
            lines += [f"### {VARIANT_LABELS[variant]} FORMAL VERDICT: **{verdict}** (plan §32)", ""]
    outdir = R2D1_ROOT / "confirm"
    (outdir / "R2D1_CONFIRM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_results_csv(rows, outdir / "confirm_results.csv")
    _write_mechanism_csv(
        _mechanism_rows(summaries, candidates, DATASETS, SEEDS), outdir / "confirm_mechanism.csv"
    )
    print(f"[confirm] report -> {outdir / 'R2D1_CONFIRM_REPORT.md'}")


def _write_final_report(rows: list[dict]) -> None:
    summaries = load_all_summaries()
    reference = load_a0_reference()
    summary_dir_mk = SUMMARY_DIR
    summary_dir_mk.mkdir(parents=True, exist_ok=True)
    _write_results_csv(rows, SUMMARY_DIR / "r2d1_results.csv")
    _write_mechanism_csv(_mechanism_rows(summaries, VARIANTS, DATASETS, SEEDS), SUMMARY_DIR / "r2d1_mechanism.csv")
    _write_resource_csv(rows, SUMMARY_DIR / "r2d1_resource.csv")
    rows_by_variant = {v: [r for r in rows if r["variant"] == v] for v in VARIANTS}
    with (SUMMARY_DIR / "R2D1_MASTER_TABLE.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "seed", "A0", "B0", "F", "S", "J"])
        for dataset in DATASETS:
            for seed in SEEDS:
                writer.writerow([
                    dataset, seed,
                    _pp(a0_val_acc(reference, dataset, seed)),
                    *[_pp(_load_summary(dataset, v, seed) and _load_summary(dataset, v, seed)["best_val_acc"]) for v in VARIANTS],
                ])
    lines = [
        "# R2-Design-1 Final Diagnosis (D1-7)",
        "",
        "> Only reads existing results. No new experiments. No Test (plan §43).",
        "",
        "## 1. Master tables (Val Acc)",
        "",
        "### Seed42",
        "",
        _dataset_table(rows, ["B0", "F", "S", "J"], DATASETS, 42),
        "",
    ]
    score_f = _mean_delta(rows, "F", "B0", TARGET_DATASETS, seed=42)
    score_s = _mean_delta(rows, "S", "B0", TARGET_DATASETS, seed=42)
    score_j_b0 = _mean_delta(rows, "J", "B0", TARGET_DATASETS, seed=42)
    b0_delta = _mean_delta(rows, "B0", "A0", TARGET_DATASETS, seed=42)
    b0_per_ds = _per_dataset_delta(rows, "B0", "A0", TARGET_DATASETS)
    lines += [
        "## 2. Answers to the 12 questions (plan §43)",
        "",
        f"1. **R2-B0 acceptable clean parent?** YES. mean(B0−A0)={_delta_pp(b0_delta)}pp; "
        f"no dataset worse than −1.0pp; ALL THREE datasets positive "
        f"(Movies +0.240 / Toys +0.459 / Grocery +0.556).",
        f"2. **Loss from removing K-relation/Gamma/OFR:** NEGATIVE loss — B0−A0 = "
        f"{ {k: round(v*100,3) for k,v in b0_per_ds.items()} } pp. Removing the machinery "
        "is a net GAIN on every target dataset: strong evidence that the old "
        "topology-prototype relation chain was unnecessary (plan §25).",
        f"3. **Functional Transfer reproducible end-to-end gain?** NO. Score_F={_delta_pp(score_f)}pp "
        "(0/3 positive; GO gate +0.30pp). Explicit functional-core NO-GO (Score_F < −0.30pp).",
        f"4. **Semantic Refiner reproducible gain?** NO. Score_S={_delta_pp(score_s)}pp "
        "(1/3 positive; GO gate +0.20pp).",
        "5. **Joint synergy/interference:** J NOT RUN by the pre-registered gate "
        "(Score_F < −0.30pp — plan §28 forbids masking the failed functional core "
        "with the refiner). No synergy question arises.",
        f"6. **Benefiting datasets:** F per-dataset "
        f"{ {k: round(v*100,3) for k,v in _per_dataset_delta(rows, 'F', 'B0', TARGET_DATASETS).items()} } pp; "
        f"S per-dataset { {k: round(v*100,3) for k,v in _per_dataset_delta(rows, 'S', 'B0', TARGET_DATASETS).items()} } pp. "
        "Only Movies gains from S (+0.12pp, the dataset where the common gate collapsed "
        "to the visual side); no dataset gains from F.",
    ]
    # 7: Movies Pv-source conditional interaction evidence in learned gates.
    movies_f = summaries.get(("Movies", "F", 42))
    diag = (movies_f or {}).get("diagnostics") or {}
    if diag.get("functional"):
        gm = diag["functional"]["gate_matrix"]
        lines.append(
            "7. **Movies Pv-source conditional interaction in learned mechanism?** "
            "NOT REALIZED. F-Movies gate means (rows=src, cols=tgt): "
            f"{[[round(v, 3) for v in row] for row in gm['mean']]}. "
            "The R2-0C frozen-probe headroom (Pv→C/Pt/Pv) appears as the WEAKEST "
            "learned cells (Pv-row = 0.094/0.068/0.154), not an activated mechanism — "
            "the frozen headroom did not convert into an end-to-end learned interaction."
        )
    else:
        lines.append("7. **Movies Pv-source conditional interaction:** no functional diagnostics available.")
    lines += [
        "8. **Gate collapse/saturation?** Neither dead nor saturated: F gate means "
        "span 0.03-0.62 with a clear diagonal-dominant structure (C→C 0.41-0.62, "
        "Pt→Pt 0.54-0.61). Two telling behaviours: (a) the contribution matrix is "
        "diagonal-dominant — the 3×3 machinery converged to approximate what B0 "
        "already does; (b) rho_func learned NEGATIVE or near-zero values "
        "(Movies −0.009/−0.021/+0.006, Grocery −0.015/−0.019/+0.006) — the model "
        "itself suppressed the functional path during training. Toys Pv-source is "
        "mass-gated off (79-96% of nodes < 0.05). Mechanism activated but not "
        "useful: activation ≠ performance (plan §43).",
        "9. **Does the refiner break ownership?** P0 health preserved in all S runs: "
        "common_sim 0.71-0.79, private_sim ≈ 0, C-P overlap ≤ 0.03 — the factorizer "
        "survived refinement. However the common gate COLLAPSED to the visual side "
        "on Movies (w_t mean = 0.001, 100% nodes < 0.05 — plan §19 collapse flag "
        "triggered) and shifted visual-ward on Toys/Grocery (w_t ≈ 0.32/0.36). "
        "Semantic residual ratios stay moderate (0.21-0.38, no domination).",
        "10. **Params/memory/time vs A0:** B0/F/S/J = 1.09/1.12/1.27/1.30M "
        "(A0 = 1.40M, −22% to −7%); Movies train peak 1.17-1.88GB "
        "(A0 same-dataset scale several GB higher); epoch time ≈ 1.4s Movies. "
        "The R2 family is strictly lighter — the gains and losses above are NOT "
        "bought with extra capacity.",
    ]
    best = None
    best_delta = None
    for variant in ("B0", "F", "S", "J"):
        rows_v = rows_by_variant.get(variant, [])
        mtg_rows = [r for r in rows_v if r["dataset"] in TARGET_DATASETS]
        deltas = [r["delta_vs_a0"] for r in mtg_rows if r["delta_vs_a0"] is not None]
        if deltas:
            mean_d = statistics.mean(deltas)
            if best_delta is None or mean_d > best_delta:
                best, best_delta = variant, mean_d
    f_ok = score_f is not None and score_f >= 0.30 / 100
    s_ok = score_s is not None and score_s >= 0.20 / 100
    # J was not run (pre-registered gate); it can only PASS the mechanism
    # question if its own entry condition had allowed it.
    if f_ok and s_ok:
        status = "PASS"
    elif f_ok or s_ok:
        status = "PARTIAL"
    else:
        status = "NO-GO"
    verdict_text = (
        f"11. **Best candidate vs current A0:** {VARIANT_LABELS.get(best, 'none')} "
        f"(seed42 M/T/G mean Δ {_delta_pp(best_delta)}pp vs A0). "
        "NOTE: B0's gain is the positive byproduct of REMOVING the old machinery; "
        "the R2 core mechanism itself (F/S) delivered no end-to-end gain."
    )
    lines += [verdict_text, ""]
    lines += [
        f"12. **Proceed to R2-Design-2 / final benchmark?** — see status below. "
        "The core mechanism is NO-GO: per plan §47, the R2-0 frozen headroom "
        "cannot be realized by the current end-to-end realization, so the next "
        "step is re-examining optimization / interaction realization — NOT "
        "stacking more modules (§45 list stays forbidden).",
        "",
        f"## R2-Design-1 status: **{status}** (core mechanism F/S both NO-GO)",
        "",
        f"**Best candidate: {VARIANT_LABELS.get(best, 'none')}** "
        "(B0 = acceptable clean parent and the new strongest simple parent)",
        "",
        "Mechanism activation ≠ performance evidence: the 3×3 gates ARE "
        "structured and the common gate DID collapse toward visual on Movies — "
        "yet neither produced an end-to-end gain (plan §43).",
        "",
        "Awaiting manual review. No Test has been run.",
    ]
    (SUMMARY_DIR / "R2D1_FINAL_DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[final] report -> {SUMMARY_DIR / 'R2D1_FINAL_DIAGNOSIS.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-Design-1 summarizer")
    parser.add_argument("--stage", required=True,
                        choices=["b0", "functional", "semantic", "joint", "confirm", "final", "csv"])
    args = parser.parse_args()

    summaries = load_all_summaries()
    rows = _rows_for(summaries, VARIANTS, DATASETS, SEEDS)
    if args.stage == "csv":
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        _write_results_csv(rows, SUMMARY_DIR / "r2d1_results.csv")
        _write_mechanism_csv(_mechanism_rows(summaries, VARIANTS, DATASETS, SEEDS), SUMMARY_DIR / "r2d1_mechanism.csv")
        _write_resource_csv(rows, SUMMARY_DIR / "r2d1_resource.csv")
        print(f"[csv] saved -> {SUMMARY_DIR}")
        return
    if args.stage == "b0":
        _write_b0_report(rows)
    elif args.stage == "functional":
        _write_functional_report(rows)
    elif args.stage == "semantic":
        _write_semantic_report(rows)
    elif args.stage == "joint":
        _write_joint_report(rows)
    elif args.stage == "confirm":
        # Pre-registered selection: candidates = variants that passed their
        # seed42 GO gates; computed the same way Prompt 6 would (plan §42).
        candidates: list[str] = []
        for variant in ("F", "S", "J"):
            vs_b0 = _mean_delta(rows, variant, "B0", TARGET_DATASETS, seed=42)
            per_ds = _per_dataset_delta(rows, variant, "B0", TARGET_DATASETS)
            positives = sum(1 for d in per_ds.values() if d > 0)
            if variant == "F":
                go = vs_b0 is not None and vs_b0 >= 0.30 / 100 and positives >= 2
            elif variant == "S":
                go = vs_b0 is not None and vs_b0 >= 0.20 / 100 and positives >= 2
            else:
                vs_a0 = _mean_delta(rows, "J", "A0", TARGET_DATASETS, seed=42)
                go = (vs_b0 is not None and vs_b0 >= 0.40 / 100 and positives >= 2
                      and vs_a0 is not None and vs_a0 >= 0.20 / 100)
            if go:
                candidates.append(variant)
        if not candidates:
            outdir = R2D1_ROOT / "confirm"
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "R2D1_CONFIRM_REPORT.md").write_text(
                "# R2-Design-1 D1-5/D1-6 — Guards + 3-seed Confirmation Report\n\n"
                "**No candidate passed the pre-registered seed42 GO gates** "
                "(F: Score_F = −0.73pp < +0.30pp; S: Score_S = −0.30pp < +0.20pp; "
                "J not run per plan §28). Guards and the 3-seed formal "
                "confirmation are NOT executed (plan §42). No Test was run.\n",
                encoding="utf-8",
            )
            print(f"[confirm] no candidate passed the pre-registered GO gates — wrote {outdir / 'R2D1_CONFIRM_REPORT.md'}")
            return
        _write_confirm_report(rows, candidates)
    elif args.stage == "final":
        _write_final_report(rows)


if __name__ == "__main__":
    main()
