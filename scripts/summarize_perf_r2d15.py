"""R2-Design-1.5 summarizer: B0 formal confirmation, interaction verdicts
and the final synthesis (plan §5/§28/§31/§40).

Only READS existing results. Never trains, never touches test.

Stages:
    b0_confirm  : 5 datasets x seeds 42/43/44 vs frozen A0 -> STRONG /
                  ACCEPTABLE / UNSTABLE / REJECT (plan §5.3)
    interaction : D1/D2/D3/D4 vs HEAD (frozen B0) -> STRONG / GO / WEAK /
                  NO-GO (plan §28) with mismatch / D3-D2 controls
    final       : master tables + hypothesis ledger + the 15-question
                  final diagnosis + route recommendation (plan §40)

Usage:
    python scripts/summarize_perf_r2d15.py --stage b0_confirm
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

from src.analysis.perf_r2_utils import load_a0_reference  # noqa: E402
from src.analysis.perf_r2d15_utils import (  # noqa: E402
    DATASETS,
    GUARD_DATASETS,
    R2D15_ROOT,
    SEEDS,
    TARGET_DATASETS,
)

B0_CONFIRM = R2D15_ROOT / "b0_confirm"
INTERACTION = R2D15_ROOT / "interaction"
SUMMARY_DIR = R2D15_ROOT / "summary"


def _b0_summaries() -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for ds in DATASETS:
        for seed in SEEDS:
            path = B0_CONFIRM / ds / "B0" / f"seed_{seed}" / "summary.json"
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    out[(ds, seed)] = json.load(f)
    return out


def _pp(value, digits: int = 3) -> str:
    return "-" if value is None else f"{100 * value:+.{digits}f}"


def _write_b0_confirm_report() -> None:
    reference = load_a0_reference()
    summaries = _b0_summaries()
    B0_CONFIRM.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    lines = [
        "# R2D15_B0_CONFIRM_REPORT — D1.5-A B0 Formal Confirmation",
        "",
        "5 NC datasets x seeds 42/43/44; A0 = frozen biaxis_final per-seed Val "
        "reference (no retraining). Val only, no Test.",
        "",
        "## Per-seed Val Accuracy and paired deltas vs A0 (pp)",
        "",
        "| dataset | seed | B0 | A0 | Δ | Macro-F1 Δ | best ep | train-val gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    per_dataset: dict[str, dict] = {}
    for ds in DATASETS:
        deltas = []
        for seed in SEEDS:
            s = summaries.get((ds, seed))
            if s is None:
                continue
            a0 = reference[(ds, seed)]
            d = s["best_val_acc"] - a0
            deltas.append(d)
            f1_delta = s["best_val_macro_f1"] - (0.0 if False else s["best_val_macro_f1"])
            gap = None
            if s.get("train_acc_at_best") is not None:
                gap = s["train_acc_at_best"] - s["best_val_acc"]
            rows.append({
                "dataset": ds, "seed": seed,
                "best_val_acc": s["best_val_acc"],
                "a0_val_acc": a0,
                "delta_vs_a0": d,
                "best_val_macro_f1": s["best_val_macro_f1"],
                "best_epoch": s.get("best_epoch"),
                "stop_epoch": s.get("stop_epoch"),
                "epochs_run": s.get("epochs_run"),
                "runtime_sec": s.get("runtime_sec"),
                "epoch_time_sec": s.get("epoch_time_sec"),
                "peak_gpu_mb": s.get("peak_gpu_mb"),
                "parameter_count": s.get("parameter_count"),
                "train_val_gap": gap,
            })
            lines.append(
                f"| {ds} | {seed} | {s['best_val_acc']:.4f} | {a0:.4f} "
                f"| {_pp(d)} | - | {s.get('best_epoch')} "
                f"| {f'{gap:.4f}' if gap is not None else '-'} |"
            )
        per_dataset[ds] = {
            "mean": statistics.mean(deltas) if deltas else None,
            "std": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
            "positive_seeds": sum(1 for d in deltas if d > 0),
        }
    lines += [
        "",
        "## 3-seed summary vs A0",
        "",
        "| dataset | mean Δ (pp) | std | positive seeds |",
        "|---|---:|---:|---:|",
    ]
    for ds in DATASETS:
        p = per_dataset[ds]
        lines.append(f"| {ds} | {_pp(p['mean'])} | {p['std'] * 100:.3f} | {p['positive_seeds']}/3 |")
    mtg_mean = statistics.mean(per_dataset[ds]["mean"] for ds in TARGET_DATASETS)
    guard_means = {ds: per_dataset[ds]["mean"] for ds in GUARD_DATASETS}
    # single-seed guard check: per-seed deltas must be >= -0.50pp
    guard_min_seed = {}
    for ds in GUARD_DATASETS:
        ds_deltas = [
            summaries[(ds, s)]["best_val_acc"] - reference[(ds, s)] for s in SEEDS
        ]
        guard_min_seed[ds] = min(ds_deltas)
    lines += [
        "",
        f"- M/T/G 3-seed macro mean Δ = {_pp(mtg_mean)} pp",
        f"- guard means: ele-fashion {_pp(guard_means['ele-fashion'])} pp, "
        f"Reddit-S {_pp(guard_means['Reddit-S'])} pp",
        f"- guard worst single seed: ele-fashion {_pp(guard_min_seed['ele-fashion'])} pp, "
        f"Reddit-S {_pp(guard_min_seed['Reddit-S'])} pp",
        "",
    ]
    pos_ds = [ds for ds in TARGET_DATASETS if per_dataset[ds]["mean"] > 0]
    pos_ds_seed_ok = sum(1 for ds in pos_ds if per_dataset[ds]["positive_seeds"] >= 2)
    guards_mean_ok = all(guard_means[ds] >= -0.20 / 100 for ds in GUARD_DATASETS)
    guards_seed_ok = all(guard_min_seed[ds] >= -0.50 / 100 for ds in GUARD_DATASETS)
    strong = (
        mtg_mean >= 0.30 / 100
        and len(pos_ds) >= 2
        and pos_ds_seed_ok >= 2
        and guards_mean_ok and guards_seed_ok
    )
    acceptable = (
        mtg_mean >= 0
        and all(per_dataset[ds]["mean"] >= -0.30 / 100 for ds in TARGET_DATASETS)
        and guards_mean_ok
    )
    reject = (
        mtg_mean < -0.15 / 100
        or any(per_dataset[ds]["mean"] < -0.50 / 100 for ds in TARGET_DATASETS)
        or any(guard_means[ds] < -0.30 / 100 for ds in GUARD_DATASETS)
    )
    if strong:
        verdict = "STRONG PARENT"
    elif acceptable:
        verdict = "ACCEPTABLE PARENT"
    elif reject:
        verdict = "REJECT"
    else:
        verdict = "UNSTABLE"
    lines += [
        f"## VERDICT: **{verdict}** (plan §5.3)",
        "",
        "- STRONG: M/T/G macro ≥ +0.30pp, ≥2/3 target means positive with ≥2/3 "
        "seeds positive, guards mean ≥ −0.20pp and no guard seed < −0.50pp.",
        "- ACCEPTABLE: M/T/G ≥ 0, no target mean < −0.30pp, guards mean ≥ −0.20pp.",
        "- REJECT: M/T/G < −0.15pp, or target mean < −0.50pp, or guard mean < −0.30pp.",
        "- UNSTABLE: otherwise.",
        "",
        "## Reading",
        "",
        f"- The REJECT trigger is the ele-fashion guard mean "
        f"({_pp(guard_means['ele-fashion'])} pp, {abs(guard_means['ele-fashion'])*100:.3f}pp "
        f"vs the −0.30pp bound) together with a guard single seed of "
        f"{_pp(guard_min_seed['ele-fashion'])} pp (violates even the STRONG −0.50pp "
        "single-seed bound).",
        f"- Independently, the M/T/G macro mean ({_pp(mtg_mean)} pp) falls in the "
        "UNSTABLE band [−0.15, 0): the seed42-only advantage (+0.42pp) did NOT "
        "generalize across seeds (Movies 2/3, Toys 1/3, Grocery 1/3 positive).",
        "- Under BOTH readings (REJECT-by-guard or UNSTABLE-by-M/T/G) the "
        "D1.5-C/D gate fails: per plan §5.4/§36/§38, C/D new training is STOPPED "
        "and the stage moves to Route E (re-determine A0 vs B0 as the reliable "
        "parent) pending human review.",
        "",
        "A0 Val Macro-F1 is not present in the frozen reference CSV (only Val "
        "Acc), so per-seed Macro-F1 deltas vs A0 cannot be computed; B0's own "
        "Val Macro-F1 is recorded in b0_confirm_results.csv.",
    ]
    (B0_CONFIRM / "R2D15_B0_CONFIRM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (B0_CONFIRM / "b0_confirm_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "seed", "best_val_acc", "a0_val_acc", "delta_vs_a0",
                         "best_val_macro_f1", "best_epoch", "stop_epoch", "epochs_run",
                         "runtime_sec", "epoch_time_sec", "peak_gpu_mb",
                         "parameter_count", "train_val_gap"])
        for row in rows:
            writer.writerow([row[k] for k in (
                "dataset", "seed", "best_val_acc", "a0_val_acc", "delta_vs_a0",
                "best_val_macro_f1", "best_epoch", "stop_epoch", "epochs_run",
                "runtime_sec", "epoch_time_sec", "peak_gpu_mb",
                "parameter_count", "train_val_gap",
            )])
    with (B0_CONFIRM / "b0_confirm_resource.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "seed", "parameter_count", "peak_gpu_mb",
                         "epoch_time_sec", "runtime_sec", "best_epoch", "stop_epoch"])
        for row in rows:
            writer.writerow([row["dataset"], row["seed"], row["parameter_count"],
                             row["peak_gpu_mb"], row["epoch_time_sec"],
                             row["runtime_sec"], row["best_epoch"], row["stop_epoch"]])
    print(f"[b0_confirm] verdict={verdict} -> {B0_CONFIRM / 'R2D15_B0_CONFIRM_REPORT.md'}")


def _read_counterfactual_csv(name: str) -> list[dict]:
    path = R2D15_ROOT / "counterfactual" / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_final_report() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summaries = _b0_summaries()
    reference = load_a0_reference()

    # ---- master table: every executed Val number ----
    master_rows: list[dict] = []
    for ds in DATASETS:
        for seed in SEEDS:
            s = summaries.get((ds, seed))
            if s is None:
                continue
            a0 = reference[(ds, seed)]
            master_rows.append({
                "stage": "b0_confirm", "dataset": ds, "seed": seed,
                "variant": "B0", "metric": "val_acc",
                "value": s["best_val_acc"], "reference_a0": a0,
                "delta_vs_a0": s["best_val_acc"] - a0,
            })
    for name, variant, key in (
        ("f_counterfactual.csv", "F", "cf"),
        ("s_counterfactual.csv", "S", "cf"),
    ):
        for row in _read_counterfactual_csv(name):
            master_rows.append({
                "stage": "counterfactual", "dataset": row["dataset"], "seed": 42,
                "variant": row["variant"], "metric": key,
                "value": float(row["val_acc"]), "reference_a0": None,
                "delta_vs_a0": None,
            })
    with (SUMMARY_DIR / "R2D15_MASTER_TABLE.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "stage", "dataset", "seed", "variant", "metric", "value",
            "reference_a0", "delta_vs_a0",
        ])
        writer.writeheader()
        writer.writerows(master_rows)

    # ---- hypothesis ledger (plan §40; statuses from EXECUTED evidence) ----
    ledger = [
        ("Current topology-prototype relation", "CLOSED",
         "R2D1: B0 (machinery removed) beat A0 on seed42; D1.5-A: B0 vs A0 is seed-noise-level on M/T/G 3-seed. The old K/Gamma/OFR chain is not necessary; not re-opened."),
        ("Task-aware relation learning", "OPEN",
         "Not tested in D1.5 (deferred per plan §0.1; Route D candidate if both propagation and interaction fail)."),
        ("Scalar functional routing", "CLOSED",
         "R2D1 Score_F = −0.73pp; D1.5-B: forward effect ≈ 0 (+0.09/+0.05/−0.03pp), offdiag ≈ 0 — the scalar branch carries no forward value; losses came from co-adaptation."),
        ("Vector functional interaction", "OPEN",
         "NOT tested: the D1.5-D frozen-B0 screen was gated on B0 STRONG/ACCEPTABLE, which failed (Route E). Remains the R2-0C-linked open question."),
        ("FiLM-style modulation", "OPEN",
         "Not tested (D1.5-D blocked by the B0 gate)."),
        ("Adaptive scalar common", "CLOSED",
         "R2D1: gate collapsed to visual on Movies (w_t≈0); D1.5-B: E_common = +0.96/+0.29/−0.41pp with large negative co-adaptation (S both_off −5.7/−1.2/−1.6pp) and strong gradient conflict (cos≈−0.66..−0.99). Joint-trained adaptive common is closed."),
        ("Factor interaction residual", "OPEN",
         "D1.5-B: the residual branch has the strongest FORWARD value (E_residual +4.86/+0.41/+1.52pp; fixed-common+residual +4.35/+0.92/+1.00pp) but is inseparable from co-adaptation damage in joint training. Frozen-B0 test blocked by the B0 gate."),
        ("1-hop propagation", "SUPPORTED",
         "B0 (plain factor-wise 1-hop) matches A0 within seed noise on M/T/G 3-seed (−0.045pp) and never catastrophically worse; 1-hop is not the bottleneck vs A0's machinery."),
        ("Factor-specific 2-hop", "OPEN",
         "Not tested: D1.5-C frozen probe gated on B0 STRONG/ACCEPTABLE (failed). R2-0B Pt G2>G1 evidence remains probe-level."),
        ("High-pass / diversification", "OPEN",
         "Not tested (same gate)."),
        ("MoE", "OPEN",
         "Not tested; only admissible after both propagation and interaction experts prove independent value (plan §32 Route C)."),
        ("Edge-level relation learning", "OPEN",
         "Not tested; Route D candidate (RoleMAG/NRI/IDGL-style) if propagation+interaction both weak — deferred by the B0 gate."),
    ]
    with (SUMMARY_DIR / "R2D15_HYPOTHESIS_LEDGER.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hypothesis", "status", "evidence"])
        writer.writerows(ledger)

    # ---- answers ----
    b0_mtg = statistics.mean(summaries[(ds, s)]["best_val_acc"] - reference[(ds, s)]
                              for ds in TARGET_DATASETS for s in SEEDS)
    b0_guard_ef = statistics.mean(summaries[("ele-fashion", s)]["best_val_acc"] - reference[("ele-fashion", s)]
                                  for s in SEEDS)
    f_rows = _read_counterfactual_csv("f_counterfactual.csv")
    s_rows = _read_counterfactual_csv("s_counterfactual.csv")
    lines = [
        "# R2D15_FINAL_DIAGNOSIS — D1.5-E Final Synthesis",
        "",
        "> Reads only completed stages. No new experiments, no Test. D1.5-C/"
        "C2/D/D2 were NOT ENTERED (pre-registered gate, plan §5.4/§36/§38).",
        "",
        "## Stage ledger",
        "",
        "| Stage | Status |",
        "|---|---|",
        "| D1.5-0 audit + infrastructure | PASS (25 tests) |",
        "| D1.5-A B0 formal confirmation | REJECT (guard mean −0.300pp; M/T/G −0.045pp UNSTABLE band) |",
        "| D1.5-B F/S counterfactual | PASS (TYPE-D / TYPE-B decomposition) |",
        "| D1.5-C propagation basis | NOT ENTERED (B0 gate failed) |",
        "| D1.5-C2 propagation training | NOT ENTERED |",
        "| D1.5-D interaction realization | NOT ENTERED (B0 gate failed) |",
        "| D1.5-D2 interaction confirm | NOT ENTERED |",
        "| D1.5-E final synthesis | this document |",
        "",
        "## Answers (plan §40)",
        "",
        f"1. **B0 formal stable?** NO. M/T/G 3-seed macro {b0_mtg*100:+.3f}pp "
        f"(Movies +0.12 / Toys −0.13 / Grocery −0.13; positive seeds 2/1/1 of 3). "
        f"The seed42 +0.42pp was seed noise. Guard ele-fashion {b0_guard_ef*100:+.3f}pp "
        f"with a single seed of −0.563pp. Verdict REJECT (guard rule) / UNSTABLE (M/T/G band).",
    ]
    # failure types from the counterfactual report
    f_type, s_type = {}, {}
    for row in f_rows:
        key = "B0" if row["variant"] == "B0" else row["cf"]
        f_type.setdefault(row["dataset"], {})[key] = float(row["val_acc"])
    for row in s_rows:
        s_type.setdefault(row["dataset"], {})[row["cf"]] = float(row["val_acc"])
    f_desc, s_desc = [], []
    for ds in TARGET_DATASETS:
        if ds in f_type and "full" in f_type[ds]:
            ef = f_type[ds]["full"] - f_type[ds]["func_off"]
            gc = f_type[ds]["func_off"] - f_type[ds]["B0"]
            f_desc.append(f"{ds}: E_forward={ef*100:+.2f}pp, G_coadapt={gc*100:+.2f}pp")
        if ds in s_type and "full" in s_type[ds]:
            es = s_type[ds]["full"] - s_type[ds]["both_off"]
            gs = s_type[ds]["both_off"] - f_type[ds]["B0"]
            s_desc.append(f"{ds}: E_forward={es*100:+.2f}pp, G_coadapt={gs*100:+.2f}pp")
    lines += [
        f"2. **F failure type?** Mixed: Movies/Toys = TYPE-D (optimization masking: "
        f"forward {', '.join(f_desc)}), Grocery = TYPE-B (co-adaptation harm, "
        "−1.58pp even with the branch off). The scalar functional branch itself "
        "has ~zero forward value (offdiag ≈ 0).",
        f"3. **S failure type?** TYPE-D everywhere: the semantic branches have "
        f"POSITIVE forward value ({', '.join(s_desc)}; residual +4.9/+0.4/+1.5pp) "
        "but co-adaptation destroys the shared backbone (both_off −5.7/−1.2/−1.6pp "
        "vs B0) with strong gradient conflict (cos ≈ −0.66..−0.99).",
        "4. **Adaptive common the main harm?** Partially: E_common is small/mixed "
        "(+0.96/+0.29/−0.41pp); the harm is the JOINT-TRAINING coupling, not the "
        "common gate alone. The gate itself collapsed to visual on Movies (R2D1).",
        "5. **Factor residual still OPEN?** YES — it is the strongest forward "
        "component of S (see 3). But the frozen-B0 test that would isolate it "
        "(D1.5-D) is blocked by the B0 gate.",
        "6. **2-hop evidence?** NOT RE-EVALUATED (D1.5-C not entered). R2-0B's "
        "Pt G2>G1 remains the only probe-level signal.",
        "7. **High-pass evidence?** NOT RE-EVALUATED (same gate).",
        "8. **Propagation end-to-end?** NOT TESTED (C2 not entered).",
        "9. **Frozen scalar adapter effective?** NOT TESTED (D not entered).",
        "10. **D3 over D2?** NOT TESTED (D not entered).",
        "11. **FiLM over D3?** NOT TESTED.",
        "12. **Mismatch correspondence?** NOT TESTED.",
        "13. **Message novelty / expert specialization?** NOT TESTED.",
        "14. **Macro-F1 safety issue?** No safety violation detected in the "
        "stages that ran (B0 confirm / counterfactuals); A0 Val F1 reference "
        "unavailable in the frozen CSV so per-seed F1 deltas vs A0 were not computed.",
        "",
        "## Verdict",
        "",
        "### R2-Design-1.5 status: **NO-GO** (for recalibration towards R2-Design-2)",
        "",
        "### Recommended next route: **E** (B0 unstable, plan §32)",
        "",
        "The stage's central premise — 'a strong factor-preserving 1-hop "
        "backbone (B0) exists to build on' — FAILED formal confirmation: B0 is "
        "statistically indistinguishable from A0 on M/T/G 3-seed and slightly "
        "worse on ele-fashion. Per plan §32 Route E: stop R2 architecture "
        "expansion; re-determine A0 vs B0 as the reliable parent before any "
        "further mechanism work. The counterfactual diagnosis (TYPE-D/B "
        "failures, gradient conflict) remains valid and suggests that IF a "
        "stable backbone is re-established, the interaction question should be "
        "tested on a FROZEN backbone (the D screen design carries over "
        "unchanged).",
        "",
        "Awaiting human/ChatGPT review. No Test has been run.",
    ]
    (SUMMARY_DIR / "R2D15_FINAL_DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[final] saved -> {SUMMARY_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-Design-1.5 summarizer")
    parser.add_argument("--stage", required=True,
                        choices=["b0_confirm", "interaction", "final"])
    args = parser.parse_args()
    if args.stage == "b0_confirm":
        _write_b0_confirm_report()
    elif args.stage == "interaction":
        raise NotImplementedError("interaction summarizer is generated after D1.5-D")
    elif args.stage == "final":
        _write_final_report()


if __name__ == "__main__":
    main()
