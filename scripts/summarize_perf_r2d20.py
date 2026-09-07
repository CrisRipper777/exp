"""R2-Design-2.0 summarizer (plan §14-§23, §37-§41).

Stages:
    m1_screen : M1 vs HEAD seed42 (GO ≥ +0.30pp, ≥2/3 pos, F1 safe) +
                PROBE-CONSISTENT check (alpha_Pt > 0 on ≥2/3 datasets and
                alpha_Pt mean > alpha_C/Pv)
    m1_confirm: guards (incl. Reddit-S NEGATIVE-CONTROL) + 3-seed formal;
                Mechanism GO vs B0 + Final GO vs A0
    m2        : M2 vs HEAD / M2 vs M1; material advantage ≥ +0.10pp
    final     : master table + hypothesis ledger + 13 questions + route

Only READS existing results. Never trains, never touches test.

Usage:
    python scripts/summarize_perf_r2d20.py --stage m1_screen
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
from src.analysis.perf_r2d16_utils import R2D16_ROOT  # noqa: E402

R2D20_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d20"
TARGET_DATASETS = ["Movies", "Toys", "Grocery"]
GUARD_DATASETS = ["ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]


def _load(root: Path, mode: str, variants, datasets, seeds) -> dict:
    out = {}
    for ds in datasets:
        for v in variants:
            for s in seeds:
                p = root / ds / v / f"seed_{s}" / "summary.json"
                if p.exists():
                    with p.open(encoding="utf-8") as f:
                        out[(ds, v, s)] = json.load(f)
    return out


def _pp(v, digits: int = 3) -> str:
    return "-" if v is None else f"{100 * v:+.{digits}f}"


def _m1_screen_report() -> None:
    root = R2D20_ROOT / "m1_screen"
    summaries = _load(root, "m1", ("HEAD", "M1"), TARGET_DATASETS, [42])
    rows = []
    for (ds, v, s), summary in summaries.items():
        h = summaries.get((ds, "HEAD", s))
        rows.append({
            "dataset": ds, "variant": v, "seed": s,
            "val_acc": summary["best_val_acc"],
            "val_macro_f1": summary["best_val_macro_f1"],
            "delta_acc_vs_head": (summary["best_val_acc"] - h["best_val_acc"]) if h else None,
            "delta_f1_vs_head": (summary["best_val_macro_f1"] - h["best_val_macro_f1"]) if h else None,
            "alpha": summary["scale"].get("alpha"),
            "best_epoch": summary["best_epoch"],
            "alpha_instability_warning": summary.get("alpha_instability_warning"),
        })
    with (root / "m1_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    gains, positives, f1_warns = [], 0, 0
    lines = [
        "# R2D20_M1_SCREEN_REPORT — M1 Seed42 Screen (frozen B0, M/T/G)",
        "",
        "M1 = H1 + alpha_f(H2−H1), 3 direct scalars on the frozen B0 parent; "
        "HEAD = same frozen B0 + same classifier init with alpha fixed 0. "
        "AdamW lr1e-3 (alpha wd=0, classifier wd=1e-4), 300ep/patience30. Val only.",
        "",
        "## Results (Val Acc; Δ vs HEAD in pp)",
        "",
        "| dataset | HEAD | M1 | Δ Acc | Δ F1 | α_C | α_Pt | α_Pv | |α|>2 warn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    alpha_pt = []
    for ds in TARGET_DATASETS:
        h = summaries.get((ds, "HEAD", 42))
        m = summaries.get((ds, "M1", 42))
        if not h or not m:
            lines.append(f"| {ds} | MISSING | - | - | - | - | - | - | - |")
            continue
        d_acc = m["best_val_acc"] - h["best_val_acc"]
        d_f1 = m["best_val_macro_f1"] - h["best_val_macro_f1"]
        gains.append(d_acc)
        positives += d_acc > 0
        if d_f1 < -0.50 / 100:
            f1_warns += 1
        a = m["scale"]["alpha"]
        alpha_pt.append((ds, a[1]))
        lines.append(
            f"| {ds} | {h['best_val_acc']:.5f} | {m['best_val_acc']:.5f} "
            f"| {_pp(d_acc)} | {_pp(d_f1)} | {a[0]:+.4f} | {a[1]:+.4f} | {a[2]:+.4f} "
            f"| {'YES' if m.get('alpha_instability_warning') else 'no'} |"
        )
    lines += ["", "## Verdict (plan §14)", ""]
    if len(gains) < 3:
        lines.append("MISSING runs — verdict deferred.")
    else:
        gain = statistics.mean(gains)
        strong = gain >= 0.50 / 100 and positives >= 2 and f1_warns == 0
        go = gain >= 0.30 / 100 and positives >= 2 and f1_warns == 0
        verdict = "STRONG" if strong else ("GO" if go else "NO-GO")
        lines += [
            f"- Gain(M1−HEAD) = {_pp(gain)} pp (positive {positives}/3, "
            f"F1 warnings {f1_warns})",
            f"- **M1 seed42 verdict: {verdict}** (GO ≥ +0.30pp, ≥2/3 positive, "
            "no F1 < −0.50pp warning; STRONG ≥ +0.50pp)",
            "",
            "## Mechanism consistency (plan §15, diagnostic only)",
            "",
        ]
        pt_pos = sum(1 for _, a in alpha_pt if a > 0)
        pt_mean = statistics.mean(a for _, a in alpha_pt)
        others = [m["scale"]["alpha"][i] for (_, m) in
                  [(ds, summaries.get((ds, "M1", 42))) for ds in TARGET_DATASETS]
                  if m for i in (0, 2)]
        pt_above = pt_mean > statistics.mean(others)
        consistent = pt_pos >= 2 and pt_above
        lines.append(
            f"- alpha_Pt > 0 on {pt_pos}/3 datasets (need ≥2/3); "
            f"mean alpha_Pt = {pt_mean:+.4f} vs mean alpha_C/Pv = "
            f"{statistics.mean(others):+.4f} → "
            f"{'**PROBE-CONSISTENT**' if consistent else 'probe-inconsistent (interpret carefully)'}"
        )
        if verdict == "GO" or verdict == "STRONG":
            lines += [
                "",
                "**Next: M1 guards + formal confirmation (plan §38).**",
            ]
        else:
            lines += [
                "",
                "**M1 NO-GO → per plan §20: M2 is NOT run; stop and wait for synthesis.**",
            ]
    (root / "R2D20_M1_SCREEN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[m1_screen] saved -> {root / 'R2D20_M1_SCREEN_REPORT.md'}")


def _m1_confirm_report() -> None:
    guards_root = R2D20_ROOT / "m1_guards"
    confirm_root = R2D20_ROOT / "m1_confirm"
    g_summaries = _load(guards_root, "m1", ("HEAD", "M1"), GUARD_DATASETS, [42])
    c_summaries = _load(confirm_root, "m1", ("HEAD", "M1"), TARGET_DATASETS, SEEDS)
    reference = load_a0_reference()
    lines = [
        "# R2D20_M1_CONFIRM_REPORT — M1 Guards + Formal Confirmation",
        "",
        "Frozen B0 per seed (b0_confirm); M1 vs HEAD (shared classifier init); "
        "A0 = frozen formal Val Acc reference. Val only.",
        "",
        "## Step A — Guards (seed42, vs HEAD)",
        "",
        "| guard | Δ Acc (pp) | Δ F1 (pp) | Acc ≥ −0.20 | F1 ≥ −0.50 | α_Pt (guard) |",
        "|---|---:|---:|---|---|---:|",
    ]
    guard_safe = True
    reddit_alpha_pt = None
    mtg_alpha_pt = None
    for ds in GUARD_DATASETS:
        h = g_summaries.get((ds, "HEAD", 42))
        m = g_summaries.get((ds, "M1", 42))
        if not h or not m:
            lines.append(f"| {ds} | MISSING | - | no | no | - |")
            guard_safe = False
            continue
        d_acc = m["best_val_acc"] - h["best_val_acc"]
        d_f1 = m["best_val_macro_f1"] - h["best_val_macro_f1"]
        ok_a, ok_f = d_acc >= -0.20 / 100, d_f1 >= -0.50 / 100
        guard_safe = guard_safe and ok_a and ok_f
        a_pt = m["scale"]["alpha"][1]
        if ds == "Reddit-S":
            reddit_alpha_pt = a_pt
        lines.append(f"| {ds} | {_pp(d_acc)} | {_pp(d_f1)} | "
                     f"{'yes' if ok_a else 'no'} | {'yes' if ok_f else 'no'} | {a_pt:+.4f} |")
    screen_pt = [
        json.load(open(R2D20_ROOT / "m1_screen" / ds / "M1" / "seed_42" / "summary.json"))["scale"]["alpha"][1]
        for ds in TARGET_DATASETS
    ]
    mtg_alpha_pt = statistics.mean(screen_pt)
    lines += [
        "",
        f"**Guards: {'SAFE' if guard_safe else 'FAILED'}**",
        "",
        "## Reddit-S negative control (plan §16)",
        "",
        f"- alpha_Pt(Reddit-S) = {reddit_alpha_pt:+.4f} vs alpha_Pt(M/T/G seed42 mean) = "
        f"{mtg_alpha_pt:+.4f}",
        f"- NEGATIVE-CONTROL {'**PASS**' if reddit_alpha_pt is not None and reddit_alpha_pt <= mtg_alpha_pt else 'FAIL'} "
        "(Reddit-S alpha_Pt should fall back to ≈0/negative; the frozen probe "
        "shows H2−H1 < 0 there)",
        "",
        "## Step B — Formal (M/T/G × 42/43/44)",
        "",
        "| dataset | s42 | s43 | s44 | 3-seed mean Δ | pos seeds | M1 3-seed mean | A0 3-seed mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    mtg_deltas = []
    for ds in TARGET_DATASETS:
        deltas, m1_vals, a0_vals = [], [], []
        for s in SEEDS:
            h = c_summaries.get((ds, "HEAD", s))
            m = c_summaries.get((ds, "M1", s))
            if m and h:
                deltas.append(m["best_val_acc"] - h["best_val_acc"])
                m1_vals.append(m["best_val_acc"])
            else:
                deltas.append(None)
            if (ds, s) in reference:
                a0_vals.append(reference[(ds, s)])
        if all(d is None for d in deltas):
            lines.append(f"| {ds} | MISSING | - | - | - | - | - | - |")
            continue
        mean_d = statistics.mean(d for d in deltas if d is not None)
        pos = sum(1 for d in deltas if d is not None and d > 0)
        mtg_deltas.append((ds, mean_d, pos))
        lines.append(
            f"| {ds} | {_pp(deltas[0])} | {_pp(deltas[1])} | {_pp(deltas[2])} "
            f"| {_pp(mean_d)} | {pos}/3 | {statistics.mean(m1_vals):.5f} | "
            f"{statistics.mean(a0_vals):.5f} |"
        )
    lines += ["", "## Verdicts (plan §19)", ""]
    if len(mtg_deltas) < 3:
        lines.append("MISSING formal runs — deferred.")
    else:
        macro_vs_head = statistics.mean(d for _, d, _ in mtg_deltas)
        pos_ds = sum(1 for _, d, _ in mtg_deltas if d > 0)
        pos_ds_seed_ok = sum(1 for _, d, p in mtg_deltas if d > 0 and p >= 2)
        mech_go = (macro_vs_head >= 0.30 / 100 and pos_ds >= 2
                   and pos_ds_seed_ok >= 2 and guard_safe)
        # Final GO vs A0: M1 val (frozen-parent HEAD-scale run) vs formal A0
        deltas_vs_a0 = []
        for ds in TARGET_DATASETS:
            for s in SEEDS:
                m = c_summaries.get((ds, "M1", s))
                if m and (ds, s) in reference:
                    deltas_vs_a0.append(m["best_val_acc"] - reference[(ds, s)])
        per_ds_a0 = {}
        for ds in TARGET_DATASETS:
            vals = [c_summaries[(ds, "M1", s)]["best_val_acc"] - reference[(ds, s)]
                    for s in SEEDS if (ds, "M1", s) in c_summaries and (ds, s) in reference]
            if vals:
                per_ds_a0[ds] = statistics.mean(vals)
        macro_vs_a0 = statistics.mean(per_ds_a0.values()) if per_ds_a0 else None
        pos_a0 = sum(1 for v in per_ds_a0.values() if v > 0)
        lines += [
            f"- Mechanism GO vs B0 scaffold: macro(M1−HEAD) = {_pp(macro_vs_head)} pp, "
            f"positive datasets {pos_ds}/3 (≥2/3 seeds: {pos_ds_seed_ok}), guards "
            f"{'SAFE' if guard_safe else 'FAILED'} → **{'GO' if mech_go else 'NO-GO'}**",
            f"- Final GO vs A0: macro(M1−A0) = {_pp(macro_vs_a0)} pp "
            f"({ {k: round(v*100, 2) for k, v in per_ds_a0.items()} } pp), "
            f"positive datasets {pos_a0}/3 → "
            f"**{'GO' if macro_vs_a0 is not None and macro_vs_a0 >= 0.30/100 and pos_a0 >= 2 else 'NO-GO'}**",
            "",
            "Only Final GO vs A0 (with guards ≥ −0.20pp and Macro-F1 safe) makes M1 "
            "a final architecture candidate (plan §19). Mechanism GO alone only "
            "unlocks M2 (plan §20).",
        ]
    outdir = R2D20_ROOT / "m1_confirm"
    (outdir / "R2D20_M1_CONFIRM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[m1_confirm] saved -> {outdir / 'R2D20_M1_CONFIRM_REPORT.md'}")


def _final_report() -> None:
    summary_dir = R2D20_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    screen = _load(R2D20_ROOT / "m1_screen", "m1", ("HEAD", "M1"), TARGET_DATASETS, [42])
    rows = []
    for (ds, v, s), summary in screen.items():
        h = screen.get((ds, "HEAD", s))
        rows.append({
            "stage": "m1_screen", "dataset": ds, "variant": v, "seed": s,
            "val_acc": summary["best_val_acc"],
            "val_macro_f1": summary["best_val_macro_f1"],
            "delta_vs_head": (summary["best_val_acc"] - h["best_val_acc"]) if h else None,
        })
    with (summary_dir / "R2D20_MASTER_TABLE.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    ledger = [
        ("Shared propagation depth", "CLOSED",
         "D1.6-B: global/joint 2-hop unsupported (final-residual weak on both parents)."),
        ("Factor-specific propagation horizon", "WEAK",
         "M1 (H1 + alpha_f*(H2-H1), 3 scalars on frozen B0): +0.058pp macro, 2/3 positive, F1 safe — direction right, magnitude negligible vs the +0.30 GO bar."),
        ("Pt-specific 2-hop demand", "SUPPORTED",
         "PROBE-LEVEL ONLY: cross-parent frozen-probe support (D1.6-B) AND M1 learned alpha_Pt > 0 on 3/3 datasets (mean +0.061 vs +0.031 for C/Pv) — but it does NOT convert to end-to-end value."),
        ("Global factor-specific scale", "WEAK",
         "The M1 realization itself: NO-GO by the pre-registered gate."),
        ("0/1/2-hop mixture", "OPEN",
         "M2 not trained: gated behind M1 Mechanism GO (plan §20 forbids rescue-by-complexity)."),
        ("Node-adaptive scale", "OPEN",
         "Not tested; only admissible after global factor-specific formal GO (plan §35 Route C3)."),
        ("High-pass/diversification", "CLOSED", "D1.6-B: no cross-parent support."),
        ("Interaction PRODDIFF", "CLOSED",
         "D1.6: B0-dependent weak (+0.20pp 3-seed, below GO) / A0 NO-GO with F1 warnings — not mainline."),
        ("K-prototype relation", "CLOSED",
         "R2D1 + D1.6-A graph-control: relation neutralization ≈ full on all 5 datasets."),
        ("Scale-based routing", "CLOSED",
         "The simplest scale realization (M1 global factor-specific) is NO-GO end-to-end; deeper scale machinery stays gated."),
        ("From-scratch training", "CLOSED",
         "R1/R2D1/D1.5-B: joint-training coupling failures."),
        ("Warm-start training", "SUPPORTED",
         "The frozen/warm-start matched protocols worked cleanly throughout D1.6/D2.0 (no co-adaptation noise in the screens)."),
        ("Graph unfreezing", "OPEN",
         "Schedule study not entered (no GO candidate); the matched-init runner remains armed."),
        ("P0 ownership preservation", "SUPPORTED",
         "Kept frozen by design in all D2.0 training; nothing in the screens contradicts the anchor role."),
    ]
    with (summary_dir / "R2D20_HYPOTHESIS_LEDGER.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hypothesis", "status", "evidence"])
        writer.writerows(ledger)

    alphas = {
        ds: screen[(ds, "M1", 42)]["scale"]["alpha"] for ds in TARGET_DATASETS
    }
    lines = [
        "# R2D20_FINAL_DIAGNOSIS — R2-Design-2.0 Final Synthesis",
        "",
        "> Reads only completed stages. No new experiments, no Test. M1 guards/"
        "formal, M2 and the schedule study were NOT ENTERED by pre-registered "
        "gates (M1 seed42 = NO-GO).",
        "",
        "## Stage ledger",
        "",
        "| Stage | Status |",
        "|---|---|",
        "| D2.0-0 audit + biaxis_r2_scale implementation | PASS (11 tests; M0 bitwise==B0, M1 exact zero-degenerate, M2 near-degenerate) |",
        "| M1 seed42 screen | PASS — **NO-GO** (+0.058pp; PROBE-CONSISTENT) |",
        "| M1 guards + formal | NOT ENTERED (gate) |",
        "| M2 screen | NOT RUN (plan §20: no rescue-by-complexity) |",
        "| Schedule study | NOT ENTERED (no GO candidate) |",
        "| Final synthesis | this document |",
        "",
        "## Answers (plan §41)",
        "",
        "1. **Factor-specific propagation horizon end-to-end?** NO. M1 macro "
        "+0.058pp on the frozen B0 scaffold — two orders of magnitude below "
        "the +0.30pp GO bar.",
        "2. **M1 beats B0 HEAD?** Nominally yes (+0.030/+0.145/+0.000pp; 2/3 "
        "positive, Macro-F1 safe) — but the margin is noise-level.",
        "3. **M1 formally beats A0?** Not tested (gate); the observed margin "
        "cannot reach the +0.30pp vs-A0 bar regardless.",
        "4. **alpha_Pt consistent with the frozen Pt 2-hop diagnosis?** YES — "
        f"PROBE-CONSISTENT: alpha_Pt > 0 on 3/3 datasets "
        f"({ {k: f'{v[1]:+.4f}' for k, v in alphas.items()} }), mean +0.061 vs "
        "+0.031 for alpha_C/Pv. The calibration learns the RIGHT direction.",
        "5. **Reddit-S automatic fallback?** Not tested (guards gated).",
        "6. **C/Pt/Pv learn different horizons?** Directionally yes (C +0.005~"
        "+0.047, Pt +0.047~+0.069, Pv −0.010~+0.054) — but all magnitudes are "
        "tiny and never trigger the |α|>2 instability flag.",
        "7. **M2 truly better than M1?** Not tested (gated by design).",
        "8. **0-hop value?** Not tested (M2-only feature).",
        "9/10. **Schedule / co-adaptation harm?** Not tested (no GO candidate).",
        "11. **Macro-F1 safe?** Yes in every executed run (0 warnings; ΔF1 "
        "+0.07..+0.30pp).",
        "12. **Next route?** E — reopen Task-aware / semantic-aware relation "
        "learning (below).",
        "13. **Enter R2-Design-2.1 consolidation?** NO — no mechanism passed "
        "the GO bar.",
        "",
        "## Verdict",
        "",
        "### R2-Design-2.0 status: **NO-GO** (scale route)",
        "",
        "Per plan §44, the falsified claim is now frozen:",
        "",
        "> **The observed Pt 2-hop probe signal is not directly realizable "
        "through simple factor-specific horizon calibration.**",
        "",
        "The mechanism learns the correct direction (probe-consistent "
        "alphas) but carries negligible end-to-end value (+0.058pp). This is "
        "the THIRD time a frozen-probe headroom (R2-0A interaction, R2-0C "
        "interaction, R2-0B Pt-2-hop) failed to convert into trainable "
        "end-to-end gain — the consistent interpretation across R2D1/1.5/1.6/"
        "2.0 is that frozen-probe Ridge headroom on the parent's OWN states "
        "mostly measures what the parent's graph path + fusion already "
        "absorbs (recall D1.6 final-residual probes ≈ 0).",
        "",
        "### Recommended R2-Design-2.1 direction: **Route E — Task-Aware / "
        "Semantic-Aware Relation Learning**",
        "",
        "Stop stacking scale mechanisms. Redefine the second axis: "
        "task-aware / semantic-aware relation learning (RoleMAG / NRI-fNRI / "
        "IDGL / ACM-FAGCN / heterophily role learning / feature-conditioned "
        "edge transformations) — i.e., edge-level relation structure that "
        "sees task-relevant signals, which none of the realized R2 variants "
        "ever did (the closed K-prototype system was topology-only).",
        "",
        "Awaiting human/ChatGPT review. No Test has been run.",
        "",
        "---",
        "",
        "## Amendment: R2-D2.0.5 Scale Realization Attribution (2026-09-05, "
        "user-directed short control)",
        "",
        "Two decisive controls (details: outputs/perf_r2d20_5/"
        "R2D205_ATTRIBUTION_REPORT.md):",
        "",
        "**A — alpha_Pt fixed response curve** (frozen B0 + fresh classifier, "
        "18 runs): a genuinely useful fixed-scale region EXISTS — concave "
        "curves peaking at alpha_Pt = 0.5-0.75 with Movies +0.63 / Toys "
        "+0.31 / Grocery +0.64pp (mean +0.53pp) — a region the M1 SGD never "
        "found (learned alpha ≈ 0.05-0.07).",
        "",
        "**B — controlled warm-start adaptation** (S1 GRAPH-UNFREEZE / "
        "S2 GRAPH+FUSION-UNFREEZE, matched init, P0 frozen): "
        "max(S1,S2) − HEAD = +0.126pp (S1 +0.114 / S2 +0.126; Movies "
        "negative in both; alpha still only reaches 0.04-0.07).",
        "",
        "### Final ruling (user rule)",
        "",
        "- D2.0 Frozen-M1 = **CLEAR NO-GO** (confirmed).",
        "- General scale hypothesis = **CLOSED**: max(S1,S2)−HEAD "
        "+0.126pp < +0.15~0.20pp → **the Scale route definitively exits the "
        "mainline**; no formal confirm.",
        "- Attribution: the Pt 2-hop representation value is REAL (A) but "
        "NOT REACHABLE by any trainable protocol (B) — an optimization "
        "realization gap. A mechanism the current training cannot learn is "
        "not a method.",
        "- Next research object: **Semantic-Ownership-Aware Neighbor Utility "
        "Learning** — learn which neighbor is useful to which ownership "
        "factor BEFORE aggregation, with a formula-level collision audit of "
        "RoleMAG / CoMAG / IDGL / NRI / DisenGCN preceding the design of an "
        "edge-level pre-aggregation relation mechanism.",
    ]
    (summary_dir / "R2D20_FINAL_DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[final] saved -> {summary_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-Design-2.0 summarizer")
    parser.add_argument("--stage", required=True,
                        choices=["m1_screen", "m1_confirm", "m2", "final"])
    args = parser.parse_args()
    if args.stage == "m1_screen":
        _m1_screen_report()
    elif args.stage == "m1_confirm":
        _m1_confirm_report()
    elif args.stage == "m2":
        raise NotImplementedError("m2 summarizer is generated after the M2 screen")
    elif args.stage == "final":
        _final_report()


if __name__ == "__main__":
    main()
