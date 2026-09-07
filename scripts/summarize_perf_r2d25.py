"""R2-Design-2.5 summarizer
(docs/BiAxis_R2_Design_2_5_Structured_Capacity_Utilization_Audit.md).

Stages:
    audit         : D2.5-0 infrastructure audit report (R2D25_AUDIT.md)
    landscape     : D2.5-A alpha landscape CSVs + R2D25_LANDSCAPE_REPORT.md
    transmission  : D2.5-B CSVs + R2D25_TRANSMISSION_REPORT.md
    capacity      : D2.5-C CSVs + R2D25_CAPACITY_REPORT.md
    optimization  : D2.5-D CSVs + R2D25_OPTIMIZATION_REPORT.md
    final         : R2D25_MASTER_TABLE.csv + R2D25_HYPOTHESIS_LEDGER.csv
                    + R2D25_FINAL_DIAGNOSIS.md (17 questions + verdict)

Usage:
    python scripts/summarize_perf_r2d25.py --stage audit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d25_utils import (  # noqa: E402
    ALPHA_GRAD_VALUES,
    ALPHA_PT_VALUES,
    CAPACITY_MODES,
    DATASETS,
    GUARD_DATASETS,
    R2D25_ROOT,
    SEEDS,
    TARGET_DATASETS,
)

AUDIT_ROOT = R2D25_ROOT / "audit"
LANDSCAPE_ROOT = R2D25_ROOT / "landscape"
TRANSMISSION_ROOT = R2D25_ROOT / "transmission"
CAPACITY_ROOT = R2D25_ROOT / "capacity"
OPTIMIZATION_ROOT = R2D25_ROOT / "optimization"
SUMMARY_ROOT = R2D25_ROOT / "summary"

# ---------------------------------------------------------------------------
# Audit stage (Prompt 1)
# ---------------------------------------------------------------------------


def _collect_audit_facts() -> dict:
    """Live checks of the D2.5 infrastructure (used by --stage audit)."""
    import subprocess

    facts: dict = {"modes": {}, "files": {}}
    # per-mode model instantiation smoke (CPU, tiny cfg)
    import torch
    from omegaconf import OmegaConf

    from src.models.biaxis_r2_capacity import EXPERT_KEYS, MODES, Model

    d, h = 16, 32
    for mode in MODES:
        cfg = OmegaConf.create({
            "model": {
                "name": "biaxis_r2_capacity", "capacity_mode": mode,
                "hidden_dim": h, "factor_dim": d, "dropout": 0.2,
                "activation": "gelu", "norm": "layernorm",
                "lambda_common": 0.02, "lambda_orth": 0.01, "lambda_recon": 0.3,
                "orth_fallback_batch": 16, "full_graph_training": True,
                "edge_chunk_size": None,
                "deep_supervision": {"enabled": False, "lambda": 0.1},
                "path_dropout_p": 0.0,
                "semantic_refiner": {"enabled": False},
                "functional_transfer": {"enabled": False},
            },
        })
        info = {"input_dim": 20, "num_nodes": 30, "num_classes": 5,
                "text_dim": 9, "visual_dim": 11}
        m = Model(cfg, info)
        facts["modes"][mode] = {
            "params": int(m.parameter_count),
            "expert_keys": list(EXPERT_KEYS[mode]),
            "has_h2_path": mode in ("early_mix", "sep_sum", "sep_concat", "inception_012"),
        }
    # reference C2/C4/C5 parameter parity on the tiny cfg
    facts["param_match"] = {
        "sep_concat": facts["modes"]["sep_concat"]["params"],
        "cap_h1_dup": facts["modes"]["cap_h1_dup"]["params"],
        "wide_b0": facts["modes"]["wide_b0"]["params"],
    }
    # pytest count for the new test files
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_biaxis_r2_capacity.py",
         "tests/test_perf_r2d25_utils.py", "-q", "--no-header", "--tb=no"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    facts["pytest"] = (proc.stdout + proc.stderr).strip().splitlines()
    return facts


def _write_audit_report(facts: dict) -> Path:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    report = AUDIT_ROOT / "R2D25_AUDIT.md"
    lines = [
        "# R2-Design-2.5 — D2.5-0 Audit & Infrastructure",
        "",
        "Protocol: seeds 42/43/44; Val only; No Test. No lightweight constraint:",
        "every added block has a defined function and a parameter-matched control.",
        "",
        "## 1. Frozen-fact review (current code, verified by reading)",
        "",
        "- **B0 graph path** (`src/models/biaxis_r2.py`): F* -> 1-hop",
        "  `neighbor_mean` -> per-factor source transform `V_f`",
        "  (Linear(d,d), bias=False) -> per-factor LayerNorm (bias=False, so",
        "  LN(0)=0 keeps isolated nodes exact) -> `rho=sigma(raw)` residual",
        "  (init 0.5) -> P0 one-layer fusion",
        "  `Linear(3d,h) -> LN -> GELU -> Dropout`.",
        "- **M1 EarlyMix** (`biaxis_r2_scale.py`): `Hmix = H1 + alpha_f (H2-H1)`",
        "  with direct scalars (init 0). B0 checkpoints load with strict key",
        "  verification (admissible missing = mixer.alpha).",
        "- **PRODDIFF source-mean compression** (`biaxis_r2d15_adapters.py`):",
        "  9 cells `D^{a->b} = MLP([F_b*N_a, |F_b-N_a|, types])` are compressed",
        "  into 3 target corrections `Delta^b = (1/3) sum_a D^{a->b}` (source",
        "  mean), then `Fhat = F_out + Delta` -> fusion. This is the",
        "  premature-compression suspect: 9 independent cells collapse to a",
        "  3-way mean BEFORE any target-side readout.",
        "- **D2.0.5 frozen facts**: fixed-alpha response curve peaked at",
        "  alpha_Pt in [0.5, 0.75] (mean +0.53pp Val on M/T/G), while SGD",
        "  (M1) only reaches alpha ~ 0.04-0.07 and all trainable schedules",
        "  max out at +0.126pp (optimization realization gap).",
        "",
        "## 2. New infrastructure",
        "",
        "### Models",
        "- `src/models/biaxis_r2_capacity.py` +",
        "  `src/models/biaxis_r2_capacity_components.py`; one class, seven",
        "  modes (config `configs/model/biaxis_r2_capacity.yaml`,",
        "  `model.capacity_mode`).",
        "- `extract_capacity_states`: H0/H1/H2, hop expert outputs, before/after",
        "  LN (`msg_pre_ln`/`msg_post_ln`), before/after residual",
        "  (`pre_residual`/`post_residual`), pre/post fusion, per-expert",
        "  ablation (`forward(..., off_hops=...)`, mode-validated).",
        "- D2.5-D hooks (default OFF): `deep_supervision` aux heads on expert",
        "  outputs (inference-free) and `path_dropout_h1` (train-only).",
        "",
        "### Scripts",
        "- `perf_r2d25_landscape.py` — D2.5-A; frozen B0 + fixed alpha_Pt,",
        "  per-(dataset, seed) shared classifier init; linear V-precompute",
        "  pipeline; diagnostic-only dTrainCE/dValCE vs alpha.",
        "- `perf_r2d25_transmission.py` — D2.5-B; S0-S4 stage probes",
        "  (StandardScaler + Ridge(1.0), TRAIN fit / VAL eval) + PRODDIFF",
        "  9-cells -> source-mean -> factor-add -> fusion retrace (M/T/G).",
        "- `perf_r2d25_capacity_train.py` — D2.5-C; unified schedule",
        "  (P0 frozen 1-20 -> lr 1e-4; graph lr 1e-3; AdamW wd 1e-4;",
        "  warmup10+cosine; 300ep/patience30/best ValAcc); ablation + expert",
        "  diagnostics at best checkpoint.",
        "- `perf_r2d25_optimization.py` — D2.5-D; single-factor interventions",
        "  (expert LR / deep supervision / path dropout), per-expert grad",
        "  norm, update ratio, output norm, classifier sensitivity.",
        "- `summarize_perf_r2d25.py` — this file.",
        "",
        "## 3. Verification (live)",
        "",
    ]
    modes = facts["modes"]
    lines.append("| mode | params (tiny cfg) | expert keys | H2 path |")
    lines.append("|---|---|---|---|")
    for mode in CAPACITY_MODES:
        m = modes[mode]
        lines.append(f"| {mode} | {m['params']} | {','.join(m['expert_keys']) or '-'} "
                     f"| {'yes' if m['has_h2_path'] else 'no'} |")
    lines.append("")
    lines.append("Parameter parity (tiny cfg): "
                 f"sep_concat={facts['param_match']['sep_concat']}, "
                 f"cap_h1_dup={facts['param_match']['cap_h1_dup']}, "
                 f"wide_b0={facts['param_match']['wide_b0']}.")
    lines.append("")
    lines.append("### Tests")
    lines.append("")
    lines.append("```")
    lines += facts["pytest"][-6:]
    lines.append("```")
    lines.append("")
    lines.append("## 4. Pre-registered interpretation matrix")
    lines.append("")
    lines.append("1. C2/C3 > A0 and > WIDE_B0/CAP_H1_DUP -> premature compression")
    lines.append("   was a real bottleneck -> Ownership-Aware Multi-Scale Expert Fusion.")
    lines.append("2. WIDE_B0 ~= C2/C3 and both improve -> generic backbone capacity ->")
    lines.append("   build a stronger backbone first.")
    lines.append("3. Representation utility remains, main CE underuses it, deep")
    lines.append("   supervision helps -> Gradient Starvation SUPPORTED.")
    lines.append("4. All fail -> Semantic-Ownership-Aware Neighbor Utility Learning")
    lines.append("   (learn which neighbor is useful for which factor BEFORE")
    lines.append("   aggregation).")
    lines.append("")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Transmission stage (Prompt 3)
# ---------------------------------------------------------------------------


def _collect_transmission() -> list[dict]:
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            p = TRANSMISSION_ROOT / ds / f"seed_{seed}" / "summary.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            for r in data["rows"]:
                row = dict(r)
                row.setdefault("cosine", None)
                row.setdefault("cka", None)
                row.setdefault("rel_norm", None)
                row.setdefault("eff_rank", None)
                row.setdefault("fixed_parent_acc", None)
                row.setdefault("fixed_parent_f1", None)
                row.setdefault("retrained_acc", None)
                row.setdefault("retrained_f1", None)
                rows.append(row)
    return rows


def _write_transmission_report(rows: list[dict]) -> Path:
    if not rows:
        raise RuntimeError("no transmission summaries found — run perf_r2d25_transmission.py first")
    fieldnames = ["dataset", "seed", "stage", "branch", "acc", "macro_f1", "cosine", "cka",
                  "rel_norm", "eff_rank", "fixed_parent_acc", "fixed_parent_f1",
                  "retrained_acc", "retrained_f1"]
    csv_path = TRANSMISSION_ROOT / "scale_transmission.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # PRODDIFF secondary rows -> interaction_transmission.csv
    pd_rows = []
    for ds in TARGET_DATASETS:
        for seed in SEEDS:
            p = TRANSMISSION_ROOT / ds / f"seed_{seed}" / "summary.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            if data.get("proddiff"):
                pd_rows.extend(data["proddiff"]["rows"])
    if pd_rows:
        pd_path = TRANSMISSION_ROOT / "interaction_transmission.csv"
        with pd_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "seed", "stage", "cell",
                                                   "utility_delta", "norm"])
            writer.writeheader()
            for r in pd_rows:
                writer.writerow(r)

    import statistics

    # per (dataset, seed): S0 delta + retention; per dataset: 3-seed means
    stage_names = ["s0_raw", "s1_src_transform", "s2_after_ln", "s3_factor_residual", "s4_fusion"]
    report = TRANSMISSION_ROOT / "R2D25_TRANSMISSION_REPORT.md"
    lines = ["# R2-D2.5-B — Layer-wise utility transmission audit", "",
             "Probes: StandardScaler + RidgeClassifier(alpha=1.0), TRAIN fit / VAL eval;",
             "delta = acc(H2 branch) - acc(H1 branch) in pp; retention = delta_stage / delta_S0.",
             ""]
    lines.append("| dataset | stage | delta H2-H1 (3-seed mean, pp) | retention |")
    lines.append("|---|---|---|---|")
    collapse: dict[str, str] = {}
    for ds in DATASETS:
        deltas: dict[str, list[float]] = {s: [] for s in stage_names}
        for seed in SEEDS:
            p = TRANSMISSION_ROOT / ds / f"seed_{seed}" / "summary.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            by_stage: dict[str, dict[str, float]] = {}
            for r in data["rows"]:
                by_stage.setdefault(r["stage"], {})[r["branch"]] = r["acc"]
            for s in stage_names:
                if s in by_stage and "h1" in by_stage[s] and "h2" in by_stage[s]:
                    deltas[s].append(100.0 * (by_stage[s]["h2"] - by_stage[s]["h1"]))
        means = {s: statistics.fmean(v) if v else None for s, v in deltas.items()}
        s0 = means["s0_raw"] if means["s0_raw"] is not None else 0.0
        first_collapse = None
        for s in stage_names:
            m = means[s]
            if m is None:
                continue
            retention = (m / s0) if abs(s0) > 1e-9 else None
            lines.append(
                f"| {ds} | {s} | {m:+.3f} | "
                f"{retention if retention is None else f'{retention:+.2f}'} |")
            if first_collapse is None and s0 > 0.10 and (m <= 0.0 or m < 0.5 * s0):
                first_collapse = s
        collapse[ds] = first_collapse or "never (within S0-S4)"
        lines.append(f"| {ds} | **first material collapse** | **{collapse[ds]}** | |")
        lines.append("")
    lines.append("### S4 fusion detail: H2-H1 delta per head (pp, 3-seed mean)")
    lines.append("")
    for ds in DATASETS:
        fixed, retr = [], []
        for seed in SEEDS:
            p = TRANSMISSION_ROOT / ds / f"seed_{seed}" / "summary.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            by_branch = {}
            for r in data["rows"]:
                if r["stage"] == "s4_fusion":
                    by_branch[r["branch"]] = r
            if "h1" in by_branch and "h2" in by_branch:
                fixed.append(100.0 * ((by_branch["h2"]["fixed_parent_acc"] or 0.0)
                                      - (by_branch["h1"]["fixed_parent_acc"] or 0.0)))
                retr.append(100.0 * ((by_branch["h2"]["retrained_acc"] or 0.0)
                                     - (by_branch["h1"]["retrained_acc"] or 0.0)))
        if fixed:
            lines.append(
                f"- {ds}: fixed-parent-head delta {statistics.fmean(fixed):+.3f}pp / "
                f"retrained-head delta {statistics.fmean(retr):+.3f}pp "
                f"({sum(1 for d in retr if d > 0)}/{len(retr)} seeds positive)")
        else:
            lines.append(f"- {ds}: no data")
    lines.append("")
    lines.append("### PRODDIFF secondary (M/T/G): strongest cell per (dataset, seed)")
    lines.append("")
    for ds in TARGET_DATASETS:
        for seed in SEEDS:
            p = TRANSMISSION_ROOT / ds / f"seed_{seed}" / "summary.json"
            if not p.exists():
                continue
            pd_ = json.loads(p.read_text()).get("proddiff")
            if pd_:
                lines.append(
                    f"- {ds} s{seed}: strongest cell **{pd_['strongest_cell']}** "
                    f"(utility {100 * pd_['strongest_cell_utility']:+.3f}pp, "
                    f"norm {pd_['strongest_cell_norm']:.3f}); fusion fixed-parent "
                    f"acc {pd_['fixed_parent_acc']:.4f} / retrained {pd_['retrained_acc']:.4f}")
    lines.append("")
    lines.append("First-collapse stage answers D2.5-B's core question: where does the")
    lines.append("Pt H2 utility materially leak on the frozen B0 path.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Capacity stage (Prompt 4)
# ---------------------------------------------------------------------------


def _collect_capacity() -> list[dict]:
    rows = []
    for ds in DATASETS:
        for v in CAPACITY_MODES:
            for seed in SEEDS:
                p = CAPACITY_ROOT / ds / v / f"seed_{seed}" / "summary.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                row = {
                    "dataset": ds, "variant": v, "seed": seed,
                    "val_acc": d["best_val_acc"], "val_macro_f1": d["best_val_macro_f1"],
                    "best_epoch": d["best_epoch"], "stop_epoch": d["stop_epoch"],
                    "params": d["parameter_count"],
                    "peak_mb": d.get("peak_allocated_mb"),
                    "runtime_sec": d.get("runtime_sec"),
                    "param_delta_pct": d.get("param_delta_pct"),
                }
                for tag, m in d.get("ablations", {}).items():
                    row[f"abl_{tag}_acc"] = m["val_acc"]
                    row[f"abl_{tag}_f1"] = m["val_macro_f1"]
                diag = d.get("diagnostics", {})
                experts = diag.get("experts", {})
                if experts:
                    row["expert_eff_rank"] = experts.get("effective_rank")
                    row["expert_pairwise_cosine"] = experts.get("pairwise_cosine")
                    row["expert_cka"] = experts.get("cka")
                rows.append(row)
    return rows


def _write_capacity_report(rows: list[dict]) -> Path:
    if not rows:
        raise RuntimeError("no capacity summaries found — run perf_r2d25_capacity_train.py first")
    import statistics

    # CSVs
    res_fields = ["dataset", "variant", "seed", "val_acc", "val_macro_f1",
                  "best_epoch", "stop_epoch"]
    with (CAPACITY_ROOT / "capacity_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=res_fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with (CAPACITY_ROOT / "capacity_resources.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "params",
                                               "peak_mb", "runtime_sec", "param_delta_pct"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with (CAPACITY_ROOT / "capacity_mechanism.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r if k.startswith(("abl_", "expert_"))}),
                                extrasaction="ignore")
        if rows:
            flds = ["dataset", "variant", "seed"] + sorted(
                {k for r in rows for k in r if k.startswith(("abl_", "expert_"))})
            writer.fieldnames = flds
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # paired deltas vs EARLY_MIX (mechanism baseline) and vs A0
    from src.analysis.perf_r2_utils import load_a0_reference

    a0_acc = load_a0_reference()  # formal per-seed val ACC
    # A0 per-seed val macro-F1: R1-baseline proxy (structure bitwise ==
    # biaxis_final, max |val acc delta| 0.176pp — disclosed, D1.6 convention).
    a0_f1: dict[tuple[str, int], float] = {}
    for ds in DATASETS:
        for seed in SEEDS:
            p = PROJECT_ROOT / "outputs" / "perf_r1" / "baseline" / ds / "A0" / f"seed_{seed}" / "summary.json"
            if p.exists():
                d = json.loads(p.read_text())
                if d.get("best_val_macro_f1") is not None:
                    a0_f1[(ds, seed)] = float(d["best_val_macro_f1"]) / 100.0
    by_key: dict[tuple[str, str, int], dict] = {
        (r["dataset"], r["variant"], r["seed"]): r for r in rows}
    report = CAPACITY_ROOT / "R2D25_CAPACITY_REPORT.md"
    lines = ["# R2-D2.5-C — Structured-capacity model matrix", "",
             "Paired per (dataset, seed); pp. Two metrics reported: Macro-F1 (the",
             "pre-registered verdict metric) and Accuracy (supplementary — the",
             "B0-family has a known systematic Macro-F1 deficit, D1.6).",
             "A0 F1 = R1-baseline A0 proxy (bitwise==biaxis_final structure,",
             "max |val acc delta| 0.176pp, disclosed). All Val only.",
             ""]
    candidates = [v for v in CAPACITY_MODES if v != "early_mix"]

    def _deltas(metric: str) -> dict[str, dict]:
        out = {}
        for v in candidates:
            ds_means, seeds_pos, f1s, a0s = [], [], [], []
            for ds in TARGET_DATASETS:
                deltas = []
                for seed in SEEDS:
                    base = by_key.get((ds, "early_mix", seed))
                    cand = by_key.get((ds, v, seed))
                    if base and cand:
                        d = 100.0 * (cand[metric] - base[metric])
                        deltas.append(d)
                        f1s.append(d)
                        a0_val = a0_f1.get((ds, seed)) if metric == "val_macro_f1" \
                            else a0_acc.get((ds, seed))
                        if a0_val is not None:
                            a0s.append(100.0 * (cand[metric] - a0_val))
                if deltas:
                    ds_means.append(statistics.fmean(deltas))
                    seeds_pos.append(sum(1 for d in deltas if d > 0))
            mean = statistics.fmean(ds_means) if ds_means else None
            out[v] = {
                "ds_means": ds_means, "mean": mean,
                "n_pos": sum(1 for d in ds_means if d > 0) if ds_means else 0,
                "seeds_pos": seeds_pos,
                "f1_safe": (statistics.fmean(f1s) >= -0.20
                            and all(d >= -0.50 for d in f1s)) if f1s else True,
                "a0_mean": statistics.fmean(a0s) if a0s else None,
                "a0_pos": sum(1 for d in a0s if d > 0) if a0s else 0,
            }
        return out

    f1 = _deltas("val_macro_f1")
    acc = _deltas("val_acc")
    for metric, name in (("val_macro_f1", "Macro-F1"), ("val_acc", "Accuracy")):
        d = f1 if metric == "val_macro_f1" else acc
        lines.append(f"## {name} deltas vs EARLY_MIX (pp, 3-seed means)")
        lines.append("")
        lines.append("| variant | Movies | Toys | Grocery | M/T/G mean | vs A0 mean | ≥2/3 ds pos | F1-safe |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for v in candidates:
            dm = d[v]["ds_means"]
            a0_str = "None" if d[v]["a0_mean"] is None else f"{d[v]['a0_mean']:+.2f}"
            lines.append(
                f"| {v} | {dm[0]:+.2f} | {dm[1]:+.2f} | {dm[2]:+.2f} "
                f"| {d[v]['mean']:+.2f} | {a0_str} "
                f"| {d[v]['n_pos']}/3 | {'yes' if d[v]['f1_safe'] else 'NO'} |")
        lines.append("")
    # mechanism-specific (vs capacity controls), both metrics
    lines.append("## Mechanism-specific: candidate - control (pp, M/T/G 3-seed mean)")
    lines.append("")
    lines.append("| candidate | vs CAP_H1_DUP (F1 / Acc) | vs WIDE_B0 (F1 / Acc) |")
    lines.append("|---|---|---|")
    for v in ("sep_sum", "sep_concat", "inception_012"):
        vs_dup_f1 = statistics.fmean(
            100.0 * (by_key[(ds, v, s)]["val_macro_f1"] - by_key[(ds, "cap_h1_dup", s)]["val_macro_f1"])
            for ds in TARGET_DATASETS for s in SEEDS)
        vs_dup_acc = statistics.fmean(
            100.0 * (by_key[(ds, v, s)]["val_acc"] - by_key[(ds, "cap_h1_dup", s)]["val_acc"])
            for ds in TARGET_DATASETS for s in SEEDS)
        vs_wb_f1 = statistics.fmean(
            100.0 * (by_key[(ds, v, s)]["val_macro_f1"] - by_key[(ds, "wide_b0", s)]["val_macro_f1"])
            for ds in TARGET_DATASETS for s in SEEDS)
        vs_wb_acc = statistics.fmean(
            100.0 * (by_key[(ds, v, s)]["val_acc"] - by_key[(ds, "wide_b0", s)]["val_acc"])
            for ds in TARGET_DATASETS for s in SEEDS)
        lines.append(f"| {v} | {vs_dup_f1:+.2f} / {vs_dup_acc:+.2f} | {vs_wb_f1:+.2f} / {vs_wb_acc:+.2f} |")
    lines.append("")
    lines.append("### H2-off causal usage (full - h2_off, Val acc pp, 3-seed mean)")
    lines.append("")
    for v in candidates:
        for ds in TARGET_DATASETS:
            drops = []
            for seed in SEEDS:
                r = by_key.get((ds, v, seed))
                if r and "abl_full_acc" in r and "abl_h2_off_acc" in r:
                    drops.append(100.0 * (r["abl_full_acc"] - r["abl_h2_off_acc"]))
            if drops:
                lines.append(f"- {v} / {ds}: H2-off drop {statistics.fmean(drops):+.3f}pp "
                             f"({sum(1 for d in drops if d > 0)}/{len(drops)} seeds positive)")
    lines.append("")
    lines.append("### Formal verdicts (pre-registered thresholds, Macro-F1)")
    lines.append("")
    lines.append("- Mechanism GO: candidate - EARLY_MIX >= +0.50pp macro, >=2/3 datasets")
    lines.append("  positive, positive datasets >=2/3 seeds positive, F1-safe.")
    lines.append("- Final GO: candidate - A0 >= +0.30pp macro with the same stability.")
    lines.append("- Mechanism-specific: candidate - CAP_H1_DUP >= +0.20pp or candidate -")
    lines.append("  WIDE_B0 >= +0.20pp; otherwise GENERIC CAPACITY GAIN.")
    lines.append("- Any candidate within +0.20pp of Final GO -> run guards x 3 seeds.")
    lines.append("")
    for v in candidates:
        if v in ("cap_h1_dup", "wide_b0"):  # controls are the references
            continue
        m = f1[v]
        mech = m["mean"] is not None and m["mean"] >= 0.50 and m["n_pos"] >= 2 and m["f1_safe"]
        final = (m["a0_mean"] is not None and m["a0_mean"] >= 0.30
                 and m["n_pos"] >= 2 and m["f1_safe"])
        dup = statistics.fmean(
            100.0 * (by_key[(ds, v, s)]["val_macro_f1"] - by_key[(ds, "cap_h1_dup", s)]["val_macro_f1"])
            for ds in TARGET_DATASETS for s in SEEDS)
        wb = statistics.fmean(
            100.0 * (by_key[(ds, v, s)]["val_macro_f1"] - by_key[(ds, "wide_b0", s)]["val_macro_f1"])
            for ds in TARGET_DATASETS for s in SEEDS)
        spec = dup >= 0.20 or wb >= 0.20
        acc_spec = acc[v]["mean"] is not None and (
            statistics.fmean(100.0 * (by_key[(ds, v, s)]["val_acc"] - by_key[(ds, "cap_h1_dup", s)]["val_acc"])
                             for ds in TARGET_DATASETS for s in SEEDS) >= 0.20
            or statistics.fmean(100.0 * (by_key[(ds, v, s)]["val_acc"] - by_key[(ds, "wide_b0", s)]["val_acc"])
                                for ds in TARGET_DATASETS for s in SEEDS) >= 0.20)
        lines.append(
            f"- **{v}**: Mechanism {'GO' if mech else 'NO-GO'} "
            f"(+{m['mean']:+.2f}pp, {m['n_pos']}/3, F1-safe={'yes' if m['f1_safe'] else 'NO'}), "
            f"Final {'GO' if final else 'NO-GO'} "
            f"(vs A0 {m['a0_mean'] if m['a0_mean'] is None else f'{m['a0_mean']:+.2f}'}pp), "
            f"Mechanism-specific {'GO' if spec else 'NO-GO'} "
            f"(vs DUP {dup:+.2f} / vs WIDE {wb:+.2f}pp F1"
            f"{'; ACC-side mechanism-specific positive' if acc_spec and not spec else ''}).")
    lines.append("")
    lines.append("ACC-side note: the Macro-F1 verdicts are the pre-registered ones; ACC")
    lines.append("deltas are reported as supplementary evidence (B0-family F1 deficit).")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Optimization stage (Prompt 5)
# ---------------------------------------------------------------------------


def _collect_optimization() -> list[dict]:
    """Intervention summaries + the reused D2.5-C base rows."""
    rows = []
    for ds in DATASETS:
        for v in CAPACITY_MODES:
            for seed in SEEDS:
                base_p = CAPACITY_ROOT / ds / v / f"seed_{seed}" / "summary.json"
                if not base_p.exists():
                    continue
                d = json.loads(base_p.read_text())
                rows.append({
                    "dataset": ds, "variant": v, "intervention": "base", "setting": "base",
                    "seed": seed, "best_val_acc": d["best_val_acc"],
                    "best_val_macro_f1": d["best_val_macro_f1"],
                    "train_ce_at_best": None, "val_ce_at_best": None,
                    "ablations": d.get("ablations", {}),
                })
    for ds in DATASETS:
        p = OPTIMIZATION_ROOT / ds
        if not p.exists():
            continue
        for v_dir in sorted(p.iterdir()):
            if not v_dir.is_dir():
                continue
            for i_dir in sorted(v_dir.iterdir()):
                if not i_dir.is_dir():
                    continue
                for set_dir in sorted(i_dir.iterdir()):  # setting_<v>
                    if not set_dir.is_dir():
                        continue
                    for seed_dir in sorted(set_dir.iterdir()):  # seed_<s>
                        sp = seed_dir / "summary.json"
                        if not sp.exists():
                            continue
                        d = json.loads(sp.read_text())
                        rows.append({
                            "dataset": ds, "variant": d["variant"], "intervention": d["intervention"],
                            "setting": d["setting"], "seed": d["seed"],
                            "best_val_acc": d["best_val_acc"],
                            "best_val_macro_f1": d["best_val_macro_f1"],
                            "train_ce_at_best": d.get("train_ce_at_best"),
                            "val_ce_at_best": d.get("val_ce_at_best"),
                            "expert_output_norm": d.get("expert_output_norm"),
                            "classifier_sensitivity": d.get("classifier_sensitivity"),
                            "expert_param_stats": d.get("expert_param_stats"),
                            "ablations": d.get("ablations", {}),
                            "best_epoch": d.get("best_epoch"),
                            "stop_epoch": d.get("stop_epoch"),
                        })
    return rows


def _write_optimization_report(rows: list[dict]) -> Path:
    if not rows:
        raise RuntimeError("no optimization summaries found — run perf_r2d25_optimization.py first")
    import statistics

    OPTIMIZATION_ROOT.mkdir(parents=True, exist_ok=True)
    with (OPTIMIZATION_ROOT / "optimization_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "intervention", "setting",
                                               "seed", "best_val_acc", "best_val_macro_f1",
                                               "train_ce_at_best", "val_ce_at_best",
                                               "best_epoch", "stop_epoch"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with (OPTIMIZATION_ROOT / "optimization_gradients.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "intervention", "setting",
                                               "seed", "expert_output_norm",
                                               "classifier_sensitivity", "expert_param_stats"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            if r["intervention"] == "base":
                continue
            r = dict(r)
            for k in ("expert_output_norm", "classifier_sensitivity", "expert_param_stats"):
                if isinstance(r.get(k), dict):
                    r[k] = json.dumps(r[k])
            writer.writerow(r)

    by_key: dict[tuple[str, str, str, str, int], dict] = {
        (r["dataset"], r["variant"], r["intervention"], r["setting"], r["seed"]): r
        for r in rows if r["intervention"] != "base"}
    base_key: dict[tuple[str, str, int], dict] = {
        (r["dataset"], r["variant"], r["seed"]): r
        for r in rows if r["intervention"] == "base"}
    variants = sorted({r["variant"] for r in rows if r["intervention"] != "base"})
    report = OPTIMIZATION_ROOT / "R2D25_OPTIMIZATION_REPORT.md"
    lines = ["# R2-D2.5-D — Optimization-accessibility interventions", "",
             "Single-factor interventions on the D2.5-C expert candidates; the base",
             "setting of every intervention is the reused D2.5-C run. Paired deltas",
             "vs base (Val Acc pp, M/T/G 3-seed mean). Pre-registered reading:",
             "- Gradient Starvation SUPPORTED if an expert-access intervention",
             "  (expert LR / deep supervision) improves Val >= +0.20pp mean with",
             "  >=2/3 datasets positive;",
             "- Objective mismatch SUPPORTED if interventions change expert usage",
             "  (output norms / H2-off drops / train CE) without Val following.",
             ""]
    for v in variants:
        lines.append(f"## {v}")
        lines.append("")
        lines.append("| intervention | setting | Movies | Toys | Grocery | M/T/G mean | ≥2/3 pos |")
        lines.append("|---|---|---|---|---|---|---|")
        interventions = ["expert_lr", "deep_sup", "path_dropout"]
        for i in interventions:
            for setting in ({"expert_lr": ("0.005", "0.01"), "deep_sup": ("0.1",),
                             "path_dropout": ("0.2",)}[i]):
                ds_means = []
                for ds in TARGET_DATASETS:
                    deltas = []
                    for seed in SEEDS:
                        cand = by_key.get((ds, v, i, setting, seed))
                        base = base_key.get((ds, v, seed))
                        if cand and base:
                            deltas.append(100.0 * (cand["best_val_acc"] - base["best_val_acc"]))
                    if deltas:
                        ds_means.append(statistics.fmean(deltas))
                if not ds_means:
                    continue
                mean = statistics.fmean(ds_means)
                n_pos = sum(1 for d in ds_means if d > 0)
                lines.append(
                    f"| {i} | {setting} | {ds_means[0]:+.2f} | {ds_means[1]:+.2f} | "
                    f"{ds_means[2]:+.2f} | {mean:+.2f} | {n_pos}/3 |")
        # expert usage shifts (M/T/G mean) for the new settings
        lines.append("")
        lines.append("Expert usage shifts vs base (M/T/G 3-seed mean):")
        for i in interventions:
            for setting in ({"expert_lr": ("0.005", "0.01"), "deep_sup": ("0.1",),
                             "path_dropout": ("0.2",)}[i]):
                norm_shifts, h2off_shifts = [], []
                for ds in TARGET_DATASETS:
                    for seed in SEEDS:
                        cand = by_key.get((ds, v, i, setting, seed))
                        base = base_key.get((ds, v, seed))
                        if not (cand and base):
                            continue
                        n_c = cand.get("expert_output_norm") or {}
                        n_b = base.get("ablations") or {}
                        if n_c:
                            norm_shifts.append({
                                k: (n_c.get(k, 0.0) or 0.0) for k in n_c})
                        abl_c = cand.get("ablations", {}).get("h2_off")
                        abl_b = base.get("ablations", {}).get("h2_off")
                        if abl_c and abl_b:
                            h2off_shifts.append(
                                100.0 * ((cand["ablations"]["full"]["val_acc"]
                                          - abl_c["val_acc"])
                                         - (base["ablations"]["full"]["val_acc"]
                                            - abl_b["val_acc"])))
                if not h2off_shifts:
                    continue
                lines.append(
                    f"- {i}/{setting}: H2-off drop shift "
                    f"{statistics.fmean(h2off_shifts):+.2f}pp "
                    f"(base-drop {100.0 * statistics.fmean([
                        base_key[(ds, v, s)]['ablations']['full']['val_acc']
                        - base_key[(ds, v, s)]['ablations']['h2_off']['val_acc']
                        for ds in TARGET_DATASETS for s in SEEDS if (ds, v, s) in base_key]):.2f}pp)")
        lines.append("")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Hop-token attention stage (Prompt 6)
# ---------------------------------------------------------------------------


def _write_hop_attention_report() -> Path:
    import statistics

    HOP_ROOT = R2D25_ROOT / "hop_attention"
    attn_rows = _collect_capacity_rows_from(HOP_ROOT)
    # base rows: EARLY_MIX from the capacity matrix, h1_attention from this
    # stage's own root (it was trained into hop_attention/<ds>/h1_attention/).
    base_rows = _collect_capacity() + _collect_capacity_rows_from(HOP_ROOT)
    if not attn_rows:
        raise RuntimeError("no hop_attention summaries found — run "
                           "perf_r2d25_capacity_train.py --out-root outputs/perf_r2d25/hop_attention")
    from src.analysis.perf_r2_utils import load_a0_reference

    a0_acc = load_a0_reference()
    report = HOP_ROOT / "R2D25_HOP_ATTN_REPORT.md"
    lines = ["# R2-D2.5-E — Mature hop-token attention (conditional)", "",
             "Per-factor 3 hop tokens -> 2 Pre-LN blocks (d, 4 heads, FFN 4d,",
             "dropout 0.1); ego/H0 token as query/summary; 2-layer fusion.",
             "Capacity control: same architecture, 3 tokens from independent H1",
             "transforms (h1_attention). Deltas per (dataset, seed), pp.",
             ""]
    by_attn: dict[tuple[str, str, int], dict] = {
        (r["dataset"], r["variant"], r["seed"]): r for r in attn_rows}
    by_base: dict[tuple[str, str, int], dict] = {
        (r["dataset"], r["variant"], r["seed"]): r for r in base_rows}
    with (HOP_ROOT / "hop_attention_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "val_acc",
                                               "val_macro_f1", "best_epoch", "stop_epoch",
                                               "params"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in attn_rows:
            writer.writerow(r)
    for metric in ("val_acc", "val_macro_f1"):
        lines.append(f"### {metric} deltas (M/T/G, 3-seed means)")
        lines.append("")
        lines.append("| comparison | Movies | Toys | Grocery | mean |")
        lines.append("|---|---|---|---|---|")
        for cmp_name, base_v in (("vs EARLY_MIX (B0 control)", "early_mix"),
                                 ("vs h1_attention (capacity control)", "h1_attention")):
            ds_means = []
            for ds in TARGET_DATASETS:
                deltas = [
                    100.0 * (by_attn[(ds, "hop_attention", s)][metric]
                             - by_base[(ds, base_v, s)][metric])
                    for s in SEEDS
                    if (ds, "hop_attention", s) in by_attn and (ds, base_v, s) in by_base
                ]
                ds_means.append(statistics.fmean(deltas) if deltas else float("nan"))
            mean = statistics.fmean([d for d in ds_means if d == d]) if any(
                d == d for d in ds_means) else float("nan")
            lines.append(
                f"| {cmp_name} | {ds_means[0]:+.2f} | {ds_means[1]:+.2f} | "
                f"{ds_means[2]:+.2f} | {mean:+.2f} |")
        lines.append("")
    lines.append("### H2-off causal usage (full - h2_off, Val acc pp)")
    lines.append("")
    for ds in TARGET_DATASETS:
        drops = []
        for s in SEEDS:
            r = by_attn.get((ds, "hop_attention", s))
            if r and "abl_full_acc" in r and "abl_h2_off_acc" in r:
                drops.append(100.0 * (r["abl_full_acc"] - r["abl_h2_off_acc"]))
        if drops:
            lines.append(f"- {ds}: {statistics.fmean(drops):+.3f}pp "
                         f"({sum(1 for d in drops if d > 0)}/{len(drops)} seeds positive)")
    lines.append("")
    lines.append("### Attention weights (mean over nodes; rows=query, cols=key)")
    lines.append("")
    for ds in TARGET_DATASETS:
        for s in SEEDS:
            r = by_attn.get((ds, "hop_attention", s))
            if not r or not r.get("attention_weights"):
                continue
            aw = r["attention_weights"]  # 6 entries: per (factor, layer) a [3,3] matrix
            lines.append(f"- {ds} s{s}:")
            for f in range(3):
                layers = aw[2 * f : 2 * f + 2]
                for li, layer in enumerate(layers):
                    rows = [" ".join(f"{v:.2f}" for v in row) for row in layer]
                    lines.append(f"  - factor {f} L{li + 1}: " + " | ".join(rows))
    lines.append("")
    lines.append("Verdict: entered because SEP_CONCAT/INCEPTION beat WIDE_B0 by")
    lines.append(">= +0.20pp (F1) while D2.5-B shows the late readout still limits")
    lines.append("utilization. The capacity control isolates the value of cross-hop")
    lines.append("token exchange from the added capacity itself.")
    HOP_ROOT.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _collect_capacity_rows_from(root: Path) -> list[dict]:
    """Capacity-style summaries from an arbitrary root."""
    rows = []
    if not root.exists():
        return rows
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        for v_dir in sorted(ds_dir.iterdir()):
            if not v_dir.is_dir():
                continue
            for s_dir in sorted(v_dir.iterdir()):
                if not s_dir.is_dir():
                    continue
                p = s_dir / "summary.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                row = {
                    "dataset": ds_dir.name, "variant": d.get("variant", v_dir.name),
                    "seed": int(s_dir.name.split("_")[-1]),
                    "val_acc": d["best_val_acc"], "val_macro_f1": d["best_val_macro_f1"],
                    "best_epoch": d.get("best_epoch"), "stop_epoch": d.get("stop_epoch"),
                    "params": d.get("parameter_count"),
                    "peak_mb": d.get("peak_allocated_mb"),
                    "runtime_sec": d.get("runtime_sec"),
                    "attention_weights": (d.get("diagnostics") or {}).get("attention_weights"),
                }
                for tag, m in d.get("ablations", {}).items():
                    row[f"abl_{tag}_acc"] = m["val_acc"]
                    row[f"abl_{tag}_f1"] = m["val_macro_f1"]
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Final synthesis (Prompt 7)
# ---------------------------------------------------------------------------


def _write_final_synthesis() -> tuple[Path, Path, Path]:
    import statistics

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    master = SUMMARY_ROOT / "R2D25_MASTER_TABLE.csv"
    ledger = SUMMARY_ROOT / "R2D25_HYPOTHESIS_LEDGER.csv"
    diagnosis = SUMMARY_ROOT / "R2D25_FINAL_DIAGNOSIS.md"

    # ---- master table: every formal run in one CSV -------------------------
    master_rows: list[dict] = []
    if (LANDSCAPE_ROOT / "alpha_landscape.csv").exists():
        with (LANDSCAPE_ROOT / "alpha_landscape.csv").open() as f:
            for r in csv.DictReader(f):
                master_rows.append({"stage": "landscape", **r})
    if (TRANSMISSION_ROOT / "scale_transmission.csv").exists():
        with (TRANSMISSION_ROOT / "scale_transmission.csv").open() as f:
            for r in csv.DictReader(f):
                sub = r.pop("stage")
                master_rows.append({"stage": "transmission", "sub_stage": sub, **r})
    if (CAPACITY_ROOT / "capacity_results.csv").exists():
        with (CAPACITY_ROOT / "capacity_results.csv").open() as f:
            for r in csv.DictReader(f):
                master_rows.append({"stage": "capacity", **r})
    if (OPTIMIZATION_ROOT / "optimization_results.csv").exists():
        with (OPTIMIZATION_ROOT / "optimization_results.csv").open() as f:
            for r in csv.DictReader(f):
                master_rows.append({"stage": "optimization", **r})
    if (R2D25_ROOT / "hop_attention" / "hop_attention_results.csv").exists():
        with (R2D25_ROOT / "hop_attention" / "hop_attention_results.csv").open() as f:
            for r in csv.DictReader(f):
                master_rows.append({"stage": "hop_attention", **r})
    if master_rows:
        fields = ["stage"] + sorted({k for r in master_rows for k in r if k != "stage"})
        with master.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in master_rows:
                writer.writerow(r)

    # ---- hypothesis ledger (hypothesis -> stage -> verdict) ----------------
    ledger_rows = [
        {"hypothesis": "fixed alpha_Pt has 3-seed stable value (D2.0.5 seed-42 curve)",
         "stage": "D2.5-A", "verdict": "REJECTED (flat, +0.001~0.003pp)",
         "evidence": "outputs/perf_r2d25/landscape/"},
        {"hypothesis": "Pt H2 utility survives the B0 factor pipeline",
         "stage": "D2.5-B", "verdict": "SUPPORTED (S0-S3 retention 0.5-1.5)",
         "evidence": "outputs/perf_r2d25/transmission/"},
        {"hypothesis": "fusion/readout is where Pt-H2 utility collapses",
         "stage": "D2.5-B", "verdict": "SUPPORTED (S4 retention 0.03-0.21; fixed parent head ~0.00pp)",
         "evidence": "outputs/perf_r2d25/transmission/"},
        {"hypothesis": "structured capacity (independent hop experts) beats the M1 control",
         "stage": "D2.5-C", "verdict": "NOT SUPPORTED (all candidates <= EARLY_MIX on M/T/G)",
         "evidence": "outputs/perf_r2d25/capacity/"},
        {"hypothesis": "independent hop experts beat parameter-matched controls",
         "stage": "D2.5-C", "verdict": "PARTIAL (ACC: sep_concat/inception beat WIDE_B0/CAP_H1_DUP +0.34~0.55pp; F1 mixed)",
         "evidence": "outputs/perf_r2d25/capacity/"},
        {"hypothesis": "generic capacity alone is sufficient",
         "stage": "D2.5-C", "verdict": "REJECTED (WIDE_B0 worst on both metrics)",
         "evidence": "outputs/perf_r2d25/capacity/"},
        {"hypothesis": "fusion depth is the binding bottleneck",
         "stage": "D2.5-C", "verdict": "PARTIAL (deep_fusion only positive variant, +0.45pp F1, 2/3 ds)",
         "evidence": "outputs/perf_r2d25/capacity/"},
        {"hypothesis": "H2 branch is actually used by trained models",
         "stage": "D2.5-C", "verdict": "SUPPORTED (H2-off drops 2.2-8.5pp, 3/3 seeds)",
         "evidence": "outputs/perf_r2d25/capacity/"},
        {"hypothesis": "H1 strong-path dominance starves the H2 branch",
         "stage": "D2.5-C ablations", "verdict": "NOT SUPPORTED (Toys/Grocery H2-off > H1-off) -> D3 path dropout skipped",
         "evidence": "outputs/perf_r2d25/capacity/capacity_mechanism.csv"},
        {"hypothesis": "expert LR unlocks the H2 value",
         "stage": "D2.5-D", "verdict": "NOT SUPPORTED (1e-2: sep_concat +0.26pp 2/3, inception -0.18pp)",
         "evidence": "outputs/perf_r2d25/optimization/"},
        {"hypothesis": "deep supervision (expert-preserving aux heads) helps",
         "stage": "D2.5-D", "verdict": "SUPPORTED (sep_concat +0.35pp 3/3 ds, inception +0.31pp 3/3 ds) -> Gradient Starvation SUPPORTED (partial)",
         "evidence": "outputs/perf_r2d25/optimization/"},
        {"hypothesis": "H1 strong-path dominance starves H2",
         "stage": "D2.5-D pre-check", "verdict": "NOT SUPPORTED (Toys/Grocery h2_off > h1_off) -> D3 path dropout skipped (pre-registered)",
         "evidence": "outputs/perf_r2d25/capacity/capacity_mechanism.csv"},
        {"hypothesis": "hop-token attention converts cross-hop token exchange into gains",
         "stage": "D2.5-E", "verdict": "PARTIAL: vs h1_attention control +1.20pp acc / +1.43pp F1 (3/3 ds) but vs EARLY_MIX only +0.13/+0.00pp (parity)",
         "evidence": "outputs/perf_r2d25/hop_attention/"},
    ]
    with ledger.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hypothesis", "stage", "verdict", "evidence"])
        writer.writeheader()
        for r in ledger_rows:
            writer.writerow(r)

    # The 17-question synthesis is authored at Prompt 7; never clobber a
    # completed diagnosis with the skeleton.
    if not diagnosis.exists() or "skeleton" in diagnosis.read_text():
        diagnosis.write_text(
            "# R2-Design-2.5 — FINAL DIAGNOSIS (skeleton, to be completed at Prompt 7)\n\n"
            "Placeholder written by --stage final; the 17-question synthesis is\n"
            "authored at Prompt 7 after D2.5-D/E completion.\n", encoding="utf-8")
    return master, ledger, diagnosis


# ---------------------------------------------------------------------------
# Later stages (implemented with their prompts)
# ---------------------------------------------------------------------------


def _write_landscape_report() -> Path:
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            p = LANDSCAPE_ROOT / ds / f"seed_{seed}" / "summary.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            rows.extend(data["rows"])
    if not rows:
        raise RuntimeError("no landscape summaries found — run perf_r2d25_landscape.py first")
    fieldnames = ["dataset", "seed", "alpha_pt", "best_val_acc", "best_val_macro_f1",
                  "best_epoch", "best_train_ce", "best_train_acc", "best_val_ce",
                  "d_train_ce", "d_val_ce"]
    csv_path = LANDSCAPE_ROOT / "alpha_landscape.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    grads_path = LANDSCAPE_ROOT / "alpha_gradients.csv"
    with grads_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "seed", "alpha_pt", "d_train_ce", "d_val_ce"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            if r.get("alpha_pt") in ALPHA_GRAD_VALUES:
                writer.writerow(r)
    # ---- verdict computation (pre-registered in the plan) -----------------
    report = LANDSCAPE_ROOT / "R2D25_LANDSCAPE_REPORT.md"
    lines = ["# R2-D2.5-A — alpha_Pt formal objective landscape", ""]
    per_ds = {}
    for ds in DATASETS:
        per_ds[ds] = {"val": {}, "train_ce": {}, "grads": {}}
        for a in ALPHA_PT_VALUES:
            per_ds[ds]["val"][a] = []
            per_ds[ds]["train_ce"][a] = []
        for r in rows:
            if r["dataset"] != ds:
                continue
            a = r["alpha_pt"]
            per_ds[ds]["val"][a].append(r["best_val_acc"])
            per_ds[ds]["train_ce"][a].append(r["best_train_ce"])
    import statistics

    verdicts = {}
    for ds in DATASETS:
        means = {a: statistics.fmean(per_ds[ds]["val"][a]) for a in ALPHA_PT_VALUES}
        base = means[0.0]
        peak = max(ALPHA_PT_VALUES, key=lambda a: means[a])
        gain = means[peak] - base
        n_pos = sum(1 for s in SEEDS if
                    max(per_ds[ds]["val"][a][SEEDS.index(s)] for a in ALPHA_PT_VALUES) >
                    per_ds[ds]["val"][0.0][SEEDS.index(s)] + 1e-9)
        train_ces = {a: statistics.fmean(per_ds[ds]["train_ce"][a]) for a in ALPHA_PT_VALUES}
        train_favors_large = train_ces[1.0] < train_ces[0.0]
        val_favors_large = peak in (0.5, 0.75, 1.0)
        if val_favors_large and train_favors_large and gain >= 0.15 and n_pos >= 2:
            verdicts[ds] = "Optimization Accessibility Failure"
        elif val_favors_large and not train_favors_large and gain >= 0.15 and n_pos >= 2:
            verdicts[ds] = "Objective-Generalization Mismatch"
        elif gain < 0.15 or n_pos < 2:
            verdicts[ds] = "3-seed headroom unstable"
        else:
            verdicts[ds] = "mixed"
        lines.append(
            f"- **{ds}**: peak alpha={peak:g} (gain +{gain:.3f}pp, {n_pos}/3 seeds "
            f"positive); train CE at 0/1.0 = {train_ces[0.0]:.4f}/{train_ces[1.0]:.4f} "
            f"-> **{verdicts[ds]}**"
        )
    lines.append("")
    lines.append("| dataset | alpha | val acc (3-seed mean) | train CE (3-seed mean) |")
    lines.append("|---|---|---|---|")
    for ds in DATASETS:
        for a in ALPHA_PT_VALUES:
            lines.append(
                f"| {ds} | {a:g} | {statistics.fmean(per_ds[ds]['val'][a]):.5f} | "
                f"{statistics.fmean(per_ds[ds]['train_ce'][a]):.4f} |"
            )
    lines.append("")
    lines.append("Aggregate verdict: see per-dataset rows. A dataset-level")
    lines.append("**Optimization Accessibility Failure** means the landscape has")
    lines.append("value that SGD cannot reach; **Objective-Generalization")
    lines.append("Mismatch** means train and val optima disagree.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-Design-2.5 summarizer")
    parser.add_argument("--stage", default="audit",
                        choices=["audit", "landscape", "transmission", "capacity",
                                 "optimization", "hop_attention", "final"])
    args = parser.parse_args()
    if args.stage == "audit":
        report = _write_audit_report(_collect_audit_facts())
        print(f"[audit] wrote {report}")
        return
    if args.stage == "landscape":
        print(f"[landscape] wrote {_write_landscape_report()}")
        return
    if args.stage == "transmission":
        print(f"[transmission] wrote {_write_transmission_report(_collect_transmission())}")
        return
    if args.stage == "capacity":
        print(f"[capacity] wrote {_write_capacity_report(_collect_capacity())}")
        return
    if args.stage == "optimization":
        print(f"[optimization] wrote {_write_optimization_report(_collect_optimization())}")
        return
    if args.stage == "hop_attention":
        print(f"[hop_attention] wrote {_write_hop_attention_report()}")
        return
    if args.stage == "final":
        m, l, d = _write_final_synthesis()
        print(f"[final] wrote {m} / {l} / {d}")
        return



if __name__ == "__main__":
    main()
