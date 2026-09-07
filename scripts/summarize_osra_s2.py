"""OSRA S2 summarizer (plan §32/§33 Prompt D + §27).

Reads outputs/osra/s2_access artifacts and produces
docs/osra/S2_relational_access_results.csv + S2_relational_access_report.md:
per-run raw table, 3-seed mean±std, paired deltas (Q1-Q0 ... Q4-Q0),
vs A0/DiP, Macro-F1, null mass, 6 source-ownership masses, N_eff,
writeback ratio, degree-bin access stats, anomalies. Anchors = S0-locked
A0/DiP 3-seed val. Val-only protocol; no test access.

Usage:
    python scripts/summarize_osra_s2.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
S2_ROOT = PROJECT_ROOT / "outputs" / "osra" / "s2_access"
OUT_CSV = PROJECT_ROOT / "docs" / "osra" / "S2_relational_access_results.csv"
OUT_MD = PROJECT_ROOT / "docs" / "osra" / "S2_relational_access_report.md"

VARIANTS = ["Q0", "Q1", "Q2", "Q3", "Q4"]
DATASETS = ["Movies", "Toys", "Grocery"]
SEEDS = [42, 43, 44]
FACTORS = ("c", "pt", "pv")
BINS = ("d1_5", "d6_10", "d11_20", "d20p")

# S0-locked anchors (3-seed val acc, pp)
A0_ANCHOR = {"Movies": 55.4489, "Toys": 79.1898, "Grocery": 83.1625}
DIP_ANCHOR = {"Movies": 56.3587, "Toys": 80.3898, "Grocery": 84.1776}

CSV_COLUMNS = [
    "variant", "dataset", "seed", "best_val_acc", "best_val_macro_f1", "best_epoch",
    "params", "peak_gpu_mem", "sec_per_epoch",
    "nullmass_c", "nullmass_pt", "nullmass_pv",
    "srcmass_c_pt", "srcmass_c_pv", "srcmass_pt_c", "srcmass_pt_pv", "srcmass_pv_c", "srcmass_pv_pt",
    "neff_c", "neff_pt", "neff_pv",
    "writeback_ratio_c", "writeback_ratio_pt", "writeback_ratio_pv",
    "nullmass_c_d1_5", "nullmass_c_d20p",
]


def _read_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _best_from_history(path: Path) -> tuple[int, float, float] | None:
    try:
        with path.open(encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)]
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    best_epoch = int(rows[0]["epoch"])
    best_acc = float(rows[0]["val_acc"])
    best_f1 = float(rows[0]["val_macro_f1"])
    for row in rows[1:]:
        acc = float(row["val_acc"])
        if acc > best_acc:
            best_acc = acc
            best_f1 = float(row["val_macro_f1"])
            best_epoch = int(row["epoch"])
    return best_epoch, best_acc, best_f1


def _aux_at_epoch(train_log: Path, epoch: int) -> dict[str, float] | None:
    try:
        lines = train_log.read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001
        return None
    for idx, line in enumerate(lines):
        m = re.search(r"Epoch (\d+) \| Train Loss", line)
        if not m or int(m.group(1)) != epoch:
            continue
        for nxt in lines[idx + 1 : idx + 3]:
            aux_pos = nxt.find("Aux ")
            if aux_pos < 0:
                continue
            stats: dict[str, float] = {}
            for token in nxt[aux_pos + 4 :].split(" | "):
                parts = token.split(" ")
                if len(parts) >= 2:
                    try:
                        stats[parts[0]] = float(parts[1])
                    except ValueError:
                        continue
            return stats
    return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _slot_mean(aux: dict[str, float], prefix: str, factor: str) -> float | None:
    vals = []
    for k in (0, 1):
        v = aux.get(f"{prefix}_{factor}_k{k}")
        if v is not None:
            vals.append(v)
    return _mean(vals) if vals else None


def _slot_mean_bin(aux: dict[str, float], factor: str, bin_name: str) -> float | None:
    """Degree-bin null-mass keys embed the factor BEFORE the bin name:
    osra_l2_nullmass_{factor}_{bin}_k{slot}."""
    vals = []
    for k in (0, 1):
        v = aux.get(f"osra_l2_nullmass_{factor}_{bin_name}_k{k}")
        if v is not None:
            vals.append(v)
    return _mean(vals) if vals else None


def _collect() -> tuple[dict, list[str]]:
    runs: dict[tuple[str, str, int], dict] = {}
    anomalies: list[str] = []
    failures = S2_ROOT / "failures.jsonl"
    if failures.exists():
        with failures.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                anomalies.append(f"FAILED run: {rec['variant']} {rec['dataset']} seed={rec['seed']} rc={rec['returncode']}")
    for variant in VARIANTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                outdir = S2_ROOT / variant / dataset / f"seed_{seed}"
                key = (variant, dataset, seed)
                results = _read_json(outdir / "hydra" / "results.json")
                if results is None:
                    anomalies.append(f"MISSING results.json: {variant} {dataset} seed={seed}")
                    continue
                if "test_acc" in results:
                    anomalies.append(f"TEST ACCESS DETECTED: {variant} {dataset} seed={seed}")
                val = results.get("val_acc", {}).get("mean")
                hist = _best_from_history(outdir / "history.csv")
                info = _read_json(outdir / "run_info.json") or {}
                run = {
                    "val": float(val) * 100 if val is not None else None,
                    "best_epoch": hist[0] if hist else None,
                    "val_f1": hist[2] * 100 if hist else None,
                    "params": info.get("params"),
                    "peak_mem": info.get("train_peak_gpu_mb"),
                    "runtime_sec": info.get("runtime_sec"),
                }
                if hist and info.get("runtime_sec"):
                    try:
                        with (outdir / "history.csv").open(encoding="utf-8") as f:
                            n_rows = sum(1 for _ in f) - 1
                        if n_rows > 0:
                            run["sec_per_epoch"] = info["runtime_sec"] / n_rows
                    except Exception:  # noqa: BLE001
                        pass
                aux = _aux_at_epoch(outdir / "train.log", run["best_epoch"]) if hist else None
                if aux and variant != "Q0":
                    for fct in FACTORS:
                        run[f"nullmass_{fct}"] = _slot_mean(aux, "osra_l2_nullmass", fct)
                        run[f"neff_{fct}"] = _slot_mean(aux, "osra_l2_neff", fct)
                        run[f"writeback_ratio_{fct}"] = aux.get(f"osra_l2_writeback_ratio_{fct}")
                        run[f"nullmass_{fct}_d1_5"] = _slot_mean_bin(aux, fct, "d1_5")
                        run[f"nullmass_{fct}_d20p"] = _slot_mean_bin(aux, fct, "d20p")
                    for a in FACTORS:
                        for b in FACTORS:
                            if a != b:
                                run[f"srcmass_{a}_{b}"] = aux.get(f"osra_l2_srcmass_{a}_{b}")
                    if variant == "Q4" and run.get("nullmass_c") is None:
                        anomalies.append(f"missing null stats: {variant} {dataset} seed={seed}")
                runs[key] = run
    return runs, anomalies


def _fmt(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def main() -> None:
    runs, anomalies = _collect()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for variant in VARIANTS:
            for dataset in DATASETS:
                for seed in SEEDS:
                    r = runs.get((variant, dataset, seed))
                    if r is None:
                        writer.writerow([variant, dataset, seed] + [""] * (len(CSV_COLUMNS) - 3))
                        continue
                    writer.writerow([
                        variant, dataset, seed,
                        f"{r['val']:.4f}" if r["val"] is not None else "",
                        f"{r['val_f1']:.4f}" if r["val_f1"] is not None else "",
                        r["best_epoch"] if r["best_epoch"] is not None else "",
                        r["params"] if r["params"] is not None else "",
                        r["peak_mem"] if r["peak_mem"] is not None else "",
                        f"{r['sec_per_epoch']:.3f}" if r.get("sec_per_epoch") else "",
                        _fmt(r.get("nullmass_c"), 4), _fmt(r.get("nullmass_pt"), 4), _fmt(r.get("nullmass_pv"), 4),
                        _fmt(r.get("srcmass_c_pt"), 4), _fmt(r.get("srcmass_c_pv"), 4),
                        _fmt(r.get("srcmass_pt_c"), 4), _fmt(r.get("srcmass_pt_pv"), 4),
                        _fmt(r.get("srcmass_pv_c"), 4), _fmt(r.get("srcmass_pv_pt"), 4),
                        _fmt(r.get("neff_c"), 3), _fmt(r.get("neff_pt"), 3), _fmt(r.get("neff_pv"), 3),
                        _fmt(r.get("writeback_ratio_c"), 4), _fmt(r.get("writeback_ratio_pt"), 4), _fmt(r.get("writeback_ratio_pv"), 4),
                        _fmt(r.get("nullmass_c_d1_5"), 4), _fmt(r.get("nullmass_c_d20p"), 4),
                    ])

    def cell(variant, dataset, metric="val"):
        vals = [runs[(variant, dataset, s)][metric] for s in SEEDS
                if (variant, dataset, s) in runs and runs[(variant, dataset, s)].get(metric) is not None]
        return (_mean(vals), _std(vals)) if len(vals) == 3 else (None, None)

    lines: list[str] = []
    add = lines.append
    add("# OSRA S2 — Relational Access Report（Val-only）")
    add("")
    add("> 协议：M/T/G × seeds 42/43/44，task=nc，evaluate_test=false（S0 冻结 unified protocol）")
    add("> Parent：S1 最强 Local parent = L3（factor-wise GATv2，M/T/G val 72.24）")
    add("> **S1 gate 前提声明**：S1 未达 GO 门槛（72.24 vs 72.50，A0−0.10），处于灰色区（未达 NO-GO 带 −0.4~−0.5）；按用户\"自动推进 S0-S2\"指示继续执行，Q4−Q0 按计划 §16 门槛判定")
    add("> Anchors（S0 锁定）：A0 = 55.45/79.19/83.16 (avg 72.60)；DiP = 56.36/80.39/84.18 (avg 73.64)")
    add("")
    add("## 1. 3-seed Val Acc 汇总（mean ± std, pp）")
    add("")
    add("| Variant | Movies | Toys | Grocery | M/T/G avg | Δ vs A0 avg | Δ vs DiP avg |")
    add("|---|---|---|---|---|---|---|")
    add("| A0 (anchor) | 55.45 | 79.19 | 83.16 | 72.60 | — | — |")
    add("| DiP (anchor) | 56.36 | 80.39 | 84.18 | 73.64 | — | — |")
    results_by_variant = {}
    for variant in VARIANTS:
        cells = [cell(variant, ds) for ds in DATASETS]
        row = [f"{m:.2f} ± {s:.2f}" if m is not None else "-" for m, s in cells]
        means = [c[0] for c in cells if c[0] is not None]
        avg = _mean(means)
        results_by_variant[variant] = (cells, avg)
        d_a0 = (avg - 72.60) if avg is not None else None
        d_dip = (avg - 73.64) if avg is not None else None
        row.append(_fmt(avg))
        row.append(f"{d_a0:+.2f}" if d_a0 is not None else "-")
        row.append(f"{d_dip:+.2f}" if d_dip is not None else "-")
        add(f"| {variant} | " + " | ".join(row) + " |")
    add("")

    add("## 2. Paired deltas（3-seed mean, pp）")
    add("")
    pairs = [("Q1", "Q0"), ("Q2", "Q1"), ("Q3", "Q2"), ("Q4", "Q3"), ("Q4", "Q0")]
    add("| Pair | Movies | Toys | Grocery | avg |")
    add("|---|---|---|---|---|")
    for newer, older in pairs:
        row, deltas = [], []
        for ds in DATASETS:
            c_new, c_old = cell(newer, ds), cell(older, ds)
            if c_new[0] is not None and c_old[0] is not None:
                d = c_new[0] - c_old[0]
                deltas.append(d)
                row.append(f"{d:+.2f}")
            else:
                row.append("-")
        row.append(f"{_mean(deltas):+.2f}" if deltas else "-")
        add(f"| {newer}−{older} | " + " | ".join(row) + " |")
    add("")

    add("## 3. 每数据集 Δ vs A0 / DiP（3-seed mean, pp）")
    add("")
    add("| Variant | ΔMovies vs A0 | ΔToys vs A0 | ΔGrocery vs A0 | ΔMovies vs DiP | ΔToys vs DiP | ΔGrocery vs DiP |")
    add("|---|---|---|---|---|---|---|")
    for variant in VARIANTS:
        cells, _avg = results_by_variant[variant]
        d_a0 = [cells[i][0] - A0_ANCHOR[ds] if cells[i][0] is not None else None for i, ds in enumerate(DATASETS)]
        d_dip = [cells[i][0] - DIP_ANCHOR[ds] if cells[i][0] is not None else None for i, ds in enumerate(DATASETS)]
        add(f"| {variant} | " + " | ".join(f"{d:+.2f}" if d is not None else "-" for d in d_a0)
            + " | " + " | ".join(f"{d:+.2f}" if d is not None else "-" for d in d_dip) + " |")
    add("")

    add("## 4. Val Macro-F1（3-seed mean, pp）")
    add("")
    add("| Variant | Movies | Toys | Grocery |")
    add("|---|---|---|---|")
    for variant in VARIANTS:
        row = []
        for ds in DATASETS:
            c = cell(variant, ds, "val_f1")
            row.append(f"{c[0]:.2f} ± {c[1]:.2f}" if c[0] is not None else "-")
        add(f"| {variant} | " + " | ".join(row) + " |")
    add("")

    add("## 5. 机制统计（best-epoch, layer 2, 3-seed mean）")
    add("")
    add("### 5.1 Null mass（E[α_∅]，slot 平均）")
    add("")
    add("| Variant | Movies C/Pt/Pv | Toys C/Pt/Pv | Grocery C/Pt/Pv |")
    add("|---|---|---|---|")
    for variant in VARIANTS[1:]:
        row = []
        for ds in DATASETS:
            parts = []
            for fct in FACTORS:
                vals = [runs[(variant, ds, s)].get(f"nullmass_{fct}") for s in SEEDS
                        if (variant, ds, s) in runs]
                vals = [v for v in vals if v is not None]
                parts.append(_fmt(_mean(vals), 3) if vals else "-")
            row.append("/".join(parts))
        add(f"| {variant} | " + " | ".join(row) + " |")
    add("")
    add("### 5.2 六个 source-ownership masses A_{a→b}（3-seed mean）")
    add("")
    add("| Variant | C→Pt | C→Pv | Pt→C | Pt→Pv | Pv→C | Pv→Pt |")
    add("|---|---|---|---|---|---|---|")
    for variant in VARIANTS[1:]:
        row = []
        for a in FACTORS:
            for b in FACTORS:
                if a == b:
                    continue
                vals = [runs[(variant, ds, s)].get(f"srcmass_{a}_{b}") for ds in DATASETS for s in SEEDS
                        if (variant, ds, s) in runs]
                vals = [v for v in vals if v is not None]
                row.append(_fmt(_mean(vals), 3) if vals else "-")
        add(f"| {variant} | " + " | ".join(row) + " |")
    add("")
    add("### 5.3 N_eff / writeback ratio / degree-bin null mass（3-seed mean）")
    add("")
    add("| Variant | N_eff C/Pt/Pv | writeback C/Pt/Pv | nullmass C d1-5 / d20+ |")
    add("|---|---|---|---|")
    for variant in VARIANTS[1:]:
        neff, wb, degbin = [], [], []
        for fct in FACTORS:
            vals = [runs[(variant, ds, s)].get(f"neff_{fct}") for ds in DATASETS for s in SEEDS
                    if (variant, ds, s) in runs]
            vals = [v for v in vals if v is not None]
            neff.append(_fmt(_mean(vals), 2) if vals else "-")
            vals = [runs[(variant, ds, s)].get(f"writeback_ratio_{fct}") for ds in DATASETS for s in SEEDS
                    if (variant, ds, s) in runs]
            vals = [v for v in vals if v is not None]
            wb.append(_fmt(_mean(vals), 3) if vals else "-")
        v1 = [runs[(variant, ds, s)].get("nullmass_c_d1_5") for ds in DATASETS for s in SEEDS
              if (variant, ds, s) in runs]
        v2 = [runs[(variant, ds, s)].get("nullmass_c_d20p") for ds in DATASETS for s in SEEDS
              if (variant, ds, s) in runs]
        v1 = [v for v in v1 if v is not None]
        v2 = [v for v in v2 if v is not None]
        add(f"| {variant} | " + "/".join(neff) + " | " + "/".join(wb) + " | "
            + f"{_fmt(_mean(v1), 3) if v1 else '-'} / {_fmt(_mean(v2), 3) if v2 else '-'} |")
    add("")

    add("## 6. 效率（3-seed mean）")
    add("")
    add("| Variant | params | peak GPU mem (MB) | sec/epoch |")
    add("|---|---|---|---|")
    for variant in VARIANTS:
        vr = [r for (v, d, s), r in runs.items() if v == variant]
        params = [r["params"] for r in vr if r.get("params")]
        mems = [r["peak_mem"] for r in vr if r.get("peak_mem")]
        secs = [r.get("sec_per_epoch") for r in vr if r.get("sec_per_epoch")]
        add(f"| {variant} | {int(_mean(params)) if params else '-'} | {_fmt(_mean(mems), 0)} | {_fmt(_mean(secs), 3)} |")
    add("")

    add("## 7. S2 Gate（计划 §16）")
    add("")
    q0_cells = [cell("Q0", ds) for ds in DATASETS]
    q4_cells = [cell("Q4", ds) for ds in DATASETS]
    deltas = [q4_cells[i][0] - q0_cells[i][0] for i in range(3)
              if q4_cells[i][0] is not None and q0_cells[i][0] is not None]
    pos = sum(1 for d in deltas if d > 0)
    avg_d = _mean(deltas) if deltas else None
    q4_avg = results_by_variant["Q4"][1]
    add(f"- Q4−Q0（M/T/G avg）= {avg_d:+.2f} pp" if avg_d is not None else "- Q4−Q0 = 数据不足")
    add(f"- 正数据集数：{pos}/3")
    if avg_d is None:
        verdict = "数据不足"
    elif avg_d >= 0.5:
        verdict = "Strong GO"
    elif avg_d >= 0.3:
        verdict = "GO"
    elif avg_d > 0:
        verdict = "Weak（只做机制诊断，不加外围模块）"
    else:
        verdict = "NO-GO（重新讨论 Query–Compress–Write family，计划 §16）"
    add(f"- 判定：**{verdict}**")
    add(f"- Q4 M/T/G avg = {q4_avg:.2f}（DiP = 73.64）")
    add("")

    add("## 8. Anomalies")
    add("")
    if anomalies:
        for a in anomalies:
            add(f"- {a}")
    else:
        add("- （无）")
    add("")
    add("## 9. 原始 evidence 说明")
    add("")
    add("- best_val_acc/best_val_macro_f1/best_epoch 取自 history.csv（argmax val_acc）；机制 stats 取自 train.log best-epoch 的 Aux 行。")
    add("- 本报告只给 raw evidence，不做论文结论。")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n = len(runs)
    total = len(VARIANTS) * len(DATASETS) * len(SEEDS)
    print(f"summarized {n}/{total} runs, {len(anomalies)} anomalies")
    print(f"csv -> {OUT_CSV}")
    print(f"md  -> {OUT_MD}")


if __name__ == "__main__":
    main()
