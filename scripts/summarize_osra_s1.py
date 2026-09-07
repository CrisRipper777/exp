"""OSRA S1 summarizer (plan §30/§27).

Reads outputs/osra/s1_local artifacts and produces
docs/osra/S1_local_parent_results.csv + S1_local_parent_report.md with the
gate check: best parent M/T/G Avg Val >= A0 - 0.10pp AND >= 2/3 datasets
not weaker than A0 (plan §8). Anchors = locked S0 values (A0/DiP 3-seed
val from outputs/r2d29/g0_reference). Val-only; no test access expected.

Usage:
    python scripts/summarize_osra_s1.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
S1_ROOT = PROJECT_ROOT / "outputs" / "osra" / "s1_local"
OUT_CSV = PROJECT_ROOT / "docs" / "osra" / "S1_local_parent_results.csv"
OUT_MD = PROJECT_ROOT / "docs" / "osra" / "S1_local_parent_report.md"

VARIANTS = ["L0", "L1", "L2", "L3"]
DATASETS = ["Movies", "Toys", "Grocery"]
SEEDS = [42, 43, 44]

# S0-locked anchors (3-seed val acc, pp)
A0_ANCHOR = {"Movies": 55.4489, "Toys": 79.1898, "Grocery": 83.1625}
DIP_ANCHOR = {"Movies": 56.3587, "Toys": 80.3898, "Grocery": 84.1776}
GATE_EPS = 0.10  # plan §8: M/T/G avg >= A0 - 0.10pp


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


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _collect() -> tuple[dict, list[str]]:
    runs: dict[tuple[str, str, int], dict] = {}
    anomalies: list[str] = []
    failures = S1_ROOT / "failures.jsonl"
    if failures.exists():
        with failures.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                anomalies.append(f"FAILED run: {rec['variant']} {rec['dataset']} seed={rec['seed']} rc={rec['returncode']}")
    for variant in VARIANTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                outdir = S1_ROOT / variant / dataset / f"seed_{seed}"
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
                runs[key] = {
                    "val": float(val) * 100 if val is not None else None,
                    "best_epoch": hist[0] if hist else None,
                    "val_f1": hist[2] * 100 if hist else None,
                    "params": info.get("params"),
                    "peak_mem": info.get("train_peak_gpu_mb"),
                    "runtime_sec": info.get("runtime_sec"),
                }
    return runs, anomalies


def _fmt(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def main() -> None:
    runs, anomalies = _collect()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant", "dataset", "seed", "best_val_acc", "best_val_macro_f1", "best_epoch",
            "params", "peak_gpu_mem", "runtime_sec",
        ])
        for variant in VARIANTS:
            for dataset in DATASETS:
                for seed in SEEDS:
                    r = runs.get((variant, dataset, seed))
                    if r is None:
                        writer.writerow([variant, dataset, seed] + [""] * 6)
                        continue
                    writer.writerow([
                        variant, dataset, seed,
                        f"{r['val']:.4f}" if r["val"] is not None else "",
                        f"{r['val_f1']:.4f}" if r["val_f1"] is not None else "",
                        r["best_epoch"] if r["best_epoch"] is not None else "",
                        r["params"] if r["params"] is not None else "",
                        r["peak_mem"] if r["peak_mem"] is not None else "",
                        f"{r['runtime_sec']:.1f}" if r["runtime_sec"] is not None else "",
                    ])

    def cell(variant, dataset):
        vals = [runs[(variant, dataset, s)]["val"] for s in SEEDS
                if (variant, dataset, s) in runs and runs[(variant, dataset, s)]["val"] is not None]
        return (_mean(vals), _std(vals)) if len(vals) == 3 else (None, None)

    lines: list[str] = []
    add = lines.append
    add("# OSRA S1 — Strong Local Parent Report（Val-only）")
    add("")
    add("> 协议：M/T/G × seeds 42/43/44，task=nc，evaluate_test=false，unified protocol（S0 冻结）")
    add("> Anchors（S0 锁定，3-seed val, pp）：A0 = 55.45/79.19/83.16 (avg 72.60)；DiP = 56.36/80.39/84.18 (avg 73.64)")
    add("")
    add("## 1. 3-seed Val Acc 汇总（mean ± std, pp）")
    add("")
    add("| Variant | Movies | Toys | Grocery | M/T/G avg | Δ vs A0 avg |")
    add("|---|---|---|---|---|---|")
    add(f"| A0 (anchor) | 55.45 | 79.19 | 83.16 | 72.60 | — |")
    add(f"| DiP (anchor) | 56.36 | 80.39 | 84.18 | 73.64 | — |")
    results_by_variant = {}
    for variant in VARIANTS:
        cells = [cell(variant, ds) for ds in DATASETS]
        row = [f"{m:.2f} ± {s:.2f}" if m is not None else "-" for m, s in cells]
        means = [c[0] for c in cells if c[0] is not None]
        avg = _mean(means)
        results_by_variant[variant] = (cells, avg)
        delta = (avg - (55.4489 + 79.1898 + 83.1625) / 3) if avg is not None else None
        row.append(_fmt(avg) if avg is not None else "-")
        row.append(f"{delta:+.2f}" if delta is not None else "-")
        add(f"| {variant} | " + " | ".join(row) + " |")
    add("")

    add("## 2. 每数据集 Δ vs A0 / DiP（3-seed mean, pp）")
    add("")
    add("| Variant | ΔMovies vs A0 | ΔToys vs A0 | ΔGrocery vs A0 | ΔMovies vs DiP | ΔToys vs DiP | ΔGrocery vs DiP |")
    add("|---|---|---|---|---|---|---|")
    for variant in VARIANTS:
        cells, _avg = results_by_variant[variant]
        d_a0 = [cells[i][0] - A0_ANCHOR[ds] if cells[i][0] is not None else None for i, ds in enumerate(DATASETS)]
        d_dip = [cells[i][0] - DIP_ANCHOR[ds] if cells[i][0] is not None else None for i, ds in enumerate(DATASETS)]
        add(f"| {variant} | " + " | ".join(
            f"{d:+.2f}" if d is not None else "-" for d in d_a0
        ) + " | " + " | ".join(f"{d:+.2f}" if d is not None else "-" for d in d_dip) + " |")
    add("")

    add("## 3. Val Macro-F1（3-seed mean, pp）")
    add("")
    add("| Variant | Movies | Toys | Grocery |")
    add("|---|---|---|---|")
    for variant in VARIANTS:
        row = []
        for ds in DATASETS:
            vals = [runs[(variant, ds, s)]["val_f1"] for s in SEEDS
                    if (variant, ds, s) in runs and runs[(variant, ds, s)]["val_f1"] is not None]
            row.append(f"{_mean(vals):.2f} ± {_std(vals):.2f}" if len(vals) == 3 else "-")
        add(f"| {variant} | " + " | ".join(row) + " |")
    add("")

    add("## 4. 效率（3-seed mean）")
    add("")
    add("| Variant | params | peak GPU mem (MB) | runtime (s) |")
    add("|---|---|---|---|")
    for variant in VARIANTS:
        vr = [r for (v, d, s), r in runs.items() if v == variant]
        params = [r["params"] for r in vr if r.get("params")]
        mems = [r["peak_mem"] for r in vr if r.get("peak_mem")]
        rt = [r["runtime_sec"] for r in vr if r.get("runtime_sec")]
        add(f"| {variant} | {int(_mean(params)) if params else '-'} | {_fmt(_mean(mems), 0)} | {_fmt(_mean(rt), 1)} |")
    add("")

    add("## 5. Anomalies")
    add("")
    if anomalies:
        for a in anomalies:
            add(f"- {a}")
    else:
        add("- （无）")
    add("")

    # ---- gate (plan §8) ----
    add("## 6. S1 Gate 判定（计划 §8）")
    add("")
    best_name, best_avg, best_cells = None, None, None
    for variant in VARIANTS:
        cells, avg = results_by_variant[variant]
        if avg is not None and (best_avg is None or avg > best_avg):
            best_name, best_avg, best_cells = variant, avg, cells
    if best_avg is None:
        add("- **数据不足，无法判定。**")
    else:
        a0_avg = (55.4489 + 79.1898 + 83.1625) / 3
        threshold = a0_avg - GATE_EPS
        weak_count = sum(
            1 for i, ds in enumerate(DATASETS)
            if best_cells[i][0] is not None and best_cells[i][0] < A0_ANCHOR[ds]
        )
        ok_avg = best_avg >= threshold
        ok_two_thirds = weak_count <= 1
        verdict = "GO" if ok_avg and ok_two_thirds else "NO-GO"
        strong = best_avg >= a0_avg
        add(f"- 最佳 parent：**{best_name}**，M/T/G Avg Val = **{best_avg:.2f}**（门槛 ≥ {threshold:.2f}）")
        add(f"- 弱于 A0 的数据集数：{weak_count}/3（要求 ≤1）")
        add(f"- 判定：**{verdict}**" + ("（Strong GO：≥ A0 avg）" if strong and verdict == "GO" else ""))
        if verdict == "NO-GO":
            add("- NO-GO：按计划 §8 暂停 cross-ownership Query，先修 Local parent。")
        else:
            add("- GO：按计划 §9 进入 S2（Ownership-Specific Relational Access）。")
    add("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n = len(runs)
    total = len(VARIANTS) * len(DATASETS) * len(SEEDS)
    print(f"summarized {n}/{total} runs, {len(anomalies)} anomalies")
    print(f"csv -> {OUT_CSV}")
    print(f"md  -> {OUT_MD}")


if __name__ == "__main__":
    main()
