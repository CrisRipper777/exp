"""R3-1 Challenge Set summarizer (plan §34/§37/§38, extended per user:
5 datasets + test reporting).

Reads ONLY the on-disk artifacts (results.json / history.csv / train.log /
run_info.json) of outputs/r3/r3_1_challenge and produces:

    docs/r3/R3_1_challenge_results.csv   per-run raw table
    docs/r3/R3_1_challenge_report.md     val + test 3-seed summary + comparisons

Comparison anchors (same commit lineage / protocol / splits, plan §37):
A0 = biaxis_final, DiP = dip, strongest = per-dataset best over the 10
formal models, all from outputs/r2d29/g0_reference (G0, 3 seeds).

Selection protocol: VAL-only (early stopping on val accuracy; test is
evaluated once from the best-val checkpoint and REPORTED, never used for
selection). Integrity checks (plan §38): 3 seeds per cell, test metrics
present (evaluate_test=true), missing runs, failures, NaN mechanism stats.

Usage:
    python scripts/summarize_r3.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_ROOT = PROJECT_ROOT / "outputs" / "r3" / "r3_1_challenge"
G0_ROOT = PROJECT_ROOT / "outputs" / "r2d29" / "g0_reference" / "main"
OUT_CSV = PROJECT_ROOT / "docs" / "r3" / "R3_1_challenge_results.csv"
OUT_MD = PROJECT_ROOT / "docs" / "r3" / "R3_1_challenge_report.md"

VARIANTS = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
FACTORS = ("c", "pt", "pv")
MECH_KEYS = ("diag_norm", "offdiag_norm", "offdiag_diag_ratio", "basis_entropy",
             "c_pt_cos", "c_pv_cos", "pt_pv_cos")

CSV_COLUMNS = [
    "variant", "dataset", "seed", "best_val_acc", "best_val_macro_f1", "best_epoch",
    "test_acc", "test_macro_f1",
    "params", "peak_gpu_mem", "sec_per_epoch",
    "diag_norm", "offdiag_norm", "offdiag_diag_ratio",
    "basis_entropy", "c_pt_cos", "c_pv_cos", "pt_pv_cos",
]


def _read_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _read_history(path: Path) -> tuple[int, float, float] | None:
    """(best_epoch, best_val_acc, best_val_macro_f1) from history.csv."""
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


def _aux_line_at_epoch(train_log: Path, epoch: int) -> dict[str, float] | None:
    """Parse the Aux stats line printed right after the given epoch line."""
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


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _float(value) -> float | None:
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _collect_runs() -> tuple[dict[tuple[str, str, int], dict], list[str]]:
    runs: dict[tuple[str, str, int], dict] = {}
    anomalies: list[str] = []
    failures_path = CHALLENGE_ROOT / "failures.jsonl"
    if failures_path.exists():
        try:
            with failures_path.open(encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    anomalies.append(
                        f"FAILED run: {rec['variant']} {rec['dataset']} seed={rec['seed']} "
                        f"rc={rec['returncode']}"
                    )
        except Exception:  # noqa: BLE001
            pass

    for variant in VARIANTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                outdir = CHALLENGE_ROOT / variant / dataset / f"seed_{seed}"
                key = (variant, dataset, seed)
                results = _read_json(outdir / "hydra" / "results.json")
                if results is None:
                    anomalies.append(f"MISSING results.json: {variant} {dataset} seed={seed}")
                    continue
                if "test_acc" not in results:
                    anomalies.append(f"no test metrics (incomplete run): {variant} {dataset} seed={seed}")

                def _split_metric(payload: dict, name: str) -> float | None:
                    entry = payload.get(name)
                    if isinstance(entry, dict):
                        return _float(entry.get("mean"))
                    return None

                run = {
                    "val_acc": _split_metric(results, "val_acc"),
                    "test_acc": _split_metric(results, "test_acc"),
                    "test_f1": _split_metric(results, "test_macro_f1"),
                }
                hist = _read_history(outdir / "history.csv")
                run["best_epoch"] = hist[0] if hist else None
                run["val_f1"] = hist[2] if hist else None

                run_info = _read_json(outdir / "run_info.json") or {}
                run["params"] = run_info.get("params")
                run["peak_gpu_mem"] = run_info.get("train_peak_gpu_mb")
                if hist and run_info.get("runtime_sec"):
                    try:
                        with (outdir / "history.csv").open(encoding="utf-8") as f:
                            n_rows = sum(1 for _ in f) - 1
                        if n_rows > 0:
                            run["sec_per_epoch"] = run_info["runtime_sec"] / n_rows
                    except Exception:  # noqa: BLE001
                        pass

                aux = _aux_line_at_epoch(outdir / "train.log", run["best_epoch"]) if hist else None
                if aux:
                    diag = [_float(aux.get(f"r3_l2_diag_norm_{n}")) for n in FACTORS]
                    off = [_float(aux.get(f"r3_l2_offdiag_norm_{n}")) for n in FACTORS]
                    ratio = [_float(aux.get(f"r3_l2_offdiag_diag_ratio_{n}")) for n in FACTORS]
                    run["diag_norm"] = _mean([v for v in diag if v is not None])
                    run["offdiag_norm"] = _mean([v for v in off if v is not None])
                    run["offdiag_diag_ratio"] = _mean([v for v in ratio if v is not None])
                    run["basis_entropy"] = _float(aux.get("r3_l2_basis_entropy"))
                    run["c_pt_cos"] = _float(aux.get("r3_l2_cos_c_pt"))
                    run["c_pv_cos"] = _float(aux.get("r3_l2_cos_c_pv"))
                    run["pt_pv_cos"] = _float(aux.get("r3_l2_cos_pt_pv"))
                    run["channels"] = {
                        f"{a}->{b}": _float(aux.get(f"r3_l2_ch_{a}_{b}"))
                        for a in FACTORS for b in FACTORS
                    }
                    run["channels_l1"] = {
                        f"{a}->{b}": _float(aux.get(f"r3_l1_ch_{a}_{b}"))
                        for a in FACTORS for b in FACTORS
                    }
                    for k in MECH_KEYS:
                        if run.get(k) is None:
                            # basis stats are legitimately absent for the
                            # non-basis variants (V0 diagonal / V1 static /
                            # V6 film)
                            if k == "basis_entropy" and variant in ("V0", "V1", "V6"):
                                continue
                            anomalies.append(f"NaN/missing aux stat {k}: {variant} {dataset} seed={seed}")
                else:
                    anomalies.append(f"no Aux line for best epoch: {variant} {dataset} seed={seed}")

                runs[key] = run
    return runs, anomalies


def _g0_anchors() -> dict[str, dict[str, dict[str, float | None]]]:
    """{dataset: {model: {"val": mean, "val_std": std, "test": mean, ...}}}."""
    anchors: dict[str, dict[str, dict[str, float | None]]] = {}
    for dataset in DATASETS:
        ds_dir = G0_ROOT / dataset
        if not ds_dir.is_dir():
            anchors[dataset] = {}
            continue
        anchors[dataset] = {}
        for model_dir in sorted(ds_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            vals, tests = [], []
            for seed in SEEDS:
                res = _read_json(model_dir / f"seed_{seed}" / "hydra" / "results.json")
                if res is None:
                    continue
                v = _float(res["val_acc"]["mean"]) if isinstance(res.get("val_acc"), dict) else None
                t = _float(res["test_acc"]["mean"]) if isinstance(res.get("test_acc"), dict) else None
                if v is not None:
                    vals.append(v)
                if t is not None:
                    tests.append(t)
            if vals:
                anchors[dataset][model_dir.name] = {
                    "val_acc": _mean(vals), "val_acc_std": _std(vals),
                    "test_acc": _mean(tests) if tests else None,
                    "test_macro_f1": None,  # filled below
                    "test_acc_std": _std(tests) if tests else None,
                }
            f1s = []
            for seed in SEEDS:
                res = _read_json(model_dir / f"seed_{seed}" / "hydra" / "results.json")
                if res and isinstance(res.get("test_macro_f1"), dict):
                    f1s.append(float(res["test_macro_f1"]["mean"]))
            if f1s and model_dir.name in anchors[dataset]:
                anchors[dataset][model_dir.name]["test_macro_f1"] = _mean(f1s)
    return anchors


def _write_csv(runs: dict, anomalies: list[str]) -> None:
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
                        f"{r['val_acc'] * 100:.4f}" if r["val_acc"] is not None else "",
                        f"{r['val_f1'] * 100:.4f}" if r["val_f1"] is not None else "",
                        r["best_epoch"] if r["best_epoch"] is not None else "",
                        f"{r['test_acc'] * 100:.4f}" if r["test_acc"] is not None else "",
                        f"{r['test_f1'] * 100:.4f}" if r["test_f1"] is not None else "",
                        r["params"] if r["params"] is not None else "",
                        r["peak_gpu_mem"] if r["peak_gpu_mem"] is not None else "",
                        f"{r['sec_per_epoch']:.2f}" if r.get("sec_per_epoch") else "",
                        f"{r['diag_norm']:.4f}" if r.get("diag_norm") is not None else "",
                        f"{r['offdiag_norm']:.4f}" if r.get("offdiag_norm") is not None else "",
                        f"{r['offdiag_diag_ratio']:.4f}" if r.get("offdiag_diag_ratio") is not None else "",
                        f"{r['basis_entropy']:.4f}" if r.get("basis_entropy") is not None else "",
                        f"{r['c_pt_cos']:.4f}" if r.get("c_pt_cos") is not None else "",
                        f"{r['c_pv_cos']:.4f}" if r.get("c_pv_cos") is not None else "",
                        f"{r['pt_pv_cos']:.4f}" if r.get("pt_pv_cos") is not None else "",
                    ])


def _write_report(runs: dict, anomalies: list[str]) -> None:
    anchors = _g0_anchors()

    def _cell(variant: str, dataset: str, metric: str) -> tuple[float, float] | None:
        vals = []
        for s in SEEDS:
            r = runs.get((variant, dataset, s))
            if r is not None and r.get(metric) is not None:
                vals.append(r[metric] * 100)
        if len(vals) < 3:
            return None
        return _mean(vals), _std(vals)

    def _anchor(dataset: str, model: str, metric: str) -> float | None:
        entry = anchors.get(dataset, {}).get(model)
        if not entry:
            return None
        key = "test_macro_f1" if metric == "test_f1" else metric
        value = entry.get(key)
        # G0 results.json stores fractions; report uses percentage points
        return value * 100 if value is not None else None

    def _strongest(dataset: str, metric: str) -> tuple[str, float]:
        key = "test_macro_f1" if metric == "test_f1" else metric
        best = max(
            ((m, a[key]) for m, a in anchors.get(dataset, {}).items() if a.get(key) is not None),
            key=lambda kv: kv[1],
            default=("?", None),
        )
        value = best[1] * 100 if best[1] is not None else None
        return best[0], value

    lines: list[str] = []
    add = lines.append
    add("# R3-1 Challenge Set Report（Val 选择 + Test 报告）")
    add("")
    add("> 数据源：outputs/r3/r3_1_challenge（R3 V0-V6）+ outputs/r2d29/g0_reference（对照）")
    add("> 协议：task=nc, seeds 42/43/44, 统一 splits/early-stopping；选择全部基于 Val（best-val checkpoint），")
    add("> Test 仅在 checkpoint 冻结后评估一次并报告，不参与选择（计划 §39.5/§39.6）。")
    add("")

    def _summary_table(metric: str, metric_label: str) -> None:
        add(f"## {metric_label} 3-seed 汇总（mean ± std, pp）")
        add("")
        add("| Model/Variant | Movies | Toys | Grocery | ele-fashion | Reddit-S | 5-set avg |")
        add("|---|---|---|---|---|---|---|")
        for model, label in [("biaxis_final", "A0 (G0 ref)"), ("dip", "DiP (G0 ref)"), ("lgmrec", "LGMRec (G0 ref)")]:
            row = []
            vals = []
            for ds in DATASETS:
                v = _anchor(ds, model, metric)
                vals.append(v)
                row.append(_fmt(v))
            row.append(_fmt(_mean([v for v in vals if v is not None])))
            add(f"| {label} | " + " | ".join(row) + " |")
        for variant in VARIANTS:
            cells = [_cell(variant, ds, metric) for ds in DATASETS]
            row = [f"{m:.2f} ± {s:.2f}" if (m is not None and s is not None) else "-" for m, s in cells]
            avg_vals = [c[0] for c in cells if c is not None]
            row.append(f"{_mean(avg_vals):.2f}" if avg_vals else "-")
            add(f"| {variant} | " + " | ".join(row) + " |")
        add("")

        add(f"### {metric_label} deltas（3-seed mean, pp）")
        add("")
        pairs = [("V1", "V0"), ("V2", "V1"), ("V3", "V2"), ("V4", "V3"), ("V5", "V4"), ("V6", "V5")]
        add("| Pair | Movies | Toys | Grocery | ele-fashion | Reddit-S | avg |")
        add("|---|---|---|---|---|---|---|")
        for newer, older in pairs:
            row, deltas = [], []
            for ds in DATASETS:
                c_new, c_old = _cell(newer, ds, metric), _cell(older, ds, metric)
                if c_new and c_old:
                    d = c_new[0] - c_old[0]
                    deltas.append(d)
                    row.append(f"{d:+.2f}")
                else:
                    row.append("-")
            row.append(f"{_mean(deltas):+.2f}" if deltas else "-")
            add(f"| {newer}−{older} | " + " | ".join(row) + " |")
        add("")

        add(f"### {metric_label} vs anchors（3-seed mean delta, pp）")
        add("")
        add("| Variant | Δ vs A0 | Δ vs DiP | Δ vs strongest |")
        add("|---|---|---|---|")
        for variant in VARIANTS:
            cells = [_cell(variant, ds, metric) for ds in DATASETS]
            deltas = {"A0": [], "DiP": [], "strongest": []}
            for ds_idx, ds in enumerate(DATASETS):
                if cells[ds_idx] is None:
                    continue
                for key, model in [("A0", "biaxis_final"), ("DiP", "dip")]:
                    a = _anchor(ds, model, metric)
                    if a is not None:
                        deltas[key].append(cells[ds_idx][0] - a)
                _, s_val = _strongest(ds, metric)
                if s_val is not None:
                    deltas["strongest"].append(cells[ds_idx][0] - s_val)
            add(
                f"| {variant} | {_fmt(_mean(deltas['A0'])) if deltas['A0'] else '-'} | "
                f"{_fmt(_mean(deltas['DiP'])) if deltas['DiP'] else '-'} | "
                f"{_fmt(_mean(deltas['strongest'])) if deltas['strongest'] else '-'} |"
            )
        add("")

    _summary_table("val_acc", "A. Val Acc")
    _summary_table("test_acc", "B. Test Acc")
    _summary_table("test_f1", "C. Test Macro-F1")

    add("## D. 9-channel transition strength（best-epoch, layer 2, 3-seed mean）")
    add("")
    for variant in VARIANTS:
        add(f"### {variant}")
        add("")
        add("| ds\\ch | C→C | Pt→C | Pv→C | C→Pt | Pt→Pt | Pv→Pt | C→Pv | Pt→Pv | Pv→Pv |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for ds in DATASETS:
            row = [ds]
            for a in FACTORS:
                for b in FACTORS:
                    vals = [
                        runs[(variant, ds, s)].get("channels", {}).get(f"{a}->{b}")
                        for s in SEEDS
                        if (variant, ds, s) in runs
                    ]
                    vals = [v for v in vals if v is not None]
                    row.append(_fmt(_mean(vals), 3) if vals else "-")
            add("| " + " | ".join(row) + " |")
        add("")

    add("## E. Basis utilization + ownership cosines（best-epoch, layer 2, 3-seed mean）")
    add("")
    add("| Variant | basis_entropy | cos(C,Pt) | cos(C,Pv) | cos(Pt,Pv) | offdiag/diag ratio |")
    add("|---|---|---|---|---|---|")
    for variant in VARIANTS:
        stats: dict[str, list[float]] = {k: [] for k in MECH_KEYS}
        for ds in DATASETS:
            for s in SEEDS:
                r = runs.get((variant, ds, s))
                if r is None:
                    continue
                for key in stats:
                    if r.get(key) is not None:
                        stats[key].append(r[key])
        add(
            f"| {variant} | {_fmt(_mean(stats['basis_entropy']), 3)} | "
            f"{_fmt(_mean(stats['c_pt_cos']), 3)} | {_fmt(_mean(stats['c_pv_cos']), 3)} | "
            f"{_fmt(_mean(stats['pt_pv_cos']), 3)} | {_fmt(_mean(stats['offdiag_diag_ratio']), 3)} |"
        )
    add("")

    add("## F. Efficiency（3-seed mean）")
    add("")
    add("| Variant | params | peak GPU mem (MB) | sec/epoch |")
    add("|---|---|---|---|")
    for variant in VARIANTS:
        vruns = [r for (v, d, s), r in runs.items() if v == variant]
        params = [r["params"] for r in vruns if r.get("params")]
        mems = [r["peak_gpu_mem"] for r in vruns if r.get("peak_gpu_mem")]
        secs = [r.get("sec_per_epoch") for r in vruns if r.get("sec_per_epoch")]
        add(
            f"| {variant} | {int(_mean(params)) if params else '-'} | "
            f"{_fmt(_mean(mems), 0)} | {_fmt(_mean(secs), 2)} |"
        )
    add("")

    add("## G. Anomalies")
    add("")
    if anomalies:
        for a in anomalies:
            add(f"- {a}")
    else:
        add("- （无）")
    add("")
    add("## H. 原始 evidence 说明")
    add("")
    add("- 每 run 的 best_val_acc/best_val_macro_f1/best_epoch 取自 history.csv（argmax val_acc）；")
    add("  test_acc/test_macro_f1 取自 results.json（best-val checkpoint 冻结后单次评估）。")
    add("- 机制 stats 取自 train.log 中 best-epoch 的 Aux 行（训练步统计，detached）。")
    add("- 本报告只给 raw evidence，不做论文结论（计划 §34 G）。")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    runs, anomalies = _collect_runs()
    _write_csv(runs, anomalies)
    _write_report(runs, anomalies)
    n = len(runs)
    total = len(VARIANTS) * len(DATASETS) * len(SEEDS)
    print(f"summarized {n}/{total} runs, {len(anomalies)} anomalies")
    print(f"csv -> {OUT_CSV}")
    print(f"md  -> {OUT_MD}")


if __name__ == "__main__":
    main()
