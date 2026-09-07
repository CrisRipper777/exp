"""R2D29 G0 summarizer: rebuild the NC Validation reference table.

Reads outputs/r2d29/g0_reference/main/<dataset>/<model>/seed_<s>/ and emits
the plan §5.3 files into outputs/r2d29/g0_reference/:

  runs.csv                        (per-seed raw: best_epoch/val_acc/val_f1/params/mem/time/commit)
  aggregate.csv                   (per model x dataset mean/std)
  strongest_baseline_by_dataset.csv (strongest external + strongest overall)
  resources.csv                   (per model x dataset params/peak mem/time)
  G0_REFERENCE_REPORT.md

Val-only: best epoch is the epoch maximizing val acc; val F1 is read from
the train log at that epoch. Test metrics are never used for any ranking.

Usage:
    python scripts/summarize_perf_r2d29_g0.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs" / "r2d29" / "g0_reference"
MAIN = OUT_ROOT / "main"

MODELS = ["mlp", "gcn", "sage", "mmgcn", "mgat", "dmgc", "dgf", "dip", "lgmrec", "biaxis_final"]
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]


def _parse_log(log_path: Path) -> tuple[float | None, float | None, int | None, int | None]:
    """Return (best_val_acc, val_f1_at_best, best_epoch, params) from train.log."""
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    params = None
    match = re.search(r"model\+head params=(\d+)", text)
    if match:
        params = int(match.group(1))
    current_epoch = None
    best_acc, best_f1, best_epoch = -1.0, None, None
    for line in text.splitlines():
        em = re.search(r"Epoch (\d+)", line)
        if em:
            current_epoch = int(em.group(1))
            continue
        vm = re.search(r"Val Acc ([\d.]+) \| Val F1 ([\d.]+)", line)
        if vm and current_epoch is not None:
            acc = float(vm.group(1))
            if acc > best_acc:
                best_acc, best_f1, best_epoch = acc, float(vm.group(2)), current_epoch
    if best_epoch is None:
        return None, None, None, params
    return best_acc, best_f1, best_epoch, params


def _load_runs() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                outdir = MAIN / dataset / model / f"seed_{seed}"
                results_json = outdir / "hydra" / "results.json"
                run_info = outdir / "run_info.json"
                row = {
                    "model": model, "dataset": dataset, "seed": seed,
                    "status": "ok",
                    "best_epoch": None, "val_acc": None, "val_f1": None,
                    "test_acc": None, "test_f1": None,
                    "param_count": None, "peak_mem_mb": None,
                    "train_seconds": None, "git_commit": None,
                }
                if not results_json.exists():
                    row["status"] = "missing"
                    rows.append(row)
                    continue
                try:
                    res = json.loads(results_json.read_text(encoding="utf-8"))
                    row["val_acc"] = res["val_acc"]["mean"] * 100.0
                    if "test_acc" in res:
                        row["test_acc"] = res["test_acc"]["mean"] * 100.0
                        row["test_f1"] = res["test_macro_f1"]["mean"] * 100.0
                except Exception:  # noqa: BLE001
                    row["status"] = "bad_results_json"
                best_acc, best_f1, best_epoch, params = _parse_log(outdir / "train.log")
                row["best_epoch"] = best_epoch
                # val_acc from results.json is authoritative; log-parse is a cross-check
                if best_acc is not None and row["val_acc"] is not None and abs(best_acc - row["val_acc"]) > 0.01:
                    row["status"] = "log_json_mismatch"
                if best_f1 is not None:
                    row["val_f1"] = best_f1
                if params is not None:
                    row["param_count"] = params
                if run_info.exists():
                    try:
                        info = json.loads(run_info.read_text(encoding="utf-8"))
                        row["param_count"] = row["param_count"] or info.get("params")
                        row["peak_mem_mb"] = info.get("train_peak_gpu_mb")
                        row["train_seconds"] = info.get("runtime_sec")
                        row["git_commit"] = info.get("git_commit")
                    except Exception:  # noqa: BLE001
                        pass
                rows.append(row)
    return rows


def _write_runs_csv(rows: list[dict]) -> None:
    fields = ["model", "dataset", "seed", "status", "best_epoch", "val_acc", "val_f1",
              "test_acc", "test_f1", "param_count", "peak_mem_mb", "train_seconds", "git_commit"]
    with (OUT_ROOT / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _write_aggregate(rows: list[dict]) -> None:
    fields = ["model", "dataset", "n_ok", "mean_acc", "std_acc", "mean_f1", "std_f1",
              "mean_params", "mean_peak_mem_mb", "mean_train_seconds"]
    with (OUT_ROOT / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            for model in MODELS:
                ok = [r for r in rows if r["model"] == model and r["dataset"] == dataset and r["status"] == "ok"]
                accs = [r["val_acc"] for r in ok if r["val_acc"] is not None]
                f1s = [r["val_f1"] for r in ok if r["val_f1"] is not None]
                params = [r["param_count"] for r in ok if r["param_count"] is not None]
                mems = [r["peak_mem_mb"] for r in ok if r["peak_mem_mb"] is not None]
                times = [r["train_seconds"] for r in ok if r["train_seconds"] is not None]
                mean_acc, std_acc = _mean_std(accs)
                mean_f1, std_f1 = _mean_std(f1s)
                writer.writerow({
                    "model": model, "dataset": dataset, "n_ok": len(ok),
                    "mean_acc": f"{mean_acc:.4f}" if mean_acc is not None else "",
                    "std_acc": f"{std_acc:.4f}" if std_acc is not None else "",
                    "mean_f1": f"{mean_f1:.4f}" if mean_f1 is not None else "",
                    "std_f1": f"{std_f1:.4f}" if std_f1 is not None else "",
                    "mean_params": f"{statistics.mean(params):.0f}" if params else "",
                    "mean_peak_mem_mb": f"{statistics.mean(mems):.0f}" if mems else "",
                    "mean_train_seconds": f"{statistics.mean(times):.1f}" if times else "",
                })


def _write_strongest(rows: list[dict]) -> None:
    """strongest_baseline_by_dataset.csv: strongest external (excl. biaxis_final)
    and strongest overall (incl. biaxis_final)."""
    fields = ["dataset", "scope", "strongest_model", "mean_acc", "std_acc", "mean_f1", "std_f1"]
    with (OUT_ROOT / "strongest_baseline_by_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            for scope, models in (("external", [m for m in MODELS if m != "biaxis_final"]),
                                  ("overall", list(MODELS))):
                best_model, best_acc, best_std, best_f1, best_f1_std = None, -1.0, None, None, None
                for model in models:
                    ok = [r for r in rows if r["model"] == model and r["dataset"] == dataset and r["status"] == "ok"]
                    accs = [r["val_acc"] for r in ok if r["val_acc"] is not None]
                    f1s = [r["val_f1"] for r in ok if r["val_f1"] is not None]
                    if not accs:
                        continue
                    mean_acc, std_acc = _mean_std(accs)
                    if mean_acc is not None and mean_acc > best_acc:
                        best_acc, best_std = mean_acc, std_acc
                        best_model = model
                        best_f1, best_f1_std = _mean_std(f1s)
                writer.writerow({
                    "dataset": dataset, "scope": scope, "strongest_model": best_model,
                    "mean_acc": f"{best_acc:.4f}", "std_acc": f"{best_std:.4f}" if best_std is not None else "",
                    "mean_f1": f"{best_f1:.4f}" if best_f1 is not None else "",
                    "std_f1": f"{best_f1_std:.4f}" if best_f1_std is not None else "",
                })


def _write_resources(rows: list[dict]) -> None:
    fields = ["model", "dataset", "params", "peak_mem_mb", "train_seconds"]
    with (OUT_ROOT / "resources.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            for model in MODELS:
                ok = [r for r in rows if r["model"] == model and r["dataset"] == dataset and r["status"] == "ok"]
                params = [r["param_count"] for r in ok if r["param_count"] is not None]
                mems = [r["peak_mem_mb"] for r in ok if r["peak_mem_mb"] is not None]
                times = [r["train_seconds"] for r in ok if r["train_seconds"] is not None]
                writer.writerow({
                    "model": model, "dataset": dataset,
                    "params": f"{statistics.mean(params):.0f}" if params else "",
                    "peak_mem_mb": f"{statistics.mean(mems):.0f}" if mems else "",
                    "train_seconds": f"{statistics.mean(times):.1f}" if times else "",
                })


def _write_report(rows: list[dict]) -> None:
    failures = [r for r in rows if r["status"] != "ok"]
    lines = [
        "# R2D29 G0 — Current-Commit NC Validation Reference",
        "",
        "## Protocol",
        "- 10 models x 5 NC datasets x seeds 42/43/44 = 150 runs, Val-only selection.",
        "- Unified NC task protocol (epochs=300, patience=30, eval_every=1); no baseline hyperparameter changes.",
        "- Test metrics are recorded but never used for ranking or selection.",
        "",
        f"## Run health",
        f"- OK runs: {sum(1 for r in rows if r['status'] == 'ok')} / {len(rows)}",
        f"- Failed/missing: {len(failures)}",
    ]
    if failures:
        lines += ["", "### Failures (reason recorded in failures.jsonl)", ""]
        for r in failures:
            lines.append(f"- {r['dataset']} / {r['model']} / seed={r['seed']} — status={r['status']}")
    lines += ["", "## Strongest reference by dataset", "", "| dataset | strongest external | external acc | strongest overall | overall acc | biaxis_final acc |", "|---|---|---|---|---|---|"]
    for dataset in DATASETS:
        ext, oa = {}, {}
        for scope, holder in (("external", ext), ("overall", oa)):
            models = [m for m in MODELS if m != "biaxis_final"] if scope == "external" else list(MODELS)
            best_model, best_acc = None, -1.0
            for model in models:
                ok = [r for r in rows if r["model"] == model and r["dataset"] == dataset and r["status"] == "ok"]
                accs = [r["val_acc"] for r in ok if r["val_acc"] is not None]
                if not accs:
                    continue
                mean_acc = statistics.mean(accs)
                if mean_acc > best_acc:
                    best_acc, best_model = mean_acc, model
            holder["model"], holder["acc"] = best_model, best_acc
        ok = [r for r in rows if r["model"] == "biaxis_final" and r["dataset"] == dataset and r["status"] == "ok"]
        accs = [r["val_acc"] for r in ok if r["val_acc"] is not None]
        bf = f"{statistics.mean(accs):.4f}" if accs else "n/a"
        lines.append(
            f"| {dataset} | {ext['model']} | {ext['acc']:.4f} | "
            f"{oa['model']} | {oa['acc']:.4f} | {bf} |"
        )
    lines += ["", "## Per-model x dataset Val Acc / Macro-F1 (mean±std over 3 seeds)", ""]
    lines.append("| model | " + " | ".join(f"{d} Acc / F1" for d in DATASETS) + " |")
    lines.append("|---|" + "---|" * len(DATASETS))
    for model in MODELS:
        cells = []
        for dataset in DATASETS:
            ok = [r for r in rows if r["model"] == model and r["dataset"] == dataset and r["status"] == "ok"]
            accs = [r["val_acc"] for r in ok if r["val_acc"] is not None]
            f1s = [r["val_f1"] for r in ok if r["val_f1"] is not None]
            if accs:
                m_a, s_a = _mean_std(accs)
                m_f, s_f = _mean_std(f1s)
                cells.append(f"{m_a:.2f}±{s_a:.2f} / {m_f:.2f}±{s_f:.2f}")
            else:
                cells.append("n/a")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    (OUT_ROOT / "G0_REFERENCE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _load_runs()
    _write_runs_csv(rows)
    _write_aggregate(rows)
    _write_strongest(rows)
    _write_resources(rows)
    _write_report(rows)
    print(f"[g0-summarizer] wrote runs.csv ({len(rows)} rows), aggregate.csv, "
          f"strongest_baseline_by_dataset.csv, resources.csv, G0_REFERENCE_REPORT.md -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
