"""R2D29 G2 summarizer: factorial statistics + global ranking (plan §7.3-§7.6).

Reads outputs/r2d29/g2_synergy/main/ + the G0 reference (outputs/r2d29/
g0_reference/aggregate.csv and strongest_baseline_by_dataset.csv) and writes:

  runs.csv / aggregate.csv / factorial_cells.csv / main_effects.csv /
  two_way_interactions.csv / higher_order_interactions.csv /
  strongest_gap.csv / matched_controls.csv / resources.csv /
  G2_SYSTEM_SYNERGY_REPORT.md

Val-only: all statistics use val acc / val f1 at the best-val-acc epoch.
Main effects are NEVER used to eliminate a component (plan §2.2).

Usage:
    python scripts/summarize_perf_r2d29_g2.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
G2_ROOT = PROJECT_ROOT / "outputs" / "r2d29" / "g2_synergy"
G0_ROOT = PROJECT_ROOT / "outputs" / "r2d29" / "g0_reference"

from src.analysis.perf_r2d29_utils import (  # noqa: E402
    DATASETS,
    G2_CELLS,
    G2_MATCHED_CONTROLS,
    SEEDS,
    parse_train_log,
)

FACTORS = ["R", "S", "W", "F"]


def _cell_level(cell: str, factor: str) -> int:
    """1 if the factor is at its high level in the cell name, else 0.
    Cell names are R{r}S{s}W{w}F{f} (e.g. R1S0W1F0)."""
    return int(cell[1 + 2 * FACTORS.index(factor)])


def _load_runs() -> list[dict]:
    rows = []
    main = G2_ROOT / "main"
    for dataset in DATASETS:
        cells = dict(G2_CELLS)
        cells.update(G2_MATCHED_CONTROLS)
        for cell in sorted(cells):
            for seed in SEEDS:
                outdir = main / dataset / cell / f"seed_{seed}"
                results_json = outdir / "hydra" / "results.json"
                row = {"dataset": dataset, "cell": cell, "seed": seed,
                       "status": "ok", "best_epoch": None, "val_acc": None,
                       "val_f1": None, "param_count": None, "peak_mem_mb": None,
                       "train_seconds": None}
                if not results_json.exists():
                    row["status"] = "missing"
                    rows.append(row)
                    continue
                res = json.loads(results_json.read_text(encoding="utf-8"))
                row["val_acc"] = res["val_acc"]["mean"] * 100.0
                parsed = parse_train_log(outdir / "train.log")
                row["best_epoch"] = parsed["best_epoch"]
                row["val_f1"] = parsed["val_f1"]
                row["param_count"] = parsed["params"]
                info_path = outdir / "run_info.json"
                if info_path.exists():
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                    row["param_count"] = row["param_count"] or info.get("params")
                    row["peak_mem_mb"] = info.get("train_peak_gpu_mb")
                    row["train_seconds"] = info.get("runtime_sec")
                rows.append(row)
    return rows


def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _cell_mean(cell: str, dataset: str, rows: list[dict], key: str = "val_acc") -> float | None:
    vals = [r[key] for r in rows if r["cell"] == cell and r["dataset"] == dataset
            and r["status"] == "ok" and r[key] is not None]
    return statistics.mean(vals) if vals else None


def _interaction_contrast(factors: list[str], dataset: str, cells: dict,
                          rows: list[dict], key: str) -> float | None:
    """n-way interaction in pp: the signed corner-sum over the n factors,
    averaged over the replicates from the remaining factors (2^(4-n)), so it
    reads as a difference-of-differences on the same scale as the main
    effects. Example: I_RS = mean over (W,F) of
    mu(R1S1WF) - mu(R1S0WF) - mu(R0S1WF) + mu(R0S0WF)."""
    total, count = 0.0, 0
    for cell in cells:
        n_low = sum(1 for f in factors if _cell_level(cell, f) == 0)
        mu = _cell_mean(cell, dataset, rows, key)
        if mu is None:
            return None
        total += (-1) ** n_low * mu
        count += 1
    if not count:
        return None
    n_repl = 2 ** (len(FACTORS) - len(factors))
    return total / n_repl


def _g0_reference() -> dict:
    """dataset -> {model -> {acc, f1}} plus strongest-external per dataset."""
    agg_path = G0_ROOT / "aggregate.csv"
    ref: dict = {}
    if agg_path.exists():
        with agg_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ref.setdefault(row["dataset"], {})[row["model"]] = {
                    "acc": float(row["mean_acc"]), "f1": float(row["mean_f1"]),
                }
    strongest_path = G0_ROOT / "strongest_baseline_by_dataset.csv"
    ext: dict = {}
    if strongest_path.exists():
        with strongest_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["scope"] == "external":
                    ext[row["dataset"]] = {
                        "model": row["strongest_model"],
                        "acc": float(row["mean_acc"]),
                        "f1": float(row["mean_f1"]),
                    }
    return ref, ext


def main() -> None:
    G2_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _load_runs()
    ref, ext = _g0_reference()

    # --- runs.csv ---
    fields = ["dataset", "cell", "seed", "status", "best_epoch", "val_acc", "val_f1",
              "param_count", "peak_mem_mb", "train_seconds"]
    with (G2_ROOT / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # --- aggregate.csv ---
    with (G2_ROOT / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "cell", "n_ok", "mean_acc", "std_acc",
                         "mean_f1", "std_f1", "mean_params", "mean_peak_mb", "mean_sec"])
        for dataset in DATASETS:
            for cell in sorted(dict(G2_CELLS, **G2_MATCHED_CONTROLS)):
                ok = [r for r in rows if r["dataset"] == dataset and r["cell"] == cell
                      and r["status"] == "ok"]
                accs = [r["val_acc"] for r in ok if r["val_acc"] is not None]
                f1s = [r["val_f1"] for r in ok if r["val_f1"] is not None]
                prms = [r["param_count"] for r in ok if r["param_count"] is not None]
                mems = [r["peak_mem_mb"] for r in ok if r["peak_mem_mb"] is not None]
                secs = [r["train_seconds"] for r in ok if r["train_seconds"] is not None]
                ma, sa = _mean_std(accs)
                mf, sf = _mean_std(f1s)
                writer.writerow([
                    dataset, cell, len(ok),
                    f"{ma:.4f}" if ma is not None else "", f"{sa:.4f}" if sa is not None else "",
                    f"{mf:.4f}" if mf is not None else "", f"{sf:.4f}" if sf is not None else "",
                    f"{statistics.mean(prms):.0f}" if prms else "",
                    f"{statistics.mean(mems):.0f}" if mems else "",
                    f"{statistics.mean(secs):.1f}" if secs else "",
                ])

    # --- factorial_cells.csv: 16 cells x 5-dataset pooled means ---
    with (G2_ROOT / "factorial_cells.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", *[f"{d}_acc" for d in DATASETS],
                         *[f"{d}_f1" for d in DATASETS], "mean_acc_5ds", "mean_f1_5ds"])
        for cell in G2_CELLS:
            accs = [_cell_mean(cell, d, rows, "val_acc") for d in DATASETS]
            f1s = [_cell_mean(cell, d, rows, "val_f1") for d in DATASETS]
            row_vals = [cell]
            row_vals += [f"{a:.4f}" if a is not None else "" for a in accs]
            row_vals += [f"{v:.4f}" if v is not None else "" for v in f1s]
            accs_ok = [a for a in accs if a is not None]
            f1s_ok = [v for v in f1s if v is not None]
            row_vals += [f"{statistics.mean(accs_ok):.4f}" if accs_ok else "",
                         f"{statistics.mean(f1s_ok):.4f}" if f1s_ok else ""]
            writer.writerow(row_vals)

    # --- main effects (Accuracy and Macro-F1) ---
    with (G2_ROOT / "main_effects.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["factor", "metric", *DATASETS, "mean_5ds"])
        for key, label in (("val_acc", "acc"), ("val_f1", "f1")):
            for factor in FACTORS:
                line = [f"delta_{factor}", label]
                for dataset in DATASETS:
                    hi = statistics.mean(
                        _cell_mean(c, dataset, rows, key) for c in G2_CELLS
                        if _cell_level(c, factor) == 1)
                    lo = statistics.mean(
                        _cell_mean(c, dataset, rows, key) for c in G2_CELLS
                        if _cell_level(c, factor) == 0)
                    line.append(f"{hi - lo:+.4f}")
                vals = [float(v) for v in line[2:]]
                line.append(f"{statistics.mean(vals):+.4f}")
                writer.writerow(line)

    # --- two-way interactions (Accuracy and Macro-F1) ---
    with (G2_ROOT / "two_way_interactions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["interaction", "metric", *DATASETS, "mean_5ds"])
        for f1_, f2_ in combinations(FACTORS, 2):
            for key, label in (("val_acc", "acc"), ("val_f1", "f1")):
                line = [f"I_{f1_}{f2_}", label]
                for dataset in DATASETS:
                    v = _interaction_contrast([f1_, f2_], dataset, G2_CELLS, rows, key)
                    line.append(f"{v:+.4f}" if v is not None else "")
                vals = [float(v) for v in line[2:] if v]
                line.append(f"{statistics.mean(vals):+.4f}" if vals else "")
                writer.writerow(line)

    # --- higher-order interactions (Accuracy and Macro-F1) ---
    with (G2_ROOT / "higher_order_interactions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["interaction", "metric", *DATASETS, "mean_5ds"])
        for combo in (["R", "S", "W"], ["R", "S", "F"], ["R", "W", "F"],
                      ["S", "W", "F"], ["R", "S", "W", "F"]):
            for key, label in (("val_acc", "acc"), ("val_f1", "f1")):
                line = ["I_" + "".join(combo), label]
                for dataset in DATASETS:
                    v = _interaction_contrast(combo, dataset, G2_CELLS, rows, key)
                    line.append(f"{v:+.4f}" if v is not None else "")
                vals = [float(v) for v in line[2:] if v]
                line.append(f"{statistics.mean(vals):+.4f}" if vals else "")
                writer.writerow(line)

    # --- global ranking vs A0 / strongest external ---
    ranking_rows = []
    for cell in G2_CELLS:
        deltas_a0, deltas_ext = [], []
        wins = ties = losses = 0
        rank_sum = 0
        accs, f1s = [], []
        worst = 0.0
        for dataset in DATASETS:
            mu = _cell_mean(cell, dataset, rows)
            if mu is None:
                continue
            accs.append(mu)
            f1s.append(_cell_mean(cell, dataset, rows, "val_f1"))
            a0 = ref.get(dataset, {}).get("biaxis_final", {}).get("acc")
            if a0 is not None:
                deltas_a0.append(mu - a0)
            e = ext.get(dataset, {})
            if e:
                d = mu - e["acc"]
                deltas_ext.append(d)
                worst = min(worst, d)
                if d > 0:
                    wins += 1
                elif abs(d) <= 1e-9:
                    ties += 1
                else:
                    losses += 1
            # rank of this cell among cells on this dataset
            cell_accs = sorted(
                {c: _cell_mean(c, dataset, rows) for c in G2_CELLS}.items(),
                key=lambda kv: -(kv[1] if kv[1] is not None else -1e9))
            for rank, (c, _a) in enumerate(cell_accs, 1):
                if c == cell:
                    rank_sum += rank
                    break
        ranking_rows.append({
            "cell": cell,
            "mean_delta_vs_A0": statistics.mean(deltas_a0) if deltas_a0 else None,
            "mean_delta_vs_strongest": statistics.mean(deltas_ext) if deltas_ext else None,
            "num_dataset_wins": wins, "ties": ties, "losses": losses,
            "worst_dataset_delta": worst,
            "mean_rank_5ds": rank_sum / len(DATASETS) if rank_sum else None,
            "mean_acc": statistics.mean(accs) if accs else None,
            "mean_f1": statistics.mean([v for v in f1s if v is not None]) if f1s else None,
        })
    with (G2_ROOT / "strongest_gap.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cell", "mean_delta_vs_A0", "mean_delta_vs_strongest",
            "num_dataset_wins", "ties", "losses", "worst_dataset_delta",
            "mean_rank_5ds", "mean_acc", "mean_f1"])
        writer.writeheader()
        for row in ranking_rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})

    # --- matched controls ---
    with (G2_ROOT / "matched_controls.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "cell", "mean_acc", "base_cell_mean_acc", "delta"])
        for mc in G2_MATCHED_CONTROLS:
            base = mc.split("+")[0]
            for dataset in DATASETS:
                a = _cell_mean(mc, dataset, rows)
                b = _cell_mean(base, dataset, rows)
                if a is None and b is None:
                    continue
                writer.writerow([dataset, mc,
                                 f"{a:.4f}" if a is not None else "",
                                 f"{b:.4f}" if b is not None else "",
                                 f"{a - b:+.4f}" if a is not None and b is not None else ""])

    # --- resources.csv ---
    with (G2_ROOT / "resources.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "cell", "params", "peak_mem_mb", "train_seconds"])
        for dataset in DATASETS:
            for cell in sorted(dict(G2_CELLS, **G2_MATCHED_CONTROLS)):
                ok = [r for r in rows if r["dataset"] == dataset and r["cell"] == cell
                      and r["status"] == "ok"]
                prms = [r["param_count"] for r in ok if r["param_count"] is not None]
                mems = [r["peak_mem_mb"] for r in ok if r["peak_mem_mb"] is not None]
                secs = [r["train_seconds"] for r in ok if r["train_seconds"] is not None]
                writer.writerow([dataset, cell,
                                 f"{statistics.mean(prms):.0f}" if prms else "",
                                 f"{statistics.mean(mems):.0f}" if mems else "",
                                 f"{statistics.mean(secs):.1f}" if secs else ""])

    # --- report ---
    # only report cells that were actually launched (>=1 run present): the
    # other matched controls were deliberately not run in this phase
    launched_cells = {(r["dataset"], r["cell"]) for r in rows if r["status"] == "ok"}
    failures = [r for r in rows if r["status"] != "ok"
                and (r["dataset"], r["cell"]) in launched_cells]
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_launched = len(launched_cells) * len(SEEDS)
    ranked = sorted(ranking_rows, key=lambda r: -(r["mean_acc"] or -1e9))
    lines = [
        "# G2_SYSTEM_SYNERGY_REPORT — Full System Synergy Matrix (plan §7)",
        "",
        f"- runs: {n_ok} OK / {n_launched} launched; failed: {len(failures)}",
        f"- cells: 16 factorial (fixed a0_augment, num_blocks=1) + {len(G2_MATCHED_CONTROLS)} matched controls",
        "- statistics: val acc / val f1 at best-val-acc epoch, 3 seeds per cell-dataset",
        "",
        "## Global ranking (5-dataset, vs G0 reference)",
        "",
        "| rank | cell | mean_acc | mean_f1 | ΔA0 | ΔStrongest | wins/ties/losses | worst_Δ | mean_rank |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {row['cell']} | {row['mean_acc']:.2f} | {row['mean_f1']:.2f} | "
            f"{row['mean_delta_vs_A0']:+.2f} | {row['mean_delta_vs_strongest']:+.2f} | "
            f"{row['num_dataset_wins']}/{row['ties']}/{row['losses']} | "
            f"{row['worst_dataset_delta']:+.2f} | {row['mean_rank_5ds']:.2f} |"
        )
    lines += [
        "",
        "## Main effects / interactions",
        "",
        "See main_effects.csv / two_way_interactions.csv / higher_order_interactions.csv.",
        "Per plan §2.2, no component is eliminated on a weak main effect alone —",
        "the coordinated-pathway interaction is the primary signal.",
        "",
        "## Top-4 recommendation (global performance + Pareto, plan §7.4)",
        "",
    ]
    for row in ranked[:4]:
        lines.append(f"- **{row['cell']}** — mean_acc {row['mean_acc']:.2f}, "
                     f"ΔStrongest {row['mean_delta_vs_strongest']:+.2f}, "
                     f"wins {row['num_dataset_wins']}, worst_Δ {row['worst_dataset_delta']:+.2f}")
    # matched controls summary (only cells that actually ran)
    lines += ["", "## Matched controls (MEAN_DUP, plan §7.5)", ""]
    mc_lines = []
    for mc in G2_MATCHED_CONTROLS:
        base = mc.split("+")[0]
        if base not in ("R1S1W1F0", "R0S1W1F0"):
            continue
        deltas = []
        for dataset in DATASETS:
            a = _cell_mean(mc, dataset, rows)
            b = _cell_mean(base, dataset, rows)
            if a is not None and b is not None:
                deltas.append(a - b)
        if deltas:
            mc_lines.append(
                f"- **{mc}**: 5-dataset mean Δ vs base = {statistics.mean(deltas):+.2f} pp "
                f"(per-dataset: {', '.join(f'{d:+.2f}' for d in deltas)})"
            )
    lines += mc_lines or ["- (no matched-control runs found)"]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(f"- {r['dataset']} / {r['cell']} / seed={r['seed']} — {r['status']}")
    lines += ["", "G2 stops here; G3 starts only after review (plan §7.6)."]
    (G2_ROOT / "G2_SYSTEM_SYNERGY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[g2-summarizer] wrote G2 outputs -> {G2_ROOT}")


if __name__ == "__main__":
    main()
