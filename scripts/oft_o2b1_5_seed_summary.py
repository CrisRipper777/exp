"""O2-B1.5 Part B — 3-seed aggregation (seed42 locked + seed43/44 replication).

Reads only locked/offline artifacts (per-class val CSVs, per-epoch history
CSVs, router stats JSONs, per-val-node router CSVs) and reports:

    * per-model per-seed Val Acc / Macro-F1 / Balanced Acc and mean ± std;
    * paired seed differences for the five pre-registered contrasts;
    * router stability across seeds (pair means, same-target correlations,
      pair ordering, inter-seed per-node probability correlation).

Balanced accuracy is the mean per-class recall and macro-F1 the mean per-class
F1, both derived from the SAME per-class CSV schema used by O1.5 / O2-A / O2-B1
(there is one per-class implementation). No training, no test split.

Usage:
    PYTHONPATH=. python scripts/oft_o2b1_5_seed_summary.py \
        --out-dir experiments/oft/o2b1_5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

MODELS = ("diag_id", "static_cross_dup", "static_cross", "conditional_cross_dup", "conditional_cross")
SEEDS = (42, 43, 44)

# Pre-registered paired contrasts (name -> (left, right)).
CONTRASTS = (
    ("CC_minus_CD", "conditional_cross", "conditional_cross_dup"),
    ("CC_minus_SD", "conditional_cross", "static_cross_dup"),
    ("CC_minus_SC", "conditional_cross", "static_cross"),
    ("SC_minus_SD", "static_cross", "static_cross_dup"),
    ("CC_minus_DIAG", "conditional_cross", "diag_id"),
)

# Locked seed42 artifacts (O1.5 / O2-A / O2-B1) and the new seed43/44 ones.
PER_CLASS = {
    42: {
        "diag_id": "experiments/oft/o1_5/diag_id_val_per_class.csv",
        "static_cross_dup": "experiments/oft/o2a/static_cross_dup_val_per_class.csv",
        "static_cross": "experiments/oft/o2a/static_cross_val_per_class.csv",
        "conditional_cross_dup": "experiments/oft/o2b1/conditional_cross_dup_val_per_class.csv",
        "conditional_cross": "experiments/oft/o2b1/conditional_cross_val_per_class.csv",
    },
    43: {v: f"experiments/oft/o2b1_5/{v}_val_per_class_seed43.csv" for v in MODELS},
    44: {v: f"experiments/oft/o2b1_5/{v}_val_per_class_seed44.csv" for v in MODELS},
}
HISTORY = {
    42: {
        "diag_id": "experiments/oft/o1_5/diag_id_rerun_movies_seed42.csv",
        "static_cross_dup": "experiments/oft/o2a/static_cross_dup_movies_seed42.csv",
        "static_cross": "experiments/oft/o2a/static_cross_movies_seed42.csv",
        "conditional_cross_dup": "experiments/oft/o2b1/conditional_cross_dup_movies_seed42.csv",
        "conditional_cross": "experiments/oft/o2b1/conditional_cross_movies_seed42.csv",
    },
    43: {v: f"experiments/oft/o2b1_5/{v}_movies_seed43.csv" for v in MODELS},
    44: {v: f"experiments/oft/o2b1_5/{v}_movies_seed44.csv" for v in MODELS},
}
ROUTER_STATS = {
    42: {
        "conditional_cross_dup": "experiments/oft/o2b1/conditional_cross_dup_stats.json",
        "conditional_cross": "experiments/oft/o2b1/conditional_cross_stats.json",
    },
    43: {v: f"experiments/oft/o2b1_5/{v}_stats_seed43.json"
         for v in ("conditional_cross_dup", "conditional_cross")},
    44: {v: f"experiments/oft/o2b1_5/{v}_stats_seed44.json"
         for v in ("conditional_cross_dup", "conditional_cross")},
}
ROUTER_NODES = {
    42: {
        "conditional_cross_dup": "experiments/oft/o2b1/conditional_cross_dup_router_nodes_val.csv",
        "conditional_cross": "experiments/oft/o2b1/conditional_cross_router_nodes_val.csv",
    },
    43: {v: f"experiments/oft/o2b1_5/{v}_router_nodes_val_seed43.csv"
         for v in ("conditional_cross_dup", "conditional_cross")},
    44: {v: f"experiments/oft/o2b1_5/{v}_router_nodes_val_seed44.csv"
         for v in ("conditional_cross_dup", "conditional_cross")},
}
PAIR_KEYS = (
    "c_to_pt", "c_to_pv", "pt_to_c", "pt_to_pv", "pv_to_c", "pv_to_pt",
)
TARGETS = {"target_c": ("pt_to_c", "pv_to_c"),
           "target_pt": ("c_to_pt", "pv_to_pt"),
           "target_pv": ("c_to_pv", "pt_to_pv")}


def metrics_from_rows(rows: list[dict]) -> dict[str, float]:
    """Acc / macro-F1 / balanced-acc from the shared per-class schema."""
    support = np.array([int(r["support"]) for r in rows], dtype=np.float64)
    recall = np.array([float(r["recall"]) for r in rows], dtype=np.float64)
    f1 = np.array([float(r["f1"]) for r in rows], dtype=np.float64)
    return {
        "acc": float((recall * support).sum() / support.sum()),
        "macro_f1": float(f1.mean()),
        "balanced_acc": float(recall.mean()),
    }


def best_row(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    best = max(rows, key=lambda r: float(r["val_acc"]))
    return {"epoch": int(best["epoch"]), "val_acc": float(best["val_acc"]),
            "val_macro_f1": float(best["val_macro_f1"]), "n_epochs": len(rows)}


def load_metrics() -> dict:
    out: dict = {}
    for seed in SEEDS:
        for model in MODELS:
            path = Path(PER_CLASS[seed][model])
            if not path.exists():
                raise SystemExit(f"missing per-class CSV: {path}")
            with path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            entry = metrics_from_rows(rows)
            entry["per_class"] = {int(r["class_id"]): float(r["f1"]) for r in rows}
            hist = best_row(HISTORY[seed][model])
            entry.update({"best_epoch": hist["epoch"], "n_epochs": hist["n_epochs"],
                          "history_val_acc": hist["val_acc"],
                          "history_val_macro_f1": hist["val_macro_f1"],
                          "acc_ckpt_vs_history_pp": (entry["acc"] - hist["val_acc"]) * 100.0})
            out[(seed, model)] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/oft/o2b1_5")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()

    # ---- per-model per-seed table + mean/std -----------------------------
    model_rows: list[dict] = []
    summary: dict = {"per_seed": {}, "mean_std": {}}
    for model in MODELS:
        accs, f1s, bals = [], [], []
        for seed in SEEDS:
            entry = metrics[(seed, model)]
            accs.append(entry["acc"]); f1s.append(entry["macro_f1"]); bals.append(entry["balanced_acc"])
            model_rows.append({
                "model": model, "seed": seed, "acc": entry["acc"], "macro_f1": entry["macro_f1"],
                "balanced_acc": entry["balanced_acc"], "best_epoch": entry["best_epoch"],
                "n_epochs": entry["n_epochs"], "history_val_acc": entry["history_val_acc"],
                "acc_ckpt_vs_history_pp": entry["acc_ckpt_vs_history_pp"],
            })
            summary["per_seed"].setdefault(model, {})[str(seed)] = {
                "acc": entry["acc"], "macro_f1": entry["macro_f1"],
                "balanced_acc": entry["balanced_acc"], "best_epoch": entry["best_epoch"],
                "n_epochs": entry["n_epochs"],
            }
        model_rows.append({
            "model": model, "seed": "mean", "acc": float(np.mean(accs)),
            "macro_f1": float(np.mean(f1s)), "balanced_acc": float(np.mean(bals)),
            "best_epoch": "", "n_epochs": "", "history_val_acc": "", "acc_ckpt_vs_history_pp": "",
        })
        model_rows.append({
            "model": model, "seed": "std", "acc": float(np.std(accs, ddof=0)),
            "macro_f1": float(np.std(f1s, ddof=0)), "balanced_acc": float(np.std(bals, ddof=0)),
            "best_epoch": "", "n_epochs": "", "history_val_acc": "", "acc_ckpt_vs_history_pp": "",
        })
        summary["mean_std"][model] = {
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=0)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s, ddof=0)),
            "balanced_acc_mean": float(np.mean(bals)),
            "balanced_acc_std": float(np.std(bals, ddof=0)),
        }

    with (out_dir / "three_seed_model_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(model_rows[0].keys()))
        writer.writeheader()
        writer.writerows(model_rows)

    # ---- paired seed differences -----------------------------------------
    pair_rows: list[dict] = []
    paired_summary: dict = {}
    for name, left, right in CONTRASTS:
        d_acc, d_f1, d_bal = [], [], []
        for seed in SEEDS:
            l, r = metrics[(seed, left)], metrics[(seed, right)]
            row = {
                "contrast": name, "left": left, "right": right, "seed": seed,
                "delta_acc_pp": (l["acc"] - r["acc"]) * 100.0,
                "delta_f1_pp": (l["macro_f1"] - r["macro_f1"]) * 100.0,
                "delta_balacc_pp": (l["balanced_acc"] - r["balanced_acc"]) * 100.0,
            }
            pair_rows.append(row)
            d_acc.append(row["delta_acc_pp"]); d_f1.append(row["delta_f1_pp"])
            d_bal.append(row["delta_balacc_pp"])
        pair_rows.append({
            "contrast": name, "left": left, "right": right, "seed": "mean",
            "delta_acc_pp": float(np.mean(d_acc)), "delta_f1_pp": float(np.mean(d_f1)),
            "delta_balacc_pp": float(np.mean(d_bal)),
        })
        pair_rows.append({
            "contrast": name, "left": left, "right": right, "seed": "std",
            "delta_acc_pp": float(np.std(d_acc, ddof=0)), "delta_f1_pp": float(np.std(d_f1, ddof=0)),
            "delta_balacc_pp": float(np.std(d_bal, ddof=0)),
        })
        paired_summary[name] = {
            "left": left, "right": right,
            "per_seed": {str(s): {"acc_pp": d_acc[i], "f1_pp": d_f1[i], "balacc_pp": d_bal[i]}
                         for i, s in enumerate(SEEDS)},
            "mean": {"acc_pp": float(np.mean(d_acc)), "f1_pp": float(np.mean(d_f1)),
                     "balacc_pp": float(np.mean(d_bal))},
            "std": {"acc_pp": float(np.std(d_acc, ddof=0)), "f1_pp": float(np.std(d_f1, ddof=0)),
                    "balacc_pp": float(np.std(d_bal, ddof=0))},
            "positive_seeds": {
                "acc": int(sum(d > 0 for d in d_acc)),
                "f1": int(sum(d > 0 for d in d_f1)),
                "balacc": int(sum(d > 0 for d in d_bal)),
            },
        }
    with (out_dir / "paired_seed_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    # ---- router stability across seeds -----------------------------------
    # Everything is derived from the per-val-node router CSVs (one row per val
    # node, one p column per pair) so the SAME computation runs for seed42
    # (locked O2-B1 artifacts) and seeds 43/44.
    stability: dict = {"pair_mean": {}, "pair_std": {}, "pair_quantiles": {},
                       "same_target_corr": {}, "pair_order": {}, "target_gate": {},
                       "source_abs_diff": {}, "inter_seed_node_corr": {}}
    for model in ("conditional_cross", "conditional_cross_dup"):
        for section in ("pair_mean", "pair_std", "pair_quantiles", "same_target_corr",
                        "pair_order", "target_gate", "source_abs_diff"):
            stability[section][model] = {}
        per_seed_nodes: dict[int, dict[int, list[float]]] = {}
        for seed in SEEDS:
            path = Path(ROUTER_NODES[seed][model])
            if not path.exists():
                raise SystemExit(f"missing per-val-node router CSV: {path}")
            with path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            per_seed_nodes[seed] = {
                int(r["node_id"]): [float(r[f"p_{key}"]) for key in PAIR_KEYS] for r in rows
            }
            mat = np.array([per_seed_nodes[seed][n] for n in sorted(per_seed_nodes[seed])])
            means = {key: float(mat[:, j].mean()) for j, key in enumerate(PAIR_KEYS)}
            stability["pair_mean"][model][str(seed)] = means
            stability["pair_std"][model][str(seed)] = {
                key: float(mat[:, j].std(ddof=0)) for j, key in enumerate(PAIR_KEYS)
            }
            stability["pair_quantiles"][model][str(seed)] = {
                key: {"p10": float(np.quantile(mat[:, j], 0.10)),
                      "p50": float(np.quantile(mat[:, j], 0.50)),
                      "p90": float(np.quantile(mat[:, j], 0.90))}
                for j, key in enumerate(PAIR_KEYS)
            }
            stability["pair_order"][model][str(seed)] = [
                key for key in sorted(means, key=lambda k: -means[k])
            ]
            corr, gates, diffs = {}, {}, {}
            for target, (a1, a2) in TARGETS.items():
                j1, j2 = PAIR_KEYS.index(a1), PAIR_KEYS.index(a2)
                corr[target] = {
                    "pair_a": a1, "pair_b": a2,
                    "pearson": float(pearsonr(mat[:, j1], mat[:, j2]).statistic),
                    "spearman": float(spearmanr(mat[:, j1], mat[:, j2]).statistic),
                }
                gates[target] = float(0.5 * (means[a1] + means[a2]))
                diffs[target] = float(np.abs(mat[:, j1] - mat[:, j2]).mean())
            stability["same_target_corr"][model][str(seed)] = corr
            stability["target_gate"][model][str(seed)] = gates
            stability["source_abs_diff"][model][str(seed)] = diffs
            # cross-check against the O2-B1 diagnostic JSON when present
            stats_path = Path(ROUTER_STATS[seed][model])
            if stats_path.exists():
                stats = json.load(open(stats_path, encoding="utf-8"))["stats"]
                stability.setdefault("stats_cross_check", {}).setdefault(model, {})[str(seed)] = {
                    key: abs(means[key] - float(stats[f"val_mean_{key}"]))
                    for key in PAIR_KEYS if f"val_mean_{key}" in stats
                }

        # inter-seed per-node probability correlation on the shared val node ids
        entry: dict = {"n_val_per_seed": {str(s): len(v) for s, v in per_seed_nodes.items()}}
        for i, s1 in enumerate(SEEDS):
            for s2 in SEEDS[i + 1:]:
                if s1 not in per_seed_nodes or s2 not in per_seed_nodes:
                    continue
                common = sorted(set(per_seed_nodes[s1]) & set(per_seed_nodes[s2]))
                if len(common) < 10:
                    entry[f"{s1}_vs_{s2}"] = {"n_common": len(common), "note": "too few shared val nodes"}
                    continue
                a = np.array([per_seed_nodes[s1][n] for n in common])
                b = np.array([per_seed_nodes[s2][n] for n in common])
                entry[f"{s1}_vs_{s2}"] = {
                    "n_common": len(common),
                    "pearson_per_pair": {key: float(pearsonr(a[:, j], b[:, j]).statistic)
                                         for j, key in enumerate(PAIR_KEYS)},
                    "spearman_per_pair": {key: float(spearmanr(a[:, j], b[:, j]).statistic)
                                          for j, key in enumerate(PAIR_KEYS)},
                    "pearson_all_pairs": float(pearsonr(a.ravel(), b.ravel()).statistic),
                    "spearman_all_pairs": float(spearmanr(a.ravel(), b.ravel()).statistic),
                }
        stability["inter_seed_node_corr"][model] = entry

    with (out_dir / "three_seed_router_stability.json").open("w", encoding="utf-8") as fh:
        json.dump({"model_summary": summary, "paired": paired_summary, "router": stability},
                  fh, indent=2, sort_keys=True)

    # ---- per-class F1 across models / seeds ------------------------------
    with (out_dir / "three_seed_per_class_f1.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["class_id", "support"] + [f"{m}_s{s}" for m in MODELS for s in SEEDS]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for c in range(20):
            row = {"class_id": c}
            for model in MODELS:
                for seed in SEEDS:
                    row[f"{model}_s{seed}"] = metrics[(seed, model)]["per_class"].get(c, "")
            writer.writerow(row)

    # ---- console report ---------------------------------------------------
    print("=== O2-B1.5 Part B 3-seed summary ===")
    print(f"{'model':<24}{'seed':>6}{'acc':>10}{'macro_f1':>10}{'bal_acc':>10}{'best_ep':>9}")
    for model in MODELS:
        for seed in SEEDS:
            e = metrics[(seed, model)]
            print(f"{model:<24}{seed:>6}{e['acc']:>10.6f}{e['macro_f1']:>10.6f}"
                  f"{e['balanced_acc']:>10.6f}{e['best_epoch']:>9}")
        ms = summary["mean_std"][model]
        print(f"{model:<24}{'mean':>6}{ms['acc_mean']:>10.6f}{ms['macro_f1_mean']:>10.6f}"
              f"{ms['balanced_acc_mean']:>10.6f}")
        print(f"{'':<24}{'std':>6}{ms['acc_std']:>10.6f}{ms['macro_f1_std']:>10.6f}"
              f"{ms['balanced_acc_std']:>10.6f}")
    print("\npaired deltas (pp):")
    for name, item in paired_summary.items():
        per = item["per_seed"]
        print(f"  {name:<14} acc  " + " ".join(f"s{s}={per[str(s)]['acc_pp']:+.2f}" for s in SEEDS)
              + f"  mean={item['mean']['acc_pp']:+.2f} (pos {item['positive_seeds']['acc']}/3)")
        print(f"  {'':<14} f1   " + " ".join(f"s{s}={per[str(s)]['f1_pp']:+.2f}" for s in SEEDS)
              + f"  mean={item['mean']['f1_pp']:+.2f} (pos {item['positive_seeds']['f1']}/3)")
        print(f"  {'':<14} bal  " + " ".join(f"s{s}={per[str(s)]['balacc_pp']:+.2f}" for s in SEEDS)
              + f"  mean={item['mean']['balacc_pp']:+.2f} (pos {item['positive_seeds']['balacc']}/3)")
    print("\nrouter pair means per seed:")
    for model in ("conditional_cross", "conditional_cross_dup"):
        for seed in SEEDS:
            means = stability["pair_mean"][model][str(seed)]
            print(f"  {model:<20} seed{seed}: " + " ".join(f"{k}={means[k]:.4f}" for k in PAIR_KEYS))
    print(f"\nwrote {out_dir}/three_seed_model_summary.csv, paired_seed_summary.csv, "
          f"three_seed_router_stability.json, three_seed_per_class_f1.csv")


if __name__ == "__main__":
    main()
