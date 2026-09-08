"""O2-B1.5 Part A — frozen routing-granularity causal audit.

Loads the trained CONDITIONAL-CROSS seed42 checkpoint (best-Val-Acc) and
re-runs the exact full-graph inference path while replacing ONLY the router's
``p_transfer`` with a counterfactual gate. Every parameter (factorizer, V/D,
the six ``T_{a->b}``, router, fusion, head) stays frozen; nothing is retrained,
finetuned or tuned. The intervention is installed by monkeypatching
``ConditionalCrossLayer.router_transfer`` (see ``oft_routing_interventions``),
so payloads / ``delta_diag`` / ``combine_cross`` / LayerNorm / fusion / head are
the layer's own code — no second copy of the model math exists here.

Protocol: VALIDATION split only. The held-out TEST split is never read and no
TEST metric is ever computed. The per-class table and metrics come from the
shared ``val_per_class_rows`` helper (same implementation as O1.5 / O2-A /
O2-B1), so the ORIGINAL replay is directly comparable to the locked O2-B1 CSV.

Interventions: ORIGINAL, PAIR_CONSTANT, TARGET_NODE, TARGET_CONSTANT,
SOURCE_SWAP, NODE_SHUFFLE (20 seeds), ALL_ON, ALL_NULL.

Usage:
    PYTHONPATH=. python scripts/oft_o2b1_5_routing_causal_audit.py \
        --ckpt outputs/oft_o2b1/conditional_cross_best.pt \
        --config outputs/2026-09-08/18-44-22/.hydra/config.yaml \
        --out-dir experiments/oft/o2b1_5 --device cuda:1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oft_o1_5_val_per_class import build_model_and_head, val_per_class_rows  # noqa: E402
from oft_routing_interventions import (  # noqa: E402
    all_null,
    all_on,
    node_shuffle,
    pair_constant,
    pair_sources,
    patched_router,
    source_swap,
    target_constant,
    target_node,
)

from src.data import load_mag_data  # noqa: E402
from src.models.oft_components import (  # noqa: E402
    CROSS_PAIR_KEYS,
    CROSS_PAIR_TARGET_SLOT,
    NUM_SLOTS,
    SLOT_NAMES,
)

# O2-B1 locked seed42 CONDITIONAL-CROSS reference (best-Val-Acc, val-only).
# The frozen audit MUST reproduce these exactly; a mismatch is
# HOLD_IMPLEMENTATION and the counterfactuals are not run.
LOCKED = {
    "val_acc": 0.5512897372245789,
    "val_macro_f1": 0.45748732111520135,
    "balanced_acc": 0.42789518276301874,
}
LOCKED_PER_CLASS = "experiments/oft/o2b1/conditional_cross_val_per_class.csv"
REPLAY_TOL = 1e-9


def _pp(delta: float) -> float:
    return float(delta) * 100.0


def _evaluate(model, head, data, device, transfer=None):
    """val-only metrics + per-class rows under an optional fixed gate."""
    with patched_router(model, transfer):
        return val_per_class_rows(model, head, data, device)


def _row_from_summary(intervention: str, summary: dict, base: dict, note: str = "") -> dict:
    return {
        "intervention": intervention,
        "acc": summary["val_acc"],
        "macro_f1": summary["val_macro_f1"],
        "balanced_acc": summary["balanced_acc"],
        "delta_acc_pp": _pp(summary["val_acc"] - base["val_acc"]),
        "delta_f1_pp": _pp(summary["val_macro_f1"] - base["val_macro_f1"]),
        "delta_balacc_pp": _pp(summary["balanced_acc"] - base["balanced_acc"]),
        "note": note,
    }


def _compare_rows(rows, ref_rows) -> float:
    """Max abs difference over the numeric per-class fields."""
    fields = ("recall", "precision", "f1")
    worst = 0.0
    for row, ref in zip(rows, ref_rows):
        if int(row["class_id"]) != int(ref["class_id"]):
            return float("inf")
        for field in fields:
            worst = max(worst, abs(float(row[field]) - float(ref[field])))
        for field in ("support", "predicted_count", "true_count"):
            if int(row[field]) != int(ref[field]):
                return float("inf")
    return worst


def _granularity(transfer: dict, val_idx: torch.Tensor, all_nodes: bool = True) -> dict:
    """§16 router granularity statistics on the (validation) node set."""
    p = {key: transfer[key].detach().cpu().double() for key in CROSS_PAIR_KEYS}
    val = val_idx.cpu()
    out: dict[str, object] = {}

    corr: dict[str, dict[str, float]] = {}
    for b in range(NUM_SLOTS):
        a1, a2 = pair_sources(b)
        x, y = p[a1][val].numpy(), p[a2][val].numpy()
        corr[f"target_{SLOT_NAMES[b]}"] = {
            "pair_a": a1,
            "pair_b": a2,
            "pearson": float(pearsonr(x, y).statistic),
            "spearman": float(spearmanr(x, y).statistic),
        }
    out["same_target_correlation_val"] = corr

    out["pair_mean_val"] = {key: float(p[key][val].mean()) for key in CROSS_PAIR_KEYS}
    out["pair_std_val"] = {key: float(p[key][val].std(unbiased=False)) for key in CROSS_PAIR_KEYS}
    out["pair_mean_all"] = {key: float(p[key].mean()) for key in CROSS_PAIR_KEYS}
    out["pair_std_all"] = {key: float(p[key].std(unbiased=False)) for key in CROSS_PAIR_KEYS}

    gate: dict[str, float] = {}
    diff: dict[str, float] = {}
    for b in range(NUM_SLOTS):
        a1, a2 = pair_sources(b)
        gate[f"target_{SLOT_NAMES[b]}"] = float(0.5 * (p[a1][val].mean() + p[a2][val].mean()))
        diff[f"target_{SLOT_NAMES[b]}"] = float((p[a1][val] - p[a2][val]).abs().mean())
    out["target_gate_val"] = gate
    out["source_abs_diff_val"] = diff
    if all_nodes:
        out["pair_mean_all_check"] = {key: float(p[key].mean()) for key in CROSS_PAIR_KEYS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, help="the run's .hydra/config.yaml")
    ap.add_argument("--out-dir", default="experiments/oft/o2b1_5")
    ap.add_argument("--reference-per-class", default=LOCKED_PER_CLASS)
    ap.add_argument("--shuffle-seed-start", type=int, default=1000)
    ap.add_argument("--num-shuffle-seeds", type=int, default=20)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--label", default="conditional_cross_seed42")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(args.config)
    seed = int(cfg.seed)
    data = load_mag_data(cfg, "nc", seed)
    device = torch.device(args.device)

    model, head = build_model_and_head(ckpt, cfg, data, device)
    variant = str(cfg.model.get("variant", "diag_id"))
    layer = model.diag_layers[0]
    if not hasattr(layer, "router_transfer"):
        raise SystemExit(f"variant {variant!r} has no router; the audit needs a conditional variant")

    params_before = {name: p.detach().clone() for name, p in model.named_parameters()}

    # ---- frozen router probabilities (checkpoint's own) -------------------
    H0, H1, layer_stats = model.encode_states(data.x, data.edge_index, device=device)
    edge = data.edge_index.to(device)
    with torch.no_grad():
        N = layer.neighbor_states(H0.to(device), edge)
        transfer = layer.router_transfer(H0.to(device), N)
    num_nodes = int(data.num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[data.train_idx.cpu()] = True
    val_idx = data.val_idx.cpu()

    # ---- ORIGINAL replay proof (§7) ---------------------------------------
    rows, original = _evaluate(model, head, data, device, transfer=None)
    replay = {
        key: {"measured": float(original[key]), "locked": float(LOCKED[key]),
              "abs_diff": abs(float(original[key]) - float(LOCKED[key]))}
        for key in ("val_acc", "val_macro_f1", "balanced_acc")
    }
    ref_path = Path(args.reference_per_class)
    per_class_max_diff = float("nan")
    if ref_path.exists():
        with ref_path.open(newline="", encoding="utf-8") as fh:
            ref_rows = list(csv.DictReader(fh))
        per_class_max_diff = _compare_rows(rows, ref_rows)
    worst = max(item["abs_diff"] for item in replay.values())
    print(f"=== O2-B1.5 Part A frozen routing audit [{args.label}] ===")
    print(f"variant={variant}  seed={seed}  ckpt={args.ckpt}  device={args.device}")
    print(f"ORIGINAL replay: acc={original['val_acc']:.10f} f1={original['val_macro_f1']:.10f} "
          f"balacc={original['balanced_acc']:.10f}")
    print(f"  max |measured - locked| = {worst:.3e}   "
          f"per-class max abs diff vs {ref_path.name} = {per_class_max_diff:.3e}")
    if worst > REPLAY_TOL or not (per_class_max_diff <= REPLAY_TOL):
        raise SystemExit(
            "HOLD_IMPLEMENTATION: ORIGINAL replay does not reproduce the O2-B1 "
            f"reference (metric diff {worst:.3e}, per-class diff {per_class_max_diff:.3e})"
        )
    print("  ORIGINAL replay VERIFIED (frozen path reproduces O2-B1 exactly).")

    with (out_dir / "routing_original_val_per_class.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- counterfactuals --------------------------------------------------
    results: list[dict] = [_row_from_summary("ORIGINAL", original, original, "checkpoint router")]
    summary_by_name: dict[str, dict] = {"ORIGINAL": original}
    for name, fixed in (
        ("PAIR_CONSTANT", pair_constant(transfer, train_mask)),
        ("TARGET_NODE", target_node(transfer)),
        ("TARGET_CONSTANT", target_constant(transfer, train_mask)),
        ("SOURCE_SWAP", source_swap(transfer)),
        ("ALL_ON", all_on(transfer)),
        ("ALL_NULL", all_null(transfer)),
    ):
        _, summary = _evaluate(model, head, data, device, transfer=fixed)
        summary_by_name[name] = summary
        results.append(_row_from_summary(name, summary, original))
        print(f"  {name:<16} acc={summary['val_acc']:.6f} f1={summary['val_macro_f1']:.6f} "
              f"balacc={summary['balanced_acc']:.6f} "
              f"(dAcc={_pp(summary['val_acc'] - original['val_acc']):+.2f}pp "
              f"dF1={_pp(summary['val_macro_f1'] - original['val_macro_f1']):+.2f}pp "
              f"dBal={_pp(summary['balanced_acc'] - original['balanced_acc']):+.2f}pp)")

    shuffle_rows: list[dict] = []
    for offset in range(int(args.num_shuffle_seeds)):
        shuffle_seed = int(args.shuffle_seed_start) + offset
        fixed = node_shuffle(transfer, val_idx, shuffle_seed)
        _, summary = _evaluate(model, head, data, device, transfer=fixed)
        shuffle_rows.append({
            "shuffle_seed": shuffle_seed,
            "acc": summary["val_acc"],
            "macro_f1": summary["val_macro_f1"],
            "balanced_acc": summary["balanced_acc"],
        })
    accs = torch.tensor([row["acc"] for row in shuffle_rows], dtype=torch.float64)
    f1s = torch.tensor([row["macro_f1"] for row in shuffle_rows], dtype=torch.float64)
    bals = torch.tensor([row["balanced_acc"] for row in shuffle_rows], dtype=torch.float64)
    shuffle_summary = {
        "n_seeds": len(shuffle_rows),
        "seed_start": int(args.shuffle_seed_start),
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std(unbiased=False)),
        "acc_min": float(accs.min()), "acc_max": float(accs.max()),
        "macro_f1_mean": float(f1s.mean()), "macro_f1_std": float(f1s.std(unbiased=False)),
        "macro_f1_min": float(f1s.min()), "macro_f1_max": float(f1s.max()),
        "balanced_acc_mean": float(bals.mean()),
        "balanced_acc_std": float(bals.std(unbiased=False)),
        "balanced_acc_min": float(bals.min()), "balanced_acc_max": float(bals.max()),
    }
    shuffle_mean = {
        "val_acc": shuffle_summary["acc_mean"],
        "val_macro_f1": shuffle_summary["macro_f1_mean"],
        "balanced_acc": shuffle_summary["balanced_acc_mean"],
    }
    results.append(_row_from_summary(
        "NODE_SHUFFLE_MEAN", shuffle_mean, original,
        f"{len(shuffle_rows)} seeds {args.shuffle_seed_start}-"
        f"{int(args.shuffle_seed_start) + len(shuffle_rows) - 1}; "
        f"acc_std={shuffle_summary['acc_std'] * 100:.3f}pp "
        f"f1_std={shuffle_summary['macro_f1_std'] * 100:.3f}pp "
        f"balacc_std={shuffle_summary['balanced_acc_std'] * 100:.3f}pp",
    ))
    print(f"  {'NODE_SHUFFLE':<16} acc={shuffle_mean['val_acc']:.6f} "
          f"f1={shuffle_mean['val_macro_f1']:.6f} balacc={shuffle_mean['balanced_acc']:.6f} "
          f"(mean of {len(shuffle_rows)} seeds)")

    # ---- invariants: no parameter changed, payload/diag untouched ---------
    param_drift = max(
        float((p.detach() - params_before[name]).abs().max())
        for name, p in model.named_parameters()
    )
    with torch.no_grad():
        payloads = layer.cross_messages(N)
        with patched_router(model, all_on(transfer)):
            H1_patched, _ = layer.propagate(H0.to(device), edge)
        with patched_router(model, all_null(transfer)):
            H1_null, _ = layer.propagate(H0.to(device), edge)
    print(f"  parameter drift after all interventions: {param_drift:.3e}")

    # ---- outputs ----------------------------------------------------------
    with (out_dir / "routing_causal_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    with (out_dir / "routing_shuffle_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(shuffle_rows[0].keys()))
        writer.writeheader()
        writer.writerows(shuffle_rows)

    granularity = _granularity(transfer, val_idx)
    payload = {
        "label": args.label,
        "variant": variant,
        "seed": seed,
        "ckpt": args.ckpt,
        "device": args.device,
        "n_val": int(data.val_idx.numel()),
        "n_train": int(data.train_idx.numel()),
        "original_replay": replay,
        "per_class_max_abs_diff_vs_reference": per_class_max_diff,
        "parameter_drift_after_interventions": param_drift,
        "shuffle": shuffle_summary,
        "granularity": granularity,
        "interventions": {
            name: {
                "acc": summary["val_acc"],
                "macro_f1": summary["val_macro_f1"],
                "balanced_acc": summary["balanced_acc"],
                "delta_acc_pp": _pp(summary["val_acc"] - original["val_acc"]),
                "delta_f1_pp": _pp(summary["val_macro_f1"] - original["val_macro_f1"]),
                "delta_balacc_pp": _pp(summary["balanced_acc"] - original["balanced_acc"]),
            }
            for name, summary in summary_by_name.items()
        },
    }
    with (out_dir / "routing_granularity_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print("\nsame-target routing correlation (val):")
    for target, item in granularity["same_target_correlation_val"].items():
        print(f"  {target:<10} {item['pair_a']:>10} vs {item['pair_b']:<10} "
              f"pearson={item['pearson']:+.4f} spearman={item['spearman']:+.4f}")
    print("\ntarget gate (mean of the two incoming sources) / source |diff|:")
    for target in granularity["target_gate_val"]:
        print(f"  {target:<10} gate={granularity['target_gate_val'][target]:.4f}  "
              f"|p_a1 - p_a2|={granularity['source_abs_diff_val'][target]:.4f}")
    print(f"\nwrote {out_dir}/routing_causal_summary.csv, routing_shuffle_results.csv, "
          f"routing_granularity_stats.json, routing_original_val_per_class.csv")
    del H1, H1_patched, H1_null, payloads, layer_stats


if __name__ == "__main__":
    main()
