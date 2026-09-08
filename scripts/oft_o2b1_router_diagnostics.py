"""O2-B1 offline router diagnostics (validation-only, post-hoc).

Reloads a best-Val-Acc O2-B1 checkpoint (``conditional_cross`` /
``conditional_cross_dup``), reruns the exact full-graph encoding path and
reports, on the VALIDATION split only:

  * per-pair transfer-probability distribution: mean / std / p10 / p25 / p50 /
    p75 / p90 / frac(<0.25) / frac(>0.75) / binary entropy — the "is the router
    actually node-dependent?" evidence;
  * the same distribution split by correctly vs incorrectly classified val
    nodes, and per class (exploratory only, never a causal claim);
  * raw vs effective per-pair cross magnitudes and the per-target cross/diag
    ratios from the layer stats;
  * post-transition ownership diagnostics (pre/post slot cosines, post norms,
    state drift);
  * Val Accuracy / Macro-F1 / balanced accuracy and the per-class table via the
    shared helper in ``scripts/oft_o1_5_val_per_class.py`` (one per-class
    implementation, reused — no second copy).

Never touches test_idx. Pure post-hoc analysis: no label information reaches
the model.

Usage:
    python scripts/oft_o2b1_router_diagnostics.py \
        --ckpt outputs/oft_o2b1/conditional_cross_best.pt \
        --config outputs/<run>/.hydra/config.yaml \
        --out-stats experiments/oft/o2b1/conditional_cross_stats.json \
        --out-per-class experiments/oft/o2b1/conditional_cross_val_per_class.csv \
        --out-nodes experiments/oft/o2b1/conditional_cross_router_nodes_val.csv \
        --label conditional_cross
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oft_o1_5_val_per_class import build_model_and_head, val_per_class_rows  # noqa: E402

from src.data import load_mag_data  # noqa: E402
from src.models.oft_components import (  # noqa: E402
    CROSS_PAIR_KEYS,
    SLOT_NAMES,
    binary_entropy,
    post_transition_stats,
)


def _dist_summary(p: torch.Tensor) -> dict[str, float]:
    """Distribution summary of a per-node transfer probability vector."""
    q = p.detach().cpu().double()
    return {
        "mean": float(q.mean()),
        "std": float(q.std(unbiased=False)),
        "p10": float(torch.quantile(q, 0.10)),
        "p25": float(torch.quantile(q, 0.25)),
        "p50": float(torch.quantile(q, 0.50)),
        "p75": float(torch.quantile(q, 0.75)),
        "p90": float(torch.quantile(q, 0.90)),
        "frac_lt_025": float((q < 0.25).double().mean()),
        "frac_gt_075": float((q > 0.75).double().mean()),
        "entropy": float(binary_entropy(q).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, help="the run's .hydra/config.yaml")
    ap.add_argument("--out-stats", default="", help="JSON with the scalar diagnostics")
    ap.add_argument("--out-per-class", default="", help="per-class val CSV")
    ap.add_argument("--out-nodes", default="", help="per-val-node router CSV")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(args.config)
    seed = int(cfg.seed)
    data = load_mag_data(cfg, "nc", seed)
    device = torch.device(args.device)

    model, head = build_model_and_head(ckpt, cfg, data, device)
    variant = str(cfg.model.get("variant", "diag_id"))
    layer = model.diag_layers[0]
    if not hasattr(layer, "router_transfer"):
        raise SystemExit(f"variant {variant!r} has no router; O2-B1 diagnostics need a conditional variant")

    H0, H1, layer_stats = model.encode_states(data.x, data.edge_index, device=device)
    edge = data.edge_index.to(device)
    H0_dev = H0.to(device)
    with torch.no_grad():
        N = layer.neighbor_states(H0_dev, edge)
        transfer = layer.router_transfer(H0_dev, N)  # {key: [N] on device}

    val_idx = data.val_idx
    val_transfer = {key: transfer[key][val_idx.to(device)].cpu() for key in CROSS_PAIR_KEYS}
    n_val = int(val_idx.numel())

    # val predictions (argmax of the best-Val-Acc checkpoint's own head)
    z = model.inference(data.x, data.edge_index, device=device)
    with torch.no_grad():
        logits = head(z[val_idx].to(device)).cpu()
    pred = logits.argmax(dim=-1)
    target = data.y[val_idx]
    correct = (pred == target)
    classes = sorted(int(c) for c in torch.unique(target).tolist())

    rows, summary = val_per_class_rows(model, head, data, device)
    post = post_transition_stats(H0, H1)

    stats: dict[str, float] = {}
    stats["params_model"] = float(sum(p.numel() for p in model.parameters()))
    stats["params_head"] = float(sum(p.numel() for p in head.parameters()))
    for slot in SLOT_NAMES:
        idx = SLOT_NAMES.index(slot)
        stats[f"diag_update_ratio_{slot}"] = float(layer_stats["diag_update_ratio"][idx])
        stats[f"neighbor_norm_{slot}"] = float(layer_stats["neighbor_norm"][idx])
        stats[f"cross_update_ratio_{slot}"] = float(layer_stats["cross_update_ratio"][idx])
        stats[f"cross_to_diag_ratio_{slot}"] = float(layer_stats["cross_to_diag_ratio"][idx])
    for key in CROSS_PAIR_KEYS:
        stats[f"cross_pair_ratio_{key}"] = float(layer_stats[f"cross_pair_ratio_{key}"])
        stats[f"effective_cross_pair_ratio_{key}"] = float(
            layer_stats[f"effective_cross_pair_ratio_{key}"]
        )
        stats[f"transfer_mean_all_{key}"] = float(layer_stats[f"transfer_mean_{key}"])
        stats[f"transfer_std_all_{key}"] = float(layer_stats[f"transfer_std_{key}"])
        stats[f"router_entropy_all_{key}"] = float(layer_stats[f"router_entropy_{key}"])
        for name, value in _dist_summary(val_transfer[key]).items():
            stats[f"val_{name}_{key}"] = value
        # exploratory splits (never a causal claim)
        p = val_transfer[key]
        stats[f"val_transfer_correct_{key}"] = float(p[correct].mean()) if bool(correct.any()) else float("nan")
        stats[f"val_transfer_incorrect_{key}"] = float(p[~correct].mean()) if bool((~correct).any()) else float("nan")
        for c in classes:
            mask = target == c
            stats[f"val_transfer_class{c}_{key}"] = float(p[mask].mean())
    for key, value in post.items():
        stats[key] = float(value)
    stats.update({key: float(value) for key, value in summary.items()})

    print(f"=== O2-B1 router diagnostics [{args.label or variant}] ===")
    print(f"variant={variant}  seed={seed}  ckpt={args.ckpt}")
    print(f"val_acc={summary['val_acc']:.6f}  val_macro_f1={summary['val_macro_f1']:.6f}  "
          f"balanced_acc={summary['balanced_acc']:.6f}  n_val={summary['n_val']}  "
          f"correct={int(correct.sum())}/{n_val}")

    print("\nper-pair transfer probability (VAL split):")
    print(f"  {'pair':>10} {'mean':>7} {'std':>7} {'p10':>7} {'p50':>7} {'p90':>7} "
          f"{'<0.25':>7} {'>0.75':>7} {'H(p)':>7}")
    for key in CROSS_PAIR_KEYS:
        d = {name: stats[f"val_{name}_{key}"] for name in
             ("mean", "std", "p10", "p50", "p90", "frac_lt_025", "frac_gt_075", "entropy")}
        print(f"  {key:>10} {d['mean']:>7.4f} {d['std']:>7.4f} {d['p10']:>7.4f} "
              f"{d['p50']:>7.4f} {d['p90']:>7.4f} {d['frac_lt_025']:>7.4f} "
              f"{d['frac_gt_075']:>7.4f} {d['entropy']:>7.4f}")

    print("\nper-target cross / diag ratios and per-pair magnitudes:")
    for slot in SLOT_NAMES:
        print(f"  target {slot:>2}: cross_ratio {stats[f'cross_update_ratio_{slot}']:.4f}  "
              f"cross/diag {stats[f'cross_to_diag_ratio_{slot}']:.4f}")
    for key in CROSS_PAIR_KEYS:
        print(f"  pair {key:>10}: raw {stats[f'cross_pair_ratio_{key}']:.4f}  "
              f"effective {stats[f'effective_cross_pair_ratio_{key}']:.4f}")

    print("\npost-transition ownership:")
    for pair in ("c_pt", "c_pv", "pt_pv"):
        print(f"  cos {pair:>6}: pre {stats[f'pre_cos_{pair}']:.4f} -> post {stats[f'post_cos_{pair}']:.4f}")
    for slot in SLOT_NAMES:
        print(f"  slot {slot:>2}: post_norm {stats[f'post_norm_{slot}']:.4f}  "
              f"state_drift {stats[f'state_drift_{slot}']:.4f}")

    print("\ncorrect vs incorrect val nodes (mean transfer):")
    for key in CROSS_PAIR_KEYS:
        print(f"  {key:>10}: correct {stats[f'val_transfer_correct_{key}']:.4f}  "
              f"incorrect {stats[f'val_transfer_incorrect_{key}']:.4f}")

    if args.out_stats:
        out = Path(args.out_stats)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump({"label": args.label or variant, "variant": variant, "seed": seed,
                       "ckpt": args.ckpt, "stats": stats}, fh, indent=2, sort_keys=True)
        print(f"\nwrote {out}")

    if args.out_per_class:
        out = Path(args.out_per_class)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")

    if args.out_nodes:
        out = Path(args.out_nodes)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            fields = ["node_id", "true", "pred", "correct"] + [f"p_{k}" for k in CROSS_PAIR_KEYS]
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for i, node in enumerate(val_idx.tolist()):
                row = {"node_id": node, "true": int(target[i]), "pred": int(pred[i]),
                       "correct": int(correct[i])}
                for key in CROSS_PAIR_KEYS:
                    row[f"p_{key}"] = float(val_transfer[key][i])
                writer.writerow(row)
        print(f"wrote {out}")

    print("\ntop 5 classes by F1:")
    for row in sorted(rows, key=lambda r: -r["f1"])[:5]:
        print(f"  cls {row['class_id']:>2}: F1 {row['f1']:.4f}  support {row['support']:>4}  "
              f"recall {row['recall']:.4f}  precision {row['precision']:.4f}  "
              f"pred {row['predicted_count']:>4}")
    print("bottom 5 classes by F1:")
    for row in sorted(rows, key=lambda r: r["f1"])[:5]:
        print(f"  cls {row['class_id']:>2}: F1 {row['f1']:.4f}  support {row['support']:>4}  "
              f"recall {row['recall']:.4f}  precision {row['precision']:.4f}  "
              f"pred {row['predicted_count']:>4}")


if __name__ == "__main__":
    main()
