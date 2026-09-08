"""O2-A offline cross-ownership diagnostics (validation-only, post-hoc).

Reloads a best-Val-Acc OFT checkpoint, reruns the exact full-graph encoding
path (``Model.encode_states``) and reports, on the VALIDATION split only:

  * post-transition ownership diagnostics (pre/post slot cosines, post norms,
    per-slot state drift) — computed for ANY variant, so diag_id / DUP /
    STATIC-CROSS are directly comparable;
  * the cross-branch learned relative update magnitudes for the O2-A variants
    (per-target cross/diag ratios and the six per-pair contributions);
  * Val Accuracy / Macro-F1 / balanced accuracy and the per-class table via the
    shared helper in ``scripts/oft_o1_5_val_per_class.py`` (single
    implementation, no duplicate per-class code).

Never touches test_idx. Pure post-hoc analysis: no label information reaches
the model.

Usage:
    python scripts/oft_o2a_diagnostics.py \
        --ckpt outputs/oft_o2a/static_cross_best.pt \
        --config outputs/<run>/.hydra/config.yaml \
        --out-stats experiments/oft/o2a/static_cross_stats.json \
        --out-per-class experiments/oft/o2a/static_cross_val_per_class.csv \
        --label static_cross
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
    post_transition_stats,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, help="the run's .hydra/config.yaml")
    ap.add_argument("--out-stats", default="", help="JSON with the scalar diagnostics")
    ap.add_argument("--out-per-class", default="", help="per-class val CSV")
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

    H0, H1, layer_stats = model.encode_states(data.x, data.edge_index, device=device)
    post = post_transition_stats(H0, H1)

    rows, summary = val_per_class_rows(model, head, data, device)

    stats: dict[str, float] = {}
    stats["params_model"] = float(sum(p.numel() for p in model.parameters()))
    stats["params_head"] = float(sum(p.numel() for p in head.parameters()))
    for slot in SLOT_NAMES:
        stats[f"diag_update_ratio_{slot}"] = float(layer_stats["diag_update_ratio"][SLOT_NAMES.index(slot)])
        stats[f"neighbor_norm_{slot}"] = float(layer_stats["neighbor_norm"][SLOT_NAMES.index(slot)])
    if "cross_update_ratio" in layer_stats:
        for slot in SLOT_NAMES:
            idx = SLOT_NAMES.index(slot)
            stats[f"cross_update_ratio_{slot}"] = float(layer_stats["cross_update_ratio"][idx])
            stats[f"cross_to_diag_ratio_{slot}"] = float(layer_stats["cross_to_diag_ratio"][idx])
        for key in CROSS_PAIR_KEYS:
            stats[f"cross_pair_ratio_{key}"] = float(layer_stats[f"cross_pair_ratio_{key}"])
    for key, value in post.items():
        stats[key] = float(value)
    stats.update({key: float(value) for key, value in summary.items()})

    print(f"=== O2-A diagnostics [{args.label or variant}] ===")
    print(f"variant={variant}  seed={seed}  ckpt={args.ckpt}")
    print(f"val_acc={summary['val_acc']:.6f}  val_macro_f1={summary['val_macro_f1']:.6f}  "
          f"balanced_acc={summary['balanced_acc']:.6f}  n_val={summary['n_val']}")
    print("\npost-transition ownership:")
    for pair in ("c_pt", "c_pv", "pt_pv"):
        print(f"  cos {pair:>6}: pre {stats[f'pre_cos_{pair}']:.4f} -> post {stats[f'post_cos_{pair}']:.4f}")
    for slot in SLOT_NAMES:
        print(f"  slot {slot:>2}: post_norm {stats[f'post_norm_{slot}']:.4f}  "
              f"state_drift {stats[f'state_drift_{slot}']:.4f}  "
              f"diag_ratio {stats[f'diag_update_ratio_{slot}']:.4f}")
    if "cross_update_ratio_c" in stats:
        print("\ncross branch:")
        for slot in SLOT_NAMES:
            print(f"  target {slot:>2}: cross_ratio {stats[f'cross_update_ratio_{slot}']:.4f}  "
                  f"cross/diag {stats[f'cross_to_diag_ratio_{slot}']:.4f}")
        for key in CROSS_PAIR_KEYS:
            print(f"  pair {key:>10}: {stats[f'cross_pair_ratio_{key}']:.4f}")

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
