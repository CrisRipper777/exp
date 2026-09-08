"""O1.5-D offline per-class validation diagnostic.

Loads an OFT O1.5 checkpoint (best-Val-Acc model+head state, saved by
task.save_ckpt_path) and the Movies data, then reports per-class
support / recall / precision / F1 / predicted count / true count on the
VALIDATION split only. Never touches test_idx. Pure post-hoc analysis:
no label information is fed into the model.

Usage:
    python scripts/oft_o1_5_val_per_class.py \
        --ckpt outputs/oft_o1_5/diag_id_best.pt \
        --config outputs/2026-09-08/17-32-40/.hydra/config.yaml \
        --out experiments/oft/o1_5/diag_id_val_per_class.csv
"""

from __future__ import annotations

import argparse
import csv

import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
)

from src.data import load_mag_data
from src.models.oft_mag import Model
from src.tasks.common import load_state_dict_cpu


def build_model_and_head(ckpt: dict, cfg, data, device: torch.device):
    """Rebuild the exact trained model+head from a saved NC checkpoint."""
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
        "y": data.y,
        "train_idx": data.train_idx,
    }
    model = Model(cfg, data_info).to(device)
    head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    load_state_dict_cpu(model, ckpt["model_state"])
    load_state_dict_cpu(head, ckpt["head_state"])
    model.eval()
    return model, head


def val_per_class_rows(model, head, data, device: torch.device):
    """Validation-only per-class table + summary metrics.

    Returns (rows, summary) where rows carry class_id / support / recall /
    precision / f1 / predicted_count / true_count (the O1.5-D schema, reused by
    the O2-A diagnostics so there is only one implementation) and summary holds
    val_acc / val_macro_f1 / balanced_acc / n_val. Never touches test_idx.
    """
    z = model.inference(data.x, data.edge_index, device=device)  # CPU [N, out]
    with torch.no_grad():
        logits = head(z[data.val_idx].to(device)).cpu()
    pred = logits.argmax(dim=-1)
    target = data.y[data.val_idx]
    num_classes = int(data.num_classes)

    acc = float((pred == target).float().mean())
    macro_f1 = float(f1_score(target.numpy(), pred.numpy(), average="macro", zero_division=0))
    labels = list(range(num_classes))
    p, r, f, _ = precision_recall_fscore_support(
        target.numpy(), pred.numpy(), labels=labels, zero_division=0
    )
    true_counts = torch.bincount(target, minlength=num_classes).numpy()
    pred_counts = torch.bincount(pred, minlength=num_classes).numpy()

    rows = [
        {
            "class_id": c,
            "support": int(true_counts[c]),
            "recall": r[c],
            "precision": p[c],
            "f1": f[c],
            "predicted_count": int(pred_counts[c]),
            "true_count": int(true_counts[c]),
        }
        for c in range(num_classes)
    ]
    summary = {
        "val_acc": acc,
        "val_macro_f1": macro_f1,
        "balanced_acc": float(r.mean()),
        "n_val": int(target.numel()),
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, help="the run's .hydra/config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--label", default="", help="free-form row/echo label")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(args.config)
    seed = int(cfg.seed)
    data = load_mag_data(cfg, "nc", seed)
    device = torch.device(args.device)

    model, head = build_model_and_head(ckpt, cfg, data, device)
    rows, summary = val_per_class_rows(model, head, data, device)

    # checkpoint epoch = the epoch whose val acc was best (best-Acc selection).
    best_ep = ckpt.get("best_epoch")
    print(f"[{args.label or 'per-class'}] n_val={summary['n_val']} "
          f"val_acc={summary['val_acc']:.6f} "
          f"val_macro_f1={summary['val_macro_f1']:.6f} "
          f"balanced_acc={summary['balanced_acc']:.6f} "
          f"(ckpt best_epoch={best_ep})")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")

    print("\ntop 5 classes by F1:")
    for row in sorted(rows, key=lambda r: -r["f1"])[:5]:
        print(f"  cls {row['class_id']:>2}: F1 {row['f1']:.4f}  support {row['support']:>4}  "
              f"recall {row['recall']:.4f}  precision {row['precision']:.4f}  pred {row['predicted_count']:>4}")
    print("bottom 5 classes by F1:")
    for row in sorted(rows, key=lambda r: r["f1"])[:5]:
        print(f"  cls {row['class_id']:>2}: F1 {row['f1']:.4f}  support {row['support']:>4}  "
              f"recall {row['recall']:.4f}  precision {row['precision']:.4f}  pred {row['predicted_count']:>4}")


if __name__ == "__main__":
    main()
