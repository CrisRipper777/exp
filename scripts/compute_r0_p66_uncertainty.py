"""R0-P6.6: per-receiver prediction probabilities (inference only).

The relation CSV stores full_confidence (max softmax prob of the frozen
classifier at the full representation z_i) but not the full probability
vector. Entropy and top1-top2 margin are not in the CSV and must not be
guessed; this script recomputes receiver-level probabilities with the FROZEN
best-val checkpoint (no training, no test labels; receivers come from the
existing relation CSV, i.e. validation receivers only).

Verification invariant: recomputed confidence must match the CSV
full_confidence row-wise (same code path, fp32) to <1e-6.

Output: one row per unique CSV receiver:
    receiver_i, confidence, entropy_norm, margin, correct, label_i

Usage:
    conda run -n yhf_env python scripts/compute_r0_p66_uncertainty.py \
        --dataset Grocery --train-seed 42 \
        --ckpt experiments/r0/checkpoints/Grocery_seed42_probe.pt \
        --csv experiments/r0/results/Grocery/seed_42/r0_relation_diagnostics.csv \
        --out experiments/r0/results/Grocery/seed_42/r0_p66_receiver_uncertainty.csv \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_r0_counterfactual import load_setup  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="Grocery")
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


@torch.no_grad()
def run(args) -> None:
    import numpy as np

    NS = argparse.Namespace(
        dataset=args.dataset, train_seed=args.train_seed,
        ckpt=args.ckpt, device=args.device,
    )
    cfg, data, model, head, device = load_setup(NS)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    labels = data.y.to(device)
    num_classes = int(data.num_classes)
    logger = print

    with args.csv.open() as f:
        rows = list(csv.DictReader(f))
    recv = np.array([int(r["receiver_i"]) for r in rows], dtype=np.int64)
    csv_conf = np.array([float(r["full_confidence"]) for r in rows])

    receivers = np.unique(recv)
    logger(f"[rows] {len(rows)} relations | unique receivers {len(receivers)}")
    assert set(receivers.tolist()) <= set(data.val_idx.tolist()), \
        "CSV contains non-validation receivers"

    # recompute z_full once (full graph forward, same code path as the CSV run)
    diag = model.forward_with_deltas(x, edge_index)
    z_i = diag["z_full"][torch.from_numpy(receivers).to(device)]
    logits = head(z_i)                          # fp32, [R, C]
    p = F.softmax(logits, dim=-1)
    conf = p.max(dim=-1).values
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1) / torch.log(
        torch.tensor(float(num_classes), device=device))
    top2 = torch.topk(p, k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    y = labels[torch.from_numpy(receivers).to(device)]
    correct = logits.argmax(dim=-1) == y

    # invariant: recomputed confidence == CSV full_confidence (row-wise)
    row_conf = conf[torch.from_numpy(np.searchsorted(receivers, recv)).to(device)]
    dev = (row_conf - torch.from_numpy(csv_conf).to(device)).abs().max().item()
    logger(f"[check] max|recomputed confidence - CSV full_confidence| = {dev:.3e} "
           f"(require <1e-6)")
    assert dev < 1e-6, "confidence recompute disagrees with CSV full_confidence"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["receiver_i", "confidence", "entropy_norm", "margin",
                    "correct", "label_i"])
        conf_c, ent_c, marg_c = conf.cpu().numpy(), ent.cpu().numpy(), margin.cpu().numpy()
        corr_c, y_c = correct.cpu().numpy(), y.cpu().numpy()
        for t in range(len(receivers)):
            w.writerow([int(receivers[t]), float(conf_c[t]), float(ent_c[t]),
                        float(marg_c[t]), bool(corr_c[t]), int(y_c[t])])
    logger(f"[out] {len(receivers)} receivers -> {args.out}")
    logger(f"[stats] mean conf={conf.mean().item():.3f} | "
           f"mean entropy_norm={ent.mean().item():.3f} | "
           f"mean margin={margin.mean().item():.3f} | "
           f"correct rate={correct.float().mean().item():.3f}")


if __name__ == "__main__":
    run(_parse_args())
