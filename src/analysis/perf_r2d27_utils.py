"""R2-Design-2.7 shared analysis layer
(docs/BiAxis_R2_Design_2_7_PreAggregation_Neighbor_Utility_Audit.md).

Discipline:
    - A0 is the ONLY primary parent (R1-baseline A0 checkpoints, disclosed);
      fully frozen in D2.7-A..E (parent adaptation optional later).
    - No auxiliary / edge-label supervision: the edge scorer learns only
      through the node-task CE (plan §18).
    - Val only, NEVER test.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
TARGET_DATASETS = ["Movies", "Toys", "Grocery"]
GUARD_DATASETS = ["ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
CLASSIFIER_SEED = 20260904

R2D27_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d27"

# variant label -> model mode
VARIANTS = {
    "A0_BASE": "a0_base",
    "UNIFORM": "uniform",
    "TARGET_NULL_ONLY": "target_null_only",
    "GENERIC_EDGE": "generic_edge",
    "DIAG_EDGE": "diag_edge",
    "PAIR_EDGE": "pair_edge",
    "SEMANTIC_SIM": "semantic_sim",
    "POST_PAIR": "post_pair",
    "SOURCE_FACTOR_ONLY": "source_factor_only",
    "TARGET_FACTOR_ONLY": "target_factor_only",
    "PAIR_TRANSFORM_UNIFORM": "pair_transform_uniform",
    "PAIR_TRANSFORM_PRE": "pair_transform_pre",
}


@dataclass
class UtilitySetup:
    dataset: str
    seed: int
    cfg: object
    data: object
    parent: nn.Module
    device: torch.device


def load_a0_parent(dataset: str, seed: int, device: torch.device) -> UtilitySetup:
    from src.analysis.perf_r1_utils import load_r1_setup

    setup = load_r1_setup(dataset, seed, "A0", device)
    assert_no_test_access(setup.data)
    return UtilitySetup(dataset, seed, setup.cfg, setup.data, setup.model, device)


def assert_no_test_access(data) -> None:
    assert data.train_idx is not None and data.val_idx is not None


def load_or_make_head_init(
    init_path: Path, out_dim: int, num_classes: int, device: torch.device
) -> nn.Module:
    from src.analysis.perf_r2d25_utils import load_or_make_head_init as _loader

    return _loader(init_path, out_dim, num_classes, device)


def scheduled_lr(epoch: int, total_epochs: int, base_lr: float,
                 warmup: int = 10, min_lr: float = 1e-5) -> float:
    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    frac = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * frac


def train_utility_model(
    data, model: nn.Module, head: nn.Module, device: torch.device,
    *, total_epochs: int = 300, patience: int = 30,
    history_callback=None,
) -> dict:
    """A0 fully frozen; side/scorer/payload/classifier lr 1e-3 wd 1e-4,
    warmup10+cosine, grad clip 1.0, best Val Acc. No aux loss (plan §18)."""
    assert_no_test_access(data)
    model = model.to(device)
    head = head.to(device)
    model.parent.eval()
    model.parent_frozen = True
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    train_idx = data.train_idx.to(device)
    y_train = data.y[data.train_idx].to(device)
    val_idx = data.val_idx.to(device)
    y_val = data.y[data.val_idx].to(device)
    criterion = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=1e-3, weight_decay=1e-4)

    def _apply_lr(epoch: int) -> None:
        for pg in opt.param_groups:
            pg["lr"] = scheduled_lr(epoch, total_epochs, 1e-3)

    history: list[dict] = []
    best_acc, best_epoch, best_state = -1.0, None, None
    patience_left = patience
    stop_epoch = total_epochs
    for epoch in range(1, total_epochs + 1):
        _apply_lr(epoch)
        opt.zero_grad(set_to_none=True)
        model.train()
        z, _, _, _, _ = model(x, ei)
        loss = criterion(head(z[train_idx]), y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        with torch.no_grad():
            model.eval()
            z_eval, _, _, _, _ = model(x, ei)
            pred_v = head(z_eval[val_idx]).argmax(-1)
            acc = float((pred_v == y_val).float().mean().item())
            del z_eval
        row = {"epoch": epoch, "lr": float(scheduled_lr(epoch, total_epochs, 1e-3)),
               "train_ce": float(loss.item()), "val_acc": acc}
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                stop_epoch = epoch
                break
        if history_callback is not None:
            history_callback(row)
        history.append(row)

    model.load_state_dict(best_state["model"])
    head.load_state_dict(best_state["head"])
    model.eval()
    head.eval()
    with torch.no_grad():
        z_best, _, _, _, _ = model(x, ei)
    from src.analysis.perf_r2d15_utils import val_metrics_with_head

    m_full = val_metrics_with_head(head, z_best, data, device)
    return {
        "best_val_acc": best_acc,
        "best_val_macro_f1": m_full["val_macro_f1"],
        "per_class_f1": m_full["per_class_f1"],
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "history": history,
        "z_best": z_best,
    }


def causal_metrics(model, head, x, ei, data, device, causal_keys) -> dict:
    from src.analysis.perf_r2d15_utils import val_metrics_with_head

    model.eval()
    head.eval()
    out = {}
    with torch.no_grad():
        for key in causal_keys:
            z, _, _, _, _ = model(x, ei, causal=key)
            m = val_metrics_with_head(head, z, data, device)
            out[key] = {"val_acc": m["val_acc"], "val_macro_f1": m["val_macro_f1"]}
            del z
    return out
