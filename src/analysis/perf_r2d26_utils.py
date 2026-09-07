"""R2-Design-2.6 shared analysis layer
(docs/BiAxis_R2_Design_2_6_Strong_Parent_Readout_Integration.md).

Discipline:
    - A0 is the ONLY primary parent (R1-baseline A0 checkpoints, structure
      bitwise == biaxis_final, max |val acc delta| 0.176pp — disclosed).
    - Frozen-parent training first (plan §25): side experts / readout / aux
      heads / classifier train; A0 untouched. Parent adaptation (D2.6-D)
      only via the explicit schedules.
    - Val only, NEVER test.
"""

from __future__ import annotations

import csv
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
FACTOR_NAMES = ("C", "Pt", "Pv")
CLASSIFIER_SEED = 20260904

R2D26_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d26"

READOUT_TYPES = (
    "no_compression_concat",
    "factor_hop_concat",
    "residual_side_fusion",
    "base_anchored_hier_attention",
    "readout_only_control",
)

# variant labels used by the experiment drivers
VARIANTS = {
    "A0_BASE": ("a0_base", "hop"),
    "NC_HOP": ("no_compression_concat", "hop"),
    "NC_H1": ("no_compression_concat", "h1"),
    "FHC_HOP": ("factor_hop_concat", "hop"),
    "FHC_H1": ("factor_hop_concat", "h1"),
    "RSF_HOP": ("residual_side_fusion", "hop"),
    "RSF_H1": ("residual_side_fusion", "h1"),
    "HIER_HOP": ("base_anchored_hier_attention", "hop"),
    "HIER_H1": ("base_anchored_hier_attention", "h1"),
    "READOUT_ONLY": ("readout_only_control", "hop"),
}

CAUSAL_KEYS = ("full", "h2_zero", "h2_to_h1", "h2_shuffle",
               "pt_h2_off", "c_h2_off", "pv_h2_off",
               "s_c_off", "s_pt_off", "s_pv_off", "side_off")

WARMUP_EPOCHS = 10
MIN_LR = 1e-5


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


@dataclass
class StrongSetup:
    dataset: str
    seed: int
    cfg: object
    data: object
    parent: nn.Module
    device: torch.device


def load_a0_parent(dataset: str, seed: int, device: torch.device) -> StrongSetup:
    """Load the A0 parent (R1-baseline A0 checkpoint). NEVER test."""
    from src.analysis.perf_r1_utils import load_r1_setup

    setup = load_r1_setup(dataset, seed, "A0", device)
    assert_no_test_access(setup.data)
    return StrongSetup(dataset, seed, setup.cfg, setup.data, setup.model, device)


def assert_no_test_access(data) -> None:
    assert data.train_idx is not None and data.val_idx is not None


def load_or_make_head_init(
    init_path: Path, out_dim: int, num_classes: int, device: torch.device
) -> nn.Module:
    from src.analysis.perf_r2d25_utils import load_or_make_head_init as _loader

    return _loader(init_path, out_dim, num_classes, device)


# ---------------------------------------------------------------------------
# Warmup10+cosine (same schedule as D2.5-C)
# ---------------------------------------------------------------------------


def scheduled_lr(epoch: int, total_epochs: int, base_lr: float,
                 warmup: int = WARMUP_EPOCHS, min_lr: float = MIN_LR) -> float:
    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    frac = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * frac


# ---------------------------------------------------------------------------
# Frozen-parent training loop (D2.6-A / D2.6-B; D2.6-D S0)
# ---------------------------------------------------------------------------


def train_strong_parent(
    data, model: nn.Module, head: nn.Module, device: torch.device,
    *, total_epochs: int = 300, patience: int = 30,
    deep_sup_lambda: float = 0.1,
    history_callback=None,
) -> dict:
    """A0 fully frozen; side branch + aux heads + classifier train.
    lr 1e-3, wd 1e-4, warmup10+cosine, best Val Acc. Val only."""
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
        lr=1e-3, weight_decay=1e-4,
    )

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
        z, tokens, _s, _attn = model.forward_with_experts(x, ei)
        loss = criterion(head(z[train_idx]), y_train)
        if deep_sup_lambda > 0.0 and tokens:
            loss = loss + deep_sup_lambda * model.deep_supervision_loss(
                tokens, train_idx, y_train)
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
        row = {
            "epoch": epoch,
            "lr": float(scheduled_lr(epoch, total_epochs, 1e-3)),
            "train_ce": float(loss.item()),
            "val_acc": acc,
        }
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


# ---------------------------------------------------------------------------
# Causal evaluation (plan §30): no retraining, best-checkpoint forward
# ---------------------------------------------------------------------------


def causal_metrics(model: nn.Module, head: nn.Module, x, ei, data, device,
                   causal_keys=CAUSAL_KEYS) -> dict:
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


# ---------------------------------------------------------------------------
# Parent-adaptation schedules (plan §34-§38)
# ---------------------------------------------------------------------------

P0_PREFIXES = ("factorizer.", "recon_text_head.", "recon_visual_head.")


def parent_unfreeze_group(model: nn.Module, schedule: str) -> list[str]:
    """Parameter-name prefixes unfrozen at epoch 31 for S1/S2. P0 factorizer
    is ALWAYS frozen (S3 is a separate optional, never automatic)."""
    if schedule == "S1":
        return ("fusion.",)  # final fusion/readout only
    if schedule == "S2":
        # graph transformation/readout blocks, P0 frozen
        return ("operator.", "graph_w0", "graph_norm.", "null_score",
                "transport_scorer.", "struct_signature_mlp.", "edge_token_mlp.",
                "relation_prototypes", "fusion.")
    raise ValueError(schedule)


def train_parent_adapt(
    data, model: nn.Module, head: nn.Module, device: torch.device,
    *, schedule: str, total_epochs: int = 300, patience: int = 30,
    deep_sup_lambda: float = 0.1, unfreeze_epoch: int = 31,
    history_callback=None,
) -> dict:
    """S0 FROZEN (== train_strong_parent) / S1 READOUT_ADAPT / S2
    GRAPH_READOUT_ADAPT. Epochs 1-30 frozen parent; 31+ the schedule's
    parent group trains at lr 1e-4 (side/head stay at 1e-3). Parent runs
    in EVAL mode throughout (no parent dropout — documented)."""
    assert schedule in ("S0", "S1", "S2"), schedule
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

    side_opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    parent_opt = None
    unfreeze_prefixes = parent_unfreeze_group(model, schedule) if schedule != "S0" else ()
    parent_unfrozen = False

    def _apply_lr(epoch: int) -> None:
        for pg in side_opt.param_groups:
            pg["lr"] = scheduled_lr(epoch, total_epochs, 1e-3)
        if parent_opt is not None:
            for pg in parent_opt.param_groups:
                pg["lr"] = scheduled_lr(epoch, total_epochs, 1e-4)

    history: list[dict] = []
    best_acc, best_epoch, best_state = -1.0, None, None
    patience_left = patience
    stop_epoch = total_epochs
    for epoch in range(1, total_epochs + 1):
        if schedule != "S0" and epoch == unfreeze_epoch:
            parent_params = [
                p for n, p in model.parent.named_parameters()
                if any(n.startswith(pre) for pre in unfreeze_prefixes)
            ]
            for p in parent_params:
                p.requires_grad_(True)
            parent_opt = torch.optim.AdamW(parent_params, lr=1e-4, weight_decay=1e-4)
            model.parent_frozen = False
            parent_unfrozen = True
        _apply_lr(epoch)
        side_opt.zero_grad(set_to_none=True)
        if parent_opt is not None:
            parent_opt.zero_grad(set_to_none=True)
        model.train()
        z, tokens, _s, _attn = model.forward_with_experts(x, ei)
        loss = criterion(head(z[train_idx]), y_train)
        if deep_sup_lambda > 0.0 and tokens:
            loss = loss + deep_sup_lambda * model.deep_supervision_loss(
                tokens, train_idx, y_train)
        loss.backward()
        all_params = list(model.parameters()) + list(head.parameters())
        if parent_unfrozen:
            all_params = all_params + [
                p for n, p in model.parent.named_parameters()
                if p.requires_grad]
        nn.utils.clip_grad_norm_(all_params, 1.0)
        side_opt.step()
        if parent_opt is not None:
            parent_opt.step()
        with torch.no_grad():
            model.eval()
            z_eval, _, _, _, _ = model(x, ei)
            pred_v = head(z_eval[val_idx]).argmax(-1)
            acc = float((pred_v == y_val).float().mean().item())
            del z_eval
        row = {"epoch": epoch, "train_ce": float(loss.item()), "val_acc": acc,
               "parent_unfrozen": int(parent_unfrozen)}
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "parent": {k: v.detach().clone() for k, v in model.parent.state_dict().items()},
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
    model.parent.load_state_dict(best_state["parent"])
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
        "parent_unfrozen": parent_unfrozen,
        "z_best": z_best,
    }


# ---------------------------------------------------------------------------
# Parent drift (plan §39): current adapted parent z vs the frozen-A0 z
# ---------------------------------------------------------------------------


def parent_drift_metrics(setup: StrongSetup, model: nn.Module, x, ei) -> dict:
    from src.analysis.perf_r2d15_utils import linear_cka, mean_cosine, mean_relative_l2

    num_nodes = int(x.size(0))
    with torch.no_grad():
        _, z_now = model._parent_pieces(x, ei, num_nodes)
        f0_ref, z_ref = _frozen_a0_reference(setup, x, ei)
    return {
        "parent_z_cka": float(linear_cka(z_now, z_ref)),
        "parent_z_cosine": mean_cosine(z_now, z_ref),
        "parent_z_rel_l2": mean_relative_l2(z_now, z_ref),
    }


def _frozen_a0_reference(setup: StrongSetup, x, ei):
    """Re-run the parent pieces with the ORIGINAL A0 weights (the setup's
    parent is overwritten by adaptation, so rebuild from the checkpoint)."""
    from src.analysis.perf_r1_utils import load_r1_setup

    ref_setup = load_r1_setup(setup.dataset, setup.seed, "A0", setup.device)
    model = ref_setup.model
    num_nodes = int(x.size(0))
    with torch.no_grad():
        factors, _ = model._encode(x)
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        graph_out = model._graph_update(f_block, ei, num_nodes)
        f_tilde = graph_out["f_tilde"]
        z = model.fusion(torch.cat([f_tilde[:, 0], f_tilde[:, 1], f_tilde[:, 2]], dim=-1))
    return f_block, z
