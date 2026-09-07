"""R2-Design-1.5 shared analysis layer (plan §3-§4).

Discipline:
    - frozen-model-read-only: counterfactuals are realized by replicating the
      model's own forward math with masks / fixed-common switches — trained
      weights are NEVER modified (plan §3: hooks / helpers / eval flags).
    - bitwise-consistency: every cf path is unit-tested against the model's
      own forward (no-cf must be torch.equal).
    - Val only, NEVER test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from hydra import compose, initialize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.biaxis_p1_components import neighbor_mean  # noqa: E402

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
TARGET_DATASETS = ["Movies", "Toys", "Grocery"]
GUARD_DATASETS = ["ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
FACTOR_NAMES = ["C", "Pt", "Pv"]

R2D1_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d1"
R2D15_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d15"

# Counterfactual labels (plan §7/§8). The base B0 diagonal path always stays.
FUNC_CFS = ("full", "func_off", "diag_only", "offdiag_only", "src_C", "src_Pt", "src_Pv")
SEM_CFS = ("full", "common_only", "fixed_common_residual", "both_off")

# Functional cell masks (rows = source, cols = target; factor order C/Pt/Pv).
_FUNC_CELLS = {
    "full": torch.ones(3, 3),
    "func_off": torch.zeros(3, 3),
    "diag_only": torch.eye(3),
    "offdiag_only": 1.0 - torch.eye(3),
    "src_C": torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    "src_Pt": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
    "src_Pv": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
}

MISMATCH_PERM_SEED = 20260904  # fixed permutation seed (plan §16/§26)

# Shared parameter groups for drift / gradient diagnostics (plan §4.1/§4.2).
SHARED_GROUPS = ("factorizer", "source_transforms", "fusion", "classifier")


@dataclass
class FrozenSetup:
    dataset: str
    seed: int
    variant: str  # "B0" | "F" | "S" | "J"
    cfg: object
    data: object
    model: object
    head: nn.Module
    device: torch.device


def resolve_cfg(dataset: str, seed: int, variant: str) -> object:
    from src.analysis.perf_r2_utils import VARIANT_YAMLS

    overrides = [
        f"dataset={dataset}", "task=nc", f"model={VARIANT_YAMLS[variant]}", f"seed={int(seed)}",
    ]
    with initialize(config_path="../../configs", version_base=None):
        return compose(config_name="config", overrides=overrides)


def load_frozen_r2_checkpoint(
    dataset: str, seed: int, variant: str, device: torch.device, root: Path | None = None
) -> FrozenSetup:
    """Load an R2 checkpoint (model + head + data). NEVER reads test labels.

    root = per-variant base directory (default outputs/perf_r2d1/<variant_root>);
    the checkpoint lives at root/<dataset>/<variant>/seed_<seed>/model.pt."""
    from src.data import load_mag_data
    from src.models.biaxis_r2 import Model
    from src.analysis.perf_r2_utils import VARIANT_ROOTS

    base = Path(root) if root is not None else R2D1_ROOT / VARIANT_ROOTS[variant]
    cfg = resolve_cfg(dataset, seed, variant)
    data = load_mag_data(cfg, "nc", int(seed))
    ckpt_path = base / dataset / variant / f"seed_{seed}" / "model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Model(cfg, ckpt["data_info"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    return FrozenSetup(dataset, seed, variant, cfg, data, model, head, device)


def assert_no_test_access(data: object) -> None:
    assert data.train_idx is not None and data.val_idx is not None


# ---------------------------------------------------------------------------
# Counterfactual machinery (plan §7/§8) — trained weights untouched
# ---------------------------------------------------------------------------


def factorize(model: object, x: torch.Tensor) -> dict[str, torch.Tensor]:
    x_t, x_v = model._split_modalities(x)
    return model.factorizer(x_t, x_v)


def ownership_states_cf(
    model: object, factors: dict[str, torch.Tensor], sem_cf: str | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Semantic counterfactual ownership states.

    sem_cf None -> the model's own enabled state (bitwise == forward).
    Returns (f0, f_star) = [N, 3, d], factor order [C, Pt, Pv].
    """
    if sem_cf is None:
        f0, f_star, _w = model._ownership_states(factors)
        return f0, f_star
    assert sem_cf in SEM_CFS, f"unknown sem_cf {sem_cf!r}"
    p_t, p_v = factors["p_t"], factors["p_v"]
    if sem_cf in ("common_only", "full"):
        c0, _w = model.adaptive_common(factors["c_t"], factors["c_v"])
    else:  # fixed_common_residual / both_off: w forced to exactly .5/.5
        c0 = 0.5 * (factors["c_t"] + factors["c_v"])
    f0 = torch.stack([c0, p_t, p_v], dim=1)
    if sem_cf in ("full", "fixed_common_residual"):
        delta = model.semantic_residual(f0)
        return f0, f0 + delta
    return f0, f0  # common_only / both_off: residual OFF


def functional_message_masked(
    model: object,
    f_star: torch.Tensor,
    n_block: torch.Tensor,
    v_block: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked 3x3 functional message, same math as
    Model._functional_message (bitwise equal for an all-ones mask, tested).

    mask[a, b] = 1 keeps cell a->b, 0 zeroes it. The 1/3 mean scaling is
    kept fixed (zeroed cells are simply removed, not renormalized).
    """
    num_nodes = int(f_star.size(0))
    src_t = model.src_type_emb.weight
    tgt_t = model.tgt_type_emb.weight
    mask = mask.to(f_star.device)
    msgs_per_target: list[torch.Tensor] = []
    for b in range(3):
        tgt_emb = tgt_t[b].unsqueeze(0).expand(num_nodes, -1)
        acc: torch.Tensor | None = None
        for a in range(3):
            u = torch.cat(
                [
                    f_star[:, b],
                    n_block[:, a],
                    f_star[:, b] * n_block[:, a],
                    (f_star[:, b] - n_block[:, a]).abs(),
                    src_t[a].unsqueeze(0).expand(num_nodes, -1),
                    tgt_emb,
                ],
                dim=-1,
            )
            g = torch.sigmoid(model.func_scorer(u))
            m = g * v_block[:, a]
            m = m * mask[a, b]
            acc = m if acc is None else acc + m
        msgs_per_target.append(model.msg_norm_func[b](acc / 3.0))
    return torch.stack(msgs_per_target, dim=1)


def forward_cf(
    model: object,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    sem_cf: str | None = None,
    func_cf: str | None = None,
) -> tuple[torch.Tensor, dict]:
    """Counterfactual full forward: z = fusion(F'), trained weights untouched.

    (sem_cf, func_cf) None -> the model's own state; the result is bitwise
    equal to model(x, edge_index) (tested). Returns (z, internals) where
    internals = {f0, f_star, n_block, v_block, base_msg, func_msg, f_out}.
    """
    factors = factorize(model, x)
    f0, f_star = ownership_states_cf(model, factors, sem_cf)
    num_nodes = int(x.size(0))
    d = model.factor_dim
    f_cat = f_star.reshape(num_nodes, 3 * d)
    n_cat = neighbor_mean(edge_index, f_cat, num_nodes, edge_chunk_size=model.edge_chunk_size)
    n_block = n_cat.reshape(num_nodes, 3, d)
    v_block = torch.stack(
        [model.source_transforms[a](n_block[:, a]) for a in range(3)], dim=1
    )
    base_msg = torch.stack(
        [model.msg_norm_base[b](v_block[:, b]) for b in range(3)], dim=1
    )
    func_msg = None
    if model.functional_enabled and func_cf != "func_off":
        mask = _FUNC_CELLS[func_cf if func_cf is not None else "full"]
        func_msg = functional_message_masked(model, f_star, n_block, v_block, mask)
    rho_base = torch.sigmoid(model.raw_rho_base)
    f_out = f_star + rho_base.view(1, 3, 1) * base_msg
    if func_msg is not None:
        f_out = f_out + model.rho_func.view(1, 3, 1) * func_msg
    z = model.fusion(torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1))
    internals = {
        "f0": f0, "f_star": f_star, "n_block": n_block, "v_block": v_block,
        "base_msg": base_msg, "func_msg": func_msg, "f_out": f_out,
    }
    return z, internals


# ---------------------------------------------------------------------------
# State extraction (plan §22) — bitwise aligned with the model forward
# ---------------------------------------------------------------------------


def extract_b0_states(
    model: object, x: torch.Tensor, edge_index: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Pre-graph factors F*, 1-hop contexts N^a, B0 graph-updated factors
    F_B0_out and z_final — the exact tensors the model forward consumes."""
    factors = factorize(model, x)
    _f0, f_star = ownership_states_cf(model, factors, None)
    num_nodes = int(x.size(0))
    f_out, n_block, base_msg, func_msg = model._graph_update(f_star, edge_index, num_nodes)
    z = model.fusion(torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1))
    return {"f_pre": f_star, "n": n_block, "f_out": f_out, "z": z, "base_msg": base_msg}


def propagation_signals(
    model: object, f_pre: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """H1 = P H0, H2 = P H1, HP = H0 - H1 per factor (plan §12).

    P = incoming-neighbor mean (same as neighbor_mean / B0's aggregation).
    Returns (H1, H2, HP), each [N, 3, d]."""
    d = model.factor_dim
    h0_cat = f_pre.reshape(num_nodes, 3 * d)
    h1 = neighbor_mean(edge_index, h0_cat, num_nodes, edge_chunk_size=model.edge_chunk_size)
    h2 = neighbor_mean(edge_index, h1, num_nodes, edge_chunk_size=model.edge_chunk_size)
    h1 = h1.reshape(num_nodes, 3, d)
    h2 = h2.reshape(num_nodes, 3, d)
    hp = f_pre - h1
    return h1, h2, hp


# ---------------------------------------------------------------------------
# Deterministic node permutation (plan §16/§26)
# ---------------------------------------------------------------------------


def fixed_node_permutation(num_nodes: int, seed: int = MISMATCH_PERM_SEED) -> torch.Tensor:
    """Deterministic node permutation for the mismatch negative control."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_nodes, generator=generator)


# ---------------------------------------------------------------------------
# Val metrics (plan §4.6)
# ---------------------------------------------------------------------------


def val_metrics_with_head(head: nn.Module, z: torch.Tensor, data: object, device: torch.device) -> dict:
    """Val Acc / Macro-F1 / per-class F1 / confusion matrix. Never test."""
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    head.eval()
    val_idx = data.val_idx.to(device)
    with torch.no_grad():
        pred = head(z[val_idx]).argmax(dim=-1).cpu().numpy()
    y_val = data.y[data.val_idx].numpy()
    classes = list(range(int(data.num_classes)))
    return {
        "val_acc": float(accuracy_score(y_val, pred)),
        "val_macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        "per_class_f1": [
            float(v) for v in f1_score(y_val, pred, average=None, labels=classes, zero_division=0)
        ],
        "confusion": [[int(v) for v in row] for row in confusion_matrix(y_val, pred, labels=classes)],
    }


# ---------------------------------------------------------------------------
# Representation comparisons (plan §4.3)
# ---------------------------------------------------------------------------


def mean_cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    x_n = torch.nn.functional.normalize(x, dim=-1)
    y_n = torch.nn.functional.normalize(y, dim=-1)
    return float((x_n * y_n).sum(dim=-1).mean().item())


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Centered linear CKA: ||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F)."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xx = x.t() @ x
    yy = y.t() @ y
    xy = x.t() @ y
    num = float(xy.norm().item()) ** 2
    den = float(xx.norm().item()) * float(yy.norm().item())
    return num / den if den > 0 else 0.0


def mean_relative_l2(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    per_node = (x - y).norm(dim=-1) / (x.norm(dim=-1) + eps)
    return float(per_node.mean().item())


def representation_drift(
    z_ref: torch.Tensor, z_var: torch.Tensor, val_idx: torch.Tensor
) -> dict[str, float]:
    """B0 z_final vs variant z on VAL nodes only (plan §4.3)."""
    z_ref_v = z_ref[val_idx]
    z_var_v = z_var[val_idx]
    return {
        "mean_cosine": mean_cosine(z_ref_v, z_var_v),
        "linear_cka": linear_cka(z_ref_v, z_var_v),
        "mean_relative_l2": mean_relative_l2(z_ref_v, z_var_v),
    }


# ---------------------------------------------------------------------------
# Parameter drift (plan §4.2)
# ---------------------------------------------------------------------------


def param_drift(model_ref: object, model_var: object, head_ref: nn.Module, head_var: nn.Module) -> dict[str, float]:
    """D(theta) = ||theta_var - theta_ref|| / (||theta_ref|| + eps) per shared
    group: factorizer / source_transforms / fusion / classifier."""
    drift: dict[str, float] = {}
    ref_states = {
        "factorizer": model_ref.factorizer.state_dict(),
        "source_transforms": {f"source_transforms.{i}.weight": m.weight for i, m in enumerate(model_ref.source_transforms)},
        "fusion": model_ref.fusion.state_dict(),
        "classifier": head_ref.state_dict(),
    }
    var_states = {
        "factorizer": model_var.factorizer.state_dict(),
        "source_transforms": {f"source_transforms.{i}.weight": m.weight for i, m in enumerate(model_var.source_transforms)},
        "fusion": model_var.fusion.state_dict(),
        "classifier": head_var.state_dict(),
    }
    for group in SHARED_GROUPS:
        ref_norm_sq = 0.0
        diff_sq = 0.0
        for key in ref_states[group]:
            if key not in var_states[group]:
                continue
            ref = ref_states[group][key].float()
            var = var_states[group][key].float()
            ref_norm_sq += float(ref.square().sum().item())
            diff_sq += float((var - ref).square().sum().item())
        drift[group] = (diff_sq ** 0.5) / (ref_norm_sq ** 0.5 + 1e-8)
    return drift


# ---------------------------------------------------------------------------
# CE-only gradient diagnostics (plan §4.1/§9)
# ---------------------------------------------------------------------------


def _group_grad_norms(
    model: object, head: nn.Module
) -> dict[str, float]:
    """Per-group L2 norm of the current .grad tensors (factorizer /
    source_transforms / fusion / classifier)."""
    norms: dict[str, float] = {}
    groups = {
        "factorizer": list(model.factorizer.parameters()),
        "source_transforms": list(model.source_transforms.parameters()),
        "fusion": list(model.fusion.parameters()),
        "classifier": list(head.parameters()),
    }
    for group, params in groups.items():
        sq = 0.0
        for p in params:
            if p.grad is not None:
                sq += float(p.grad.square().sum().item())
        norms[group] = sq ** 0.5
    return norms


def _grad_dict(
    model: object, head: nn.Module
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    prefixes = {
        "factorizer": "factorizer.",
        "source_transforms": "source_transforms.",
        "fusion": "fusion.",
    }
    for name, p in model.named_parameters():
        for group, prefix in prefixes.items():
            if name.startswith(prefix) and p.grad is not None:
                out[f"{group}.{name[len(prefix):]}"] = p.grad.detach().clone().float()
                break
    for name, p in head.named_parameters():
        if p.grad is not None:
            out[f"classifier.{name}"] = p.grad.detach().clone().float()
    return out


def ce_only_gradient_pair(
    model: object,
    head: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    data: object,
    device: torch.device,
    cf_a: tuple[str | None, str | None],
    cf_b: tuple[str | None, str | None],
) -> dict[str, dict[str, float]]:
    """CE-only gradient diagnostics between two counterfactual forwards
    (plan §4.1). Both forwards run in EVAL mode with grad enabled
    (dropout off, aux not computed) => deterministic.

    cf_a = (sem_cf, func_cf) full variant; cf_b = branch-off variant.
    Returns per-group {norm_off, norm_delta, delta_off_ratio, cos_off_delta}
    where off = grad of cf_b (branch off), delta = grad_a - grad_b.
    """
    model.eval()
    train_idx = data.train_idx.to(device)
    y_train = data.y[data.train_idx].to(device)
    grads: dict[str, dict[str, torch.Tensor]] = {}
    for tag, cf in (("full", cf_a), ("off", cf_b)):
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        with torch.enable_grad():
            z, _ = forward_cf(model, x, edge_index, sem_cf=cf[0], func_cf=cf[1])
            loss = torch.nn.functional.cross_entropy(head(z[train_idx]), y_train)
            loss.backward()
        grads[tag] = _grad_dict(model, head)
        del z, loss
        torch.cuda.empty_cache()
    out: dict[str, dict[str, float]] = {}
    for group in SHARED_GROUPS:
        g_off = {k: v for k, v in grads["off"].items() if k.startswith(group + ".")}
        g_full = {k: v for k, v in grads["full"].items() if k.startswith(group + ".")}
        if not g_off:
            continue
        norm_off = sum(float(v.square().sum().item()) for v in g_off.values()) ** 0.5
        delta = {
            k: g_full[k] - g_off[k] for k in g_off if k in g_full
        }
        norm_delta = sum(float(v.square().sum().item()) for v in delta.values()) ** 0.5
        cos = 0.0
        if norm_off > 0 and norm_delta > 0:
            dot = sum(float((g_off[k] * delta[k]).sum().item()) for k in delta)
            cos = dot / (norm_off * norm_delta)
        out[group] = {
            "norm_off": norm_off,
            "norm_delta": norm_delta,
            "delta_off_ratio": norm_delta / (norm_off + 1e-12),
            "cos_off_delta": cos,
        }
    return out
