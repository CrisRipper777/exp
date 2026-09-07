"""R2-Design-2.5 shared analysis layer
(docs/BiAxis_R2_Design_2_5_Structured_Capacity_Utilization_Audit.md).

Discipline:
    - Frozen-B0 read-only: the fixed-alpha pipeline and the transmission
      stages replicate the B0 forward math with the model's OWN trained
      modules; trained weights are never modified. (The pipeline's linear
      precompute V(H1)/V(H2) -> v(alpha) = v1 + alpha*(v2 - v1) is
      numerically equivalent to the M1 forward up to fp noise; unit-tested.)
    - Val only, NEVER test: all helpers assert train/val presence and the
      trainers never index test labels.
    - No labels inside the capacity model; label access lives in the
      trainers here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.biaxis_p1_components import neighbor_mean  # noqa: E402

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
TARGET_DATASETS = ["Movies", "Toys", "Grocery"]
GUARD_DATASETS = ["ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
FACTOR_NAMES = ("C", "Pt", "Pv")
ALPHA_PT_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA_GRAD_VALUES = (0.0, 0.25, 0.5)
CLASSIFIER_SEED = 20260904

R2D25_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d25"
B0_CONFIRM_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d15" / "b0_confirm"

CAPACITY_MODES = (
    "early_mix", "sep_sum", "sep_concat", "inception_012",
    "cap_h1_dup", "wide_b0", "deep_fusion",
    "hop_attention", "h1_attention",
)
# Hop branches per mode (mirrors EXPERT_KEYS in the model).
MODE_EXPERTS = {
    "early_mix": (),
    "sep_sum": ("e1", "e2"),
    "sep_concat": ("e1", "e2"),
    "inception_012": ("e0", "e1", "e2"),
    "cap_h1_dup": ("e1a", "e1b"),
    "wide_b0": (),
    "deep_fusion": (),
    "hop_attention": ("e0", "e1", "e2"),
    "h1_attention": ("e1a", "e1b", "e1c"),
}

# Warmup10+cosine (plan D2.5-C unified schedule).
WARMUP_EPOCHS = 10
MIN_LR = 1e-5


# ---------------------------------------------------------------------------
# Frozen B0 setup
# ---------------------------------------------------------------------------


def load_r2d25_b0_setup(dataset: str, seed: int, device: torch.device):
    """Frozen B0 best checkpoint (outputs/perf_r2d15/b0_confirm). NEVER test."""
    from src.analysis.perf_r2d15_utils import load_frozen_r2_checkpoint

    return load_frozen_r2_checkpoint(dataset, seed, "B0", device, root=B0_CONFIRM_ROOT)


def resolve_capacity_cfg(dataset: str, seed: int, mode: str):
    from hydra import compose, initialize_config_dir

    overrides = [
        f"dataset={dataset}", "task=nc", "model=biaxis_r2_capacity",
        f"model.capacity_mode={mode}", f"seed={int(seed)}",
    ]
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        return compose(config_name="config", overrides=overrides)


def assert_no_test_access(data) -> None:
    assert data.train_idx is not None and data.val_idx is not None


def load_mag_data_wrap(cfg, seed: int):
    """NC data with the no-test guard (Val-only protocol)."""
    from src.data import load_mag_data

    data = load_mag_data(cfg, "nc", int(seed))
    assert_no_test_access(data)
    return data


# ---------------------------------------------------------------------------
# Deterministic classifier init (bitwise replay, plan Prompt 1 tests)
# ---------------------------------------------------------------------------


def save_state(path: Path, module: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)


def load_state_into(path: Path, module: nn.Module) -> None:
    module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def make_classifier_init(seed: int, out_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    """Deterministic fresh classifier init (same RNG for every consumer).

    Uses the analytic default-uniform std 1/sqrt(3*fan_in) instead of
    reading head.weight.std() — the default Linear init draws from the
    GLOBAL RNG, which would leak call-order dependence into the scale."""
    import math

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        head = nn.Linear(out_dim, num_classes).to(device)
        std = 1.0 / math.sqrt(3.0 * float(out_dim))
        head.weight.normal_(0.0, std, generator=generator)
    head.bias.data.zero_()
    return head


def load_or_make_head_init(
    init_path: Path, out_dim: int, num_classes: int, device: torch.device
) -> nn.Module:
    """Per-(dataset, seed) shared classifier init file."""
    if not init_path.exists():
        torch.manual_seed(CLASSIFIER_SEED)
        init_head = nn.Linear(out_dim, num_classes).to(device)
        save_state(init_path, init_head)
    head = nn.Linear(out_dim, num_classes).to(device)
    load_state_into(init_path, head)
    return head


# ---------------------------------------------------------------------------
# Ridge probe machinery (plan D2.5-B: StandardScaler + Ridge(alpha=1.0))
# ---------------------------------------------------------------------------


def ridge_probe(feat_train: torch.Tensor, y_train: torch.Tensor, feat_val: torch.Tensor, y_val: torch.Tensor) -> dict:
    """TRAIN-fit / VAL-eval RidgeClassifier(alpha=1.0) with StandardScaler.
    Returns {acc, macro_f1}. CPU tensors / numpy arrays accepted."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifier

    x_tr = feat_train.detach().cpu().numpy() if torch.is_tensor(feat_train) else feat_train
    x_va = feat_val.detach().cpu().numpy() if torch.is_tensor(feat_val) else feat_val
    y_tr = y_train.detach().cpu().numpy() if torch.is_tensor(y_train) else y_train
    y_va = y_val.detach().cpu().numpy() if torch.is_tensor(y_val) else y_val
    pipe = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    pipe.fit(x_tr, y_tr)
    pred = pipe.predict(x_va)
    return {
        "acc": float(accuracy_score(y_va, pred)),
        "macro_f1": float(f1_score(y_va, pred, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Representation comparison metrics (reused from perf_r2d15_utils)
# ---------------------------------------------------------------------------


def _import_r2d15():
    from src.analysis.perf_r2d15_utils import linear_cka, mean_cosine, mean_relative_l2

    return linear_cka, mean_cosine, mean_relative_l2


def effective_rank(x: torch.Tensor) -> float:
    """Shannon-entropy effective rank of the centered matrix (CPU float64)."""
    x64 = (x.detach().cpu() - x.detach().cpu().mean(dim=0, keepdim=True)).double()
    s = torch.linalg.svdvals(x64)
    s = s / (s.sum() + 1e-12)
    return float(torch.exp(-(s * torch.log(s + 1e-12)).sum()).item())


# ---------------------------------------------------------------------------
# Fixed-alpha pipeline (plan D2.5-A): frozen B0, linear V precompute
# ---------------------------------------------------------------------------


@dataclass
class FixedAlphaPipeline:
    """Frozen B0 + fixed alpha_Pt (alpha_C = alpha_Pv = 0).

    Precomputes V(H1)/V(H2) once and evaluates the M1 representation as
    v_f(alpha) = v1_f + alpha_f*(v2_f - v1_f) — exact up to fp noise
    (V is linear), then LN -> rho residual -> B0 fusion."""

    setup: object
    f_star: torch.Tensor
    v1: torch.Tensor
    v2: torch.Tensor
    rho: torch.Tensor

    def z_at(self, alpha_pt: float) -> torch.Tensor:
        model = self.setup.model
        v = self.v1.clone()
        v[:, 1] = self.v1[:, 1] + float(alpha_pt) * (self.v2[:, 1] - self.v1[:, 1])
        msg = torch.stack([model.msg_norm_base[b](v[:, b]) for b in range(3)], dim=1)
        f_out = self.f_star + self.rho.view(1, 3, 1) * msg
        return model.fusion(torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1))

    def z_at_tensor(self, alpha_pt: torch.Tensor) -> torch.Tensor:
        """Differentiable z(alpha) for the CE-gradient diagnostics."""
        model = self.setup.model
        v = self.v1
        v_pt = v[:, 1] + alpha_pt.unsqueeze(0) * (self.v2[:, 1] - self.v1[:, 1])
        msgs = [
            model.msg_norm_base[0](v[:, 0]),
            model.msg_norm_base[1](v_pt),
            model.msg_norm_base[2](v[:, 2]),
        ]
        msg = torch.stack(msgs, dim=1)
        f_out = self.f_star + self.rho.view(1, 3, 1) * msg
        return model.fusion(torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1))


def build_fixed_alpha_pipeline(setup, x: torch.Tensor, edge_index: torch.Tensor) -> FixedAlphaPipeline:
    """Precompute f_star / V(H1) / V(H2) / rho on the frozen B0 checkpoint."""
    from src.analysis.perf_r2d15_utils import extract_b0_states, propagation_signals

    model = setup.model
    with torch.no_grad():
        states = extract_b0_states(model, x, edge_index)
        f_star = states["f_pre"]
        h1, h2, _ = propagation_signals(model, f_star, edge_index, int(x.size(0)))
        v1 = torch.stack([model.source_transforms[a](h1[:, a]) for a in range(3)], dim=1)
        v2 = torch.stack([model.source_transforms[a](h2[:, a]) for a in range(3)], dim=1)
        rho = torch.sigmoid(model.raw_rho_base)
    return FixedAlphaPipeline(setup, f_star, v1, v2, rho)


def alpha_ce_gradients(
    pipeline: FixedAlphaPipeline,
    head: nn.Module,
    data,
    alpha_pt: float,
) -> dict[str, float]:
    """Diagnostic-only dTrainCE/dalpha_Pt and dValCE/dalpha_Pt with the
    trained head FIXED (never used to update parameters)."""
    device = pipeline.f_star.device
    alpha = torch.tensor(float(alpha_pt), dtype=torch.float32, device=device, requires_grad=True)
    head.eval()
    train_idx = data.train_idx.to(device)
    y_train = data.y[data.train_idx].to(device)
    val_idx = data.val_idx.to(device)
    y_val = data.y[data.val_idx].to(device)
    out: dict[str, float] = {}
    with torch.enable_grad():
        z = pipeline.z_at_tensor(alpha)
        loss_train = torch.nn.functional.cross_entropy(head(z[train_idx]), y_train)
        g_train = torch.autograd.grad(loss_train, alpha, retain_graph=True)[0]
        loss_val = torch.nn.functional.cross_entropy(head(z[val_idx]), y_val)
        g_val = torch.autograd.grad(loss_val, alpha)[0]
        out["d_train_ce"] = float(g_train.item())
        out["d_val_ce"] = float(g_val.item())
    return out


# ---------------------------------------------------------------------------
# Frozen-B0 head training (plan D2.5-A / D2.5-B S4 retrained head)
# ---------------------------------------------------------------------------


def train_head_on_frozen_z(
    z: torch.Tensor,
    head: nn.Module,
    data,
    device: torch.device,
    epochs: int = 300,
    patience: int = 30,
    lr: float = 1e-3,
    wd: float = 1e-4,
) -> dict:
    """Train ONLY the classifier on a FROZEN representation z. Best Val Acc.
    Returns {best_val_acc, best_val_f1, best_epoch, best_train_ce,
    best_train_acc, best_val_ce, stop_epoch}."""
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss()
    train_idx = data.train_idx.to(device)
    y_train = data.y[data.train_idx].to(device)
    val_idx = data.val_idx.to(device)
    y_val = data.y[data.val_idx].to(device)
    z_tr = z[train_idx]
    z_va = z[val_idx]
    best_acc, best_epoch, best_state = -1.0, None, None
    best_train_ce = best_train_acc = best_val_ce = best_f1 = None
    patience_left = patience
    stop_epoch = epochs
    for epoch in range(1, epochs + 1):
        head.train()
        opt.zero_grad(set_to_none=True)
        logits = head(z_tr)
        loss = criterion(logits, y_train)
        loss.backward()
        opt.step()
        with torch.no_grad():
            train_ce = float(loss.item())
            train_acc = float((logits.argmax(-1) == y_train).float().mean().item())
        head.eval()
        with torch.no_grad():
            logits_v = head(z_va)
            val_ce = float(criterion(logits_v, y_val).item())
            acc = float((logits_v.argmax(-1) == y_val).float().mean().item())
        if acc > best_acc:
            from sklearn.metrics import f1_score

            best_acc, best_epoch = acc, epoch
            best_f1 = float(f1_score(y_val.cpu().numpy(), logits_v.argmax(-1).cpu().numpy(),
                                     average="macro", zero_division=0))
            best_train_ce, best_train_acc, best_val_ce = train_ce, train_acc, val_ce
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                stop_epoch = epoch
                break
    head.load_state_dict(best_state)
    return {
        "best_val_acc": best_acc, "best_val_macro_f1": best_f1, "best_epoch": best_epoch,
        "best_train_ce": best_train_ce, "best_train_acc": best_train_acc,
        "best_val_ce": best_val_ce, "stop_epoch": stop_epoch,
    }


# ---------------------------------------------------------------------------
# D2.5-B transmission stages (plan: S0 raw / S1 V / S2 LN / S3 residual / S4 fusion)
# ---------------------------------------------------------------------------


def pt_transmission_features(setup, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
    """Pt-factor H1/H2 trace through the actual frozen-B0 computation."""
    from src.analysis.perf_r2d15_utils import extract_b0_states, propagation_signals

    model = setup.model
    with torch.no_grad():
        states = extract_b0_states(model, x, edge_index)
        f_star = states["f_pre"]
        pt = f_star[:, 1]
        h1, h2, _ = propagation_signals(model, f_star, edge_index, int(x.size(0)))
        v1 = torch.stack([model.source_transforms[a](h1[:, a]) for a in range(3)], dim=1)
        v2 = torch.stack([model.source_transforms[a](h2[:, a]) for a in range(3)], dim=1)
        ln1 = torch.stack([model.msg_norm_base[b](v1[:, b]) for b in range(3)], dim=1)
        ln2 = torch.stack([model.msg_norm_base[b](v2[:, b]) for b in range(3)], dim=1)
        rho = torch.sigmoid(model.raw_rho_base)
        # S4 counterfactual: only the Pt graph context is replaced by H2.
        base_msg_cf = ln1.clone()
        base_msg_cf[:, 1] = ln2[:, 1]
        f_out_cf = f_star + rho.view(1, 3, 1) * base_msg_cf
        z_cf = model.fusion(torch.cat([f_out_cf[:, 0], f_out_cf[:, 1], f_out_cf[:, 2]], dim=-1))
        z = states["z"]
    return {
        "pt": pt,
        "s0_h1": torch.cat([pt, h1[:, 1]], dim=-1),
        "s0_h2": torch.cat([pt, h2[:, 1]], dim=-1),
        "s1_h1": torch.cat([pt, v1[:, 1]], dim=-1),
        "s1_h2": torch.cat([pt, v2[:, 1]], dim=-1),
        "s2_h1": torch.cat([pt, ln1[:, 1]], dim=-1),
        "s2_h2": torch.cat([pt, ln2[:, 1]], dim=-1),
        "s3_h1": pt + rho[1] * ln1[:, 1],
        "s3_h2": pt + rho[1] * ln2[:, 1],
        "z": z,
        "z_cf": z_cf,
    }


# ---------------------------------------------------------------------------
# LR schedule (plan D2.5-C: warmup10 + cosine, unified across variants)
# ---------------------------------------------------------------------------


def scheduled_lr(epoch: int, total_epochs: int, base_lr: float,
                 warmup: int = WARMUP_EPOCHS, min_lr: float = MIN_LR) -> float:
    import math

    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    frac = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * frac


# ---------------------------------------------------------------------------
# Parameter grouping (plan D2.5-C unified schedule / D2.5-D expert LR)
# ---------------------------------------------------------------------------

P0_PREFIXES = ("factorizer.", "recon_text_head.", "recon_visual_head.")


def is_p0_param(name: str) -> bool:
    return name.startswith(P0_PREFIXES)


def group_parameters(model: nn.Module, expert_lr_group: str | None = None) -> dict[str, dict]:
    """{group_name: {"params": [...], "base_lr": ..., "wd": ...}}.
    expert_lr_group, e.g. "hop_experts.e2", splits that prefix out."""
    p0, graph, expert, other = [], [], [], []
    for name, p in model.named_parameters():
        if is_p0_param(name):
            p0.append(p)
        elif expert_lr_group and name.startswith(expert_lr_group + "."):
            expert.append(p)
        else:
            graph.append(p)
    groups = {
        "graph": {"params": graph, "base_lr": 1e-3, "wd": 1e-4},
        "p0": {"params": p0, "base_lr": 1e-4, "wd": 1e-4},
    }
    if expert:
        groups["expert_lr"] = {"params": expert, "base_lr": 1e-3, "wd": 1e-4}
    return {k: v for k, v in groups.items() if v["params"]}


def group_grad_norms(model: nn.Module, head: nn.Module) -> dict[str, float]:
    """L2 grad norms per group (P0 / graph / fusion / classifier)."""
    norms = {"p0": 0.0, "graph": 0.0, "fusion": 0.0, "classifier": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.square().sum().item())
        if is_p0_param(name):
            norms["p0"] += g
        elif name.startswith("fusion."):
            norms["fusion"] += g
        else:
            norms["graph"] += g
    for p in head.parameters():
        if p.grad is not None:
            norms["classifier"] += float(p.grad.square().sum().item())
    return {k: v ** 0.5 for k, v in norms.items()}


def group_update_ratios(model: nn.Module, head: nn.Module, group_lrs: dict[str, float]) -> dict[str, float]:
    """Mean over params of lr_g * ||grad|| / ||param|| per group."""
    acc: dict[str, list[float]] = {"p0": [], "graph": [], "classifier": []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.norm().item()
        w = p.detach().norm().item()
        if w <= 0:
            continue
        key = "p0" if is_p0_param(name) else "graph"
        acc[key].append(group_lrs.get("p0" if key == "p0" else "graph", 1e-3) * g / w)
    for p in head.parameters():
        if p.grad is None:
            continue
        w = p.detach().norm().item()
        if w > 0:
            acc["classifier"].append(group_lrs.get("classifier", 1e-3) * p.grad.norm().item() / w)
    return {k: float(sum(v) / len(v)) if v else 0.0 for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Capacity training loop (plan D2.5-C unified schedule; D2.5-D hooks)
# ---------------------------------------------------------------------------


def train_capacity_variant(
    cfg,
    data,
    model: nn.Module,
    head: nn.Module,
    device: torch.device,
    *,
    total_epochs: int = 300,
    patience: int = 30,
    freeze_p0_epochs: int = 20,
    expert_lr_group: str | None = None,
    expert_lr: float = 1e-3,
    deep_sup_lambda: float = 0.0,
    path_dropout_p: float = 0.0,
    sample_grad_epochs: tuple[int, ...] = (1, 10, 20, 21, 30, 60, 120),
    history_callback=None,
) -> dict:
    """Unified D2.5-C schedule: epochs 1-20 P0 frozen, 21+ P0 lr 1e-4;
    graph/fusion/classifier lr 1e-3; AdamW wd 1e-4; warmup10+cosine;
    patience30 best Val Acc. Val only.

    Single-factor D2.5-D hooks (never stacked):
        expert_lr_group/expert_lr : expert-specific LR
        deep_sup_lambda           : expert aux heads (model must have them)
        path_dropout_p            : H1-branch per-node dropout during training
    """
    assert_no_test_access(data)
    model = model.to(device)
    head = head.to(device)
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    train_idx = data.train_idx.to(device)
    y_train = data.y[data.train_idx].to(device)
    val_idx = data.val_idx.to(device)
    y_val = data.y[data.val_idx].to(device)
    criterion = nn.CrossEntropyLoss()

    # requires_grad initial state: P0 frozen.
    for n, p in model.named_parameters():
        p.requires_grad_(not is_p0_param(n))

    groups = group_parameters(model, expert_lr_group=expert_lr_group)
    if expert_lr_group and "expert_lr" in groups:
        groups["expert_lr"]["base_lr"] = expert_lr
    # P0 is FROZEN for epochs 1..freeze_p0_epochs: keep it out of the
    # optimizer entirely (AdamW rejects parameters appearing twice; the
    # group is added at the unfreeze epoch).
    groups.pop("p0", None)
    opt_params = []
    group_base = {}
    for gname, g in groups.items():
        opt_params.append({"params": g["params"], "lr": g["base_lr"], "weight_decay": g["wd"]})
        group_base[gname] = g["base_lr"]
    opt_params.append({"params": head.parameters(), "lr": 1e-3, "weight_decay": 1e-4})
    group_base["classifier"] = 1e-3
    opt = torch.optim.AdamW(opt_params)
    opt_group_names = list(group_base.keys())

    def _apply_lr(epoch: int) -> None:
        for i, pg in enumerate(opt.param_groups):
            pg["lr"] = scheduled_lr(epoch, total_epochs, group_base[opt_group_names[i]])

    history: list[dict] = []
    grad_samples: list[dict] = []
    best_acc, best_epoch, best_state = -1.0, None, None
    patience_left = patience
    stop_epoch = total_epochs
    p0_unfrozen = False
    for epoch in range(1, total_epochs + 1):
        if epoch == freeze_p0_epochs + 1:
            for n, p in model.named_parameters():
                if is_p0_param(n):
                    p.requires_grad_(True)
            opt.add_param_group({
                "params": [p for n, p in model.named_parameters() if is_p0_param(n)],
                "lr": scheduled_lr(epoch, total_epochs, 1e-4), "weight_decay": 1e-4,
            })
            group_base["p0_late"] = 1e-4
            opt_group_names.append("p0_late")
            p0_unfrozen = True
        _apply_lr(epoch)
        opt.zero_grad(set_to_none=True)
        model.train()
        if deep_sup_lambda > 0.0:
            z, experts = model.forward_with_experts(
                x, ei, path_dropout_h1=path_dropout_p)
            loss = criterion(head(z[train_idx]), y_train)
            loss = loss + deep_sup_lambda * model.deep_supervision_loss(
                experts, train_idx, y_train)
        else:
            z, _, _, _, _ = model(x, ei, path_dropout_h1=path_dropout_p)
            loss = criterion(head(z[train_idx]), y_train)
        loss.backward()
        if epoch in sample_grad_epochs:
            lr_map = {
                "p0": 1e-4 if p0_unfrozen else 0.0, "graph": 1e-3, "classifier": 1e-3,
            }
            grad_samples.append({
                "epoch": epoch,
                "grad_norm": group_grad_norms(model, head),
                "update_ratio": group_update_ratios(model, head, lr_map),
            })
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad]
            + list(head.parameters()), 1.0,
        )
        opt.step()
        with torch.no_grad():
            model.eval()
            z_eval, _, _, _, _ = model(x, ei)
            pred_v = head(z_eval[val_idx]).argmax(-1)
            acc = float((pred_v == y_val).float().mean().item())
        del z_eval
        train_ce = float(loss.item())
        row = {
            "epoch": epoch,
            "lr_graph": float(scheduled_lr(epoch, total_epochs, 1e-3)),
            "lr_p0": float(scheduled_lr(epoch, total_epochs, 1e-4)) if p0_unfrozen else 0.0,
            "train_ce": train_ce,
            "val_acc": acc,
            "p0_unfrozen": int(p0_unfrozen),
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
        "grad_samples": grad_samples,
        "p0_unfrozen": p0_unfrozen,
        "z_best": z_best,
    }


# ---------------------------------------------------------------------------
# Ablation evaluation (plan D2.5-C causal usage: FULL / H2-OFF / H1-OFF / H0-OFF)
# ---------------------------------------------------------------------------


def ablation_metrics(model: nn.Module, head: nn.Module, x, ei, data, device) -> dict:
    """Val Acc/F1 under per-branch ablations at the best checkpoint."""
    from src.analysis.perf_r2d15_utils import val_metrics_with_head

    model.eval()
    head.eval()
    out: dict = {}
    mode = model.capacity_mode
    plan: list[tuple[str, set[str]]] = [("full", set())]
    if mode == "early_mix":
        plan.append(("h2_off", {"e2"}))
    elif mode in ("sep_sum", "sep_concat"):
        plan += [("h2_off", {"e2"}), ("h1_off", {"e1"})]
    elif mode == "inception_012":
        plan += [("h2_off", {"e2"}), ("h1_off", {"e1"}), ("h0_off", {"e0"})]
    elif mode == "cap_h1_dup":
        plan += [("h2_off", {"e1b"}), ("h1_off", {"e1a"})]
    elif mode == "hop_attention":
        plan += [("h2_off", {"e2"}), ("h1_off", {"e1"}), ("h0_off", {"e0"})]
    elif mode == "h1_attention":
        plan += [("h2_off", {"e1c"}), ("h1_off", {"e1a"})]
    with torch.no_grad():
        for tag, off in plan:
            z, _, _, _, _ = model(x, ei, off_hops=off)
            m = val_metrics_with_head(head, z, data, device)
            out[tag] = {"val_acc": m["val_acc"], "val_macro_f1": m["val_macro_f1"]}
            del z
    return out
