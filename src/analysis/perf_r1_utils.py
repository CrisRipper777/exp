"""R1 performance-diagnostic shared layer (plan §33 / audit §2).

Same discipline as perf_r0_utils: frozen-model-read-only, no test access,
chunked where needed. One structural difference from R0: the relation
contexts g_perm come from the checkpoint model's OWN ``_graph_update``
(dynamic dispatch), so A0 checkpoints use the frozen aggregation and A1
checkpoints use the eta-weighted aggregation — the scripts never reimplement
the aggregation themselves.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from hydra import compose, initialize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.biaxis_p1_components import neighbor_mean  # noqa: E402

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
FACTOR_NAMES = ["C", "Pt", "Pv"]

# Plan §13: A0 = R1 same-code-path baseline, A1 = semantic_reliability.
# A1R1..A1R4 = review option B control (regularized A1 variants).
# A2 = semantic_relation_calibration (user-authorized amendment).
MODE_LABELS = {
    "A0": "baseline",
    "A1": "semantic_reliability",
    "A2": "semantic_relation_calibration",
    "BL": "baseline",
    "BR": "baseline",
    "BLR": "baseline",
    "C1SG": "detached_2hop",
}
MODE_ROOTS = {
    "A0": "baseline",
    "A1": "reliability",
    "A2": "relation_calibration",
    "BL": "routing",
    "BR": "routing",
    "BLR": "routing",
    "C1SG": "multihop",
}
# R1-B decoupled variants (user §5/§9): BL dynamic local only,
# BR support relation only, BLR both. Parent = A0 baseline.
EXTRA_OVERRIDES = {
    "BL": ["model.r1.router_mode=local_only"],
    "BR": ["model.r1.router_mode=relation_only"],
    "BLR": ["model.r1.router_mode=evidence"],
}
# Variant label -> extra hydra overrides on top of model.r1.mode=semantic_reliability.
REG_VARIANTS = {
    "A1R1": ["model.r1.reg_type=mean1", "model.r1.reg_weight=0.1"],
    "A1R2": ["model.r1.reg_type=mean1", "model.r1.reg_weight=1.0"],
    "A1R3": ["model.r1.reg_type=band", "model.r1.reg_weight=0.1"],
    "A1R4": ["model.r1.reg_type=band", "model.r1.reg_weight=1.0"],
}
for _v in REG_VARIANTS:
    MODE_ROOTS[_v] = "reliability"
CKPT_ROOT = PROJECT_ROOT / "outputs" / "perf_r1"


@dataclass
class R1Setup:
    dataset: str
    seed: int
    mode: str  # "A0" | "A1"
    cfg: object
    data: object
    model: object
    head: torch.nn.Module
    device: torch.device


def resolve_cfg(dataset: str, seed: int, mode: str) -> object:
    # NOTE: hydra resolves config_path relative to THIS file (src/analysis/).
    overrides = [
        f"dataset={dataset}", "task=nc", "model=biaxis_perf_r1",
        f"model.r1.mode={MODE_LABELS.get(mode, 'semantic_reliability')}",
        f"seed={int(seed)}",
    ]
    overrides += REG_VARIANTS.get(mode, [])
    overrides += EXTRA_OVERRIDES.get(mode, [])
    with initialize(config_path="../../configs", version_base=None):
        return compose(config_name="config", overrides=overrides)


def load_r1_setup(dataset: str, seed: int, mode: str, device: torch.device) -> R1Setup:
    """Load R1 checkpoint model + head + data. NEVER reads test labels."""
    from src.data import load_mag_data
    from src.models.biaxis_perf_r1 import Model

    cfg = resolve_cfg(dataset, seed, mode)
    data = load_mag_data(cfg, "nc", int(seed))
    ckpt = torch.load(
        CKPT_ROOT / MODE_ROOTS[mode] / dataset / mode / f"seed_{seed}" / "model.pt",
        map_location="cpu", weights_only=False,
    )
    model = Model(cfg, ckpt["data_info"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(ckpt["head_state"])
    return R1Setup(dataset, seed, mode, cfg, data, model, head, device)


def assert_no_test_access(data: object) -> None:
    """Guard: R1 scripts must only use train/val supervision."""
    assert data.train_idx is not None and data.val_idx is not None


@torch.no_grad()
def r1_pipeline(setup: R1Setup) -> dict:
    """One frozen full-graph forward returning every intermediate the probes
    and counterfactuals need.

    Returns (all on GPU except indices):
        factors, z_local, z_final [N, hidden]
        f_block [N, 3, d], graph_out (incl. g_perm / gamma / r / availability
        / effective_mass when present), s_rel / s_aug, beta / alpha,
        deg [N], edge_index, num_nodes
    """
    model, data, device = setup.model, setup.data, setup.device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    num_nodes = int(x.size(0))
    factors, z_local = model._encode(x)
    f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
    num_factors = int(f_block.size(1))
    graph_out = model._graph_update(f_block, edge_index, num_nodes)
    deg = torch.bincount(edge_index[1], minlength=num_nodes).to(torch.float32)
    # Rebuild the score assembly EXACTLY as the model's forward does
    # (BL/BR/BLR carry zero-init residual scorers that the A0-era
    # reconstruction must not drop).
    s_rel = model.transport_scorer(f_block, graph_out["g_perm"])
    if getattr(model, "_use_relation_residual", False):
        availability = graph_out["availability"]
        mass = availability * deg.unsqueeze(-1)  # structural m_ik [N, K]
        rel_res = model.relation_score_residual(torch.log1p(mass), availability)  # [N, K, 1]
        s_rel = s_rel + rel_res.permute(0, 2, 1).expand(num_nodes, num_factors, model.num_relations)
    s_aug = torch.cat(
        [model.null_score.reshape(1, -1, 1).expand(num_nodes, -1, 1), s_rel], dim=-1
    )
    if getattr(model, "_use_local_residual", False):
        f_cat = f_block.reshape(num_nodes, num_factors * model.factor_dim)
        g_bar = neighbor_mean(
            edge_index, f_cat, num_nodes, edge_chunk_size=model.edge_chunk_size
        ).reshape(num_nodes, num_factors, model.factor_dim)
        local = model.local_score_residual(f_block, g_bar)  # [N, F, 1]
        s_aug[..., 0] = s_aug[..., 0] + local.squeeze(-1)
    if getattr(model, "r1_mode", None) == "detached_2hop":
        # C1SG: the final embedding includes the trajectory readout — the
        # manual hop-1-only fusion is NOT the model's output.
        z_final = model.forward(x, edge_index)[0]
    else:
        z_final = model.fusion(
            torch.cat(
                [graph_out["f_tilde"][:, 0], graph_out["f_tilde"][:, 1], graph_out["f_tilde"][:, 2]],
                dim=-1,
            )
        )
    return {
        "factors": factors,
        "z_local": z_local,
        "z_final": z_final,
        "f_block": f_block,
        "graph_out": graph_out,
        "scores": {"s_rel": s_rel, "s_aug": s_aug},
        "beta": graph_out["beta"],
        "alpha": graph_out["alpha"],
        "deg": deg,
        "edge_index": edge_index,
        "num_nodes": num_nodes,
    }


@torch.no_grad()
def apply_plan(setup: R1Setup, internals: dict, gamma: torch.Tensor) -> torch.Tensor:
    """Frozen operator + fusion with an overridden Gamma -> z [N, hidden].
    Contexts / operator / fusion are the checkpoint's own (never retrained)."""
    model = setup.model
    f_block = internals["f_block"]
    g_perm = internals["graph_out"]["g_perm"]
    num_nodes, num_factors, factor_dim = f_block.shape
    m_f = model.operator(g_perm, gamma[..., 1:], model.graph_w0)
    f_tilde = model.graph_norm((f_block + m_f).reshape(num_nodes * num_factors, factor_dim))
    f_tilde = f_tilde.reshape(num_nodes, num_factors, factor_dim)
    return model.fusion(torch.cat([f_tilde[:, 0], f_tilde[:, 1], f_tilde[:, 2]], dim=-1))


@torch.no_grad()
def val_acc_with_head(setup: R1Setup, z: torch.Tensor) -> float:
    """Val acc of head(z) on the VAL split (train/val only; R0-identical).
    Val Macro-F1 for the main protocol comes from the trainer log only —
    the counterfactual layer keeps the R0 acc-only scope."""
    data, device = setup.data, setup.device
    y = data.y.to(device)
    pred = setup.head(z).argmax(dim=-1)
    val_idx = data.val_idx.to(device)
    return float((pred[val_idx] == y[val_idx]).float().mean().item())


# ---------------------------------------------------------------------------
# Gamma counterfactuals (R0 D5 definitions, plan §11: at least
# Current / Uniform / Availability)
# ---------------------------------------------------------------------------


def _cf1_uniform(gamma: torch.Tensor, beta: torch.Tensor, alpha: torch.Tensor, _avail) -> torch.Tensor:
    out = gamma.clone()
    k = int(alpha.size(-1))
    out[..., 1:] = (beta.unsqueeze(-1) / k).expand_as(alpha)
    return out


def _cf2_availability(gamma: torch.Tensor, beta: torch.Tensor, _alpha, availability) -> torch.Tensor:
    out = gamma.clone()
    out[..., 1:] = beta.unsqueeze(-1) * availability.unsqueeze(1)
    return out


def _cf4_no_local(gamma: torch.Tensor, _beta, alpha: torch.Tensor, _avail) -> torch.Tensor:
    """R0 D5 CF4: all graph mass, Local column removed, relation plan kept."""
    out = gamma.clone()
    out[..., 0] = 0.0
    out[..., 1:] = alpha
    return out


def _cf5_fixed_factor_mean_beta(gamma: torch.Tensor, beta: torch.Tensor, alpha: torch.Tensor, _avail) -> torch.Tensor:
    """R0 D5 CF5: per-factor graph demand replaced by its dataset mean."""
    out = gamma.clone()
    beta_bar = beta.mean(dim=0, keepdim=True)  # [1, F]
    out[..., 0] = 1.0 - beta_bar
    out[..., 1:] = beta_bar.unsqueeze(-1) * alpha
    return out


COUNTERFACTUALS = {
    "CF0_current": None,
    "CF1_uniform": _cf1_uniform,
    "CF2_availability": _cf2_availability,
    "CF4_no_local": _cf4_no_local,
    "CF5_mean_beta": _cf5_fixed_factor_mean_beta,
}
