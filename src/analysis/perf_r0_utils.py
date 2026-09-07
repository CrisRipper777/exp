"""R0 performance-diagnostic shared layer (plan §32 Prompt 2).

Discipline:
    - frozen models are NEVER modified; everything here reads them.
    - TEST labels/metrics are never touched (the layer exposes only
      x / edge_index / y_train / y_val / train_idx / val_idx).
    - big tensors live only within one (dataset, seed) lifecycle; statistics
      are chunked over N / E where needed (ele-fashion safe).
    - all counterfactuals compare against the checkpoint's OWN current-Gamma
      forward, never against fresh benchmark runs.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from hydra import compose, initialize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CKPT_ROOT = PROJECT_ROOT / "outputs" / "p3" / "operator"
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
FACTOR_NAMES = ["C", "Pt", "Pv"]


@dataclass
class R0Setup:
    dataset: str
    seed: int
    cfg: object
    data: object
    model: object
    head: torch.nn.Module
    device: torch.device


def resolve_cfg(dataset: str, seed: int) -> object:
    # NOTE: hydra resolves config_path relative to THIS file (src/analysis/).
    with initialize(config_path="../../configs", version_base=None):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}", "task=nc", "model=biaxis_p3",
                "model.p3.operator_mode=full_interaction", f"seed={int(seed)}",
            ],
        )


def load_setup(dataset: str, seed: int, device: torch.device) -> R0Setup:
    """Load checkpoint model + head + data. NEVER reads test labels."""
    from src.data import load_mag_data
    from src.models.biaxis_p3 import Model

    cfg = resolve_cfg(dataset, seed)
    data = load_mag_data(cfg, "nc", int(seed))
    ckpt = torch.load(
        CKPT_ROOT / dataset / "OFR" / f"seed_{seed}" / "model.pt",
        map_location="cpu", weights_only=False,
    )
    model = Model(cfg, ckpt["data_info"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(ckpt["head_state"])
    return R0Setup(dataset, seed, cfg, data, model, head, device)


def assert_no_test_access(data: object) -> None:
    """Guard: R0 scripts must only use train/val supervision."""
    assert data.train_idx is not None and data.val_idx is not None


@torch.no_grad()
def extract_forward(setup: R0Setup) -> dict:
    """One eval full-graph forward returning every intermediate R0 needs.

    Returns (all on GPU except indices):
        factors: h_t,h_v,c_t,c_v,C,Pt,Pv      [N, d_h]/[N, d_f]
        z_local [N, hidden], z_final [N, hidden]
        f_block [N,3,d_f], f_tilde [N,3,d_f]
        graph_out: r [E,K], availability [N,K], g_perm [N,F,K,d_f],
                   gamma [N,F,K+1], beta, alpha
        scores: s_rel [N,F,K], s_aug [N,F,K+1]
        deg [N]
    """
    model, data, device = setup.model, setup.data, setup.device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    factors, z_local = model._encode(x)
    num_nodes = int(x.size(0))
    f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
    graph_out = model._graph_update(f_block, edge_index, num_nodes)
    # scores are not returned by _graph_update: recompute with frozen weights
    s_rel = model.transport_scorer(f_block, graph_out["g_perm"])
    s_aug = torch.cat(
        [model.null_score.reshape(1, -1, 1).expand(num_nodes, -1, 1), s_rel], dim=-1
    )
    z_final = model.fusion(
        torch.cat(
            [graph_out["f_tilde"][:, 0], graph_out["f_tilde"][:, 1], graph_out["f_tilde"][:, 2]],
            dim=-1,
        )
    )
    deg = torch.bincount(edge_index[1], minlength=num_nodes).to(torch.float32)
    return {
        "factors": factors,
        "z_local": z_local,
        "z_final": z_final,
        "f_block": f_block,
        "f_tilde": graph_out["f_tilde"],
        "graph_out": graph_out,
        "scores": {"s_rel": s_rel, "s_aug": s_aug},
        "deg": deg,
        "edge_index": edge_index,
    }


@torch.no_grad()
def val_metrics_with_head(setup: R0Setup, z: torch.Tensor) -> dict[str, float]:
    """Val acc / macro-F1 of head(z) on the VAL split (train/val only)."""
    data, device = setup.data, setup.device
    y = data.y.to(device)
    logits = setup.head(z)
    pred = logits.argmax(dim=-1)
    val_idx = data.val_idx.to(device)
    acc = float((pred[val_idx] == y[val_idx]).float().mean().item())
    return {"val_acc": acc}


@torch.no_grad()
def _sanities(graph_out: dict, deg: torch.Tensor) -> dict[str, float]:
    gamma = graph_out["gamma"]
    r = graph_out["r"]
    availability = graph_out["availability"]
    non_isolated = deg > 0
    row_sum_gamma = float((gamma.sum(dim=-1) - 1.0).abs().max().item())
    row_sum_r = float((r.sum(dim=-1) - 1.0).abs().max().item())
    if bool(non_isolated.any()):
        row_sum_avail = float((availability[non_isolated].sum(dim=-1) - 1.0).abs().max().item())
    else:
        row_sum_avail = float("nan")
    return {
        "gamma_row_sum_maxdev": row_sum_gamma,
        "r_row_sum_maxdev": row_sum_r,
        "availability_row_sum_maxdev": row_sum_avail,
    }


# ---------------------------------------------------------------------------
# Chunked statistics (ele-fashion safe: pure sum/sumsq aggregations)
# ---------------------------------------------------------------------------


def chunked_mean_var(values: torch.Tensor, chunk: int = 200_000) -> tuple[float, float]:
    """Mean and population variance over the leading dim (streaming)."""
    total = float(values.size(0))
    s = torch.zeros(values.size(-1), dtype=torch.float64, device=values.device)
    s2 = torch.zeros(values.size(-1), dtype=torch.float64, device=values.device)
    for start in range(0, values.size(0), chunk):
        block = values[start : start + chunk].double()
        s += block.sum(dim=0)
        s2 += (block * block).sum(dim=0)
    mean = s / total
    var = (s2 / total - mean * mean).clamp_min(0.0)
    return float(mean.mean().item()), float(var.mean().item())


def chunked_mean_cos(a: torch.Tensor, b: torch.Tensor, chunk: int = 200_000) -> float:
    """mean_i cos(a_i, b_i) over rows (streaming)."""
    total = float(a.size(0))
    s = 0.0
    for start in range(0, a.size(0), chunk):
        a_b, b_b = a[start : start + chunk], b[start : start + chunk]
        cos = torch.nn.functional.cosine_similarity(a_b, b_b, dim=-1)
        s += float(cos.sum().item())
    return s / total


def chunked_pairwise_overlap(a: torch.Tensor, b: torch.Tensor, chunk: int = 200_000) -> dict[str, float]:
    """mean cosine + mean absolute cross-covariance between a and b."""
    total = float(a.size(0))
    cos_sum = 0.0
    xcov_sum = 0.0
    d = a.size(1)
    for start in range(0, a.size(0), chunk):
        a_b, b_b = a[start : start + chunk], b[start : start + chunk]
        a_c = a_b - a_b.mean(dim=0, keepdim=True)
        b_c = b_b - b_b.mean(dim=0, keepdim=True)
        cos = torch.nn.functional.cosine_similarity(a_b, b_b, dim=-1)
        xcov = (a_c * b_c).sum(dim=-1).abs().mean()
        cos_sum += float(cos.sum().item())
        xcov_sum += float(xcov.sum().item())
    return {"mean_cos": cos_sum / total, "mean_abs_xcov": xcov_sum / total}


# ---------------------------------------------------------------------------
# Fixed linear probe (plan §5.3): StandardScaler + Ridge(alpha=1.0),
# fit TRAIN only, eval VAL only. No hyperparameter tuning.
# ---------------------------------------------------------------------------


def ridge_probe(features: torch.Tensor, setup: R0Setup) -> dict[str, float]:
    """features [N, d] (GPU). fit on train_idx, eval on val_idx."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler

    data, device = setup.data, setup.device
    feats = features.detach().cpu().numpy()
    y = data.y.numpy()
    train_idx = data.train_idx.numpy()
    val_idx = data.val_idx.numpy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(feats[train_idx])
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, y[train_idx])
    X_val = scaler.transform(feats[val_idx])
    pred = clf.predict(X_val)
    return {
        "val_acc": float(accuracy_score(y[val_idx], pred)),
        "val_macro_f1": float(f1_score(y[val_idx], pred, average="macro")),
    }


def concat_rows(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tensors, dim=-1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
