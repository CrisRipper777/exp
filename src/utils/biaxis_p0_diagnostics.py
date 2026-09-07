"""P0 diagnostics for the Bi-Axis MAG project.

Unsupervised, label-free, read-only utilities:
  - P0-B: factorization sanity (cross-modal sim, C-P overlap, effective rank)
  - P0-C: factor-dependent edge statistics (cosine, Spearman, top-q Jaccard, gaps)
  - fixed one-hop GCN propagation (D^-1/2 (A+I) D^-1/2) shared by NC/LP probes
  - generic conflict statistics over per-unit graph-utility deltas

Discipline: everything runs under torch.no_grad, never mutates model
parameters, uses fixed seeds for any sampling, and never touches labels
(probe functions are the only label consumers and live in separate steps).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import scatter

from src.data.graph_utils import canonicalize_edges


# ---------------------------------------------------------------------------
# Sampling helpers (fixed seeds, reproducible across models)
# ---------------------------------------------------------------------------


def _sample_nodes(num_nodes: int, max_nodes: int, seed: int = 42) -> torch.Tensor:
    if num_nodes <= max_nodes:
        return torch.arange(num_nodes)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_nodes, generator=generator)[:max_nodes]


def _sample_edges(edges: torch.Tensor, max_edges: int, seed: int = 42) -> torch.Tensor:
    if edges.size(0) <= max_edges:
        return edges
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(edges.size(0), generator=generator)[:max_edges]
    return edges[perm]


# ---------------------------------------------------------------------------
# P0-B: factorization sanity
# ---------------------------------------------------------------------------


@torch.no_grad()
def _effective_rank(matrix: torch.Tensor) -> float:
    """Spectral effective rank r_eff = exp(-sum q log q) over centered SVD."""
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum()
    if total <= 0:
        return 0.0
    q = singular / total
    q = q[q > 0]
    return float(torch.exp(-(q * q.log()).sum()).item())


@torch.no_grad()
def _cross_cov_overlap(c: torch.Tensor, p: torch.Tensor) -> float:
    """||Cov(C, P)||_F / d over the given (batch, d) tensors."""
    c_c = c - c.mean(dim=0, keepdim=True)
    p_c = p - p.mean(dim=0, keepdim=True)
    cov = c_c.t() @ p_c / max(c.size(0) - 1, 1)
    return float((cov.norm() / c.size(1)).item())


@torch.no_grad()
def compute_factor_sanity(factors: dict[str, torch.Tensor], max_nodes: int = 10000, seed: int = 42) -> dict:
    """P0-B sanity: are C / P_t / P_v meaningful, separated, non-collapsed spaces?"""
    idx = _sample_nodes(int(factors["c"].size(0)), max_nodes, seed)
    c = factors["c"][idx]
    c_t = factors["c_t"][idx]
    c_v = factors["c_v"][idx]
    p_t = factors["p_t"][idx]
    p_v = factors["p_v"][idx]

    common_sim = float(F.cosine_similarity(c_t, c_v, dim=-1).mean().item())
    private_sim = float(F.cosine_similarity(p_t, p_v, dim=-1).mean().item())
    return {
        "num_nodes_sampled": int(idx.numel()),
        "common_sim": common_sim,
        "private_sim": private_sim,
        "cp_overlap_t": _cross_cov_overlap(c_t, p_t),
        "cp_overlap_v": _cross_cov_overlap(c_v, p_v),
        "effrank_c": _effective_rank(c),
        "effrank_pt": _effective_rank(p_t),
        "effrank_pv": _effective_rank(p_v),
        "c_norm": float(c.norm(dim=-1).mean().item()),
        "pt_norm": float(p_t.norm(dim=-1).mean().item()),
        "pv_norm": float(p_v.norm(dim=-1).mean().item()),
    }


# ---------------------------------------------------------------------------
# P0-C: factor-dependent edge statistics
# ---------------------------------------------------------------------------


@torch.no_grad()
def _edge_cosine_similarity(h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    a = F.normalize(h[src], dim=-1)
    b = F.normalize(h[dst], dim=-1)
    return (a * b).sum(dim=-1)


@torch.no_grad()
def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pairwise Spearman; NaN when either side is constant (degenerate factor)."""
    if a.numel() < 2 or float(a.std().item()) == 0.0 or float(b.std().item()) == 0.0:
        return float("nan")
    from scipy.stats import spearmanr

    return float(spearmanr(a.numpy(), b.numpy()).correlation)


@torch.no_grad()
def _top_ratio_jaccard(a: torch.Tensor, b: torch.Tensor, ratio: float) -> float:
    k = max(int(round(ratio * a.numel())), 1)
    top_a = set(torch.topk(a, k).indices.tolist())
    top_b = set(torch.topk(b, k).indices.tolist())
    union = top_a | top_b
    if not union:
        return 0.0
    return len(top_a & top_b) / len(union)


@torch.no_grad()
def compute_edge_factor_statistics(
    factors: dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    max_edges: int = 500000,
    seed: int = 42,
    top_ratios: tuple[float, ...] = (0.1, 0.2),
    gap_threshold: float = 0.25,
) -> dict:
    """P0-C: does the SAME observed edge look different in C / P_t / P_v?"""
    c, p_t, p_v = factors["c"], factors["p_t"], factors["p_v"]
    edges = canonicalize_edges(edge_index)  # one canonical direction per undirected edge
    edges = _sample_edges(edges, max_edges, seed)
    src, dst = edges[:, 0], edges[:, 1]

    s_c = _edge_cosine_similarity(c, src, dst)
    s_t = _edge_cosine_similarity(p_t, src, dst)
    s_v = _edge_cosine_similarity(p_v, src, dst)
    sims = {"C": s_c, "Pt": s_t, "Pv": s_v}

    result: dict = {
        "num_edges_sampled": int(edges.size(0)),
        "mean_sim_C": float(s_c.mean().item()),
        "mean_sim_Pt": float(s_t.mean().item()),
        "mean_sim_Pv": float(s_v.mean().item()),
        "std_sim_C": float(s_c.std().item()),
        "std_sim_Pt": float(s_t.std().item()),
        "std_sim_Pv": float(s_v.std().item()),
    }
    pairs = [("C", "Pt"), ("C", "Pv"), ("Pt", "Pv")]
    for name_a, name_b in pairs:
        a, b = sims[name_a], sims[name_b]
        result[f"rho_{name_a}_{name_b}"] = _spearman(a, b)
        for ratio in top_ratios:
            pct = int(round(ratio * 100))
            result[f"jaccard_top{pct}_{name_a}_{name_b}"] = _top_ratio_jaccard(a, b, ratio)
        gap = (a - b).abs()
        result[f"mean_abs_gap_{name_a}_{name_b}"] = float(gap.mean().item())
        result[f"frac_gap_gt_{gap_threshold}_{name_a}_{name_b}"] = float(
            (gap > gap_threshold).float().mean().item()
        )
    return result


# ---------------------------------------------------------------------------
# Fixed one-hop GCN propagation (no learned relation, no attention)
# ---------------------------------------------------------------------------


@torch.no_grad()
def aggregate_fixed_gcn(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int | None = None,
) -> torch.Tensor:
    """G^f = D^{-1/2} (A + I) D^{-1/2} F^f, source_to_target.

    Self-loops are added here on purpose: dataset configs keep
    ``add_self_loops: false``, so the fixed propagation must add them itself
    (audit §8).
    """
    if num_nodes is None:
        num_nodes = int(h.size(0))
    norm_ei, norm_w = gcn_norm(
        edge_index,
        edge_weight=None,
        num_nodes=num_nodes,
        improved=False,
        add_self_loops=True,
        flow="source_to_target",
        dtype=h.dtype,
    )
    msg = h[norm_ei[0]] * norm_w.view(-1, 1)
    return scatter(msg, norm_ei[1], dim=0, dim_size=num_nodes, reduce="sum")


@torch.no_grad()
def propagate_fixed_gcn(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int | None = None,
) -> torch.Tensor:
    """tilde F^f = LayerNorm(F^f + G^f) — the P0-D 'graph' representation."""
    agg = aggregate_fixed_gcn(h, edge_index, num_nodes=num_nodes)
    return F.layer_norm(h + agg, (h.size(1),))


# ---------------------------------------------------------------------------
# Conflict statistics over per-unit deltas (NC: CE delta; LP: RR delta)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_conflict_statistics(
    delta_dict: dict[str, torch.Tensor],
    names: tuple[str, ...] = ("C", "Pt", "Pv"),
) -> dict:
    """Per-unit graph-utility deltas. delta > 0: graph helps; < 0: graph hurts.

    Returns pairwise delta correlations, sign-conflict rates, and the
    three-factor agreement pattern counts (plan §11.1 / §12.2).
    """
    deltas = {name: delta_dict[name].float() for name in names if name in delta_dict}
    if len(deltas) < 2:
        raise ValueError(f"need at least two deltas, got {sorted(deltas)}")

    num_units = next(iter(deltas.values())).numel()
    for name, delta in deltas.items():
        if delta.numel() != num_units:
            raise ValueError(f"delta[{name}] has {delta.numel()} units, expected {num_units}")

    result: dict = {"num_units": num_units}
    result["mean_delta"] = {name: float(delta.mean().item()) for name, delta in deltas.items()}
    result["frac_positive"] = {name: float((delta > 0).float().mean().item()) for name, delta in deltas.items()}

    names_present = list(deltas)
    for idx_a, name_a in enumerate(names_present):
        for name_b in names_present[idx_a + 1 :]:
            a, b = deltas[name_a], deltas[name_b]
            conflict = (a > 0) & (b < 0) | (a < 0) & (b > 0)
            result[f"conflict_{name_a}_{name_b}"] = float(conflict.float().mean().item())
            result[f"corr_spearman_delta_{name_a}_{name_b}"] = _spearman(a, b)
            result[f"corr_pearson_delta_{name_a}_{name_b}"] = float(
                torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            )
            result[f"help_hurt_{name_a}_{name_b}"] = float(
                ((a > 0) & (b < 0)).float().mean().item()
            )
            result[f"hurt_help_{name_a}_{name_b}"] = float(
                ((a < 0) & (b > 0)).float().mean().item()
            )

    if len(deltas) == 3:
        a, b, c = (deltas[name] for name in names_present)
        result["pattern_all_help"] = float(((a > 0) & (b > 0) & (c > 0)).float().mean().item())
        result["pattern_all_hurt"] = float(((a < 0) & (b < 0) & (c < 0)).float().mean().item())
        result["pattern_mixed"] = float(
            (((a > 0) != (b > 0)) | ((b > 0) != (c > 0))).float().mean().item()
        )
    return result
