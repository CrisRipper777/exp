"""R2-Design-2.7 pre-aggregation neighbor-utility components
(docs/BiAxis_R2_Design_2_7_PreAggregation_Neighbor_Utility_Audit.md).

    FactorPairScorer   shared psi across the 9 source->target factor pairs
                       input [F_i^b, F_j^a, F_i^b*F_j^a, |F_i^b-F_j^a|, e_a, e_b]
                       Linear(4d+2t, 2d) -> LN -> GELU -> Dropout(0.1)
                       -> Linear(2d, d) -> GELU -> Linear(d, 1)      (plan §2)
    NullScorer         target-conditioned null score s_0,i^{a->b} =
                       phi(F_i^b, e_a, e_b)                          (plan §3)
    GenericEdgeScorer  one score per observed edge from the local
                       ownership projection z0 (plan §14)
    PairTransform      T_ab per factor pair (plan §42, D2.7-E)
    ChunkedEngine      chunked null-augmented segment softmax + weighted
                       message scatter (plan §9: no [E,3,3,d] materialization)

Discipline:
    - tau = 1.0 fixed, no temperature sweep, no top-k during training;
    - no edge-label supervision — the scorer learns only through CE;
    - selection-only payload: U_a source transforms shared across target b;
    - observed graph as support only: no edge addition / reconstruction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_activation, make_norm


class FactorPairScorer(nn.Module):
    """Shared psi for all 9 factor pairs (plan §2)."""

    def __init__(self, factor_dim: int, type_dim: int = 16,
                 dropout: float = 0.1, activation: str = "gelu",
                 norm: str = "layernorm") -> None:
        super().__init__()
        d = int(factor_dim)
        t = int(type_dim)
        self.net = nn.Sequential(
            nn.Linear(4 * d + 2 * t, 2 * d),
            make_norm(norm, 2 * d),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * d, d),
            get_activation(activation),
            nn.Linear(d, 1),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: [*, 4d+2t] -> score [*, 1]."""
        return self.net(u).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class NullScorer(nn.Module):
    """Target-conditioned null score phi(F_i^b, e_a, e_b) (plan §3)."""

    def __init__(self, factor_dim: int, type_dim: int = 16,
                 hidden: int | None = None, activation: str = "gelu",
                 norm: str = "layernorm") -> None:
        super().__init__()
        d = int(factor_dim)
        t = int(type_dim)
        h = int(hidden) if hidden is not None else d
        self.net = nn.Sequential(
            nn.Linear(d + 2 * t, h),
            make_norm(norm, h),
            get_activation(activation),
            nn.Linear(h, 1),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: [N, d+2t] -> null score [N]."""
        return self.net(u).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GenericEdgeScorer(nn.Module):
    """One score per edge from the local ownership projection z0 (plan §14):
    s_ji = MLP([z0_i, z0_j, z0_i*z0_j, |z0_i-z0_j|]). Width solved by the
    model so the total scorer params match PAIR_EDGE within +/-5%."""

    def __init__(self, factor_dim: int, width: int,
                 dropout: float = 0.1, activation: str = "gelu",
                 norm: str = "layernorm") -> None:
        super().__init__()
        d = int(factor_dim)
        w = int(width)
        self.net = nn.Sequential(
            nn.Linear(4 * d, w),
            make_norm(norm, w),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(w, 1),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: [*, 4d] -> score [*]."""
        return self.net(u).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PairTransform(nn.Module):
    """T_ab per factor pair (plan §42): Linear(d,2d) -> LN -> GELU ->
    Linear(2d,d). 9 independent transforms."""

    def __init__(self, factor_dim: int, activation: str = "gelu",
                 norm: str = "layernorm") -> None:
        super().__init__()
        d = int(factor_dim)
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 2 * d),
                make_norm(norm, 2 * d),
                get_activation(activation),
                nn.Linear(2 * d, d),
            )
            for _ in range(9)  # (a, b) row-major
        ])

    def forward(self, f_src: torch.Tensor, a: int, b: int) -> torch.Tensor:
        return self.transforms[3 * a + b](f_src)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Chunked null-augmented segment softmax (plan §9)
# ---------------------------------------------------------------------------


def chunked_pair_message(
    f_block: torch.Tensor,  # [N, 3, d] frozen A0 pre-graph factors
    edge_index: torch.Tensor,  # [2, E]
    num_nodes: int,
    scores: torch.Tensor | None,  # [E] edge scores for this pair (None = uniform)
    null_scores: torch.Tensor,  # [N] per-target null scores for this pair
    payload_a: torch.Tensor,  # [N, d] = U_a(F[:, a])
    edge_chunk_size: int = 50000,
    edge_mask: torch.Tensor | None = None,  # [E] bool: False = dropped
) -> torch.Tensor:
    """m_i = sum_j alpha_ji * U_a(F_j^a) with
    alpha_ji = exp(s_ji) / (exp(s_null_i) + sum_k exp(s_ki)) over the KEPT
    neighbors (null always available). Chunked; no [E,3,3,d] materialized.
    scores=None => uniform real-neighbor weights (TARGET_NULL_ONLY uses a
    constant-0 score = the same path with scores = zeros)."""
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    if edge_mask is not None:
        src = src[edge_mask]
        dst = dst[edge_mask]
        if scores is not None:
            scores = scores[edge_mask]
        num_edges = int(src.size(0))

    m = torch.zeros(num_nodes, payload_a.size(-1), dtype=f_block.dtype,
                    device=f_block.device)
    if num_edges == 0:
        return m
    if scores is None:
        scores = torch.zeros(num_edges, dtype=f_block.dtype, device=f_block.device)

    # pass 1: per-target max over real neighbors
    max_i = torch.full((num_nodes,), float("-inf"), dtype=scores.dtype,
                       device=scores.device)
    seg_max = torch.zeros(num_nodes, dtype=scores.dtype, device=scores.device)
    seg_max = seg_max.scatter_reduce(0, dst, scores, reduce="amax", include_self=False)
    max_i = torch.maximum(max_i, seg_max)
    # targets without any kept neighbor: max_i = -inf -> their null gets all mass
    max_i = torch.where(torch.isfinite(max_i), max_i,
                        torch.zeros_like(max_i))

    # pass 2: denominator = exp(null - max) + sum exp(s - max)
    denom = torch.exp(null_scores - max_i)  # [N]
    for start in range(0, num_edges, edge_chunk_size):
        end = min(start + edge_chunk_size, num_edges)
        d_c = dst[start:end]
        e_c = torch.exp(scores[start:end] - max_i[d_c])
        denom = denom.scatter_add(0, d_c, e_c)

    # pass 3: weighted message
    for start in range(0, num_edges, edge_chunk_size):
        end = min(start + edge_chunk_size, num_edges)
        s_c, d_c = src[start:end], dst[start:end]
        alpha = torch.exp(scores[start:end] - max_i[d_c]) / denom[d_c]
        contrib = alpha.unsqueeze(-1) * payload_a[s_c]
        m = m.scatter_add(0, d_c.unsqueeze(-1).expand_as(contrib), contrib)
    return m
