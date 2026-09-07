"""P1 Bi-Axis components: topology-only structural relation decomposition (M2)
and the factor-side graph consumers (M3: budget / selector), plus the sparse
relation-weighted aggregation helpers.

Discipline (plan §5.1):
    R = f(A),  R does NOT read x_t / x_v / C / Pt / Pv.
    Semantic factors only enter at the coupling stage (budget / selector).

Memory discipline (plan §9):
    - never materialize [N,N], [K,N,N] or [E,K,d] tensors;
    - only edge_index [2,E] and r [E,K] are edge-level objects;
    - aggregation is scatter-based (bincount + index_add) with optional
      edge chunking (edge_chunk_size); the selector scores are computed
      one relation at a time so its input is [N,F,3d+1] at most.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_activation, make_norm

_EPS = 1e-8


# ---------------------------------------------------------------------------
# M2: Topology-only Structural Relation Decomposition
# ---------------------------------------------------------------------------


def compute_raw_struct_signature(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Deterministic topology-only signature [u0, u1, u2] (plan §6).

    u0 = log(1 + d_i)
    u1 = P u0,  u2 = P u1   with row-normalized transition P = D^{-1} A.

    edge_index is the message direction (src -> dst); for the undirected
    graphs used in P1 the row-normalized transition equals the message
    direction D^{-1}A with D = in-degree. Isolated nodes yield zeros.

    Returns [N, 3] float32, NOT z-scored. Contains no parameters and no
    dependency on node features.
    """
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, 3, dtype=torch.float32, device=edge_index.device)
    src, dst = edge_index[0], edge_index[1]
    deg = torch.bincount(dst, minlength=num_nodes).to(torch.float32)
    u0 = torch.log1p(deg)  # [N]

    def _step(v: torch.Tensor) -> torch.Tensor:
        # P v = D^{-1} A v, computed by scatter-add of v[src] onto dst.
        aggr = torch.zeros_like(v)
        aggr.index_add_(0, dst, v[src])
        return aggr / (deg + _EPS)

    u1 = _step(u0)
    u2 = _step(u1)
    return torch.stack([u0, u1, u2], dim=1)  # [N, 3]


def zscore_columns(raw: torch.Tensor) -> torch.Tensor:
    """Whole-graph z-score normalization per column: (s - mu) / (sigma + eps)."""
    mean = raw.mean(dim=0, keepdim=True)
    std = raw.std(dim=0, keepdim=True, unbiased=False)
    return (raw - mean) / (std + _EPS)


class TopologyDiffusionSignature(nn.Module):
    """s_i = MLP_S(zscore([u0, u1, u2])) in R^{d_r} (plan §6).

    Input: ONLY edge_index / num_nodes (plus an optional cached raw
    signature). The caller owns the cache; this module never reads x.
    """

    def __init__(self, relation_dim: int = 32, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, int(relation_dim)),
            get_activation(activation),
            nn.Linear(int(relation_dim), int(relation_dim)),
        )
        self.relation_dim = int(relation_dim)

    def forward(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        raw_signature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if raw_signature is None:
            raw_signature = compute_raw_struct_signature(edge_index, num_nodes)
        s_bar = zscore_columns(raw_signature.to(torch.float32))
        return self.net(s_bar)  # [N, d_r]


class EdgeStructuralToken(nn.Module):
    """Symmetric edge token (plan §7):

        e_ij = MLP_E([s_i + s_j || |s_i - s_j| || s_i * s_j])

    Symmetric in (i, j) by construction so the two directions of an undirected
    edge get identical tokens.
    """

    def __init__(self, relation_dim: int = 32, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * int(relation_dim), int(relation_dim)),
            get_activation(activation),
            nn.Linear(int(relation_dim), int(relation_dim)),
        )

    def forward(self, s: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        s_i, s_j = s[src], s[dst]
        token = torch.cat([s_i + s_j, (s_i - s_j).abs(), s_i * s_j], dim=-1)
        return self.net(token)  # [E, d_r]


class RelationPrototypes(nn.Module):
    """K learnable prototypes rho_k and soft assignment (plan §8):

        r_ij,k = exp(cos(e_ij, rho_k) / tau) / sum_l exp(...)

    K == 1 must be handled by the caller with a strict fast path (r = ones,
    no cosine softmax). Prototype names are R1..RK only; prototype
    permutation across seeds is expected and must not be read as stable
    semantics.
    """

    def __init__(self, num_relations: int, relation_dim: int, temperature: float = 0.5) -> None:
        super().__init__()
        self.num_relations = int(num_relations)
        self.temperature = float(temperature)
        prototypes = torch.empty(self.num_relations, int(relation_dim))
        nn.init.normal_(prototypes, std=1.0)
        prototypes = F.normalize(prototypes, dim=-1)  # cosine starts well-scaled
        self.prototypes = nn.Parameter(prototypes)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        e_n = F.normalize(e, dim=-1)
        rho = F.normalize(self.prototypes, dim=-1)
        logits = e_n @ rho.t() / self.temperature  # [E, K]
        return torch.softmax(logits, dim=-1)  # [E, K]


# ---------------------------------------------------------------------------
# Relation-weighted sparse aggregation (plan §9)
# ---------------------------------------------------------------------------


def compute_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """In-degree per node (self-loops are absent in the P1 graphs)."""
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    return torch.bincount(edge_index[1], minlength=num_nodes).to(torch.float32)


def relation_mass(
    edge_index: torch.Tensor,
    r: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """m_i,k = sum_{j in N(i)} r_ji,k  -> [N, K] (scatter over dst)."""
    dst = edge_index[1]
    mass = torch.zeros(num_nodes, r.size(1), dtype=r.dtype, device=r.device)
    mass.index_add_(0, dst, r)
    return mass


def relation_availability(
    mass: torch.Tensor,
    degree: torch.Tensor,
) -> torch.Tensor:
    """a_i,k = m_i,k / (d_i + eps); for non-isolated nodes sum_k a = 1."""
    return mass / (degree.unsqueeze(-1) + _EPS)


def relation_weighted_mean(
    edge_index: torch.Tensor,
    r: torch.Tensor,
    features: torch.Tensor,
    num_nodes: int,
    edge_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-relation weighted mean of ``features`` over incoming edges.

        g_tilde_i,k = sum_{j in N(i)} r_ji,k * f_j          [N, K, d]
        g_i,k       = g_tilde_i,k / (m_i,k + eps)            [N, K, d]
        m           = relation mass                          [N, K]

    features: [N, d] (typically the concatenated factor block, plan §9.2).
    Loops over relations so the peak transient is [chunk, d] — never [E, K, d].
    Returns (g, m).
    """
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    k = int(r.size(1))
    dim = int(features.size(-1))
    dtype = features.dtype

    mass = relation_mass(edge_index, r, num_nodes)
    acc = torch.zeros(num_nodes, k, dim, dtype=dtype, device=features.device)

    if edge_chunk_size is None or edge_chunk_size >= num_edges:
        # Memory fix (2026-09-04): hoist the [E, d] source-feature gather OUT
        # of the relation loop — one gather shared by all K relations instead
        # of K identical gathers (bitwise identical values; the four
        # r-scaled products still exist and are what autograd retains).
        features_src = features[src]  # [E, d]
        for rel in range(k):
            weighted = r[:, rel].unsqueeze(-1) * features_src  # [E, d]
            acc[:, rel].index_add_(0, dst, weighted)
    else:
        for start in range(0, num_edges, int(edge_chunk_size)):
            end = min(start + int(edge_chunk_size), num_edges)
            src_c, dst_c = src[start:end], dst[start:end]
            r_c = r[start:end]
            features_src_c = features[src_c]  # hoisted per chunk
            for rel in range(k):
                weighted = r_c[:, rel].unsqueeze(-1) * features_src_c  # [chunk, d]
                acc[:, rel].index_add_(0, dst_c, weighted)
    g = acc / (mass.unsqueeze(-1) + _EPS)  # [N, K, d]
    return g, mass


def neighbor_mean(
    edge_index: torch.Tensor,
    features: torch.Tensor,
    num_nodes: int,
    edge_chunk_size: int | None = None,
) -> torch.Tensor:
    """Plain incoming-neighbor mean of ``features`` (weight 1 per edge).

    Used for the relation-averaged context g_bar (audit §4: because
    sum_k r_ji,k = 1, g_bar == neighbor mean regardless of r).
    """
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    deg = compute_degree(edge_index, num_nodes)
    acc = torch.zeros(num_nodes, features.size(-1), dtype=features.dtype, device=features.device)
    if edge_chunk_size is None or edge_chunk_size >= num_edges:
        acc.index_add_(0, dst, features[src])
    else:
        for start in range(0, num_edges, int(edge_chunk_size)):
            end = min(start + int(edge_chunk_size), num_edges)
            acc.index_add_(0, dst[start:end], features[src[start:end]])
    return acc / (deg.unsqueeze(-1) + _EPS)


# ---------------------------------------------------------------------------
# M3a: Factor Graph Budget (plan §11)
# ---------------------------------------------------------------------------


class FactorGraphBudget(nn.Module):
    """beta_i^f = sigmoid(MLP_B([f_i || g_bar_i^f])) in (0, 1).

    Initialized so beta ~ 0.5 (final linear zero-init, bias 0).
    beta = how much graph evidence each factor uses at each node.
    """

    def __init__(
        self,
        factor_dim: int,
        hidden_dim: int = 64,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * int(factor_dim), int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), 1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, f: torch.Tensor, g_bar: torch.Tensor) -> torch.Tensor:
        """f, g_bar: [N, F, d_f] -> beta: [N, F] in (0, 1)."""
        return torch.sigmoid(self.net(torch.cat([f, g_bar], dim=-1))).squeeze(-1)


# ---------------------------------------------------------------------------
# M3b: Factor-Relation Selector (plan §12)
# ---------------------------------------------------------------------------


class FactorRelationSelector(nn.Module):
    """alpha_i,f,k = Softmax_k(MLP_R([f_i || g_i,k^f || f_i * g_i,k^f || a_i,k])).

    alpha = which structural relations each factor prefers; sum_k alpha = 1.
    K == 1 uses a strict fast path: alpha = 1, no MLP is evaluated.
    Scores are computed one relation at a time: peak transient [N, F, 3d+1].
    """

    def __init__(
        self,
        num_relations: int,
        factor_dim: int,
        hidden_dim: int = 64,
        activation: str = "gelu",
        input_norm: str | None = None,
    ) -> None:
        super().__init__()
        self.num_relations = int(num_relations)
        in_dim = 3 * int(factor_dim) + 1
        self.input_norm = make_norm(input_norm, in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        f: torch.Tensor,
        g: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        """f: [N, F, d_f]; g: [N, F, K, d_f]; a: [N, K] -> alpha: [N, F, K].

        Scores are computed one relation at a time: peak transient [N, F, 3d+1].
        """
        if self.num_relations == 1:
            return torch.ones(f.size(0), f.size(1), 1, dtype=f.dtype, device=f.device)
        num_nodes, num_factors, d = f.shape
        k = int(g.size(2))
        scores_list = []
        for rel in range(k):
            g_rel = g[:, :, rel]  # [N, F, d]
            a_rel = availability[:, rel]  # [N]
            a_rel = a_rel.unsqueeze(0).expand(num_factors, num_nodes).t().unsqueeze(-1)  # [N, F, 1]
            feat = torch.cat(
                [f, g_rel, f * g_rel, a_rel], dim=-1
            )  # [N, F, 3d+1]
            scores_list.append(self.net(self.input_norm(feat)))  # [N, F, 1]
        scores = torch.cat(scores_list, dim=-1)  # [N, F, K]
        return torch.softmax(scores, dim=-1)
