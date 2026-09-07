"""R1-A Bi-Axis performance components (plan §36 Prompt 2; audit §1bis/§2/§7).

Factor-conditioned Edge Reliability:

    eta_ji^f = 2 * sigmoid( MLP_eta( [u_i^f + u_j^f
                                      | |u_i^f - u_j^f|
                                      | u_i^f * u_j^f
                                      | cos(u_i^f, u_j^f)] ) )
    u_i^f = P_f f_i                    per-factor projection d -> 32

    g_ifk             = sum_j r_ji,k * eta_ji^f * f_j / (sum_j r_ji,k * eta_ji^f + eps)
    effective_mass_ifk = sum_j r_ji,k * eta_ji^f

Discipline:
    - eta is factor-conditioned but relation-INDEPENDENT (audit §1bis): it
      answers "is this neighbor reliable for factor f?", NOT "which relation
      should this edge play for factor f" (that is R1-A2).
    - zero-init final layer: delta == 0 -> eta == 1 EXACTLY at step 0
      (2 * sigmoid(0) == 1.0 in float32). A fresh A1 model is therefore
      mathematically equivalent to the baseline; the float grouping of the
      aggregation differs -> equivalence tests use allclose, never equal.
    - eta in (0, 2) strictly (sigmoid is open-interval).
    - eta only ever exists as [chunk, F]; [E, F] / [E, F, d] are never
      materialized on the training path (the no-grad diagnostics helper
      below is the sole exception, documented at its site).
    - never reads labels or raw modalities; factor semantics only.
    - r (structural relation posterior) is consumed read-only, never modified.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import get_activation

_EPS = 1e-8


class FactorConditionedEdgeReliability(nn.Module):
    """eta_ji^f in (0, 2): factor-specific semantic reliability of edge j->i.

    Symmetric by construction (token built from u_i + u_j / |u_i - u_j| /
    u_i * u_j / cos(u_i, u_j)): on an undirected graph the two directions of
    an edge get bitwise-identical eta.
    """

    def __init__(
        self,
        num_factors: int = 3,
        factor_dim: int = 128,
        proj_dim: int = 32,
        hidden_dim: int = 64,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.num_factors = int(num_factors)
        self.proj_dim = int(proj_dim)
        # One projection per factor (audit §7); bias=True so a factor living
        # in the kernel of P_f still yields a nonzero u (cosine stays defined;
        # a zero u_i or u_j makes the numerator 0 -> cos = 0, no clamp needed).
        self.projections = nn.ModuleList(
            [
                nn.Linear(int(factor_dim), int(proj_dim), bias=True)
                for _ in range(self.num_factors)
            ]
        )
        self.mlp = nn.Sequential(
            nn.Linear(3 * int(proj_dim) + 1, int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), 1),
        )
        # Zero-init final layer: delta == 0 -> eta == 1 exactly (audit §7).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def extra_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def forward(
        self,
        f_src: torch.Tensor,
        f_dst: torch.Tensor,
    ) -> torch.Tensor:
        """f_src / f_dst: [C, F, d] edge-chunk factor states -> eta [C, F]."""
        etas = []
        for f in range(self.num_factors):
            u_i = self.projections[f](f_src[:, f])  # [C, p]
            u_j = self.projections[f](f_dst[:, f])  # [C, p]
            cos = (u_i * u_j).sum(dim=-1) / (u_i.norm(dim=-1) * u_j.norm(dim=-1) + _EPS)
            token = torch.cat(
                [u_i + u_j, (u_i - u_j).abs(), u_i * u_j, cos.unsqueeze(-1)], dim=-1
            )  # [C, 3p+1]
            etas.append(2.0 * torch.sigmoid(self.mlp(token)).squeeze(-1))
        return torch.stack(etas, dim=1)  # [C, F]


def reliable_relation_weighted_mean(
    edge_index: torch.Tensor,
    r: torch.Tensor,
    f_block: torch.Tensor,
    reliability: nn.Module,
    num_nodes: int,
    edge_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-relation reliable weighted mean of the factor block (audit §Q4/Q5).

        g_ifk             = sum_j r_ji,k * eta_ji^f * f_j / (sum_j r_ji,k * eta_ji^f + eps)
        effective_mass_ifk = sum_j r_ji,k * eta_ji^f

    f_block: [N, F, d]. Returns (g_perm [N, F, K, d], effective_mass [N, F, K]).

    eta is computed per edge chunk and consumed immediately inside the
    (chunk, factor, relation) loops; no [E, F] / [E, F, d] tensor is kept.
    With eta == 1 the result is mathematically equal to
    ``relation_weighted_mean`` (float grouping differs -> allclose only).
    Isolated nodes: acc = 0, effective_mass = 0 -> g = 0, no NaN.
    Empty graph (E = 0): returns zeros, no crash.
    """
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    num_factors = int(f_block.size(1))
    num_relations = int(r.size(1))
    dim = int(f_block.size(2))
    dtype = f_block.dtype
    device = f_block.device

    acc = torch.zeros(num_nodes, num_factors, num_relations, dim, dtype=dtype, device=device)
    eff_mass = torch.zeros(num_nodes, num_factors, num_relations, dtype=dtype, device=device)

    # E = 0 -> chunk = 1 keeps range()'s step nonzero (empty loop, no crash).
    chunk = max(num_edges, 1) if edge_chunk_size is None else max(int(edge_chunk_size), 1)
    for start in range(0, num_edges, chunk):
        end = min(start + chunk, num_edges)
        src_c, dst_c = src[start:end], dst[start:end]
        r_c = r[start:end]  # [C, K]
        f_src_c = f_block[src_c]  # [C, F, d]
        f_dst_c = f_block[dst_c]  # [C, F, d]
        eta_c = reliability(f_src_c, f_dst_c)  # [C, F]
        for f in range(num_factors):
            f_src_f = f_src_c[:, f]  # [C, d]
            for k in range(num_relations):
                w = r_c[:, k] * eta_c[:, f]  # [C]
                acc[:, f, k].index_add_(0, dst_c, w.unsqueeze(-1) * f_src_f)
                eff_mass[:, f, k].index_add_(0, dst_c, w)
    g_perm = acc / (eff_mass.unsqueeze(-1) + _EPS)  # [N, F, K, d]
    return g_perm, eff_mass


def reliability_regularization(
    edge_index: torch.Tensor,
    f_block: torch.Tensor,
    reliability: nn.Module,
    num_nodes: int,
    reg_type: str = "mean1",
    edge_chunk_size: int | None = None,
) -> torch.Tensor:
    """Regularization for the reliability gate (review option B: test whether
    the eta collapse is an unregularized overshoot).

    Reg types (both EXACTLY zero at eta == 1):
        mean1 : mean_{e,f} (eta - 1)^2          pulls eta back toward 1
        band  : mean_{e,f} max(0, 0.5-eta)^2 + max(0, eta-1.5)^2
                permissive inside the healthy (0.5, 1.5) band, penalizes
                collapse / overshoot only

    Chunked over edges; no [E, F] tensor is kept. Returns a scalar tensor
    (differentiable; callers add reg_weight * value to the aux loss).
    """
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    assert reg_type in ("mean1", "band"), f"unknown reg_type {reg_type!r}"

    acc = torch.zeros((), dtype=f_block.dtype, device=f_block.device)
    count = 0
    chunk = max(num_edges, 1) if edge_chunk_size is None else max(int(edge_chunk_size), 1)
    for start in range(0, num_edges, chunk):
        end = min(start + chunk, num_edges)
        eta_c = reliability(f_block[src[start:end]], f_block[dst[start:end]])  # [C, F]
        if reg_type == "mean1":
            acc = acc + (eta_c - 1.0).square().sum()
        else:  # band
            acc = acc + (0.5 - eta_c).clamp_min(0.0).square().sum()
            acc = acc + (eta_c - 1.5).clamp_min(0.0).square().sum()
        count += eta_c.numel()
    return acc / max(count, 1)


class FactorConditionedRelationCalibration(nn.Module):
    """R1-A2: factor-conditioned semantic relation calibration (plan §41,
    user-authorized amendment; A1 reliability is OFF in this mode).

        u_i^f = P_f f_i                      (per-factor d -> 32, bias=True)
        q_ij^f = MLP([u_i+u_j | |u_i-u_j| | u_i*u_j | cos(u_i,u_j)])   in R^K
        qhat   = q - mean_k q               (shift-invariant de-mean)
        delta  = tanh(qhat)                 bounded residual, in (-1, 1)
        r^f_ij,k = Softmax_k( log(r_str_ij,k + eps) + delta_ij,k^f )

    - shared MLP output = K; final layer zero-init => delta == 0 =>
      r^f == r^str at step 0 (mathematical equivalence; the log/softmax
      roundtrip is not bitwise — equivalence tests use allclose).
    - symmetric token => reverse edges get bitwise-identical r^f.
    - r^str is read-only: the structural prior / availability / capacity
      reference all keep the ORIGINAL r^str (audit Q6 discipline).
    """

    def __init__(
        self,
        num_factors: int = 3,
        factor_dim: int = 128,
        proj_dim: int = 32,
        hidden_dim: int = 64,
        num_relations: int = 4,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.num_factors = int(num_factors)
        self.num_relations = int(num_relations)
        self.proj_dim = int(proj_dim)
        self.projections = nn.ModuleList(
            [
                nn.Linear(int(factor_dim), int(proj_dim), bias=True)
                for _ in range(self.num_factors)
            ]
        )
        self.mlp = nn.Sequential(
            nn.Linear(3 * int(proj_dim) + 1, int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), int(num_relations)),
        )
        # Zero-init final layer: delta == 0 -> r^f == r^str at step 0.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def extra_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def forward(
        self,
        r_str: torch.Tensor,
        f_src: torch.Tensor,
        f_dst: torch.Tensor,
    ) -> torch.Tensor:
        """r_str: [C, K]; f_src / f_dst: [C, F, d] -> r_f [C, F, K]."""
        log_r = torch.log(r_str + _EPS)
        rf_list = []
        for f in range(self.num_factors):
            u_i = self.projections[f](f_src[:, f])  # [C, p]
            u_j = self.projections[f](f_dst[:, f])  # [C, p]
            cos = (u_i * u_j).sum(dim=-1) / (u_i.norm(dim=-1) * u_j.norm(dim=-1) + _EPS)
            token = torch.cat(
                [u_i + u_j, (u_i - u_j).abs(), u_i * u_j, cos.unsqueeze(-1)], dim=-1
            )  # [C, 3p+1]
            q = self.mlp(token)  # [C, K]
            qhat = q - q.mean(dim=-1, keepdim=True)
            delta = torch.tanh(qhat)
            rf_list.append(torch.softmax(log_r + delta, dim=-1))
        return torch.stack(rf_list, dim=1)  # [C, F, K]


def relation_calibrated_weighted_mean(
    edge_index: torch.Tensor,
    r_str: torch.Tensor,
    f_block: torch.Tensor,
    calibration: nn.Module,
    num_nodes: int,
    edge_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A2 per-factor calibrated relation-weighted context aggregation.

        g_ifk             = sum_j r_ji,k^f f_j / (sum_j r_ji,k^f + eps)
        eff_mass_ifk      = sum_j r_ji,k^f

    r^f is computed per edge chunk and consumed immediately; no [E, F, K]
    tensor is kept. At delta == 0 the result is mathematically equal to
    ``relation_weighted_mean`` (float grouping differs -> allclose only).
    Isolated nodes / empty graphs return zeros, no NaN.
    """
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    num_factors = int(f_block.size(1))
    num_relations = int(r_str.size(1))
    dim = int(f_block.size(2))
    dtype = f_block.dtype
    device = f_block.device

    acc = torch.zeros(num_nodes, num_factors, num_relations, dim, dtype=dtype, device=device)
    eff_mass = torch.zeros(num_nodes, num_factors, num_relations, dtype=dtype, device=device)

    chunk = max(num_edges, 1) if edge_chunk_size is None else max(int(edge_chunk_size), 1)
    for start in range(0, num_edges, chunk):
        end = min(start + chunk, num_edges)
        src_c, dst_c = src[start:end], dst[start:end]
        r_str_c = r_str[start:end]  # [C, K]
        f_src_c = f_block[src_c]  # [C, F, d]
        f_dst_c = f_block[dst_c]  # [C, F, d]
        rf_c = calibration(r_str_c, f_src_c, f_dst_c)  # [C, F, K]
        for f in range(num_factors):
            f_src_f = f_src_c[:, f]  # [C, d]
            for k in range(num_relations):
                w = rf_c[:, f, k]  # [C]
                acc[:, f, k].index_add_(0, dst_c, w.unsqueeze(-1) * f_src_f)
                eff_mass[:, f, k].index_add_(0, dst_c, w)
    g_perm = acc / (eff_mass.unsqueeze(-1) + _EPS)  # [N, F, K, d]
    return g_perm, eff_mass


@torch.no_grad()
def calibration_edge_statistics(
    edge_index: torch.Tensor,
    r_str: torch.Tensor,
    f_block: torch.Tensor,
    calibration: nn.Module,
    num_nodes: int,
    edge_chunk_size: int | None = None,
) -> dict:
    """A2 edge-level mechanism statistics (user-specified diagnostic list),
    one chunked pass. JSON-safe.

    Per factor f:
        js_str_f   : mean_e JS(r^f_e,f, r^str_e)   (how far the calibrated
                     posterior moved from the structural prior)
        kl_f2str / kl_str2f : mean KL(r^f || r^str) / KL(r^str || r^f)
        sim_{f,k}  : r^f-weighted semantic coherence; range over k
        entropy_f  : mean_e H(r^f_e,f);  k_eff_f = exp(entropy)
    Pairwise factor routing divergence:
        js_C_Pt / js_C_Pv / js_Pt_Pv : mean_e JS(r^f_e,f1, r^f_e,f2)
        (THE key A2 mechanism metric: do factors route differently?)
    """
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    num_factors = int(f_block.size(1))
    num_relations = int(r_str.size(1))
    device = f_block.device

    js_str = torch.zeros(num_factors, dtype=torch.float64, device=device)
    kl_f2s = torch.zeros(num_factors, dtype=torch.float64, device=device)
    kl_s2f = torch.zeros(num_factors, dtype=torch.float64, device=device)
    ent = torch.zeros(num_factors, dtype=torch.float64, device=device)
    js_pair = torch.zeros(3, dtype=torch.float64, device=device)
    sim_num = torch.zeros(num_factors, num_relations, dtype=torch.float64, device=device)
    sim_den = torch.zeros(num_factors, num_relations, dtype=torch.float64, device=device)

    def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return (p * torch.log((p + _EPS) / (q + _EPS))).sum(dim=-1)

    def _js(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        m = 0.5 * (p + q)
        return 0.5 * (_kl(p, m) + _kl(q, m))

    chunk = max(num_edges, 1) if edge_chunk_size is None else max(int(edge_chunk_size), 1)
    for start in range(0, num_edges, chunk):
        end = min(start + chunk, num_edges)
        src_c, dst_c = src[start:end], dst[start:end]
        r_str_c = r_str[start:end].double()  # [C, K]
        f_src_c = f_block[src_c]
        f_dst_c = f_block[dst_c]
        rf_c = calibration(r_str[start:end], f_src_c, f_dst_c).double()  # [C, F, K]
        for f in range(num_factors):
            js_str[f] += _js(rf_c[:, f], r_str_c).sum()
            kl_f2s[f] += _kl(rf_c[:, f], r_str_c).sum()
            kl_s2f[f] += _kl(r_str_c, rf_c[:, f]).sum()
            ent[f] += (-(rf_c[:, f] * torch.log(rf_c[:, f] + _EPS)).sum(dim=-1)).sum()
            a = f_src_c[:, f]
            b = f_dst_c[:, f]
            cos = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + _EPS)
            cos = cos.double()
            for k in range(num_relations):
                w = rf_c[:, f, k]
                sim_num[f, k] += (w * cos).sum()
                sim_den[f, k] += w.sum()
        js_pair[0] += _js(rf_c[:, 0], rf_c[:, 1]).sum()
        js_pair[1] += _js(rf_c[:, 0], rf_c[:, 2]).sum()
        js_pair[2] += _js(rf_c[:, 1], rf_c[:, 2]).sum()

    total = max(num_edges, 1)
    sim = sim_num / (sim_den + _EPS)
    return {
        "js_str": [float(js_str[f].item() / total) for f in range(num_factors)],
        "kl_f2str": [float(kl_f2s[f].item() / total) for f in range(num_factors)],
        "kl_str2f": [float(kl_s2f[f].item() / total) for f in range(num_factors)],
        "js_pairwise": {
            "C_Pt": float(js_pair[0].item() / total),
            "C_Pv": float(js_pair[1].item() / total),
            "Pt_Pv": float(js_pair[2].item() / total),
        },
        "semantic_coherence": sim.cpu().tolist(),
        "semantic_coherence_range": [float((sim[f].max() - sim[f].min()).item()) for f in range(num_factors)],
        "entropy": [float(ent[f].item() / total) for f in range(num_factors)],
        "k_eff": [float(torch.exp(ent[f] / total).item()) for f in range(num_factors)],
    }


class DynamicLocalScoreResidual(nn.Module):
    """R1-BL (plan §19, user-decoupled variant): node-adaptive Local score
    residual. Answers ONLY "how much Local should this node/factor keep" —
    it never touches the relation scores, so the conditional relation plan
    alpha = Softmax_k(s_rel/eps) is mathematically unchanged (user §7).

        s_if0 = z_f + delta_if0,   delta_if0 = MLP_0([f_i | g_bar_i^f |
                                      |f_i-g_bar_i^f| | f_i*g_bar_i^f])

    Final layer zero-init: delta == 0 exactly at step 0 -> the Local score
    is bitwise-equal to the parent's global z_f.
    """

    def __init__(
        self,
        factor_dim: int,
        hidden_dim: int = 64,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * int(factor_dim), int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def extra_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def forward(self, f: torch.Tensor, g_bar: torch.Tensor) -> torch.Tensor:
        """f / g_bar: [N, F, d] -> delta_local [N, F, 1]."""
        feat = torch.cat([f, g_bar, (f - g_bar).abs(), f * g_bar], dim=-1)
        return self.net(feat)


class SupportRelationScoreResidual(nn.Module):
    """R1-BR (plan §19, user-decoupled high-risk branch): support-aware
    relation score residual.

        s_ifk = s_base + delta_evidence([log1p(m_ik) | a_ik])

    Availability stays a FEATURE only (plan §20: never a hard capacity
    prior, never top-1 routing). Final layer zero-init: delta == 0 exactly
    at step 0 -> relation scores equal the parent's.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def extra_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def forward(
        self, log1p_mass: torch.Tensor, availability: torch.Tensor
    ) -> torch.Tensor:
        """log1p_mass / availability: [N, K] -> delta_evidence [N, K, 1]
        (factor-independent support evidence, broadcast over F)."""
        feat = torch.stack([log1p_mass, availability], dim=-1)
        return self.net(feat)


@torch.no_grad()
def reliability_edge_statistics(
    edge_index: torch.Tensor,
    r: torch.Tensor,
    f_block: torch.Tensor,
    reliability: nn.Module,
    num_nodes: int,
    edge_chunk_size: int | None = None,
) -> dict:
    """Best-checkpoint eta statistics (audit §10 + review §8), one chunked pass.

    Sole documented exception to the no-[E, F] discipline: eta_full [E, F]
    (<= 4.8 MB on ele-fashion) is materialized INSIDE this no-grad
    diagnostics function for exact quantiles. The training path never does.

    Returns JSON-safe dict:
        eta:      per-factor mean/std/CV/p10/p50/p90/frac_lt05/frac_gt15
                  (CV^f = std/mean — the weighted mean is invariant to a
                  uniform eta scale, so relative differentiation is the
                  signal that matters, review §8)
        neighbor: per-factor mean over nodes of within-neighborhood
                  std_{j in N(i)} eta_ji^f and its CV (deg > 0 mask)
        corr_eta_cos: per-factor Pearson corr(eta, semantic cosine)
        weighted_semantic_coherence: [F, K] Sim_{f,k} = sum r*eta*cos / sum r*eta
    """
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    num_factors = int(f_block.size(1))
    num_relations = int(r.size(1))
    device = f_block.device

    eta_chunks: list[torch.Tensor] = []
    # f64 accumulators (R0 convention): moments + cross-moments per factor.
    s_eta = torch.zeros(num_factors, dtype=torch.float64, device=device)
    s_eta2 = torch.zeros(num_factors, dtype=torch.float64, device=device)
    s_cos = torch.zeros(num_factors, dtype=torch.float64, device=device)
    s_cos2 = torch.zeros(num_factors, dtype=torch.float64, device=device)
    s_etacos = torch.zeros(num_factors, dtype=torch.float64, device=device)
    # per-node, per-factor eta moments for the neighbor-wise statistics.
    n_s1 = torch.zeros(num_nodes, num_factors, dtype=torch.float64, device=device)
    n_s2 = torch.zeros(num_nodes, num_factors, dtype=torch.float64, device=device)
    deg = torch.bincount(dst, minlength=num_nodes).to(torch.float64)
    # weighted semantic coherence per (f, k).
    sim_num = torch.zeros(num_factors, num_relations, dtype=torch.float64, device=device)
    sim_den = torch.zeros(num_factors, num_relations, dtype=torch.float64, device=device)

    chunk = max(num_edges, 1) if edge_chunk_size is None else max(int(edge_chunk_size), 1)
    for start in range(0, num_edges, chunk):
        end = min(start + chunk, num_edges)
        src_c, dst_c = src[start:end], dst[start:end]
        r_c = r[start:end].double()  # [C, K]
        f_src_c = f_block[src_c]  # [C, F, d]
        f_dst_c = f_block[dst_c]  # [C, F, d]
        eta_c = reliability(f_src_c, f_dst_c).double()  # [C, F]
        eta_chunks.append(eta_c.float())
        for f in range(num_factors):
            a = f_src_c[:, f]
            b = f_dst_c[:, f]
            cos = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + _EPS)
            cos = cos.double()
            eta_f = eta_c[:, f]
            s_eta[f] += eta_f.sum()
            s_eta2[f] += (eta_f * eta_f).sum()
            s_cos[f] += cos.sum()
            s_cos2[f] += (cos * cos).sum()
            s_etacos[f] += (eta_f * cos).sum()
            n_s1[:, f].index_add_(0, dst_c, eta_f)
            n_s2[:, f].index_add_(0, dst_c, eta_f * eta_f)
            for k in range(num_relations):
                w = r_c[:, k] * eta_f
                sim_num[f, k] += (w * cos).sum()
                sim_den[f, k] += w.sum()

    eta_full = torch.cat(eta_chunks, dim=0) if eta_chunks else torch.zeros(
        0, num_factors, dtype=torch.float32, device=device
    )  # [E, F] diagnostics-only
    quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32, device=device)

    eta_stats: dict[str, dict[str, float]] = {}
    neighbor_stats: dict[str, dict[str, float]] = {}
    corr_stats: dict[str, float] = {}
    total = max(num_edges, 1)
    for f in range(num_factors):
        mean_e = float((s_eta[f] / total).item())
        var_e = max(float((s_eta2[f] / total - mean_e * mean_e).item()), 0.0)
        std_e = var_e ** 0.5
        qs = torch.quantile(eta_full[:, f], quantiles) if eta_full.size(0) else torch.zeros(3)
        eta_stats[f"F{f + 1}"] = {
            "mean": mean_e,
            "std": std_e,
            "cv": std_e / (mean_e + 1e-8),
            "p10": float(qs[0].item()),
            "p50": float(qs[1].item()),
            "p90": float(qs[2].item()),
            "frac_lt_0.5": float((eta_full[:, f] < 0.5).float().mean().item()) if eta_full.size(0) else 0.0,
            "frac_gt_1.5": float((eta_full[:, f] > 1.5).float().mean().item()) if eta_full.size(0) else 0.0,
        }
        # neighbor-wise: per-node std of incoming eta. clamp_min(1.0) keeps
        # deg > 0 nodes exact (no eps drift: eta == 1 -> std exactly 0);
        # deg = 0 rows are masked out below.
        cnt = deg.clamp_min(1.0)
        node_mean = n_s1[:, f] / cnt
        node_var = (n_s2[:, f] / cnt - node_mean * node_mean).clamp_min(0.0)
        node_std = node_var.sqrt()
        node_cv = node_std / (node_mean + _EPS)
        mask = deg > 0
        neighbor_stats[f"F{f + 1}"] = {
            "neighbor_std_mean": float(node_std[mask].mean().item()) if bool(mask.any()) else 0.0,
            "neighbor_cv_mean": float(node_cv[mask].mean().item()) if bool(mask.any()) else 0.0,
        }
        # Pearson r(eta, cos): zero when either variance is ~0.
        mean_c = float((s_cos[f] / total).item())
        var_c = max(float((s_cos2[f] / total - mean_c * mean_c).item()), 0.0)
        cov = s_etacos[f] / total - mean_e * mean_c
        corr_stats[f"F{f + 1}"] = float((cov / ((std_e * (var_c ** 0.5)) + _EPS)).item())

    sim = sim_num / (sim_den + _EPS)
    return {
        "eta": eta_stats,
        "neighbor": neighbor_stats,
        "corr_eta_cos": corr_stats,
        "weighted_semantic_coherence": sim.cpu().tolist(),
    }
