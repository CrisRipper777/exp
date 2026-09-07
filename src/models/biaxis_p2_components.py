"""P2 Bi-Axis transport layer: Null-Augmented Factor-Relation Plan (plan §2).

Unifies P1's independent budget beta and selector alpha into one plan:

    Gamma_i in R_+^{F x (K+1)},  sum_{k=0}^K Gamma_i,f,k = 1
    column 0          = Local / No-Transport state
    columns 1..K      = latent structural relations

Derived quantities (plan §2):
    beta_i^f  = 1 - Gamma_i,f,0           (graph mass)
    alpha_ifk = Gamma_i,f,k / sum_{l>=1} Gamma_i,f,l   (conditional relation plan)

Discipline:
    - shared scorer: [f || g_k || f*g_k], NO availability inside the score
      (availability is the relation-side supply prior, not a compatibility
      feature; plan §6).
    - reference capacity nu is built from DETACHED availability (§10).
    - relation confidence q is DETACHED (§17).
    - log-domain generalized Sinkhorn, hard row marginal mu = 1_F, soft
      column KL towards nu; tau -> 0 degenerates to NullSoftmax exactly
      (theta = tau / (tau + eps)).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import get_activation

_EPS = 1e-8


class FactorRelationScore(nn.Module):
    """Shared relation compatibility scorer (plan §6):

        s_i,f,k = MLP([f_i || g_i,k^f || f_i * g_i,k^f])

    ONE shared network for all factors C/Pt/Pv and all relations R1..RK.
    Availability is deliberately NOT an input.
    """

    def __init__(self, factor_dim: int, hidden_dim: int = 64, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * int(factor_dim), int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """f: [N, F, d]; g: [N, F, K, d] -> scores [N, F, K].

        Scored one relation at a time: peak transient [N, F, 3d].
        """
        k = int(g.size(2))
        scores_list = []
        for rel in range(k):
            g_rel = g[:, :, rel]  # [N, F, d]
            feat = torch.cat([f, g_rel, f * g_rel], dim=-1)  # [N, F, 3d]
            scores_list.append(self.net(feat).squeeze(-1))  # [N, F]
        return torch.stack(scores_list, dim=-1)  # [N, F, K]


def build_augmented_scores(
    relation_scores: torch.Tensor,
    null_score: torch.Tensor,
) -> torch.Tensor:
    """S = [s_f,0 ; s_f,1:K] -> [N, F, K+1] (plan §8).

    null_score: [F] learnable scalars (plan §7: per-factor global thresholds).
    """
    num_nodes = int(relation_scores.size(0))
    null = null_score.to(relation_scores.dtype).reshape(1, -1, 1).expand(num_nodes, -1, 1)
    return torch.cat([null, relation_scores], dim=-1)


def build_reference_capacity(
    availability: torch.Tensor,
    num_factors: int,
    null_prior: float = 0.5,
    degree: torch.Tensor | None = None,
    detach: bool = True,
) -> torch.Tensor:
    """Relation-side reference capacity (plan §9):

        nu_i = F * [pi0, (1-pi0)*a_i1, ..., (1-pi0)*a_iK]     [N, K+1]

    detach=True (default, plan §10 stop-gradient): the capacity prior must
    not receive gradients from the relation module. detach=False exists only
    for the explicit config switch (review §17b) — default experiments keep
    the stop-gradient.
    Isolated nodes get pi0=1 (pure Local reference); they never run the
    transport anyway (fast path).
    """
    availability = torch.as_tensor(availability)
    if detach:
        availability = availability.detach()
    num_nodes, k = availability.shape
    ref = torch.cat(
        [
            torch.full((num_nodes, 1), float(null_prior), dtype=availability.dtype, device=availability.device),
            (1.0 - float(null_prior)) * availability,
        ],
        dim=-1,
    )  # [N, K+1]
    if degree is not None:
        isolated = torch.as_tensor(degree).le(0)
        if bool(isolated.any()):
            ref[isolated] = 0.0
            ref[isolated, 0] = 1.0
    return float(num_factors) * ref


def compute_node_relation_confidence(
    r: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    detach: bool = True,
) -> torch.Tensor:
    """Node-wise relation specialization confidence (plan §16):

        h_ji = -sum_k r_ji,k log(r_ji,k + eps)
        hbar_i = mean over incoming edges
        q_i = clamp(1 - hbar_i / log(K), 0, 1)

    Isolated nodes: q = 0. detach=True by default (plan §17 stop-gradient);
    False only for the explicit config switch (review §17b).
    """
    r = torch.as_tensor(r)
    num_relations = int(r.size(1))
    edge_entropy = -(r * torch.log(r + _EPS)).sum(dim=-1)  # [E]
    dst = edge_index[1]
    acc = torch.zeros(num_nodes, dtype=r.dtype, device=r.device)
    acc.index_add_(0, dst, edge_entropy)
    degree = torch.bincount(dst, minlength=num_nodes).to(r.dtype)
    hbar = acc / (degree + _EPS)
    q = (1.0 - hbar / (torch.log(torch.tensor(float(num_relations), dtype=r.dtype, device=r.device)))).clamp(0.0, 1.0)
    q = torch.where(degree <= 0, torch.zeros_like(q), q)
    if detach:
        q = q.detach()
    return q


def null_augmented_softmax(scores: torch.Tensor, epsilon: float) -> torch.Tensor:
    """P2-Variant 1 (plan §12): independent row softmax over {Local, R1..RK}.

    gamma = Softmax(S / epsilon); no inter-factor or capacity coupling.
    """
    return torch.softmax(scores / float(epsilon), dim=-1)


def semi_relaxed_transport(
    scores: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float,
    tau_base: float,
    sinkhorn_iters: int,
    theta_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """P2-Variant 2/3 (plan §13/§14): fixed or adaptive semi-relaxed UOT.

    Per node i (vectorized):
        Gamma_i = argmin <C, Gamma> - eps H(Gamma) + tau KL(Gamma^T 1 || nu_i)
                  s.t.  Gamma 1 = 1_F            (hard row marginal)
    with C = -S.

    Log-domain generalized Sinkhorn (plan §20):
        u <- mu / (K v)
        v <- (nu / (K^T u))^theta,   theta = tau / (tau + eps)

    - theta_override None -> fixed mode: theta = tau_base/(tau_base+eps) scalar.
    - theta_override [N,1] -> adaptive mode: theta_i = tau_i/(tau_i+eps)
      (tau_i = tau_base * q_i^R, plan §18).
    - theta_override [N,K+1] -> per-column theta (used by relation_uot:
      Local column theta=0 removes the Local-capacity constraint, review §19).
    Row constants are removed from S for numerical stability (absorbed by u;
    plan unchanged under hard row marginal).
    """
    num_nodes, num_factors, k_aug = scores.shape
    log_k = scores / float(epsilon)
    log_k = log_k - log_k.max(dim=-1, keepdim=True).values  # stability
    log_mu = torch.zeros(num_nodes, num_factors, 1, dtype=scores.dtype, device=scores.device)
    log_nu = torch.log(nu + _EPS)  # [N, K+1]
    log_v = torch.zeros(num_nodes, k_aug, dtype=scores.dtype, device=scores.device)

    if theta_override is None:
        theta = float(tau_base) / (float(tau_base) + float(epsilon))
    else:
        theta = theta_override.to(scores.dtype)  # [N, 1] or [N, K+1]

    for _ in range(int(sinkhorn_iters)):
        log_u = log_mu - torch.logsumexp(log_k + log_v[:, None, :], dim=-1, keepdim=True)  # [N,F,1]
        log_col = torch.logsumexp(log_k + log_u, dim=1)  # [N,K+1]
        log_v = theta * (log_nu - log_col)

    # Final u update so the row marginal holds exactly (plan §20).
    log_u = log_mu - torch.logsumexp(log_k + log_v[:, None, :], dim=-1, keepdim=True)
    log_gamma = log_u + log_k + log_v[:, None, :]
    return torch.exp(log_gamma)


# ---------------------------------------------------------------------------
# Deterministic aggregation (verification mode, 2026-09-02 reproducibility
# analysis): replaces GPU-atomic index_add accumulation with CSR SpMM /
# cumsum-based segment reductions. Results are bitwise reproducible across
# runs on the same GPU. Default training path is UNCHANGED (atomics are
# faster); this mode exists for determinism-verification sweeps only.
# ---------------------------------------------------------------------------


def _hillis_steele_scan(x: torch.Tensor) -> torch.Tensor:
    """Inclusive scan with an order FIXED BY CODE (Hillis-Steele doubling).

    Pure elementwise / slice / cat ops — no atomics, no cumsum (torch's CUDA
    cumsum is empirically cross-process unstable at 1e6-element float32:
    its inter-block prefix goes through atomic ordering, and float32 ulp at
    1e6 magnitude is 2^-4 — exactly the quanta observed). The summation
    TREE here is fixed, so results are bitwise reproducible."""
    y = x
    shift = 1
    num_rows, d = x.shape
    while shift < num_rows:
        shifted = torch.cat(
            [torch.zeros(shift, d, dtype=x.dtype, device=x.device), y[:-shift]], dim=0
        )
        y = y + shifted
        shift *= 2
    return y


def _csr_chunk_aggregate(
    dst: torch.Tensor,
    values: torch.Tensor,
    num_nodes: int,
    block_size: int = 2048,
) -> torch.Tensor:
    """Deterministic segment aggregation (sum of values grouped by dst).

    No atomics, no cumsum, no sparse.mm (all measured cross-process
    nondeterministic): argsort (stable) + per-block Hillis-Steele scan with
    a serial cross-block prefix — the entire accumulation order is fixed by
    code, bitwise reproducible across processes."""
    num_edges = int(dst.size(0))
    order = torch.argsort(dst, stable=True)
    s_dst = dst[order]
    s_vals = values[order]

    scanned_blocks: list[torch.Tensor] = []
    prefix = torch.zeros(s_vals.size(-1), dtype=s_vals.dtype, device=s_vals.device)
    for start in range(0, num_edges, block_size):
        end = min(start + block_size, num_edges)
        block = s_vals[start:end]
        scanned = _hillis_steele_scan(block) + prefix
        # Block total in the FIXED scan order (last row of the inclusive
        # scan) — avoids torch.sum, whose reduction may use atomics.
        prefix = scanned[-1].clone()
        scanned_blocks.append(scanned)
    cum = torch.cat(scanned_blocks, dim=0)  # [E, d] inclusive scan
    # cum_aug[i] = sum of the first i sorted rows (leading zero row) so that
    # segment sum [start, end) = cum_aug[end] - cum_aug[start].
    cum_aug = torch.cat(
        [torch.zeros(1, s_vals.size(-1), dtype=s_vals.dtype, device=s_vals.device), cum], dim=0
    )  # [E+1, d]

    change = torch.ones(num_edges, dtype=torch.bool, device=dst.device)
    change[1:] = s_dst[1:] != s_dst[:-1]
    starts = change.nonzero(as_tuple=True)[0]
    ends = torch.cat([starts[1:], torch.tensor([num_edges], device=dst.device)])
    sums = cum_aug[ends] - cum_aug[starts]  # [S, d]
    out = torch.zeros(num_nodes, s_vals.size(-1), dtype=s_vals.dtype, device=s_vals.device)
    out[s_dst[starts]] = sums  # unique-index scatter: one write per row
    return out


def deterministic_relation_weighted_mean(
    edge_index: torch.Tensor,
    r: torch.Tensor,
    features: torch.Tensor,
    num_nodes: int,
    edge_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic twin of ``relation_weighted_mean`` (same signature and
    return: g [N,K,d], mass [N,K]). Chunks are accumulated serially in fixed
    order, each chunk via CSR SpMM -> bitwise reproducible."""
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    k = int(r.size(1))
    dim = int(features.size(-1))
    dtype = features.dtype

    acc = torch.zeros(num_nodes, k, dim, dtype=dtype, device=features.device)
    mass = torch.zeros(num_nodes, k, dtype=r.dtype, device=r.device)

    chunk = num_edges if edge_chunk_size is None else int(edge_chunk_size)
    for start in range(0, num_edges, chunk):
        end = min(start + chunk, num_edges)
        src_c, dst_c = src[start:end], dst[start:end]
        r_c = r[start:end]
        mass += _csr_chunk_aggregate(dst_c, r_c, num_nodes)  # [N, K]
        features_src_c = features[src_c]  # hoisted per chunk (memory fix)
        for rel in range(k):
            weighted = r_c[:, rel].unsqueeze(-1) * features_src_c  # [E_c, d]
            acc[:, rel] += _csr_chunk_aggregate(dst_c, weighted, num_nodes)
    g = acc / (mass.unsqueeze(-1) + _EPS)
    return g, mass


def deterministic_node_relation_confidence(
    r: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    detach: bool = True,
) -> torch.Tensor:
    """Deterministic twin of ``compute_node_relation_confidence`` (CSR SpMM
    instead of index_add atomics; no cumsum — cumsum_cuda_kernel has no
    deterministic implementation in torch 2.4)."""
    r = torch.as_tensor(r)
    num_relations = int(r.size(1))
    edge_entropy = -(r * torch.log(r + _EPS)).sum(dim=-1)  # [E]
    dst = edge_index[1]
    acc = _csr_chunk_aggregate(dst, edge_entropy.unsqueeze(-1), num_nodes).squeeze(-1)  # [N]
    degree = torch.bincount(dst, minlength=num_nodes).to(r.dtype)
    hbar = acc / (degree + _EPS)
    q = (1.0 - hbar / (torch.log(torch.tensor(float(num_relations), dtype=r.dtype, device=r.device)))).clamp(0.0, 1.0)
    q = torch.where(degree <= 0, torch.zeros_like(q), q)
    if detach:
        q = q.detach()
    return q
