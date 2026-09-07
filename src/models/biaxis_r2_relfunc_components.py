"""R2-Design-2.8 v2 relational-function components
(docs/BiAxis_R2_Design_2_8_v2_Identifiable_Relational_Function_Decomposition.md).

Unified diagnostic formulation (v2 §1):

    m_i^b = sum_a lambda_i^{ab} r_i^{ab} sum_j pi_ji^{ab} Ohat_ji^{ab}(U_a F_j^a)

    lambda : simplex over source factor a           (Rule III)
    pi     : simplex over real neighbors only       (Rule II, no null token)
    r      : exposure scalar in [0,1]               (Rule I)
    Ohat   : content operator, NormMatch in the     (Rule IV)
             primary diagnostic

Identifiability (v2 §2): r / pi / lambda / operator routing coefficients are
dynamic outputs of shared predictor networks — never free node/edge tables.

Causal machinery (v2 §5):
    within_target_perm / shuffle_scores_within_target — exact integer-segment
        permutation (NO float composite key; seed 20260904).
    per_target_edge_mask — REMOVE_TOP/RANDOM/BOTTOM + KEEP_TOP inside each
        target's own real neighborhood; random removes the same per-target
        count as top/bottom.
    chunked_coupled_message — exact r*pi factorization of the old
        null-augmented softmax (v2 §5.3, COUPLED_EQUIV bridge).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_activation, make_norm

MISMATCH_PERM_SEED = 20260904
NORM_MATCH_EPS = 1e-6


# ---------------------------------------------------------------------------
# Exact segmented permutation (v2 §5.1)
# ---------------------------------------------------------------------------


def within_target_perm(edge_index: torch.Tensor,
                       num_edges: int | None = None,
                       seed: int = MISMATCH_PERM_SEED) -> torch.Tensor:
    """For every edge e, perm[e] is another edge of the SAME target dst, such
    that within each target's segment e -> perm[e] is a deterministic local
    permutation — guaranteed NON-IDENTITY for every target with degree > 1.
    Exact integer arithmetic only: key = (dst << 32) | r32.
    The old float composite key (dst.float() * 1e7 + tie_break) is forbidden
    here (v2 §5.1): float32 quantization collapsed distinct targets on large
    graphs and silently degenerated to near-identity permutations.
    Sparse graphs are dominated by degree-2 targets, where a uniform local
    permutation is the identity with probability 1/2; without a fix the v2
    >=95% non-identity-targets requirement would be mathematically
    unreachable. Identity segments get a deterministic cyclic shift."""
    dst = edge_index[1]
    num_edges = int(edge_index.size(1)) if num_edges is None else int(num_edges)
    generator = torch.Generator().manual_seed(int(seed))
    # torch.Generator is CPU-only: generate on CPU, then move (D2.7 pitfall)
    r = torch.randint(0, 2 ** 32, (num_edges,), generator=generator,
                      dtype=torch.int64).to(dst.device)
    key = (dst.to(torch.int64) << 32) | r
    ord_dst = torch.argsort(dst.to(torch.int64), stable=True)
    ord_key = torch.argsort(key, stable=True)
    inv_dst = torch.empty(num_edges, dtype=torch.long, device=dst.device)
    inv_dst[ord_dst] = torch.arange(num_edges, device=dst.device)
    # edge e sits at dst-grouped position k = inv_dst[e]; its permuted partner
    # is the edge at key-grouped position k (same target segment).
    perm = ord_key[inv_dst]
    # non-identity guarantee for degree>1 targets
    n_nodes = int(dst.max().item()) + 1 if num_edges else 1
    is_fixed = perm == torch.arange(num_edges, device=dst.device)
    seg_fixed = torch.zeros(n_nodes, dtype=torch.float32, device=dst.device)
    seg_fixed = seg_fixed.scatter_reduce(
        0, dst, is_fixed.float(), reduce="amin", include_self=False)
    deg = torch.bincount(dst, minlength=n_nodes)
    all_fixed = (seg_fixed > 0.5) & (deg > 1)
    starts = _segment_starts_by_value(dst[ord_dst], n_nodes)
    k = inv_dst
    shifted = starts[dst] + ((k - starts[dst] + 1) % deg[dst])
    perm_shift = ord_dst[shifted]
    return torch.where(all_fixed[dst], perm_shift, perm)


def shuffle_scores_within_target(scores: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 seed: int = MISMATCH_PERM_SEED) -> torch.Tensor:
    """s_perm[e] = scores[perm[e]]: preserves the per-target multiset of
    scores exactly, destroys score-to-neighbor correspondence (v2 §5.1)."""
    perm = within_target_perm(edge_index, int(scores.numel()), seed)
    return scores[perm]


def _segment_starts_by_value(seg_ids: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """seg_ids must be grouped (equal values contiguous). Returns
    starts_by_value[value] = first position of that value's block."""
    n = int(seg_ids.numel())
    is_start = torch.empty(n, dtype=torch.bool, device=seg_ids.device)
    is_start[0] = True
    is_start[1:] = seg_ids[1:] != seg_ids[:-1]
    start_k = torch.nonzero(is_start, as_tuple=False).squeeze(-1)
    starts = torch.zeros(num_nodes, dtype=torch.long, device=seg_ids.device)
    starts[seg_ids[start_k]] = start_k
    return starts


def per_target_rank(scores: torch.Tensor, edge_index: torch.Tensor,
                    num_nodes: int, kind: str,
                    seed: int = MISMATCH_PERM_SEED) -> torch.Tensor:
    """rank[e] in [0, deg_i) within edge e's own target segment.
    kind="score": ascending score (ties broken by edge order, stable).
    kind="random": deterministic local random order (exact integer key)."""
    dst = edge_index[1].to(torch.int64)
    num_edges = int(scores.numel())
    if kind == "score":
        # rank by score WITHIN target without any float composite key:
        # 1) global stable argsort by score -> pos[e] = score-order position;
        # 2) group those positions by dst (contiguous blocks);
        # 3) within-block index = per-target ascending score rank.
        ord_s = torch.argsort(scores, stable=True)
        pos = torch.empty(num_edges, dtype=torch.long, device=dst.device)
        pos[ord_s] = torch.arange(num_edges, device=dst.device)
        dst_s = dst[ord_s]
        ord_g = torch.argsort(dst_s, stable=True)
        starts = _segment_starts_by_value(dst_s[ord_g], num_nodes)
        inv_g = torch.empty(num_edges, dtype=torch.long, device=dst.device)
        inv_g[ord_g] = torch.arange(num_edges, device=dst.device)
        k_of_p = inv_g[pos]  # grouped-listing position of edge e's score position
        return k_of_p - starts[dst]
    if kind == "random":
        generator = torch.Generator().manual_seed(int(seed))
        # torch.Generator is CPU-only: generate on CPU, then move
        r = torch.randint(0, 2 ** 32, (num_edges,), generator=generator,
                          dtype=torch.int64).to(dst.device)
        key = (dst << 32) | r
        ord_k = torch.argsort(key, stable=True)
        starts = _segment_starts_by_value(dst[ord_k], num_nodes)
        inv_k = torch.empty(num_edges, dtype=torch.long, device=dst.device)
        inv_k[ord_k] = torch.arange(num_edges, device=dst.device)
        return inv_k - starts[dst]
    raise ValueError(kind)


def per_target_edge_mask(scores: torch.Tensor, edge_index: torch.Tensor,
                         num_nodes: int, op: str, pct: float,
                         seed: int = MISMATCH_PERM_SEED) -> torch.Tensor:
    """[E] bool keep-mask. Each target independently has the requested
    fraction of ITS OWN real neighbors selected (v2 §5.2). Random and
    top/bottom select identical per-target counts (floor(deg*pct))."""
    dst = edge_index[1]
    deg = torch.bincount(dst, minlength=num_nodes)
    n_sel = (deg.to(torch.float64) * float(pct)).to(torch.int64)
    if op in ("remove_top", "remove_bottom", "keep_top"):
        rank = per_target_rank(scores, edge_index, num_nodes, "score", seed)
    elif op == "remove_random":
        rank = per_target_rank(scores, edge_index, num_nodes, "random", seed)
    else:
        raise ValueError(op)
    keep_above = deg[dst] - n_sel[dst]
    if op == "remove_top":     # drop the highest n_sel scores
        return rank < keep_above
    if op == "remove_bottom":  # drop the lowest n_sel scores
        return rank >= n_sel[dst]
    if op == "remove_random":  # drop n_sel random neighbors
        return rank >= n_sel[dst]
    return rank >= keep_above  # keep_top


def validate_shuffle(scores: torch.Tensor, scores_perm: torch.Tensor,
                     edge_index: torch.Tensor, num_nodes: int,
                     tol: float = 1e-9) -> dict:
    """v2 §5.1 mandatory checks. Returns JSON-safe stats:
      frac_score_changed: among degree>1 edges, fraction whose score changed
      frac_nonidentity_targets: among degree>1 targets, fraction with any change
      sums_preserved: per-target sums equal within tolerance (no cross-target)
      histogram_exact: per-target sorted-histogram equality (small graphs only)
    Mandatory thresholds: frac_score_changed >= 0.80 and
    frac_nonidentity_targets >= 0.95; a no-op shuffle must fail."""
    dst = edge_index[1]
    deg = torch.bincount(dst, minlength=num_nodes)
    deg1 = deg > 1
    changed = (scores_perm - scores).abs() > tol
    any_changed = torch.zeros(num_nodes, dtype=torch.float32, device=scores.device)
    any_changed = any_changed.scatter_reduce(
        0, dst, changed.float(), reduce="amax", include_self=False)
    any_changed = any_changed > 0.5
    n_deg1_edges = int(changed[deg1[dst]].sum().item())
    den_deg1_edges = max(int(deg1[dst].sum().item()), 1)
    frac_changed = n_deg1_edges / den_deg1_edges
    frac_nonid = float(any_changed[deg1].float().mean().item()) if deg1.any() else 1.0
    # necessary cross-target conditions: per-target sum preserved
    ssum = torch.zeros(num_nodes, dtype=torch.float64, device=scores.device)
    ssum = ssum.scatter_add(0, dst, scores.to(torch.float64))
    ssum_p = torch.zeros(num_nodes, dtype=torch.float64, device=scores.device)
    ssum_p = ssum_p.scatter_add(0, dst, scores_perm.to(torch.float64))
    sums_preserved = bool(torch.allclose(ssum, ssum_p, atol=1e-3, rtol=1e-5))
    histogram_exact = None
    if num_nodes <= 2048 and scores.numel() <= 200000:
        ok = True
        for i in range(num_nodes):
            m = dst == i
            if m.sum() <= 1:
                continue
            if not torch.allclose(torch.sort(scores[m])[0],
                                  torch.sort(scores_perm[m])[0],
                                  atol=1e-5, rtol=1e-5):
                ok = False
                break
        histogram_exact = bool(ok)
    return {"frac_score_changed": frac_changed,
            "frac_nonidentity_targets": frac_nonid,
            "sums_preserved": sums_preserved,
            "histogram_exact": histogram_exact}


# ---------------------------------------------------------------------------
# COUPLED_EQUIV: exact factorization of the old null-augmented softmax (§5.3)
# ---------------------------------------------------------------------------


def chunked_coupled_message(
    f_block: torch.Tensor,  # [N, 3, d] (dtype/device source)
    edge_index: torch.Tensor,  # [2, E]
    num_nodes: int,
    scores: torch.Tensor,  # [E] same edge logits as the old model
    null_scores: torch.Tensor,  # [N] same null logits as the old model
    payload_a: torch.Tensor,  # [N, d]
    edge_chunk_size: int = 50000,
    edge_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """m_i = sum_j r_i * pi_ji * payload_j with (v2 §5.3):

        Z_i   = sum_j exp(s_ji)
        r_i   = Z_i / (exp(s_null_i) + Z_i)
        pi_ji = exp(s_ji) / Z_i

    Same edge/null logits as the old null-augmented softmax, but the exposure
    r and the composition pi are computed explicitly and multiplied. Must
    reproduce the old message to < 1e-6."""
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    if edge_mask is not None:
        src, dst = src[edge_mask], dst[edge_mask]
        scores = scores[edge_mask]
        num_edges = int(src.size(0))
    m = torch.zeros(num_nodes, payload_a.size(-1), dtype=f_block.dtype,
                    device=f_block.device)
    if num_edges == 0:
        return m
    # pass 1: per-target max over null AND real neighbors
    max_i = null_scores.clone()
    seg = torch.zeros(num_nodes, dtype=scores.dtype, device=scores.device)
    seg = seg.scatter_reduce(0, dst, scores, reduce="amax", include_self=False)
    max_i = torch.maximum(max_i, seg)
    # pass 2: Z_i = sum_j exp(s - max); denom = exp(null - max) + Z
    z = torch.zeros(num_nodes, dtype=scores.dtype, device=scores.device)
    for start in range(0, num_edges, edge_chunk_size):
        end = min(start + edge_chunk_size, num_edges)
        d_c = dst[start:end]
        z = z.scatter_add(0, d_c, torch.exp(scores[start:end] - max_i[d_c]))
    denom = torch.exp(null_scores - max_i) + z
    r_i = z / denom
    # pass 3: alpha_j = r_i * pi_j ; m += alpha * payload
    for start in range(0, num_edges, edge_chunk_size):
        end = min(start + edge_chunk_size, num_edges)
        s_c, d_c = src[start:end], dst[start:end]
        pi = torch.exp(scores[start:end] - max_i[d_c]) / z[d_c]
        contrib = (r_i[d_c] * pi).unsqueeze(-1) * payload_a[s_c]
        m = m.scatter_add(0, d_c.unsqueeze(-1).expand_as(contrib), contrib)
    return m


# ---------------------------------------------------------------------------
# NormMatch (v2 Rule IV)
# ---------------------------------------------------------------------------


def norm_match(v_out: torch.Tensor, v_ref: torch.Tensor,
               eps: float = NORM_MATCH_EPS) -> torch.Tensor:
    """Ohat(v) = v_tilde / (||v_tilde|| + eps) * ||v_ref|| — the operator may
    change feature DIRECTION/CONTENT but not magnitude; r stays the explicit
    graph-amplitude variable (v2 Rule IV)."""
    n_out = v_out.norm(dim=-1, keepdim=True)
    n_ref = v_ref.norm(dim=-1, keepdim=True)
    return v_out / (n_out + eps) * n_ref


# ---------------------------------------------------------------------------
# Exposure predictors (v2 §7)
# ---------------------------------------------------------------------------


def exposure_param_count(in_dim: int, hidden: int) -> int:
    return int(in_dim * hidden + hidden + hidden * 1 + 1)


def solve_exposure_width(in_dim: int, d: int, target_params: int) -> int:
    """Hidden width so ExposureNet(in_dim, w) params match target_params
    (capacity-matched granularity comparison, v2 §7)."""
    w = int(round((target_params - 1) / (in_dim + 2)))
    return max(w, 8)


class ExposureNet(nn.Module):
    """r predictor (v2 §7): Linear(in, h) -> LN -> GELU -> Linear(h, 1) ->
    sigmoid. Shared learnable function; never a per-node table (v2 §2)."""

    def __init__(self, in_dim: int, hidden: int, activation: str = "gelu",
                 norm: str = "layernorm") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            make_norm(norm, int(hidden)),
            get_activation(activation),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: [*, in] -> r [*] in (0, 1)."""
        return torch.sigmoid(self.net(u).squeeze(-1))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Composition scorers (v2 §8; softmax over real neighbors only)
# ---------------------------------------------------------------------------


class CompScorer(nn.Module):
    """Shared psi: Linear(in, 2d) -> LN -> GELU -> Dropout -> Linear(2d, d)
    -> GELU -> Linear(d, 1). in = 4d+t (target/source) or 4d+2t (pair)."""

    def __init__(self, in_dim: int, factor_dim: int, dropout: float = 0.1,
                 activation: str = "gelu", norm: str = "layernorm") -> None:
        super().__init__()
        d = int(factor_dim)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), 2 * d),
            make_norm(norm, 2 * d),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * d, d),
            get_activation(activation),
            nn.Linear(d, 1),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: [*, in] -> score [*]."""
        return self.net(u).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GenericCompScorer(nn.Module):
    """C1: one edge distribution shared across semantic factors (v2 §8).
    s_ji = MLP([z0_i, z0_j, z0_i*z0_j, |z0_i-z0_j|]) from the local ownership
    projection z0. Width is solved by the model so that (local_proj + scorer)
    match the pair scorer params within +/-5% (D2.7 precedent)."""

    def __init__(self, factor_dim: int, width: int, dropout: float = 0.1,
                 activation: str = "gelu", norm: str = "layernorm") -> None:
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


def solve_generic_comp_width(d: int, pair_scorer_params: int) -> int:
    """Width so (local_proj 3d->d) + GenericCompScorer(4d->w->1) params match
    the pair scorer params within +/-5%."""
    base = 3 * d * d + d
    w = int(round((pair_scorer_params - base - 1) / (4 * d + 5)))
    return max(w, d)


# ---------------------------------------------------------------------------
# Source-channel integration (v2 §9)
# ---------------------------------------------------------------------------


class SourceScalarMix(nn.Module):
    """M1: lambda_i^{ab} = Softmax_a(g_lambda(F_i^b, e_a)) with
    g: Linear(d+t, d) -> LN -> GELU -> Linear(d, 1). The lambda is a simplex
    over source factors a, so it cannot become an unconstrained second
    exposure gate (v2 Rule III)."""

    def __init__(self, factor_dim: int, type_dim: int = 16,
                 activation: str = "gelu", norm: str = "layernorm") -> None:
        super().__init__()
        d, t = int(factor_dim), int(type_dim)
        self.net = nn.Sequential(
            nn.Linear(d + t, d),
            make_norm(norm, d),
            get_activation(activation),
            nn.Linear(d, 1),
        )

    def forward(self, f_target: torch.Tensor, e_embs: torch.Tensor) -> torch.Tensor:
        """f_target [N, d]; e_embs [3, t] -> lambda [N, 3] simplex over a."""
        logits = []
        for a in range(3):
            u = torch.cat([f_target,
                           e_embs[a].unsqueeze(0).expand(f_target.size(0), -1)],
                          dim=-1)
            logits.append(self.net(u).squeeze(-1))
        return F.softmax(torch.stack(logits, dim=-1), dim=-1)


class ConcatMixer(nn.Module):
    """M2 per target factor b (v2 §9): concat[m_Cb|m_Ptb|m_Pvb] (3d) ->
    Linear(3d,2d) -> LN -> GELU -> Dropout(.1) -> Linear(2d,d).
    MEAN_DUP control feeds [mean|mean|mean] to the same architecture."""

    def __init__(self, factor_dim: int, dropout: float = 0.1,
                 activation: str = "gelu", norm: str = "layernorm") -> None:
        super().__init__()
        d = int(factor_dim)
        self.mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * d, 2 * d),
                make_norm(norm, 2 * d),
                get_activation(activation),
                nn.Dropout(float(dropout)),
                nn.Linear(2 * d, d),
            )
            for _ in range(3)  # per target factor b
        ])

    def forward(self, x: torch.Tensor, b: int) -> torch.Tensor:
        """x [N, 3d] -> [N, d]."""
        return self.mixers[b](x)


class _PreLNAttnBlock(nn.Module):
    def __init__(self, d: int, num_heads: int = 4, dropout: float = 0.1,
                 ffn_mult: int = 4) -> None:
        super().__init__()
        assert d % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        # hand-rolled MHA (nn.MultiheadAttention's internal SDPA dispatch
        # raises "invalid configuration argument" at ele-fashion batch size
        # 97K x 4 tokens; explicit matmuls keep full control)
        self.qkv = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.drop_attn = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d, d),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        b, l, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        w = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1)
        w = self.drop_attn(w)
        h = (w @ v).transpose(1, 2).reshape(b, l, d)
        x = x + self.drop1(self.out_proj(h))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x, (w.mean(dim=1) if return_attn else None)


class SourceAttnMixer(nn.Module):
    """M3 (v2 §9): three source-channel tokens, target factor F_i^b as query,
    2 Pre-LN blocks, 4 heads, FFN 4d, dropout .1, one d-dim output message.
    MEAN_DUP control feeds the mean message as all three tokens."""

    def __init__(self, factor_dim: int, num_blocks: int = 2,
                 num_heads: int = 4, ffn_mult: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        d = int(factor_dim)
        self.blocks = nn.ModuleList([
            _PreLNAttnBlock(d, num_heads, dropout, ffn_mult)
            for _ in range(num_blocks)])
        self.ln_out = nn.LayerNorm(d)

    def forward(self, query: torch.Tensor, tokens: torch.Tensor,
                return_attn: bool = False) -> torch.Tensor:
        """query [N, d]; tokens [N, 3, d] -> [N, d]."""
        x = torch.cat([query.unsqueeze(1), tokens], dim=1)  # [N, 4, d]
        attn = None
        for i, blk in enumerate(self.blocks):
            x, w = blk(x, return_attn=return_attn and i == len(self.blocks) - 1)
            if w is not None:
                attn = w
        out = self.ln_out(x[:, 0])
        return (out, attn) if return_attn else out


# ---------------------------------------------------------------------------
# Functional operators (v2 §10); zero/small-init so step 0 == O0
# ---------------------------------------------------------------------------


def _zero_final(seq: nn.Sequential) -> None:
    nn.init.zeros_(seq[-1].weight)
    nn.init.zeros_(seq[-1].bias)


class StaticPairResidual(nn.Module):
    """O1: Otilde_ab(v) = v + DeltaT_ab(v), 9 pair transforms
    Linear(d,2d) -> LN -> GELU -> Linear(2d,d), final layer zero-init so
    step 0 equals O0 exactly (v2 §10)."""

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
        for t in self.transforms:
            _zero_final(t)

    def forward(self, v: torch.Tensor, a: int, b: int) -> torch.Tensor:
        return v + self.transforms[3 * a + b](v)


class TargetFilm(nn.Module):
    """O2: [Delta_gamma_i^{ab}, beta_i^{ab}] = phi(F_i^b, e_a, e_b);
    Otilde = (1+Delta_gamma) * v + beta. phi:
    Linear(d+2t, 2d) -> LN -> GELU -> Linear(2d, 2d), final zero-init so
    step 0 equals O0 exactly (v2 §10)."""

    def __init__(self, factor_dim: int, type_dim: int = 16,
                 activation: str = "gelu", norm: str = "layernorm") -> None:
        super().__init__()
        d, t = int(factor_dim), int(type_dim)
        self.net = nn.Sequential(
            nn.Linear(d + 2 * t, 2 * d),
            make_norm(norm, 2 * d),
            get_activation(activation),
            nn.Linear(2 * d, 2 * d),
        )
        _zero_final(self.net)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: [*, d+2t] -> [*, 2d] = [gamma | beta]."""
        return self.net(u)


class EdgeFilm(nn.Module):
    """O3: [Delta_gamma_ji^{ab}, beta_ji^{ab}] = phi(F_i^b, F_j^a, product,
    |diff|, e_a, e_b). Same architecture as TargetFilm with 4d+2t input;
    final zero-init so step 0 equals O0 exactly. Chunked by the caller
    (v2 §10)."""

    def __init__(self, factor_dim: int, type_dim: int = 16,
                 activation: str = "gelu", norm: str = "layernorm") -> None:
        super().__init__()
        d, t = int(factor_dim), int(type_dim)
        self.net = nn.Sequential(
            nn.Linear(4 * d + 2 * t, 2 * d),
            make_norm(norm, 2 * d),
            get_activation(activation),
            nn.Linear(2 * d, 2 * d),
        )
        _zero_final(self.net)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: [*, 4d+2t] -> [*, 2d] = [gamma | beta]."""
        return self.net(u)


class BasisOperator(nn.Module):
    """O4 (v2 §10): K residual basis operators B_k (Linear(d,d,bias=False),
    small-init) and router q = Softmax_k(rho(F_i^b, F_j^a, e_a, e_b));
    Otilde = v + sum_k q_k B_k(v).
    Controls: uniform_router (q = 1/K), target_router (rho conditioned on
    F_i^b, e_a, e_b only)."""

    def __init__(self, factor_dim: int, type_dim: int = 16, k: int = 4,
                 uniform_router: bool = False, target_router: bool = False,
                 activation: str = "gelu", norm: str = "layernorm",
                 init_scale: float = 0.001) -> None:
        super().__init__()
        d, t = int(factor_dim), int(type_dim)
        self.k = int(k)
        self.uniform_router = bool(uniform_router)
        self.target_router = bool(target_router)
        self.bases = nn.ModuleList([nn.Linear(d, d, bias=False)
                                    for _ in range(self.k)])
        for base in self.bases:
            nn.init.normal_(base.weight, std=float(init_scale))
        in_dim = d + 2 * t if self.target_router else 2 * d + 2 * t
        self.router = nn.Sequential(
            nn.Linear(in_dim, d),
            make_norm(norm, d),
            get_activation(activation),
            nn.Linear(d, self.k),
        )

    def router_logits(self, u: torch.Tensor) -> torch.Tensor:
        """u: [*, in] -> [*, K]."""
        return self.router(u)

    def apply(self, v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """v [C, d]; q [C, K] -> v + sum_k q_k B_k(v) [C, d]."""
        out = v
        for k in range(self.k):
            out = out + q[:, k:k + 1] * self.bases[k](v)
        return out
