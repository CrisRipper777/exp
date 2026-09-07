"""R2D29 CORT components
(docs/BiAxis_R2D29_System_Level_Performance_Advancement_Plan.md, §3-§4).

CORT = Coordinated Ownership-Relational Transfer. One CORT block implements
the full computation pathway (plan §3):

    Relational Allocation -> Source Preservation -> Target-conditioned
    Interaction -> Ownership-state Update

Plan §4 formulation (factors a=source, b=target, H = {C, Pt, Pv}):

    s_ji^{a->b}  = psi(H_i^b, H_j^a, H_i^b*H_j^a, |H_i^b-H_j^a|, e_a, e_b)
    {alpha_0, alpha_ji}^{ab} = Softmax_{{0} u N(i)}({s_0, s_ji}^{ab})
    m_i^{a->b}   = sum_j alpha_ji^{ab} U_a H_j^a          (source channel)
    h_i^{ab}     = phi([H_i^b | m_i^{ab} | H_i^b*m_i^{ab} | |H_i^b-m_i^{ab}| | e_a | e_b])
    Delta_i^b    = Phi_b([h_i^{Cb} | h_i^{Pt b} | h_i^{Pv b}])
    H~_i^b       = LN(H_i^b + rho_b Delta_i^b)             (factor write-back)

The coupled null-augmented softmax (message mass x neighbor ranking in one
softmax over {null} u N(i)) is the D2.7-positive formulation; the message
math below is bit-identical to biaxis_r2_relfunc_components.chunked_coupled_message
(D2.8 v2 §5.3 COUPLED_EQUIV), extended to also return routing statistics.

Router modes (plan §6.2):
    uniform    : fixed 1/deg_i neighbor mean, no null state
    pair_null  : one null-augmented softmax per (a, b) pair (9)
    target_null: per-target-b scoring on the neighbor factor-mean summary;
                 source identity kept in the payload U_a H_j^a only

Source modes:
    mean            : m_i^b = (1/3) sum_a m_i^{a->b}
    preserve_concat : per-pair phi + per-target Phi over the 3 channels
    preserve_attn   : per-target attention over the 3 source tokens
    mean_dup flag   : matched control — the mean message duplicated into the
                      3 channels, then the exact preserve_concat path

Memory discipline: per-pair edge scores are computed in chunks and, when
memory_checkpoint is on, the score segment runs under activation
checkpointing (D2.7 pitfall: pair-composite scores computed outside
checkpoint OOM ele-fashion). The [E]-level score buffer is transient per
pair; only [N, d]-level messages are retained for the interaction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .biaxis_r2_relfunc_components import SourceAttnMixer  # reuse D2.8 v2 M3
from .common import get_activation, make_norm

CORT_EPS = 1e-8
NUM_FACTORS = 3


# ---------------------------------------------------------------------------
# Coupled null-augmented message with routing stats
# ---------------------------------------------------------------------------


def cort_coupled_message(
    f_block: torch.Tensor,  # [N, 3, d] (dtype/device source)
    edge_index: torch.Tensor,  # [2, E]
    num_nodes: int,
    scores: torch.Tensor,  # [E] edge logits
    null_scores: torch.Tensor,  # [N] null logits
    payload_a: torch.Tensor,  # [N, d]
    edge_chunk_size: int = 50000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """m_i = sum_j alpha_ji * payload_j with the coupled softmax over
    {null} u N(i) (D2.8 v2 §5.3 COUPLED_EQUIV numerics):

        Z_i   = sum_j exp(s_ji)
        r_i   = Z_i / (exp(s_null_i) + Z_i)          (graph mass)
        pi_ji = exp(s_ji) / Z_i
        m_i   = r_i * sum_j pi_ji * payload_j

    Returns (m, null_mass, entropy) with null_mass = 1 - r_i and entropy the
    full per-target distribution entropy over {null} u N(i) (NaN-free for
    isolated nodes: they have no edges, r=0, null_mass=1, entropy=0)."""
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    m = torch.zeros(num_nodes, payload_a.size(-1), dtype=f_block.dtype, device=f_block.device)
    null_mass = torch.ones(num_nodes, dtype=scores.dtype, device=scores.device)
    entropy = torch.zeros(num_nodes, dtype=scores.dtype, device=scores.device)
    if num_edges == 0:
        return m, null_mass, entropy
    # pass 1: per-target max over null AND real neighbors
    max_i = null_scores.clone()
    seg = torch.zeros(num_nodes, dtype=scores.dtype, device=scores.device)
    seg = seg.scatter_reduce(0, dst, scores, reduce="amax", include_self=False)
    max_i = torch.maximum(max_i, seg)
    # pass 2: Z_i, denom, per-target entropy over the real neighbors
    z = torch.zeros(num_nodes, dtype=scores.dtype, device=scores.device)
    for start in range(0, num_edges, edge_chunk_size):
        end = min(start + edge_chunk_size, num_edges)
        d_c = dst[start:end]
        z = z.scatter_add(0, d_c, torch.exp(scores[start:end] - max_i[d_c]))
    denom = torch.exp(null_scores - max_i) + z
    null_mass = torch.exp(null_scores - max_i) / denom
    # null contribution to entropy: -p0 log p0 (0 when null_mass == 0)
    ent = -null_mass * torch.log(null_mass.clamp_min(CORT_EPS))
    for start in range(0, num_edges, edge_chunk_size):
        end = min(start + edge_chunk_size, num_edges)
        d_c = dst[start:end]
        p = torch.exp(scores[start:end] - max_i[d_c]) / denom[d_c]
        ent = ent.scatter_add(0, d_c, -p * torch.log(p.clamp_min(CORT_EPS)))
    entropy = ent
    # pass 3: alpha_j = r_i * pi_j ; m += alpha * payload
    for start in range(0, num_edges, edge_chunk_size):
        end = min(start + edge_chunk_size, num_edges)
        s_c, d_c = src[start:end], dst[start:end]
        pi = torch.exp(scores[start:end] - max_i[d_c]) / z[d_c]
        contrib = ((1.0 - null_mass[d_c]) * pi).unsqueeze(-1) * payload_a[s_c]
        m = m.scatter_add(0, d_c.unsqueeze(-1).expand_as(contrib), contrib)
    return m, null_mass, entropy


# ---------------------------------------------------------------------------
# Factor type embeddings (e_a / e_b, plan §4.1)
# ---------------------------------------------------------------------------


class FactorTypeEmbedding(nn.Module):
    """Learnable per-factor type tokens e_C / e_Pt / e_Pv, [3, t]."""

    def __init__(self, type_dim: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(NUM_FACTORS, int(type_dim))

    def forward(self) -> torch.Tensor:
        return self.emb.weight  # [3, t]


# ---------------------------------------------------------------------------
# Relational allocation scorers
# ---------------------------------------------------------------------------


class CortPairScorer(nn.Module):
    """psi_ab (plan §4.1): shared-per-pair allocation scorer.
    Input [H^b | H^a | H^b*H^a | |H^b-H^a| | e_a | e_b] = 4d+2t -> score."""

    def __init__(self, factor_dim: int, type_dim: int = 16, dropout: float = 0.2,
                 activation: str = "gelu", norm: str = "layernorm") -> None:
        super().__init__()
        d, t = int(factor_dim), int(type_dim)
        self.net = nn.Sequential(
            nn.Linear(4 * d + 2 * t, 2 * d),
            make_norm(norm, 2 * d),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * d, 1),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u).squeeze(-1)


class CortTargetScorer(nn.Module):
    """target_null scorer (plan §8.2): real-edge ranking decided by the
    target factor b against the neighbor's factor-mean summary mbar_j:
    [H^b | mbar | H^b*mbar | |H^b-mbar| | e_b] = 4d+t -> score."""

    def __init__(self, factor_dim: int, type_dim: int = 16, dropout: float = 0.2,
                 activation: str = "gelu", norm: str = "layernorm") -> None:
        super().__init__()
        d, t = int(factor_dim), int(type_dim)
        self.net = nn.Sequential(
            nn.Linear(4 * d + t, 2 * d),
            make_norm(norm, 2 * d),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * d, 1),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u).squeeze(-1)


def _chunk_pair_scores(
    scorer: nn.Module,
    f_pre: torch.Tensor,  # [N, 3, d]
    edge_index: torch.Tensor,  # [2, E]
    e_embs: torch.Tensor,  # [3, t]
    a: int,
    b: int,
    edge_chunk_size: int,
) -> torch.Tensor:
    """Chunked per-edge allocation scores s_ji^{a->b} (plan §4.1)."""
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    out = torch.empty(num_edges, dtype=f_pre.dtype, device=f_pre.device)
    if num_edges == 0:
        return out
    fa = f_pre[:, a]
    fb = f_pre[:, b]
    ea = e_embs[a].unsqueeze(0)
    eb = e_embs[b].unsqueeze(0)
    for start in range(0, num_edges, edge_chunk_size):
        end = min(start + edge_chunk_size, num_edges)
        s_c, d_c = src[start:end], dst[start:end]
        fb_d = fb[d_c]
        fa_s = fa[s_c]
        u = torch.cat(
            [fb_d, fa_s, fb_d * fa_s, (fb_d - fa_s).abs(),
             ea.expand(end - start, -1), eb.expand(end - start, -1)],
            dim=-1,
        )
        out[start:end] = scorer(u)
    return out


def _chunk_target_scores(
    scorer: nn.Module,
    f_pre: torch.Tensor,  # [N, 3, d]
    edge_index: torch.Tensor,  # [2, E]
    e_embs: torch.Tensor,  # [3, t]
    b: int,
    edge_chunk_size: int,
) -> torch.Tensor:
    """target_null chunked scores: target factor vs neighbor factor-mean."""
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    out = torch.empty(num_edges, dtype=f_pre.dtype, device=f_pre.device)
    if num_edges == 0:
        return out
    fb = f_pre[:, b]
    fbar = f_pre.mean(dim=1)
    eb = e_embs[b].unsqueeze(0)
    for start in range(0, num_edges, edge_chunk_size):
        end = min(start + edge_chunk_size, num_edges)
        s_c, d_c = src[start:end], dst[start:end]
        fb_d = fb[d_c]
        fbar_s = fbar[s_c]
        u = torch.cat(
            [fb_d, fbar_s, fb_d * fbar_s, (fb_d - fbar_s).abs(),
             eb.expand(end - start, -1)],
            dim=-1,
        )
        out[start:end] = scorer(u)
    return out


class CortRouter(nn.Module):
    """Relational Allocation (plan §4.1) + Source Preservation (plan §4.2).

    forward(f_pre [N,3,d], edge_index, num_nodes) ->
        msgs: dict[(a,b)] -> [N, d]  (or {b: [N,d]} for source_mode="mean"),
        stats: {null_mass_<a><b> / entropy_<a><b> / ...}
    """

    def __init__(self, factor_dim: int, router_mode: str, source_mode: str,
                 type_emb: FactorTypeEmbedding, type_dim: int = 16,
                 dropout: float = 0.2, activation: str = "gelu",
                 norm: str = "layernorm", edge_chunk_size: int = 50000,
                 memory_checkpoint: bool = True, mean_dup: bool = False) -> None:
        super().__init__()
        d = int(factor_dim)
        self.factor_dim = d
        self.router_mode = str(router_mode)
        self.source_mode = str(source_mode)
        self.mean_dup = bool(mean_dup)
        self.edge_chunk_size = int(edge_chunk_size)
        self.memory_checkpoint = bool(memory_checkpoint)
        self.type_emb = type_emb

        # per-source payload projectors U_a (plan §4.1)
        self.payload = nn.ModuleList([nn.Linear(d, d) for _ in range(NUM_FACTORS)])

        if self.router_mode == "pair_null":
            # 9 scorers (a, b) row-major + 9 null thresholds (plan §4.1)
            self.scorers = nn.ModuleList([
                CortPairScorer(d, type_dim, dropout, activation, norm)
                for _ in range(NUM_FACTORS * NUM_FACTORS)
            ])
            self.null_score = nn.Parameter(torch.zeros(NUM_FACTORS, NUM_FACTORS))
        elif self.router_mode == "target_null":
            self.scorers = nn.ModuleList([
                CortTargetScorer(d, type_dim, dropout, activation, norm)
                for _ in range(NUM_FACTORS)
            ])
            self.null_score = nn.Parameter(torch.zeros(NUM_FACTORS))
        elif self.router_mode != "uniform":
            raise AssertionError(f"unknown router_mode {self.router_mode!r}")

    def _pair_scores(self, f_pre: torch.Tensor, edge_index: torch.Tensor,
                     e_embs: torch.Tensor, a: int, b: int) -> torch.Tensor:
        return _chunk_pair_scores(
            self.scorers[NUM_FACTORS * a + b], f_pre, edge_index, e_embs,
            a, b, self.edge_chunk_size,
        )

    def _target_scores(self, f_pre: torch.Tensor, edge_index: torch.Tensor,
                       e_embs: torch.Tensor, b: int) -> torch.Tensor:
        return _chunk_target_scores(
            self.scorers[b], f_pre, edge_index, e_embs, b, self.edge_chunk_size,
        )

    def _message(self, f_pre: torch.Tensor, edge_index: torch.Tensor,
                 num_nodes: int, scores: torch.Tensor, null: torch.Tensor,
                 payload: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return cort_coupled_message(
            f_pre, edge_index, num_nodes, scores, null, payload,
            edge_chunk_size=self.edge_chunk_size,
        )

    def forward(self, f_pre: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int) -> tuple[dict, dict]:
        e_embs = self.type_emb()  # [3, t]
        # payloads: U_a f^a per source factor
        payloads = [self.payload[a](f_pre[:, a]) for a in range(NUM_FACTORS)]
        msgs: dict = {}
        stats: dict = {}

        if self.router_mode == "uniform":
            # fixed neighbor mean per source payload; no null state (plan §6.2)
            for a in range(NUM_FACTORS):
                m_a = self._uniform_mean(payloads[a], edge_index, num_nodes)
                for b in range(NUM_FACTORS):
                    msgs[(a, b)] = m_a
            deg = torch.bincount(edge_index[1], minlength=num_nodes).to(torch.float32)
            stats["entropy_mean"] = float(
                torch.log(deg + 1.0).mean().item()
            )
            stats["null_mass_mean"] = 0.0
        elif self.router_mode == "pair_null":
            for a in range(NUM_FACTORS):
                for b in range(NUM_FACTORS):
                    if self.memory_checkpoint and torch.is_grad_enabled() and self.training:
                        scores = torch.utils.checkpoint.checkpoint(
                            self._pair_scores, f_pre, edge_index, e_embs,
                            a, b, use_reentrant=False,
                        )
                    else:
                        scores = self._pair_scores(f_pre, edge_index, e_embs, a, b)
                    null = self.null_score[a, b].expand(num_nodes)
                    m, null_mass, entropy = self._message(
                        f_pre, edge_index, num_nodes, scores, null, payloads[a]
                    )
                    msgs[(a, b)] = m
                    stats[f"null_mass_{a}{b}"] = float(null_mass.mean().item())
                    stats[f"entropy_{a}{b}"] = float(entropy.mean().item())
        else:  # target_null
            for b in range(NUM_FACTORS):
                if self.memory_checkpoint and torch.is_grad_enabled() and self.training:
                    scores = torch.utils.checkpoint.checkpoint(
                        self._target_scores, f_pre, edge_index, e_embs,
                        b, use_reentrant=False,
                    )
                else:
                    scores = self._target_scores(f_pre, edge_index, e_embs, b)
                null = self.null_score[b].expand(num_nodes)
                for a in range(NUM_FACTORS):
                    m, null_mass, entropy = self._message(
                        f_pre, edge_index, num_nodes, scores, null, payloads[a]
                    )
                    msgs[(a, b)] = m
                stats[f"null_mass_{b}"] = float(null_mass.mean().item())
                stats[f"entropy_{b}"] = float(entropy.mean().item())

        if self.mean_dup:
            # matched control (plan §7.5): mean duplicated into 3 channels,
            # then the exact preserve_concat path
            for b in range(NUM_FACTORS):
                mean_b = sum(msgs[(a, b)] for a in range(NUM_FACTORS)) / NUM_FACTORS
                for a in range(NUM_FACTORS):
                    msgs[(a, b)] = mean_b
        elif self.source_mode == "mean":
            msgs = {
                b: sum(msgs[(a, b)] for a in range(NUM_FACTORS)) / NUM_FACTORS
                for b in range(NUM_FACTORS)
            }
        return msgs, stats

    def _uniform_mean(self, payload: torch.Tensor, edge_index: torch.Tensor,
                      num_nodes: int) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        num_edges = int(edge_index.size(1))
        deg = torch.bincount(dst, minlength=num_nodes).to(payload.dtype)
        acc = torch.zeros(num_nodes, payload.size(-1), dtype=payload.dtype, device=payload.device)
        for start in range(0, num_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, num_edges)
            acc.index_add_(0, dst[start:end], payload[src[start:end]])
        return acc / (deg.unsqueeze(-1) + CORT_EPS)


# ---------------------------------------------------------------------------
# Target-conditioned interaction (plan §4.3)
# ---------------------------------------------------------------------------


class _PhiNet(nn.Module):
    """Phi_b: 2-layer MLP (plan §4.3) — per target factor."""

    def __init__(self, in_dim: int, factor_dim: int, hidden_mult: float,
                 dropout: float = 0.2, activation: str = "gelu") -> None:
        super().__init__()
        h = max(int(hidden_mult * factor_dim), factor_dim)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), h),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(h, int(factor_dim)),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class _PhiPairNet(nn.Module):
    """phi_ab: 2-layer MLP + GELU + LayerNorm + Dropout (plan §4.3)."""

    def __init__(self, in_dim: int, factor_dim: int, hidden_mult: float,
                 dropout: float = 0.2, activation: str = "gelu",
                 norm: str = "layernorm") -> None:
        super().__init__()
        h = max(int(hidden_mult * factor_dim), factor_dim)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), h),
            make_norm(norm, h),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(h, int(factor_dim)),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class CortInteraction(nn.Module):
    """Target-conditioned vector interaction (plan §4.3): per-pair phi_ab over
    [H^b | m^ab | H^b*m^ab | |H^b-m^ab| | e_a | e_b], then per-target
    Phi_b over the three source channels -> Delta_i^b [N, d].

    source_mode:
        preserve_concat : phi_ab x 9 + Phi_b x 3
        preserve_attn   : per-b SourceAttnMixer over the 3 source tokens
        mean            : phi_b x 3 (4d input) + Phi_b x 3
    """

    def __init__(self, factor_dim: int, source_mode: str, type_dim: int = 16,
                 interaction_hidden_mult: float = 2.0, dropout: float = 0.2,
                 activation: str = "gelu", norm: str = "layernorm") -> None:
        super().__init__()
        d = int(factor_dim)
        t = int(type_dim)
        self.source_mode = str(source_mode)
        self.factor_dim = d
        self.interaction_hidden_mult = float(interaction_hidden_mult)

        if self.source_mode == "preserve_concat":
            self.phis = nn.ModuleList([
                _PhiPairNet(4 * d + 2 * t, d, interaction_hidden_mult,
                            dropout, activation, norm)
                for _ in range(NUM_FACTORS * NUM_FACTORS)
            ])
            self.phi_out = nn.ModuleList([
                _PhiNet(3 * d, d, interaction_hidden_mult, dropout, activation)
                for _ in range(NUM_FACTORS)
            ])
        elif self.source_mode == "preserve_attn":
            self.attns = nn.ModuleList([
                SourceAttnMixer(d, num_blocks=2, num_heads=4, ffn_mult=4, dropout=dropout)
                for _ in range(NUM_FACTORS)
            ])
        elif self.source_mode == "mean":
            self.phis = nn.ModuleList([
                _PhiPairNet(4 * d, d, interaction_hidden_mult, dropout, activation, norm)
                for _ in range(NUM_FACTORS)
            ])
            self.phi_out = nn.ModuleList([
                _PhiNet(d, d, interaction_hidden_mult, dropout, activation)
                for _ in range(NUM_FACTORS)
            ])
        else:
            raise AssertionError(f"unknown source_mode {self.source_mode!r}")

    def forward(self, f_pre: torch.Tensor, msgs: dict,
                e_embs: torch.Tensor) -> torch.Tensor:
        """f_pre [N,3,d]; msgs per (a,b) (or {b:} for mean); -> deltas [N,3,d]."""
        num_nodes = int(f_pre.size(0))
        d = self.factor_dim
        deltas = f_pre.new_zeros(num_nodes, NUM_FACTORS, d)

        if self.source_mode == "preserve_concat":
            for b in range(NUM_FACTORS):
                hb = f_pre[:, b]
                hs = []
                for a in range(NUM_FACTORS):
                    m = msgs[(a, b)]
                    u = torch.cat(
                        [hb, m, hb * m, (hb - m).abs(),
                         e_embs[a].expand(num_nodes, -1), e_embs[b].expand(num_nodes, -1)],
                        dim=-1,
                    )
                    hs.append(self.phis[NUM_FACTORS * a + b](u))
                deltas[:, b] = self.phi_out[b](torch.cat(hs, dim=-1))
        elif self.source_mode == "preserve_attn":
            for b in range(NUM_FACTORS):
                tokens = torch.stack(
                    [msgs[(0, b)], msgs[(1, b)], msgs[(2, b)]], dim=1
                )
                deltas[:, b] = self.attns[b](query=f_pre[:, b], tokens=tokens)
        else:  # mean
            for b in range(NUM_FACTORS):
                hb = f_pre[:, b]
                m = msgs[b]
                u = torch.cat([hb, m, hb * m, (hb - m).abs()], dim=-1)
                deltas[:, b] = self.phi_out[b](self.phis[b](u))
        return deltas


# ---------------------------------------------------------------------------
# Factor-space write-back (plan §4.4)
# ---------------------------------------------------------------------------


class CortWriteback(nn.Module):
    """H~_i^b = LN_b(H_i^b + rho_b Delta_i^b) with ReZero-style per-factor
    rho init residual_init (0 or small positive, plan §4.4)."""

    def __init__(self, factor_dim: int, residual_init: float = 0.0,
                 norm: str = "layernorm") -> None:
        super().__init__()
        d = int(factor_dim)
        self.rhos = nn.Parameter(torch.full((NUM_FACTORS,), float(residual_init)))
        self.norms = nn.ModuleList([make_norm(norm, d) for _ in range(NUM_FACTORS)])

    def forward(self, f_in: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        out = []
        for b in range(NUM_FACTORS):
            out.append(self.norms[b](f_in[:, b] + self.rhos[b] * deltas[:, b]))
        return torch.stack(out, dim=1)


# ---------------------------------------------------------------------------
# CORT block (plan §4: the full computation pathway)
# ---------------------------------------------------------------------------


class CortBlock(nn.Module):
    """One CORT block: Routing -> Source Preservation -> Target Interaction
    -> Write-back. ``writeback`` off means the block returns f_in unchanged
    and only the deltas are kept (writeback_mode=late; the caller projects
    them into z-space)."""

    def __init__(self, factor_dim: int, type_emb: FactorTypeEmbedding,
                 router_mode: str = "pair_null", source_mode: str = "preserve_concat",
                 writeback: bool = True, type_dim: int = 16,
                 interaction_hidden_mult: float = 2.0, residual_init: float = 0.0,
                 pre_norm: bool = True, dropout: float = 0.2,
                 activation: str = "gelu", norm: str = "layernorm",
                 edge_chunk_size: int = 50000, memory_checkpoint: bool = True,
                 mean_dup: bool = False) -> None:
        super().__init__()
        d = int(factor_dim)
        self.pre_norm = bool(pre_norm)
        self.writeback = bool(writeback)
        self.pre_norms = (
            nn.ModuleList([make_norm(norm, d) for _ in range(NUM_FACTORS)])
            if self.pre_norm else None
        )
        self.router = CortRouter(
            d, router_mode, source_mode, type_emb, type_dim, dropout, activation,
            norm, edge_chunk_size, memory_checkpoint, mean_dup,
        )
        self.type_emb = type_emb
        self.interaction = CortInteraction(
            d, source_mode, type_dim, interaction_hidden_mult, dropout,
            activation, norm,
        )
        self.writeback_mod = (
            CortWriteback(d, residual_init, norm) if self.writeback else None
        )
        self.type_dim = int(type_dim)

    def _pre_norm_apply(self, f_in: torch.Tensor) -> torch.Tensor:
        if self.pre_norms is None:
            return f_in
        return torch.stack(
            [self.pre_norms[b](f_in[:, b]) for b in range(NUM_FACTORS)], dim=1
        )

    def forward(self, f_in: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        f_pre = self._pre_norm_apply(f_in)
        msgs, rstats = self.router(f_pre, edge_index, num_nodes)
        deltas = self.interaction(f_pre, msgs, self.type_emb())
        stats = dict(rstats)
        if self.writeback and self.writeback_mod is not None:
            f_out = self.writeback_mod(f_in, deltas)
        else:
            f_out = f_in
        f_norm = f_in.norm(dim=-1, keepdim=True).clamp_min(CORT_EPS)
        ratio = (deltas.norm(dim=-1, keepdim=True) / f_norm).squeeze(-1)  # [N,3]
        for b in range(NUM_FACTORS):
            stats[f"delta_ratio_{b}"] = float(ratio[:, b].mean().item())
        if self.writeback_mod is not None:
            for b in range(NUM_FACTORS):
                stats[f"rho_{b}"] = float(self.writeback_mod.rhos[b].item())
        return f_out, deltas, stats


# ---------------------------------------------------------------------------
# Ownership Interaction Fusion (plan §4.5)
# ---------------------------------------------------------------------------


class OifFusion(nn.Module):
    """q = [C,Pt,Pv,C*Pt,C*Pv,Pt*Pv,|C-Pt|,|C-Pv|,|Pt-Pv|] (9d) -> MLP -> z."""

    def __init__(self, factor_dim: int, hidden_dim: int,
                 fusion_hidden_mult: float = 2.0, dropout: float = 0.2,
                 activation: str = "gelu", norm: str = "layernorm") -> None:
        super().__init__()
        d, hd = int(factor_dim), int(hidden_dim)
        hf = max(int(fusion_hidden_mult * hd), hd)
        self.net = nn.Sequential(
            nn.Linear(9 * d, hf),
            make_norm(norm, hf),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(hf, hd),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        c, pt, pv = f[:, 0], f[:, 1], f[:, 2]
        q = torch.cat(
            [c, pt, pv, c * pt, c * pv, pt * pv,
             (c - pt).abs(), (c - pv).abs(), (pt - pv).abs()],
            dim=-1,
        )
        return self.net(q)


class FactorAttnFusion(nn.Module):
    """factor_attn fusion (plan §4.5 / G4 axis): 3 factor tokens through
    hand-rolled pre-LN attention blocks, mean-pooled, projected to z.
    (nn.MultiheadAttention is avoided: D2.8 pitfall — its internal SDPA
    dispatch raises at ele-fashion batch sizes.)"""

    def __init__(self, factor_dim: int, hidden_dim: int, num_blocks: int = 2,
                 num_heads: int = 4, ffn_mult: int = 4, dropout: float = 0.1,
                 activation: str = "gelu") -> None:
        super().__init__()
        d, hd = int(factor_dim), int(hidden_dim)
        from .biaxis_r2_relfunc_components import _PreLNAttnBlock
        self.blocks = nn.ModuleList([
            _PreLNAttnBlock(d, num_heads, dropout, ffn_mult)
            for _ in range(int(num_blocks))
        ])
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, hd),
            get_activation(activation),
            nn.Dropout(float(dropout)),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        x = f  # [N, 3, d]
        for blk in self.blocks:
            x, _w = blk(x)
        z = self.ln_out(x.mean(dim=1))
        return self.head(z)


class CortMerger(nn.Module):
    """Hybrid factor-space merger (plan §6.3): H^b = H^b_A0 + G_b([H^b_A0,
    H^b_CORT]) with zero-initialized per-factor G_b, so step 0 == the A0
    path exactly (no bare concat before the classifier)."""

    def __init__(self, factor_dim: int) -> None:
        super().__init__()
        d = int(factor_dim)
        self.gs = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(NUM_FACTORS)])
        for g in self.gs:
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)

    def forward(self, f_a0: torch.Tensor, f_cort: torch.Tensor) -> torch.Tensor:
        out = []
        for b in range(NUM_FACTORS):
            out.append(
                f_a0[:, b]
                + self.gs[b](torch.cat([f_a0[:, b], f_cort[:, b]], dim=-1))
            )
        return torch.stack(out, dim=1)
