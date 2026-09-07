"""R3 Ownership-Structured Semantic Transition components
(docs/R3_Ownership_Structured_Transition_阶段推进计划.md, §2-§7).

Ownership-Structured Semantic Transition Layer (plan §3/§4/§5/§6/§7):

    H: [N, 3, d] ownership states (factor order: 0=C, 1=Pt, 2=Pv)

    S_i       = phi_s([C_i | Pt_i | Pv_i])              (same-node context,
                                                          conditioning ONLY,
                                                          plan §3.1)
    v_j^a     = V_a H_j^a ;  q_i^b = Q_b H_i^b           (relational space,
                                                          plan §3.2)
    diagonal : m_ji^{b->b} = D_b^diag(W_b^diag v_j^b)   (static, plan §3.3)
    off-diag : r_ji^{ab}    = phi_r([q_i^b | v_j^a | S_i | e_a | e_b])
               m_ji^{a->b}  = eps_l * D_b^cross( OP_ab(v_j^a, r_ji^{ab}) )
               OP_ab in {static linear, FiLM, shared low-rank dynamic basis}
               (plan §3.4/§4; eps_l = offdiag init scale, plan §4.4)
    mbar_i^{a->b} = mean_{j in N(i)} m_ji^{a->b}        (pre-aggregation
                                                          conditional compute,
                                                          plan §5)
    Delta H_i^b = U_b( LN( [H_i^b | mbar^{C->b} | mbar^{Pt->b} | mbar^{Pv->b}] ) )
    H_i^{b,l+1} = H_i^{b,l} + eta_l * Delta H_i^b       (pre-LN residual,
                                                          NO post-LN,
                                                          plan §7/§16.7)

Memory discipline (plan §17): every edge-level tensor is computed inside
chunks; the per-chunk message segments run under activation checkpointing
(memory_checkpoint=true, the p3/CORT ele-fashion fix): their [E, d]-level
intermediates are recomputed in backward instead of retained. The segment
is randomness-free (no dropout inside the router / context / basis — see
below), so the recomputation is bitwise identical (no dropout consumed in
the segment). The [E, d_r, d_r] matrix is never materialized; basis
transforms are low-rank z_r = A_r(act(C_r(v))), mixed by the softmax router
over functional bases (NOT over neighbors, plan §4.3).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .common import get_activation, make_norm

# torch 2.4's non-reentrant checkpoint internally calls the deprecated
# torch.cpu.amp.autocast(...) constructor on every invocation; the kwargs are
# empty unless CPU autocast is enabled (we never enable it), so the
# FutureWarning is purely cosmetic. Silencing it here, scoped to
# torch.utils.checkpoint, keeps training logs clean (same pattern as
# biaxis_p3.py).
warnings.filterwarnings(
    "ignore",
    message=".*torch.cpu.amp.autocast.*deprecated.*",
    category=FutureWarning,
    module=r"torch\.utils\.checkpoint",
)

FACTOR_NAMES = ("c", "pt", "pv")
NUM_FACTORS = 3
OFFDIAG_PAIRS = tuple((a, b) for a in range(NUM_FACTORS) for b in range(NUM_FACTORS) if a != b)
R3_EPS = 1.0e-8


class OwnershipTransitionLayer(nn.Module):
    """One Ownership-Structured Semantic Transition layer (plan §3)."""

    def __init__(
        self,
        factor_dim: int,
        relation_dim: int,
        factor_id_dim: int,
        context_dim: int,
        transition_mode: str,
        cross_factor: bool,
        use_dual_space: bool,
        use_same_node_context: bool,
        preserve_source_channels: bool,
        num_bases: int,
        basis_rank: int,
        router_hidden_dim: int,
        offdiag_init_scale: float,
        layer_scale_init: float,
        edge_chunk_size: int,
        dropout: float,
        activation: str,
        norm: str,
        memory_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        d = int(factor_dim)
        self.factor_dim = d
        self.transition_mode = str(transition_mode)
        assert self.transition_mode in ("diagonal", "static", "film", "basis"), (
            f"transition_mode must be diagonal|static|film|basis, got {self.transition_mode!r}"
        )
        self.cross_factor = bool(cross_factor)
        self.use_dual_space = bool(use_dual_space)
        self.use_same_node_context = bool(use_same_node_context)
        self.preserve_source_channels = bool(preserve_source_channels)
        self.offdiag_enabled = self.cross_factor and self.transition_mode != "diagonal"
        self.edge_chunk_size = max(int(edge_chunk_size), 1)
        self.memory_checkpoint = bool(memory_checkpoint)
        self.d_r = int(relation_dim) if self.use_dual_space else d
        self.activation = get_activation(activation)

        # ---- scales -------------------------------------------------------
        # layer_scale_init == 0 -> FROZEN zero scale: the layer short-circuits
        # to exact identity (unit-test mode, plan §16.7). Real configs must
        # keep it > 0 (plan §16.6: no zero-init starvation). Same convention
        # for the off-diagonal scale (plan §4.4: small-but-nonzero).
        if float(layer_scale_init) == 0.0:
            self.layer_scale = None
        else:
            self.layer_scale = nn.Parameter(torch.tensor(float(layer_scale_init)))
        if float(offdiag_init_scale) == 0.0:
            self.offdiag_scale = None
        else:
            self.offdiag_scale = nn.Parameter(torch.tensor(float(offdiag_init_scale)))

        # ---- same-node context (conditioning ONLY, plan §3.1) -------------
        # NOTE: no dropout here — the context feeds the checkpointed message
        # segment, which must be randomness-free (see module docstring).
        self.context_dim = int(context_dim) if self.use_same_node_context else 0
        if self.use_same_node_context:
            self.context_mlp = nn.Sequential(
                nn.Linear(NUM_FACTORS * d, self.context_dim),
                make_norm(norm, self.context_dim),
                get_activation(activation),
                nn.Linear(self.context_dim, self.context_dim),
            )

        # ---- semantic -> relational projections (plan §3.2) ---------------
        if self.use_dual_space:
            self.src_proj = nn.ModuleList([nn.Linear(d, self.d_r) for _ in range(NUM_FACTORS)])
            self.tgt_proj = nn.ModuleList([nn.Linear(d, self.d_r) for _ in range(NUM_FACTORS)])

        # factor identity embeddings e_a / e_b (plan §3.4)
        self.factor_id_dim = int(factor_id_dim)
        self.factor_emb = nn.Embedding(NUM_FACTORS, self.factor_id_dim)

        # ---- diagonal block: ownership-preserving static propagation ------
        # (plan §3.3). W_b^diag (d_r->d_r) composed with D_b^diag (d_r->d)
        # collapses to one linear map; kept as a single Linear per factor.
        self.diag = nn.ModuleList([nn.Linear(self.d_r, d) for _ in range(NUM_FACTORS)])

        # ---- off-diagonal block (plan §3.4/§4) ----------------------------
        if self.offdiag_enabled:
            # target-specific decode back into the target ownership space
            self.target_decode = nn.ModuleList([nn.Linear(self.d_r, d) for _ in range(NUM_FACTORS)])
            if self.transition_mode == "static":
                # one static structured transformation per (a,b) channel; no
                # descriptor / router (the transform does not depend on the
                # edge or the source-target state).
                self.cross_static = nn.ModuleList(
                    [nn.Linear(self.d_r, self.d_r) for _ in OFFDIAG_PAIRS]
                )
                self._pair_idx = {pair: idx for idx, pair in enumerate(OFFDIAG_PAIRS)}
            elif self.transition_mode == "film":
                # feature-wise affine modulation conditioned on the descriptor
                self.router = self._build_router(2 * self.d_r, router_hidden_dim, activation, norm)
            elif self.transition_mode == "basis":
                # shared low-rank functional bases (plan §4.3):
                # B_r(x) = A_r(act(C_r(x))), shared across ALL off-diagonal
                # (a,b) pairs; source/target identity enters via the router
                # descriptor only (plan §16: no 6/9 independent networks).
                self.num_bases = int(num_bases)
                self.basis_down = nn.ModuleList(
                    [nn.Linear(self.d_r, int(basis_rank)) for _ in range(self.num_bases)]
                )
                self.basis_up = nn.ModuleList(
                    [nn.Linear(int(basis_rank), self.d_r) for _ in range(self.num_bases)]
                )
                self.router = self._build_router(self.num_bases, router_hidden_dim, activation, norm)

        # ---- target update (plan §6/§7): pre-LN, no post-LN ---------------
        if self.offdiag_enabled and self.preserve_source_channels:
            update_in = (NUM_FACTORS + 1) * d  # [H_b | C->b | Pt->b | Pv->b]
        else:
            update_in = 2 * d  # [H_b | mbar^{b->b}] or mean-merged channels
        self.update_ln = nn.ModuleList([nn.LayerNorm(update_in) for _ in range(NUM_FACTORS)])
        self.update = nn.ModuleList([nn.Linear(update_in, d) for _ in range(NUM_FACTORS)])

    def _build_router(
        self,
        out_dim: int,
        router_hidden_dim: int,
        activation: str,
        norm: str,
    ) -> nn.Module:
        # NOTE: no dropout — the router runs inside the checkpointed message
        # segment, which must be deterministic for numerics-preserving
        # recomputation (same reasoning as the p3 memory fix).
        in_dim = 2 * self.d_r + self.context_dim + 2 * self.factor_id_dim
        return nn.Sequential(
            nn.Linear(in_dim, int(router_hidden_dim)),
            make_norm(norm, int(router_hidden_dim)),
            get_activation(activation),
            nn.Linear(int(router_hidden_dim), out_dim),
        )

    # ------------------------------------------------------------------
    # Per-chunk message segments (checkpointable, randomness-free)
    # ------------------------------------------------------------------

    def _diag_chunk(
        self,
        b: int,
        v0: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
        s_c: torch.Tensor,
    ) -> torch.Tensor:
        v_s = (v0, v1, v2)[b][s_c]
        return self.diag[b](v_s)

    def _offdiag_chunk(
        self,
        a: int,
        b: int,
        v0: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
        q0: torch.Tensor,
        q1: torch.Tensor,
        q2: torch.Tensor,
        S: torch.Tensor | None,
        e_embs: torch.Tensor,
        s_c: torch.Tensor,
        d_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One (a,b) off-diagonal message segment for one edge chunk.

        Returns (m [c, d] — decoded AND offdiag-scaled, omega [c, R]
        detached for router statistics, or None for non-basis modes).
        """
        v_s = (v0, v1, v2)[a][s_c]
        if self.transition_mode == "static":
            m_rel = self.cross_static[self._pair_idx[(a, b)]](v_s)
            omega = None
        else:
            q_d = (q0, q1, q2)[b][d_c]
            parts = [q_d, v_s]
            if S is not None:
                parts.append(S[d_c])
            c = int(q_d.size(0))
            parts.append(e_embs[a].unsqueeze(0).expand(c, -1))
            parts.append(e_embs[b].unsqueeze(0).expand(c, -1))
            desc = torch.cat(parts, dim=-1)
            if self.transition_mode == "film":
                gamma, beta = self.router(desc).chunk(2, dim=-1)
                m_rel = gamma * v_s + beta
                omega = None
            else:  # basis
                omega = F.softmax(self.router(desc), dim=-1)  # [c, R]
                zs = torch.stack(
                    [
                        self.basis_up[r](self.activation(self.basis_down[r](v_s)))
                        for r in range(self.num_bases)
                    ],
                    dim=1,
                )  # [c, R, d_r]
                m_rel = torch.einsum("cr,crd->cd", omega, zs)
                omega = omega.detach()
        m = self.target_decode[b](m_rel)
        return self.offdiag_scale * m, omega

    # ------------------------------------------------------------------
    # Layer forward
    # ------------------------------------------------------------------

    def forward(
        self,
        H: torch.Tensor,  # [N, 3, d]
        edge_index: torch.Tensor,  # [2, E]
        num_nodes: int,
        collect_stats: bool = True,
    ) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
        """One semantic-state evolution step.

        Returns (H_out [N, 3, d], stats {"transition": {...}, "basis": {...}})
        with detached float aggregates (empty dicts when collect_stats=False).
        """
        # frozen zero layer scale -> EXACT identity (plan §16.7 test mode)
        if self.layer_scale is None:
            return H, {}

        num_nodes = int(num_nodes)
        d = self.factor_dim
        device = H.device

        if self.use_same_node_context:
            S = self.context_mlp(H.reshape(num_nodes, NUM_FACTORS * d))  # [N, d_ctx]
        else:
            S = None

        # semantic -> relational projections (plan §3.2)
        if self.use_dual_space:
            v = [self.src_proj[a](H[:, a]) for a in range(NUM_FACTORS)]  # [N, d_r]
            q = [self.tgt_proj[b](H[:, b]) for b in range(NUM_FACTORS)]
        else:
            v = [H[:, a] for a in range(NUM_FACTORS)]
            q = v

        src, dst = edge_index[0], edge_index[1]
        num_edges = int(edge_index.size(1))
        deg = torch.bincount(dst, minlength=num_nodes).to(H.dtype)
        deg_mean = deg.clamp_min(1.0)

        # [N, d] channel accumulators (plan §6: source channels preserved)
        acc: dict[tuple[int, int], torch.Tensor] = {
            (a, b): torch.zeros(num_nodes, d, dtype=H.dtype, device=device)
            for a in range(NUM_FACTORS)
            for b in range(NUM_FACTORS)
        }

        # basis router statistics accumulated across chunks (basis mode only)
        omega_sum = None
        omega_ent_sum = 0.0
        omega_top1 = None
        omega_count = 0
        e_embs = self.factor_emb.weight

        use_ckpt = self.memory_checkpoint and torch.is_grad_enabled() and self.training

        # ---- pre-aggregation functional messages, edge-chunked (plan §17)
        for start in range(0, num_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, num_edges)
            s_c, d_c = src[start:end], dst[start:end]

            # diagonal: ownership-preserving propagation (plan §3.3)
            for b in range(NUM_FACTORS):
                if use_ckpt:
                    m = torch.utils.checkpoint.checkpoint(
                        self._diag_chunk, b, v[0], v[1], v[2], s_c, use_reentrant=False
                    )
                else:
                    m = self._diag_chunk(b, v[0], v[1], v[2], s_c)
                acc[(b, b)] = acc[(b, b)].index_add(0, d_c, m)

            if not self.offdiag_enabled or self.offdiag_scale is None:
                # frozen zero off-diagonal scale (test mode): the off-diagonal
                # path contributes EXACTLY nothing and is skipped entirely
                continue

            for a in range(NUM_FACTORS):
                for b in range(NUM_FACTORS):
                    if a == b:
                        continue
                    if use_ckpt:
                        m, omega = torch.utils.checkpoint.checkpoint(
                            self._offdiag_chunk,
                            a, b, v[0], v[1], v[2], q[0], q[1], q[2],
                            S, e_embs, s_c, d_c,
                            use_reentrant=False,
                        )
                    else:
                        m, omega = self._offdiag_chunk(
                            a, b, v[0], v[1], v[2], q[0], q[1], q[2],
                            S, e_embs, s_c, d_c,
                        )
                    acc[(a, b)] = acc[(a, b)].index_add(0, d_c, m)
                    if omega is not None:
                        # router statistics (already detached)
                        if omega_sum is None:
                            omega_sum = omega.sum(dim=0)
                            omega_top1 = torch.zeros(
                                self.num_bases, dtype=omega.dtype, device=device
                            )
                        else:
                            omega_sum = omega_sum + omega.sum(dim=0)
                        omega_top1 = omega_top1 + torch.bincount(
                            omega.argmax(dim=-1), minlength=self.num_bases
                        ).to(omega.dtype)
                        omega_ent_sum += float(
                            -(omega * torch.log(omega + R3_EPS)).sum(dim=-1).sum().item()
                        )
                        omega_count += int(omega.size(0))

        # ---- mean neighbor aggregation (plan §5) --------------------------
        for key in acc:
            acc[key] = acc[key] / deg_mean.unsqueeze(-1)

        # ---- source-channel preservation -> per-target message (plan §6) --
        if self.offdiag_enabled and self.preserve_source_channels:
            M = [
                torch.cat([acc[(a, b)] for a in range(NUM_FACTORS)], dim=-1)
                for b in range(NUM_FACTORS)
            ]  # [N, 3d] each
        elif self.offdiag_enabled:
            M = [
                (acc[(0, b)] + acc[(1, b)] + acc[(2, b)]) / 3.0
                for b in range(NUM_FACTORS)
            ]
        else:
            M = [acc[(b, b)] for b in range(NUM_FACTORS)]

        # ---- ownership-preserving target update (plan §7) -----------------
        out_parts: list[torch.Tensor] = []
        for b in range(NUM_FACTORS):
            u = self.update_ln[b](torch.cat([H[:, b], M[b]], dim=-1))
            delta = self.update[b](u)
            out_parts.append(H[:, b] + self.layer_scale * delta)
        H_out = torch.stack(out_parts, dim=1)  # [N, 3, d]

        if not collect_stats:
            return H_out, {}

        stats_t: dict[str, float] = {}
        stats_b: dict[str, float] = {}

        # ---- transition magnitude (plan §18.1) ----------------------------
        for b, name in enumerate(FACTOR_NAMES):
            dnorm = float(acc[(b, b)].norm(dim=-1).mean().item())
            if self.offdiag_enabled:
                off = torch.stack(
                    [acc[(a, b)].norm(dim=-1) for a in range(NUM_FACTORS) if a != b], dim=1
                )
                onorm = float(off.sum(dim=1).mean().item())
            else:
                onorm = 0.0
            stats_t[f"diag_norm_{name}"] = dnorm
            stats_t[f"offdiag_norm_{name}"] = onorm
            stats_t[f"offdiag_diag_ratio_{name}"] = onorm / (dnorm + R3_EPS)

        # ---- 9 source->target channel strengths (plan §18.2) --------------
        for a in range(NUM_FACTORS):
            for b in range(NUM_FACTORS):
                stats_t[f"ch_{FACTOR_NAMES[a]}_{FACTOR_NAMES[b]}"] = float(
                    acc[(a, b)].norm(dim=-1).mean().item()
                )

        # ---- ownership preservation (plan §18.4) --------------------------
        def _cos(u: torch.Tensor, w: torch.Tensor) -> float:
            return float(
                (F.normalize(u, dim=-1) * F.normalize(w, dim=-1)).sum(dim=-1).mean().item()
            )

        stats_t["cos_c_pt"] = _cos(H_out[:, 0], H_out[:, 1])
        stats_t["cos_c_pv"] = _cos(H_out[:, 0], H_out[:, 2])
        stats_t["cos_pt_pv"] = _cos(H_out[:, 1], H_out[:, 2])
        for b, name in enumerate(FACTOR_NAMES):
            stats_t[f"norm_{name}"] = float(H_out[:, b].norm(dim=-1).mean().item())
            delta_norm = (H_out[:, b] - H[:, b]).norm(dim=-1).mean()
            state_norm = H[:, b].norm(dim=-1).mean()
            stats_t[f"update_ratio_{name}"] = float((delta_norm / (state_norm + R3_EPS)).item())

        # ---- stability (plan §18.5) ---------------------------------------
        stats_t["offdiag_scale"] = (
            0.0 if self.offdiag_scale is None else float(self.offdiag_scale.detach().item())
        )
        stats_t["layer_scale"] = float(self.layer_scale.detach().item())
        stats_t["max_activation"] = float(H_out.abs().max().item())

        # ---- basis utilization (plan §18.3) -------------------------------
        if self.transition_mode == "basis" and self.offdiag_enabled and omega_count > 0:
            w_mean = omega_sum / omega_count
            for r in range(self.num_bases):
                stats_b[f"basis_mean_w{r}"] = float(w_mean[r].item())
                stats_b[f"basis_top1_{r}"] = float((omega_top1[r] / omega_count).item())
            stats_b["basis_entropy"] = omega_ent_sum / omega_count
        elif self.transition_mode == "basis":
            for r in range(self.num_bases):
                stats_b[f"basis_mean_w{r}"] = 0.0
                stats_b[f"basis_top1_{r}"] = 0.0
            stats_b["basis_entropy"] = 0.0

        return H_out, {"transition": stats_t, "basis": stats_b}
