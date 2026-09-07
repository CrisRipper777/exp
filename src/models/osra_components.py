"""OSRA components
(docs/OSRA_下一阶段推进计划.md, §5-§7).

Strong Same-Ownership Local Block (plan §6). For each ownership factor
b in {C, Pt, Pv}, INDEPENDENTLY (no cross-factor interaction in S1):

    N_i^{b}  = Mean_{j in N(i)} W_n^b H_j^b            (local_mode=strong)
               or GATv2-attention-weighted mean        (local_mode=gatv2)
    U_i^{b}  = W_s^b H_i^b + N_i^{b}
    Delta    = W_2^b GELU( W_1^b LN(U_i^{b}) )         (ffn_expand_ratio=2)
    Hhat_i^b = H_i^b + eta_l * Dropout(Delta)          (NO post-LN, plan §7)

Optional initial ownership anchor (plan §6, L2):
    H~_i^b = (1 - alpha) * Hhat_i^b + alpha * H_i^{b,0}

r3_diag control mode reproduces the R3 V0 diagonal layer semantics
(dual-space projection -> static diagonal linear -> mean aggregation ->
pre-LN concat update) inside the same block interface.

Memory discipline: edge-chunked scatter_add aggregation; the per-chunk
GATv2 score segment runs under activation checkpointing (memory_checkpoint,
randomness-free: no dropout in the score path). No dense [E, d, d] or
[N, deg, d] tensors.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .common import get_activation, make_norm

warnings.filterwarnings(
    "ignore",
    message=".*torch.cpu.amp.autocast.*deprecated.*",
    category=FutureWarning,
    module=r"torch\.utils\.checkpoint",
)

FACTOR_NAMES = ("c", "pt", "pv")
NUM_FACTORS = 3
OSRA_EPS = 1.0e-8


def _chunked_segment_mean(values_per_factor, edge_index, num_nodes, edge_chunk_size):
    """Mean over N(i) of per-factor node values [N, d] (chunked scatter_add).

    Returns [N, d] per factor; isolated nodes get exactly 0."""
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    deg = torch.bincount(dst, minlength=num_nodes).to(values_per_factor[0].dtype).clamp_min(1.0)
    outs = []
    for val in values_per_factor:
        out = torch.zeros(num_nodes, val.size(-1), dtype=val.dtype, device=val.device)
        for start in range(0, num_edges, edge_chunk_size):
            end = min(start + edge_chunk_size, num_edges)
            out = out.index_add(0, dst[start:end], val[src[start:end]])
        outs.append(out / deg.unsqueeze(-1))
    return outs


class OsraRelationalAccess(nn.Module):
    """Target-Driven Cross-Ownership Relational Access (plan §9-§13).

    For target factor b, access ONLY {(j, a) : j in N(i), a != b}:

        q_{i,b,k} = W_Q^b H~_i^b + s_k + e_b          (K slots; s_k/e_b are
                                                       d_r-dimensional)
        k_{j,a}   = W_K^a H~_j^a + e_a
        v_{j,a}   = W_V^a H~_j^a
        score     = q^T k / sqrt(d_r)                 (real neighbors, a!=b)
        null      : learnable key k_0^{b,k}, FIXED ZERO value; softmax over
                    {null} u {(j,a)} — if alpha_null -> 1 the slot reads no
                    cross-ownership evidence (NO extra cross gate, plan §12)
        R_{i,b,k} = sum alpha * v
        R_i^b     = FFN_R([R_{i,b,1} || ... || R_{i,b,K}])     (temporary
                    relational workspace, not a 4th factor)
        Delta_rel = FFN_b([H~_i^b || R_i^b])          (target-specific PreNorm)
        H^{b,l+1} = H~_i^b + rho_l * Delta_rel        (rho init 0.05, > 0)

    Modes:
        direct : static per-(a,b) linear transfer + mean + merge (Q1 control,
                 no query / no slots / no null)
        static : learned slot queries WITHOUT target-state dependence (Q2)
        target : W_Q^b H~_i^b conditioned queries (Q3/Q4)

    Memory discipline: scores are [E]-level scalars computed in edge chunks;
    the per-chunk score segment is randomness-free (checkpointable); the
    segment softmax couples the two off-diagonal source types + the null row
    per target node (scatter_reduce amax / scatter_add passes, coupled-message
    pattern). No dense [N, deg, ...] tensors.
    """

    def __init__(
        self,
        factor_dim: int,
        relation_dim: int,
        num_slots: int,
        access_mode: str,
        query_mode: str,
        use_null: bool,
        relational_scale_init: float,
        edge_chunk_size: int,
        dropout: float,
        activation: str,
        norm: str,
        memory_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        d = int(factor_dim)
        d_r = int(relation_dim)
        self.factor_dim = d
        self.d_r = d_r
        self.num_slots = int(num_slots)
        self.access_mode = str(access_mode)
        assert self.access_mode in ("direct", "query")
        self.query_mode = str(query_mode)
        assert self.query_mode in ("static", "target")
        self.use_null = bool(use_null)
        self.edge_chunk_size = max(int(edge_chunk_size), 1)
        self.memory_checkpoint = bool(memory_checkpoint)
        self.activation = get_activation(activation)

        # frozen-zero rho -> the access module short-circuits to exact
        # identity (plan §26 test mode); real configs keep it > 0.
        if float(relational_scale_init) == 0.0:
            self.relational_scale = None
        else:
            self.relational_scale = nn.Parameter(torch.tensor(float(relational_scale_init)))

        self.factor_emb = nn.Embedding(NUM_FACTORS, d_r)  # e_a / e_b

        if self.access_mode == "direct":
            # Q1 control: static per-(a,b) cross transform + mean + merge
            self.cross_static = nn.ModuleList(
                [nn.Linear(d, d) for _ in range(NUM_FACTORS * (NUM_FACTORS - 1))]
            )
            self._pair_idx = {
                (a, b): idx
                for idx, (a, b) in enumerate(
                    (a, b) for a in range(NUM_FACTORS) for b in range(NUM_FACTORS) if a != b
                )
            }
            self.merge = nn.ModuleList([nn.Linear(2 * d, d_r) for _ in range(NUM_FACTORS)])
        else:
            # query access (Q2-Q4)
            self.q_proj = nn.ModuleList([nn.Linear(d, d_r) for _ in range(NUM_FACTORS)])
            self.k_proj = nn.ModuleList([nn.Linear(d, d_r) for _ in range(NUM_FACTORS)])
            self.v_proj = nn.ModuleList([nn.Linear(d, d_r) for _ in range(NUM_FACTORS)])
            self.slots = nn.Parameter(torch.randn(num_slots, d_r) * 0.02)  # s_k
            if self.use_null:
                # learnable null key per (target b, slot k); null VALUE is
                # the fixed zero vector (plan §12 — never a parameter)
                self.null_key = nn.Parameter(torch.zeros(NUM_FACTORS, num_slots, d_r))
            # bias-free: null-only attention (R_k == 0 exactly) must map to
            # ZERO relational workspace (plan §26.3 test requirement)
            self.ffn_r = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(num_slots * d_r, 2 * d_r, bias=False),
                        get_activation(activation),
                        nn.Linear(2 * d_r, d_r, bias=False),
                    )
                    for _ in range(NUM_FACTORS)
                ]
            )

        # target-specific writeback PreNorm FFN (plan §13)
        self.write_ln = nn.ModuleList([nn.LayerNorm(d + d_r) for _ in range(NUM_FACTORS)])
        self.write_in = nn.ModuleList([nn.Linear(d + d_r, 2 * d) for _ in range(NUM_FACTORS)])
        self.write_out = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(NUM_FACTORS)])

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    def _queries(self, H: torch.Tensor, intervention: str | None = None) -> list[torch.Tensor]:
        """[q_{i,b,k}] per b: [N, K, d_r]. Q2: no target-state dependence.

        S2.5 causal interventions on the target-state term W_Q^b H_i^b only
        (plan: factor identity / slots / graph / K/V / Null untouched):
            target_state_permute : permute nodes WITHIN each target factor b
            target_state_mean    : replace with the factor's node mean
        """
        out = []
        for b in range(NUM_FACTORS):
            if self.query_mode == "target":
                base = self.q_proj[b](H[:, b])  # [N, d_r]
                if intervention == "target_state_permute":
                    perm = torch.randperm(base.size(0), device=base.device)
                    base = base[perm]
                elif intervention == "target_state_mean":
                    base = base.mean(dim=0, keepdim=True).expand(base.size(0), -1)
                q = base.unsqueeze(1) + self.slots.unsqueeze(0) + self.factor_emb.weight[b]
            else:
                q = (self.slots.unsqueeze(0) + self.factor_emb.weight[b]).expand(
                    int(H.size(0)), -1, -1
                )
            out.append(q)  # [N, K, d_r]
        return out

    # ------------------------------------------------------------------
    # Score segments (randomness-free, checkpointable)
    # ------------------------------------------------------------------

    def _score_chunk(self, q_k, k_a, s_c, d_c):
        """s = (q_k[dst] * k_a[src]).sum(-1) / sqrt(d_r) for one chunk."""
        return (q_k[d_c] * k_a[s_c]).sum(dim=-1) / (self.d_r ** 0.5)

    def _evidence_chunk(self, max_i, z, s_a0, s_a1, v_a0, v_a1, s_c, d_c):
        """One chunk of the (b,k) pass-3 weighted scatter for BOTH
        off-diagonal source types (checkpointable): the scatter happens
        INSIDE the segment, so only the [N, d_r] partial sum is retained,
        never the [E_chunk, d_r] gathers (ele-fashion OOM fix)."""
        out = torch.zeros(max_i.size(0), self.d_r, dtype=s_a0.dtype, device=s_a0.device)
        for s_a, v_a in ((s_a0, v_a0), (s_a1, v_a1)):
            alpha = torch.exp(s_a - max_i[d_c]) / z[d_c]
            out = out.index_add(0, d_c, alpha.unsqueeze(-1) * v_a[s_c])
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        H: torch.Tensor,  # [N, 3, d] (local-propagated states)
        edge_index: torch.Tensor,
        num_nodes: int,
        collect_stats: bool = True,
        intervention: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """``intervention`` is the S2.5 causal-usage audit switch (all
        interventions are parameter-frozen, same-checkpoint inference-only;
        None keeps the production path untouched):
            target_state_permute           (A1)
            target_state_mean              (A2)
            uniform_neighbor_within_source (A3)
            flatten_source_ownership       (A4)
            null_off                       (A5)
            null_only                      (A6)
        """
        if self.relational_scale is None:
            return H, {}
        num_nodes = int(num_nodes)
        d = self.factor_dim
        device = H.device
        src, dst = edge_index[0], edge_index[1]
        num_edges = int(edge_index.size(1))
        deg = torch.bincount(dst, minlength=num_nodes).to(H.dtype)
        deg_mean = deg.clamp_min(1.0)

        R_parts: list[torch.Tensor] = []
        stats: dict[str, float] = {}

        for b in range(NUM_FACTORS):
            off = [a for a in range(NUM_FACTORS) if a != b]
            if intervention == "null_only":
                # A6: all attention to Null -> cross evidence EXACTLY zero;
                # the writeback FFN still runs on [H || 0]
                R_parts.append(torch.zeros(num_nodes, self.d_r, dtype=H.dtype, device=device))
                continue
            if self.access_mode == "direct":
                # --- Q1: static cross transform + mean + merge ------------
                msgs = [
                    _chunked_segment_mean(
                        [self.cross_static[self._pair_idx[(a, b)]](H[:, a])],
                        edge_index, num_nodes, self.edge_chunk_size,
                    )[0]
                    for a in off
                ]
                R_parts.append(self.merge[b](torch.cat(msgs, dim=-1)))  # [N, d_r]
                continue

            # --- Q2-Q4: slot queries ----------------------------------------
            queries = self._queries(H, intervention)[b]  # [N, K, d_r]
            # typed source evidence for the OFF-diagonal sources only (plan §11)
            k_vals = [self.k_proj[a](H[:, a]) + self.factor_emb.weight[a] for a in off]
            v_vals = [self.v_proj[a](H[:, a]) for a in off]
            R_bk = []
            for kk in range(self.num_slots):
                q_k = queries[:, kk]  # [N, d_r]
                # A5: null_off forbids the null state (logit -inf, no
                # out-of-model renormalization)
                s_null = (
                    (q_k * self.null_key[b, kk]).sum(dim=-1) / (self.d_r ** 0.5)
                    if self.use_null and intervention != "null_off"
                    else torch.full((num_nodes,), float("-inf"), dtype=H.dtype, device=device)
                )
                # pass 0: chunked edge scores per off-diagonal source type
                s_edges = []
                use_ckpt = self.memory_checkpoint and torch.is_grad_enabled() and self.training
                for k_a in k_vals:
                    s_a = torch.empty(num_edges, dtype=H.dtype, device=device)
                    for start in range(0, num_edges, self.edge_chunk_size):
                        end = min(start + self.edge_chunk_size, num_edges)
                        s_c, d_c = src[start:end], dst[start:end]
                        if use_ckpt:
                            s_a[start:end] = torch.utils.checkpoint.checkpoint(
                                self._score_chunk, q_k, k_a, s_c, d_c,
                                use_reentrant=False,
                            )
                        else:
                            s_a[start:end] = self._score_chunk(q_k, k_a, s_c, d_c)
                    s_edges.append(s_a)
                # pass 1: per-target max over null + both source types
                # (floored at 0: keeps isolated nodes NaN-free when
                # use_null=false and s_null == -inf)
                max_i = s_null.clone()
                for s_a in s_edges:
                    seg = torch.zeros(num_nodes, dtype=H.dtype, device=device)
                    seg = seg.scatter_reduce(0, dst, s_a, reduce="amax", include_self=False)
                    max_i = torch.maximum(max_i, seg)
                max_i = torch.maximum(max_i, torch.zeros_like(max_i))
                # pass 2: Z_i = exp(s_null - max) + sum_a sum_j exp(s - max)
                z = torch.exp(s_null - max_i)
                for s_a in s_edges:
                    for start in range(0, num_edges, self.edge_chunk_size):
                        end = min(start + self.edge_chunk_size, num_edges)
                        d_c = dst[start:end]
                        z = z.scatter_add(0, d_c, torch.exp(s_a[start:end] - max_i[d_c]))
                z = z.clamp_min(OSRA_EPS)
                # pass 3: weighted value scatter for BOTH source types inside
                # the checkpointed per-chunk segment (memory fix)
                R_k = torch.zeros(num_nodes, self.d_r, dtype=H.dtype, device=device)
                if intervention in ("uniform_neighbor_within_source", "flatten_source_ownership"):
                    # --- S2.5 audit pass 3: per-source alphas with
                    # intervention transforms (inference-only, no checkpoint
                    # needed). A3 keeps the per-node source mass but makes the
                    # neighbor ranking uniform; A4 keeps the within-source
                    # ranking but equalizes the two sources' masses.
                    orig_chunks = []
                    node_masses = []
                    for s_a in s_edges:
                        chunks = []
                        mass = torch.zeros(num_nodes, dtype=H.dtype, device=device)
                        for start in range(0, num_edges, self.edge_chunk_size):
                            end = min(start + self.edge_chunk_size, num_edges)
                            d_c = dst[start:end]
                            a_chunk = torch.exp(s_a[start:end] - max_i[d_c]) / z[d_c]
                            chunks.append(a_chunk)
                            mass = mass.scatter_add(0, d_c, a_chunk)
                        orig_chunks.append(chunks)
                        node_masses.append(mass)
                    if intervention == "flatten_source_ownership":
                        equal = (node_masses[0] + node_masses[1]) / 2.0
                    else:
                        equal = None
                    for v_a, chunks, mass in zip(v_vals, orig_chunks, node_masses):
                        for idx, start in enumerate(range(0, num_edges, self.edge_chunk_size)):
                            end = min(start + self.edge_chunk_size, num_edges)
                            s_c, d_c = src[start:end], dst[start:end]
                            if equal is None:
                                # A3: uniform over N(i) within this source
                                alpha = mass[d_c] / deg_mean[d_c]
                            else:
                                # A4: rescale the learned within-source ranking
                                # to the equalized per-node source mass
                                alpha = chunks[idx] * (equal[d_c] / mass[d_c].clamp_min(OSRA_EPS))
                            R_k = R_k.index_add(0, d_c, alpha.unsqueeze(-1) * v_a[s_c])
                else:
                    s0, s1 = s_edges
                    v0, v1 = v_vals
                    for start in range(0, num_edges, self.edge_chunk_size):
                        end = min(start + self.edge_chunk_size, num_edges)
                        s_c, d_c = src[start:end], dst[start:end]
                        if use_ckpt:
                            partial = torch.utils.checkpoint.checkpoint(
                                self._evidence_chunk, max_i, z,
                                s0[start:end], s1[start:end], v0, v1, s_c, d_c,
                                use_reentrant=False,
                            )
                        else:
                            partial = self._evidence_chunk(
                                max_i, z, s0[start:end], s1[start:end], v0, v1, s_c, d_c
                            )
                        R_k = R_k + partial
                R_bk.append(R_k)
                # stats: null mass + entropy (only when requested)
                if collect_stats and self.use_null:
                    stats[f"nullmass_{FACTOR_NAMES[b]}_k{kk}"] = float(
                        torch.exp(s_null - max_i).div(z).mean().item()
                    )
                    ent = -(torch.exp(s_null - max_i) / z * torch.log(
                        torch.exp(s_null - max_i) / z + OSRA_EPS
                    ))
                    for a, s_a in zip(off, s_edges):
                        for start in range(0, num_edges, self.edge_chunk_size):
                            end = min(start + self.edge_chunk_size, num_edges)
                            d_c = dst[start:end]
                            p = torch.exp(s_a[start:end] - max_i[d_c]) / z[d_c]
                            ent = ent.scatter_add(0, d_c, -p * torch.log(p + OSRA_EPS))
                    stats[f"neff_{FACTOR_NAMES[b]}_k{kk}"] = float(torch.exp(ent).mean().item())
                if collect_stats:
                    # source ownership mass A_{a->b} (plan §17)
                    for a, s_a in zip(off, s_edges):
                        mass = torch.zeros(num_nodes, dtype=H.dtype, device=device)
                        for start in range(0, num_edges, self.edge_chunk_size):
                            end = min(start + self.edge_chunk_size, num_edges)
                            d_c = dst[start:end]
                            alpha = torch.exp(s_a[start:end] - max_i[d_c]) / z[d_c]
                            mass = mass.scatter_add(0, d_c, alpha)
                        stats[f"srcmass_{FACTOR_NAMES[a]}_{FACTOR_NAMES[b]}"] = float(mass.mean().item())
                    # null mass by degree bins (plan §17)
                    if self.use_null:
                        p_null = torch.exp(s_null - max_i) / z
                        for bin_name, lo, hi in (
                            ("d1_5", 1, 5), ("d6_10", 6, 10), ("d11_20", 11, 20), ("d20p", 21, None),
                        ):
                            mask = (deg >= lo) & ((deg <= hi) if hi is not None else torch.ones_like(deg).bool())
                            if bool(mask.any()):
                                stats[f"nullmass_{FACTOR_NAMES[b]}_{bin_name}_k{kk}"] = float(
                                    p_null[mask].mean().item()
                                )
            R_parts.append(self.ffn_r[b](torch.cat(R_bk, dim=-1)))  # [N, d_r]

        # --- writeback (plan §13) ------------------------------------------
        out_parts = []
        for b in range(NUM_FACTORS):
            u = self.write_ln[b](torch.cat([H[:, b], R_parts[b]], dim=-1))
            delta = self.write_out[b](self.activation(self.write_in[b](u)))
            out_parts.append(H[:, b] + self.relational_scale * delta)
            if collect_stats:
                delta_norm = (self.relational_scale * delta).norm(dim=-1).mean()
                state_norm = H[:, b].norm(dim=-1).mean()
                stats[f"writeback_ratio_{FACTOR_NAMES[b]}"] = float(
                    (delta_norm / (state_norm + OSRA_EPS)).item()
                )
        H_out = torch.stack(out_parts, dim=1)
        if collect_stats:
            stats["relational_scale"] = float(self.relational_scale.detach().item())
        return H_out, stats


class OsraBlock(nn.Module):
    """One OSRA dual-mode graph block (plan §3): strong same-ownership
    local propagation, then (when enabled) target-driven cross-ownership
    relational access."""

    def __init__(self, local: nn.Module, access: nn.Module | None) -> None:
        super().__init__()
        self.local = local
        self.access = access

    def forward(
        self,
        H: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        H0: torch.Tensor | None = None,
        collect_local: bool = True,
        collect_access: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        H_loc, stats = self.local(H, edge_index, num_nodes, H0, collect_stats=collect_local)
        if self.access is None:
            return H_loc, stats
        H_out, access_stats = self.access(H_loc, edge_index, num_nodes, collect_stats=collect_access)
        if collect_access:
            stats.update(access_stats)
        return H_out, stats


class OsraLocalBlock(nn.Module):
    """Strong same-ownership local propagation block (plan §6)."""

    def __init__(
        self,
        factor_dim: int,
        ffn_expand_ratio: float,
        layer_scale_init: float,
        anchor_mix: float,
        local_mode: str,
        edge_chunk_size: int,
        dropout: float,
        activation: str,
        norm: str,
        memory_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        d = int(factor_dim)
        self.factor_dim = d
        self.local_mode = str(local_mode)
        assert self.local_mode in ("strong", "gatv2", "r3_diag"), (
            f"osra.local_mode must be strong|gatv2|r3_diag, got {self.local_mode!r}"
        )
        self.anchor_mix = float(anchor_mix)
        assert 0.0 <= self.anchor_mix <= 1.0
        self.edge_chunk_size = max(int(edge_chunk_size), 1)
        self.memory_checkpoint = bool(memory_checkpoint)
        self.activation = get_activation(activation)
        expand = int(round(float(ffn_expand_ratio) * d))

        # frozen-zero scale convention (plan §7 tests): init 0 -> exact
        # identity residual; real configs keep it > 0.
        if float(layer_scale_init) == 0.0:
            self.layer_scale = None
        else:
            self.layer_scale = nn.Parameter(torch.tensor(float(layer_scale_init)))

        if self.local_mode == "r3_diag":
            # R3 V0 diagonal control: dual-space projection + static linear
            # message + mean + pre-LN concat update (R3 semantics).
            self.src_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(NUM_FACTORS)])
            self.diag = nn.ModuleList([nn.Linear(d, d) for _ in range(NUM_FACTORS)])
            self.update_ln = nn.ModuleList([nn.LayerNorm(2 * d) for _ in range(NUM_FACTORS)])
            self.update = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(NUM_FACTORS)])
        else:
            # neighbor transform (no bias: mean(W_n H_j) must not inject a
            # constant into isolated nodes)
            self.neighbor_w = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(NUM_FACTORS)])
            self.self_w = nn.ModuleList([nn.Linear(d, d) for _ in range(NUM_FACTORS)])
            self.ffn_ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(NUM_FACTORS)])
            self.ffn_in = nn.ModuleList([nn.Linear(d, expand) for _ in range(NUM_FACTORS)])
            self.ffn_out = nn.ModuleList([nn.Linear(expand, d) for _ in range(NUM_FACTORS)])
            self.ffn_dropout = nn.Dropout(float(dropout))
            if self.local_mode == "gatv2":
                # factor-wise GATv2 attention (control only, plan §7):
                # e_ji = a^T LeakyReLU(W_att [H_i^b || H_j^b])
                self.att_w = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(NUM_FACTORS)])
                self.att_a = nn.ModuleList([nn.Linear(d, 1, bias=False) for _ in range(NUM_FACTORS)])

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _agg_strong(self, H, edge_index, num_nodes):
        """Neighbor mean of W_n^b H^b (chunked)."""
        vals = [self.neighbor_w[b](H[:, b]) for b in range(NUM_FACTORS)]
        return _chunked_segment_mean(vals, edge_index, num_nodes, self.edge_chunk_size)

    def _gatv2_scores_chunk(self, b, H0, H1, H2, s_c, d_c):
        """Randomness-free per-chunk GATv2 logits for factor b (checkpointable)."""
        h_d = (H0, H1, H2)[b][d_c]
        h_s = (H0, H1, H2)[b][s_c]
        u = self.att_w[b](torch.cat([h_d, h_s], dim=-1))
        return self.att_a[b](F.leaky_relu(u, negative_slope=0.2)).squeeze(-1)

    def _gatv2_evidence_chunk(self, max_i, z, scores_chunk, vals_b, s_c, d_c):
        """One chunk of the GATv2 weighted-mean pass (checkpointable): the
        scatter happens INSIDE the segment so only the [N, d] partial sum is
        retained, never the [E_chunk, d] gathers (ele-fashion OOM fix)."""
        alpha = torch.exp(scores_chunk - max_i[d_c]) / z[d_c].clamp_min(OSRA_EPS)
        out = torch.zeros(max_i.size(0), vals_b.size(-1), dtype=vals_b.dtype, device=vals_b.device)
        return out.index_add(0, d_c, alpha.unsqueeze(-1) * vals_b[s_c])

    def _agg_gatv2(self, H, edge_index, num_nodes):
        """Factor-wise GATv2: segment softmax over N(i) within each factor,
        weighted mean of W_n^b H_j^b. Chunked scatter passes (coupled-message
        pattern); the score and evidence segments run under checkpoint so
        [E, d]-level intermediates are recomputed in backward."""
        src, dst = edge_index[0], edge_index[1]
        num_edges = int(edge_index.size(1))
        device = H.device
        vals = [self.neighbor_w[b](H[:, b]) for b in range(NUM_FACTORS)]
        use_ckpt = self.memory_checkpoint and torch.is_grad_enabled() and self.training
        outs = []
        for b in range(NUM_FACTORS):
            # pass 0: all edge logits (chunked)
            scores = torch.empty(num_edges, dtype=H.dtype, device=device)
            for start in range(0, num_edges, self.edge_chunk_size):
                end = min(start + self.edge_chunk_size, num_edges)
                s_c, d_c = src[start:end], dst[start:end]
                if use_ckpt:
                    scores[start:end] = torch.utils.checkpoint.checkpoint(
                        self._gatv2_scores_chunk, b, H[:, 0], H[:, 1], H[:, 2],
                        s_c, d_c, use_reentrant=False,
                    )
                else:
                    scores[start:end] = self._gatv2_scores_chunk(
                        b, H[:, 0], H[:, 1], H[:, 2], s_c, d_c
                    )
            # pass 1: per-target max
            seg = torch.zeros(num_nodes, dtype=H.dtype, device=device)
            seg = seg.scatter_reduce(0, dst, scores, reduce="amax", include_self=False)
            max_i = seg
            # pass 2: Z_i = sum exp(s - max)
            z = torch.zeros(num_nodes, dtype=H.dtype, device=device)
            for start in range(0, num_edges, self.edge_chunk_size):
                end = min(start + self.edge_chunk_size, num_edges)
                d_c = dst[start:end]
                z = z.scatter_add(0, d_c, torch.exp(scores[start:end] - max_i[d_c]))
            # pass 3: weighted mean of W_n H (scatter inside the checkpointed
            # per-chunk segment)
            out = torch.zeros(num_nodes, self.factor_dim, dtype=H.dtype, device=device)
            for start in range(0, num_edges, self.edge_chunk_size):
                end = min(start + self.edge_chunk_size, num_edges)
                s_c, d_c = src[start:end], dst[start:end]
                if use_ckpt:
                    partial = torch.utils.checkpoint.checkpoint(
                        self._gatv2_evidence_chunk, max_i, z,
                        scores[start:end], vals[b], s_c, d_c, use_reentrant=False,
                    )
                else:
                    partial = self._gatv2_evidence_chunk(
                        max_i, z, scores[start:end], vals[b], s_c, d_c
                    )
                out = out + partial
            outs.append(out)
        return outs

    # ------------------------------------------------------------------
    # Block forward
    # ------------------------------------------------------------------

    def forward(
        self,
        H: torch.Tensor,  # [N, 3, d]
        edge_index: torch.Tensor,  # [2, E]
        num_nodes: int,
        H0: torch.Tensor | None = None,  # initial ownership state (anchor)
        collect_stats: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        num_nodes = int(num_nodes)
        d = self.factor_dim

        if self.local_mode == "r3_diag":
            # --- R3 V0 diagonal control semantics -------------------------
            v = [self.src_proj[b](H[:, b]) for b in range(NUM_FACTORS)]
            mbar = _chunked_segment_mean(
                [self.diag[b](v[b]) for b in range(NUM_FACTORS)],
                edge_index, num_nodes, self.edge_chunk_size,
            )
            out_parts = []
            for b in range(NUM_FACTORS):
                u = self.update_ln[b](torch.cat([H[:, b], mbar[b]], dim=-1))
                delta = self.update[b](u)
                if self.layer_scale is None:
                    out_parts.append(H[:, b])
                else:
                    out_parts.append(H[:, b] + self.layer_scale * delta)
            H_out = torch.stack(out_parts, dim=1)
        else:
            # --- strong local propagation (plan §6) -----------------------
            if self.local_mode == "gatv2":
                neighbors = self._agg_gatv2(H, edge_index, num_nodes)
            else:
                neighbors = self._agg_strong(H, edge_index, num_nodes)
            out_parts = []
            for b in range(NUM_FACTORS):
                u = self.self_w[b](H[:, b]) + neighbors[b]
                delta = self.ffn_out[b](self.activation(self.ffn_in[b](self.ffn_ln[b](u))))
                if self.layer_scale is None:
                    out_parts.append(H[:, b])
                else:
                    out_parts.append(H[:, b] + self.layer_scale * self.ffn_dropout(delta))
            H_out = torch.stack(out_parts, dim=1)

        # --- initial ownership anchor (plan §6, L2) ------------------------
        if H0 is not None and self.anchor_mix > 0.0:
            H_out = (1.0 - self.anchor_mix) * H_out + self.anchor_mix * H0

        stats: dict[str, float] = {}
        if collect_stats:
            for b, name in enumerate(FACTOR_NAMES):
                delta_norm = (H_out[:, b] - H[:, b]).norm(dim=-1).mean()
                state_norm = H[:, b].norm(dim=-1).mean()
                stats[f"update_ratio_{name}"] = float((delta_norm / (state_norm + OSRA_EPS)).item())
            stats["layer_scale"] = 0.0 if self.layer_scale is None else float(self.layer_scale.detach().item())
            stats["max_activation"] = float(H_out.abs().max().item())
        return H_out, stats
