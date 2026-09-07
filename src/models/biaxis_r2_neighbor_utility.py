"""R2-Design-2.7 pre-aggregation neighbor-utility model
(docs/BiAxis_R2_Design_2_7_PreAggregation_Neighbor_Utility_Audit.md).

    F = [C, Pt, Pv]   (frozen A0 pre-graph ownership factors)
    z_base = A0(x, G) (frozen strong parent; side only ADDS)
    s_ji^{a->b} = psi([F_i^b, F_j^a, prod, |diff|, e_a, e_b])  (shared psi)
    alpha over {null} U N(i) with tau=1.0                      (plan §3)
    m_i^{a->b} = sum_j alpha_ji^{a->b} U_a(F_j^a)              (selection-only)
    m_i^b = (1/3) sum_a m_i^{a->b}
    z_util = [z_base | m_C | m_Pt | m_Pv]                      (no projection)

Modes:
    a0_base / uniform / target_null_only / generic_edge / diag_edge /
    pair_edge / semantic_sim (D2.7-A matrix)
    post_pair / source_factor_only / target_factor_only (D2.7-C/D)
    pair_transform_uniform / pair_transform_pre (D2.7-E)

Discipline:
    - A0 parent frozen, always eval (parent_frozen flag for the optional
      D2.7 adaptation; never automatic).
    - edge chunking + stable segment softmax (components); the forbidden
      [E,3,3,d] is never materialized.
    - observed graph as support only: no edge addition, no role labels,
      no learned adjacency (plan §1).
    - causal overrides never modify trained weights (plan §27/§28).
    - No Test: labels never enter the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_r2_neighbor_utility_components import (
    FactorPairScorer,
    GenericEdgeScorer,
    NullScorer,
    PairTransform,
    chunked_pair_message,
)
from .biaxis_r2_relfunc_components import (
    chunked_coupled_message,
    per_target_edge_mask,
    shuffle_scores_within_target,
)

FACTOR_NAMES = ("C", "Pt", "Pv")

MODES = (
    "a0_base",
    "uniform",
    "target_null_only",
    "generic_edge",
    "diag_edge",
    "pair_edge",
    "semantic_sim",
    "post_pair",
    "source_factor_only",
    "target_factor_only",
    "pair_transform_uniform",
    "pair_transform_pre",
    "coupled_equiv",
)

# pair list per mode: (a, b) row-major over source a -> target b
ALL_PAIRS = tuple((a, b) for a in range(3) for b in range(3))
DIAG_PAIRS = tuple((a, a) for a in range(3))

CAUSAL_OVERRIDES = (
    "full", "side_off",
    "remove_top_10", "remove_top_25", "remove_top_50",
    "remove_random_10", "remove_random_25", "remove_random_50",
    "remove_bottom_10", "remove_bottom_25", "remove_bottom_50",
    "keep_top_25", "keep_top_50",
    # D2.8 v2 repaired causal machinery (§5): the shuffle below is the fixed
    # exact integer-segment permutation (the float composite key is gone);
    # per-target removal selects inside each target's own neighborhood.
    "within_target_shuffle", "within_target_shuffle_fixed",
    "remove_top_per_target_10", "remove_top_per_target_25", "remove_top_per_target_50",
    "remove_random_per_target_10", "remove_random_per_target_25", "remove_random_per_target_50",
    "remove_bottom_per_target_10", "remove_bottom_per_target_25", "remove_bottom_per_target_50",
    "keep_top_per_target_25", "keep_top_per_target_50",
    "source_shuffle", "factor_id_shuffle",
    "noise_10", "noise_25",
)

MISMATCH_PERM_SEED = 20260904


class Model(nn.Module):
    """Pre-aggregation neighbor-utility model wrapping a frozen A0 parent."""

    def __init__(self, cfg, data_info, parent: nn.Module):
        super().__init__()
        object.__setattr__(self, "parent", parent)  # not a submodule
        self.parent.eval()
        for p in self.parent.parameters():
            p.requires_grad_(False)
        self.parent_frozen = True

        self.factor_dim = int(parent.factor_dim)
        self.hidden_dim = int(parent.hidden_dim)
        self.edge_chunk_size = int(cfg.model.get("edge_chunk_size", 50000))
        self.mode = str(cfg.model.mode)
        assert self.mode in MODES, self.mode
        self.num_classes = int(data_info["num_classes"])

        d = self.factor_dim
        h = self.hidden_dim
        dropout = float(cfg.model.get("dropout", 0.2))
        activation = str(cfg.model.get("activation", "gelu"))
        norm = str(cfg.model.get("norm", "layernorm"))
        self.type_dim = int(cfg.model.get("type_dim", 16))

        # --- payload transforms U_a (selection-only payload, plan §4) ------
        self.has_side = self.mode != "a0_base"
        if self.has_side:
            self.payload = nn.ModuleList(
                [nn.Linear(d, d, bias=False) for _ in range(3)])
            if self.mode.startswith("pair_transform"):
                self.pair_transform = PairTransform(d, activation, norm)

        # --- scorers ----------------------------------------------------------
        self.scorer: FactorPairScorer | None = None
        self.generic_scorer: GenericEdgeScorer | None = None
        self.local_proj: nn.Linear | None = None
        if self.mode in ("pair_edge", "diag_edge", "post_pair", "pair_transform_pre",
                         "coupled_equiv"):
            self.scorer = FactorPairScorer(d, self.type_dim, dropout, activation, norm)
        elif self.mode == "generic_edge":
            # width solved so scorer params match PAIR_EDGE's psi within +/-5%
            self.local_proj = nn.Linear(3 * d, d)
            width = self._solve_generic_width(d)
            self.generic_scorer = GenericEdgeScorer(d, width, dropout, activation, norm)
        elif self.mode in ("source_factor_only", "target_factor_only"):
            self.scorer = FactorPairScorer(d, self.type_dim, dropout, activation, norm)

        # --- null scorer (per (i,a,b); target-conditioned) -------------------
        if self.has_side and self.mode not in ("uniform", "pair_transform_uniform"):
            self.null_scorer = NullScorer(d, self.type_dim, hidden=d,
                                          activation=activation, norm=norm)

        # --- factor type embeddings (shared with the scorers) ----------------
        self.type_emb = nn.Embedding(3, self.type_dim)

        self.out_dim = h if not self.has_side else h + 3 * d
        self.side_parameter_count = int(sum(p.numel() for p in self.parameters()))
        self.parameter_count = self.side_parameter_count

    def _solve_generic_width(self, d: int) -> int:
        """GENERIC_EDGE scorer width so (local_proj + generic scorer) params
        match the PAIR_EDGE scorer params within +/-5%."""
        import math

        ref = FactorPairScorer(d, self.type_dim)
        target = ref.parameter_count()  # psi params only
        # local_proj: 3d*d + d; scorer: 4d*w + w + 2w + w = 4d*w + 4w + w
        # => total = 3d^2 + d + w(4d + 5)
        base = 3 * d * d + d
        w = int(round((target - base) / (4 * d + 5)))
        self._generic_match = {"target_pair_scorer_params": int(target),
                               "solved_width": max(w, d)}
        return max(w, d)

    # ------------------------------------------------------------------
    # Parent pieces
    # ------------------------------------------------------------------

    def _parent_forward(self, x, edge_index, num_nodes):
        factors, _z_local = self.parent._encode(x)
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        graph_out = self.parent._graph_update(f_block, edge_index, num_nodes)
        f_tilde = graph_out["f_tilde"]
        z_base = self.parent.fusion(
            torch.cat([f_tilde[:, 0], f_tilde[:, 1], f_tilde[:, 2]], dim=-1))
        return f_block, z_base

    def _parent_ctx(self, x, edge_index, num_nodes):
        if self.parent_frozen:
            with torch.no_grad():
                return self._parent_forward(x, edge_index, num_nodes)
        return self._parent_forward(x, edge_index, num_nodes)

    # ------------------------------------------------------------------
    # Edge scores
    # ------------------------------------------------------------------

    def _pair_input(self, f_block, edge_index, a, b, chunk):
        src, dst = edge_index[0], edge_index[1]
        s_c, d_c = src[chunk], dst[chunk]
        f_a, f_b = f_block[s_c, a], f_block[d_c, b]
        e_a = self.type_emb.weight[a].unsqueeze(0).expand(len(chunk), -1)
        e_b = self.type_emb.weight[b].unsqueeze(0).expand(len(chunk), -1)
        return torch.cat([f_b, f_a, f_b * f_a, (f_b - f_a).abs(), e_a, e_b], dim=-1)

    def _pair_scores_chunked(self, f_block, edge_index, a, b, num_edges,
                             scorer=None):
        """s^{a->b} for every edge (chunked; stored [E])."""
        scorer = scorer or self.scorer
        out = torch.zeros(num_edges, dtype=f_block.dtype, device=f_block.device)
        for start in range(0, num_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, num_edges)
            u = self._pair_input(f_block, edge_index, a, b,
                                 torch.arange(start, end, device=f_block.device))
            out[start:end] = scorer(u)
        return out

    def _null_scores(self, f_block, a, b):
        """s_null^{a->b} per target node."""
        e_a = self.type_emb.weight[a].unsqueeze(0).expand(f_block.size(0), -1)
        e_b = self.type_emb.weight[b].unsqueeze(0).expand(f_block.size(0), -1)
        return self.null_scorer(torch.cat([f_block[:, b], e_a, e_b], dim=-1))

    def _cos_scores(self, f_block, edge_index, a, b, num_edges):
        out = torch.zeros(num_edges, dtype=f_block.dtype, device=f_block.device)
        for start in range(0, num_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, num_edges)
            s_c = edge_index[0, start:end]
            d_c = edge_index[1, start:end]
            out[start:end] = torch.nn.functional.cosine_similarity(
                f_block[d_c, b], f_block[s_c, a], dim=-1)
        return out

    # ------------------------------------------------------------------
    # Per-mode side messages
    # ------------------------------------------------------------------

    def _side_messages(self, f_block, edge_index, num_nodes, causal="full"):
        """m [N, 3, d]: m[:, b] = (1/3) sum_a m^{a->b}."""
        mode = self.mode
        d = self.factor_dim
        num_edges = int(edge_index.size(1))
        src, dst = edge_index[0], edge_index[1]

        # optional causal masks / permutations (eval-time only). The
        # remove/keep masks are per-pair (score-dependent) and applied
        # inside the pair loop below.
        if causal == "source_shuffle":
            from src.analysis.perf_r2d15_utils import fixed_node_permutation

            perm = fixed_node_permutation(num_nodes, MISMATCH_PERM_SEED)
            f_block = f_block[perm]
        if causal == "factor_id_shuffle":
            from src.analysis.perf_r2d15_utils import fixed_node_permutation

            perm = fixed_node_permutation(3, MISMATCH_PERM_SEED)
            f_block = f_block[:, perm]
            # also permute the pair interpretation below via pair_perm
            pair_perm = {(a, b): (int(perm[a]), int(perm[b])) for (a, b) in ALL_PAIRS}
        else:
            pair_perm = {(a, b): (a, b) for (a, b) in ALL_PAIRS}

        payloads = [self.payload[a](f_block[:, a]) for a in range(3)]  # [N, d]
        if mode == "uniform" or mode == "pair_transform_uniform":
            # uniform: m^{a->b} = mean_j U_a(F_j^a) (or T_ab mean for pair_transform)
            from .biaxis_p1_components import neighbor_mean

            msgs = []
            for b in range(3):
                acc = None
                for a in range(3):
                    src_feat = f_block[:, a]
                    if mode == "pair_transform_uniform":
                        src_feat = self.pair_transform(src_feat, a, b)
                    else:
                        src_feat = payloads[a]
                    nm = neighbor_mean(
                        edge_index, src_feat, num_nodes,
                        edge_chunk_size=self.edge_chunk_size)
                    acc = nm if acc is None else acc + nm
                msgs.append(acc / 3.0)
            return torch.stack(msgs, dim=1)

        if mode == "target_null_only":
            # real-neighbor weights uniform; only null mass is target-conditioned
            from .biaxis_p1_components import neighbor_mean

            deg = torch.bincount(dst, minlength=num_nodes).to(f_block.dtype)
            msgs = []
            for b in range(3):
                acc = None
                for a in range(3):
                    nm = neighbor_mean(
                        edge_index, payloads[a], num_nodes,
                        edge_chunk_size=self.edge_chunk_size)
                    null = self._null_scores(f_block, a, b)
                    gate = deg / (deg + torch.exp(null))
                    contrib = gate.unsqueeze(-1) * nm
                    acc = contrib if acc is None else acc + contrib
                msgs.append(acc / 3.0)
            return torch.stack(msgs, dim=1)

        if mode == "post_pair":
            # aggregate uniformly FIRST, then a per-node pair gate (plan §30)
            from .biaxis_p1_components import neighbor_mean

            means = [
                neighbor_mean(edge_index, f_block[:, a], num_nodes,
                              edge_chunk_size=self.edge_chunk_size)
                for a in range(3)]
            msgs = []
            for b in range(3):
                acc = None
                for a in range(3):
                    n_a = means[a]
                    u = torch.cat(
                        [f_block[:, b], n_a, f_block[:, b] * n_a,
                         (f_block[:, b] - n_a).abs(),
                         self.type_emb.weight[a].unsqueeze(0).expand(num_nodes, -1),
                         self.type_emb.weight[b].unsqueeze(0).expand(num_nodes, -1)],
                        dim=-1)
                    g = torch.sigmoid(self.scorer(u))
                    contrib = g.unsqueeze(-1) * self.payload[a](n_a)
                    acc = contrib if acc is None else acc + contrib
                msgs.append(acc / 3.0)
            return torch.stack(msgs, dim=1)

        if mode == "generic_edge":
            z0 = self.local_proj(f_block.reshape(num_nodes, 3 * d))  # [N, d]
            s = torch.zeros(num_edges, dtype=f_block.dtype, device=f_block.device)
            for start in range(0, num_edges, self.edge_chunk_size):
                end = min(start + self.edge_chunk_size, num_edges)
                s_c, d_c = src[start:end], dst[start:end]
                zi, zj = z0[d_c], z0[s_c]
                u = torch.cat([zi, zj, zi * zj, (zi - zj).abs()], dim=-1)
                s[start:end] = self.generic_scorer(u)
            return self._weighted_messages(f_block, edge_index, num_nodes,
                                           payloads, s, causal)

        if mode == "coupled_equiv":
            # D2.8 v2 §5.3: explicit r*pi factorization of PAIR_EDGE using the
            # same edge/null logits (COUPLED_EQUIV bridge). Never trained;
            # used to verify max|m_coupled - m_pair_edge| < 1e-6.
            msgs = []
            for b in range(3):
                acc = None
                for a in range(3):
                    pa, pb = pair_perm[(a, b)]
                    s = self._pair_scores_chunked(f_block, edge_index, pa, pb,
                                                  num_edges)
                    null = self._null_scores(f_block, a, b)
                    m_ab = chunked_coupled_message(
                        f_block, edge_index, num_nodes, s, null, payloads[a],
                        edge_chunk_size=self.edge_chunk_size)
                    acc = m_ab if acc is None else acc + m_ab
                msgs.append(acc / 3.0)
            return torch.stack(msgs, dim=1)

        if mode in ("pair_edge", "pair_transform_pre", "semantic_sim",
                    "diag_edge", "source_factor_only", "target_factor_only"):
            pairs = DIAG_PAIRS if mode == "diag_edge" else ALL_PAIRS
            msgs = []
            for b in range(3):
                acc = None
                for a in range(3):
                    if (a, b) not in pairs:
                        continue
                    pa, pb = pair_perm[(a, b)]
                    payload = payloads[a]
                    m_ab = self._pair_message_ckpt(
                        f_block, edge_index, num_nodes, num_edges,
                        a, b, pa, pb, payload, causal)
                    acc = m_ab if acc is None else acc + m_ab
                msgs.append(acc / 3.0 if acc is not None else torch.zeros(
                    num_nodes, d, dtype=f_block.dtype, device=f_block.device))
            return torch.stack(msgs, dim=1)

        raise ValueError(f"mode={mode} has no side message path")

    def _pair_message_ckpt(self, f_block, edge_index, num_nodes, num_edges,
                           a, b, pa, pb, payload, causal):
        """One factor-pair message computation. Activation-checkpointed in
        any grad-enabled pass (the chunked intermediates of all 9 pairs
        would otherwise coexist until backward — ~21GB on ele-fashion)."""

        def _run(fb, ei):
            if self.mode == "semantic_sim":
                s = self._cos_scores(fb, ei, pa, pb, int(ei.size(1)))
            elif self.mode == "source_factor_only":
                s = self._pair_scores_chunked(fb, ei, a, a, int(ei.size(1)))
            elif self.mode == "target_factor_only":
                s = self._pair_scores_chunked(fb, ei, b, b, int(ei.size(1)))
            else:
                s = self._pair_scores_chunked(fb, ei, pa, pb, int(ei.size(1)))
            if causal in ("within_target_shuffle", "within_target_shuffle_fixed"):
                s = self._within_target_shuffle(s, ei, num_nodes, int(ei.size(1)))
            edge_mask = None
            if causal.startswith(("remove", "keep")) and "_per_target_" in causal:
                edge_mask = self._pair_mask_per_target(s, ei, num_nodes, causal)
            elif causal.startswith(("remove", "keep")):
                edge_mask = self._pair_mask(s, causal)
            null = self._null_scores(fb, a, b)
            pl = payload
            if self.mode == "pair_transform_pre":
                pl = self.pair_transform(fb[:, a], a, b)
            return chunked_pair_message(
                fb, ei, num_nodes, s, null, pl,
                edge_chunk_size=self.edge_chunk_size, edge_mask=edge_mask)

        if self.training or torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                _run, f_block, edge_index, use_reentrant=False)
        return _run(f_block, edge_index)

    def _weighted_messages(self, f_block, edge_index, num_nodes, payloads,
                           s, causal):
        msgs = []
        for b in range(3):
            acc = None
            for a in range(3):
                null = self._null_scores(f_block, a, b)
                m_ab = chunked_pair_message(
                    f_block, edge_index, num_nodes, s, null, payloads[a],
                    edge_chunk_size=self.edge_chunk_size)
                acc = m_ab if acc is None else acc + m_ab
            msgs.append(acc / 3.0)
        return torch.stack(msgs, dim=1)

    # ------------------------------------------------------------------
    # Causal machinery (eval-time only, weights untouched)
    # ------------------------------------------------------------------

    def _pair_edge_scores_all(self, f_block, edge_index, a, b, num_edges):
        return self._pair_scores_chunked(f_block, edge_index, a, b, num_edges)

    def _pair_mask(self, scores, causal, seed=20260904):
        """[E] bool mask for remove/keep overrides on this pair's scores."""
        num_edges = scores.numel()
        keep = torch.ones(num_edges, dtype=torch.bool, device=scores.device)
        if causal.startswith("remove_top") or causal.startswith("remove_bottom"):
            pct = int(causal.split("_")[-1]) / 100.0
            n_remove = int(num_edges * pct)
            largest = causal.startswith("remove_top")
            _, idx = torch.topk(scores, n_remove, largest=largest)
            keep = torch.ones(num_edges, dtype=torch.bool, device=scores.device)
            keep[idx] = False
        elif causal.startswith("remove_random"):
            pct = int(causal.split("_")[-1]) / 100.0
            generator = torch.Generator().manual_seed(seed)
            perm = torch.randperm(num_edges, generator=generator)
            n_remove = int(num_edges * pct)
            keep = torch.ones(num_edges, dtype=torch.bool, device=scores.device)
            keep[perm[:n_remove]] = False
        elif causal.startswith("keep_top"):
            pct = int(causal.split("_")[-1]) / 100.0
            n_keep = int(num_edges * pct)
            _, top_idx = torch.topk(scores, n_keep, largest=True)
            keep = torch.zeros(num_edges, dtype=torch.bool, device=scores.device)
            keep[top_idx] = True
        return keep

    def _within_target_shuffle(self, scores, edge_index, num_nodes, num_edges):
        """Permute the scores across each target's own neighbors (per pair):
        preserves per-target score histograms, destroys score-to-neighbor
        correspondence. D2.8 v2 §5.1 repair: exact integer-segment
        permutation — the previous float32 composite key (dst.float()*1e7 +
        tie_break) collided on large graphs and degenerated toward identity;
        it is gone."""
        return shuffle_scores_within_target(scores, edge_index,
                                            MISMATCH_PERM_SEED)

    def _pair_mask_per_target(self, scores, edge_index, num_nodes, causal):
        """D2.8 v2 §5.2: per-target remove/keep — each target independently
        has the requested fraction of ITS OWN real neighbors selected; the
        null is preserved; the remaining real-neighbor composition is
        renormalized by the softmax itself."""
        op, pct = causal.split("_per_target_")
        return per_target_edge_mask(scores, edge_index, num_nodes, op,
                                    float(pct) / 100.0, MISMATCH_PERM_SEED)

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def _augment_edges(self, edge_index, num_nodes, causal):
        """Eval-time random-edge injection for the SIDE branch only
        (D2.7-F stress test, plan §46): noise_10 / noise_25 append random
        node pairs to a COPY of the observed edge list. The parent path
        always consumes the original graph."""
        pct = int(causal.split("_")[1]) / 100.0
        generator = torch.Generator().manual_seed(MISMATCH_PERM_SEED)
        n_add = max(1, int(edge_index.size(1) * pct))
        src_r = torch.randint(0, num_nodes, (n_add,), generator=generator).to(edge_index.device)
        dst_r = torch.randint(0, num_nodes, (n_add,), generator=generator).to(edge_index.device)
        noise = torch.stack([src_r, dst_r])
        return torch.cat([edge_index, noise], dim=1)

    def forward(self, x, edge_index=None, causal="full"):
        assert causal in CAUSAL_OVERRIDES, causal
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        num_nodes = int(x.size(0))
        f_block, z_base = self._parent_ctx(x, edge_index, num_nodes)
        if causal == "side_off" or self.mode == "a0_base":
            return z_base, None, None, x.new_tensor(0.0), {}
        side_ei = edge_index
        if causal.startswith("noise_"):
            side_ei = self._augment_edges(edge_index, num_nodes, causal)
        m = self._side_messages(f_block, side_ei, num_nodes, causal)
        z = torch.cat([z_base, m[:, 0], m[:, 1], m[:, 2]], dim=-1)
        return z, None, None, x.new_tensor(0.0), {}

    @torch.no_grad()
    def inference(self, x, edge_index=None, device=None, batch_size=65536):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        else:
            edge_index = edge_index.to(device)
        z, _, _, _, _ = self.forward(x, edge_index)
        return z.detach().cpu()

    # ------------------------------------------------------------------
    # Edge diagnostics (plan §23-§26): scores, null mass, ranking stats
    # ------------------------------------------------------------------

    @torch.no_grad()
    def edge_scores_and_mass(self, f_block, edge_index, num_nodes,
                             a, b) -> dict:
        """Per-pair edge scores, alpha (real-neighbor weight), null mass."""
        num_edges = int(edge_index.size(1))
        dst = edge_index[1]
        s = self._pair_scores_chunked(f_block, edge_index, a, b, num_edges)
        null = self._null_scores(f_block, a, b)
        max_i = torch.zeros(num_nodes, dtype=s.dtype, device=s.device)
        seg_max = torch.zeros(num_nodes, dtype=s.dtype, device=s.device)
        seg_max = seg_max.scatter_reduce(0, dst, s, reduce="amax", include_self=False)
        max_i = torch.maximum(max_i, seg_max)
        denom = torch.exp(null - max_i)
        denom = denom.scatter_add(
            0, dst, torch.exp(s - max_i[dst]))
        alpha = torch.exp(s - max_i[dst]) / denom[dst]
        null_mass = torch.exp(null - max_i) / denom
        return {"scores": s, "alpha": alpha, "null_mass": null_mass}

    def ranking_stats(self, alpha: torch.Tensor, edge_index, num_nodes,
                      null_mass: torch.Tensor) -> dict:
        """Entropy / Gini / top-k mass per target (plan §23)."""
        dst = edge_index[1]
        deg = torch.bincount(dst, minlength=num_nodes).to(alpha.dtype)
        total = alpha.new_zeros(num_nodes)
        total = total.scatter_add(0, dst, alpha)
        # per-target: normalized entropy over real neighbors + null
        def _seg_stats(values, idx, num_nodes):
            z = values.new_zeros(num_nodes)
            z = z.scatter_add(0, idx, values)
            return z

        real_mass = _seg_stats(alpha, dst, num_nodes)
        p_ent = alpha / (real_mass[dst] + 1e-12)
        ent_contrib = -alpha * torch.log(p_ent + 1e-12)
        entropy = _seg_stats(ent_contrib, dst, num_nodes)
        # Gini over real neighbors per target: mean |alpha_i - alpha_j| /
        # (2 * mean). Exact Gini needs per-target sorting; the audit script
        # computes it from the exported per-edge alpha.
        eff_count = torch.exp(entropy)
        return {
            "real_mass": real_mass, "null_mass": null_mass,
            "entropy": entropy, "eff_neighbor_count": eff_count,
            "deg": deg,
        }

    @torch.no_grad()
    def compute_edge_diagnostics(self, x, edge_index) -> dict:
        """JSON-safe per-pair aggregates: null mass, entropy, top-10/25%
        mass, cosine correlation, source/target degree correlation."""
        self.eval()
        num_nodes = int(x.size(0))
        f_block, _z_base = self._parent_ctx(x, edge_index, num_nodes)
        src, dst = edge_index[0], edge_index[1]
        num_edges = int(edge_index.size(1))
        src_deg = torch.bincount(src, minlength=num_nodes).to(torch.float32)
        dst_deg = torch.bincount(dst, minlength=num_nodes).to(torch.float32)
        out = {"mode": self.mode}
        for a in range(3):
            for b in range(3):
                if self.mode == "diag_edge" and a != b:
                    continue
                stats = self.edge_scores_and_mass(f_block, edge_index, num_nodes, a, b)
                s, alpha, null = stats["scores"], stats["alpha"], stats["null_mass"]
                real = 1.0 - null
                # top-k mass (over real neighbors per target)
                top10 = torch.quantile(s, 0.90) if num_edges else s.new_tensor(0.0)
                top25 = torch.quantile(s, 0.75) if num_edges else s.new_tensor(0.0)
                mass_top10 = alpha[s >= top10].sum() / (alpha.sum() + 1e-12)
                mass_top25 = alpha[s >= top25].sum() / (alpha.sum() + 1e-12)
                # heuristic correlations
                cos_ab = torch.nn.functional.cosine_similarity(
                    f_block[dst, b], f_block[src, a], dim=-1)
                out[f"pair_{a}{b}"] = {
                    "null_mass_mean": float(null.mean().item()),
                    "null_mass_frac_zero": float((null < 0.05).float().mean().item()),
                    "null_mass_frac_one": float((null > 0.95).float().mean().item()),
                    "real_mass_mean": float(real.mean().item()),
                    "alpha_entropy_mean": float(
                        -(alpha * torch.log(alpha + 1e-12)).sum().item()
                        / max(alpha.numel(), 1)),
                    "mass_top10": float(mass_top10.item()),
                    "mass_top25": float(mass_top25.item()),
                    "corr_score_cos": float(
                        torch.corrcoef(torch.stack([s, cos_ab]))[0, 1].item()
                        if num_edges > 2 else 0.0),
                    "corr_score_src_deg": float(
                        torch.corrcoef(torch.stack([s, src_deg[src]]))[0, 1].item()
                        if num_edges > 2 else 0.0),
                    "corr_score_dst_deg": float(
                        torch.corrcoef(torch.stack([s, dst_deg[dst]]))[0, 1].item()
                        if num_edges > 2 else 0.0),
                }
        return out

    @torch.no_grad()
    def injected_edge_utility(self, x, edge_index, pct: float) -> dict:
        """D2.7-F diagnostics: alpha mass assigned to INJECTED random edges
        vs original edges (mean over the 9 pairs), plus the injected edges'
        share of each pair's top-10%/top-25% alpha."""
        self.eval()
        num_nodes = int(x.size(0))
        f_block, _ = self._parent_ctx(x, edge_index, num_nodes)
        aug = self._augment_edges(edge_index, num_nodes, f"noise_{int(pct)}")
        n_orig = int(edge_index.size(1))
        n_add = int(aug.size(1)) - n_orig
        inj_alpha, orig_alpha = [], []
        inj_top10, inj_top25 = 0, 0
        for a in range(3):
            for b in range(3):
                if self.mode == "diag_edge" and a != b:
                    continue
                s = self._pair_scores_chunked(f_block, aug, a, b, int(aug.size(1)))
                stats_alpha = self._alpha_from_scores(f_block, aug, num_nodes, a, b, s)
                inj_alpha.append(stats_alpha[n_orig:].mean().item())
                orig_alpha.append(stats_alpha[:n_orig].mean().item())
                top10_q = torch.quantile(stats_alpha, 0.90)
                top25_q = torch.quantile(stats_alpha, 0.75)
                inj_top10 += int((stats_alpha[n_orig:] >= top10_q).sum().item())
                inj_top25 += int((stats_alpha[n_orig:] >= top25_q).sum().item())
        return {
            "injected_mean_alpha": float(sum(inj_alpha) / len(inj_alpha)),
            "original_mean_alpha": float(sum(orig_alpha) / len(orig_alpha)),
            "injected_frac_top10": inj_top10 / max(9 * n_add, 1),
            "injected_frac_top25": inj_top25 / max(9 * n_add, 1),
        }

    def _alpha_from_scores(self, f_block, edge_index, num_nodes, a, b, s):
        """alpha per edge for pair (a,b) given precomputed scores."""
        dst = edge_index[1]
        null = self._null_scores(f_block, a, b)
        max_i = torch.zeros(num_nodes, dtype=s.dtype, device=s.device)
        seg_max = torch.zeros(num_nodes, dtype=s.dtype, device=s.device)
        seg_max = seg_max.scatter_reduce(0, dst, s, reduce="amax", include_self=False)
        max_i = torch.maximum(max_i, seg_max)
        denom = torch.exp(null - max_i)
        denom = denom.scatter_add(0, dst, torch.exp(s - max_i[dst]))
        return torch.exp(s - max_i[dst]) / denom[dst]

    @torch.no_grad()
    def export_pair_scores(self, x, edge_index) -> dict:
        """Per-edge raw scores + alpha + null mass for the audit script
        (all pairs; [E] tensors on CPU)."""
        self.eval()
        num_nodes = int(x.size(0))
        f_block, _z_base = self._parent_ctx(x, edge_index, num_nodes)
        out = {}
        for a in range(3):
            for b in range(3):
                if self.mode == "diag_edge" and a != b:
                    continue
                stats = self.edge_scores_and_mass(f_block, edge_index, num_nodes, a, b)
                out[f"pair_{a}{b}"] = {
                    "scores": stats["scores"].cpu(),
                    "alpha": stats["alpha"].cpu(),
                    "null_mass": stats["null_mass"].cpu(),
                }
        return out
