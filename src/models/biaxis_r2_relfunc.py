"""R2-Design-2.8 v2 relational-function decomposition model
(docs/BiAxis_R2_Design_2_8_v2_Identifiable_Relational_Function_Decomposition.md).

Unified diagnostic formulation (v2 §1):

    m_i^b = sum_a lambda_i^{ab} * r_i^{ab} * sum_j pi_ji^{ab} * Ohat_ji^{ab}(U_a F_j^a)

        lambda : simplex over source factor a        (Rule III)
        pi     : simplex over real neighbors only    (Rule II, no null token)
        r      : exposure scalar in [0,1]            (Rule I)
        Ohat   : content operator, NormMatch in the  (Rule IV)
                 primary diagnostic

Identifiability rules (v2 §3):
    Rule I  exposure tests: pi uniform, O = U_a, lambda = 1/3.
    Rule II composition: real-neighbor softmax only (no null inside pi);
            the chosen exposure E* is loaded and frozen.
    Rule III channel: lambda = Softmax_a (simplex; cannot become a second
            exposure gate).
    Rule IV primary operator diagnostic uses NormMatch: content vs magnitude.
    Rule V  staged freezing: previously selected functions are frozen when
            testing a new one; joint co-training only as a secondary
            confirmation after a single-mechanism GO.

Config knobs (configs/model/biaxis_r2_relfunc.yaml):
    exposure:    fixed_full | node | target | source | pair
    composition: uniform | generic | target | source | pair
    channel:     mean | softmax | concat | attn
    operator:    linear | static_pair | target_film | edge_film | basis
    norm_match / mean_dup / uniform_router / target_router / basis_k
    freeze_exposure / freeze_composition / freeze_channel / freeze_operator

Discipline:
    - A0 parent frozen, always eval; side branch only ADDS (v2 strong-parent).
    - r / pi / lambda / operator routing are shared-predictor outputs, never
      free node/edge tables (v2 §2).
    - causal overrides never modify trained weights.
    - No Test: labels never enter the model.
    - observed graph as support only: no edge addition, no role labels,
      no learned adjacency (collision guardrails).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_r2_relfunc_components import (
    MISMATCH_PERM_SEED,
    BasisOperator,
    CompScorer,
    ConcatMixer,
    EdgeFilm,
    ExposureNet,
    GenericCompScorer,
    SourceAttnMixer,
    SourceScalarMix,
    StaticPairResidual,
    TargetFilm,
    norm_match,
    per_target_edge_mask,
    shuffle_scores_within_target,
    solve_exposure_width,
    solve_generic_comp_width,
    within_target_perm,
)

FACTOR_NAMES = ("C", "Pt", "Pv")

EXPOSURE_KINDS = ("fixed_full", "node", "target", "source", "pair")
COMPOSITION_KINDS = ("uniform", "generic", "target", "source", "pair")
CHANNEL_KINDS = ("mean", "softmax", "concat", "attn")
OPERATOR_KINDS = ("linear", "static_pair", "target_film", "edge_film", "basis")

ALL_PAIRS = tuple((a, b) for a in range(3) for b in range(3))

CAUSAL_OVERRIDES = (
    "full", "side_off",
    "within_target_shuffle",
    "remove_top_per_target_10", "remove_top_per_target_25", "remove_top_per_target_50",
    "remove_random_per_target_10", "remove_random_per_target_25", "remove_random_per_target_50",
    "remove_bottom_per_target_10", "remove_bottom_per_target_25", "remove_bottom_per_target_50",
    "keep_top_per_target_25", "keep_top_per_target_50",
    "source_shuffle", "factor_id_shuffle",
    "film_neutralize", "operator_shuffle", "router_uniformize", "router_permute",
)

_COMP_CAUSAL = ("within_target_shuffle",) + tuple(
    k for k in CAUSAL_OVERRIDES if k.startswith(("remove_", "keep_")))


def _parse_per_target(causal: str) -> tuple[str, float]:
    """'remove_top_per_target_10' -> ('remove_top', 0.10)."""
    op, pct = causal.split("_per_target_")
    return op, float(pct) / 100.0


class Model(nn.Module):
    """Relational-function decomposition model wrapping a frozen A0 parent."""

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
        self.num_classes = int(data_info["num_classes"])

        d = self.factor_dim
        h = self.hidden_dim
        dropout = float(cfg.model.get("dropout", 0.2))
        activation = str(cfg.model.get("activation", "gelu"))
        norm = str(cfg.model.get("norm", "layernorm"))
        self.type_dim = t = int(cfg.model.get("type_dim", 16))

        self.exposure_kind = str(cfg.model.get("exposure", "fixed_full"))
        self.composition_kind = str(cfg.model.get("composition", "uniform"))
        self.channel_kind = str(cfg.model.get("channel", "mean"))
        self.operator_kind = str(cfg.model.get("operator", "linear"))
        assert self.exposure_kind in EXPOSURE_KINDS, self.exposure_kind
        assert self.composition_kind in COMPOSITION_KINDS, self.composition_kind
        assert self.channel_kind in CHANNEL_KINDS, self.channel_kind
        assert self.operator_kind in OPERATOR_KINDS, self.operator_kind

        self.norm_match = bool(cfg.model.get("norm_match", True))
        self.mean_dup = bool(cfg.model.get("mean_dup", False))
        self.basis_k = int(cfg.model.get("basis_k", 4))
        self.freeze_exposure = bool(cfg.model.get("freeze_exposure", False))
        self.freeze_composition = bool(cfg.model.get("freeze_composition", False))
        self.freeze_channel = bool(cfg.model.get("freeze_channel", False))
        self.freeze_operator = bool(cfg.model.get("freeze_operator", False))

        # --- base semantic content: v_j^a = U_a F_j^a (v2 §1) --------------
        self.payload = nn.ModuleList([nn.Linear(d, d, bias=False)
                                      for _ in range(3)])

        # --- exposure predictor (v2 §7) -------------------------------------
        self.exposure_net: ExposureNet | None = None
        self.exposure_emb: nn.Embedding | None = None
        if self.exposure_kind != "fixed_full":
            in_dim = {"node": d, "target": d + t, "source": d + t,
                      "pair": d + 2 * t}[self.exposure_kind]
            # capacity match every granularity against the pair predictor
            target_params = (d + 2 * t) * d + d + d * 1 + 1
            hidden = solve_exposure_width(in_dim, d, target_params)
            self.exposure_net = ExposureNet(in_dim, hidden, activation, norm)
            self.exposure_emb = nn.Embedding(3, t)

        # --- composition scorer (v2 §8) -------------------------------------
        self.comp_net: CompScorer | GenericCompScorer | None = None
        self.comp_local_proj: nn.Linear | None = None
        self.comp_emb: nn.Embedding | None = None
        if self.composition_kind == "generic":
            self.comp_local_proj = nn.Linear(3 * d, d)
            ref = (4 * d + 2 * t) * 2 * d + 2 * d + 2 * d * d + d + d * 1 + 1
            width = solve_generic_comp_width(d, ref)
            self.comp_net = GenericCompScorer(d, width, dropout, activation, norm)
            self._comp_match = {"pair_scorer_params": int(ref),
                                "solved_width": width}
        elif self.composition_kind in ("target", "source"):
            self.comp_net = CompScorer(4 * d + t, d, dropout, activation, norm)
            self.comp_emb = nn.Embedding(3, t)
        elif self.composition_kind == "pair":
            self.comp_net = CompScorer(4 * d + 2 * t, d, dropout, activation, norm)
            self.comp_emb = nn.Embedding(3, t)

        # --- source-channel integration (v2 §9) -----------------------------
        self.channel_net: nn.Module | None = None
        self.channel_emb: nn.Embedding | None = None
        if self.channel_kind == "softmax":
            self.channel_net = SourceScalarMix(d, t, activation, norm)
            self.channel_emb = nn.Embedding(3, t)
        elif self.channel_kind == "concat":
            self.channel_net = ConcatMixer(d, dropout, activation, norm)
        elif self.channel_kind == "attn":
            self.channel_net = SourceAttnMixer(d, dropout=0.1)

        # --- functional operator (v2 §10) -----------------------------------
        self.operator_net: nn.Module | None = None
        self.operator_emb: nn.Embedding | None = None
        if self.operator_kind == "static_pair":
            self.operator_net = StaticPairResidual(d, activation, norm)
        elif self.operator_kind in ("target_film", "edge_film"):
            cls = TargetFilm if self.operator_kind == "target_film" else EdgeFilm
            self.operator_net = cls(d, t, activation, norm)
            self.operator_emb = nn.Embedding(3, t)
        elif self.operator_kind == "basis":
            self.operator_net = BasisOperator(
                d, t, k=self.basis_k,
                uniform_router=bool(cfg.model.get("uniform_router", False)),
                target_router=bool(cfg.model.get("target_router", False)),
                activation=activation, norm=norm)
            self.operator_emb = nn.Embedding(3, t)

        self.out_dim = h + 3 * d
        self.side_parameter_count = int(sum(p.numel() for p in self.parameters()))
        self.parameter_count = self.side_parameter_count
        self._apply_freezes()

    # ------------------------------------------------------------------
    # Staged freezing (v2 Rule V)
    # ------------------------------------------------------------------

    def _frozen_groups(self) -> dict:
        groups = {}
        if self.freeze_exposure:
            groups["exposure"] = [self.exposure_net, self.exposure_emb, self.payload]
        if self.freeze_composition:
            groups["composition"] = [self.comp_net, self.comp_local_proj, self.comp_emb]
        if self.freeze_channel:
            groups["channel"] = [self.channel_net, self.channel_emb]
        if self.freeze_operator:
            groups["operator"] = [self.operator_net, self.operator_emb]
        return groups

    def _apply_freezes(self) -> None:
        for _name, modules in self._frozen_groups().items():
            for m in modules:
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad_(False)

    def frozen_parameter_count(self) -> int:
        return sum(p.numel() for _name, modules in self._frozen_groups().items()
                   for m in modules if m is not None for p in m.parameters())

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
    # Exposure (v2 §7)
    # ------------------------------------------------------------------

    def _exposure_values(self, f_block):
        """Distinct r tensors keyed by granularity; r in (0,1) via sigmoid.
        fixed_full -> {} (r == 1 everywhere)."""
        if self.exposure_kind == "fixed_full":
            return {}
        n = f_block.size(0)
        emb = self.exposure_emb.weight
        if self.exposure_kind == "node":
            u = f_block.mean(dim=1)  # node-local summary
            return {"node": self.exposure_net(u)}
        if self.exposure_kind == "target":
            out = {}
            for b in range(3):
                u = torch.cat([f_block[:, b],
                               emb[b].unsqueeze(0).expand(n, -1)], dim=-1)
                out[f"target_{b}"] = self.exposure_net(u)
            return out
        if self.exposure_kind == "source":
            out = {}
            for a in range(3):
                u = torch.cat([f_block[:, a],
                               emb[a].unsqueeze(0).expand(n, -1)], dim=-1)
                out[f"source_{a}"] = self.exposure_net(u)
            return out
        out = {}
        for (a, b) in ALL_PAIRS:
            u = torch.cat([f_block[:, b],
                           emb[a].unsqueeze(0).expand(n, -1),
                           emb[b].unsqueeze(0).expand(n, -1)], dim=-1)
            out[f"pair_{a}{b}"] = self.exposure_net(u)
        return out

    def _r_for_pair(self, rvals, a, b):
        """[N] exposure for pair a->b (None = 1.0)."""
        if self.exposure_kind == "fixed_full":
            return None
        if self.exposure_kind == "node":
            return rvals["node"]
        if self.exposure_kind == "target":
            return rvals[f"target_{b}"]
        if self.exposure_kind == "source":
            return rvals[f"source_{a}"]
        return rvals[f"pair_{a}{b}"]

    # ------------------------------------------------------------------
    # Composition scores (v2 §8)
    # ------------------------------------------------------------------

    def _comp_target_scores(self, f_block, edge_index, b, num_edges):
        src, dst = edge_index[0], edge_index[1]
        e_b = self.comp_emb.weight[b]
        out = torch.zeros(num_edges, dtype=f_block.dtype, device=f_block.device)
        for start in range(0, num_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, num_edges)
            s_c, d_c = src[start:end], dst[start:end]
            fb, fa = f_block[d_c, b], f_block[s_c, b]
            u = torch.cat([fb, fa, fb * fa, (fb - fa).abs(),
                           e_b.unsqueeze(0).expand(end - start, -1)], dim=-1)
            out[start:end] = self.comp_net(u)
        return out

    def _comp_source_scores(self, f_block, edge_index, a, num_edges):
        src, dst = edge_index[0], edge_index[1]
        e_a = self.comp_emb.weight[a]
        out = torch.zeros(num_edges, dtype=f_block.dtype, device=f_block.device)
        for start in range(0, num_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, num_edges)
            s_c, d_c = src[start:end], dst[start:end]
            fb, fa = f_block[d_c, a], f_block[s_c, a]
            u = torch.cat([fb, fa, fb * fa, (fb - fa).abs(),
                           e_a.unsqueeze(0).expand(end - start, -1)], dim=-1)
            out[start:end] = self.comp_net(u)
        return out

    def _comp_pair_scores(self, f_block, edge_index, a, b, num_edges):
        src, dst = edge_index[0], edge_index[1]
        e_a = self.comp_emb.weight[a]
        e_b = self.comp_emb.weight[b]
        out = torch.zeros(num_edges, dtype=f_block.dtype, device=f_block.device)
        for start in range(0, num_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, num_edges)
            s_c, d_c = src[start:end], dst[start:end]
            fb, fa = f_block[d_c, b], f_block[s_c, a]
            u = torch.cat([fb, fa, fb * fa, (fb - fa).abs(),
                           e_a.unsqueeze(0).expand(end - start, -1),
                           e_b.unsqueeze(0).expand(end - start, -1)], dim=-1)
            out[start:end] = self.comp_net(u)
        return out

    def _comp_generic_scores(self, f_block, edge_index, num_nodes, num_edges):
        src, dst = edge_index[0], edge_index[1]
        z0 = self.comp_local_proj(f_block.reshape(num_nodes, 3 * self.factor_dim))
        out = torch.zeros(num_edges, dtype=f_block.dtype, device=f_block.device)
        for start in range(0, num_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, num_edges)
            s_c, d_c = src[start:end], dst[start:end]
            zi, zj = z0[d_c], z0[s_c]
            u = torch.cat([zi, zj, zi * zj, (zi - zj).abs()], dim=-1)
            out[start:end] = self.comp_net(u)
        return out

    def _shared_comp_scores(self, f_block, edge_index, num_nodes, num_edges):
        """Scores shared across pairs (computed once per forward, outside the
        per-pair checkpoint); pair granularity is None (computed per pair)."""
        kind = self.composition_kind
        if kind == "uniform" or kind == "pair":
            return {}
        if kind == "generic":
            return {"generic": self._comp_generic_scores(
                f_block, edge_index, num_nodes, num_edges)}
        if kind == "target":
            return {"target": {b: self._comp_target_scores(
                f_block, edge_index, b, num_edges) for b in range(3)}}
        return {"source": {a: self._comp_source_scores(
            f_block, edge_index, a, num_edges) for a in range(3)}}

    def _comp_scores_for_pair(self, f_block, edge_index, a, b, num_edges, shared):
        kind = self.composition_kind
        if kind == "uniform":
            return None
        if kind == "generic":
            return shared["generic"]
        if kind == "target":
            return shared["target"][b]
        if kind == "source":
            return shared["source"][a]
        return self._comp_pair_scores(f_block, edge_index, a, b, num_edges)

    # ------------------------------------------------------------------
    # Operator application (v2 §10) — inside the chunked scatter pass
    # ------------------------------------------------------------------

    def _operator_film(self, fb, g_src, g_dst, a, b, causal):
        """FiLM modulation [C, 2d] computed from the (possibly permuted)
        gathered features; None = film_neutralize (identity)."""
        c = int(g_src.size(0))
        e_a = self.operator_emb.weight[a]
        e_b = self.operator_emb.weight[b]
        if causal == "film_neutralize":
            return None
        if self.operator_kind == "target_film":
            u = torch.cat([fb[g_dst, b],
                           e_a.unsqueeze(0).expand(c, -1),
                           e_b.unsqueeze(0).expand(c, -1)], dim=-1)
        else:
            u = torch.cat([fb[g_dst, b], fb[g_src, a],
                           fb[g_dst, b] * fb[g_src, a],
                           (fb[g_dst, b] - fb[g_src, a]).abs(),
                           e_a.unsqueeze(0).expand(c, -1),
                           e_b.unsqueeze(0).expand(c, -1)], dim=-1)
        return self.operator_net(u)

    def _operator_basis(self, fb, v, g_src, g_dst, a, b, causal):
        """Basis-operator output [C, d] and router assignment [C, K]."""
        c = int(g_src.size(0))
        k = self.operator_net.k
        if causal == "router_uniformize" or self.operator_net.uniform_router:
            q = torch.full((c, k), 1.0 / k, dtype=fb.dtype, device=fb.device)
            return self.operator_net.apply(v, q), q
        e_a = self.operator_emb.weight[a]
        e_b = self.operator_emb.weight[b]
        if self.operator_net.target_router:
            u = torch.cat([fb[g_dst, b],
                           e_a.unsqueeze(0).expand(c, -1),
                           e_b.unsqueeze(0).expand(c, -1)], dim=-1)
        else:
            u = torch.cat([fb[g_dst, b], fb[g_src, a],
                           e_a.unsqueeze(0).expand(c, -1),
                           e_b.unsqueeze(0).expand(c, -1)], dim=-1)
        q = torch.softmax(self.operator_net.router_logits(u), dim=-1)
        return self.operator_net.apply(v, q), q

    # ------------------------------------------------------------------
    # Per-pair message: sum_j pi_ji * Ohat(v_j^a)  (real neighbors only)
    # ------------------------------------------------------------------

    def _real_pair_message(self, f_block, edge_index, num_nodes, num_edges,
                           a, b, payload_a, s, causal):
        """m_ab = sum_j pi_ji * Ohat_ji(v_j^a), pi = Softmax_{j in N(i)}(s_ji)
        over REAL neighbors only (no null inside the softmax, v2 Rule II).
        Exposure r multiplies afterwards (Rule I keeps it the only explicit
        graph-amplitude variable)."""
        src, dst = edge_index[0], edge_index[1]
        if s is None:
            s = torch.zeros(num_edges, dtype=f_block.dtype, device=f_block.device)
        if causal in _COMP_CAUSAL and self.composition_kind == "uniform":
            raise ValueError(f"{causal} needs a learned composition")
        if causal == "within_target_shuffle":
            s = shuffle_scores_within_target(s, edge_index, MISMATCH_PERM_SEED)
        edge_mask = None
        if causal.startswith(("remove_", "keep_")):
            op, pct = _parse_per_target(causal)
            edge_mask = per_target_edge_mask(s, edge_index, num_nodes, op, pct,
                                             MISMATCH_PERM_SEED)
        if edge_mask is not None:
            src, dst = src[edge_mask], dst[edge_mask]
            s = s[edge_mask]
            num_edges = int(src.size(0))
        if num_edges == 0:
            return torch.zeros(num_nodes, self.factor_dim, dtype=f_block.dtype,
                               device=f_block.device)

        # pass 1: per-target max over kept real neighbors
        max_i = torch.full((num_nodes,), float("-inf"), dtype=s.dtype,
                           device=s.device)
        seg = torch.zeros(num_nodes, dtype=s.dtype, device=s.device)
        seg = seg.scatter_reduce(0, dst, s, reduce="amax", include_self=False)
        max_i = torch.maximum(max_i, seg)
        max_i = torch.where(torch.isfinite(max_i), max_i, torch.zeros_like(max_i))

        # pass 2: denom = sum_j exp(s_ji - max_i)
        denom = torch.zeros(num_nodes, dtype=s.dtype, device=s.device)
        for start in range(0, num_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, num_edges)
            d_c = dst[start:end]
            denom = denom.scatter_add(0, d_c, torch.exp(s[start:end] - max_i[d_c]))

        # operator-condition permutation (edge-conditioned operators only)
        op_perm = None
        if causal in ("operator_shuffle", "router_permute"):
            op_perm = within_target_perm(torch.stack([src, dst]), num_edges,
                                         MISMATCH_PERM_SEED)

        # per-node operator pieces (target-conditioned kinds), computed once
        d = self.factor_dim
        static_v = None
        target_gb = None
        if self.operator_kind == "static_pair":
            static_v = self.operator_net(payload_a, a, b)  # [N, d]
        elif self.operator_kind == "target_film" and causal != "film_neutralize":
            n = f_block.size(0)
            e_a = self.operator_emb.weight[a]
            e_b = self.operator_emb.weight[b]
            u = torch.cat([f_block[:, b],
                           e_a.unsqueeze(0).expand(n, -1),
                           e_b.unsqueeze(0).expand(n, -1)], dim=-1)
            target_gb = self.operator_net(u)  # [N, 2d]

        # pass 3: operator transform per chunk + weighted scatter
        m = torch.zeros(num_nodes, d, dtype=f_block.dtype, device=f_block.device)
        for start in range(0, num_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, num_edges)
            s_c, d_c = src[start:end], dst[start:end]
            v = payload_a[s_c]
            g_src, g_dst = s_c, d_c
            if op_perm is not None:
                g_src = src[op_perm[start:end]]
                g_dst = dst[op_perm[start:end]]
            kind = self.operator_kind
            if kind == "linear":
                v_out = v
            elif kind == "static_pair":
                v_out = static_v[s_c]
            elif kind in ("target_film", "edge_film"):
                if target_gb is not None:
                    dg, beta = target_gb[d_c, :d], target_gb[d_c, d:]
                else:
                    gb = self._operator_film(f_block, g_src, g_dst, a, b, causal)
                    dg = beta = None
                    if gb is not None:
                        dg, beta = gb[:, :d], gb[:, d:]
                v_out = v if dg is None else (1.0 + dg) * v + beta
            elif kind == "basis":
                v_out, _q = self._operator_basis(f_block, v, g_src, g_dst,
                                                 a, b, causal)
            else:
                raise ValueError(kind)
            if kind != "linear" and self.norm_match:
                v_out = norm_match(v_out, v)
            alpha = torch.exp(s[start:end] - max_i[d_c]) / denom[d_c]
            contrib = alpha.unsqueeze(-1) * v_out
            m = m.scatter_add(0, d_c.unsqueeze(-1).expand_as(contrib), contrib)
        return m

    def _pair_message_ckpt(self, f_block, edge_index, num_nodes, num_edges,
                           a, b, payload_a, s, causal):
        """Per-pair message with activation checkpointing (grad-enabled pass
        only; ele-fashion OOM guard from D2.7: the chunked intermediates of
        all 9 pairs must not coexist until backward)."""

        def _run(fb, ei, s_in):
            if s_in.numel() == 0:
                s_in = None
                if self.composition_kind == "pair":
                    s_in = self._comp_pair_scores(fb, ei, a, b, int(ei.size(1)))
            return self._real_pair_message(fb, ei, num_nodes, num_edges,
                                           a, b, payload_a, s_in, causal)

        sentinel = s if s is not None else torch.empty(
            0, dtype=f_block.dtype, device=f_block.device)
        if self.training or torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                _run, f_block, edge_index, sentinel, use_reentrant=False)
        return _run(f_block, edge_index, sentinel)

    # ------------------------------------------------------------------
    # Source-channel integration (v2 §9)
    # ------------------------------------------------------------------

    def _channel_mix(self, f_block, m_list, b):
        """m_i^b = channel(m^{C->b}, m^{Pt->b}, m^{Pv->b}) for target b."""
        kind = self.channel_kind
        m0, m1, m2 = m_list
        if kind == "mean":
            return (m0 + m1 + m2) / 3.0
        if kind == "softmax":
            lam = self.channel_net(f_block[:, b], self.channel_emb.weight)  # [N,3]
            return lam[:, 0:1] * m0 + lam[:, 1:2] * m1 + lam[:, 2:3] * m2
        if kind == "concat":
            if self.mean_dup:
                mean = (m0 + m1 + m2) / 3.0
                x = torch.cat([mean, mean, mean], dim=-1)
            else:
                x = torch.cat([m0, m1, m2], dim=-1)
            return self.channel_net(x, b)
        # attn
        tokens = torch.stack(m_list, dim=1)  # [N, 3, d]
        if self.mean_dup:
            mean = tokens.mean(dim=1, keepdim=True)
            tokens = mean.expand(-1, 3, -1)
        return self.channel_net(f_block[:, b], tokens)

    # ------------------------------------------------------------------
    # Side messages
    # ------------------------------------------------------------------

    def _side_messages(self, f_block, edge_index, num_nodes, causal="full",
                       return_components=False):
        """m [N, 3, d]: m[:, b] = channel-mixed message for target factor b.
        return_components=True also returns the per-pair pre-channel messages
        and exposure values (diagnostic exports only, no_grad)."""
        d = self.factor_dim
        num_edges = int(edge_index.size(1))

        if causal == "film_neutralize" and self.operator_kind not in (
                "target_film", "edge_film"):
            raise ValueError("film_neutralize needs a FiLM operator")
        if causal in ("operator_shuffle", "router_permute") and \
                self.operator_kind not in ("edge_film", "basis"):
            raise ValueError(f"{causal} needs an edge-conditioned operator "
                             f"(got {self.operator_kind})")
        if causal == "router_uniformize" and self.operator_kind != "basis":
            raise ValueError("router_uniformize needs the basis operator")

        pair_perm = {(a, b): (a, b) for (a, b) in ALL_PAIRS}
        if causal == "factor_id_shuffle":
            from src.analysis.perf_r2d15_utils import fixed_node_permutation

            perm = fixed_node_permutation(3, MISMATCH_PERM_SEED)
            f_block = f_block[:, perm]
            pair_perm = {(a, b): (int(perm[a]), int(perm[b]))
                         for (a, b) in ALL_PAIRS}
        if causal == "source_shuffle":
            from src.analysis.perf_r2d15_utils import fixed_node_permutation

            f_block = f_block[fixed_node_permutation(num_nodes, MISMATCH_PERM_SEED)]

        payloads = [self.payload[a](f_block[:, a]) for a in range(3)]
        rvals = self._exposure_values(f_block)
        shared = self._shared_comp_scores(f_block, edge_index, num_nodes, num_edges)

        msgs = []
        components = {"m_ab": {}, "r": rvals}
        for b in range(3):
            m_ab = []
            for a in range(3):
                pa, pb = pair_perm[(a, b)]
                # pair-granularity scores are computed INSIDE the checkpointed
                # pair function (else the chunked scorer graphs of all 9 pairs
                # coexist until backward — ~21GB OOM on ele-fashion, D2.7
                # pitfall); shared granularities pass their precomputed scores
                s = (None if self.composition_kind == "pair"
                     else self._comp_scores_for_pair(f_block, edge_index, pa, pb,
                                                     num_edges, shared))
                m0 = self._pair_message_ckpt(f_block, edge_index, num_nodes,
                                             num_edges, pa, pb, payloads[a],
                                             s, causal)
                r = self._r_for_pair(rvals, pa, pb)
                m_ab.append(m0 if r is None else r.unsqueeze(-1) * m0)
            if self.channel_kind == "attn" and \
                    (self.training or torch.is_grad_enabled()):
                # attention activations at ele-fashion batch size (97K) are
                # ~10GB+; checkpoint the mixer like the pair messages
                def _mix(fb, m0, m1, m2, bb):
                    return self._channel_mix(fb, [m0, m1, m2], bb)

                msgs.append(torch.utils.checkpoint.checkpoint(
                    _mix, f_block, m_ab[0], m_ab[1], m_ab[2], b,
                    use_reentrant=False))
            else:
                msgs.append(self._channel_mix(f_block, m_ab, b))
            if return_components:
                for a in range(3):
                    components["m_ab"][(a, b)] = m_ab[a]
        stacked = torch.stack(msgs, dim=1)
        if return_components:
            return stacked, components
        return stacked

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(self, x, edge_index=None, causal="full"):
        assert causal in CAUSAL_OVERRIDES, causal
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        num_nodes = int(x.size(0))
        f_block, z_base = self._parent_ctx(x, edge_index, num_nodes)
        if causal == "side_off":
            return z_base, None, None, x.new_tensor(0.0), {}
        m = self._side_messages(f_block, edge_index, num_nodes, causal)
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
    # Stage diagnostics (eval-only, no_grad; v2 §7/§8/§9/§10)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def export_exposure_stats(self, x, edge_index, train_idx, train_y) -> dict:
        """r mean/std/quantiles, frac<0.1/>0.9, degree correlation, per-
        factor/pair r matrix (diag vs off-diag), TRAIN-label-only class
        diagnostics (v2 §7)."""
        self.eval()
        num_nodes = int(x.size(0))
        f_block, _z = self._parent_ctx(x, edge_index, num_nodes)
        rvals = self._exposure_values(f_block)
        if not rvals:
            return {"exposure_kind": self.exposure_kind, "fixed_full": True}
        deg = torch.bincount(edge_index[1], minlength=num_nodes).to(torch.float32)
        per_key = {}
        for key, r in rvals.items():
            r = r.detach()
            per_key[key] = {
                "mean": float(r.mean().item()),
                "std": float(r.std().item()),
                "q10": float(torch.quantile(r, 0.10).item()),
                "q50": float(torch.quantile(r, 0.50).item()),
                "q90": float(torch.quantile(r, 0.90).item()),
                "frac_lt_0.1": float((r < 0.1).float().mean().item()),
                "frac_gt_0.9": float((r > 0.9).float().mean().item()),
                "degree_corr": float(
                    torch.corrcoef(torch.stack([r, deg]))[0, 1].item()
                    if num_nodes > 2 else 0.0),
            }
            # TRAIN-label-only class diagnostics
            tr = train_idx.to(r.device)
            cls = []
            for c in range(self.num_classes):
                mask = (train_y.to(r.device) == c)
                cls.append(round(float(r[tr[mask]].mean().item()), 6)
                           if mask.any() else None)
            per_key[key]["mean_r_train_class"] = cls
        # broadcast to the 9 pair cells
        r_mat = torch.zeros(3, 3, dtype=f_block.dtype, device=f_block.device)
        for (a, b) in ALL_PAIRS:
            r_mat[a, b] = self._r_for_pair(rvals, a, b).mean()
        return {
            "exposure_kind": self.exposure_kind,
            "per_key": per_key,
            "r_matrix_mean": [[round(float(r_mat[a, b].item()), 6)
                               for b in range(3)] for a in range(3)],
            "diag_mean": round(float(r_mat.diagonal().mean().item()), 6),
            "offdiag_mean": round(float(
                (r_mat.sum() - r_mat.diagonal().sum()).item() / 6.0), 6),
        }

    @torch.no_grad()
    def export_comp_stats(self, x, edge_index, subsample: int = 50000) -> dict:
        """Per-ranking score stats, per-target pi entropy, top-10/25% mass,
        pairwise Spearman between pair rankings (v2 §8 diagnostics)."""
        self.eval()
        num_nodes = int(x.size(0))
        f_block, _z = self._parent_ctx(x, edge_index, num_nodes)
        num_edges = int(edge_index.size(1))
        shared = self._shared_comp_scores(f_block, edge_index, num_nodes, num_edges)
        if self.composition_kind == "uniform":
            return {"composition_kind": "uniform"}
        dst = edge_index[1]
        out = {"composition_kind": self.composition_kind}
        scores_by_pair = {}
        for (a, b) in ALL_PAIRS:
            s = self._comp_scores_for_pair(f_block, edge_index, a, b, num_edges, shared)
            key = {"generic": "generic",
                   "target": f"target_{b}",
                   "source": f"source_{a}",
                   "pair": f"pair_{a}{b}"}[self.composition_kind]
            scores_by_pair[(a, b)] = (key, s)
            if key not in out:
                max_i = torch.zeros(num_nodes, dtype=s.dtype, device=s.device)
                seg = torch.zeros(num_nodes, dtype=s.dtype, device=s.device)
                seg = seg.scatter_reduce(0, dst, s, reduce="amax", include_self=False)
                max_i = torch.maximum(max_i, seg)
                denom = torch.zeros(num_nodes, dtype=s.dtype, device=s.device)
                denom = denom.scatter_add(0, dst, torch.exp(s - max_i[dst]))
                pi = torch.exp(s - max_i[dst]) / denom[dst]
                ent = torch.zeros(num_nodes, dtype=s.dtype, device=s.device)
                ent = ent.scatter_add(0, dst, -pi * torch.log(pi + 1e-12))
                ent = ent[denom > 0]
                top10_q = torch.quantile(s, 0.90) if num_edges else s.new_tensor(0.0)
                top25_q = torch.quantile(s, 0.75) if num_edges else s.new_tensor(0.0)
                out[key] = {
                    "score_mean": float(s.mean().item()),
                    "score_std": float(s.std().item()),
                    "pi_entropy_mean": float(ent.mean().item()),
                    "mass_top10": float((pi[s >= top10_q].sum() / (pi.sum() + 1e-12)).item()),
                    "mass_top25": float((pi[s >= top25_q].sum() / (pi.sum() + 1e-12)).item()),
                }
        if self.composition_kind == "pair" and num_edges > 100:
            # pairwise Spearman between the 9 pair rankings on a subsample
            import numpy as np
            from scipy.stats import rankdata

            idx = torch.randperm(num_edges, device=dst.device)[:subsample].cpu().numpy()
            arr = np.stack([scores_by_pair[(a, b)][1][idx].cpu().numpy()
                            for (a, b) in ALL_PAIRS])
            corr = np.corrcoef(rankdata(arr, axis=1))
            out["pair_spearman"] = [[round(float(v), 4) for v in row]
                                    for row in corr]
        return out

    @torch.no_grad()
    def export_channel_stats(self, x, edge_index) -> dict:
        """M1: mean lambda matrix + entropy; M2/M3: channel norms, pairwise
        channel cosine, output-to-channel cosine (v2 §9)."""
        self.eval()
        num_nodes = int(x.size(0))
        f_block, _z = self._parent_ctx(x, edge_index, num_nodes)
        _, comps = self._side_messages(f_block, edge_index, num_nodes,
                                       return_components=True)
        m_ab = comps["m_ab"]
        if self.channel_kind == "softmax":
            lams = [self.channel_net(f_block[:, b], self.channel_emb.weight)
                    for b in range(3)]
            mat = torch.zeros(3, 3)
            for b in range(3):
                mat[:, b] = lams[b].mean(dim=0)
            lam_all = torch.cat(lams, dim=0)
            ent = -(lam_all * torch.log(lam_all + 1e-12)).sum(dim=-1).mean()
            return {"channel_kind": "softmax",
                    "lambda_matrix_mean": [[round(float(mat[a, b].item()), 4)
                                            for b in range(3)] for a in range(3)],
                    "lambda_entropy_mean": float(ent.item())}
        out = {"channel_kind": self.channel_kind}
        for b in range(3):
            ch = [m_ab[(a, b)] for a in range(3)]
            norms = [float(c.norm(dim=-1).mean().item()) for c in ch]
            cos = [[float(torch.nn.functional.cosine_similarity(
                ch[i], ch[j], dim=-1).mean().item()) for j in range(3)]
                for i in range(3)]
            mean = sum(ch) / 3.0
            out[f"target_{b}"] = {"channel_norms": norms,
                                  "channel_pairwise_cos": cos,
                                  "channel_mean_cos": [
                                      float(torch.nn.functional.cosine_similarity(
                                          c, mean, dim=-1).mean().item())
                                      for c in ch]}
        return out

    @torch.no_grad()
    def export_operator_stats(self, x, edge_index, subsample: int = 20000) -> dict:
        """FiLM: gamma/beta mean/std, feature-wise variance, pair divergence.
        Basis: expert usage/entropy/effective count, basis-output cosine,
        router JSD across factor pairs (v2 §10)."""
        self.eval()
        num_nodes = int(x.size(0))
        f_block, _z = self._parent_ctx(x, edge_index, num_nodes)
        num_edges = int(edge_index.size(1))
        src, dst = edge_index[0], edge_index[1]
        d = self.factor_dim
        if self.operator_kind in ("target_film", "edge_film"):
            out = {"operator_kind": self.operator_kind}
            idx = torch.randperm(num_edges, device=dst.device)[:subsample]
            pats = {}
            for (a, b) in ALL_PAIRS:
                e_a = self.operator_emb.weight[a]
                e_b = self.operator_emb.weight[b]
                if self.operator_kind == "target_film":
                    u = torch.cat([f_block[:, b],
                                   e_a.unsqueeze(0).expand(num_nodes, -1),
                                   e_b.unsqueeze(0).expand(num_nodes, -1)], dim=-1)
                    gb = self.operator_net(u)  # [N, 2d]
                    dg = gb[:, :d]
                    pats[(a, b)] = dg  # per-node gamma pattern
                else:
                    s_c, d_c = src[idx], dst[idx]
                    fa, fb = f_block[s_c, a], f_block[d_c, b]
                    u = torch.cat([fb, fa, fb * fa, (fb - fa).abs(),
                                   e_a.unsqueeze(0).expand(idx.size(0), -1),
                                   e_b.unsqueeze(0).expand(idx.size(0), -1)], dim=-1)
                    gb = self.operator_net(u)
                    dg = gb[:, :d]
                    pats[(a, b)] = dg  # sampled-edge gamma pattern
                out[f"pair_{a}{b}"] = {
                    "gamma_mean": float(dg.mean().item()),
                    "gamma_std": float(dg.std().item()),
                    "beta_mean": float(gb[:, d:].mean().item()),
                    "beta_std": float(gb[:, d:].std().item()),
                    "featurewise_var_mean": float(dg.var(dim=0).mean().item()),
                }
            # pair divergence: pairwise cosine of per-pair gamma patterns
            keys = list(ALL_PAIRS)
            div = [[round(float(torch.nn.functional.cosine_similarity(
                pats[ki], pats[kj], dim=-1).mean().item()), 4)
                for kj in keys] for ki in keys]
            out["gamma_pattern_pairwise_cos"] = div
            return out
        if self.operator_kind == "basis":
            out = {"operator_kind": "basis", "k": self.operator_net.k}
            idx = torch.randperm(num_edges, device=dst.device)[:subsample]
            q_by_pair = {}
            for (a, b) in ALL_PAIRS:
                s_c, d_c = src[idx], dst[idx]
                e_a = self.operator_emb.weight[a]
                e_b = self.operator_emb.weight[b]
                if self.operator_net.target_router:
                    u = torch.cat([f_block[d_c, b],
                                   e_a.unsqueeze(0).expand(idx.size(0), -1),
                                   e_b.unsqueeze(0).expand(idx.size(0), -1)], dim=-1)
                else:
                    u = torch.cat([f_block[d_c, b], f_block[s_c, a],
                                   e_a.unsqueeze(0).expand(idx.size(0), -1),
                                   e_b.unsqueeze(0).expand(idx.size(0), -1)], dim=-1)
                q = torch.softmax(self.operator_net.router_logits(u), dim=-1)
                q_by_pair[(a, b)] = q
                ent = -(q * torch.log(q + 1e-12)).sum(dim=-1)
                out[f"pair_{a}{b}"] = {
                    "q_mean": [round(float(v), 4) for v in q.mean(dim=0).tolist()],
                    "router_entropy_mean": float(ent.mean().item()),
                    "effective_experts": float(torch.exp(ent.mean()).item()),
                }
            # basis output pairwise cosine (functional diversity of operators)
            v = f_block[src[idx], 0]
            outs = torch.stack([self.operator_net.bases[k](v)
                                for k in range(self.operator_net.k)])
            cos = [[round(float(torch.nn.functional.cosine_similarity(
                outs[i], outs[j], dim=-1).mean().item()), 4)
                for j in range(self.operator_net.k)]
                for i in range(self.operator_net.k)]
            out["basis_output_pairwise_cos"] = cos
            # router JSD across factor pairs
            jsd = [[round(float(_jsd(q_by_pair[ki], q_by_pair[kj])), 4)
                    for kj in ALL_PAIRS] for ki in ALL_PAIRS]
            out["router_jsd"] = jsd
            return out
        return {"operator_kind": self.operator_kind}


def _jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    m = 0.5 * (p + q)
    kl_p = (p * torch.log((p + 1e-12) / (m + 1e-12))).sum(dim=-1)
    kl_q = (q * torch.log((q + 1e-12) / (m + 1e-12))).sum(dim=-1)
    return float(0.5 * (kl_p + kl_q).mean().item())
