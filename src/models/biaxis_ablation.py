"""Paper-facing ablation model (plan §22 Prompt 6).

Formal ablations are organized by PAPER CLAIMS, not by development stages:

    full_reference          == biaxis_final (Full Cell-conditioned Operator)
    no_factor_axis          w/o Semantic Factor Axis: graph side F=1 (factor-
                            blind q), K=4 relations, NullSoftmax, T_k=W0+B_k
    no_relation_axis        w/o Structural Relation Axis: F=3, K=1 strict
                            plain-neighbor context, NullSoftmax Local/Graph,
                            T_f=W0+A_f
    no_adaptive_allocation  w/o Adaptive Allocation: F=3, K=4, Gamma_if0=0,
                            Gamma_ifk = a_ik (topology relation availability,
                            factor-independent), full hierarchical T_fk kept
    shared_operator         w/o Hierarchical Operator: T_fk = W0 (P3 O0)
    no_cell_correction      w/o Cell-specific Correction: T = W0+A_f+B_k
                            (P3 OADD)

Discipline:
    - P0/P1/P2/P3/final frozen files are NEVER modified; all modes live here.
    - the passthrough modes (full_reference/shared_operator/
      no_cell_correction) construct the EXACT biaxis_p3 path with the mapped
      operator mode -> same weights => bitwise-identical outputs (tested).
    - zero-init discipline preserved in every mode (step 0 = W0 path).
    - aux decision (no_factor_axis): the semantic factorizer and its aux
      objective STILL train (the multimodal encoder stays; only the graph
      side loses factor identity) — same semantics as P1's factor-blind
      F0R1 path; documented here and in the ablation yaml.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_p1_components import relation_weighted_mean
from .biaxis_p2_components import (
    build_augmented_scores,
    compute_node_relation_confidence,
    null_augmented_softmax,
)
from .biaxis_p3 import Model as P3Model
from .biaxis_p3_components import FullResidualFactorRelationOperator
from .common import get_activation, make_norm

# paper-mode -> P3 operator_mode (passthrough modes)
_PASSTHROUGH = {
    "full_reference": "full_interaction",
    "shared_operator": "shared",
    "no_cell_correction": "additive",
}

ABLATION_MODES = ("full_reference", "no_factor_axis", "no_relation_axis",
                  "no_adaptive_allocation", "shared_operator", "no_cell_correction")


class Model(P3Model):
    """Bi-Axis ablation model. Inherits all frozen machinery; custom graph
    updates per paper-facing ablation mode."""

    def __init__(self, cfg, data_info):
        mode = str(cfg.model.ablation.mode)
        assert mode in ABLATION_MODES, f"unknown ablation mode {mode!r}"
        self.ablation_mode = mode

        if mode in _PASSTHROUGH:
            cfg.model.p3.operator_mode = _PASSTHROUGH[mode]
            super().__init__(cfg, data_info)
        elif mode == "no_factor_axis":
            super().__init__(cfg, data_info)
            self._setup_no_factor_axis(cfg)
        elif mode == "no_relation_axis":
            super().__init__(cfg, data_info)
            self._setup_no_relation_axis()
        else:  # no_adaptive_allocation
            super().__init__(cfg, data_info)
            self._setup_no_adaptive_allocation()

    # ------------------------------------------------------------------
    # no_factor_axis: graph side F=1 (factor-blind q), T_k = W0 + B_k
    # ------------------------------------------------------------------

    def _setup_no_factor_axis(self, cfg) -> None:
        norm = str(cfg.model.get("norm", "layernorm"))
        activation = str(cfg.model.get("activation", "gelu"))
        # Recreate the P1 factor-blind projectors (deleted by P2's init).
        self.proj_q = nn.Sequential(
            nn.Linear(self.hidden_dim, self.factor_dim),
            make_norm(norm, self.factor_dim),
            get_activation(activation),
            nn.Dropout(float(cfg.model.dropout)),
        )
        self.fusion_q = nn.Sequential(
            nn.Linear(self.factor_dim, self.hidden_dim),
            make_norm(norm, self.hidden_dim),
            get_activation(activation),
            nn.Dropout(float(cfg.model.dropout)),
        )
        # F=1: single Local/No-Transport threshold.
        self.null_score = nn.Parameter(torch.zeros(1))
        # Minimal identifiable operator: T_k = W0 + B_k (factor main effect
        # absorbed into W0, cell correction merged into the relation effect).
        self.operator = FullResidualFactorRelationOperator(
            num_factors=1, num_relations=self.num_relations, dim=self.factor_dim, mode="relation"
        )

    def forward(self, x: torch.Tensor, edge_index=None):
        if self.ablation_mode == "no_factor_axis":
            return self._forward_no_factor_axis(x, edge_index)
        # all other modes keep the frozen factor-aware P1 forward (dynamic
        # dispatch reaches the mode-specific _graph_update override).
        return super().forward(x, edge_index)

    def _forward_no_factor_axis(self, x: torch.Tensor, edge_index=None):
        factors, z_local = self._encode(x)
        if self.training:
            aux_loss, aux_info = self._compute_aux(factors)
        else:
            aux_loss = z_local.new_tensor(0.0)
            aux_info = {}
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        num_nodes = int(x.size(0))
        q = self.proj_q(z_local)  # [N, d_f]
        graph_out = self._graph_update(q.unsqueeze(1), edge_index, num_nodes)
        z = self.fusion_q(graph_out["f_tilde"][:, 0])
        return z, None, None, aux_loss, aux_info

    # ------------------------------------------------------------------
    # no_relation_axis: K=1 strict plain-neighbor context, T_f = W0 + A_f
    # ------------------------------------------------------------------

    def _setup_no_relation_axis(self) -> None:
        self.num_relations = 1  # strict fast path in _decompose_relations
        # M2 relation modules are unreachable at K=1: drop for clean params.
        del self.struct_signature_mlp
        del self.edge_token_mlp
        del self.relation_prototypes
        self.operator = FullResidualFactorRelationOperator(
            num_factors=3, num_relations=1, dim=self.factor_dim, mode="factor"
        )

    # ------------------------------------------------------------------
    # no_adaptive_allocation: Gamma = a_ik (topology availability, factor-
    # independent), full hierarchical operator kept
    # ------------------------------------------------------------------

    def _setup_no_adaptive_allocation(self) -> None:
        del self.transport_scorer
        del self.null_score

    # ------------------------------------------------------------------
    # Graph updates
    # ------------------------------------------------------------------

    def _graph_update(
        self,
        f_block: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> dict[str, torch.Tensor]:
        if self.ablation_mode == "no_factor_axis":
            return self._graph_update_no_factor_axis(f_block, edge_index, num_nodes)
        if self.ablation_mode == "no_relation_axis":
            return self._graph_update_no_relation_axis(f_block, edge_index, num_nodes)
        if self.ablation_mode == "no_adaptive_allocation":
            return self._graph_update_fixed_allocation(f_block, edge_index, num_nodes)
        # passthrough modes: exactly the P3 update (W0 / W0+A+B / full).
        return super()._graph_update(f_block, edge_index, num_nodes)

    def _graph_update_no_factor_axis(
        self,
        f_block: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> dict[str, torch.Tensor]:
        """F=1 factor-blind graph update: NullSoftmax over {Local, R1..RK},
        operator W0+B_k."""
        num_factors = 1
        factor_dim = int(f_block.size(2))
        device = f_block.device
        r, availability, deg = self._decompose_relations(edge_index, num_nodes)
        f_cat = f_block.reshape(num_nodes, num_factors * factor_dim)
        g_cat, _mass = relation_weighted_mean(
            edge_index, r, f_cat, num_nodes, edge_chunk_size=self.edge_chunk_size
        )  # [N, K, d]
        g_perm = g_cat.reshape(num_nodes, self.num_relations, num_factors, factor_dim)
        g_perm = g_perm.permute(0, 2, 1, 3)  # [N, 1, K, d]

        s_rel = self.transport_scorer(f_block, g_perm)  # [N, 1, K]
        s_aug = build_augmented_scores(s_rel, self.null_score)  # [N, 1, K+1]
        gamma = null_augmented_softmax(s_aug, self.p2_epsilon)
        isolated = deg <= 0
        if bool(isolated.any()):
            local_plan = torch.zeros_like(gamma)
            local_plan[..., 0] = 1.0
            gamma = torch.where(isolated[:, None, None], local_plan, gamma)
        graph_mass = 1.0 - gamma[..., 0]
        alpha_diag = gamma[..., 1:] / (graph_mass.unsqueeze(-1) + self.eps)
        m_f = self.operator(g_perm, gamma[..., 1:], self.graph_w0)  # [N, 1, d]
        f_tilde = self.graph_norm((f_block + m_f).reshape(num_nodes * num_factors, factor_dim))
        f_tilde = f_tilde.reshape(num_nodes, num_factors, factor_dim)
        return {
            "f_tilde": f_tilde,
            "beta": graph_mass,
            "alpha": alpha_diag,
            "r": r,
            "availability": availability,
            "gamma": gamma,
            "null_mass": gamma[..., 0],
            "relation_confidence": compute_node_relation_confidence(r, edge_index, num_nodes),
            "theta": torch.zeros(num_nodes, dtype=f_block.dtype, device=device),
            "g_perm": g_perm,
        }

    def _graph_update_no_relation_axis(
        self,
        f_block: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> dict[str, torch.Tensor]:
        """K=1 strict: r=ones -> plain neighbor mean; NullSoftmax over
        {Local, Graph}; operator W0+A_f."""
        num_factors = int(f_block.size(1))
        factor_dim = int(f_block.size(2))
        device = f_block.device
        r, availability, deg = self._decompose_relations(edge_index, num_nodes)
        f_cat = f_block.reshape(num_nodes, num_factors * factor_dim)
        g_cat, _mass = relation_weighted_mean(
            edge_index, r, f_cat, num_nodes, edge_chunk_size=self.edge_chunk_size
        )  # [N, 1, F*d]
        g_perm = g_cat.reshape(num_nodes, 1, num_factors, factor_dim)
        g_perm = g_perm.permute(0, 2, 1, 3)  # [N, F, 1, d]

        s_rel = self.transport_scorer(f_block, g_perm)  # [N, F, 1]
        s_aug = build_augmented_scores(s_rel, self.null_score)  # [N, F, 2]
        gamma = null_augmented_softmax(s_aug, self.p2_epsilon)
        isolated = deg <= 0
        if bool(isolated.any()):
            local_plan = torch.zeros_like(gamma)
            local_plan[..., 0] = 1.0
            gamma = torch.where(isolated[:, None, None], local_plan, gamma)
        graph_mass = 1.0 - gamma[..., 0]
        alpha_diag = gamma[..., 1:] / (graph_mass.unsqueeze(-1) + self.eps)
        m_f = self.operator(g_perm, gamma[..., 1:], self.graph_w0)  # [N, F, d]
        f_tilde = self.graph_norm((f_block + m_f).reshape(num_nodes * num_factors, factor_dim))
        f_tilde = f_tilde.reshape(num_nodes, num_factors, factor_dim)
        return {
            "f_tilde": f_tilde,
            "beta": graph_mass,
            "alpha": alpha_diag,
            "r": r,
            "availability": availability,
            "gamma": gamma,
            "null_mass": gamma[..., 0],
            "relation_confidence": compute_node_relation_confidence(r, edge_index, num_nodes),
            "theta": torch.zeros(num_nodes, dtype=f_block.dtype, device=device),
            "g_perm": g_perm,
        }

    def _graph_update_fixed_allocation(
        self,
        f_block: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> dict[str, torch.Tensor]:
        """Gamma_if0 = 0, Gamma_ifk = a_ik (topology availability, identical
        across factors); the full hierarchical operator still transforms the
        relation contexts. No scorer influence on messages."""
        num_factors = int(f_block.size(1))
        factor_dim = int(f_block.size(2))
        r, availability, deg = self._decompose_relations(edge_index, num_nodes)
        f_cat = f_block.reshape(num_nodes, num_factors * factor_dim)
        g_cat, _mass = relation_weighted_mean(
            edge_index, r, f_cat, num_nodes, edge_chunk_size=self.edge_chunk_size
        )  # [N, K, F*d]
        g_perm = g_cat.reshape(num_nodes, self.num_relations, num_factors, factor_dim)
        g_perm = g_perm.permute(0, 2, 1, 3)  # [N, F, K, d]

        gamma_graph = availability.unsqueeze(1).expand(num_nodes, num_factors, self.num_relations)
        gamma = torch.cat(
            [torch.zeros(num_nodes, num_factors, 1, dtype=f_block.dtype, device=f_block.device),
             gamma_graph],
            dim=-1,
        )  # [N, F, K+1], Gamma_if0 = 0
        graph_mass = 1.0 - gamma[..., 0]
        alpha_diag = gamma_graph / (graph_mass.unsqueeze(-1) + self.eps)
        m_f = self.operator(g_perm, gamma_graph, self.graph_w0)  # [N, F, d]
        f_tilde = self.graph_norm((f_block + m_f).reshape(num_nodes * num_factors, factor_dim))
        f_tilde = f_tilde.reshape(num_nodes, num_factors, factor_dim)
        return {
            "f_tilde": f_tilde,
            "beta": graph_mass,
            "alpha": alpha_diag,
            "r": r,
            "availability": availability,
            "gamma": gamma,
            "null_mass": gamma[..., 0],
            "relation_confidence": compute_node_relation_confidence(r, edge_index, num_nodes),
            "theta": torch.zeros(num_nodes, dtype=f_block.dtype, device=f_block.device),
            "g_perm": g_perm,
        }
