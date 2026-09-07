"""P3 Bi-Axis model: Factor-Relation-specific graph transformation (plan §5).

Inherits P2 (M1 semantic factorization + M2 relation decomposition +
Null-Augmented plan Gamma) and changes ONLY how graph messages are
interpreted:

    P2:  g_mix = sum_k Gamma*g ;  m = W0(g_mix)              (shared operator)
    P3:  m_i^f = sum_k Gamma_ifk * T_fk(g_ik^f)              (cell operator)

with the full residual decomposition (P3-A):

    T_fk = W0 + A_f + B_k + C_fk

Local column Gamma_if0 still only feeds the residual factor (f_block),
never a graph operator (plan §28).

P3 discipline:
    - p2.mode MUST be null_softmax, p2.deterministic MUST be false
      (hard asserts: P3 runs the fast default training path, plan §2.4)
    - all residuals zero-initialized: step 0 T_fk = W0 for every mode
    - no new aux losses; operator regularizers exist but default 0 (plan §7)
    - P0/P1/P2 files are never modified
"""

from __future__ import annotations

import warnings

import torch
import torch.utils.checkpoint

from .biaxis_p1_components import relation_weighted_mean

# torch 2.4's non-reentrant checkpoint internally calls the deprecated
# torch.cpu.amp.autocast(...) constructor (checkpoint.py:~1399) on every
# invocation; the kwargs are empty unless CPU autocast is enabled (we never
# enable it), so the FutureWarning is purely cosmetic. Silencing it here,
# scoped to torch.utils.checkpoint, keeps training logs clean.
warnings.filterwarnings(
    "ignore",
    message=".*torch.cpu.amp.autocast.*deprecated.*",
    category=FutureWarning,
    module=r"torch\.utils\.checkpoint",
)
from .biaxis_p2 import Model as P2Model
from .biaxis_p2_components import (
    build_augmented_scores,
    build_reference_capacity,
    compute_node_relation_confidence,
    null_augmented_softmax,
    semi_relaxed_transport,
)
from .biaxis_p3_components import (
    BasisCellOperator,
    DirectCellOperator,
    FullResidualFactorRelationOperator,
    LowRankFactorRelationOperator,
)


class Model(P2Model):
    """Bi-Axis P3. Overrides `_graph_update` so the transport plan selects
    evidence (how much / which relation) and a per-cell operator decides how
    that evidence is transformed."""

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)

        # P3 main experiments are frozen on the NullSoftmax coupler and the
        # fast (atomic-aggregation) training path (plan §2.3/§2.4).
        # composition_uot is admitted ONLY for the post-freeze compatibility
        # check (plan §24), never for the main structure study.
        assert self.p2_mode in ("null_softmax", "composition_uot"), (
            f"biaxis_p3 requires model.p2.mode=null_softmax (composition_uot "
            f"only for the §24 compatibility check), got {self.p2_mode!r}"
        )
        assert not self.p2_deterministic, (
            "biaxis_p3 requires model.p2.deterministic=false (P3 never uses "
            "the deterministic verification mode)"
        )

        p3 = cfg.model.p3
        self.p3_operator_mode = str(p3.operator_mode)
        self.p3_lowrank_rank = int(p3.get("lowrank_rank", 16))
        self.p3_operator_reg_weight = float(p3.get("operator_reg_weight", 0.0))
        self.p3_interaction_reg_weight = float(p3.get("interaction_reg_weight", 0.0))
        # Memory fix (2026-09-04): activation checkpointing for the context +
        # scorer segment. Numerics-preserving: the recomputed activations are
        # bitwise identical to the forward pass (no dropout / randomness in
        # the segment), so gradients and results are unchanged.
        self.p3_memory_checkpoint = bool(p3.get("memory_checkpoint", False))

        if self.p3_operator_mode in FullResidualFactorRelationOperator.MODES:
            self.operator = FullResidualFactorRelationOperator(
                num_factors=3,  # C / Pt / Pv (factor_aware asserted by P2)
                num_relations=self.num_relations,
                dim=self.factor_dim,
                mode=self.p3_operator_mode,
            )
        elif self.p3_operator_mode in LowRankFactorRelationOperator.MODES:
            self.operator = LowRankFactorRelationOperator(
                num_factors=3,
                num_relations=self.num_relations,
                dim=self.factor_dim,
                rank=self.p3_lowrank_rank,
                mode=self.p3_operator_mode,
            )
        elif self.p3_operator_mode in BasisCellOperator.MODES:
            self.operator = BasisCellOperator(
                num_factors=3,
                num_relations=self.num_relations,
                dim=self.factor_dim,
                num_bases=int(p3.get("basis_num_bases", 8)),
                mode=self.p3_operator_mode,
            )
        elif self.p3_operator_mode in DirectCellOperator.MODES:
            self.operator = DirectCellOperator(
                num_factors=3,
                num_relations=self.num_relations,
                dim=self.factor_dim,
                mode=self.p3_operator_mode,
            )
        else:
            raise AssertionError(f"unknown p3.operator_mode {self.p3_operator_mode!r}")

    # ------------------------------------------------------------------
    # P3 graph update: per-cell operator BEFORE the relation sum
    # ------------------------------------------------------------------

    def _graph_update(
        self,
        f_block: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> dict[str, torch.Tensor]:
        """P2 plan solver unchanged; only the message interpretation differs
        (plan §28): m_i^f = sum_k Gamma_ifk T_fk(g_ik^f)."""
        num_factors = int(f_block.size(1))
        factor_dim = int(f_block.size(2))
        device = f_block.device

        r, availability, deg = self._decompose_relations(edge_index, num_nodes)

        # Relation-specific factor contexts (P1 aggregation, P2 plan §5) +
        # relation compatibility scores (plan §6). With memory_checkpoint the
        # whole context+scorer segment runs under activation checkpointing:
        # its [E, d]-level intermediates are recomputed in backward instead
        # of retained (bitwise-identical recomputation: no dropout or
        # randomness inside the segment).
        f_cat = f_block.reshape(num_nodes, num_factors * factor_dim)
        if self.p3_memory_checkpoint and torch.is_grad_enabled() and self.training:

            def _context_scores(r_seg: torch.Tensor, f_cat_seg: torch.Tensor):
                g_cat, _m = relation_weighted_mean(
                    edge_index, r_seg, f_cat_seg, num_nodes,
                    edge_chunk_size=self.edge_chunk_size,
                )  # [N, K, F*d]
                g_perm = g_cat.reshape(num_nodes, self.num_relations, num_factors, factor_dim)
                g_perm = g_perm.permute(0, 2, 1, 3)  # [N, F, K, d]
                s_rel = self.transport_scorer(f_block, g_perm)  # [N, F, K]
                return g_perm, s_rel

            g_perm, s_rel = torch.utils.checkpoint.checkpoint(
                _context_scores, r, f_cat, use_reentrant=False
            )
        else:
            g_cat, _mass = relation_weighted_mean(
                edge_index, r, f_cat, num_nodes, edge_chunk_size=self.edge_chunk_size
            )  # [N, K, F*d]
            g_perm = g_cat.reshape(num_nodes, self.num_relations, num_factors, factor_dim)
            g_perm = g_perm.permute(0, 2, 1, 3)  # [N, F, K, d]
            s_rel = self.transport_scorer(f_block, g_perm)  # [N, F, K]
        s_aug = build_augmented_scores(s_rel, self.null_score)  # [N, F, K+1]
        nu = build_reference_capacity(
            availability,
            num_factors,
            null_prior=self.p2_null_prior,
            degree=deg,
            detach=self.p2_detach_capacity_prior,
        )
        q = compute_node_relation_confidence(
            r, edge_index, num_nodes, detach=self.p2_detach_relation_confidence
        )
        if self.p2_mode == "composition_uot":
            # §24 compatibility check only (P2 review §9/§10): total graph
            # mass from the unconstrained NullSoftmax plan (detached), the
            # UOT constraint only redistributes it across R1..RK.
            gamma_ns = null_augmented_softmax(s_aug, self.p2_epsilon)
            m_ns = gamma_ns[..., 1:].sum(dim=(1, 2)).detach()  # [N]
            nu_rel = torch.cat(
                [
                    torch.zeros(num_nodes, 1, dtype=f_block.dtype, device=device),
                    (m_ns.unsqueeze(-1) * availability),
                ],
                dim=-1,
            )
            theta_col = torch.full(
                (num_nodes, self.num_relations + 1), self.p2_tau_base / (self.p2_tau_base + self.p2_epsilon),
                dtype=f_block.dtype, device=device,
            )
            theta_col[:, 0] = 0.0
            gamma = semi_relaxed_transport(
                s_aug, nu_rel, self.p2_epsilon, self.p2_tau_base, self.p2_sinkhorn_iters, theta_col
            )
            theta = torch.full(
                (num_nodes,), self.p2_tau_base / (self.p2_tau_base + self.p2_epsilon),
                dtype=f_block.dtype, device=device,
            )
        else:
            gamma = null_augmented_softmax(s_aug, self.p2_epsilon)
            theta = torch.zeros(num_nodes, dtype=f_block.dtype, device=device)

        # --- isolated node fast path (plan §21): all Local, no transport ---
        isolated = deg <= 0
        if bool(isolated.any()):
            local_plan = torch.zeros_like(gamma)
            local_plan[..., 0] = 1.0
            gamma = torch.where(isolated[:, None, None], local_plan, gamma)

        # --- message passing (plan §28): operator applied per (f, k) cell
        # before the relation sum; Local column never passes an operator ----
        graph_mass = 1.0 - gamma[..., 0]  # [N, F]
        alpha_diag = gamma[..., 1:] / (graph_mass.unsqueeze(-1) + self.eps)  # [N, F, K]
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
            "relation_confidence": q,
            "theta": theta,
            "g_perm": g_perm,
        }

    # ------------------------------------------------------------------
    # Operator regularizers (plan §7: weights 0 by default)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None):
        z, _, _, aux_loss, aux_info = super().forward(x, edge_index)
        if self.training:
            if self.p3_operator_reg_weight > 0:
                aux_loss = aux_loss + self.p3_operator_reg_weight * self.operator.reg_operator()
            if self.p3_interaction_reg_weight > 0:
                aux_loss = aux_loss + self.p3_interaction_reg_weight * self.operator.reg_interaction()
        return z, None, None, aux_loss, aux_info

    # ------------------------------------------------------------------
    # P3 mechanism diagnostics (plan §13 / §35 Prompt 4)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_p3_diagnostics(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict:
        """P2 plan diagnostics + operator diagnostics. Never uses labels;
        does not modify model state. The operator pass re-runs
        `_graph_update` (the raw topology signature is cached, so the
        extra cost is only the aggregations — post-hoc best-checkpoint
        analysis only, not on the training path)."""
        diag = super().compute_p2_diagnostics(x, edge_index)
        self.eval()
        factors, _z_local = self._encode(x)
        num_nodes = int(x.size(0))
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        graph_out = self._graph_update(f_block, edge_index, num_nodes)
        diag["operator"] = self.operator.compute_diagnostics(
            graph_out["g_perm"], graph_out["gamma"][..., 1:], self.graph_w0
        )
        return diag
