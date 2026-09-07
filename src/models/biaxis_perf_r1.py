"""R1 performance-branch model (plan §37 Prompt 3; audit §1/§1bis/§2).

Inherits the frozen P3 stack (OFR operator + NullSoftmax + all inherited
hard asserts) and changes ONLY the relation-conditioned context aggregation:

    mode = baseline              == biaxis_final: same weights => bitwise
                                   identical. _graph_update delegates to
                                   super() and NO new module is constructed,
                                   so state_dict keys stay identical.
    mode = semantic_reliability   R1-A1 (HARD NO-GO, kept for the record):
                                   reliable_relation_weighted_mean with
                                   eta_ji^f (factor-conditioned edge
                                   reliability). Never combined with A2.
    mode = semantic_relation_calibration  R1-A2 (user-authorized amendment):
                                   relation_calibrated_weighted_mean with
                                   r^f = Softmax(log r^str + tanh(q - mean q)),
                                   the factor-conditioned semantic relation
                                   posterior. r^str / availability / capacity
                                   / scorer / Gamma / operator / Local / P0
                                   aux losses are all unchanged.

Frozen discipline:
    - p2.mode MUST be null_softmax, p2.deterministic MUST be false
      (inherited P2/P3 hard asserts, plus an explicit R1 assert).
    - p3.operator_mode MUST be full_interaction (the frozen OFR parent).
    - P0/P1/P2/P3/final files are never modified.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .biaxis_p1_components import neighbor_mean, relation_weighted_mean
from .common import get_activation, make_norm
from .biaxis_p2_components import (
    build_augmented_scores,
    build_reference_capacity,
    compute_node_relation_confidence,
    null_augmented_softmax,
)
from .biaxis_p3 import Model as P3Model
from .biaxis_perf_r1_components import (
    DynamicLocalScoreResidual,
    FactorConditionedEdgeReliability,
    FactorConditionedRelationCalibration,
    SupportRelationScoreResidual,
    calibration_edge_statistics,
    relation_calibrated_weighted_mean,
    reliability_regularization,
    reliable_relation_weighted_mean,
    reliability_edge_statistics,
)

R1_MODES = (
    "baseline",
    "semantic_reliability",
    "semantic_relation_calibration",
    "detached_2hop",
)


class Model(P3Model):
    """Bi-Axis R1. Only the context aggregation differs from the frozen
    parent; everything downstream (scores, Gamma, operator, Local, fusion,
    aux losses) is untouched."""

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)

        assert self.p2_mode == "null_softmax", (
            "biaxis_perf_r1 requires model.p2.mode=null_softmax (R1 does not "
            f"reopen the coupler question), got {self.p2_mode!r}"
        )
        assert self.p3_operator_mode == "full_interaction", (
            "biaxis_perf_r1 requires model.p3.operator_mode=full_interaction "
            f"(the frozen OFR parent), got {self.p3_operator_mode!r}"
        )

        r1 = cfg.model.r1
        self.r1_mode = str(r1.mode)
        assert self.r1_mode in R1_MODES, f"unknown r1.mode {self.r1_mode!r}"
        # R1-B router (plan §19, user-decoupled variants): base == parent
        # Gamma; local_only (BL) / relation_only (BR) / evidence (BLR) add
        # the zero-init residual scorers. Experimentally used ONLY on top of
        # the A0 baseline — A1/A2 are Hard NO-GO and are never combined with
        # later modules (user ruling).
        self.r1_router_mode = str(r1.get("router_mode", "base"))
        assert self.r1_router_mode in ("base", "local_only", "relation_only", "evidence"), (
            f"unknown r1.router_mode {self.r1_router_mode!r}"
        )
        router_hidden = int(r1.get("router_hidden_dim", 64))
        activation = str(cfg.model.get("activation", "gelu"))
        # Construct ONLY the module(s) the variant uses (clean param counts
        # for the decoupled ablation, user §5/§9).
        self._use_local_residual = self.r1_router_mode in ("local_only", "evidence")
        self._use_relation_residual = self.r1_router_mode in ("relation_only", "evidence")
        if self._use_local_residual:
            self.local_score_residual = DynamicLocalScoreResidual(
                factor_dim=self.factor_dim,
                hidden_dim=router_hidden,
                activation=activation,
            )
        if self._use_relation_residual:
            self.relation_score_residual = SupportRelationScoreResidual(
                hidden_dim=router_hidden,
                activation=activation,
            )
        # Regularizer defaults (review option B): plain A1 unless enabled.
        self.r1_reg_type = None
        self.r1_reg_weight = 0.0

        # A1 reliability / A2 calibration modules ONLY in their own modes:
        # baseline mode must construct nothing new so state_dict keys ==
        # biaxis_final. A1 and A2 are mutually exclusive (user ruling: A1 is
        # Hard NO-GO and is never combined with later modules).
        if self.r1_mode == "semantic_reliability":
            self.reliability = FactorConditionedEdgeReliability(
                num_factors=3,  # C / Pt / Pv (factor_aware asserted by P2)
                factor_dim=self.factor_dim,
                proj_dim=int(r1.get("rel_proj_dim", 32)),
                hidden_dim=int(r1.get("rel_hidden_dim", 64)),
                activation=str(cfg.model.get("activation", "gelu")),
            )
            self.r1_rel_chunk_size = r1.get("rel_chunk_size")
            # Review option B control: regularizers are OFF by default (A1
            # unchanged); reg_type null|mean1|band, reg_weight 0 disables.
            self.r1_reg_type = str(r1.get("reg_type", "null"))
            if self.r1_reg_type == "null":
                self.r1_reg_type = None
            self.r1_reg_weight = float(r1.get("reg_weight", 0.0))
        elif self.r1_mode == "semantic_relation_calibration":
            self.calibration = FactorConditionedRelationCalibration(
                num_factors=3,
                factor_dim=self.factor_dim,
                proj_dim=int(r1.get("rel_proj_dim", 32)),
                hidden_dim=int(r1.get("rel_hidden_dim", 64)),
                num_relations=self.num_relations,
                activation=str(cfg.model.get("activation", "gelu")),
            )
            self.r1_rel_chunk_size = r1.get("rel_chunk_size")
        elif self.r1_mode == "detached_2hop":
            # R1-C1SG (user ruling): detached adaptive 2-hop trajectory.
            # Shared-across-factors W_traj + LN; depth gate MLP last layer
            # zero-init => lam == 0 => F_out == F1 exactly at step 0.
            norm = str(cfg.model.get("norm", "layernorm"))
            self.traj_w = nn.Linear(self.factor_dim, self.factor_dim, bias=False)
            self.traj_norm = make_norm(norm, self.factor_dim)
            self.depth_mlp = nn.Sequential(
                nn.Linear(3 * self.factor_dim, int(r1.get("hop_hidden_dim", 64))),
                get_activation(str(cfg.model.get("activation", "gelu"))),
                nn.Linear(int(r1.get("hop_hidden_dim", 64)), 1),
            )
            nn.init.zeros_(self.depth_mlp[-1].weight)
            nn.init.zeros_(self.depth_mlp[-1].bias)

    # ------------------------------------------------------------------
    # R1-A graph update: reliable relation-weighted context (plan §6)
    # ------------------------------------------------------------------

    def _graph_update(
        self,
        f_block: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> dict[str, torch.Tensor]:
        """Baseline+base delegates to the frozen P3 path (bitwise identity).
        Any other (mode, router) combination reproduces the P3 body with
        changes ONLY at the sanctioned insertion points: the context
        aggregation (audit Q3: the g_cat/g_perm block) and — for the
        evidence router (plan §19) — the score assembly. r_str /
        availability / capacity / Gamma solver / operator are all untouched."""
        if self.r1_mode == "baseline" and self.r1_router_mode == "base":
            return super()._graph_update(f_block, edge_index, num_nodes)

        num_factors = int(f_block.size(1))
        factor_dim = int(f_block.size(2))
        device = f_block.device

        r, availability, deg = self._decompose_relations(edge_index, num_nodes)

        # --- context aggregation per mode (audit Q4/Q5) --------------------
        effective_mass = None
        if self.r1_mode in ("baseline", "detached_2hop"):
            # baseline / C1SG: the frozen A0 context aggregation (C1SG keeps
            # K / relation / NullSoftmax / OFR exactly as A0).
            f_cat = f_block.reshape(num_nodes, num_factors * factor_dim)
            g_cat, _mass = relation_weighted_mean(
                edge_index, r, f_cat, num_nodes, edge_chunk_size=self.edge_chunk_size
            )
            g_perm = g_cat.reshape(num_nodes, self.num_relations, num_factors, factor_dim)
            g_perm = g_perm.permute(0, 2, 1, 3)  # [N, F, K, d]
        elif self.r1_mode == "semantic_reliability":
            g_perm, effective_mass = reliable_relation_weighted_mean(
                edge_index, r, f_block, self.reliability, num_nodes,
                edge_chunk_size=self.r1_rel_chunk_size,
            )  # [N, F, K, d], [N, F, K]
        else:  # semantic_relation_calibration
            g_perm, effective_mass = relation_calibrated_weighted_mean(
                edge_index, r, f_block, self.calibration, num_nodes,
                edge_chunk_size=self.r1_rel_chunk_size,
            )  # [N, F, K, d], [N, F, K]

        # --- scores / capacity / confidence / plan (P2 §6-§17) -------------
        s_rel = self.transport_scorer(f_block, g_perm)  # [N, F, K]
        if self._use_relation_residual:
            # R1-BR (plan §19): support-aware relation residual — a FEATURE
            # only, never a hard capacity prior (plan §20).
            mass = availability * deg.unsqueeze(-1)  # structural m_ik [N, K]
            rel_res = self.relation_score_residual(
                torch.log1p(mass), availability
            )  # [N, K, 1]
            s_rel = s_rel + rel_res.permute(0, 2, 1).expand(num_nodes, num_factors, self.num_relations)
        s_aug = build_augmented_scores(s_rel, self.null_score)  # [N, F, K+1]
        if self._use_local_residual:
            # R1-BL (plan §19): dynamic Local residual. Only column 0 moves,
            # so the conditional relation plan alpha stays exactly
            # Softmax_k(s_rel/eps) (user §7).
            f_cat_ev = f_block.reshape(num_nodes, num_factors * factor_dim)
            g_bar = neighbor_mean(
                edge_index, f_cat_ev, num_nodes, edge_chunk_size=self.edge_chunk_size
            ).reshape(num_nodes, num_factors, factor_dim)
            local_res = self.local_score_residual(f_block, g_bar)  # [N, F, 1]
            s_aug[..., 0] = s_aug[..., 0] + local_res.squeeze(-1)
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
        gamma = null_augmented_softmax(s_aug, self.p2_epsilon)
        theta = torch.zeros(num_nodes, dtype=f_block.dtype, device=device)

        # --- isolated node fast path (plan §21): all Local, no transport ---
        isolated = deg <= 0
        if bool(isolated.any()):
            local_plan = torch.zeros_like(gamma)
            local_plan[..., 0] = 1.0
            gamma = torch.where(isolated[:, None, None], local_plan, gamma)

        # --- message passing (P3 §28): per-cell operator before the sum ----
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
            "effective_mass": effective_mass,
        }

    # ------------------------------------------------------------------
    # Forward: reliability regularizer (option B) / detached 2-hop (C1SG)
    # ------------------------------------------------------------------

    def _hop_readout(
        self,
        f_block: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """R1-C1SG trajectory readout (user ruling):
            D_if = LN[W_traj(F(2)-F(1))]    W_traj shared across factors
            lam_if = tanh(MLP_d([F(0)|F(1)|sg(F(2))]))
            F_out = F(1) + lam * D
        Zero-init depth gate => lam == 0 => F_out == F1 EXACTLY.
        F2 must already be detached (second hop ran under no_grad)."""
        num_nodes = int(f1.size(0))
        diff = (f2 - f1).reshape(num_nodes * 3, self.factor_dim)
        d = self.traj_norm(self.traj_w(diff)).reshape(num_nodes, 3, self.factor_dim)
        feat = torch.cat([f_block, f1, f2.detach()], dim=-1)  # [N, F, 3d]
        lam = torch.tanh(self.depth_mlp(feat).squeeze(-1))  # [N, F]
        f_out = f1 + lam.unsqueeze(-1) * d
        return f_out, lam, d

    def forward(self, x: torch.Tensor, edge_index=None):
        if self.r1_mode != "detached_2hop":
            z, _, _, aux_loss, aux_info = super().forward(x, edge_index)
            if self.training and self.r1_reg_weight > 0:
                if edge_index is None:
                    edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
                factors, _ = self._encode(x)
                f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
                aux_loss = aux_loss + self.r1_reg_weight * reliability_regularization(
                    edge_index, f_block, self.reliability, int(x.size(0)),
                    reg_type=self.r1_reg_type, edge_chunk_size=self.r1_rel_chunk_size,
                )
            return z, None, None, aux_loss, aux_info

        # --- R1-C1SG: detached adaptive 2-hop trajectory -------------------
        # Hop 1 = the A0 path (normal gradients, unchanged). Hop 2 reuses the
        # same graph block under no_grad: F2 carries NO gradient to the P0 /
        # M2 / Gamma / operator / GraphBlock parameters by construction.
        factors, z_local = self._encode(x)
        if self.training:
            aux_loss, aux_info = self._compute_aux(factors)
        else:
            aux_loss = z_local.new_tensor(0.0)
            aux_info = {}
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        num_nodes = int(x.size(0))
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)  # F(0)
        graph_out1 = self._graph_update(f_block, edge_index, num_nodes)  # F(1)
        f1 = graph_out1["f_tilde"]
        with torch.no_grad():
            graph_out2 = self._graph_update(f1.detach(), edge_index, num_nodes)  # F(2)
        f2 = graph_out2["f_tilde"]
        f_out, lam, d = self._hop_readout(f_block, f1, f2)
        if self.training and (
            self.relation_balance_weight or self.alpha_entropy_weight or self.budget_reg_weight
        ):
            aux_loss = aux_loss + self._graph_regularization(
                graph_out1["r"], graph_out1["alpha"], graph_out1["beta"]
            )
        z = self.fusion(torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1))
        return z, None, None, aux_loss, aux_info

    # ------------------------------------------------------------------
    # R1 mechanism diagnostics (plan §10/§11 + review §8/§9)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_r1_diagnostics(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict:
        """P3 plan/operator diagnostics + R1-A reliability diagnostics.
        Never uses labels; does not modify model state. Best-checkpoint
        analysis only (the extra full-graph pass is acceptable — P3 pattern).
        Baseline mode reports None for the reliability sections."""
        diag = super().compute_p3_diagnostics(x, edge_index)

        self.eval()
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=x.device)
        factors, _z_local = self._encode(x)
        num_nodes = int(x.size(0))
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        graph_out = self._graph_update(f_block, edge_index, num_nodes)
        r = graph_out["r"]
        deg = torch.bincount(edge_index[1], minlength=num_nodes).to(torch.float32)
        # Structural mass stays the UNCHANGED r-only definition (audit Q6);
        # it only masks D_ctx / context-change aggregations for comparability.
        mass = graph_out["availability"] * deg.unsqueeze(-1)  # [N, K]
        factor_names = ["C", "Pt", "Pv"]

        # D_ctx per factor (R0 definition: structural-mass mask >= 0.5,
        # mean pairwise 1-cos over valid relation cells, nodes with >= 2).
        # Computed for BOTH modes (baseline == R0-comparable A0 value).
        d_ctx: dict[str, float | None] = {}
        for f, fname in enumerate(factor_names):
            g = graph_out["g_perm"][:, f]  # [N, K, d]
            valid = mass >= 0.5  # [N, K]
            pairwise = 1.0 - F.cosine_similarity(g.unsqueeze(2), g.unsqueeze(1), dim=-1)  # [N,K,K]
            n_valid = valid.sum(dim=-1)
            usable = n_valid >= 2
            if bool(usable.any()):
                sel = pairwise[usable]
                vsel = valid[usable]
                tri = torch.triu(sel, diagonal=1)
                mask = torch.triu(vsel.unsqueeze(1) & vsel.unsqueeze(2), diagonal=1)
                d_ctx[fname] = float(
                    ((tri * mask.float()).sum(dim=(1, 2)) / (mask.float().sum(dim=(1, 2)) + 1e-8)).mean().item()
                )
            else:
                d_ctx[fname] = None
        diag["d_ctx"] = d_ctx
        diag["hop"] = None

        if self.r1_mode == "detached_2hop":
            # R1-C1SG mechanism stats (user ruling): per-factor lam
            # distribution, |lam|<0.05 fraction, correction/base norm ratio,
            # cos(F1, F2). Eval path; two extra full-graph passes.
            graph_out1 = self._graph_update(f_block, edge_index, num_nodes)
            f1 = graph_out1["f_tilde"]
            with torch.no_grad():
                graph_out2 = self._graph_update(f1.detach(), edge_index, num_nodes)
            f2 = graph_out2["f_tilde"]
            _f_out, lam, d = self._hop_readout(f_block, f1, f2)
            qs = torch.tensor([0.1, 0.5, 0.9], dtype=lam.dtype, device=lam.device)
            hop: dict[str, dict[str, float]] = {}
            for fi, fname in enumerate(factor_names):
                l = lam[:, fi]
                q = torch.quantile(l, qs)
                corr_ratio = float(
                    (d[:, fi].norm(dim=-1) / (f1[:, fi].norm(dim=-1) + 1e-8)).mean().item()
                )
                cos = float(F.cosine_similarity(
                    f1[:, fi].reshape(-1, self.factor_dim),
                    f2[:, fi].reshape(-1, self.factor_dim), dim=-1,
                ).mean().item())
                hop[fname] = {
                    "lam_mean": float(l.mean().item()),
                    "lam_std": float(l.std(unbiased=False).item()),
                    "lam_abs_mean": float(l.abs().mean().item()),
                    "lam_p10": float(q[0].item()),
                    "lam_p50": float(q[1].item()),
                    "lam_p90": float(q[2].item()),
                    "frac_abs_lt_0.05": float((l.abs() < 0.05).float().mean().item()),
                    "correction_base_ratio": corr_ratio,
                    "cos_F1F2": cos,
                }
            diag["hop"] = hop
            diag["reliability"] = None
            diag["calibration"] = None
            diag["context_change"] = None
            diag["effective_mass"] = None
            return diag

        if self.r1_mode == "baseline":
            diag["reliability"] = None
            diag["calibration"] = None
            diag["context_change"] = None
            diag["effective_mass"] = None
            return diag

        g_a1 = graph_out["g_perm"]  # [N, F, K, d]
        eff_mass = graph_out["effective_mass"]  # [N, F, K]

        # Mode-specific edge-level statistics (one chunked pass each).
        if self.r1_mode == "semantic_reliability":
            # eta-side statistics (review §8).
            diag["reliability"] = reliability_edge_statistics(
                edge_index, r, f_block, self.reliability, num_nodes,
                edge_chunk_size=self.r1_rel_chunk_size,
            )
            diag["calibration"] = None
        else:  # semantic_relation_calibration (A2)
            diag["calibration"] = calibration_edge_statistics(
                edge_index, r, f_block, self.calibration, num_nodes,
                edge_chunk_size=self.r1_rel_chunk_size,
            )
            diag["reliability"] = None

        # Baseline r^str contexts for the context-change diagnostic
        # (review §9): g^A0 via the frozen relation_weighted_mean.
        f_cat = f_block.reshape(num_nodes, 3 * self.factor_dim)
        g0, _m0 = relation_weighted_mean(
            edge_index, r, f_cat, num_nodes, edge_chunk_size=self.edge_chunk_size
        )
        g_a0 = g0.reshape(num_nodes, self.num_relations, 3, self.factor_dim).permute(0, 2, 1, 3)

        # context_change: delta_g_ifk = 1 - cos(g^mode, g^A0) (review §9).
        # For A2 this measures how far the calibrated routing moved the
        # contexts away from the r^str baseline.
        change: dict[str, dict[str, float | None]] = {}
        for f, fname in enumerate(factor_names):
            for k in range(self.num_relations):
                cos = F.cosine_similarity(g_a1[:, f, k], g_a0[:, f, k], dim=-1)
                dg = 1.0 - cos
                valid = mass[:, k] >= 0.5
                change[f"{fname}_R{k + 1}"] = {
                    "mean_all": float(dg.mean().item()),
                    "mean_valid": float(dg[valid].mean().item()) if bool(valid.any()) else None,
                }

        diag["context_change"] = change
        diag["effective_mass"] = {
            "mean": float(eff_mass.mean().item()),
            "frac_below_0.5": float((eff_mass < 0.5).float().mean().item()),
            "per_cell_mean": [[float(v) for v in row] for row in eff_mass.mean(dim=0).cpu().tolist()],
        }
        return diag
