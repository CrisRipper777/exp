"""P1 Bi-Axis model: Semantic Factor x Structural Relation decoupled graph
learning on top of the P0 semantic factorizer (architecture/objective kept
unchanged and JOINTLY optimized — review §16: P1 reuses, not weight-freezes).

    x_t, x_v -> P0 factorizer -> C / Pt / Pv           (M1, unchanged arch/obj, jointly trained)
    A -> topology signature -> R1..RK                  (M2, topology-only)
    beta_i^f  = sigmoid(MLP_B[f_i || g_bar_i^f])       (M3a, how much graph)
    alpha_ifk = Softmax_k(MLP_R[f_i || g_ik^f || f_i*g_ik^f || a_ik])  (M3b, which relation)
    g_i^f = sum_k alpha_ifk g_ik^f,  m_i^f = W0 g_i^f  (shared W0 only)
    f_i'  = LayerNorm(f_i + beta_i^f m_i^f)
    z = fusion([C' || Pt' || Pv'])                     (F1) or fusion_q(q') (F0)

Variants (plan §15/§22), one model + config switches:
    F0R0  model.p1.factor_aware=false model.p1.num_relations=1
    F1R0  model.p1.factor_aware=true  model.p1.num_relations=1
    F0R1  model.p1.factor_aware=false model.p1.num_relations=4
    F1R1  model.p1.factor_aware=true  model.p1.num_relations=4  (default)

P1 discipline: the relation axis reads ONLY edge_index; semantic factors enter
only at the coupling stage. graph modules never see raw x.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .biaxis_components import SemanticFactorizer
from .biaxis_p0 import Model as P0Model
from .biaxis_p1_components import (
    EdgeStructuralToken,
    FactorGraphBudget,
    FactorRelationSelector,
    RelationPrototypes,
    TopologyDiffusionSignature,
    compute_degree,
    compute_raw_struct_signature,
    neighbor_mean,
    relation_availability,
    relation_mass,
    relation_weighted_mean,
)
from .common import get_activation, make_norm


class Model(P0Model):
    """Bi-Axis P1. Inherits the P0 factorizer / recon heads / fusion / aux
    losses (architecture & objective unchanged, jointly optimized) and adds
    the graph-side modules (M2 + M3)."""

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)

        p1 = cfg.model.p1
        self.factor_aware = bool(p1.factor_aware)
        self.num_relations = int(p1.num_relations)
        self.relation_dim = int(p1.relation_dim)
        self.relation_temperature = float(p1.relation_temperature)
        self.use_graph_budget = bool(p1.get("use_graph_budget", True))
        self.budget_shared = bool(p1.get("budget_shared", False))
        self.edge_chunk_size = p1.get("edge_chunk_size")
        self.eps = float(p1.get("eps", 1.0e-8))
        self.relation_balance_weight = float(p1.get("relation_balance_weight", 0.0))
        self.alpha_entropy_weight = float(p1.get("alpha_entropy_weight", 0.0))
        self.budget_reg_weight = float(p1.get("budget_reg_weight", 0.0))

        activation = str(cfg.model.get("activation", "gelu"))
        norm = str(cfg.model.get("norm", "layernorm"))

        # --- M2: topology-only relation decomposition ---------------------
        self.struct_signature_mlp = TopologyDiffusionSignature(self.relation_dim, activation)
        self.edge_token_mlp = EdgeStructuralToken(self.relation_dim, activation)
        self.relation_prototypes = RelationPrototypes(
            self.num_relations, self.relation_dim, self.relation_temperature
        )

        # --- M3: factor-side graph consumers ------------------------------
        # Shared budget (B1) uses the SAME 128-d network as factor-specific
        # (B2): f_shared = mean_f f (review §17 — the old [F*d]-concat input
        # gave B1 MORE parameters than B2, an unclean ablation).
        self.graph_budget = FactorGraphBudget(
            self.factor_dim,
            hidden_dim=int(p1.budget_hidden_dim),
            activation=activation,
        )
        self.factor_selector = FactorRelationSelector(
            self.num_relations,
            self.factor_dim,
            hidden_dim=int(p1.selector_hidden_dim),
            activation=activation,
            input_norm=p1.get("selector_input_norm"),
        )

        # --- shared graph operator (P1 has ONLY W0, plan §13) -------------
        self.graph_w0 = nn.Linear(self.factor_dim, self.factor_dim, bias=False)
        self.graph_norm = make_norm(norm, self.factor_dim)

        # --- F0 path projectors (plan §15: q = Proj_q(Fusion_P0 output)) --
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

        # --- raw topology signature cache (plan §10) -----------------------
        # persistent=False: never enters state_dict; rebuilt from edge_index.
        self.register_buffer("_sig_cache_raw", torch.empty(0), persistent=False)
        self._sig_cache_ptr = -1
        self._sig_cache_n = -1

        # P1 graph modules need the complete topology at every step.
        self.requires_full_graph_training = bool(cfg.model.get("full_graph_training", True))

    # ------------------------------------------------------------------
    # Topology-only relation decomposition (M2)
    # ------------------------------------------------------------------

    def _get_raw_signature(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Cached deterministic raw signature s_raw = f(A) (plan §10).

        Cache is keyed on (num_nodes, edge_index.data_ptr()); the full-graph
        runner reuses the same hosted edge_index tensor, so the signature is
        computed once per process and only MLP_S / MLP_E / prototypes learn.
        """
        if self._sig_cache_n == num_nodes and self._sig_cache_ptr == edge_index.data_ptr():
            return self._sig_cache_raw
        raw = compute_raw_struct_signature(edge_index, num_nodes).detach()
        self._sig_cache_raw.resize_(raw.size(0), raw.size(1)).copy_(raw)
        self._sig_cache_ptr = edge_index.data_ptr()
        self._sig_cache_n = num_nodes
        return self._sig_cache_raw

    def _decompose_relations(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A -> (r [E,K], a [N,K], deg [N]).

        K == 1: strict fast path, r = ones, no prototype softmax (plan §22).
        Returns (r, availability, degree).
        """
        num_edges = int(edge_index.size(1))
        deg = compute_degree(edge_index, num_nodes)
        device = edge_index.device
        if self.num_relations == 1:
            r = torch.ones(num_edges, 1, dtype=torch.float32, device=device)
            mass = deg.unsqueeze(-1)
        else:
            raw = self._get_raw_signature(edge_index, num_nodes)
            s = self.struct_signature_mlp(edge_index, num_nodes, raw_signature=raw)
            e = self.edge_token_mlp(s, edge_index)
            r = self.relation_prototypes(e)
            mass = relation_mass(edge_index, r, num_nodes)
        availability = relation_availability(mass, deg)
        return r, availability, deg

    # ------------------------------------------------------------------
    # Graph update (M3, plan §11-§13)
    # ------------------------------------------------------------------

    def _graph_update(
        self,
        f_block: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> dict[str, torch.Tensor]:
        """Apply budget + selection + shared W0 to the factor block.

        f_block: [N, F, d_f] (F=3 factor-aware, F=1 factor-blind).
        Returns dict with f_tilde [N, F, d_f], beta [N, F], alpha [N, F, K],
        r [E, K], availability [N, K] (last ones for diagnostics).
        """
        num_factors = int(f_block.size(1))
        factor_dim = int(f_block.size(2))
        r, availability, _deg = self._decompose_relations(edge_index, num_nodes)

        # Relation-averaged context == plain neighbor mean (audit §4):
        # g_bar_i^f = sum_k a_ik g_ik^f = neighbor_mean(f).
        f_cat = f_block.reshape(num_nodes, num_factors * factor_dim)
        g_bar = neighbor_mean(edge_index, f_cat, num_nodes, edge_chunk_size=self.edge_chunk_size)
        g_bar_f = g_bar.reshape(num_nodes, num_factors, factor_dim)  # [N, F, d]

        # Per-relation weighted mean over the concatenated factor block
        # (one aggregation per relation, plan §9.2).
        g_cat, mass = relation_weighted_mean(
            edge_index, r, f_cat, num_nodes, edge_chunk_size=self.edge_chunk_size
        )  # [N, K, F*d]
        g_perm = g_cat.reshape(num_nodes, self.num_relations, num_factors, factor_dim)
        g_perm = g_perm.permute(0, 2, 1, 3)  # [N, F, K, d]

        # --- M3a: factor graph budget (how much graph evidence) -----------
        if self.use_graph_budget:
            if self.budget_shared:
                # B1: one per-node budget shared across factors, computed from
                # the factor-mean state with the same 128-d network as B2
                # (review §17: matched parameter capacity).
                f_shared = f_block.mean(dim=1)  # [N, d]
                g_bar_shared = g_bar_f.mean(dim=1)  # [N, d]
                beta_shared = self.graph_budget(f_shared, g_bar_shared)  # [N]
                beta = beta_shared.unsqueeze(-1).expand(num_nodes, num_factors)
            else:
                beta = self.graph_budget(f_block, g_bar_f)  # [N, F]
        else:
            beta = torch.ones(num_nodes, num_factors, dtype=f_block.dtype, device=f_block.device)

        # --- M3b: factor-relation selector (which relations) --------------
        # Loop over factors so the selector input never exceeds [N, 1, 3d+1].
        alpha_list = [
            self.factor_selector(f_block[:, idx : idx + 1], g_perm[:, idx : idx + 1], availability)
            for idx in range(num_factors)
        ]
        alpha = torch.cat(alpha_list, dim=1)  # [N, F, K]

        # --- shared W0 (plan §13) -----------------------------------------
        g_f = (alpha.unsqueeze(-1) * g_perm).sum(dim=2)  # [N, F, d]
        m_f = self.graph_w0(g_f.reshape(num_nodes * num_factors, factor_dim))
        m_f = m_f.reshape(num_nodes, num_factors, factor_dim)
        f_tilde = f_block + beta.unsqueeze(-1) * m_f
        f_tilde = self.graph_norm(f_tilde.reshape(num_nodes * num_factors, factor_dim))
        f_tilde = f_tilde.reshape(num_nodes, num_factors, factor_dim)

        return {
            "f_tilde": f_tilde,
            "beta": beta,
            "alpha": alpha,
            "r": r,
            "availability": availability,
        }

    # ------------------------------------------------------------------
    # Graph regularization losses (plan §20: OFF by default, weights 0)
    # ------------------------------------------------------------------

    def _graph_regularization(
        self,
        r: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Optional weak diagnostics-only regularizers; all weights are 0.0 by
        default and must be enabled as separate experiments (plan §20)."""
        loss = torch.zeros((), dtype=beta.dtype, device=beta.device)
        if self.relation_balance_weight > 0 and self.num_relations > 1:
            mean_r = r.mean(dim=0) + self.eps  # [K]
            loss = loss + self.relation_balance_weight * torch.sum(
                mean_r * torch.log(mean_r * self.num_relations)
            )
        if self.alpha_entropy_weight > 0 and self.num_relations > 1:
            log_alpha = torch.log(alpha + self.eps)
            entropy = -(alpha * log_alpha).sum(dim=-1).mean()
            loss = loss + self.alpha_entropy_weight * (-entropy)
        if self.budget_reg_weight > 0:
            loss = loss + self.budget_reg_weight * (beta * (1.0 - beta)).mean()
        return loss

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None):
        factors, z_local = self._encode(x)
        if self.training:
            aux_loss, aux_info = self._compute_aux(factors)
        else:
            aux_loss = z_local.new_tensor(0.0)
            aux_info = {}

        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        num_nodes = int(x.size(0))

        if self.factor_aware:
            f_block = torch.stack(
                [factors["c"], factors["p_t"], factors["p_v"]], dim=1
            )  # [N, 3, d_f]
        else:
            # Factor OFF (plan §15): still the SAME P0 factorizer, but the
            # graph module sees a single fused state q with no factor identity.
            q = self.proj_q(z_local)  # [N, d_f]
            f_block = q.unsqueeze(1)  # [N, 1, d_f]

        graph_out = self._graph_update(f_block, edge_index, num_nodes)
        f_tilde = graph_out["f_tilde"]

        if self.training and (self.relation_balance_weight or self.alpha_entropy_weight or self.budget_reg_weight):
            aux_loss = aux_loss + self._graph_regularization(
                graph_out["r"], graph_out["alpha"], graph_out["beta"]
            )

        if self.factor_aware:
            z = self.fusion(
                torch.cat([f_tilde[:, 0], f_tilde[:, 1], f_tilde[:, 2]], dim=-1)
            )
        else:
            z = self.fusion_q(f_tilde[:, 0])
        return z, None, None, aux_loss, aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """P1 inference = ONE exact full-graph forward (plan §23).

        P0's chunked per-node inference is invalid here: the graph module
        needs the complete neighborhood of every node. ``batch_size`` is
        accepted for API compatibility but unused.
        """
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
    # Mechanism diagnostics (plan §19)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_p1_diagnostics(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict:
        """Best-checkpoint mechanism diagnostics, JSON-safe aggregates only
        (plan §19). All nodes / all edges; NEVER uses labels; does not modify
        model state. Expects x / edge_index already on the model device."""
        self.eval()
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=x.device)
        factors, z_local = self._encode(x)
        num_nodes = int(x.size(0))
        if self.factor_aware:
            f_block = torch.stack(
                [factors["c"], factors["p_t"], factors["p_v"]], dim=1
            )
            factor_names = ["C", "Pt", "Pv"]
        else:
            f_block = self.proj_q(z_local).unsqueeze(1)
            factor_names = ["q"]

        graph_out = self._graph_update(f_block, edge_index, num_nodes)
        r, beta, alpha = graph_out["r"], graph_out["beta"], graph_out["alpha"]

        # --- relation occupancy / effective number (§19.1) -----------------
        occ = r.mean(dim=0)  # [K]
        if float(occ.sum()) > 0:
            occ = occ / occ.sum()
        effective_num = float(torch.exp(-(occ * torch.log(occ + self.eps)).sum()).item())
        mean_edge_entropy = float((-(r * torch.log(r + self.eps)).sum(dim=-1)).mean().item())

        # --- budget statistics (§19.2) -------------------------------------
        budget_stats: dict[str, dict[str, float]] = {}
        quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=beta.dtype, device=beta.device)
        for idx, name in enumerate(factor_names):
            b = beta[:, idx].reshape(-1)
            qs = torch.quantile(b, quantiles)
            budget_stats[name] = {
                "mean": float(b.mean().item()),
                "std": float(b.std(unbiased=False).item()),
                "p10": float(qs[0].item()),
                "p50": float(qs[1].item()),
                "p90": float(qs[2].item()),
                "low_frac": float((b < 0.05).float().mean().item()),
                "high_frac": float((b > 0.95).float().mean().item()),
            }

        # --- alpha entropy + node-wise JS (§19.3) --------------------------
        log_alpha = torch.log(alpha + self.eps)
        ent = -(alpha * log_alpha).sum(dim=-1)  # [N, F]
        alpha_entropy = {name: float(ent[:, idx].mean().item()) for idx, name in enumerate(factor_names)}

        js: dict[str, float] = {}
        num_factors = len(factor_names)
        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                p = alpha[:, i] + self.eps
                q = alpha[:, j] + self.eps
                m = 0.5 * (p + q)
                js_val = 0.5 * (
                    (p * torch.log(p / m)).sum(dim=-1) + (q * torch.log(q / m)).sum(dim=-1)
                )
                js[f"{factor_names[i]}_{factor_names[j]}"] = float(js_val.mean().item())

        # --- factor-relation usage matrix (§19.4) --------------------------
        usage = (beta.unsqueeze(-1) * alpha).mean(dim=0)  # [F, K]

        return {
            "relation": {
                "occ": [float(v) for v in occ.cpu().tolist()],
                "effective_num": effective_num,
                "mean_edge_entropy": mean_edge_entropy,
            },
            "budget": budget_stats,
            "alpha_entropy": alpha_entropy,
            "alpha_js": js,
            "usage_matrix": {
                "factors": factor_names,
                "relations": [f"R{i + 1}" for i in range(self.num_relations)],
                "values": [[float(v) for v in row] for row in usage.cpu().tolist()],
            },
        }
