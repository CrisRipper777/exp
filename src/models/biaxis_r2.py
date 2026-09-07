"""R2-Design-1 model: minimal end-to-end Factor-Context Functional Modulation.

Plan: docs/BiAxis_R2_Design_1_Implementation_Validation_Plan.md.

    Text / Visual embeddings
            |
            v
    P0 Semantic Ownership (unchanged factorizer + aux losses)
    c_t, c_v, p_t, p_v            [aux losses act ONLY here, plan §8]
            |
            v
    (S/J only) Semantic Refiner
      C^0 = node-adaptive common consensus (AdaptiveCommonGate)
      F*  = F^0 + Delta^b (zero-init factor interaction residual)
            |                    (F: F* = F^0, plan §9)
            v
    Simple 1-hop factor-wise aggregation  N^a = neighbor_mean(F^{a,*})
            |
            +--- B0 diagonal path (ALWAYS):  rho_base^b * LN_base^b(V_b(N^b))
            |
            +--- (F/J only) Functional residual:
                     m^{a->b} = sigmoid(scorer(u^{a->b})) * V_a(N^a)
                     M_func^b = mean_a m^{a->b};  rho_func^b * LN_func^b(M_func^b)
            |
            v
    F'^b = F^{b,*} + base + func      -> existing P0 fusion -> z_final

Four variants share this ONE implementation (plan §3/§36):
    B0: semantic OFF / functional OFF (clean parent)
    F : semantic OFF / functional ON  (B0 + minimal functional residual)
    S : semantic ON  / functional OFF
    J : semantic ON  / functional ON

Discipline:
    - P0 factorizer / recon heads / fusion / aux losses are inherited
      UNCHANGED from biaxis_p0 (plan §1.1/§8/§15); P1/P2/P3 files untouched.
    - No K relations / Gamma / OFR / Gsim-Gdiff / multi-hop / edge attention
      (plan §1.3/§10).
    - msg LayerNorms use bias=False so LN(0)=0: isolated nodes get an EXACT
      zero graph residual (plan §4.4, unit-tested).
    - Functional scorer gates are independent sigmoids; Softmax over sources
      is forbidden (plan §11).
    - rho_func is a direct LayerScale parameter init 0.01 (no sigmoid,
      plan §13); rho_base = sigmoid(raw), init 0.5 (plan §4.4).
    - No new losses: architecture + original P0 aux only (plan §8).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_p0 import Model as P0Model
from .biaxis_p1_components import neighbor_mean
from .biaxis_r2_components import (
    AdaptiveCommonGate,
    FunctionalScorer,
    SemanticInteractionResidual,
)

FACTOR_NAMES = ("C", "Pt", "Pv")


class Model(P0Model):
    """Bi-Axis R2. Inherits the P0 factorizer / recon heads / fusion / aux
    losses and adds (1) the always-on B0 diagonal 1-hop path, (2) the
    optional ownership-preserving semantic refiner, (3) the optional
    target-conditioned functional residual."""

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)

        sem = cfg.model.semantic_refiner
        func = cfg.model.functional_transfer
        self.semantic_enabled = bool(sem.enabled)
        self.functional_enabled = bool(func.enabled)

        activation = str(cfg.model.get("activation", "gelu"))
        norm = str(cfg.model.get("norm", "layernorm"))
        self.edge_chunk_size = cfg.model.get("edge_chunk_size")

        # --- B0 diagonal path (always present, plan §4) -------------------
        # One light source transform per factor; NO 9 full W_ab (plan §4.3).
        self.source_transforms = nn.ModuleList(
            [nn.Linear(self.factor_dim, self.factor_dim, bias=False) for _ in range(3)]
        )
        # bias=False: LN(0)=0 so isolated nodes keep F' = F* EXACTLY (§4.4).
        self.msg_norm_base = nn.ModuleList(
            [nn.LayerNorm(self.factor_dim, bias=False) for _ in range(3)]
        )
        self.raw_rho_base = nn.Parameter(torch.zeros(3))  # rho = sigmoid(raw) = 0.5

        # --- Semantic Refiner (S/J only, plan §6/§7) ----------------------
        if self.semantic_enabled:
            self.adaptive_common = AdaptiveCommonGate(
                self.factor_dim,
                hidden_dim=int(sem.get("gate_hidden", 64)),
                activation=activation,
            )
            self.semantic_residual = SemanticInteractionResidual(
                self.factor_dim,
                dropout=float(sem.get("dropout", cfg.model.dropout)),
                activation=activation,
                norm=norm,
            )

        # --- Functional Transfer (F/J only, plan §11-§13) -----------------
        if self.functional_enabled:
            self.type_dim = int(func.get("type_dim", 8))
            self.func_scorer = FunctionalScorer(
                self.factor_dim,
                type_dim=self.type_dim,
                hidden_dim=int(func.get("gate_hidden", 64)),
                activation=activation,
            )
            self.src_type_emb = nn.Embedding(3, self.type_dim)
            self.tgt_type_emb = nn.Embedding(3, self.type_dim)
            self.msg_norm_func = nn.ModuleList(
                [nn.LayerNorm(self.factor_dim, bias=False) for _ in range(3)]
            )
            self.rho_func = nn.Parameter(
                torch.full((3,), float(func.get("rho_func_init", 0.01)))
            )

        # Full-graph training protocol (same as P1/P2/P3).
        self.requires_full_graph_training = bool(cfg.model.get("full_graph_training", True))

    # ------------------------------------------------------------------
    # Semantic ownership states (plan §6/§7)
    # ------------------------------------------------------------------

    def _ownership_states(
        self, factors: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """F^0 / F* = [N, 3, d], factor order [C, Pt, Pv] (unit-tested).

        Returns (f0, f_star, w) where w = common gate weights [N, 2]
        (None when the refiner is off).
        """
        if self.semantic_enabled:
            c0, w = self.adaptive_common(factors["c_t"], factors["c_v"])
            f0 = torch.stack([c0, factors["p_t"], factors["p_v"]], dim=1)
            delta = self.semantic_residual(f0)  # [N, 3, d], zero-init
            return f0, f0 + delta, w
        # B0/F: current fixed common consensus c = (c_t + c_v) / 2 (plan §4.1).
        f0 = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        return f0, f0, None

    # ------------------------------------------------------------------
    # Graph update (plan §4/§10-§13)
    # ------------------------------------------------------------------

    def _graph_update(
        self,
        f_star: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """B0 diagonal path + optional functional residual.

        Returns (f_out [N,3,d], n_block [N,3,d], base_msg [N,3,d],
        func_msg [N,3,d] | None). Factor order [C, Pt, Pv] everywhere.
        """
        d = self.factor_dim
        f_cat = f_star.reshape(num_nodes, 3 * d)
        n_cat = neighbor_mean(
            edge_index, f_cat, num_nodes, edge_chunk_size=self.edge_chunk_size
        )
        n_block = n_cat.reshape(num_nodes, 3, d)  # N^a for a in [C, Pt, Pv]

        # v_i^a = V_a(N_i^a): shared by the B0 diagonal path AND the
        # functional path (plan §12: source transform 与 B0 共用).
        v_block = torch.stack(
            [self.source_transforms[a](n_block[:, a]) for a in range(3)], dim=1
        )  # [N, 3, d]

        base_msg = torch.stack(
            [self.msg_norm_base[b](v_block[:, b]) for b in range(3)], dim=1
        )  # [N, 3, d]

        func_msg = self._functional_message(f_star, n_block, v_block) if self.functional_enabled else None

        rho_base = torch.sigmoid(self.raw_rho_base)  # [3]
        f_out = f_star + rho_base.view(1, 3, 1) * base_msg
        if func_msg is not None:
            f_out = f_out + self.rho_func.view(1, 3, 1) * func_msg
        return f_out, n_block, base_msg, func_msg

    def _functional_message(
        self,
        f_star: torch.Tensor,
        n_block: torch.Tensor,
        v_block: torch.Tensor,
    ) -> torch.Tensor:
        """3x3 functional cells -> per-target functional message [N, 3, d].

        On-the-fly cell construction (plan §17): the [N, 4d+2*type_dim]
        scorer input is built and freed per (b, a) cell; the forbidden
        [N,3,3,4d] / [N,9,d] shapes are never materialized.
        """
        num_nodes = int(f_star.size(0))
        src_t = self.src_type_emb.weight  # [3, td]
        tgt_t = self.tgt_type_emb.weight  # [3, td]
        msgs_per_target: list[torch.Tensor] = []
        for b in range(3):
            tgt_emb = tgt_t[b].unsqueeze(0).expand(num_nodes, -1)  # [N, td]
            acc: torch.Tensor | None = None
            for a in range(3):
                u = torch.cat(
                    [
                        f_star[:, b],
                        n_block[:, a],
                        f_star[:, b] * n_block[:, a],
                        (f_star[:, b] - n_block[:, a]).abs(),
                        src_t[a].unsqueeze(0).expand(num_nodes, -1),
                        tgt_emb,
                    ],
                    dim=-1,
                )  # [N, 4d + 2td]
                g = torch.sigmoid(self.func_scorer(u))  # [N, 1]
                m = g * v_block[:, a]  # m^{a->b} = g^{a->b} * v^a
                acc = m if acc is None else acc + m
            # M_func^b = (1/3) sum_a m^{a->b} (plan §12).
            msgs_per_target.append(self.msg_norm_func[b](acc / 3.0))
        return torch.stack(msgs_per_target, dim=1)  # [N, 3, d]

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None):
        factors, z_local = self._encode(x)
        if self.training:
            # P0 aux losses act on the BASE decomposition only (plan §8).
            aux_loss, aux_info = self._compute_aux(factors)
        else:
            aux_loss = z_local.new_tensor(0.0)
            aux_info = {}

        f0, f_star, _w = self._ownership_states(factors)
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        num_nodes = int(x.size(0))
        f_out, _n_block, _base_msg, _func_msg = self._graph_update(f_star, edge_index, num_nodes)

        # Existing P0 fusion, unchanged (plan §15).
        z = self.fusion(torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1))
        return z, None, None, aux_loss, aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """One exact full-graph forward (same as P1; chunked per-node
        inference is invalid because the graph path needs full neighborhoods).
        ``batch_size`` accepted for API compatibility but unused."""
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
    # Mechanism diagnostics (plan §18)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_r2_diagnostics(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict:
        """Best-checkpoint mechanism diagnostics, JSON-safe aggregates only.
        NEVER uses labels; does not modify model state. Expects x /
        edge_index already on the model device."""
        self.eval()
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=x.device)
        factors, _z_local = self._encode(x)
        num_nodes = int(x.size(0))

        # --- P0 ownership health (same keys as the training aux info) -----
        aux_loss, aux_info = self._compute_aux(factors)
        diag: dict = {
            "p0": {key: float(value.item()) for key, value in aux_info.items()},
            "p0_aux_loss": float(aux_loss.item()),
        }

        # --- semantic refiner diagnostics (plan §18.1) --------------------
        f0, f_star, w = self._ownership_states(factors)
        diag["semantic"] = None
        if self.semantic_enabled:
            diag["semantic"] = self._semantic_diagnostics(w, f0, f_star)

        # --- graph path diagnostics (plan §18.2) --------------------------
        f_out, n_block, base_msg, func_msg = self._graph_update(
            f_star, edge_index, num_nodes
        )
        rho_base = torch.sigmoid(self.raw_rho_base)  # [3]
        diag["rho_base"] = [float(v) for v in rho_base.cpu().tolist()]
        diag["base_residual_ratio"] = self._residual_ratio_stats(
            rho_base.view(1, 3, 1) * base_msg, f_star
        )

        diag["functional"] = None
        if self.functional_enabled:
            diag["functional"] = self._functional_diagnostics(f_star, n_block, func_msg)
        return diag

    def _semantic_diagnostics(
        self,
        w: torch.Tensor,
        f0: torch.Tensor,
        f_star: torch.Tensor,
    ) -> dict:
        """Common gate weight stats + per-factor semantic residual ratio
        (plan §18.1)."""

        def _weight_stats(col: torch.Tensor) -> dict[str, float]:
            return {
                "mean": float(col.mean().item()),
                "std": float(col.std(unbiased=False).item()),
                "frac_lt_05": float((col < 0.05).float().mean().item()),
                "frac_gt_95": float((col > 0.95).float().mean().item()),
            }

        delta = f_star - f0
        return {
            "w_t": _weight_stats(w[:, 0]),
            "w_v": _weight_stats(w[:, 1]),
            "sem_residual_ratio": self._residual_ratio_stats(delta, f0),
        }

    def _residual_ratio_stats(
        self, residual: torch.Tensor, reference: torch.Tensor
    ) -> dict[str, dict[str, float]]:
        """Per-factor node-wise ratio ||residual^b|| / (||reference^b|| + eps)
        -> {factor: {mean, std}} (plan §18.1/§18.2)."""
        eps = 1e-8
        ratio = residual.norm(dim=-1) / (reference.norm(dim=-1) + eps)  # [N, 3]
        out: dict[str, dict[str, float]] = {}
        for idx, name in enumerate(FACTOR_NAMES):
            r = ratio[:, idx]
            out[name] = {
                "mean": float(r.mean().item()),
                "std": float(r.std(unbiased=False).item()),
            }
        return out

    def _functional_diagnostics(
        self,
        f_star: torch.Tensor,
        n_block: torch.Tensor,
        func_msg: torch.Tensor,
    ) -> dict:
        """3x3 gate matrix, 3x3 message contribution matrix, rho_func and
        the functional residual ratio (plan §18.2). Recomputes the cells
        under no-grad (deterministic: no dropout inside the scorer); the
        inputs f_star / n_block are the SAME tensors the forward path used,
        so the recomputed gates equal the trained gates."""
        num_nodes = int(f_star.size(0))
        src_t = self.src_type_emb.weight  # [3, td]
        tgt_t = self.tgt_type_emb.weight  # [3, td]
        gates = torch.zeros(num_nodes, 3, 3, dtype=f_star.dtype, device=f_star.device)
        contrib = torch.zeros(3, 3, dtype=f_star.dtype, device=f_star.device)
        for b in range(3):
            tgt_emb = tgt_t[b].unsqueeze(0).expand(num_nodes, -1)
            cell_msgs: list[torch.Tensor] = []
            cell_gates: list[torch.Tensor] = []
            for a in range(3):
                n_a = n_block[:, a]
                u = torch.cat(
                    [
                        f_star[:, b],
                        n_a,
                        f_star[:, b] * n_a,
                        (f_star[:, b] - n_a).abs(),
                        src_t[a].unsqueeze(0).expand(num_nodes, -1),
                        tgt_emb,
                    ],
                    dim=-1,
                )
                g = torch.sigmoid(self.func_scorer(u)).squeeze(-1)  # [N]
                v_a = self.source_transforms[a](n_a)
                cell_msgs.append(g.unsqueeze(-1) * v_a)
                cell_gates.append(g)
            # Plan §18.2: C_{a->b} = E_i||m^{a->b}|| / E_i sum_{a'}||m^{a'->b}||
            # (ratio of expectations, NOT expectation of per-node ratios).
            norm_means = torch.stack([m.norm(dim=-1).mean() for m in cell_msgs])  # [3]
            norm_sum_mean = norm_means.sum()
            for a in range(3):
                contrib[a, b] = norm_means[a] / (norm_sum_mean + 1e-8)
            gates[:, :, b] = torch.stack(cell_gates, dim=1)  # [N, 3]

        quantiles = torch.tensor([0.05, 0.5, 0.95], dtype=gates.dtype, device=gates.device)
        stats = {
            "mean": [[0.0] * 3 for _ in range(3)],
            "std": [[0.0] * 3 for _ in range(3)],
            "p05": [[0.0] * 3 for _ in range(3)],
            "p50": [[0.0] * 3 for _ in range(3)],
            "p95": [[0.0] * 3 for _ in range(3)],
            "frac_lt_05": [[0.0] * 3 for _ in range(3)],
            "frac_gt_95": [[0.0] * 3 for _ in range(3)],
        }
        for a in range(3):
            for b in range(3):
                col = gates[:, a, b]
                qs = torch.quantile(col, quantiles)
                stats["mean"][a][b] = float(col.mean().item())
                stats["std"][a][b] = float(col.std(unbiased=False).item())
                stats["p05"][a][b] = float(qs[0].item())
                stats["p50"][a][b] = float(qs[1].item())
                stats["p95"][a][b] = float(qs[2].item())
                stats["frac_lt_05"][a][b] = float((col < 0.05).float().mean().item())
                stats["frac_gt_95"][a][b] = float((col > 0.95).float().mean().item())

        return {
            "gate_matrix": {
                "rows": ["src_C", "src_Pt", "src_Pv"],
                "cols": ["tgt_C", "tgt_Pt", "tgt_Pv"],
                **stats,
            },
            "contribution_matrix": {
                "rows": ["src_C", "src_Pt", "src_Pv"],
                "cols": ["tgt_C", "tgt_Pt", "tgt_Pv"],
                "values": [[float(v) for v in row] for row in contrib.cpu().tolist()],
            },
            "rho_func": [float(v) for v in self.rho_func.cpu().tolist()],
            "func_residual_ratio": self._residual_ratio_stats(
                self.rho_func.view(1, 3, 1) * func_msg, f_star
            ),
        }
