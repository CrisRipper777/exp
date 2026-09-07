"""OSRA model: Ownership-Specific Relational Access for Multimodal
Attributed Graphs (docs/OSRA_下一阶段推进计划.md).

Core principle: "Propagate within ownership; query across ownership."

    x_t, x_v -> P0 factorizer -> C / Pt / Pv            (Stage I, unchanged)
    H^(0) = [C; Pt; Pv]     ownership state tensor [N, 3, d]
    H^(l+1) = OsraBlock(H^(l), A, H^(0))                (Stage II, L blocks)
    z = P0 fusion([C' | Pt' | Pv'])                     (Stage III)

S1 (plan §5-§7): the block is the Strong Same-Ownership Local Block —
per-factor neighbor mean / GATv2 aggregation + ego + PreNorm 2-layer GELU
FFN + eta-scaled residual, optional initial ownership anchor. NO
cross-factor interaction.

S2 (plan §9-§13, gated on S1): the block additionally performs
Target-Driven Cross-Ownership Relational Access (cross_access=true):
target-conditioned slot queries over typed source evidence of the OTHER
ownership factors only, with a learnable Null/No-Access state, then
target-specific write-back.

Discipline: single code path + config switches for all L0-L3 / Q0-Q4
variants; P0 factorizer/aux/fusion reused unchanged; val-only selection
until S7.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_p0 import Model as P0Model
from .osra_components import OsraBlock, OsraLocalBlock, OsraRelationalAccess


class Model(P0Model):
    """OSRA encoder. Inherits the P0 factorizer / recon heads / aux losses /
    final fusion; adds the ownership-structured block stack."""

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)
        o = cfg.model.osra
        self.num_blocks = int(o.num_blocks)
        self.local_mode = str(o.local_mode)
        self.anchor_mix = float(o.get("anchor_mix", 0.0))
        self.cross_access = bool(o.get("cross_access", False))
        self.access_mode = str(o.get("access_mode", "query"))
        self.query_mode = str(o.get("query_mode", "target"))
        self.use_null = bool(o.get("use_null", True))
        self.log_local_stats = bool(o.get("log_local_stats", True))
        self.log_access_stats = bool(o.get("log_access_stats", True))
        self.edge_chunk_size = int(o.get("edge_chunk_size", 200000))
        self.memory_checkpoint = bool(o.get("memory_checkpoint", True))

        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            local = OsraLocalBlock(
                factor_dim=self.factor_dim,
                ffn_expand_ratio=float(o.get("ffn_expand_ratio", 2.0)),
                layer_scale_init=float(o.get("layer_scale_init", 0.1)),
                anchor_mix=self.anchor_mix,
                local_mode=self.local_mode,
                edge_chunk_size=self.edge_chunk_size,
                dropout=float(cfg.model.dropout),
                activation=str(cfg.model.get("activation", "gelu")),
                norm=str(cfg.model.get("norm", "layernorm")),
                memory_checkpoint=self.memory_checkpoint,
            )
            access = None
            if self.cross_access:
                access = OsraRelationalAccess(
                    factor_dim=self.factor_dim,
                    relation_dim=int(o.get("relation_dim", 128)),
                    num_slots=int(o.get("num_slots", 2)),
                    access_mode=self.access_mode,
                    query_mode=self.query_mode,
                    use_null=self.use_null,
                    relational_scale_init=float(o.get("relational_scale_init", 0.05)),
                    edge_chunk_size=self.edge_chunk_size,
                    dropout=float(cfg.model.dropout),
                    activation=str(cfg.model.get("activation", "gelu")),
                    norm=str(cfg.model.get("norm", "layernorm")),
                    memory_checkpoint=self.memory_checkpoint,
                )
            self.blocks.append(OsraBlock(local, access))

        self.requires_full_graph_training = bool(cfg.model.get("full_graph_training", True))

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None):
        x_t, x_v = self._split_modalities(x)
        factors = self.factorizer(x_t, x_v)
        H = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)  # [N, 3, d]

        if self.training:
            aux_loss, aux_info = self._compute_aux(factors)
        else:
            aux_loss = H.new_tensor(0.0)
            aux_info = {}

        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        num_nodes = int(x.size(0))

        H0 = H if self.anchor_mix > 0.0 else None
        for layer_idx, block in enumerate(self.blocks):
            H, block_stats = block(
                H, edge_index, num_nodes, H0,
                collect_local=self.training and self.log_local_stats,
                collect_access=self.training and self.log_access_stats,
            )
            if self.training:
                for key, value in block_stats.items():
                    aux_info[f"osra_l{layer_idx + 1}_{key}"] = value

        z = self.fusion(torch.cat([H[:, 0], H[:, 1], H[:, 2]], dim=-1))
        return z, None, None, aux_loss, aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """OSRA inference = ONE exact full-graph forward (same convention as
        P1/R3): the graph operator needs every node's complete neighborhood.
        ``batch_size`` is accepted for API compatibility but unused. Eval-mode
        forward has no dropout, so inference == eval forward."""
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
