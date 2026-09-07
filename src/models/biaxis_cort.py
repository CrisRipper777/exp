"""R2D29 CORT model: Coordinated Ownership-Relational Transfer
(docs/BiAxis_R2D29_System_Level_Performance_Advancement_Plan.md).

System-level architecture on top of the A0 parent (biaxis_p3 / biaxis_final):
the A0 factorizer, `_graph_update` and fusion are REUSED (never copied), and
the CORT block adds the complete pathway

    Relational Allocation -> Source Preservation -> Target-conditioned
    Interaction -> Ownership-state Update   ->   Ownership Interaction Fusion

Backbone modes (plan §6.3):
    a0_augment : P0 factors -> A0 graph_update -> CORT x L -> fusion
    pre_a0     : P0 factors -> CORT x L -> A0 graph_update -> fusion
    sandwich   : P0 factors -> CORT_pre -> A0 graph_update -> CORT_post -> fusion
    replace    : P0 factors -> CORT x L -> fusion        (no K=4/Gamma/OFR)
    hybrid     : P0 factors -> A0 path || CORT path -> factor-space merger -> fusion

Discipline:
    - recurrent blocks recompute routing from the UPDATED factor states every
      layer (plan §8.3) — edge weights are never fixed across layers;
    - late write-back projects the deltas to hidden_dim and adds them as a
      z-space residual (plan §7.2 W0);
    - every cort_* training statistic is detached and JSON-safe;
    - no Test access anywhere (labels never touch the model).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_p3 import Model as P3Model
from .biaxis_cort_components import (
    CortBlock,
    CortMerger,
    FactorAttnFusion,
    FactorTypeEmbedding,
    NUM_FACTORS,
    OifFusion,
)

BACKBONE_MODES = ("a0_augment", "pre_a0", "sandwich", "replace", "hybrid")
ROUTER_MODES = ("uniform", "target_null", "pair_null")
SOURCE_MODES = ("mean", "preserve_concat", "preserve_attn")
WRITEBACK_MODES = ("late", "factor")
FUSION_MODES = ("legacy", "oif", "factor_attn")


class Model(P3Model):
    """Bi-Axis CORT. Inherits the A0 (P3) parent: P0 factorizer + aux losses,
    M2 relation decomposition, NullSoftmax transport graph_update, and the
    legacy fusion. Adds the configurable CORT pathway."""

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)

        cort = cfg.model.cort
        self.cort_backbone_mode = str(cort.backbone_mode)
        assert self.cort_backbone_mode in BACKBONE_MODES, self.cort_backbone_mode
        self.cort_router_mode = str(cort.router_mode)
        assert self.cort_router_mode in ROUTER_MODES, self.cort_router_mode
        self.cort_source_mode = str(cort.source_mode)
        assert self.cort_source_mode in SOURCE_MODES, self.cort_source_mode
        self.cort_writeback_mode = str(cort.writeback_mode)
        assert self.cort_writeback_mode in WRITEBACK_MODES, self.cort_writeback_mode
        self.cort_fusion_mode = str(cort.fusion_mode)
        assert self.cort_fusion_mode in FUSION_MODES, self.cort_fusion_mode

        self.cort_num_blocks = int(cort.num_blocks)
        self.cort_num_blocks_pre = int(cort.get("num_blocks_pre", 1))
        self.cort_num_blocks_post = int(cort.get("num_blocks_post", 1))
        self.cort_share_blocks = bool(cort.get("share_blocks", False))
        self.cort_type_dim = int(cort.get("type_dim", 16))
        self.cort_interaction_hidden_mult = float(cort.get("interaction_hidden_mult", 2.0))
        self.cort_fusion_hidden_mult = float(cort.get("fusion_hidden_mult", 2.0))
        self.cort_residual_init = float(cort.get("residual_init", 0.0))
        self.cort_pre_norm = bool(cort.get("pre_norm", True))
        self.cort_dropout = float(cort.get("dropout", float(cfg.model.dropout)))
        self.cort_edge_chunk_size = int(cort.get("edge_chunk_size", 50000))
        self.cort_memory_checkpoint = bool(cort.get("memory_checkpoint", True))
        self.cort_mean_dup = bool(cort.get("mean_dup", False))

        activation = str(cfg.model.get("activation", "gelu"))
        norm = str(cfg.model.get("norm", "layernorm"))

        self.type_emb = FactorTypeEmbedding(self.cort_type_dim)

        def make_block() -> CortBlock:
            return CortBlock(
                self.factor_dim, self.type_emb,
                router_mode=self.cort_router_mode,
                source_mode=self.cort_source_mode,
                writeback=(self.cort_writeback_mode == "factor"),
                type_dim=self.cort_type_dim,
                interaction_hidden_mult=self.cort_interaction_hidden_mult,
                residual_init=self.cort_residual_init,
                pre_norm=self.cort_pre_norm,
                dropout=self.cort_dropout,
                activation=activation,
                norm=norm,
                edge_chunk_size=self.cort_edge_chunk_size,
                memory_checkpoint=self.cort_memory_checkpoint,
                mean_dup=self.cort_mean_dup,
            )

        # Only the block sets the backbone mode actually uses are built
        # (unused sets would add ~3.3M dead params each at d=128).
        needs_main = self.cort_backbone_mode in ("a0_augment", "pre_a0", "replace", "hybrid")
        needs_pre_post = self.cort_backbone_mode == "sandwich"
        if self.cort_share_blocks:
            # one shared block instance for every application (plan §6.2)
            self.cort_block = make_block()
            self.cort_blocks = nn.ModuleList()
            self.cort_pre_blocks = nn.ModuleList()
            self.cort_post_blocks = nn.ModuleList()
        else:
            self.cort_block = None
            self.cort_blocks = (
                nn.ModuleList([make_block() for _ in range(self.cort_num_blocks)])
                if needs_main else nn.ModuleList()
            )
            self.cort_pre_blocks = (
                nn.ModuleList([make_block() for _ in range(self.cort_num_blocks_pre)])
                if needs_pre_post else nn.ModuleList()
            )
            self.cort_post_blocks = (
                nn.ModuleList([make_block() for _ in range(self.cort_num_blocks_post)])
                if needs_pre_post else nn.ModuleList()
            )

        # fusion (plan §4.5); legacy = the inherited P0/A0 fusion (which
        # expects the flattened [C|Pt|Pv] concat, not the [N,3,d] block)
        if self.cort_fusion_mode == "legacy":
            self.cort_fusion = self.fusion
            self.cort_fusion_flatten = True
        elif self.cort_fusion_mode == "oif":
            self.cort_fusion = OifFusion(
                self.factor_dim, self.hidden_dim, self.cort_fusion_hidden_mult,
                self.cort_dropout, activation, norm,
            )
        else:  # factor_attn
            self.cort_fusion = FactorAttnFusion(
                self.factor_dim, self.hidden_dim, dropout=self.cort_dropout,
            )
            self.cort_fusion_flatten = False
        if self.cort_fusion_mode == "oif":
            self.cort_fusion_flatten = False

        # late integration: z-space residual via projection to hidden_dim
        # (plan §7.2 W0; rho gating per factor, ReZero-style init)
        if self.cort_writeback_mode == "late":
            self.cort_late_proj = nn.Linear(NUM_FACTORS * self.factor_dim, self.hidden_dim)
            self.cort_late_rhos = nn.Parameter(
                torch.full((NUM_FACTORS,), self.cort_residual_init)
            )
        else:
            self.cort_late_proj = None

        if self.cort_backbone_mode == "hybrid":
            self.cort_merger = CortMerger(self.factor_dim)
        else:
            self.cort_merger = None

    # ------------------------------------------------------------------
    # Block iteration helpers
    # ------------------------------------------------------------------

    def _block_list(self, kind: str) -> list[CortBlock]:
        if self.cort_share_blocks:
            assert self.cort_block is not None
            n = {
                "main": self.cort_num_blocks,
                "pre": self.cort_num_blocks_pre,
                "post": self.cort_num_blocks_post,
            }[kind]
            return [self.cort_block] * n
        return list({"main": self.cort_blocks, "pre": self.cort_pre_blocks,
                     "post": self.cort_post_blocks}[kind])

    def _apply_blocks(self, blocks: list[CortBlock], f_cur: torch.Tensor,
                      edge_index: torch.Tensor, num_nodes: int, tag: str
                      ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Applies CORT blocks sequentially; routing is recomputed from the
        updated factor states at every layer (plan §8.3). Returns
        (f_cur, deltas_sum, merged_stats)."""
        deltas_sum = torch.zeros_like(f_cur)
        stats: dict = {}
        for l, blk in enumerate(blocks):
            f_cur, deltas, lstats = blk(f_cur, edge_index, num_nodes)
            deltas_sum = deltas_sum + deltas
            for key, value in lstats.items():
                stats[f"{tag}{l}_{key}"] = value
        return f_cur, deltas_sum, stats

    # ------------------------------------------------------------------
    # Forward
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
        f_block = torch.stack(
            [factors["c"], factors["p_t"], factors["p_v"]], dim=1
        )  # [N, 3, d]

        cort_stats: dict = {}
        deltas_sum = None
        mode = self.cort_backbone_mode

        if mode == "replace":
            f_cur, deltas_sum, cort_stats = self._apply_blocks(
                self._block_list("main"), f_block, edge_index, num_nodes, "L")
        elif mode == "pre_a0":
            f_cur, deltas_sum, cort_stats = self._apply_blocks(
                self._block_list("main"), f_block, edge_index, num_nodes, "L")
            f_cur = self._graph_update(f_cur, edge_index, num_nodes)["f_tilde"]
        elif mode == "a0_augment":
            f_cur = self._graph_update(f_block, edge_index, num_nodes)["f_tilde"]
            f_cur, deltas_sum, cort_stats = self._apply_blocks(
                self._block_list("main"), f_cur, edge_index, num_nodes, "L")
        elif mode == "sandwich":
            f_cur, d_pre, s_pre = self._apply_blocks(
                self._block_list("pre"), f_block, edge_index, num_nodes, "Lpre")
            f_cur = self._graph_update(f_cur, edge_index, num_nodes)["f_tilde"]
            f_cur, d_post, s_post = self._apply_blocks(
                self._block_list("post"), f_cur, edge_index, num_nodes, "Lpost")
            deltas_sum = d_pre + d_post
            cort_stats = {**s_pre, **s_post}
        elif mode == "hybrid":
            f_a0 = self._graph_update(f_block, edge_index, num_nodes)["f_tilde"]
            f_cort, deltas_sum, cort_stats = self._apply_blocks(
                self._block_list("main"), f_block, edge_index, num_nodes, "L")
            f_cur = self.cort_merger(f_a0, f_cort)

        if self.cort_fusion_flatten:
            z = self.cort_fusion(f_cur.reshape(num_nodes, -1))
        else:
            z = self.cort_fusion(f_cur)

        if self.cort_writeback_mode == "late" and deltas_sum is not None:
            gated = torch.stack(
                [self.cort_late_rhos[b] * deltas_sum[:, b] for b in range(NUM_FACTORS)],
                dim=1,
            )  # [N, 3, d]
            z = z + self.cort_late_proj(gated.reshape(num_nodes, -1))

        if self.training and cort_stats:
            aux_info = dict(aux_info)
            aux_info["cort_stats"] = cort_stats
        return z, None, None, aux_loss, aux_info

    # ------------------------------------------------------------------
    # Framework interface (P1 full-graph inference, exact same path)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """One exact full-graph forward (the CORT block needs the complete
        neighborhood of every node); ``batch_size`` accepted for API
        compatibility but unused."""
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
    # CORT mechanism diagnostics (eval-mode, JSON-safe, no labels)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_cort_diagnostics(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict:
        """One eval-mode pass through the CORT pathway (same math as forward)
        and per-block routing/interaction/write-back statistics. Never uses
        labels; does not modify model state."""
        self.eval()
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=x.device)
        factors, _z_local = self._encode(x)
        num_nodes = int(x.size(0))
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)

        out: dict = {"backbone_mode": self.cort_backbone_mode,
                     "router_mode": self.cort_router_mode,
                     "source_mode": self.cort_source_mode,
                     "writeback_mode": self.cort_writeback_mode,
                     "fusion_mode": self.cort_fusion_mode,
                     "num_blocks": self.cort_num_blocks}

        mode = self.cort_backbone_mode
        if mode == "replace":
            _f, _, s = self._apply_blocks(
                self._block_list("main"), f_block, edge_index, num_nodes, "L")
            out["main_stats"] = s
        elif mode == "pre_a0":
            _f, _, s = self._apply_blocks(
                self._block_list("main"), f_block, edge_index, num_nodes, "L")
            out["main_stats"] = s
        elif mode == "a0_augment":
            f_t = self._graph_update(f_block, edge_index, num_nodes)["f_tilde"]
            _f, _, s = self._apply_blocks(
                self._block_list("main"), f_t, edge_index, num_nodes, "L")
            out["main_stats"] = s
        elif mode == "sandwich":
            f_pre, _, s_pre = self._apply_blocks(
                self._block_list("pre"), f_block, edge_index, num_nodes, "Lpre")
            f_t = self._graph_update(f_pre, edge_index, num_nodes)["f_tilde"]
            _f, _, s_post = self._apply_blocks(
                self._block_list("post"), f_t, edge_index, num_nodes, "Lpost")
            out["pre_stats"] = s_pre
            out["post_stats"] = s_post
        elif mode == "hybrid":
            # CORT path starts from the raw factors; the A0 path is the
            # inherited graph_update (its stats live in P2/P3 diagnostics)
            _f, _, s = self._apply_blocks(
                self._block_list("main"), f_block, edge_index, num_nodes, "L")
            out["main_stats"] = s
        return out
