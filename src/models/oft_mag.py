from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .biaxis_p0 import Model as P0Model
from .oft_components import DiagIDLayer, NUM_SLOTS, SLOT_NAMES

# Slot <-> factorizer key map. Permanently fixed: 0=C, 1=Pt, 2=Pv.
_FACTOR_KEYS = ("c", "p_t", "p_v")


def stack_factor_slots(factors: dict[str, Tensor]) -> Tensor:
    """H = stack([C, Pt, Pv], dim=1) -> [N, 3, F]. Slot order is fixed."""
    return torch.stack([factors[key] for key in _FACTOR_KEYS], dim=1)


class Model(P0Model):
    """OFT-MAG O1 — ``OFT-DIAG-ID``: clean diagonal graph propagation states.

    The P0 semantic factorizer is reused verbatim (``biaxis_p0.Model``):
    x = [x_t | x_v] -> C / P_t / P_v, then the three factors become explicit
    graph states ``H = stack([C, Pt, Pv])``, each propagated by one DIAG-ID
    block (see ``DiagIDLayer``: per-factor V_a -> incoming neighbor mean ->
    D_a -> LayerNorm residual). O1 allows NO cross-factor transition, no
    self-loops, no multi-scale and exactly ``num_layers=1``; the final
    representation is the P0 fusion over the post-propagation H1 slots.

    Framework interface (unchanged from P0):
        forward(x, edge_index) -> (z, None, None, aux_loss, aux_info)
        inference(x, edge_index, device, batch_size) -> z (CPU)

    Training aux losses are the exact P0 ones, computed on the raw factorizer
    outputs (pre-propagation factors). New aux_info diagnostics:
        oft_l1_{c,pt,pv}_diag_update_ratio / oft_l1_{c,pt,pv}_neighbor_norm.
    """

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)
        num_layers = int(cfg.model.get("num_layers", 1))
        if num_layers != 1:
            raise ValueError(
                f"O1 OFT-DIAG-ID supports num_layers=1 only, got {num_layers} "
                "(single-layer scaffold; no multi-scale)"
            )
        self.num_layers = num_layers
        self.diag_layers = nn.ModuleList(
            [DiagIDLayer(factor_dim=self.factor_dim) for _ in range(self.num_layers)]
        )
        self.requires_full_graph_training = bool(cfg.model.get("full_graph_training", True))

    # ------------------------------------------------------------------
    # OFT encoding
    # ------------------------------------------------------------------

    def _encode_oft(
        self, x: torch.Tensor, edge_index: torch.Tensor | None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """factorizer -> H0 -> 1 x DIAG-ID block -> H1 -> P0 fusion.

        Returns (factors, z, layer_stats). With an empty graph the stats are
        exactly zero (no messages -> delta = 0), which keeps the training
        diagnostics well-defined in every case."""
        x_t, x_v = self._split_modalities(x)
        factors = self.factorizer(x_t, x_v)
        H0 = stack_factor_slots(factors)  # [N, 3, F]
        H = H0
        stats: dict[str, torch.Tensor] = {}
        for layer in self.diag_layers:
            H, stats = layer.propagate(H, edge_index)
        H1 = H
        z = self.fusion(torch.cat([H1[:, a] for a in range(NUM_SLOTS)], dim=-1))
        return factors, z, stats

    def _oft_aux_info(self, stats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Map layer stats (indexed by fixed slot order) onto oft_l1_* keys."""
        info = {}
        for slot_idx, name in enumerate(SLOT_NAMES):
            info[f"oft_l1_{name}_diag_update_ratio"] = stats["diag_update_ratio"][slot_idx]
            info[f"oft_l1_{name}_neighbor_norm"] = stats["neighbor_norm"][slot_idx]
        return info

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None):
        if edge_index is not None:
            edge_index = edge_index.to(x.device)
        factors, z, layer_stats = self._encode_oft(x, edge_index)
        if self.training:
            aux_loss, aux_info = self._compute_aux(factors)  # exact P0 reuse
            aux_info.update(self._oft_aux_info(layer_stats))
        else:
            aux_loss = z.new_tensor(0.0)
            aux_info = {}
        return z, None, None, aux_loss, aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """Full-graph inference -> CPU z.

        Graph propagation couples all nodes (incoming neighbor means), so the
        P0-style per-node chunking is not applicable here: one exact full-graph
        forward is run (like DiP). ``batch_size`` is accepted for interface
        compatibility and unused.
        """
        del batch_size
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self(
            x.to(device),
            edge_index.to(device) if edge_index is not None else None,
        )
        return z.detach().cpu()
