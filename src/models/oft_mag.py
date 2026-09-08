from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .biaxis_p0 import Model as P0Model
from .oft_components import (
    CROSS_PAIR_KEYS,
    ConditionalCrossLayer,
    DiagIDLayer,
    NUM_SLOTS,
    SLOT_NAMES,
    StaticCrossLayer,
    post_transition_stats,
)

# Slot <-> factorizer key map. Permanently fixed: 0=C, 1=Pt, 2=Pv.
_FACTOR_KEYS = ("c", "p_t", "p_v")

# O1 / O1.5 / O2-A / O2-B1 variants (config ``model.variant``):
#   diag_id               — O1 OFT-DIAG-ID: real graph propagation through V/D/LN.
#   diag_nograph          — O1.5 strict matched no-topology control: the graph
#                           input is routed away so the neighbor response is
#                           exactly zero (see Model docstring). Same params /
#                           init / fusion / losses.
#   static_cross_dup      — O2-A capacity-matched control: same DIAG backbone plus
#                           a static cross branch that reads the TARGET's own
#                           neighbor state N^b through every off-diagonal map.
#   static_cross          — O2-A treatment: the cross branch reads the real
#                           SOURCE ownership states N^a (a != b).
#   conditional_cross_dup — O2-B1 capacity-matched control: identical O2-A cross
#                           branch + a target-conditioned Null-vs-Transfer router,
#                           but every payload still reads the target's own N^b.
#   conditional_cross     — O2-B1 treatment: same router, real source payload N^a.
VARIANTS = (
    "diag_id",
    "diag_nograph",
    "static_cross_dup",
    "static_cross",
    "conditional_cross_dup",
    "conditional_cross",
)
_STATIC_CROSS_VARIANTS = ("static_cross_dup", "static_cross")
_CONDITIONAL_CROSS_VARIANTS = ("conditional_cross_dup", "conditional_cross")
_CROSS_VARIANTS = _STATIC_CROSS_VARIANTS + _CONDITIONAL_CROSS_VARIANTS


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

    O1.5 ``diag_nograph`` (variant switch): strict matched no-topology control
    for the O1.5 graph-causality cleanup. It is diag_id with every parameter,
    structure, initialization, fusion and P0 loss unchanged, and the single
    allowed difference that the graph neighbor response is forced to zero:
    the propagation is fed ``edge_index=None``, so ``incoming_mean`` returns
    exact zeros, ``delta_a = D_a(0) = 0`` exactly (D_a has bias=False) and
    ``H_next_a = LayerNorm(H_a + 0)``. V_a / D_a / LN stay instantiated and in
    the optimizer parameter set (their weights never move because the graph
    channel is dead) — parameter count is exactly diag_id's.

    O2-A ``static_cross`` / ``static_cross_dup`` (variant switch): the DIAG-ID
    backbone above is untouched and an additive static cross branch is added
    (``StaticCrossLayer``): six bias-free zero-initialized maps C_{a->b} for the
    off-diagonal ownership pairs, averaged over the two sources of each target
    and added inside the same per-slot LayerNorm residual. STATIC-CROSS feeds
    the real other-ownership neighborhood state N^a; the capacity-matched
    control STATIC-CROSS-DUP feeds the target's own N^b through the identical
    modules. Zero init makes step 0 exactly DIAG-ID for both.

    O2-B1 ``conditional_cross`` / ``conditional_cross_dup`` (variant switch):
    the O2-A construction is reused and the unconditional cross application is
    replaced by a shared target-conditioned Null-vs-Transfer router
    (``ConditionalCrossLayer``): one embedding table for factor identity, one
    shared MLP reading ``[H^b | N^b | e_a | e_b]``, ``p_transfer`` multiplying
    the same six zero-init maps, ``p_null`` realizing the Null operator. The
    router NEVER reads source ownership content (see the layer docstring), so
    the two variants differ only in the payload's ownership information.

    Framework interface (unchanged from P0):
        forward(x, edge_index) -> (z, None, None, aux_loss, aux_info)
        inference(x, edge_index, device, batch_size) -> z (CPU)

    Training aux losses are the exact P0 ones, computed on the raw factorizer
    outputs (pre-propagation factors). New aux_info diagnostics:
        oft_l1_{c,pt,pv}_diag_update_ratio / oft_l1_{c,pt,pv}_neighbor_norm,
    plus, for the O2-A cross variants only,
        oft_l1_{c,pt,pv}_cross_update_ratio / _cross_to_diag_ratio,
        oft_l1_{a_to_b}_cross_pair_ratio (6 pairs),
        oft_l1_post_cos_{a}_{b} / pre_cos_{a}_{b}, oft_l1_{c,pt,pv}_post_norm,
        oft_l1_{c,pt,pv}_state_drift.
    and, for the O2-B1 conditional variants only,
        oft_l1_{a_to_b}_transfer_mean / _transfer_std / _router_entropy,
        oft_l1_{a_to_b}_effective_cross_pair_ratio (6 pairs each).
    """

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)
        variant = str(cfg.model.get("variant", "diag_id")).strip().lower()
        if variant not in VARIANTS:
            raise ValueError(
                f"OFT model.variant must be one of {VARIANTS}, got {variant!r}"
            )
        self.variant = variant
        num_layers = int(cfg.model.get("num_layers", 1))
        if num_layers != 1:
            raise ValueError(
                f"OFT variants {VARIANTS} support num_layers=1 only, got {num_layers} "
                "(single-layer scaffold; no multi-scale)"
            )
        self.num_layers = num_layers
        if variant in _CONDITIONAL_CROSS_VARIANTS:
            dup = variant == "conditional_cross_dup"
            self.diag_layers = nn.ModuleList(
                [
                    ConditionalCrossLayer(factor_dim=self.factor_dim, dup=dup)
                    for _ in range(self.num_layers)
                ]
            )
        elif variant in _STATIC_CROSS_VARIANTS:
            dup = variant == "static_cross_dup"
            self.diag_layers = nn.ModuleList(
                [
                    StaticCrossLayer(factor_dim=self.factor_dim, dup=dup)
                    for _ in range(self.num_layers)
                ]
            )
        else:
            # O1 / O1.5 path: unchanged DiagIDLayer construction.
            self.diag_layers = nn.ModuleList(
                [DiagIDLayer(factor_dim=self.factor_dim) for _ in range(self.num_layers)]
            )
        self.requires_full_graph_training = bool(cfg.model.get("full_graph_training", True))

    # ------------------------------------------------------------------
    # OFT encoding
    # ------------------------------------------------------------------

    def _encode_oft(
        self, x: torch.Tensor, edge_index: torch.Tensor | None
    ) -> tuple[
        dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        """factorizer -> H0 -> 1 x graph block -> H1 -> P0 fusion.

        Returns (factors, z, layer_stats, H0, H1). With an empty graph the
        stats are exactly zero (no messages -> delta = 0), which keeps the
        training diagnostics well-defined in every case.

        O1.5 ``diag_nograph``: the graph input is routed away here (single
        choke point, shared by forward/inference), so the layer executes its
        empty-graph path: N = 0, delta = D(0) = 0 exactly, H1 = LN(H0).
        ``diag_id`` and all cross variants propagate on the real graph; the
        O2-A / O2-B1 variants only differ inside their cross layer."""
        x_t, x_v = self._split_modalities(x)
        factors = self.factorizer(x_t, x_v)
        H0 = stack_factor_slots(factors)  # [N, 3, F]
        prop_edge = None if self.variant == "diag_nograph" else edge_index
        H = H0
        stats: dict[str, torch.Tensor] = {}
        for layer in self.diag_layers:
            H, stats = layer.propagate(H, prop_edge)
        H1 = H
        z = self.fusion(torch.cat([H1[:, a] for a in range(NUM_SLOTS)], dim=-1))
        return factors, z, stats, H0, H1

    def _oft_aux_info(
        self,
        stats: dict[str, torch.Tensor],
        H0: torch.Tensor | None = None,
        H1: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Map layer stats (indexed by fixed slot order) onto oft_l1_* keys.

        O2-A cross variants additionally emit the cross-branch update ratios,
        the six per-pair contributions and the post-transition ownership
        diagnostics. O2-B1 conditional variants add, per pair, the router's
        transfer mean/std/entropy and the effective (routed) pair magnitude.
        diag_id / diag_nograph keep exactly their O1/O1.5 key set and the O2-A
        variants keep exactly their O2-A key set (numerical semantics and log
        surface unchanged)."""
        info = {}
        for slot_idx, name in enumerate(SLOT_NAMES):
            info[f"oft_l1_{name}_diag_update_ratio"] = stats["diag_update_ratio"][slot_idx]
            info[f"oft_l1_{name}_neighbor_norm"] = stats["neighbor_norm"][slot_idx]
        if self.variant not in _CROSS_VARIANTS:
            return info
        for slot_idx, name in enumerate(SLOT_NAMES):
            info[f"oft_l1_{name}_cross_update_ratio"] = stats["cross_update_ratio"][slot_idx]
            info[f"oft_l1_{name}_cross_to_diag_ratio"] = stats["cross_to_diag_ratio"][slot_idx]
        for key in CROSS_PAIR_KEYS:
            info[f"oft_l1_{key}_cross_pair_ratio"] = stats[f"cross_pair_ratio_{key}"]
        if self.variant in _CONDITIONAL_CROSS_VARIANTS:
            for key in CROSS_PAIR_KEYS:
                info[f"oft_l1_{key}_transfer_mean"] = stats[f"transfer_mean_{key}"]
                info[f"oft_l1_{key}_transfer_std"] = stats[f"transfer_std_{key}"]
                info[f"oft_l1_{key}_router_entropy"] = stats[f"router_entropy_{key}"]
                info[f"oft_l1_{key}_effective_cross_pair_ratio"] = stats[
                    f"effective_cross_pair_ratio_{key}"
                ]
        if H0 is not None and H1 is not None:
            for key, value in post_transition_stats(H0, H1).items():
                info[f"oft_l1_{key}"] = value
        return info

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None):
        if edge_index is not None:
            edge_index = edge_index.to(x.device)
        factors, z, layer_stats, H0, H1 = self._encode_oft(x, edge_index)
        if self.training:
            aux_loss, aux_info = self._compute_aux(factors)  # exact P0 reuse
            aux_info.update(self._oft_aux_info(layer_stats, H0, H1))
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

    @torch.no_grad()
    def encode_states(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Offline O2-A diagnostics: full-graph (H0, H1, layer stats) on CPU.

        Exact same code path as ``forward`` (including the diag_nograph routing
        and the cross branch), just without the P0 aux losses. Used by
        ``scripts/oft_o2a_diagnostics.py`` on a best-Val-Acc checkpoint.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        _, _, stats, H0, H1 = self._encode_oft(
            x.to(device),
            edge_index.to(device) if edge_index is not None else None,
        )
        return H0.detach().cpu(), H1.detach().cpu(), {
            key: value.detach().cpu() for key, value in stats.items()
        }
