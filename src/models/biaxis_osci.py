"""R5-1 Ownership-Conditioned Source Integration (OSCI)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_scope_v2 import Model as ScopeModel
from .biaxis_osci_components import OwnershipConditionedSourceComposer


class Model(ScopeModel):
    """SCOPE-MAG OACC with post-aggregation source-factor composition."""

    OSCI_MODES = ("c0-diag", "c1-dup", "c2-src-static", "c3-osci")
    SOURCE_ORDER = {
        0: (1, 2),  # C <- (Pt, Pv)
        1: (0, 2),  # Pt <- (C, Pv)
        2: (0, 1),  # Pv <- (C, Pt)
    }

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)
        osci = cfg.model.get("osci", {})
        self.osci_mode = str(osci.get("mode", "c0-diag")).lower()
        if self.osci_mode not in self.OSCI_MODES:
            raise ValueError(f"model.osci.mode must be one of {self.OSCI_MODES}, got {self.osci_mode!r}")
        if self.osci_mode != "c0-diag":
            cross_scale_init = float(osci.get("cross_scale_init", 0.05))
            if abs(cross_scale_init - 0.05) > 1e-12:
                raise ValueError("R5-1 fixes osci.cross_scale_init at 0.05; scale sweeps are not permitted")
            activation = str(cfg.model.get("activation", "gelu"))
            self.cross_composers = nn.ModuleList([
                OwnershipConditionedSourceComposer(
                    factor_dim=self.factor_dim,
                    hidden_dim=int(osci.get("composer_hidden_dim", 256)),
                    activation=activation,
                )
                for _ in range(self.num_blocks)
            ])
            self.cross_scales = nn.ParameterList([
                nn.Parameter(torch.full((3,), 0.05))
                for _ in range(self.num_blocks)
            ])

    def _context(self, h: torch.Tensor, edge_index: torch.Tensor, layer: int):
        local_messages, context_diag = super()._context(h, edge_index, layer)
        if self.osci_mode == "c0-diag":
            return local_messages, context_diag

        cross_messages = []
        intervention = getattr(self, "_osci_intervention", None) or {}
        intervention_mode = intervention.get("mode")
        source_permutations = intervention.get("source_permutations", {})
        target_permutations = intervention.get("target_permutations", {})
        for target_factor in range(3):
            source_a, source_b = self.SOURCE_ORDER[target_factor]
            u1 = local_messages[:, source_a]
            u2 = local_messages[:, source_b]
            if self.osci_mode == "c1-dup" or intervention_mode == "duplicate":
                u1 = local_messages[:, target_factor]
                u2 = local_messages[:, target_factor]
            elif intervention_mode == "shuffle_sources":
                u1 = u1[source_permutations[(layer, target_factor, 0)]]
                u2 = u2[source_permutations[(layer, target_factor, 1)]]

            include_target = self.osci_mode in ("c1-dup", "c3-osci")
            q = self.cross_composers[layer].make_query(
                h[:, target_factor], target_factor, include_target=include_target
            )
            if intervention_mode == "shuffle_target_q":
                q = q[target_permutations[(layer, target_factor)]]
            cross = self.cross_composers[layer](q, u1, u2)
            if intervention_mode == "zero_cross":
                cross = torch.zeros_like(cross)
            cross_messages.append(cross)

            factor_name = ("c", "pt", "pv")[target_factor]
            base = local_messages[:, target_factor]
            scale = self.cross_scales[layer][target_factor]
            context_diag[f"scope_l{layer + 1}_{factor_name}_cross_context_ratio"] = self._safe_ratio(
                cross, base
            ).detach()
            context_diag[f"scope_l{layer + 1}_{factor_name}_cross_scale"] = scale.detach()
            context_diag[f"scope_l{layer + 1}_{factor_name}_cross_update_ratio"] = self._safe_ratio(
                scale * cross, base
            ).detach()

        cross_messages = torch.stack(cross_messages, dim=1)
        return local_messages + torch.stack([
            self.cross_scales[layer][factor] * cross_messages[:, factor]
            for factor in range(3)
        ], dim=1), context_diag
