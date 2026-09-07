"""SCOPE-MAG v2: ownership-separated context followed by ORB.

This file is intentionally independent of the historical R3 implementation.
The public model switches are:

``graph_mode``
    ``local`` for OSC, ``local_global`` for OSC+FGR, or ``none`` for the
    pointwise control used by equivalence tests.
``reconciliation.mode``
    ``none`` (IND), ``orb`` (proposed bottleneck), or ``full`` (unrestricted
    mixing control).

The P0 factorizer, auxiliary losses, and final fusion are inherited unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_p0 import Model as P0Model
from .biaxis_scope_v2_components import (
    NUM_FACTORS,
    FactorGlobalRelay,
    OACCBlock,
    OwnershipReconciliationBottleneck,
    OwnershipSeparatedContext,
    chunked_factor_mean,
    chunked_factor_sym_norm,
)


class Model(P0Model):
    """Two-block SCOPE-MAG v2 model."""

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)
        scope = cfg.model.scope
        self.graph_mode = str(scope.get("graph_mode", "local"))
        if self.graph_mode not in ("none", "local", "local_global"):
            raise ValueError(f"scope.graph_mode must be none|local|local_global, got {self.graph_mode!r}")
        self.num_blocks = int(scope.get("num_blocks", 2))
        if self.num_blocks < 1:
            raise ValueError("scope.num_blocks must be >= 1")
        self.edge_chunk_size = max(int(scope.get("edge_chunk_size", 200000)), 1)
        self.local_aggregation = str(scope.get("local_aggregation", "mean"))
        if self.local_aggregation not in ("mean", "sym_norm"):
            raise ValueError(
                "scope.local_aggregation must be mean|sym_norm, "
                f"got {self.local_aggregation!r}"
            )
        self.use_source_projection = bool(scope.get("use_source_projection", False))
        self.post_context_norm = bool(scope.get("post_context_norm", False))
        self.anchor_mix = float(scope.get("anchor_mix", 0.1))
        self.osc_scale_init = float(scope.get("osc_scale_init", 0.1))
        self.orb_scale_init = float(scope.get("orb_scale_init", 0.1))
        self.osc_hidden_dim = int(scope.get("osc_hidden_dim", 256))
        self.factor_id_dim = int(scope.get("factor_id_dim", 16))
        activation = str(cfg.model.get("activation", "gelu"))
        norm = str(cfg.model.get("norm", "layernorm"))
        self.context_core = str(scope.get("context_core", "osc"))
        if self.context_core not in ("osc", "oacc"):
            raise ValueError(f"scope.context_core must be osc|oacc, got {self.context_core!r}")
        reconciliation = scope.get("reconciliation", {})
        self.reconciliation_mode = str(reconciliation.get("mode", "none"))

        if self.context_core == "osc":
            self.context_layers = nn.ModuleList([
                OwnershipSeparatedContext(
                    factor_dim=self.factor_dim,
                    hidden_dim=self.osc_hidden_dim,
                    factor_id_dim=self.factor_id_dim,
                    dropout=float(cfg.model.dropout),
                    activation=activation,
                    norm=norm,
                )
                for _ in range(self.num_blocks)
            ])
        else:
            self.oacc_layers = nn.ModuleList([
                OACCBlock(
                    factor_dim=self.factor_dim,
                    factor_id_dim=self.factor_id_dim,
                    adapter_rank=int(scope.get("oacc_adapter_rank", 16)),
                    ffn_hidden_dim=int(scope.get("oacc_ffn_hidden_dim", 256)),
                    beta_hidden_dim=int(scope.get("oacc_beta_hidden_dim", 64)),
                    beta_mode=str(scope.get("beta_mode", "adaptive")),
                    rho=float(scope.get("oacc_rho", 0.1)),
                    gamma=float(scope.get("oacc_gamma", 0.1)),
                    activation=activation,
                    norm=norm,
                )
                for _ in range(self.num_blocks)
            ])
        # Explicit phi_n,f from the R4 specification. Keeping it before edge
        # aggregation makes the source-side structural transform identifiable
        # instead of asking the target FFN to infer it from a raw mean.
        if self.use_source_projection:
            self.source_projectors = nn.ModuleList([
                nn.ModuleList([nn.Linear(self.factor_dim, self.factor_dim) for _ in range(NUM_FACTORS)])
                for _ in range(self.num_blocks)
            ])
        if self.context_core == "osc":
            self.osc_scales = nn.ParameterList([
                nn.Parameter(torch.full((NUM_FACTORS,), self.osc_scale_init))
                for _ in range(self.num_blocks)
            ])
            self.update_norms = nn.ModuleList([nn.LayerNorm(self.factor_dim) for _ in range(self.num_blocks)])
        if self.reconciliation_mode != "none":
            self.orb_scales = nn.ParameterList([
                nn.Parameter(torch.full((NUM_FACTORS,), self.orb_scale_init))
                for _ in range(self.num_blocks)
            ])
            self.reconcile_norms = nn.ModuleList([nn.LayerNorm(self.factor_dim) for _ in range(self.num_blocks)])

        # R5-0 MEAN-LN is an explicitly opt-in LayerNorm after the structural
        # ContextCore update.  Keeping the ModuleList absent when disabled
        # preserves the old parameter set and old forward behavior exactly.
        if self.post_context_norm:
            self.post_context_norms = nn.ModuleList([
                nn.LayerNorm(self.factor_dim) for _ in range(self.num_blocks)
            ])

        if self.graph_mode == "local_global":
            assignment_mode = str(scope.get("global_assignment_mode", "learned"))
            if assignment_mode not in FactorGlobalRelay.ASSIGNMENT_MODES:
                raise ValueError(
                    "scope.global_assignment_mode must be learned|uniform, "
                    f"got {assignment_mode!r}"
                )
            self.global_relays = nn.ModuleList([
                FactorGlobalRelay(
                    factor_dim=self.factor_dim,
                    num_slots=int(scope.get("global_num_slots", 4)),
                    activation=activation,
                    assignment_mode=assignment_mode,
                )
                for _ in range(self.num_blocks)
            ])
            self.global_scales = nn.ParameterList([
                nn.Parameter(torch.full((NUM_FACTORS,), float(scope.get("global_scale_init", 0.05))))
                for _ in range(self.num_blocks)
            ])

        self.reconciliation = nn.ModuleList([
            OwnershipReconciliationBottleneck(
                factor_dim=self.factor_dim,
                rank=int(reconciliation.get("rank", 32)),
                hidden_dim=int(reconciliation.get("hidden_dim", 128)),
                mode=self.reconciliation_mode,
                activation=activation,
                norm=norm,
                use_graph_condition=bool(reconciliation.get("use_graph_condition", True)),
            )
            for _ in range(self.num_blocks)
        ])

        # R4 uses full-graph training for all variants so the same context
        # core is evaluated on exactly the same nodes/edges in every control.
        self.requires_full_graph_training = bool(cfg.model.get("full_graph_training", True))

    def _context(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        num_nodes = int(h.size(0))
        if self.graph_mode == "none":
            return h.new_zeros(h.shape), {}
        messages = []
        context_diag: dict[str, torch.Tensor] = {}
        for factor in range(NUM_FACTORS):
            source = h[:, factor]
            if self.use_source_projection:
                source = self.source_projectors[layer][factor](source)
            if self.local_aggregation == "mean":
                local = chunked_factor_mean(
                    edge_index=edge_index,
                    values=source,
                    num_nodes=num_nodes,
                    edge_chunk_size=self.edge_chunk_size,
                )
            else:
                local = chunked_factor_sym_norm(
                    edge_index=edge_index,
                    values=source,
                    num_nodes=num_nodes,
                    edge_chunk_size=self.edge_chunk_size,
                )
            message = local
            factor_name = ("c", "pt", "pv")[factor]
            context_diag[
                f"scope_l{layer + 1}_{factor_name}_local_message_ratio"
            ] = self._safe_ratio(local, h[:, factor]).detach()
            if self.graph_mode == "local_global":
                global_message, relay_stats = self.global_relays[layer](
                    h[:, factor], factor, return_stats=True
                )
                message = message + self.global_scales[layer][factor] * global_message
                context_diag.update({
                    f"scope_l{layer + 1}_{factor_name}_global_local_ratio": self._safe_ratio(
                        global_message, local
                    ).detach(),
                    f"scope_l{layer + 1}_{factor_name}_global_scale": self.global_scales[
                        layer
                    ][factor].detach(),
                    f"scope_l{layer + 1}_{factor_name}_slot_entropy": relay_stats[
                        "slot_entropy"
                    ],
                    f"scope_l{layer + 1}_{factor_name}_effective_active_slots": relay_stats[
                        "effective_active_slots"
                    ],
                    f"scope_l{layer + 1}_{factor_name}_active_slots": relay_stats[
                        "active_slots"
                    ],
                })
                for slot_idx, usage in enumerate(relay_stats["slot_usage"]):
                    context_diag[
                        f"scope_l{layer + 1}_{factor_name}_slot_usage_{slot_idx}"
                    ] = usage
            messages.append(message)
        return torch.stack(messages, dim=1), context_diag

    @staticmethod
    def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        return num.norm(dim=-1).mean() / den.norm(dim=-1).mean().clamp_min(1e-8)

    def forward(self, x: torch.Tensor, edge_index=None):
        x_t, x_v = self._split_modalities(x)
        factors = self.factorizer(x_t, x_v)
        h0 = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        h = h0

        if self.training:
            aux_loss, aux_info = self._compute_aux(factors)
        else:
            aux_loss = h.new_tensor(0.0)
            aux_info = {}

        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        else:
            edge_index = edge_index.to(x.device)

        # Exact pointwise P0 fallback: useful for unit tests and parameter
        # matching controls, and avoids accidentally treating no-edge as a
        # graph operation.
        if self.graph_mode == "none" and self.reconciliation_mode == "none":
            z = self.fusion(torch.cat([h[:, 0], h[:, 1], h[:, 2]], dim=-1))
            return z, None, None, aux_loss, aux_info

        for layer in range(self.num_blocks):
            message, context_diag = self._context(h, edge_index, layer)
            if self.graph_mode == "none":
                h_struct = h
                core_stats = None
            elif self.context_core == "oacc":
                h_struct, core_stats = self.oacc_layers[layer](h, h0, message)
            else:
                anchor = (1.0 - self.anchor_mix) * h + self.anchor_mix * h0
                h_next = []
                for factor in range(NUM_FACTORS):
                    updated = anchor[:, factor]
                    updated = updated + self.osc_scales[layer][factor] * self.context_layers[layer](
                        h[:, factor], message[:, factor], factor
                    )
                    updated = self.update_norms[layer](updated)
                    h_next.append(updated)
                h_struct = torch.stack(h_next, dim=1)
                core_stats = {
                    "graph_update": message,
                    "payload": message,
                    "beta": message.new_ones((message.size(0), NUM_FACTORS, 1)),
                }

            if self.post_context_norm:
                h_struct = torch.stack([
                    self.post_context_norms[layer](h_struct[:, factor])
                    for factor in range(NUM_FACTORS)
                ], dim=1)

            # ORB is deliberately post-context: it reads the structurally
            # contextualized state, never the pre-context h in parallel.
            delta = self.reconciliation[layer](h_struct, message)
            h_next = []
            for factor in range(NUM_FACTORS):
                updated = h_struct[:, factor]
                if self.reconciliation_mode != "none":
                    updated = updated + self.orb_scales[layer][factor] * delta[:, factor]
                    updated = self.reconcile_norms[layer](updated)
                h_next.append(updated)
                if self.training:
                    factor_name = ("c", "pt", "pv")[factor]
                    graph_update = core_stats["graph_update"][:, factor]
                    aux_info[f"scope_l{layer + 1}_{factor_name}_osc_ratio"] = self._safe_ratio(
                        h_struct[:, factor] - h[:, factor], h[:, factor]
                    ).detach()
                    aux_info[f"scope_l{layer + 1}_{factor_name}_orb_ratio"] = self._safe_ratio(
                        delta[:, factor], h[:, factor]
                    ).detach()
                    aux_info[f"scope_l{layer + 1}_{factor_name}_graph_update_ratio"] = self._safe_ratio(
                        graph_update, h[:, factor]
                    ).detach()
                    if core_stats.get("beta") is not None:
                        beta_f = core_stats["beta"][:, factor]
                        aux_info[f"scope_l{layer + 1}_{factor_name}_beta_mean"] = beta_f.mean().detach()
                        aux_info[f"scope_l{layer + 1}_{factor_name}_beta_std"] = beta_f.std().detach()
                    aux_info[f"scope_l{layer + 1}_{factor_name}_anchor_drift"] = self._safe_ratio(
                        h_struct[:, factor] - h0[:, factor], h0[:, factor]
                    ).detach()
                    for key, value in context_diag.items():
                        if key.startswith(f"scope_l{layer + 1}_{factor_name}_"):
                            aux_info[key] = value
            h = torch.stack(h_next, dim=1)

        z = self.fusion(torch.cat([h[:, 0], h[:, 1], h[:, 2]], dim=-1))
        return z, None, None, aux_loss, aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        z, _, _, _, _ = self.forward(x, edge_index)
        return z.detach().cpu()
