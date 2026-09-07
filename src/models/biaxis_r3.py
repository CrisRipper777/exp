"""R3 Bi-Axis model: Ownership-Structured Semantic Transition Network
(docs/R3_Ownership_Structured_Transition_阶段推进计划.md, §2-§11).

    x_t, x_v -> P0 factorizer -> C / Pt / Pv            (Stage I, unchanged)
    H^(0) = [C; Pt; Pv]     ownership state tensor [N, 3, d]
    H^(l+1) = OwnershipTransitionLayer(H^(l), A)        (Stage II, L layers)
    Hbar_b = M_b([H_b^(0) | H_b^(1) | ...])             (multi-scale, optional)
    z = MLP_f([Cbar | Ptbar | Pvbar])                   (Stage III = P0 fusion)

R3 discipline:
    - P0 factorizer / aux objective (L_common/L_orth/L_recon) / final fusion
      are reused UNCHANGED (plan §2.1/§10); biaxis_p0.py is never modified.
    - every R3 variant (V0-V6) is produced by the ONE code path + config
      switches (plan §16.1); no per-variant model files.
    - off-diagonal functional computation happens strictly BEFORE neighbor
      aggregation (plan §1.1 C); aggregation is plain mean (plan §5).
    - no neighbor attention / MoE / pseudo nodes / hop router / Transformer
      fusion / exposure gate (plan §1.2/§12).
    - layer/offdiag scales are never zero-initialized in real configs
      (plan §16.6); zero init is reserved for exact-identity test mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_p0 import Model as P0Model
from .biaxis_r3_components import NUM_FACTORS, OwnershipTransitionLayer


class Model(P0Model):
    """R3 Ownership-Structured Semantic Transition Network.

    Inherits the P0 factorizer / recon heads / aux losses / final fusion;
    replaces the P0 readout with the ownership-state transition stack."""

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)
        t = cfg.model.transition

        self.transition_mode = str(t.transition_mode)
        self.cross_factor = bool(t.cross_factor)
        self.use_dual_space = bool(t.use_dual_space)
        self.use_same_node_context = bool(t.use_same_node_context)
        self.preserve_source_channels = bool(t.preserve_source_channels)
        self.multi_scale = str(t.multi_scale)
        assert self.multi_scale in ("last", "concat"), (
            f"transition.multi_scale must be last|concat, got {self.multi_scale!r}"
        )
        self.num_transition_layers = int(t.num_transition_layers)
        assert self.num_transition_layers >= 1
        self.log_transition_stats = bool(t.get("log_transition_stats", True))
        self.log_basis_stats = bool(t.get("log_basis_stats", True))
        self.log_grad_stats = bool(t.get("log_grad_stats", False))
        self.memory_checkpoint = bool(t.get("memory_checkpoint", True))
        if bool(t.get("use_exposure", False)):
            raise NotImplementedError(
                "R3-v1 does not implement the exposure gate (plan §12); "
                "transition.use_exposure must stay false"
            )
        self.edge_chunk_size = int(t.get("edge_chunk_size", 200000))

        self.transition_layers = nn.ModuleList(
            [
                OwnershipTransitionLayer(
                    factor_dim=self.factor_dim,
                    relation_dim=int(t.relation_dim),
                    factor_id_dim=int(t.factor_id_dim),
                    context_dim=int(t.context_dim),
                    transition_mode=self.transition_mode,
                    cross_factor=self.cross_factor,
                    use_dual_space=self.use_dual_space,
                    use_same_node_context=self.use_same_node_context,
                    preserve_source_channels=self.preserve_source_channels,
                    num_bases=int(t.num_bases),
                    basis_rank=int(t.basis_rank),
                    router_hidden_dim=int(t.router_hidden_dim),
                    offdiag_init_scale=float(t.offdiag_init_scale),
                    layer_scale_init=float(t.layer_scale_init),
                    edge_chunk_size=self.edge_chunk_size,
                    dropout=float(cfg.model.dropout),
                    activation=str(cfg.model.get("activation", "gelu")),
                    norm=str(cfg.model.get("norm", "layernorm")),
                    memory_checkpoint=self.memory_checkpoint,
                )
                for _ in range(self.num_transition_layers)
            ]
        )

        # multi-scale state retention (plan §9): per-factor M_b over the
        # (L+1) retained states, concat mode only
        if self.multi_scale == "concat":
            self.ms_proj = nn.ModuleList(
                [
                    nn.Linear((self.num_transition_layers + 1) * self.factor_dim, self.factor_dim)
                    for _ in range(NUM_FACTORS)
                ]
            )

        self.requires_full_graph_training = bool(cfg.model.get("full_graph_training", True))

        # gradient audit (plan §18.6, debug flag only): per-module grad norms
        # recorded via backward hooks on the FIRST transition layer + shared
        # modules, surfaced in aux_info as r3_grad_<name>.
        self._grad_sq: dict[str, float] = {}
        self._grad_handles: list = []
        if self.log_grad_stats:
            first = self.transition_layers[0]
            named_modules = [
                ("factorizer", self.factorizer),
                ("diag", first.diag),
                ("router", getattr(first, "router", None)),
                ("basis", getattr(first, "basis_down", None)),
                ("basis_up", getattr(first, "basis_up", None)),
                ("target_update", first.update),
                ("fusion", self.fusion),
            ]
            for name, module in named_modules:
                if module is None:
                    continue
                for p in module.parameters():
                    self._grad_handles.append(p.register_hook(self._make_grad_hook(name)))

    def _make_grad_hook(self, name: str):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is not None:
                self._grad_sq[name] = self._grad_sq.get(name, 0.0) + float(
                    (grad.detach() ** 2).sum().item()
                )
            return grad

        return hook

    def _collect_grad_stats(self, aux_info: dict[str, float]) -> None:
        """Surface the PREVIOUS step's per-module grad norms (hooks fire
        during backward, i.e. after the forward that builds aux_info)."""
        for name, sq in self._grad_sq.items():
            aux_info[f"r3_grad_{name}"] = sq ** 0.5
        self._grad_sq.clear()

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None):
        x_t, x_v = self._split_modalities(x)
        factors = self.factorizer(x_t, x_v)
        # ownership state tensor (plan §2.2): [N, 3, d], order C / Pt / Pv
        H = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)

        if self.training:
            aux_loss, aux_info = self._compute_aux(factors)
        else:
            aux_loss = H.new_tensor(0.0)
            aux_info = {}

        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        num_nodes = int(x.size(0))

        # ---- Stage II: iterative semantic-state evolution (plan §8) -------
        states = [H]
        for layer_idx, layer in enumerate(self.transition_layers):
            H, layer_stats = layer(
                H,
                edge_index,
                num_nodes,
                collect_stats=self.training and (self.log_transition_stats or self.log_basis_stats),
            )
            states.append(H)
            if self.training:
                if self.log_transition_stats:
                    for key, value in layer_stats.get("transition", {}).items():
                        aux_info[f"r3_l{layer_idx + 1}_{key}"] = value
                if self.log_basis_stats:
                    for key, value in layer_stats.get("basis", {}).items():
                        aux_info[f"r3_l{layer_idx + 1}_{key}"] = value

        # ---- Stage III: multi-scale retention + final fusion (plan §9/§10)
        if self.multi_scale == "concat":
            cat = torch.cat(states, dim=-1)  # [N, 3, (L+1)*d]
            Hbar = torch.stack([self.ms_proj[b](cat[:, b]) for b in range(NUM_FACTORS)], dim=1)
        else:
            Hbar = states[-1]
        z = self.fusion(
            torch.cat([Hbar[:, 0], Hbar[:, 1], Hbar[:, 2]], dim=-1)
        )

        if self.training and self.log_grad_stats:
            self._collect_grad_stats(aux_info)
        return z, None, None, aux_loss, aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """R3 inference = ONE exact full-graph forward (same convention as
        P1, plan §16.5). The graph operator needs the complete neighborhood
        of every node, so per-node chunking is invalid; ``batch_size`` is
        accepted for API compatibility but unused. Eval-mode forward has no
        dropout, so inference == eval forward up to float noise."""
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
