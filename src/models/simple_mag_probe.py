"""R0 SimpleMAGProbe (FIRM-MAG diagnostic probe).

NOT a method to chase SOTA: it builds a diagnostic environment in which every
receiver node's representation is exactly decomposable into per-relation /
per-modality message contributions:

    z_i = h_self_i + sum_{j -> i} (delta_text_ij + delta_visual_ij)

so that later stages can run Null / Text / Visual / Text+Visual local
counterfactual interventions on single directed relations j -> i.

Hard constraints (R0 plan §3.3):
- exactly ONE layer of message passing;
- no attention / router / MoE / OT / Q-Former / cross-modal transformer /
  edge-conditioned nonlinear gating / complex MLP after aggregation;
- additive relation decomposition must hold to float precision.

edge_index convention (matches the loader): [2, E], row 0 = source j,
row 1 = target i; loader graphs are self-loop-free and bidirectional.
The probe therefore models the self contribution explicitly (h_self) and must
NOT add self-loops itself (existing protocol: dataset add_self_loops=false).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_activation, make_norm


def _normalize_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """alpha_ij = 1 / sqrt(d_i * d_j) over the (self-loop-free) graph.

    The loader already stores each undirected relation as a directed pair, so
    degrees counted on edge_index[1] equal the true neighbour counts."""
    deg = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device))
    deg = deg.clamp(min=1.0)
    src = edge_index[0]
    tgt = edge_index[1]
    alpha = 1.0 / torch.sqrt(deg[src] * deg[tgt])
    return edge_index, alpha


class Model(nn.Module):
    def __init__(self, cfg, data_info):
        super().__init__()
        text_dim = int(data_info["text_dim"]) if data_info.get("text_dim") else 0
        visual_dim = int(data_info["visual_dim"]) if data_info.get("visual_dim") else 0
        hidden_dim = int(cfg.model.hidden_dim)
        dropout = float(cfg.model.get("dropout", 0.0))
        activation = str(cfg.model.get("activation", "relu"))
        norm_name = cfg.model.get("norm", "none")
        num_proj_layers = int(cfg.model.get("num_proj_layers", 1))
        self.normalize_modalities = bool(cfg.model.get("normalize_modalities", True))

        # Full-graph training only: per-edge deltas and degree normalization
        # require the whole graph (also keeps intervention semantics exact).
        self.requires_full_graph_training = True

        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

        def _make_projector(in_dim: int) -> nn.Module:
            layers: list[nn.Module] = []
            dims = [in_dim] + [hidden_dim] * num_proj_layers
            for layer_idx in range(num_proj_layers):
                layers.append(nn.Linear(dims[layer_idx], dims[layer_idx + 1]))
                layers.append(make_norm(norm_name, dims[layer_idx + 1]))
                layers.append(self.activation)
                layers.append(self.dropout)
            return nn.Sequential(*layers)

        if text_dim > 0:
            self.proj_t = _make_projector(text_dim)
        if visual_dim > 0:
            self.proj_v = _make_projector(visual_dim)

        # h_self_i = W_self [h_text_i || h_visual_i]
        n_mod = int(text_dim > 0) + int(visual_dim > 0)
        if n_mod == 0:
            raise ValueError("simple_mag_probe needs at least one modality (data_info text_dim/visual_dim)")
        self.w_self = nn.Linear(n_mod * hidden_dim, hidden_dim)
        # delta^T_ij = alpha_ij * W_T_msg h_text_j ; likewise for visual.
        self.w_msg_t = nn.Linear(hidden_dim, hidden_dim)
        self.w_msg_v = nn.Linear(hidden_dim, hidden_dim)

        self.out_dim = hidden_dim
        self._text_dim = text_dim
        self._visual_dim = visual_dim

    # ------------------------------------------------------------------ #
    # shared computation: everything (forward / inference / diagnostics)  #
    # goes through _decompose so train/eval/diag outputs stay consistent. #
    # ------------------------------------------------------------------ #

    def _split_modalities(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x_t = x[:, : self._text_dim] if self._text_dim > 0 else None
        x_v = x[:, self._text_dim:] if self._visual_dim > 0 else None
        if self.normalize_modalities:
            if x_t is not None:
                x_t = F.normalize(x_t, p=2, dim=-1)
            if x_v is not None:
                x_v = F.normalize(x_v, p=2, dim=-1)
        return x_t, x_v

    def _decompose(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Return (h_text, h_visual, h_self, z_full, delta_text, delta_visual,
        alpha). All tensors except alpha live on the current device.

        For directed edge r = (j -> i): source = edge_index[0, r] = j,
        target = edge_index[1, r] = i, and
            delta_text[r]   = alpha[r] * W_T_msg h_text_j
            delta_visual[r] = alpha[r] * W_V_msg h_visual_j
        """
        num_nodes = int(x.size(0))
        edge_index = edge_index.to(x.device)
        x_t, x_v = self._split_modalities(x)

        h_text = self.proj_t(x_t) if x_t is not None else None
        h_visual = self.proj_v(x_v) if x_v is not None else None

        parts = [p for p in (h_text, h_visual) if p is not None]
        h_self = self.w_self(torch.cat(parts, dim=-1))

        _, alpha = _normalize_edges(edge_index, num_nodes, x.dtype)
        alpha = alpha.view(-1, 1)  # [E, 1]

        src = edge_index[0]
        if h_text is not None:
            delta_text = alpha * self.w_msg_t(h_text)[src]  # [E, H]
        else:
            delta_text = None
        if h_visual is not None:
            delta_visual = alpha * self.w_msg_v(h_visual)[src]
        else:
            delta_visual = None

        msg_parts = [d for d in (delta_text, delta_visual) if d is not None]
        total_delta = msg_parts[0]
        for extra in msg_parts[1:]:
            total_delta = total_delta + extra
        agg = torch.zeros_like(h_self)
        agg.index_add_(0, edge_index[1], total_delta)  # scatter over targets
        z_full = h_self + agg
        return h_text, h_visual, h_self, z_full, delta_text, delta_visual, alpha.view(-1)

    # ------------------------------------------------------------------ #
    # interface used by src/tasks (nc.py)                                 #
    # ------------------------------------------------------------------ #

    def forward(self, x, edge_index):
        _, _, _, z_full, _, _, _ = self._decompose(x, edge_index)
        aux_loss = z_full.new_tensor(0.0)
        return z_full, None, None, aux_loss, {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """Whole-graph CPU output (probe aggregation is one hop over the full
        transductive graph, so layerwise sampling would change the message
        set; graphs small enough for NC full mode are fine here)."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self.forward(x.to(device), edge_index.to(device))
        return z.detach().cpu()

    # ------------------------------------------------------------------ #
    # R0 diagnostics (plan §3.4): per-edge aligned messages               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forward_with_deltas(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        """Diagnostic forward. Returns:
          h_text / h_visual / h_self  : [N, H] node projections,
          z_full                      : [N, H] receiver representations,
          edge_delta_text[r]          : delta^T_{ij} for edge_index[:, r],
          edge_delta_visual[r]        : delta^V_{ij} for edge_index[:, r],
          edge_norm[r]                : alpha_ij = 1/sqrt(d_i d_j).
        """
        self.eval()
        h_text, h_visual, h_self, z_full, delta_text, delta_visual, alpha = self._decompose(x, edge_index)
        out = {
            "h_text": h_text,
            "h_visual": h_visual,
            "h_self": h_self,
            "z_full": z_full,
            "edge_delta_text": delta_text,
            "edge_delta_visual": delta_visual,
            "edge_norm": alpha,
        }
        return {k: v for k, v in out.items() if v is not None}
