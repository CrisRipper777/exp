from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# O1 slot order is permanently fixed: 0=C, 1=Pt, 2=Pv.
NUM_SLOTS = 3
SLOT_NAMES = ("c", "pt", "pv")


def incoming_mean(X: Tensor, edge_index: Tensor | None, num_nodes: int) -> Tensor:
    """Per-node incoming-neighbor mean: messages flow source -> target, i.e.

        N_i = (1 / |{j : (j, i) in E}|) * sum_j X_j

    with ``edge_index[0] = source`` and ``edge_index[1] = target``. No
    self-loops are ever added, and no GCN-style normalization is applied.
    Nodes with no incoming edges get EXACTLY 0.0 (``0 / clamp_min(1) = 0``),
    so an isolated node's graph delta is strictly zero by construction.

    The aggregation uses one ``index_add_`` over gathered source rows
    (``X[src]``, shape ``[E, S, F]``); it never materializes an
    ``[E, S, S, ...]`` edge-pair tensor.

    Shapes: X ``[N, S, F]``, edge_index ``[2, E]`` -> ``[N, S, F]``.
    """
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros_like(X)
    src, dst = edge_index
    msg = X[src]  # [E, S, F] gathered source rows
    out = torch.zeros_like(X)
    out.index_add_(0, dst, msg)
    deg = torch.zeros(num_nodes, dtype=X.dtype, device=X.device)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=X.dtype))
    return out / deg.clamp_min(1.0).view(num_nodes, 1, 1)


class DiagIDLayer(nn.Module):
    """O1 OFT-DIAG-ID graph propagation layer (functional operator = Identity).

    Applied per factor slot ``a`` independently (diagonal, factor-preserving):

        X_i^a       = V_a H_i^a                 # Linear 128->128, no bias
        N_i^a       = incoming neighbor mean    # mean over in-neighbors j -> i
        delta_i^a   = D_a N_i^a                 # Linear 128->128, no bias
        H_i^{a,+}   = LayerNorm(H_i^a + delta_i^a)

    There is no self-loop (delta of an isolated node is exactly 0; the ego is
    kept through the residual), no GCN normalization, and no cross-factor
    transfer anywhere inside the layer: slot ``a`` only ever reads H[:, a].
    """

    def __init__(self, factor_dim: int = 128, eps: float = 1e-8) -> None:
        super().__init__()
        self.factor_dim = int(factor_dim)
        self.num_slots = NUM_SLOTS
        self.eps = float(eps)
        # One independent 128->128 bias-free map per slot.
        self.V = nn.ModuleList(
            [nn.Linear(self.factor_dim, self.factor_dim, bias=False) for _ in range(self.num_slots)]
        )
        self.D = nn.ModuleList(
            [nn.Linear(self.factor_dim, self.factor_dim, bias=False) for _ in range(self.num_slots)]
        )
        # Per-slot LayerNorm keeps the LN statistics factor-diagonal.
        self.norm = nn.ModuleList(
            [nn.LayerNorm(self.factor_dim) for _ in range(self.num_slots)]
        )

    def propagate(
        self, H: Tensor, edge_index: Tensor | None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """H ``[N, S, F]`` -> (H_next ``[N, S, F]``, stats).

        stats holds one detached scalar per slot:
            diag_update_ratio[a] = mean_i ||delta_i^a|| / (||H_i^a|| + eps)
            neighbor_norm[a]     = mean_i ||N_i^a||
        """
        num_nodes = int(H.size(0))
        X = torch.stack(
            [self.V[a](H[:, a]) for a in range(self.num_slots)], dim=1
        )  # [N, S, F]
        N = incoming_mean(X, edge_index, num_nodes)  # [N, S, F]
        delta = torch.stack(
            [self.D[a](N[:, a]) for a in range(self.num_slots)], dim=1
        )  # [N, S, F]
        H_next = torch.stack(
            [self.norm[a](H[:, a] + delta[:, a]) for a in range(self.num_slots)], dim=1
        )  # [N, S, F]

        with torch.no_grad():
            h_norm = H.norm(dim=-1)  # [N, S]
            d_norm = delta.norm(dim=-1)
            n_norm = N.norm(dim=-1)
            update_ratio = (d_norm / (h_norm + self.eps)).mean(dim=0)  # [S]
            neighbor_norm = n_norm.mean(dim=0)  # [S]
        stats = {
            "diag_update_ratio": update_ratio.detach(),
            "neighbor_norm": neighbor_norm.detach(),
        }
        return H_next, stats

    def forward(self, H: Tensor, edge_index: Tensor | None) -> Tensor:
        return self.propagate(H, edge_index)[0]
