from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, coalesce, degree, to_undirected


class Model(nn.Module):
    """DGF filtering-core supervised adaptation (reference: OpenMAG DGF.py).

    Keeps the dual graph filtering core: L2-normalized per-modality
    projections, fixed mean fusion ZM = (ZM1 + ZM2) / 2, feature-domain
    symmetric-softmax shift operators, and the node-domain truncated Neumann
    series sum_{t} (alpha/(alpha+1) * NAM)^t ZM.

    OpenMAG's clustering/contrastive losses (cross-modal NCE, random-walk
    graph contrastive, k-means community loss) are intentionally NOT
    included — this adaptation uses the unified supervised CE protocol
    (RPTA baseline_adaptations.md: DGF filtering-core supervised adaptation).
    """

    def __init__(self, cfg, data_info):
        super().__init__()
        text_dim = int(data_info["text_dim"])
        visual_dim = int(data_info["visual_dim"])
        if text_dim <= 0 or visual_dim <= 0:
            raise ValueError(
                f"dgf requires both modalities, got text_dim={text_dim}, visual_dim={visual_dim}"
            )
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = int(cfg.model.hidden_dim)
        self.alpha = float(cfg.model.get("alpha", 1.0))
        self.beta = float(cfg.model.get("beta", 1.0))
        self.num_layers = int(cfg.model.get("num_layers", 10))
        # Official core: no activation, no dropout, no norm layers — L2 only.
        self.linear1 = nn.Linear(text_dim, self.hidden_dim)  # text
        self.linear2 = nn.Linear(visual_dim, self.hidden_dim)  # image
        self.out_dim = self.hidden_dim
        self._nam_cache: tuple | None = None

    # ------------------------------------------------------------------
    # Feature-domain shift operator (OpenMAG DGFCore.symmetric_softmax)
    # ------------------------------------------------------------------

    def _symmetric_softmax(self, zm: torch.Tensor) -> torch.Tensor:
        n, _d = zm.shape
        scale = 1.0 / math.sqrt(n)
        similarity = zm.t() @ zm * scale
        exp_sim = torch.exp(similarity - similarity.max())  # numerical stability
        row_sum = torch.sqrt(exp_sim.sum(dim=1, keepdim=True) + 1e-10)
        col_sum = torch.sqrt(exp_sim.sum(dim=0, keepdim=True) + 1e-10)
        return exp_sim / (row_sum @ col_sum)

    # ------------------------------------------------------------------
    # Node-domain normalized adjacency (self-loops added: OpenMAG NC
    # loader uses self_loop=True; RPTA protocol allows standard conv
    # self-loops).
    # ------------------------------------------------------------------

    def _build_nam(
        self, edge_index: torch.Tensor, num_nodes: int, device: torch.device
    ) -> torch.Tensor:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)[0]
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float32)
        inv_sqrt = (deg + 1e-10).pow(-0.5)
        values = inv_sqrt[row] * inv_sqrt[col]
        return torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).to(device)

    def _get_nam(
        self, edge_index: torch.Tensor, num_nodes: int, device: torch.device
    ) -> torch.Tensor:
        # Identity-keyed cache: full-graph training reuses the same tensors
        # every epoch; LinkNeighborLoader subgraphs get fresh tensors per
        # batch and are rebuilt (their edge_index genuinely differs).
        cache = self._nam_cache
        if cache is not None and cache[0] is edge_index and cache[1] == num_nodes:
            return cache[2]
        nam = self._build_nam(edge_index, num_nodes, device)
        self._nam_cache = (edge_index, num_nodes, nam)
        return nam

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        device = x.device
        num_nodes = int(x.size(0))
        x_t = x[:, : self.text_dim]
        x_v = x[:, self.text_dim : self.text_dim + self.visual_dim]

        zm1 = F.normalize(self.linear1(x_t), p=2, dim=1)
        zm2 = F.normalize(self.linear2(x_v), p=2, dim=1)
        zm = 0.5 * (zm1 + zm2)

        with torch.no_grad():
            sm = 0.5 * (self._symmetric_softmax(zm1) + self._symmetric_softmax(zm2))
            beta_term = self.beta / (self.beta + 1.0)
            d = zm.size(1)
            sum_beta = torch.zeros(d, d, device=device)
            current_power = torch.eye(d, device=device)
            for _ in range(self.num_layers):
                sum_beta = sum_beta + current_power
                current_power = current_power @ (beta_term * sm)

        nam = self._get_nam(edge_index, num_nodes, device)
        alpha_term = self.alpha / (self.alpha + 1.0)
        part_alpha = zm.clone()
        current = zm.clone()
        for _ in range(self.num_layers):
            current = alpha_term * torch.sparse.mm(nam, current)
            part_alpha = part_alpha + current

        h = (1.0 / ((self.alpha + 1.0) * (self.beta + 1.0))) * (part_alpha @ sum_beta)
        return F.normalize(h, p=2, dim=1)

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None):
        if edge_index is None:
            raise ValueError("DGF requires edge_index")
        z = self._encode(x, edge_index)
        return z, None, None, z.new_tensor(0.0), {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """Chunked layerwise-style eval: per-node projections on device, then
        full-graph filtering on CPU (identical math to a full-graph forward)."""
        self.eval()
        if edge_index is None:
            raise ValueError("DGF requires edge_index")
        if device is None:
            device = next(self.parameters()).device
        num_nodes = int(x.size(0))
        zm1 = torch.empty((num_nodes, self.hidden_dim), dtype=x.dtype)
        zm2 = torch.empty_like(zm1)
        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)
            chunk = x[start:end].to(device)
            zm1[start:end] = F.normalize(self.linear1(chunk[:, : self.text_dim]), p=2, dim=1).cpu()
            zm2[start:end] = F.normalize(
                self.linear2(chunk[:, self.text_dim : self.text_dim + self.visual_dim]), p=2, dim=1
            ).cpu()
        zm = 0.5 * (zm1 + zm2)
        sm = 0.5 * (self._symmetric_softmax(zm1) + self._symmetric_softmax(zm2))
        beta_term = self.beta / (self.beta + 1.0)
        d = zm.size(1)
        sum_beta = torch.zeros(d, d)
        current_power = torch.eye(d)
        for _ in range(self.num_layers):
            sum_beta = sum_beta + current_power
            current_power = current_power @ (beta_term * sm)
        nam = self._build_nam(edge_index.cpu(), num_nodes, torch.device("cpu"))
        alpha_term = self.alpha / (self.alpha + 1.0)
        part_alpha = zm.clone()
        current = zm.clone()
        for _ in range(self.num_layers):
            current = alpha_term * torch.sparse.mm(nam, current)
            part_alpha = part_alpha + current
        h = (1.0 / ((self.alpha + 1.0) * (self.beta + 1.0))) * (part_alpha @ sum_beta)
        return F.normalize(h, p=2, dim=1)
