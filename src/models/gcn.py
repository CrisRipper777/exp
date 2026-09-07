from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from .common import get_activation, make_norm


class Model(nn.Module):
    def __init__(self, cfg, data_info):
        super().__init__()
        input_dim = int(data_info["input_dim"])
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.num_layers)
        dropout = float(cfg.model.dropout)
        activation = str(cfg.model.get("activation", "relu"))
        norm_name = cfg.model.get("norm", "batchnorm")

        dims = [input_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1], normalize=False, add_self_loops=False) for i in range(num_layers)]
        )
        self.norms = nn.ModuleList([make_norm(norm_name, hidden_dim) for _ in range(max(num_layers - 1, 0))])
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def _normalize_edges(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return gcn_norm(
            edge_index,
            edge_weight=None,
            num_nodes=num_nodes,
            improved=False,
            add_self_loops=True,
            flow="source_to_target",
            dtype=dtype,
        )

    def forward(self, x, edge_index):
        norm_edge_index, norm_edge_weight = self._normalize_edges(edge_index, int(x.size(0)), x.dtype)
        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, norm_edge_index, edge_weight=norm_edge_weight)
            if layer_idx != len(self.convs) - 1:
                x = self.norms[layer_idx](x)
                x = self.activation(x)
                x = self.dropout(x)
        aux_loss = x.new_tensor(0.0)
        return x, None, None, aux_loss, {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        h = x.cpu()
        edge_index = edge_index.cpu()
        num_nodes = int(h.size(0))
        input_nodes = torch.arange(num_nodes, dtype=torch.long)
        norm_edge_index, norm_edge_weight = self._normalize_edges(edge_index, num_nodes, h.dtype)

        for layer_idx, conv in enumerate(self.convs):
            data = Data(x=h, edge_index=norm_edge_index, edge_weight=norm_edge_weight)
            loader = NeighborLoader(
                data,
                input_nodes=input_nodes,
                num_neighbors=[-1],
                batch_size=batch_size,
                shuffle=False,
            )
            out = torch.empty((num_nodes, conv.out_channels), dtype=h.dtype, device="cpu")
            for batch in loader:
                batch = batch.to(device)
                z = conv(batch.x, batch.edge_index, edge_weight=batch.edge_weight)[: batch.batch_size]
                if layer_idx != len(self.convs) - 1:
                    z = self.norms[layer_idx](z)
                    z = self.activation(z)
                out[batch.n_id[: batch.batch_size].cpu()] = z.detach().cpu()
            h = out

        return h
