from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import degree, remove_self_loops, softmax

from .common import make_norm


class GraphGAT(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        bias: bool = True,
        aggr: str = "add",
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        uniform(self.in_channels, self.weight)
        if self.bias is not None:
            uniform(self.in_channels, self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size=None,
        node_degree: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x = torch.matmul(x, self.weight)
        if size is None:
            size = (x.size(0), x.size(0))
        if node_degree is not None and node_degree.dim() == 1:
            node_degree = node_degree.view(-1, 1)
        return self.propagate(edge_index, size=size, x=x, node_degree=node_degree)

    def message(
        self,
        edge_index_i: torch.Tensor,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        size_i: int,
        edge_index: torch.Tensor,
        size,
        node_degree_j: torch.Tensor | None,
    ) -> torch.Tensor:
        x_i = x_i.view(-1, self.out_channels)
        x_j = x_j.view(-1, self.out_channels)
        inner_product = torch.mul(x_i, F.leaky_relu(x_j)).sum(dim=-1)

        if node_degree_j is None:
            row, _ = edge_index
            node_degree_j = degree(row, size[0], dtype=x_i.dtype)[row]
        else:
            node_degree_j = node_degree_j.view(-1)
        deg_inv_sqrt = node_degree_j.clamp(min=1).pow(-0.5)
        gate_w = torch.sigmoid(torch.mul(deg_inv_sqrt, inner_product))

        attention_w = softmax(torch.mul(inner_product, gate_w), edge_index_i, num_nodes=size_i)
        return torch.mul(x_j, attention_w.view(-1, 1))

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out


class MgatBranch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        latent_dim: int,
        norm: str | None = "batchnorm",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.MLP = nn.Linear(in_dim, latent_dim)

        layer_input_dims = [latent_dim] + [hidden_dim] * max(num_layers - 1, 0)
        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.g_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_in_dim in layer_input_dims:
            conv = GraphGAT(layer_in_dim, layer_in_dim, normalize=True, aggr="add")
            linear = nn.Linear(layer_in_dim, hidden_dim)
            g_layer = nn.Linear(layer_in_dim, hidden_dim)
            nn.init.xavier_normal_(conv.weight)
            nn.init.xavier_normal_(linear.weight)
            nn.init.xavier_normal_(g_layer.weight)
            self.convs.append(conv)
            self.linears.append(linear)
            self.g_layers.append(g_layer)
            self.norms.append(make_norm(norm, hidden_dim))

    def _project_features(self, x_feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(torch.tanh(self.MLP(x_feat)), p=2, dim=-1)

    def forward(self, x_feat: torch.Tensor, id_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self._project_features(x_feat)
        layer_outputs: list[torch.Tensor] = []
        for conv, linear, g_layer, norm in zip(self.convs, self.linears, self.g_layers, self.norms):
            h = F.leaky_relu(conv(x, edge_index))
            x_hat = F.leaky_relu(linear(x)) + id_emb
            x = norm(g_layer(h) + x_hat)
            x = F.leaky_relu(x)
            layer_outputs.append(x)

        return torch.cat(layer_outputs, dim=1)

    @torch.no_grad()
    def project_all(
        self,
        x_feat: torch.Tensor,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        out_dim = self.MLP.out_features
        output = torch.empty((x_feat.size(0), out_dim), dtype=x_feat.dtype, device="cpu")
        for start in range(0, x_feat.size(0), batch_size):
            end = min(start + batch_size, x_feat.size(0))
            projected = self._project_features(x_feat[start:end].to(device))
            output[start:end] = projected.detach().cpu()
        return output

    @torch.no_grad()
    def inference(
        self,
        x_feat: torch.Tensor,
        id_embedding: nn.Embedding,
        edge_index: torch.Tensor,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        h = self.project_all(x_feat, device, batch_size)
        edge_index = edge_index.cpu()
        num_nodes = int(x_feat.size(0))
        input_nodes = torch.arange(num_nodes, dtype=torch.long)
        layer_outputs: list[torch.Tensor] = []
        clean_edge_index, _ = remove_self_loops(edge_index)
        full_degree = degree(clean_edge_index[0], num_nodes, dtype=h.dtype)

        for conv, linear, g_layer, norm in zip(self.convs, self.linears, self.g_layers, self.norms):
            data = Data(x=h, edge_index=edge_index, node_degree=full_degree)
            loader = NeighborLoader(
                data,
                input_nodes=input_nodes,
                num_neighbors=[-1],
                batch_size=batch_size,
                shuffle=False,
            )
            out = torch.empty((num_nodes, linear.out_features), dtype=h.dtype, device="cpu")
            for batch in loader:
                batch = batch.to(device)
                id_emb = id_embedding[batch.n_id]
                h_g = F.leaky_relu(conv(batch.x, batch.edge_index, node_degree=batch.node_degree))
                x_hat = F.leaky_relu(linear(batch.x)) + id_emb
                z = norm(g_layer(h_g) + x_hat)
                z = F.leaky_relu(z)[: batch.batch_size]
                out[batch.n_id[: batch.batch_size].cpu()] = z.detach().cpu()
            layer_outputs.append(out)
            h = out

        return torch.cat(layer_outputs, dim=1)


class Model(nn.Module):
    def __init__(self, cfg, data_info: dict):
        super().__init__()
        input_dim = int(data_info["input_dim"])
        num_nodes = int(data_info["num_nodes"])
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.get("num_layers", 2))
        norm = cfg.model.get("norm", "batchnorm")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.text_dim = int(data_info.get("text_dim", 0) or 0)
        self.visual_dim = int(data_info.get("visual_dim", 0) or 0)
        if self.text_dim <= 0 or self.visual_dim <= 0:
            self.text_dim = input_dim // 2
            self.visual_dim = input_dim - self.text_dim
        if self.text_dim + self.visual_dim > input_dim:
            raise ValueError(
                f"text_dim+visual_dim={self.text_dim + self.visual_dim} exceeds input_dim={input_dim}"
            )

        # Match the MMGCN convention in this repo: the per-node ID table looks
        # learnable but is intentionally NOT registered, so it is absent from
        # model.parameters() and never updated by Adam. A trainable per-node
        # table memorizes train nodes on small graphs (observed: val stuck at
        # majority-class while train loss -> 0).
        self.id_embedding = nn.init.xavier_normal_(
            torch.empty(num_nodes, hidden_dim, requires_grad=True)
        )
        self._batch_n_id: torch.Tensor | None = None

        self.v_branch = MgatBranch(
            in_dim=self.visual_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            latent_dim=256,
            norm=norm,
        )
        self.t_branch = MgatBranch(
            in_dim=self.text_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            latent_dim=100,
            norm=norm,
        )

        self.out_dim = hidden_dim * num_layers

    def _apply(self, fn):
        super()._apply(fn)
        with torch.no_grad():
            self.id_embedding = fn(self.id_embedding)
        self.id_embedding.requires_grad_(True)
        return self

    def _get_id_emb(self, num_nodes: int) -> torch.Tensor:
        n_id = getattr(self, "_batch_n_id", None)
        if n_id is not None:
            return self.id_embedding[n_id]
        return self.id_embedding[:num_nodes]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        text_feat = x[:, : self.text_dim]
        visual_feat = x[:, self.text_dim : self.text_dim + self.visual_dim]
        id_emb = self._get_id_emb(int(x.size(0)))

        v_rep = self.v_branch(visual_feat, id_emb, edge_index)
        t_rep = self.t_branch(text_feat, id_emb, edge_index)
        z = (v_rep + t_rep) / 2.0

        aux_loss = z.new_tensor(0.0)
        return z, None, None, aux_loss, {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        self.eval()
        self._batch_n_id = None
        if device is None:
            device = next(self.parameters()).device

        text_feat = x[:, : self.text_dim].cpu()
        visual_feat = x[:, self.text_dim : self.text_dim + self.visual_dim].cpu()
        edge_index = edge_index.cpu()

        v_rep = self.v_branch.inference(visual_feat, self.id_embedding, edge_index, device, batch_size)
        t_rep = self.t_branch.inference(text_feat, self.id_embedding, edge_index, device, batch_size)
        return (v_rep + t_rep) / 2.0
