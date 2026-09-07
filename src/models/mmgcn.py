from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform

from .common import make_norm


# ---------------------------------------------------------------------------
# BaseModel: simple message passing, no normalization/self-loops
# Identical to mm-graph-benchmark's BaseModel.
# ---------------------------------------------------------------------------
class BaseModel(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, aggr: str = "add", **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index, size=None):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


# ---------------------------------------------------------------------------
# MMGCNBranch: one modality branch with configurable number of GCN layers.
# Per-layer: h = LeakyReLU(conv(x)), x_hat = LeakyReLU(linear(x)) + id_emb,
#            x = LeakyReLU(g_layer(h) + x_hat)
# ---------------------------------------------------------------------------
class MMGCNBranch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim_id: int,
        num_layers: int,
        aggr: str = "mean",
        dropout: float = 0.0,
        norm: str | None = "batchnorm",
        has_mlp: bool = False,
        mlp_dim: int = 256,
    ):
        super().__init__()
        self.has_mlp = has_mlp
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList([make_norm(norm, dim_id) for _ in range(max(num_layers - 1, 0))])
        first_dim = mlp_dim if has_mlp else in_dim

        if has_mlp:
            self.MLP = nn.Linear(in_dim, mlp_dim)

        # Layer 1
        self.conv1 = BaseModel(first_dim, first_dim, aggr=aggr)
        nn.init.xavier_normal_(self.conv1.weight)
        self.linear1 = nn.Linear(first_dim, dim_id)
        nn.init.xavier_normal_(self.linear1.weight)
        self.g1 = nn.Linear(first_dim, dim_id)
        nn.init.xavier_normal_(self.g1.weight)

        # Layers 2..num_layers
        for i in range(2, num_layers + 1):
            conv = BaseModel(dim_id, dim_id, aggr=aggr)
            nn.init.xavier_normal_(conv.weight)
            linear = nn.Linear(dim_id, dim_id)
            nn.init.xavier_normal_(linear.weight)
            g = nn.Linear(dim_id, dim_id)
            nn.init.xavier_normal_(g.weight)
            setattr(self, f"conv{i}", conv)
            setattr(self, f"linear{i}", linear)
            setattr(self, f"g{i}", g)

    def forward(self, x_feat: torch.Tensor, id_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.MLP(x_feat) if self.has_mlp else x_feat
        x = F.normalize(x)

        for i in range(1, self.num_layers + 1):
            conv = getattr(self, f"conv{i}")
            linear = getattr(self, f"linear{i}")
            g = getattr(self, f"g{i}")
            h = F.leaky_relu(conv(x, edge_index))
            x_hat = F.leaky_relu(linear(x)) + id_emb
            x = g(h) + x_hat
            if i != self.num_layers:
                x = self.norms[i - 1](x)
            x = F.leaky_relu(x)
            if i != self.num_layers:
                x = self.dropout(x)

        return x


# ---------------------------------------------------------------------------
# Model: top-level MMGCN for MAG_baseline.
# forward(x, edge_index) -> (z, None, None, aux_loss, {})
# inference(x, edge_index, device, batch_size) -> Tensor
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, cfg, data_info: dict):
        super().__init__()
        input_dim = int(data_info["input_dim"])
        num_nodes = int(data_info["num_nodes"])
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.get("num_layers", 3))
        aggr = str(cfg.model.get("aggr", "mean"))
        dropout = float(cfg.model.get("dropout", 0.0))
        norm = cfg.model.get("norm", "batchnorm")

        # MAG_baseline stores joint features in mm-graph-benchmark order: [text, visual].
        self.text_dim = int(data_info.get("text_dim", 0) or 0)
        self.visual_dim = int(data_info.get("visual_dim", 0) or 0)
        if self.text_dim <= 0 or self.visual_dim <= 0:
            self.text_dim = input_dim // 2
            self.visual_dim = input_dim - self.text_dim
        if self.text_dim + self.visual_dim > input_dim:
            raise ValueError(
                f"text_dim+visual_dim={self.text_dim + self.visual_dim} exceeds input_dim={input_dim}"
            )

        # Match the reference MMGCN implementations: this looks learnable
        # (`requires_grad=True`) but is intentionally not an nn.Parameter, so it
        # is absent from model.parameters() and is not updated by Adam.
        self.id_embedding = nn.init.xavier_normal_(
            torch.empty(num_nodes, hidden_dim, requires_grad=True)
        )
        self._batch_n_id: torch.Tensor | None = None

        # Visual branch: MLP projection (dim_latent=256 in reference)
        self.v_branch = MMGCNBranch(
            in_dim=self.visual_dim,
            dim_id=hidden_dim,
            num_layers=num_layers,
            aggr=aggr,
            dropout=dropout,
            norm=norm,
            has_mlp=True,
            mlp_dim=256,
        )
        # Text branch: no MLP projection
        self.t_branch = MMGCNBranch(
            in_dim=self.text_dim,
            dim_id=hidden_dim,
            num_layers=num_layers,
            aggr=aggr,
            dropout=dropout,
            norm=norm,
            has_mlp=False,
        )

        self.out_dim = hidden_dim

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

    def forward(self, x, edge_index):
        text_feat = x[:, : self.text_dim]
        visual_feat = x[:, self.text_dim : self.text_dim + self.visual_dim]
        id_emb = self._get_id_emb(int(x.size(0)))

        v_rep = self.v_branch(visual_feat, id_emb, edge_index)
        t_rep = self.t_branch(text_feat, id_emb, edge_index)
        z = (v_rep + t_rep) / 2.0

        aux_loss = z.new_tensor(0.0)
        return z, None, None, aux_loss, {}

    @torch.no_grad()
    def inference(self, x, edge_index, device=None, batch_size=65536):
        self.eval()
        self._batch_n_id = None
        if device is None:
            device = next(self.parameters()).device

        h_text = x[:, : self.text_dim].cpu()
        h_visual = x[:, self.text_dim : self.text_dim + self.visual_dim].cpu()
        edge_index = edge_index.cpu()
        num_nodes = int(x.size(0))
        input_nodes = torch.arange(num_nodes, dtype=torch.long)
        dim_id = self.out_dim

        # --- Visual branch (has MLP) ---
        h = self.v_branch.MLP(h_visual.to(device)).cpu()
        for i in range(1, self.v_branch.num_layers + 1):
            conv = getattr(self.v_branch, f"conv{i}")
            linear = getattr(self.v_branch, f"linear{i}")
            g = getattr(self.v_branch, f"g{i}")

            if i == 1:
                h = F.normalize(h)
            data = Data(x=h, edge_index=edge_index)
            loader = NeighborLoader(
                data, input_nodes=input_nodes,
                num_neighbors=[-1], batch_size=batch_size, shuffle=False,
            )
            out = torch.empty((num_nodes, dim_id), dtype=h.dtype)
            for batch in loader:
                batch = batch.to(device)
                id_emb = self.id_embedding[batch.n_id]
                h_g = F.leaky_relu(conv(batch.x, batch.edge_index))
                x_hat = F.leaky_relu(linear(batch.x)) + id_emb
                z = g(h_g) + x_hat
                if i != self.v_branch.num_layers:
                    z = self.v_branch.norms[i - 1](z)
                z = F.leaky_relu(z)[: batch.batch_size]
                out[batch.n_id[: batch.batch_size].cpu()] = z.detach().cpu()
            h = out

        v_out = h

        # --- Text branch (no MLP) ---
        h = h_text
        for i in range(1, self.t_branch.num_layers + 1):
            conv = getattr(self.t_branch, f"conv{i}")
            linear = getattr(self.t_branch, f"linear{i}")
            g = getattr(self.t_branch, f"g{i}")

            if i == 1:
                h = F.normalize(h)
            data = Data(x=h, edge_index=edge_index)
            loader = NeighborLoader(
                data, input_nodes=input_nodes,
                num_neighbors=[-1], batch_size=batch_size, shuffle=False,
            )
            out = torch.empty((num_nodes, dim_id), dtype=h.dtype)
            for batch in loader:
                batch = batch.to(device)
                id_emb = self.id_embedding[batch.n_id]
                h_g = F.leaky_relu(conv(batch.x, batch.edge_index))
                x_hat = F.leaky_relu(linear(batch.x)) + id_emb
                z = g(h_g) + x_hat
                if i != self.t_branch.num_layers:
                    z = self.t_branch.norms[i - 1](z)
                z = F.leaky_relu(z)[: batch.batch_size]
                out[batch.n_id[: batch.batch_size].cpu()] = z.detach().cpu()
            h = out

        t_out = h
        return (v_out + t_out) / 2.0
