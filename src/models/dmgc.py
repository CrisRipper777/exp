from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, coalesce, degree, to_undirected

EPS = 1e-10


class _GCNConv(nn.Module):
    """OpenMAG GCNConv_dense with sparse support: Linear then A @ hidden."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(x)
        if a.layout == torch.sparse_coo:
            return torch.sparse.mm(a, hidden)
        return a @ hidden


class _GraphEncoder(nn.Module):
    """Stack of GCN convs; ReLU + dropout after every layer except the last."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, nlayers: int):
        super().__init__()
        if nlayers < 1:
            raise ValueError(f"nlayers must be >= 1, got {nlayers}")
        self.gnn_encoder_layers = nn.ModuleList(
            [_GCNConv(in_dim, hidden_dim)] + [_GCNConv(hidden_dim, hidden_dim) for _ in range(nlayers - 1)]
        )
        self.act = nn.ReLU()
        self.dropout = dropout

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, a)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gnn_encoder_layers[-1](x, a)


class _AttentionShared(nn.Module):
    """OpenMAG Attention_shared: learnable cross-modal attention fusion."""

    def __init__(self, hidden_dim: int, attn_drop: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.tanh = nn.Tanh()
        self.att_l = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att_l.data, gain=1.414)
        self.att_h = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att_h.data, gain=1.414)
        self.softmax = nn.Softmax(dim=0)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds_l, embeds_h):
        beta = []
        attn_l = self.attn_drop(self.att_l)
        attn_h = self.attn_drop(self.att_h)
        for embed in embeds_l:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_l.matmul(sp.t()))
        for embed in embeds_h:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_h.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)

        z_fusion = 0
        embeds = embeds_l + embeds_h
        for i in range(len(embeds)):
            z_fusion = z_fusion + embeds[i] * beta[i]
        return F.normalize(z_fusion, dim=1, p=2)


class _FusionRepresentation(nn.Module):
    """Learnable sigmoid-weighted low/high-pass mix (OpenMAG)."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))

    def forward(self, z1, z2):
        a = torch.sigmoid(self.a)
        return (1 - a) * z1 + a * z2


class Model(nn.Module):
    """DMGC supervised adaptation (reference: OpenMAG DMGC.py).

    Keeps the DMGC architecture: per-modality projections, shared dual-
    frequency GraphEncoder over the low-pass (normalized adjacency) and
    high-pass (Laplacian) views, per-modality sigmoid low/high fusion, and
    learned attention-based cross-modal fusion with L2 normalization.

    OpenMAG's InfoNCE losses (dual-frequency + cross-modal contrastive) are
    intentionally NOT included — the adaptation uses the unified supervised
    CE protocol. The dense [N, N] operators are computed with mathematically
    identical sparse matmuls so large graphs fit in memory.
    """

    def __init__(self, cfg, data_info):
        super().__init__()
        text_dim = int(data_info["text_dim"])
        visual_dim = int(data_info["visual_dim"])
        if text_dim <= 0 or visual_dim <= 0:
            raise ValueError(
                f"dmgc requires both modalities, got text_dim={text_dim}, visual_dim={visual_dim}"
            )
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = int(cfg.model.hidden_dim)
        self.num_layers = int(cfg.model.get("num_layers", 1))
        self.dropout = float(cfg.model.get("dropout", 0.5))

        self.t_proj = nn.Linear(text_dim, self.hidden_dim)
        self.v_proj = nn.Linear(visual_dim, self.hidden_dim)

        self.encoder = _GraphEncoder(self.hidden_dim, self.hidden_dim, self.dropout, self.num_layers)
        self.att = _AttentionShared(self.hidden_dim)
        self.fusion_1 = _FusionRepresentation()
        self.fusion_2 = _FusionRepresentation()
        self.multi_view_projections_t = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout))
                for _ in range(2)
            ]
        )
        self.multi_view_projections_m = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout))
                for _ in range(2)
            ]
        )

        self.out_dim = self.hidden_dim
        self._graph_cache: tuple | None = None

    # ------------------------------------------------------------------
    # Graph construction (sparse equivalent of OpenMAG's dense operators)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_sparse_adj(adj: torch.Tensor, mode: str = "sym") -> torch.Tensor:
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()
        deg = torch.zeros(adj.size(0), device=adj.device)
        deg.index_add_(0, indices[0], values)
        if mode == "sym":
            inv_sqrt_deg = 1.0 / (torch.sqrt(deg) + EPS)
            new_values = values * inv_sqrt_deg[indices[0]] * inv_sqrt_deg[indices[1]]
        else:
            inv_deg = 1.0 / (deg + EPS)
            new_values = values * inv_deg[indices[0]]
        return torch.sparse_coo_tensor(indices, new_values, adj.size()).coalesce()

    def _build_graphs(
        self, edge_index: torch.Tensor, num_nodes: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)[0]  # OpenMAG NC self_loop=True
        adj = torch.sparse_coo_tensor(
            edge_index, torch.ones(edge_index.size(1), device=device), (num_nodes, num_nodes)
        ).coalesce()
        adj_l = self._normalize_sparse_adj(adj, mode="sym")
        # L_h = I - A_norm as sparse: L @ X == X - A_norm @ X (identical to the
        # dense OpenMAG Laplacian; both modalities share it).
        idx = adj_l.indices()
        l_values = -adj_l.values()
        l_adj = torch.sparse_coo_tensor(idx, l_values, (num_nodes, num_nodes), device=device)
        diag = torch.arange(num_nodes, device=device)
        l_identity = torch.sparse_coo_tensor(
            torch.stack([diag, diag]), torch.ones(num_nodes, device=device), (num_nodes, num_nodes)
        )
        l_h = (l_adj + l_identity).coalesce()
        return adj_l, l_h

    def _get_graphs(
        self, edge_index: torch.Tensor, num_nodes: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._graph_cache
        if cache is not None and cache[0] is edge_index and cache[1] == num_nodes:
            return cache[2], cache[3]
        adj_l, l_h = self._build_graphs(edge_index, num_nodes, device)
        self._graph_cache = (edge_index, num_nodes, adj_l, l_h)
        return adj_l, l_h

    # ------------------------------------------------------------------
    # DMGC core encode (OpenMAG DMGCCore.forward without the losses)
    # ------------------------------------------------------------------

    def _encode(self, t_feat: torch.Tensor, v_feat: torch.Tensor, adj_l: torch.Tensor, l_h: torch.Tensor) -> torch.Tensor:
        zt = self.multi_view_projections_t[0](t_feat)
        zm = self.multi_view_projections_m[0](v_feat)

        zt_l = self.encoder(zt, adj_l)
        zt_h = self.encoder(zt, l_h)
        zm_l = self.encoder(zm, adj_l)
        zm_h = self.encoder(zm, l_h)

        z_t = self.fusion_1(zt_l, zt_h)
        z_m = self.fusion_2(zm_l, zm_h)

        return self.att([z_t], [z_m])

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None):
        if edge_index is None:
            raise ValueError("DMGC requires edge_index")
        device = x.device
        num_nodes = int(x.size(0))
        t_feat = self.t_proj(x[:, : self.text_dim])
        v_feat = self.v_proj(x[:, self.text_dim : self.text_dim + self.visual_dim])
        adj_l, l_h = self._get_graphs(edge_index, num_nodes, device)
        z = self._encode(t_feat, v_feat, adj_l, l_h)
        return z, None, None, z.new_tensor(0.0), {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """Chunked projections on device, then full-graph filtering on CPU
        (identical math to a full-graph forward)."""
        self.eval()
        if edge_index is None:
            raise ValueError("DMGC requires edge_index")
        if device is None:
            device = next(self.parameters()).device
        num_nodes = int(x.size(0))
        t_feat = torch.empty((num_nodes, self.hidden_dim), dtype=x.dtype)
        v_feat = torch.empty_like(t_feat)
        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)
            chunk = x[start:end].to(device)
            t_feat[start:end] = self.t_proj(chunk[:, : self.text_dim]).cpu()
            v_feat[start:end] = self.v_proj(
                chunk[:, self.text_dim : self.text_dim + self.visual_dim]
            ).cpu()
        adj_l, l_h = self._build_graphs(edge_index.cpu(), num_nodes, torch.device("cpu"))
        return self._encode(t_feat, v_feat, adj_l, l_h).cpu()
