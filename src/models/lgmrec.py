"""LGMRec (Local-Global Multimodal Recommender, RecSys 2024) — ported from
the OpenMAG implementation (OpenMAG/src/model/models.py) and adapted to the
UNIFIED NC benchmark protocol of this repository.

Architecture (OpenMAG LGMRec):
    feat_encoder -> 3x LGConv (LightGCN, self-loops added) -> mean pool
      -> local_structure_emb + alpha * L2-normalize(hypergraph emb)
    hypergraph: HGNNLayer = linear hyperedge projector -> gumbel-softmax H
      -> node -> hyperedge -> node propagation (n_layers=1)

Protocol unification (user decision 2026-09-03): the OpenMAG trainer's
modality InfoNCE reconstruction aux (lambda_v/lambda_t = 0.5) is NOT used —
LGMRec runs under the SAME plain-CE protocol as every other baseline
(no aux losses; the OpenMAG vision/text heads and decoders exist only to
serve that aux and are therefore dropped). lr/wd preset 5e-3 / 1e-5 from
the OpenMAG NC task config; no label smoothing (unified protocol = plain
CE); eval uses deterministic softmax(score / tau) in place of gumbel noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.utils import add_self_loops


class HGNNLayer(nn.Module):
    """Hypergraph layer (OpenMAG): hyperedge assignment H = gumbel_softmax(
    Linear(x), tau), then one node -> hyperedge -> node propagation round."""

    def __init__(self, in_dim: int, hyper_num: int, tau: float = 0.5) -> None:
        super().__init__()
        self.hyper_num = int(hyper_num)
        self.tau = float(tau)
        self.hyper_projector = nn.Linear(int(in_dim), self.hyper_num, bias=False)
        nn.init.xavier_uniform_(self.hyper_projector.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hyper_score = self.hyper_projector(x)
        if self.training:
            h = F.gumbel_softmax(hyper_score, tau=self.tau, dim=1, hard=False)
        else:
            # deterministic approximation of the expected training-time H
            h = F.softmax(hyper_score / self.tau, dim=1)
        lat = torch.mm(h.t(), x)  # hyperedge aggregation
        return torch.mm(h, lat)  # hyperedge -> node distribution


class Model(nn.Module):
    """LGMRec NC encoder (unified protocol: no aux losses). Repo interface:
    forward -> (z, None, None, aux_loss, aux_info); inference -> CPU z;
    ``out_dim`` = hidden_dim."""

    def __init__(self, cfg, data_info):
        super().__init__()
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.num_layers)
        self.dropout = float(cfg.model.dropout)
        self.alpha = float(cfg.model.alpha)

        self.feat_encoder = nn.Linear(int(data_info["input_dim"]), hidden_dim)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.hgnn = HGNNLayer(hidden_dim, hyper_num=int(cfg.model.hyper_num))
        self.out_dim = hidden_dim

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_emb = F.dropout(self.feat_encoder(x), p=self.dropout, training=self.training)
        embs = [x_emb]
        current = x_emb
        for conv in self.convs:
            current = conv(current, edge_index)
            embs.append(current)
        local = torch.stack(embs, dim=1).mean(dim=1)
        global_hyper = self.hgnn(x_emb)
        return local + self.alpha * F.normalize(global_hyper, p=2, dim=1)

    def forward(self, x: torch.Tensor, edge_index=None):
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        # LightGCN convention: normalized adjacency includes self-loops
        # (OpenMAG task config self_loop=True; dataset configs here set
        # add_self_loops=false, models add their own).
        edge_index, _ = add_self_loops(edge_index, num_nodes=int(x.size(0)))
        z = self._encode(x, edge_index)
        aux_loss = z.new_tensor(0.0)
        return z, None, None, aux_loss, {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """One exact full-graph forward (LGConv needs full neighborhoods)."""
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
