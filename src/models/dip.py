from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import add_remaining_self_loops, scatter, to_torch_coo_tensor


class PathIntegral(nn.Module):
    def __init__(self, q_dim: int, n_q: int):
        super().__init__()
        self.lambda_copies = nn.Parameter(torch.randn(n_q, 1, 1))
        self.n_q = n_q
        self.q_dim = q_dim

    def forward(self, in_subsystem: torch.Tensor, out_subsystem: torch.Tensor) -> torch.Tensor:
        if in_subsystem is None or out_subsystem is None:
            raise ValueError("Path integral requires computational objects.")

        if in_subsystem.shape[-3] == 1 and out_subsystem.shape[-3] == self.n_q:
            out_subsystem = out_subsystem * self.lambda_copies
            out_subsystem_sum = out_subsystem.sum(-3) / (self.n_q * self.q_dim)
            weighted_dist_sum = torch.matmul(
                in_subsystem.squeeze(-3),
                out_subsystem_sum.transpose(-2, -1),
            )
        elif in_subsystem.shape[-3] == self.n_q and out_subsystem.shape[-3] == 1:
            in_subsystem = in_subsystem * self.lambda_copies
            in_subsystem_sum = in_subsystem.sum(-3) / (self.n_q * self.q_dim)
            weighted_dist_sum = torch.matmul(
                in_subsystem_sum,
                out_subsystem.squeeze(-3).transpose(-2, -1),
            )
        else:
            dist = torch.matmul(in_subsystem, out_subsystem.transpose(-2, -1)) / self.q_dim
            weighted_dist = dist * self.lambda_copies
            weighted_dist_sum = weighted_dist.sum(-3) / self.n_q

        with torch.no_grad():
            clip_check = torch.abs(weighted_dist_sum.sum(-1, keepdim=True))
            scaler = torch.where(clip_check > 1e4, 1e4 / clip_check, torch.ones_like(clip_check))
        return scaler * weighted_dist_sum


class PNodeCommunicator(nn.Module):
    def __init__(self, d_in: int, d_out: int, q_dim: int, n_q: int, dropout: float):
        super().__init__()
        self.q_dim = q_dim
        self.pnode_agg = PathIntegral(q_dim, n_q)
        self.glob2disp = nn.Sequential(
            nn.Linear(d_in, q_dim * n_q),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.glob2value = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, state: torch.Tensor, glob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        glob_updater = self.pnode_agg(state, state)
        glob_update = torch.matmul(glob_updater, glob)
        displacement = self.glob2disp(glob_update)
        displacement = displacement.unflatten(-1, (self.q_dim, -1))
        displacement = displacement.permute(0, 3, 1, 2)
        dispatch_value = self.glob2value(glob_update)
        return displacement, dispatch_value


class NodePseudoSubsystem(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_pnode: int,
        q_dim: int,
        n_q: int,
        dropout: float = 0.0,
        norm: bool = True,
    ):
        super().__init__()
        self.collection1 = PathIntegral(q_dim, n_q)
        self.pnode_agg1 = PNodeCommunicator(d_model, d_model, q_dim, n_q, dropout)
        self.inspection = PathIntegral(q_dim, n_q)
        self.hstate_interface = nn.Sequential(
            nn.Linear(d_model * 2 + q_dim, q_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.collection2 = PathIntegral(q_dim, n_q)
        self.pnode_agg2 = PNodeCommunicator(d_model * 3 + q_dim, q_dim, q_dim, n_q, dropout)
        self.dispatch = PathIntegral(q_dim, n_q)
        self.feat_ff = nn.Sequential(
            nn.Linear(q_dim, d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.norm = norm
        if norm:
            self.phidden_norm = nn.LayerNorm(q_dim)
            self.hidden_norm = nn.LayerNorm(q_dim)
            self.pout_norm = nn.LayerNorm(q_dim)
            self.out_norm = nn.LayerNorm(q_dim)
            self.feat_norm = nn.LayerNorm(d_model)
        self.pnode_num = n_pnode

    def _feature_inspection(
        self,
        features: torch.Tensor,
        node_state: torch.Tensor,
        pnode_state: torch.Tensor,
        node_num: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ipn2n_dist = self.collection1(pnode_state, node_state)
        glob_init = torch.matmul(ipn2n_dist, features) / node_num
        pnode_disp1, self.str_inspector = self.pnode_agg1(pnode_state, glob_init)
        pnode_state = pnode_disp1 + pnode_state
        if self.norm:
            pnode_state = self.phidden_norm(pnode_state)
        n2ipn_dist = self.inspection(node_state, pnode_state)
        inspector = torch.matmul(n2ipn_dist, self.str_inspector)
        return inspector, pnode_state

    def _pnode_aggregator(
        self,
        pnode_state: torch.Tensor,
        hnode_state: torch.Tensor,
        insp_out: torch.Tensor,
        node_num: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        opn2n_dist = self.collection2(pnode_state, hnode_state)
        glob_info = torch.matmul(opn2n_dist, insp_out) / node_num
        glob_info = torch.cat((glob_info, self.str_inspector), dim=-1)
        pnode_disp2, dispatch_value = self.pnode_agg2(pnode_state, glob_info)
        pnode_state = pnode_state + pnode_disp2
        if self.norm:
            pnode_state = self.pout_norm(pnode_state)
        n2opn_dist = self.dispatch(hnode_state, pnode_state)
        dispatch_value = torch.matmul(n2opn_dist, dispatch_value)
        return dispatch_value, pnode_state

    def _edge_aggregation(
        self,
        insp_in: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        adj = to_torch_coo_tensor(edge_index, edge_weight, size=size)
        return torch.matmul(adj, insp_in)

    def forward(
        self,
        *,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        features: torch.Tensor,
        node_state: torch.Tensor,
        pnode_state: torch.Tensor,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_nodes = features.shape[:2]
        node_num = num_nodes

        insp, pnode_state = self._feature_inspection(
            features,
            node_state.unsqueeze(1),
            pnode_state,
            node_num,
        )
        insp_in = torch.cat((features, insp, node_state), dim=-1)
        insp_out = self._edge_aggregation(
            insp_in.flatten(0, -2),
            edge_index,
            edge_weight,
            size,
        ).view(batch_size, num_nodes, -1)

        hnode_state = self.hstate_interface(insp_out) + node_state
        if self.norm:
            hnode_state = self.hidden_norm(hnode_state)

        dispatch_value, pnode_state = self._pnode_aggregator(
            pnode_state,
            hnode_state.unsqueeze(1),
            insp_out,
            node_num,
        )
        features = self.feat_ff(dispatch_value) + features
        node_state = hnode_state + dispatch_value
        if self.norm:
            node_state = self.out_norm(node_state)
            features = self.feat_norm(features)

        return node_state, pnode_state, features


class MultiModalNodePseudoSubsystem(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_pnode_v: int,
        n_pnode_t: int,
        q_dim: int,
        n_q: int,
        dropout: float = 0.0,
        norm: bool = True,
        fusion_type: str = "path_integral",
    ):
        super().__init__()
        self.visual_subsystem = NodePseudoSubsystem(
            d_model=d_model,
            n_pnode=n_pnode_v,
            q_dim=q_dim,
            n_q=n_q,
            dropout=dropout,
            norm=norm,
        )
        self.text_subsystem = NodePseudoSubsystem(
            d_model=d_model,
            n_pnode=n_pnode_t,
            q_dim=q_dim,
            n_q=n_q,
            dropout=dropout,
            norm=norm,
        )

        if fusion_type == "pnode_comm":
            self.modal_fusion_v2t = PNodeCommunicator(q_dim, q_dim, q_dim, n_q, dropout)
            self.modal_fusion_t2v = PNodeCommunicator(q_dim, q_dim, q_dim, n_q, dropout)
        elif fusion_type == "path_integral":
            self.modal_fusion = PathIntegral(q_dim, n_q)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.fusion_type = fusion_type
        self.norm = norm
        if norm:
            self.pnode_v_norm = nn.LayerNorm(q_dim)
            self.pnode_t_norm = nn.LayerNorm(q_dim)

    def forward(
        self,
        *,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        feat_v: torch.Tensor,
        feat_t: torch.Tensor,
        pstate_v: torch.Tensor,
        pstate_t: torch.Tensor,
        nstate_v: torch.Tensor,
        nstate_t: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        size = (num_nodes, num_nodes)
        nstate_v, pstate_v, feat_v = self.visual_subsystem(
            edge_index=edge_index,
            edge_weight=edge_weight,
            features=feat_v,
            node_state=nstate_v,
            pnode_state=pstate_v,
            size=size,
        )
        nstate_t, pstate_t, feat_t = self.text_subsystem(
            edge_index=edge_index,
            edge_weight=edge_weight,
            features=feat_t,
            node_state=nstate_t,
            pnode_state=pstate_t,
            size=size,
        )

        if self.fusion_type == "pnode_comm":
            fusion_disp_v2t, _ = self.modal_fusion_v2t(pstate_v, pstate_t)
            fusion_disp_t2v, _ = self.modal_fusion_t2v(pstate_t, pstate_v)
            pstate_v = pstate_v + fusion_disp_t2v
            pstate_t = pstate_t + fusion_disp_v2t
        else:
            fusion_v2t = self.modal_fusion(pstate_v, pstate_t)
            fusion_t2v = self.modal_fusion(pstate_t, pstate_v)
            pstate_v = pstate_v + torch.matmul(fusion_v2t, pstate_t)
            pstate_t = pstate_t + torch.matmul(fusion_t2v, pstate_v)

        if self.norm:
            pstate_v = self.pnode_v_norm(pstate_v)
            pstate_t = self.pnode_t_norm(pstate_t)
        return nstate_v, nstate_t, pstate_v, pstate_t, feat_v, feat_t


class Model(nn.Module):
    def __init__(self, cfg, data_info: dict):
        super().__init__()
        input_dim = int(data_info["input_dim"])
        d_model = int(cfg.model.get("d_model", cfg.model.get("hidden_dim", 256)))
        q_dim = int(cfg.model.get("q_dim", d_model))
        n_q = int(cfg.model.get("n_q", 8))
        n_pnode_v = int(cfg.model.get("n_pnode_v", 128))
        n_pnode_t = int(cfg.model.get("n_pnode_t", 128))
        dropout = float(cfg.model.get("dropout", 0.1))
        self.num_steps = int(cfg.model.get("mp_hops", cfg.model.get("num_layers", 3)))
        norm = bool(cfg.model.get("norm", True))
        fusion_type = str(cfg.model.get("fusion_type", "path_integral"))
        self.requires_full_graph_training = bool(cfg.model.get("full_graph_training", True))

        self.text_dim = int(data_info.get("text_dim", 0) or 0)
        self.visual_dim = int(data_info.get("visual_dim", 0) or 0)
        if self.text_dim <= 0 or self.visual_dim <= 0:
            self.text_dim = input_dim // 2
            self.visual_dim = input_dim - self.text_dim
        if self.text_dim + self.visual_dim > input_dim:
            raise ValueError(
                f"text_dim+visual_dim={self.text_dim + self.visual_dim} exceeds input_dim={input_dim}"
            )

        feature_dim = self.text_dim + self.visual_dim
        self.node_state_interface = nn.Sequential(
            nn.Linear(feature_dim, q_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.feat_ff = nn.Sequential(
            nn.Linear(feature_dim, d_model * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.pnode_state_v = nn.Parameter(torch.randn(1, n_pnode_v, q_dim))
        self.pnode_state_t = nn.Parameter(torch.randn(1, n_pnode_t, q_dim))
        self.multimodal_subsystem = MultiModalNodePseudoSubsystem(
            d_model=d_model,
            n_pnode_v=n_pnode_v,
            n_pnode_t=n_pnode_t,
            q_dim=q_dim,
            n_q=n_q,
            dropout=dropout,
            norm=norm,
            fusion_type=fusion_type,
        )

        embedding_dim = cfg.model.get("embedding_dim", None)
        if embedding_dim is None:
            self.output_proj = nn.Identity()
            self.out_dim = q_dim * 2
        else:
            embedding_dim = int(embedding_dim)
            self.output_proj = nn.Linear(q_dim * 2, embedding_dim)
            self.out_dim = embedding_dim

    def _split_and_order_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.long:
            x = x.float()
        text_feat = x[:, : self.text_dim]
        visual_feat = x[:, self.text_dim : self.text_dim + self.visual_dim]
        return torch.cat([visual_feat, text_feat], dim=-1).contiguous()

    def _normalize_edges(
        self,
        edge_index: torch.Tensor | None,
        num_nodes: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_index = edge_index.to(device)
        edge_index, edge_weight = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
        else:
            edge_weight = edge_weight.float()

        row, col = edge_index[0], edge_index[1]
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce="sum")
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor | None) -> torch.Tensor:
        features = self._split_and_order_features(x).unsqueeze(0)
        num_nodes = int(features.size(1))
        edge_index, edge_weight = self._normalize_edges(edge_index, num_nodes, features.device)

        node_states = self.node_state_interface(features)
        features = self.feat_ff(features)
        feat_v, feat_t = torch.chunk(features, 2, dim=-1)
        nstate_v, nstate_t = torch.chunk(node_states, 2, dim=-1)

        pstate_v = self.pnode_state_v.expand(features.size(0), -1, -1)
        pstate_t = self.pnode_state_t.expand(features.size(0), -1, -1)

        for _ in range(self.num_steps):
            nstate_v, nstate_t, pstate_v, pstate_t, feat_v, feat_t = self.multimodal_subsystem(
                edge_index=edge_index,
                edge_weight=edge_weight,
                feat_v=feat_v,
                feat_t=feat_t,
                pstate_v=pstate_v,
                pstate_t=pstate_t,
                nstate_v=nstate_v,
                nstate_t=nstate_t,
                num_nodes=num_nodes,
            )

        z = torch.cat([nstate_v, nstate_t], dim=-1).flatten(0, 1)
        return self.output_proj(z)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        z = self._encode(x, edge_index)
        aux_loss = z.new_tensor(0.0)
        return z, None, None, aux_loss, {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        del batch_size
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        edge_index = edge_index.to(device) if edge_index is not None else None
        z, _, _, _, _ = self(x.to(device), edge_index)
        return z.detach().cpu()
