from __future__ import annotations

import torch
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected


def ensure_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.dim() != 2:
        raise ValueError(f"edge_index must be 2-D, got shape {tuple(edge_index.shape)}")
    if edge_index.size(0) == 2:
        return edge_index.contiguous()
    if edge_index.size(1) == 2:
        return edge_index.t().contiguous()
    raise ValueError(f"cannot interpret edge_index shape {tuple(edge_index.shape)}")


def preprocess_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    make_undirected: bool = True,
    with_self_loops: bool = False,
) -> torch.Tensor:
    edge_index = ensure_edge_index(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    if make_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    if with_self_loops:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index.contiguous()


def edge_dict_to_index(edge_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    src = torch.as_tensor(edge_dict["source_node"], dtype=torch.long)
    dst = torch.as_tensor(edge_dict["target_node"], dtype=torch.long)
    return torch.stack([src, dst], dim=0).contiguous()


def canonicalize_edges(edge_index: torch.Tensor) -> torch.Tensor:
    row, col = ensure_edge_index(edge_index)
    src = torch.minimum(row, col)
    dst = torch.maximum(row, col)
    edges = torch.stack([src, dst], dim=1)
    edges = edges[src != dst]
    return torch.unique(edges, dim=0)
