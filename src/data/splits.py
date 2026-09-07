from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .graph_utils import canonicalize_edges, ensure_edge_index


def generate_nc_split(
    labels: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, torch.Tensor | float | int]:
    labels_np = labels.cpu().numpy()
    idx = np.arange(labels_np.shape[0])
    train_idx, rest_idx = train_test_split(
        idx,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels_np,
    )
    relative_val_ratio = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=relative_val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels_np[rest_idx],
    )
    return {
        "train_idx": torch.as_tensor(train_idx, dtype=torch.long),
        "val_idx": torch.as_tensor(val_idx, dtype=torch.long),
        "test_idx": torch.as_tensor(test_idx, dtype=torch.long),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
    }


def _build_adjacency(edges: torch.Tensor, num_nodes: int, undirected: bool) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(num_nodes)]
    for src, dst in edges.tolist():
        if src == dst:
            continue
        adj[src].add(dst)
        if undirected:
            adj[dst].add(src)
    return adj


def _sample_filtered_negatives(
    sources: torch.Tensor,
    num_nodes: int,
    adj: list[set[int]],
    num_neg: int,
    generator: torch.Generator,
) -> torch.Tensor:
    out = torch.empty((sources.numel(), num_neg), dtype=torch.long)
    for row, src_value in enumerate(sources.tolist()):
        used: set[int] = set()
        forbidden = adj[src_value]
        while len(used) < num_neg:
            need = num_neg - len(used)
            candidates = torch.randint(
                low=0,
                high=num_nodes,
                size=(max(need * 4, 64),),
                generator=generator,
            ).tolist()
            for dst_value in candidates:
                if dst_value == src_value:
                    continue
                if dst_value in forbidden or dst_value in used:
                    continue
                used.add(dst_value)
                if len(used) == num_neg:
                    break
        out[row] = torch.as_tensor(list(used), dtype=torch.long)
    return out


def generate_lp_split(
    edge_index: torch.Tensor,
    num_nodes: int,
    val_ratio: float,
    test_ratio: float,
    num_neg: int,
    seed: int,
    undirected: bool = True,
) -> dict[str, dict[str, torch.Tensor] | dict[str, float | int | bool]]:
    edge_index = ensure_edge_index(edge_index)
    if undirected:
        positive_edges = canonicalize_edges(edge_index)
    else:
        positive_edges = torch.unique(edge_index.t().contiguous(), dim=0)
        positive_edges = positive_edges[positive_edges[:, 0] != positive_edges[:, 1]]

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(positive_edges.size(0), generator=generator)
    positive_edges = positive_edges[perm]

    num_total = positive_edges.size(0)
    num_test = int(round(num_total * test_ratio))
    num_val = int(round(num_total * val_ratio))
    num_train = num_total - num_val - num_test
    if num_train <= 0:
        raise ValueError("LP split ratios leave no training edges")

    train_edges = positive_edges[:num_train]
    valid_edges = positive_edges[num_train : num_train + num_val]
    test_edges = positive_edges[num_train + num_val :]

    adj = _build_adjacency(positive_edges, num_nodes=num_nodes, undirected=undirected)
    valid_neg = _sample_filtered_negatives(valid_edges[:, 0], num_nodes, adj, num_neg, generator)
    test_neg = _sample_filtered_negatives(test_edges[:, 0], num_nodes, adj, num_neg, generator)

    def pack(edges: torch.Tensor, neg: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        data = {
            "source_node": edges[:, 0].long().contiguous(),
            "target_node": edges[:, 1].long().contiguous(),
        }
        if neg is not None:
            data["target_node_neg"] = neg.long().contiguous()
        return data

    return {
        "train": pack(train_edges),
        "valid": pack(valid_edges, valid_neg),
        "test": pack(test_edges, test_neg),
        "metadata": {
            "seed": seed,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "num_neg": num_neg,
            "filtered": True,
            "undirected": undirected,
            "num_positive_edges": int(num_total),
        },
    }


def load_or_create_nc_split(
    path: str | Path,
    labels: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    auto_generate: bool,
) -> dict:
    path = Path(path)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    if not auto_generate:
        raise FileNotFoundError(f"NC split not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    split = generate_nc_split(labels, train_ratio, val_ratio, seed)
    torch.save(split, path)
    return split


def load_or_create_lp_split(
    path: str | Path,
    edge_index: torch.Tensor,
    num_nodes: int,
    val_ratio: float,
    test_ratio: float,
    num_neg: int,
    seed: int,
    undirected: bool,
    auto_generate: bool,
) -> dict:
    path = Path(path)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    if not auto_generate:
        raise FileNotFoundError(f"LP split not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    split = generate_lp_split(edge_index, num_nodes, val_ratio, test_ratio, num_neg, seed, undirected)
    torch.save(split, path)
    return split
