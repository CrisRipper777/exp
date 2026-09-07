from __future__ import annotations

import torch

from src.data import EdgeSplit
from src.tasks.lp import (
    _build_epoch_train_labels,
    _build_forbidden_edge_keys,
    _exclude_positive_label_edges_from_message_graph,
)


def _pack(src: list[int], dst: list[int]) -> dict[str, torch.Tensor]:
    return {
        "source_node": torch.tensor(src, dtype=torch.long),
        "target_node": torch.tensor(dst, dtype=torch.long),
    }


def test_epoch_train_negative_edges_are_globally_filtered() -> None:
    edge_split = EdgeSplit(
        train=_pack([0, 2], [1, 3]),
        valid=_pack([0], [2]) | {"target_node_neg": torch.tensor([[4, 5]])},
        test=_pack([4], [5]) | {"target_node_neg": torch.tensor([[0, 1]])},
    )
    num_nodes = 6
    forbidden = _build_forbidden_edge_keys(edge_split, num_nodes=num_nodes, undirected=True)
    edge_label_index, edge_label = _build_epoch_train_labels(
        edge_split,
        num_nodes=num_nodes,
        num_neg=2,
        forbidden_keys=forbidden,
        generator=torch.Generator().manual_seed(11),
    )

    assert edge_label_index.size(1) == 6
    assert edge_label.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert edge_label_index[:, :2].tolist() == [[0, 2], [1, 3]]

    positive_pairs = {(0, 1), (1, 0), (2, 3), (3, 2), (0, 2), (2, 0), (4, 5), (5, 4)}
    sampled_pairs = list(zip(edge_label_index[0, 2:].tolist(), edge_label_index[1, 2:].tolist(), strict=True))
    per_source: dict[int, list[int]] = {}
    for src, dst in sampled_pairs:
        assert src != dst
        assert (src, dst) not in positive_pairs
        per_source.setdefault(src, []).append(dst)
    assert all(len(targets) == len(set(targets)) for targets in per_source.values())


def test_positive_label_edges_are_excluded_from_message_graph() -> None:
    message_edge_index = torch.tensor(
        [
            [0, 1, 0, 2, 3, 4, 5],
            [1, 0, 2, 0, 4, 3, 0],
        ],
        dtype=torch.long,
    )
    edge_label_index = torch.tensor(
        [
            [0, 3],
            [1, 5],
        ],
        dtype=torch.long,
    )
    edge_label = torch.tensor([1.0, 0.0])

    filtered = _exclude_positive_label_edges_from_message_graph(
        message_edge_index,
        edge_label_index,
        edge_label,
        num_nodes=6,
    )

    assert filtered.tolist() == [[0, 2, 3, 4, 5], [2, 0, 4, 3, 0]]
