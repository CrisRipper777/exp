from __future__ import annotations

import torch

from src.data.splits import generate_lp_split, generate_nc_split


def test_generate_nc_split_covers_all_nodes() -> None:
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    split = generate_nc_split(labels, train_ratio=0.5, val_ratio=0.25, seed=1)
    merged = torch.cat([split["train_idx"], split["val_idx"], split["test_idx"]])
    assert sorted(merged.tolist()) == list(range(labels.numel()))


def test_lp_negatives_are_filtered() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 0],
        ]
    )
    split = generate_lp_split(edge_index, num_nodes=8, val_ratio=0.2, test_ratio=0.2, num_neg=3, seed=7)
    positives = set()
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True):
        positives.add((min(src, dst), max(src, dst)))
    for section in ["valid", "test"]:
        src = split[section]["source_node"]
        neg = split[section]["target_node_neg"]
        for s, row in zip(src.tolist(), neg.tolist(), strict=True):
            for d in row:
                assert s != d
                assert (min(s, d), max(s, d)) not in positives
