from __future__ import annotations

import torch

from src.utils.metrics import accuracy, macro_f1, mrr_hits_from_ranks


def test_nc_metrics() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 3.0], [4.0, 1.0]])
    labels = torch.tensor([0, 1, 1])
    assert abs(accuracy(logits, labels) - 2 / 3) < 1e-6
    assert 0.0 <= macro_f1(logits, labels) <= 1.0


def test_rank_metrics() -> None:
    metrics = mrr_hits_from_ranks(torch.tensor([1.0, 2.0, 10.0]))
    assert abs(metrics["hits@1"] - 1 / 3) < 1e-6
    assert metrics["hits@10"] == 1.0
    assert abs(metrics["mrr"] - ((1.0 + 0.5 + 0.1) / 3.0)) < 1e-6
