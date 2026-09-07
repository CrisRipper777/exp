from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models import LinkPredictor
from src.tasks.lp import _evaluate_split
from src.tasks.nc import _resolve_training_mode
from src.utils.summary import mean_std


def test_mean_std_uses_population_std() -> None:
    mean, std = mean_std([1.0, 3.0])
    assert mean == 2.0
    assert abs(std - 1.0) < 1e-12  # ddof=0 (RPTA reporting convention)


def test_lp_ranking_pessimistic_ties() -> None:
    # All embeddings identical -> every negative ties with the positive.
    # RPTA/OpenMAG rule: ties rank BEHIND the positive => rank = 1 + num_neg.
    z = torch.zeros((8, 4))
    predictor = LinkPredictor(in_dim=4, hidden_dim=8, num_layers=2, dropout=0.0)
    split = {
        "source_node": torch.tensor([0, 1]),
        "target_node": torch.tensor([2, 3]),
        "target_node_neg": torch.tensor([[4, 5], [6, 7]]),
    }
    metrics = _evaluate_split(z, predictor, split, torch.device("cpu"), batch_size=2)
    assert abs(metrics["mrr"] - 1.0 / 3.0) < 1e-6
    assert metrics["hits@1"] == 0.0
    assert metrics["hits@3"] == 1.0
    assert metrics["hits@10"] == 1.0


class _FakeModel:
    def __init__(self, requires_full_graph: bool):
        self.requires_full_graph_training = requires_full_graph


def test_resolve_training_mode() -> None:
    cfg = OmegaConf.create({"task": {"training_mode": "sampled"}, "model": {"name": "gcn"}})
    assert _resolve_training_mode(cfg, _FakeModel(False)) == "sampled"
    assert _resolve_training_mode(cfg, _FakeModel(True)) == "full_graph"
    cfg = OmegaConf.create({"task": {}, "model": {"name": "gcn"}})
    assert _resolve_training_mode(cfg, _FakeModel(False)) == "full_graph"  # default
