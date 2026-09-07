from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class EdgeSplit:
    train: dict[str, torch.Tensor]
    valid: dict[str, torch.Tensor]
    test: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MAGData:
    name: str
    source: str
    task: str
    x: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int
    x_i: torch.Tensor | None = None
    x_t: torch.Tensor | None = None
    y: torch.Tensor | None = None
    train_idx: torch.Tensor | None = None
    val_idx: torch.Tensor | None = None
    test_idx: torch.Tensor | None = None
    edge_split: EdgeSplit | None = None
    num_classes: int | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def input_dim(self) -> int:
        return int(self.x.size(-1))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))
