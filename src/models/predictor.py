from __future__ import annotations

import torch
import torch.nn as nn


class LinkPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * max(num_layers - 1, 0) + [1]
        for layer_idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[layer_idx], dims[layer_idx + 1]))
            if layer_idx != len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, pair_feature: torch.Tensor) -> torch.Tensor:
        return self.net(pair_feature).view(-1)

    def score_pairs(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        return self(z_src * z_dst)
