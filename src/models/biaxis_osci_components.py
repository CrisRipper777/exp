"""Components for R5-1 Ownership-Conditioned Source Integration (OSCI)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import get_activation


class OwnershipConditionedSourceComposer(nn.Module):
    """Compose two source-factor messages for one target ownership factor.

    The module is deliberately mode agnostic.  C1, C2, and C3 instantiate the
    same parameters and differ only in the tensors supplied as ``q``, ``u1``,
    and ``u2`` by the model wrapper.
    """

    def __init__(self, factor_dim: int, hidden_dim: int, activation: str) -> None:
        super().__init__()
        d = int(factor_dim)
        self.factor_dim = d
        self.factor_embedding = nn.Embedding(3, d)
        self.target_norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.LayerNorm(7 * d),
            nn.Linear(7 * d, int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), d),
        )

    def make_query(self, target: torch.Tensor, factor: int, include_target: bool) -> torch.Tensor:
        embedding = self.factor_embedding.weight[int(factor)].to(dtype=target.dtype)
        embedding = embedding.view(1, -1).expand(target.size(0), -1)
        if not include_target:
            # C2 remains independent of node target state.  This zero-valued
            # anchor keeps the shared target_norm parameters in the graph so
            # C1--C3 have identical parameter usage without changing values.
            parameter_anchor = (self.target_norm.weight.sum() + self.target_norm.bias.sum()) * 0.0
            return embedding + parameter_anchor
        return self.target_norm(target) + embedding

    def forward(
        self,
        q: torch.Tensor,
        u1: torch.Tensor,
        u2: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([
            q, u1, u2,
            q * u1, q * u2,
            q - u1, q - u2,
        ], dim=-1)
        return self.mlp(features)
