from __future__ import annotations

import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


def make_norm(name: str | None, dim: int) -> nn.Module:
    if not name or str(name).lower() in {"none", "null"}:
        return nn.Identity()
    name = str(name).lower()
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if name == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unsupported norm: {name}")
