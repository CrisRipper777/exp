"""R2-Design-1.6 adapters (plan §35/§25-§27).

Two adapter families on frozen parents:

1. SemanticResidualAdapter (D1.6-D, plan §35): the factor interaction
   residual ONLY, with the adaptive common gate strictly removed
   (C = 0.5*(c_t+c_v) fixed). Inserted BEFORE the parent graph path.

        I = [C*Pt, C*Pv, Pt*Pv, |C-Pt|, |C-Pv|, |Pt-Pv|]        (6d)
        Delta F = Linear(6d,128) -> LN -> GELU -> Linear(128,3d)
        F* = F0 + Delta F                                     (last layer zero-init)

2. Interaction adapters CONCAT / PRODDIFF / FiLM (D1.6-C): re-exported
   from biaxis_r2d15_adapters (identical specifications, plan §25-§27),
   inserted AFTER the parent graph path: Fhat = F_parent_out + Delta.

All zero-initialized: a fresh adapter degenerates EXACTLY to the frozen
parent output (unit-tested). No dropout inside adapters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_r2d15_adapters import (  # noqa: F401 — re-export for D1.6-C
    ConcatVectorAdapter,
    FiLMVectorAdapter,
    ProdDiffVectorAdapter,
)
from .common import get_activation, make_norm


class SemanticResidualAdapter(nn.Module):
    """Factor interaction residual BEFORE parent graph propagation
    (plan §35/§36): F* = F0 + Delta with F0 = [C, Pt, Pv] (fixed common).

    The final Linear(128, 3d) is zero-initialized: step 0 F* == F0 exactly.
    """

    def __init__(self, factor_dim: int, hidden: int = 128,
                 activation: str = "gelu", norm: str = "layernorm") -> None:
        super().__init__()
        d = int(factor_dim)
        self.factor_dim = d
        self.trunk = nn.Sequential(
            nn.Linear(6 * d, int(hidden)),
            make_norm(norm, int(hidden)),
            get_activation(activation),
        )
        self.head = nn.Linear(int(hidden), 3 * d)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def interaction_block(self, f0: torch.Tensor) -> torch.Tensor:
        """I = [C*Pt, C*Pv, Pt*Pv, |C-Pt|, |C-Pv|, |Pt-Pv|] (plan §35)."""
        c, pt, pv = f0[:, 0], f0[:, 1], f0[:, 2]
        return torch.cat(
            [c * pt, c * pv, pt * pv, (c - pt).abs(), (c - pv).abs(), (pt - pv).abs()],
            dim=-1,
        )

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """f0 [N, 3, d] -> F* [N, 3, d] = f0 + Delta."""
        delta = self.head(self.trunk(self.interaction_block(f0)))  # [N, 3d]
        return f0 + delta.reshape(f0.size(0), 3, self.factor_dim)

    def residual(self, f0: torch.Tensor) -> torch.Tensor:
        """Delta [N, 3, d] only (for diagnostics)."""
        delta = self.head(self.trunk(self.interaction_block(f0)))
        return delta.reshape(f0.size(0), 3, self.factor_dim)


def build_interaction_adapter(name: str, factor_dim: int, **kwargs) -> nn.Module:
    """CONCAT / PRODDIFF / FiLM (the D1.5-D2/D3/D4 specifications carry over
    unchanged, plan §19/§25-§27)."""
    mapping = {
        "CONCAT": ConcatVectorAdapter,
        "PRODDIFF": ProdDiffVectorAdapter,
        "FiLM": FiLMVectorAdapter,
    }
    if name not in mapping:
        raise ValueError(f"unknown interaction adapter {name!r}")
    return mapping[name](factor_dim, **kwargs)
