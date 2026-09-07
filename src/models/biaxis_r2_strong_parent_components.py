"""R2-Design-2.6 strong-parent side components
(docs/BiAxis_R2_Design_2_6_Strong_Parent_Readout_Integration.md).

    StrongParentExpert : per (factor, hop) expert, plan §6
        Linear(d, 2d) -> LN -> GELU -> Dropout(0.1) -> Linear(2d, d)
    FactorHopAttention: factor-local attention over 3 hop tokens
        (2 Pre-LN blocks, d, 4 heads, FFN 4d, dropout 0.1; ego/H0 summary)
    CrossFactorAttention: base-anchored attention over [z_base, q_C, q_Pt, q_Pv]
        (2 Pre-LN blocks, h, 4 heads, FFN 4h, dropout 0.1; returns ALL final
        tokens so the residual z_base + W_o(T_final[0] - z_base) is possible)
    ResidualFusion     : re-exported from the R2D2.5 components (B2 side readout)
    ReadoutOnlyMLP     : parameter-matched deep residual MLP on z_base (B4 control)

Discipline (plan §55):
    - The side branch may only ADD information; it never replaces z_base.
      Every residual path (RSF / HIER / READOUT_ONLY) keeps z_base as a
      direct skip; final residual projections use small nonzero init
      (std 1e-3, bias 0) — never a scalar gate.
    - HOP and H1-control variants share EVERY parameter shape (mandatory
      architecture-identical controls, plan §7).
    - Aux expert heads (deep supervision) live OUTSIDE these components —
      they are removed at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_r2_capacity_components import ResidualFusion  # noqa: F401 (re-export)
from .biaxis_r2_capacity_components import _PreLNTokenBlock, HopTokenAttention
from .common import get_activation, make_norm


class StrongParentExpert(nn.Module):
    """Per (factor, hop) expert (plan §6):

        Linear(d, 2d) -> LN -> GELU -> Dropout(0.1) -> Linear(2d, d)

    Independent parameters for every factor-hop pair; default
    initialization (the side branch must receive gradients from step 0).
    """

    def __init__(self, factor_dim: int, dropout: float = 0.1,
                 activation: str = "gelu", norm: str = "layernorm") -> None:
        super().__init__()
        d = int(factor_dim)
        self.net = nn.Sequential(
            nn.Linear(d, 2 * d),
            make_norm(norm, 2 * d),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * d, d),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class CrossFactorAttention(nn.Module):
    """Base-anchored hierarchical attention (plan §20): 2 Pre-LN transformer
    blocks over the per-node token sequence [z_base, q_C, q_Pt, q_Pv],
    embed_dim = h, 4 heads, FFN 4h, dropout 0.1. Returns the FULL final
    token sequence; the caller forms
    z = z_base + W_o(T_final[0] - z_base) so the base skip is explicit."""

    def __init__(self, dim: int, heads: int = 4, ff_mult: int = 4,
                 dropout: float = 0.1, activation: str = "gelu",
                 norm: str = "layernorm") -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_PreLNTokenBlock(dim, heads, ff_mult, dropout, activation, norm)
             for _ in range(2)]
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens: [N, 4, h] -> (final tokens [N, 4, h], mean attention
        [2, 4, 4] = layers x query x key)."""
        mean_attns = []
        for block in self.blocks:
            tokens = block(tokens)
            mean_attns.append(block.mean_attn)
        return tokens, torch.stack(mean_attns, dim=0)


class ReadoutOnlyMLP(nn.Module):
    """B4 control (plan §22): a deep residual MLP on z_base with its width
    solved so its parameter count matches the HIER side branch within
    +/-5% (the width is solved by the model constructor and passed in).

        z = z_base + M(z_base),  M = input proj + 2 residual blocks
    """

    def __init__(self, dim: int, width: int, dropout: float = 0.1,
                 activation: str = "gelu", norm: str = "layernorm",
                 final_std: float = 1e-3) -> None:
        super().__init__()
        h = int(dim)
        w = int(width)
        self.proj = nn.Linear(h, w)
        self.norm0 = make_norm(norm, w)
        self.act = get_activation(activation)
        self.drop = nn.Dropout(float(dropout))

        def _block() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(w, w),
                make_norm(norm, w),
                get_activation(activation),
                nn.Dropout(float(dropout)),
            )

        self.block1 = _block()
        self.block2 = _block()
        self.out = nn.Linear(w, h)
        nn.init.normal_(self.out.weight, std=float(final_std))
        nn.init.zeros_(self.out.bias)

    def forward(self, z_base: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.norm0(self.proj(z_base))))
        x = x + self.block1(x)
        x = x + self.block2(x)
        return self.out(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
