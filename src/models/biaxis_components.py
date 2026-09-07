from __future__ import annotations

import torch
import torch.nn as nn

from .common import get_activation, make_norm


class ModalityProjector(nn.Module):
    """Per-modality projection: Linear -> Norm -> Act -> Dropout -> Linear.

    Text and visual projectors do NOT share parameters (different input dims
    and different modality statistics).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            make_norm(norm, int(hidden_dim)),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemanticFactorizer(nn.Module):
    """Decouples projected text/visual embeddings into semantic factors.

    Factors::
        c   = (c_t + c_v) / 2          cross-modal common consensus
        c_t = E_C(h_t), c_v = E_C(h_v) shared common encoder
        p_t = E_t^P(h_t)               text-private
        p_v = E_v^P(h_v)               visual-private

    P0 discipline: this module is strictly topology-free (per-node MLPs only).
    """

    def __init__(
        self,
        text_dim: int,
        visual_dim: int,
        hidden_dim: int = 256,
        factor_dim: int = 128,
        dropout: float = 0.2,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        self.text_projector = ModalityProjector(text_dim, hidden_dim, dropout, activation, norm)
        self.visual_projector = ModalityProjector(visual_dim, hidden_dim, dropout, activation, norm)
        # Shared across modalities: c_t and c_v go through the SAME module instance.
        self.common_encoder = self._build_factor_mlp(hidden_dim, factor_dim, activation, norm)
        # Independent per modality.
        self.private_text_encoder = self._build_factor_mlp(hidden_dim, factor_dim, activation, norm)
        self.private_visual_encoder = self._build_factor_mlp(hidden_dim, factor_dim, activation, norm)
        self.hidden_dim = int(hidden_dim)
        self.factor_dim = int(factor_dim)

    @staticmethod
    def _build_factor_mlp(in_dim: int, out_dim: int, activation: str, norm: str) -> nn.Module:
        return nn.Sequential(
            nn.Linear(int(in_dim), int(out_dim)),
            make_norm(norm, int(out_dim)),
            get_activation(activation),
            nn.Linear(int(out_dim), int(out_dim)),
        )

    def forward(self, x_t: torch.Tensor, x_v: torch.Tensor) -> dict[str, torch.Tensor]:
        h_t = self.text_projector(x_t)
        h_v = self.visual_projector(x_v)
        c_t = self.common_encoder(h_t)
        c_v = self.common_encoder(h_v)
        p_t = self.private_text_encoder(h_t)
        p_v = self.private_visual_encoder(h_v)
        c = 0.5 * (c_t + c_v)
        return {
            "h_t": h_t,
            "h_v": h_v,
            "c_t": c_t,
            "c_v": c_v,
            "c": c,
            "p_t": p_t,
            "p_v": p_v,
        }


class ReconstructionHead(nn.Module):
    """Light decoder: reconstructs the projected modality embedding h_mod from
    [c_mod || p_mod]. Prevents private collapse and common information hoarding.
    """

    def __init__(self, factor_dim: int, hidden_dim: int, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * int(factor_dim), int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )

    def forward(self, c_mod: torch.Tensor, p_mod: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([c_mod, p_mod], dim=-1))
