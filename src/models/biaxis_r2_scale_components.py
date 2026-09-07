"""R2-Design-2.0 scale components: factor-specific propagation-horizon mixers.

Plan: docs/BiAxis_R2_Design_2_0_Factor_Specific_Propagation_Horizon_Plan.md.

    M0 : Hmix = H1                        (exact B0 1-hop)
    M1 : Hmix = H1 + alpha_f * (H2 - H1)  (3 direct scalars, init 0)
    M2 : Hmix = sum_k gamma_fk * Hk       (softmax(theta/1.0), theta=[-4,4,-4])

Discipline:
    - NO sigmoid / softmax / clamp on M1 alphas (plan §6.2);
    - M1 at init is EXACTLY H1; M2 at init is numerically ~H1 (gamma1 ≈
      0.9993; the residual ~3.3e-4 * (H0+H2-2H1) is reported, plan §36);
    - no high-pass, no K-relation/Gamma/OFR, no node-wise routing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FactorHopMixer(nn.Module):
    """Global factor-specific hop mixture (plan §2.2: factor-global, NOT
    node-wise)."""

    M0 = "m0"
    M1 = "m1"
    M2 = "m2"
    MODES = (M0, M1, M2)

    def __init__(self, mode: str, num_factors: int = 3, tau: float = 1.0) -> None:
        super().__init__()
        assert mode in self.MODES, mode
        self.mode = mode
        self.num_factors = int(num_factors)
        if mode == self.M1:
            # direct learnable scalars, init 0 -> step 0 exactly H1 (plan §6.2)
            self.alpha = nn.Parameter(torch.zeros(self.num_factors))
        elif mode == self.M2:
            # softmax(theta/tau); theta=[-4, 4, -4] -> gamma1 ≈ 0.9993 (plan §8.1)
            self.hop_logits = nn.Parameter(
                torch.tensor([[-4.0, 4.0, -4.0]] * self.num_factors)
            )
            self.tau = float(tau)

    def forward(
        self, h0: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor
    ) -> torch.Tensor:
        """h0/h1/h2: [N, F, d] -> Hmix [N, F, d]."""
        if self.mode == self.M0:
            return h1
        if self.mode == self.M1:
            return h1 + self.alpha.view(1, -1, 1) * (h2 - h1)
        # M2: gamma = softmax(theta / tau) over the hop axis
        gamma = torch.softmax(self.hop_logits / self.tau, dim=-1)  # [F, 3]
        return (
            gamma[:, 0].view(1, -1, 1) * h0
            + gamma[:, 1].view(1, -1, 1) * h1
            + gamma[:, 2].view(1, -1, 1) * h2
        )

    def scale_diagnostics(self) -> dict:
        """JSON-safe learned coefficients (plan §29)."""
        if self.mode == self.M1:
            return {
                "mode": self.mode,
                "alpha": [float(v) for v in self.alpha.detach().cpu().tolist()],
            }
        if self.mode == self.M2:
            gamma = torch.softmax(self.hop_logits / self.tau, dim=-1).detach().cpu()
            entropy = -(gamma * torch.log(gamma + 1e-12)).sum(dim=-1)  # [F]
            depth = (gamma * torch.arange(3, dtype=gamma.dtype)).sum(dim=-1)  # [F]
            return {
                "mode": self.mode,
                "gamma": [[float(v) for v in row] for row in gamma.tolist()],
                "entropy": [float(v) for v in entropy.tolist()],
                "effective_depth": [float(v) for v in depth.tolist()],
            }
        return {"mode": self.mode}
