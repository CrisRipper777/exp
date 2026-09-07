"""R2-Design-1 components: node-adaptive common consensus, ownership-preserving
factor interaction residual, and the shared factor-context functional scorer.

Plan: docs/BiAxis_R2_Design_1_Implementation_Validation_Plan.md (§6/§7/§11).

Discipline:
    - These modules are per-node only (topology-free); the graph coupling
      lives in biaxis_r2.Model.
    - Every residual/gate entry point is zero-initialized so step 0 exactly
      degenerates to the fixed 50/50 common average / zero interaction
      residual (plan §6/§7), and the functional scorer starts at g~0.5 with
      an additional rho_func=0.01 LayerScale (plan §13/§14).
    - No dropout inside the scorer (deterministic compatibility gate); the
      semantic trunk uses the current model dropout (plan §7).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_activation, make_norm


class AdaptiveCommonGate(nn.Module):
    """Node-adaptive common consensus (plan §6):

        u_i^c   = [c_t, c_v, c_t * c_v, |c_t - c_v|]          (4d)
        logits  = Linear(4d, 64) -> GELU -> Linear(64, 2)     (last layer zero-init)
        [w_t, w_v] = Softmax(logits)
        C_i^0   = w_t * c_t + w_v * c_v

    Zero-initialized final layer => step 0 logits = 0 => w_t = w_v = 0.5,
    i.e. the model strictly degenerates to the current common average
    (mandatory stability design, plan §6).
    """

    def __init__(
        self,
        factor_dim: int,
        hidden_dim: int = 64,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * int(factor_dim), int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), 2),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, c_t: torch.Tensor, c_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (C0 [N,d], w [N,2] = [w_t, w_v])."""
        u = torch.cat([c_t, c_v, c_t * c_v, (c_t - c_v).abs()], dim=-1)  # [N, 4d]
        logits = self.net(u)  # [N, 2]
        w = torch.softmax(logits, dim=-1)  # [N, 2]
        c0 = w[:, 0:1] * c_t + w[:, 1:2] * c_v  # [N, d]
        return c0, w


class SemanticInteractionResidual(nn.Module):
    """Ownership-preserving factor interaction residual (plan §7):

        I      = [C0*Pt, C0*Pv, Pt*Pv, |C0-Pt|, |C0-Pv|, |Pt-Pv|]   (6d)
        r_sem  = Linear(6d, d) -> LN -> GELU -> Dropout              (shared trunk)
        Delta^b = W_b^sem * r_sem                                    (3 heads)

    The 3 factor-specific heads are zero-initialized: step 0 Delta = 0
    exactly, so refinement can only grow from the shared interaction trunk.
    No full mixing / transformer here (plan §7).
    """

    def __init__(
        self,
        factor_dim: int,
        dropout: float = 0.2,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        d = int(factor_dim)
        self.trunk = nn.Sequential(
            nn.Linear(6 * d, d),
            make_norm(norm, d),
            get_activation(activation),
            nn.Dropout(float(dropout)),
        )
        self.heads = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(3)])
        for head in self.heads:
            nn.init.zeros_(head.weight)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """f0: [N, 3, d] with factor order [C, Pt, Pv] -> Delta [N, 3, d]."""
        c0, pt, pv = f0[:, 0], f0[:, 1], f0[:, 2]
        interaction = torch.cat(
            [
                c0 * pt,
                c0 * pv,
                pt * pv,
                (c0 - pt).abs(),
                (c0 - pv).abs(),
                (pt - pv).abs(),
            ],
            dim=-1,
        )  # [N, 6d]
        r_sem = self.trunk(interaction)  # [N, d]
        deltas = [head(r_sem) for head in self.heads]  # 3 x [N, d]
        return torch.stack(deltas, dim=1)  # [N, 3, d]


class FunctionalScorer(nn.Module):
    """Shared factor-context functional compatibility scorer (plan §11):

        s^{a->b} = Linear(4d + 2*type_dim, 64) -> GELU -> Linear(64, 1)

    The final layer is small-normal initialized (std=1e-3, bias=0) so the
    gate starts at g ~ 0.5; combined with rho_func=0.01 the new path starts
    at ~0.005 x message (plan §14) — gradients flow without perturbing B0.

    The gate is an INDEPENDENT sigmoid per (a -> b) cell. Softmax over
    sources is forbidden: different source factors may be useful
    simultaneously (R1 competitive-routing lesson, plan §11).
    """

    def __init__(
        self,
        factor_dim: int,
        type_dim: int = 8,
        hidden_dim: int = 64,
        activation: str = "gelu",
        final_std: float = 1.0e-3,
    ) -> None:
        super().__init__()
        in_dim = 4 * int(factor_dim) + 2 * int(type_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), 1),
        )
        final = self.net[-1]
        nn.init.normal_(final.weight, std=float(final_std))
        nn.init.zeros_(final.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: [N, 4d + 2*type_dim] -> score [N, 1] (caller applies sigmoid)."""
        return self.net(u)
