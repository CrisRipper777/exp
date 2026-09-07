"""R2-Design-1.5 frozen-B0 interaction adapters (plan §23-§25).

All adapters consume FROZEN B0 states — pre-graph ownership factors F*
[pre-graph factor F_b] and 1-hop contexts N^a — and output a zero-init
correction Delta [N, 3, d] that is added to the B0 graph-updated factors:

    Fhat^b = F_B0_out^b + Delta^b,   zhat = Fusion_B0(Fhat),   fresh classifier

Realization candidates (plan §23):
    D1 SCALAR       : independent scalar gates, U_a NOT shared with B0
    D2 CONCAT       : [F_b, N_a] concat -> shared vector MLP (capacity control)
    D3 PRODDIFF     : [F_b*N_a, |F_b-N_a|] -> parameter-matched with D2
    D4 FiLM         : delta_gamma * U_a(N_a) + beta correction

Discipline:
    - every adapter is EXACTLY zero at step 0 (zero-init last layer / alpha=0),
      so a fresh init degenerates to the frozen B0 output (unit-tested).
    - D2 and D3 have identical module shapes => identical parameter counts
      (the parameter-matched control, plan §23).
    - cell aggregation is the plain 1/3 mean over sources; no softmax /
      MoE / competitive router (plan §24).
    - no dropout inside adapters (deterministic; B0 runs in eval mode).
    - every adapter exposes cell_deltas() returning the exact per-cell
      corrections [b][a] that forward aggregates — used by the message
      novelty / expert-specialization diagnostics (plan §27).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import get_activation


class _CellAdapterBase(nn.Module):
    """Shared plumbing: type embeddings and the 1/3-mean target aggregation."""

    def __init__(self, factor_dim: int, type_dim: int = 8) -> None:
        super().__init__()
        self.factor_dim = int(factor_dim)
        self.type_dim = int(type_dim)
        self.src_type_emb = nn.Embedding(3, self.type_dim)
        self.tgt_type_emb = nn.Embedding(3, self.type_dim)

    def cell_input(self, f_pre: torch.Tensor, n_block: torch.Tensor, a: int, b: int, use_int: bool) -> torch.Tensor:
        """Interaction input for cell (a -> b). use_int selects the
        PRODDIFF form [F*N, |F-N|]; otherwise the plain concat [F, N]."""
        num_nodes = int(f_pre.size(0))
        f_b = f_pre[:, b]
        n_a = n_block[:, a]
        if use_int:
            core = torch.cat([f_b * n_a, (f_b - n_a).abs()], dim=-1)  # [N, 2d]
        else:
            core = torch.cat([f_b, n_a], dim=-1)  # [N, 2d]
        types = torch.cat(
            [
                self.src_type_emb.weight[a].unsqueeze(0).expand(num_nodes, -1),
                self.tgt_type_emb.weight[b].unsqueeze(0).expand(num_nodes, -1),
            ],
            dim=-1,
        )
        return torch.cat([core, types], dim=-1)

    def aggregate(self, cell_deltas: list[torch.Tensor]) -> torch.Tensor:
        """Delta^b = (1/3) sum_a Delta^{a->b} -> [N, 3, d]."""
        return torch.stack(cell_deltas, dim=1)


class ScalarAdapter(_CellAdapterBase):
    """D1 SCALAR (plan §23): decoupled scalar control on a frozen B0.

        g_ab = sigmoid(MLP([F_b, N_a, F_b*N_a, |F_b-N_a|, e_a, e_b]))
        Delta_ab = alpha_b * g_ab * U_a(N_a)

    U_a is an INDEPENDENT source projection (not shared with B0) and
    alpha_b is zero-initialized: strict zero-correction start. The scorer
    last layer is small-normal initialized (g ~ 0.5), matching R2-F.
    """

    def __init__(self, factor_dim: int, type_dim: int = 8, gate_hidden: int = 64,
                 activation: str = "gelu", final_std: float = 1.0e-3) -> None:
        super().__init__(factor_dim, type_dim)
        d = self.factor_dim
        self.source_proj = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(3)]
        )
        self.scorer = nn.Sequential(
            nn.Linear(4 * d + 2 * self.type_dim, int(gate_hidden)),
            get_activation(activation),
            nn.Linear(int(gate_hidden), 1),
        )
        nn.init.normal_(self.scorer[-1].weight, std=float(final_std))
        nn.init.zeros_(self.scorer[-1].bias)
        self.alpha = nn.Parameter(torch.zeros(3))  # per target b, init 0

    def forward(self, f_pre: torch.Tensor, n_block: torch.Tensor) -> torch.Tensor:
        # Delta^b = (1/3) sum_a Delta^{a->b} with Delta_ab = alpha_b*g*U_a(N_a)
        # (alpha applied INSIDE the cells, plan §23).
        deltas_per_target: list[torch.Tensor] = []
        for b in range(3):
            cells = [
                self.alpha[b] * cell for cell in self._cells_for_target(f_pre, n_block, b)
            ]
            acc = cells[0]
            for cell in cells[1:]:
                acc = acc + cell
            deltas_per_target.append(acc / 3.0)
        return torch.stack(deltas_per_target, dim=1)

    def _cells_for_target(
        self, f_pre: torch.Tensor, n_block: torch.Tensor, b: int
    ) -> list[torch.Tensor]:
        """Per-cell messages g^{a->b} * U_a(N^a) BEFORE the alpha scale."""
        num_nodes = int(f_pre.size(0))
        src_t = self.src_type_emb.weight
        tgt_t = self.tgt_type_emb.weight
        tgt_emb = tgt_t[b].unsqueeze(0).expand(num_nodes, -1)
        cells: list[torch.Tensor] = []
        for a in range(3):
            u = torch.cat(
                [
                    f_pre[:, b],
                    n_block[:, a],
                    f_pre[:, b] * n_block[:, a],
                    (f_pre[:, b] - n_block[:, a]).abs(),
                    src_t[a].unsqueeze(0).expand(num_nodes, -1),
                    tgt_emb,
                ],
                dim=-1,
            )
            g = torch.sigmoid(self.scorer(u))  # [N, 1]
            cells.append(g * self.source_proj[a](n_block[:, a]))
        return cells

    def cell_deltas(
        self, f_pre: torch.Tensor, n_block: torch.Tensor
    ) -> list[list[torch.Tensor]]:
        """Delta^{a->b} = alpha_b * g^{a->b} * U_a(N^a) per cell (plan §27)."""
        return [
            [self.alpha[b] * cell for cell in self._cells_for_target(f_pre, n_block, b)]
            for b in range(3)
        ]


class ConcatVectorAdapter(_CellAdapterBase):
    """D2 CONCAT-VECTOR (plan §23): capacity control without explicit
    product/difference. Shared MLP: Linear(2d+2td, 128) GELU Linear(128, d),
    last layer zero-init. Delta_b = (1/3) sum_a Delta_ab."""

    def __init__(self, factor_dim: int, type_dim: int = 8, hidden: int = 128,
                 activation: str = "gelu") -> None:
        super().__init__(factor_dim, type_dim)
        d = self.factor_dim
        self.net = nn.Sequential(
            nn.Linear(2 * d + 2 * self.type_dim, int(hidden)),
            get_activation(activation),
            nn.Linear(int(hidden), d),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _cells_for_target(
        self, f_pre: torch.Tensor, n_block: torch.Tensor, b: int
    ) -> list[torch.Tensor]:
        return [
            self.net(self.cell_input(f_pre, n_block, a, b, use_int=False)) for a in range(3)
        ]

    def forward(self, f_pre: torch.Tensor, n_block: torch.Tensor) -> torch.Tensor:
        deltas_per_target: list[torch.Tensor] = []
        for b in range(3):
            cells = self._cells_for_target(f_pre, n_block, b)
            acc = cells[0]
            for cell in cells[1:]:
                acc = acc + cell
            deltas_per_target.append(acc / 3.0)
        return torch.stack(deltas_per_target, dim=1)

    def cell_deltas(
        self, f_pre: torch.Tensor, n_block: torch.Tensor
    ) -> list[list[torch.Tensor]]:
        return [self._cells_for_target(f_pre, n_block, b) for b in range(3)]


class ProdDiffVectorAdapter(_CellAdapterBase):
    """D3 PRODDIFF-VECTOR (plan §23, core candidate): only the R2-0C-
    supported interaction [F_b*N_a, |F_b-N_a|] + type embeddings, through
    an MLP with EXACTLY the same shape as D2 (parameter-matched control)."""

    def __init__(self, factor_dim: int, type_dim: int = 8, hidden: int = 128,
                 activation: str = "gelu") -> None:
        super().__init__(factor_dim, type_dim)
        d = self.factor_dim
        self.net = nn.Sequential(
            nn.Linear(2 * d + 2 * self.type_dim, int(hidden)),
            get_activation(activation),
            nn.Linear(int(hidden), d),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _cells_for_target(
        self, f_pre: torch.Tensor, n_block: torch.Tensor, b: int
    ) -> list[torch.Tensor]:
        return [
            self.net(self.cell_input(f_pre, n_block, a, b, use_int=True)) for a in range(3)
        ]

    def forward(self, f_pre: torch.Tensor, n_block: torch.Tensor) -> torch.Tensor:
        deltas_per_target: list[torch.Tensor] = []
        for b in range(3):
            cells = self._cells_for_target(f_pre, n_block, b)
            acc = cells[0]
            for cell in cells[1:]:
                acc = acc + cell
            deltas_per_target.append(acc / 3.0)
        return torch.stack(deltas_per_target, dim=1)

    def cell_deltas(
        self, f_pre: torch.Tensor, n_block: torch.Tensor
    ) -> list[list[torch.Tensor]]:
        return [self._cells_for_target(f_pre, n_block, b) for b in range(3)]


class FiLMVectorAdapter(_CellAdapterBase):
    """D4 FiLM-VECTOR (plan §23): target-conditioned feature-wise modulation.

        [delta_gamma_ab, beta_ab] = MLP([F_b, N_a, F_b*N_a, |F_b-N_a|, types])
        Delta_ab = delta_gamma_ab * U_a(N_a) + beta_ab

    Last layer zero-init (delta_gamma = beta = 0). NOT a replacement of the
    B0 message — a zero-init correction (plan §23)."""

    def __init__(self, factor_dim: int, type_dim: int = 8, hidden: int = 128,
                 activation: str = "gelu") -> None:
        super().__init__(factor_dim, type_dim)
        d = self.factor_dim
        self.source_proj = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(3)]
        )
        self.net = nn.Sequential(
            nn.Linear(4 * d + 2 * self.type_dim, int(hidden)),
            get_activation(activation),
            nn.Linear(int(hidden), 2 * d),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _cells_for_target(
        self, f_pre: torch.Tensor, n_block: torch.Tensor, b: int
    ) -> list[torch.Tensor]:
        num_nodes = int(f_pre.size(0))
        src_t = self.src_type_emb.weight
        tgt_t = self.tgt_type_emb.weight
        tgt_emb = tgt_t[b].unsqueeze(0).expand(num_nodes, -1)
        cells: list[torch.Tensor] = []
        for a in range(3):
            u = torch.cat(
                [
                    f_pre[:, b],
                    n_block[:, a],
                    f_pre[:, b] * n_block[:, a],
                    (f_pre[:, b] - n_block[:, a]).abs(),
                    src_t[a].unsqueeze(0).expand(num_nodes, -1),
                    tgt_emb,
                ],
                dim=-1,
            )
            out = self.net(u)  # [N, 2d]
            delta_gamma, beta = out.chunk(2, dim=-1)
            v_a = self.source_proj[a](n_block[:, a])  # [N, d]
            cells.append(delta_gamma * v_a + beta)
        return cells

    def forward(self, f_pre: torch.Tensor, n_block: torch.Tensor) -> torch.Tensor:
        deltas_per_target: list[torch.Tensor] = []
        for b in range(3):
            cells = self._cells_for_target(f_pre, n_block, b)
            acc = cells[0]
            for cell in cells[1:]:
                acc = acc + cell
            deltas_per_target.append(acc / 3.0)
        return torch.stack(deltas_per_target, dim=1)

    def cell_deltas(
        self, f_pre: torch.Tensor, n_block: torch.Tensor
    ) -> list[list[torch.Tensor]]:
        return [self._cells_for_target(f_pre, n_block, b) for b in range(3)]


def build_adapter(name: str, factor_dim: int, **kwargs) -> _CellAdapterBase:
    if name == "HEAD":
        raise ValueError("HEAD is the no-adapter control, not an adapter")
    if name == "D1":
        return ScalarAdapter(factor_dim, **kwargs)
    if name == "D2":
        return ConcatVectorAdapter(factor_dim, **kwargs)
    if name == "D3":
        return ProdDiffVectorAdapter(factor_dim, **kwargs)
    if name == "D4":
        return FiLMVectorAdapter(factor_dim, **kwargs)
    raise ValueError(f"unknown adapter {name!r}")
