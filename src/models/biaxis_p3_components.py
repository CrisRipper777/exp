"""P3 Bi-Axis operator layer: Factor-Relation-specific graph transformation
(plan §5-§9).

    T_fk = W0 + A_f + B_k + C_fk

- W0 : shared P2 graph operator (owned by the model, passed in per call)
- A_f: semantic-factor main effect      [F, d, d]   zero init
- B_k: structural-relation main effect  [K, d, d]   zero init
- C_fk: pair-specific residual          [F, K, d, d] zero init

Modes (plan §6):
    shared            T = W0
    factor            T = W0 + A_f
    relation          T = W0 + B_k
    additive          T = W0 + A_f + B_k
    full_interaction  T = W0 + A_f + B_k + C_fk

Discipline:
    - bias=False everywhere; every residual zero-initialized (step 0:
      T_fk = W0 for ALL modes, plan §7)
    - only the parameters a mode uses are allocated (clean param accounting,
      plan §8: OF 3d^2 / OR 4d^2 / OADD 7d^2 / OFR 19d^2 extra)
    - never materialize [N, F, K, d, d] or a persistent transformed
      [N, F, K, d] copy: shared/factor terms aggregate first over k
      (transient [N, F, d]), relation term loops K (transient [N, F, d]),
      pair term loops F x K cells (transient [N, d]), all accumulated
      immediately (plan §9)
    - no internal regularizer: regularization hooks exist but are applied
      by the model with config weights (default 0, plan §7)
"""

from __future__ import annotations

import torch
import torch.nn as nn

_EPS = 1e-8


class FullResidualFactorRelationOperator(nn.Module):
    """Cell-indexed graph operator T_fk with residual decomposition.

    ``forward(g_perm, gamma_graph, w0)`` applies

        m_i^f = sum_k Gamma_ifk * T_fk(g_ik^f)

    without ever materializing a node-wise operator tensor. ``w0`` is the
    shared P2 Linear (bias=False); it remains the model's single shared
    operator parameter.
    """

    MODES = ("shared", "factor", "relation", "additive", "full_interaction")

    def __init__(
        self,
        num_factors: int,
        num_relations: int,
        dim: int,
        mode: str = "shared",
    ) -> None:
        super().__init__()
        assert mode in self.MODES, f"unknown operator mode {mode!r}"
        self.mode = str(mode)
        self.num_factors = int(num_factors)
        self.num_relations = int(num_relations)
        self.dim = int(dim)
        self._use_factor = self.mode in ("factor", "additive", "full_interaction")
        self._use_relation = self.mode in ("relation", "additive", "full_interaction")
        self._use_pair = self.mode == "full_interaction"

        # All residuals zero-initialized (plan §7: step 0 T_fk = W0).
        if self._use_factor:
            self.A = nn.Parameter(torch.zeros(self.num_factors, self.dim, self.dim))
        if self._use_relation:
            self.B = nn.Parameter(torch.zeros(self.num_relations, self.dim, self.dim))
        if self._use_pair:
            self.C = nn.Parameter(torch.zeros(self.num_factors, self.num_relations, self.dim, self.dim))

    def extra_residual_params(self) -> int:
        """Extra residual parameters vs the shared W0-only operator."""
        return sum(int(p.numel()) for p in self.parameters())

    def _cell_transform(self, x: torch.Tensor, f: int, k: int, w0: nn.Linear) -> torch.Tensor:
        """T_fk(x) for a single cell (diagnostics path; [N, d] in/out)."""
        out = w0(x)
        if self._use_factor:
            out = out + x @ self.A[f].t()
        if self._use_relation:
            out = out + x @ self.B[k].t()
        if self._use_pair:
            out = out + x @ self.C[f, k].t()
        return out

    def forward(
        self,
        g_perm: torch.Tensor,
        gamma_graph: torch.Tensor,
        w0: nn.Linear,
    ) -> torch.Tensor:
        """g_perm: [N, F, K, d]; gamma_graph: [N, F, K] (gamma[..., 1:]).
        Returns m [N, F, d] = sum_k Gamma_ifk T_fk(g_ik^f)."""
        num_nodes, num_factors, num_relations, dim = g_perm.shape

        # --- shared term, aggregate-first (plan §9.1): the SAME op order as
        # P2's g_mix -> W0 path, so with zero residuals all modes are
        # bitwise identical to the P2 shared operator. ---
        agg = (gamma_graph.unsqueeze(-1) * g_perm).sum(dim=2)  # [N, F, d]
        m = w0(agg.reshape(num_nodes * num_factors, dim)).reshape(num_nodes, num_factors, dim)

        # --- factor term (plan §9.2): A_f is k-independent, aggregate first.
        if self._use_factor:
            for f in range(num_factors):
                m[:, f] = m[:, f] + agg[:, f] @ self.A[f].t()

        # --- relation term (plan §9.3): loop K, transient [N, F, d]. -------
        if self._use_relation:
            for k in range(num_relations):
                t = g_perm[:, :, k].reshape(num_nodes * num_factors, dim) @ self.B[k].t()
                m = m + gamma_graph[:, :, k : k + 1] * t.reshape(num_nodes, num_factors, dim)

        # --- pair term (plan §9.4): loop F x K cells, transient [N, d]. ----
        if self._use_pair:
            for f in range(num_factors):
                for k in range(num_relations):
                    t = g_perm[:, f, k] @ self.C[f, k].t()  # [N, d]
                    m[:, f] = m[:, f] + gamma_graph[:, f, k : k + 1] * t

        return m

    # ------------------------------------------------------------------
    # Residual regularization hooks (plan §7: weights 0 by default)
    # ------------------------------------------------------------------

    def reg_operator(self) -> torch.Tensor:
        """Squared Frobenius norm of the MAIN-effect residuals (A, B)."""
        loss = torch.zeros((), dtype=torch.float32)
        if self._use_factor:
            loss = loss + self.A.square().sum()
        if self._use_relation:
            loss = loss + self.B.square().sum()
        return loss

    def reg_interaction(self) -> torch.Tensor:
        """Squared Frobenius norm of the pair-specific residual C."""
        if not self._use_pair:
            return torch.zeros((), dtype=torch.float32)
        return self.C.square().sum()

    # ------------------------------------------------------------------
    # Mechanism diagnostics (plan §13) — best checkpoint, no_grad
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_diagnostics(
        self,
        g_perm: torch.Tensor,
        gamma_graph: torch.Tensor,
        w0: nn.Linear,
    ) -> dict:
        """Operator-side diagnostics (plan §13). JSON-safe floats only.
        Never touches labels; does not modify model state.

        g_perm [N, F, K, d], gamma_graph [N, F, K] (gamma[..., 1:]).
        """
        num_nodes, num_factors, num_relations, dim = g_perm.shape
        w0_norm = float(w0.weight.norm(p="fro").item())

        # --- residual Frobenius norms, relative to ||W0|| (plan §13.1) ----
        r_a: list[float] = []
        r_b: list[float] = []
        r_c: list[list[float]] = []
        if self._use_factor:
            r_a = [(float(self.A[f].norm(p="fro").item()) / (w0_norm + _EPS)) for f in range(num_factors)]
        if self._use_relation:
            r_b = [(float(self.B[k].norm(p="fro").item()) / (w0_norm + _EPS)) for k in range(num_relations)]
        if self._use_pair:
            r_c = [
                [(float(self.C[f, k].norm(p="fro").item()) / (w0_norm + _EPS)) for k in range(num_relations)]
                for f in range(num_factors)
            ]

        # --- usage (plan §13.2): u_fk = mean_i Gamma_ifk -------------------
        usage = gamma_graph.mean(dim=0)  # [F, K]
        u_fk = usage.cpu().tolist()

        # --- materialize the per-cell operators T_fk (small: [F, K, d, d]) -
        t_cells = w0.weight.detach().clone().unsqueeze(0).unsqueeze(0).expand(
            num_factors, num_relations, dim, dim
        ).clone()
        if self._use_factor:
            t_cells = t_cells + self.A.detach().unsqueeze(1)
        if self._use_relation:
            t_cells = t_cells + self.B.detach().unsqueeze(0)
        if self._use_pair:
            t_cells = t_cells + self.C.detach()

        # --- usage-weighted pair strength (plan §13.2) --------------------
        s_pair = 0.0
        if self._use_pair:
            pair_norms = self.C.detach().norm(p="fro", dim=(2, 3)) / (w0_norm + _EPS)  # [F, K]
            s_pair = float((usage * pair_norms).sum().item())

        # --- operator distances (plan §13.3) -------------------------------
        # normalized Frobenius distance + flattened cosine; means over pairs.
        def _pair_stats(mats: torch.Tensor) -> dict:
            """mats: [P, d, d] -> mean normalized Frobenius distance + cosine."""
            p = int(mats.size(0))
            frob, cos = [], []
            for i in range(p):
                for j in range(i + 1, p):
                    diff = (mats[i] - mats[j]).norm(p="fro")
                    frob.append(float((diff / (mats[i].norm(p="fro") + mats[j].norm(p="fro") + _EPS)).item()))
                    cos.append(float(torch.nn.functional.cosine_similarity(
                        mats[i].flatten(), mats[j].flatten(), dim=0
                    ).item()))
            mean = lambda vals: sum(vals) / len(vals) if vals else 0.0  # noqa: E731
            return {"norm_frob_dist": mean(frob), "flattened_cosine": mean(cos)}

        same_relation = [_pair_stats(t_cells[:, k]) for k in range(num_relations)]  # across factors
        same_factor = [_pair_stats(t_cells[f]) for f in range(num_factors)]  # across relations

        # --- message-level effect (plan §13.4): usage-weighted mean of
        # delta_ifk = ||T_fk(g) - W0 g||_2 / (||W0 g||_2 + eps) -------------
        delta_weighted_sum = 0.0
        gamma_total = 0.0
        for f in range(num_factors):
            for k in range(num_relations):
                g_cell = g_perm[:, f, k]  # [N, d]
                base = w0(g_cell)
                trans = self._cell_transform(g_cell, f, k, w0)
                delta = (trans - base).norm(dim=-1) / (base.norm(dim=-1) + _EPS)  # [N]
                w = gamma_graph[:, f, k]  # [N]
                delta_weighted_sum += float((w * delta).sum().item())
                gamma_total += float(w.sum().item())
        message_deviation = delta_weighted_sum / (gamma_total + _EPS)

        return {
            "mode": self.mode,
            "w0_norm": w0_norm,
            "residual_norms": {
                "factor": r_a,            # [F]  ||A_f|| / ||W0||
                "relation": r_b,          # [K]  ||B_k|| / ||W0||
                "pair": r_c,              # [F, K] ||C_fk|| / ||W0||
            },
            "usage": u_fk,                # [F, K] mean_i Gamma_ifk
            "pair_strength": s_pair,      # sum_fk u_fk ||C_fk|| / ||W0||
            "operator_distance": {
                "same_relation_across_factors": same_relation,   # per k
                "same_factor_across_relations": same_factor,     # per f
            },
            "message_deviation_usage_weighted": message_deviation,
            "extra_residual_params": self.extra_residual_params(),
            "interaction": _interaction_diagnostic(t_cells, w0_norm, usage),
        }


def _interaction_diagnostic(
    t_cells: torch.Tensor,
    w0_norm: float,
    usage: torch.Tensor,
) -> dict:
    """Double-centered interaction of the EFFECTIVE cell operators
    (review §12):

        I_fk = T_fk - Tbar_f· - Tbar_·k + Tbar_··

    Defined on the final effective operators, invariant to how A/B/C absorb
    each other (unlike ||C_fk||). Reports ||I_fk||_F / ||W0||_F per cell plus
    TWO summary strengths (review-2 §2):
      - usage_weighted_strength (raw): sum_fk u_fk ||I_fk|| / ||W0||
        (total interaction exposure; u_fk sums to the total graph mass < 1)
      - usage_weighted_average (normalized): sum_fk u_fk ||I_fk|| /
        ((sum_fk u_fk) ||W0|| + eps) — mean interaction strength over
        actually-used cells, comparable across datasets.
    """
    tbar_f = t_cells.mean(dim=1, keepdim=True)  # [F, 1, d, d]
    tbar_k = t_cells.mean(dim=0, keepdim=True)  # [1, K, d, d]
    tbar_all = t_cells.mean(dim=(0, 1), keepdim=True)  # [1, 1, d, d]
    interaction = t_cells - tbar_f - tbar_k + tbar_all  # [F, K, d, d]
    i_norms = interaction.norm(p="fro", dim=(2, 3)) / (w0_norm + _EPS)  # [F, K]
    raw = (usage * i_norms).sum()
    total = usage.sum()
    return {
        "norms": [[float(v) for v in row] for row in i_norms.cpu().tolist()],
        "usage_weighted_strength": float(raw.item()),
        "usage_weighted_average": float((raw / (total + _EPS)).item()),
    }


class LowRankFactorRelationOperator(nn.Module):
    """Parameter-matched low-rank cell operator (P3-B, plan §14-§19):

        T_fk(x) = W0 x + U [ c_fk * (V^T x) ]

    with the rank-r cell coefficient:

        lowrank_add:          c_fk = a_f + b_k
        lowrank_interaction:  c_fk = a_f + b_k + a_f * b_k

    Parameters: U [d, r], V [d, r] (Xavier), a [F, r], b [K, r] (zeros).
    LR-ADD and LR-INT share the EXACT same parameter set — the only
    difference is the explicit factor x relation interaction term a*b
    (plan §17: the cleanest parameter-matched interaction test).

    Step 0: a=b=0 -> c=0 -> T_fk = W0 exactly (plan §18). a/b receive
    gradient immediately; U/V start with zero residual coefficient and
    receive gradient once c != 0 (expected, not a bug).

    Memory discipline: aggregation happens in the rank-r latent space
    (plan §38): z_f = sum_k Gamma_ifk * c_fk * (V^T g_ik^f), then a single
    U-projection. Transients are [N, F, r]; no [N, F, K, d, d] anywhere.
    """

    MODES = ("lowrank_add", "lowrank_interaction")

    def __init__(
        self,
        num_factors: int,
        num_relations: int,
        dim: int,
        rank: int = 16,
        mode: str = "lowrank_interaction",
    ) -> None:
        super().__init__()
        assert mode in self.MODES, f"unknown low-rank mode {mode!r}"
        self.mode = str(mode)
        self.num_factors = int(num_factors)
        self.num_relations = int(num_relations)
        self.dim = int(dim)
        self.rank = int(rank)
        self._use_interaction = self.mode == "lowrank_interaction"

        # U, V: Xavier (plan §18); a, b: zeros.
        self.U = nn.Parameter(torch.empty(self.dim, self.rank))
        self.V = nn.Parameter(torch.empty(self.dim, self.rank))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        self.a = nn.Parameter(torch.zeros(self.num_factors, self.rank))
        self.b = nn.Parameter(torch.zeros(self.num_relations, self.rank))

    def extra_residual_params(self) -> int:
        """LR-ADD and LR-INT share exactly these params (plan §17/§29.7)."""
        return sum(int(p.numel()) for p in self.parameters())

    def _cell_coefficients(self) -> torch.Tensor:
        """c_fk [F, K, r] (plan §15/§16)."""
        c = self.a.unsqueeze(1) + self.b.unsqueeze(0)
        if self._use_interaction:
            c = c + self.a.unsqueeze(1) * self.b.unsqueeze(0)
        return c

    def _cell_transform(self, x: torch.Tensor, f: int, k: int, w0: nn.Linear) -> torch.Tensor:
        """T_fk(x) for a single cell (diagnostics path; [N, d] in/out)."""
        out = w0(x)
        c = self._cell_coefficients()[f, k]  # [r]
        latent = (x @ self.V) * c  # [N, r]
        return out + latent @ self.U.t()

    def forward(
        self,
        g_perm: torch.Tensor,
        gamma_graph: torch.Tensor,
        w0: nn.Linear,
    ) -> torch.Tensor:
        """g_perm: [N, F, K, d]; gamma_graph: [N, F, K]. Returns m [N, F, d].

        Aggregate-first in the rank-r latent space (U is linear):
            sum_k Gamma * U[c ⊙ V^T g_k] = U( sum_k Gamma * c ⊙ V^T g_k )
        """
        num_nodes, num_factors, num_relations, dim = g_perm.shape

        agg = (gamma_graph.unsqueeze(-1) * g_perm).sum(dim=2)  # [N, F, d]
        m = w0(agg.reshape(num_nodes * num_factors, dim)).reshape(num_nodes, num_factors, dim)

        c = self._cell_coefficients()  # [F, K, r]
        z = torch.zeros(num_nodes, num_factors, self.rank, dtype=g_perm.dtype, device=g_perm.device)
        for k in range(num_relations):
            v = g_perm[:, :, k].reshape(num_nodes * num_factors, dim) @ self.V  # [N*F, r]
            v = v.reshape(num_nodes, num_factors, self.rank)
            for f in range(num_factors):
                z[:, f] = z[:, f] + gamma_graph[:, f, k : k + 1] * (c[f, k] * v[:, f])
        m = m + (z.reshape(num_nodes * num_factors, self.rank) @ self.U.t()).reshape(
            num_nodes, num_factors, dim
        )
        return m

    # ------------------------------------------------------------------
    # Regularization hooks (same interface as the full operator)
    # ------------------------------------------------------------------

    def reg_operator(self) -> torch.Tensor:
        return self.a.square().sum() + self.b.square().sum()

    def reg_interaction(self) -> torch.Tensor:
        return (self.a.unsqueeze(1) * self.b.unsqueeze(0)).square().sum()

    # ------------------------------------------------------------------
    # Mechanism diagnostics (plan §13 adapted to the low-rank cell op)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_diagnostics(
        self,
        g_perm: torch.Tensor,
        gamma_graph: torch.Tensor,
        w0: nn.Linear,
    ) -> dict:
        """Same diagnostic interface as the full operator: ||W0||, per-cell
        residual strength (here: ||c_fk|| as the cell-modulation strength),
        usage-weighted pair strength, operator distances (T_fk materialized),
        usage-weighted message deviation, param count."""
        num_nodes, num_factors, num_relations, dim = g_perm.shape
        w0_norm = float(w0.weight.norm(p="fro").item())
        c = self._cell_coefficients()  # [F, K, r]

        r_c_tensor = c.norm(dim=-1) / (w0_norm + _EPS)  # [F, K]
        r_c = [[float(v) for v in row] for row in r_c_tensor.cpu().tolist()]
        usage = gamma_graph.mean(dim=0)  # [F, K]
        pair_strength = float((usage * r_c_tensor).sum().item())

        # materialize per-cell operators T_fk (small) for distances.
        # The residual matrix is U diag(c) V^T = sum_r c_r (u_r v_r^T):
        residual = torch.einsum("dr,fkr,er->fkde", self.U.detach(), c.detach(), self.V.detach())
        t_cells = w0.weight.detach().clone().unsqueeze(0).unsqueeze(0).expand(
            num_factors, num_relations, dim, dim
        ).clone() + residual

        def _pair_stats(mats: torch.Tensor) -> dict:
            p = int(mats.size(0))
            frob, cos = [], []
            for i in range(p):
                for j in range(i + 1, p):
                    diff = (mats[i] - mats[j]).norm(p="fro")
                    frob.append(float((diff / (mats[i].norm(p="fro") + mats[j].norm(p="fro") + _EPS)).item()))
                    cos.append(float(torch.nn.functional.cosine_similarity(
                        mats[i].flatten(), mats[j].flatten(), dim=0
                    ).item()))
            mean = lambda vals: sum(vals) / len(vals) if vals else 0.0  # noqa: E731
            return {"norm_frob_dist": mean(frob), "flattened_cosine": mean(cos)}

        same_relation = [_pair_stats(t_cells[:, k]) for k in range(num_relations)]
        same_factor = [_pair_stats(t_cells[f]) for f in range(num_factors)]

        delta_weighted_sum = 0.0
        gamma_total = 0.0
        for f in range(num_factors):
            for k in range(num_relations):
                g_cell = g_perm[:, f, k]
                base = w0(g_cell)
                trans = self._cell_transform(g_cell, f, k, w0)
                delta = (trans - base).norm(dim=-1) / (base.norm(dim=-1) + _EPS)
                w = gamma_graph[:, f, k]
                delta_weighted_sum += float((w * delta).sum().item())
                gamma_total += float(w.sum().item())

        return {
            "mode": self.mode,
            "rank": self.rank,
            "w0_norm": w0_norm,
            "residual_norms": {
                "factor": [float(self.a[f].norm().item() / (w0_norm + _EPS)) for f in range(num_factors)],
                "relation": [float(self.b[k].norm().item() / (w0_norm + _EPS)) for k in range(num_relations)],
                "pair": r_c,
            },
            "usage": usage.cpu().tolist(),
            "pair_strength": pair_strength,
            "operator_distance": {
                "same_relation_across_factors": same_relation,
                "same_factor_across_relations": same_factor,
            },
            "message_deviation_usage_weighted": float(
                delta_weighted_sum / (gamma_total + _EPS)
            ),
            "extra_residual_params": self.extra_residual_params(),
            "interaction": _interaction_diagnostic(t_cells, w0_norm, usage),
        }


class BasisCellOperator(nn.Module):
    """Basis-decomposed Cell Operator (review §20):

        T_fk = W0 + sum_{b=1}^{B} c_fkb * V_b

    with FULL-matrix basis V_b in R^{d x d} (Xavier) and per-cell
    coefficients c in R^{F x K x B} (zeros). Unlike the low-rank operator
    (shared rank-1 bases u_b v_b^T), each V_b is an arbitrary full-rank
    matrix, so the parameter count scales as B*d^2 + F*K*B and interpolates
    the capacity axis: O0 -> B4 (65.6K) -> OADD (115K) -> B8 (131.2K) ->
    B16 (262.3K) -> OFR (311K).

    Step 0: c=0 -> T_fk = W0 exactly (same zero-init discipline, plan §7).
    c receives gradient immediately (through V_b x); V_b's residual
    coefficient starts at 0 and activates once c != 0 (plan §18 dynamics,
    expected).

    Memory discipline: the cell operator matrices T_fk are materialized ONCE
    per forward as [F, K, d, d] (786 KB at d=128 — no node dimension), then
    applied per (f, k) cell with transient [N, d] tensors (same cost profile
    as the full operator's pair term).
    """

    MODES = ("basis",)

    def __init__(
        self,
        num_factors: int,
        num_relations: int,
        dim: int,
        num_bases: int = 8,
        mode: str = "basis",
    ) -> None:
        super().__init__()
        assert mode in self.MODES, f"unknown basis mode {mode!r}"
        self.mode = str(mode)
        self.num_factors = int(num_factors)
        self.num_relations = int(num_relations)
        self.dim = int(dim)
        self.num_bases = int(num_bases)

        # Per-basis Xavier (review §9): a 3-D [B, d, d] Xavier call uses the
        # WHOLE tensor's fan-in/out (scale ~0.018 at d=128), ~8x smaller than
        # the per-matrix bound sqrt(6/(d+d)) ~= 0.153. Each V_b must be
        # initialized as an independent [d, d] linear map.
        self.V = nn.Parameter(torch.empty(self.num_bases, self.dim, self.dim))
        for b in range(self.num_bases):
            nn.init.xavier_uniform_(self.V[b])
        self.c = nn.Parameter(torch.zeros(self.num_factors, self.num_relations, self.num_bases))

    def extra_residual_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def _cell_residual(self) -> torch.Tensor:
        """Effective per-cell residual matrices: [F, K, d, d] =
        sum_b c_fkb * V_b (small, no node dimension)."""
        return torch.einsum("bde,fkb->fkde", self.V, self.c)

    def _cell_transform(self, x: torch.Tensor, f: int, k: int, w0: nn.Linear) -> torch.Tensor:
        out = w0(x)
        resid = torch.einsum("bde,b->de", self.V, self.c[f, k])
        return out + x @ resid.t()

    def forward(
        self,
        g_perm: torch.Tensor,
        gamma_graph: torch.Tensor,
        w0: nn.Linear,
    ) -> torch.Tensor:
        """g_perm: [N, F, K, d]; gamma_graph: [N, F, K]. Returns m [N, F, d]."""
        num_nodes, num_factors, num_relations, dim = g_perm.shape

        agg = (gamma_graph.unsqueeze(-1) * g_perm).sum(dim=2)  # [N, F, d]
        m = w0(agg.reshape(num_nodes * num_factors, dim)).reshape(num_nodes, num_factors, dim)

        resid = self._cell_residual()  # [F, K, d, d], materialized once
        for f in range(num_factors):
            for k in range(num_relations):
                t = g_perm[:, f, k] @ resid[f, k].t()  # [N, d]
                m[:, f] = m[:, f] + gamma_graph[:, f, k : k + 1] * t
        return m

    # ------------------------------------------------------------------
    # Regularization hooks (same interface as the other operators)
    # ------------------------------------------------------------------

    def reg_operator(self) -> torch.Tensor:
        return self.c.square().sum()

    def reg_interaction(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Mechanism diagnostics (same interface; review §20)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_diagnostics(
        self,
        g_perm: torch.Tensor,
        gamma_graph: torch.Tensor,
        w0: nn.Linear,
    ) -> dict:
        num_nodes, num_factors, num_relations, dim = g_perm.shape
        w0_norm = float(w0.weight.norm(p="fro").item())
        resid = self._cell_residual()  # [F, K, d, d]
        t_cells = w0.weight.detach().clone().unsqueeze(0).unsqueeze(0).expand(
            num_factors, num_relations, dim, dim
        ).clone() + resid

        r_c_tensor = resid.norm(p="fro", dim=(2, 3)) / (w0_norm + _EPS)  # [F, K]
        r_c = [[float(v) for v in row] for row in r_c_tensor.cpu().tolist()]
        usage = gamma_graph.mean(dim=0)  # [F, K]
        pair_strength = float((usage * r_c_tensor).sum().item())

        def _pair_stats(mats: torch.Tensor) -> dict:
            p = int(mats.size(0))
            frob, cos = [], []
            for i in range(p):
                for j in range(i + 1, p):
                    diff = (mats[i] - mats[j]).norm(p="fro")
                    frob.append(float((diff / (mats[i].norm(p="fro") + mats[j].norm(p="fro") + _EPS)).item()))
                    cos.append(float(torch.nn.functional.cosine_similarity(
                        mats[i].flatten(), mats[j].flatten(), dim=0
                    ).item()))
            mean = lambda vals: sum(vals) / len(vals) if vals else 0.0  # noqa: E731
            return {"norm_frob_dist": mean(frob), "flattened_cosine": mean(cos)}

        same_relation = [_pair_stats(t_cells[:, k]) for k in range(num_relations)]
        same_factor = [_pair_stats(t_cells[f]) for f in range(num_factors)]

        delta_weighted_sum = 0.0
        gamma_total = 0.0
        for f in range(num_factors):
            for k in range(num_relations):
                g_cell = g_perm[:, f, k]
                base = w0(g_cell)
                trans = self._cell_transform(g_cell, f, k, w0)
                delta = (trans - base).norm(dim=-1) / (base.norm(dim=-1) + _EPS)
                w = gamma_graph[:, f, k]
                delta_weighted_sum += float((w * delta).sum().item())
                gamma_total += float(w.sum().item())

        return {
            "mode": self.mode,
            "num_bases": self.num_bases,
            "w0_norm": w0_norm,
            "residual_norms": {
                "factor": [],
                "relation": [],
                "pair": r_c,
            },
            "usage": usage.cpu().tolist(),
            "pair_strength": pair_strength,
            "operator_distance": {
                "same_relation_across_factors": same_relation,
                "same_factor_across_relations": same_factor,
            },
            "message_deviation_usage_weighted": float(
                delta_weighted_sum / (gamma_total + _EPS)
            ),
            "extra_residual_params": self.extra_residual_params(),
            "interaction": _interaction_diagnostic(t_cells, w0_norm, usage),
        }


class DirectCellOperator(nn.Module):
    """Direct Cell Operator (review-2 §10-§12):

        T_fk = W0 + D_fk,   D in R^{F x K x d x d}, zero-init.

    No A_f / B_k / C_fk decomposition. The effective function class is
    EXACTLY that of OFR (review-2 §11: D_fk = A_f+B_k+C_fk in one direction;
    A=B=0, C=D in the other), while residual parameters drop from
    19*d^2 = 311,296 to 12*d^2 = 196,608 (−36.8%) with no A/B/C
    identifiability ambiguity.

    Step 0: D=0 -> T_fk = W0 exactly (zero-init discipline preserved).
    The interaction diagnostic still double-centers the effective T_fk.

    Memory discipline: per-cell loop with transient [N, d] tensors — the
    same cost profile as the full operator's pair term.
    """

    MODES = ("direct",)

    def __init__(
        self,
        num_factors: int,
        num_relations: int,
        dim: int,
        mode: str = "direct",
    ) -> None:
        super().__init__()
        assert mode in self.MODES, f"unknown direct mode {mode!r}"
        self.mode = str(mode)
        self.num_factors = int(num_factors)
        self.num_relations = int(num_relations)
        self.dim = int(dim)
        self.D = nn.Parameter(torch.zeros(self.num_factors, self.num_relations, self.dim, self.dim))

    def extra_residual_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def _cell_transform(self, x: torch.Tensor, f: int, k: int, w0: nn.Linear) -> torch.Tensor:
        return w0(x) + x @ self.D[f, k].t()

    def forward(
        self,
        g_perm: torch.Tensor,
        gamma_graph: torch.Tensor,
        w0: nn.Linear,
    ) -> torch.Tensor:
        """g_perm: [N, F, K, d]; gamma_graph: [N, F, K]. Returns m [N, F, d]."""
        num_nodes, num_factors, num_relations, dim = g_perm.shape

        agg = (gamma_graph.unsqueeze(-1) * g_perm).sum(dim=2)  # [N, F, d]
        m = w0(agg.reshape(num_nodes * num_factors, dim)).reshape(num_nodes, num_factors, dim)

        for f in range(num_factors):
            for k in range(num_relations):
                t = g_perm[:, f, k] @ self.D[f, k].t()  # [N, d]
                m[:, f] = m[:, f] + gamma_graph[:, f, k : k + 1] * t
        return m

    # ------------------------------------------------------------------
    # Regularization hooks (same interface as the other operators)
    # ------------------------------------------------------------------

    def reg_operator(self) -> torch.Tensor:
        return self.D.square().sum()

    def reg_interaction(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Mechanism diagnostics (same interface; review-2 §10)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_diagnostics(
        self,
        g_perm: torch.Tensor,
        gamma_graph: torch.Tensor,
        w0: nn.Linear,
    ) -> dict:
        num_nodes, num_factors, num_relations, dim = g_perm.shape
        w0_norm = float(w0.weight.norm(p="fro").item())
        t_cells = w0.weight.detach().clone().unsqueeze(0).unsqueeze(0).expand(
            num_factors, num_relations, dim, dim
        ).clone() + self.D.detach()

        r_c_tensor = self.D.detach().norm(p="fro", dim=(2, 3)) / (w0_norm + _EPS)  # [F, K]
        r_c = [[float(v) for v in row] for row in r_c_tensor.cpu().tolist()]
        usage = gamma_graph.mean(dim=0)  # [F, K]
        pair_strength = float((usage * r_c_tensor).sum().item())

        def _pair_stats(mats: torch.Tensor) -> dict:
            p = int(mats.size(0))
            frob, cos = [], []
            for i in range(p):
                for j in range(i + 1, p):
                    diff = (mats[i] - mats[j]).norm(p="fro")
                    frob.append(float((diff / (mats[i].norm(p="fro") + mats[j].norm(p="fro") + _EPS)).item()))
                    cos.append(float(torch.nn.functional.cosine_similarity(
                        mats[i].flatten(), mats[j].flatten(), dim=0
                    ).item()))
            mean = lambda vals: sum(vals) / len(vals) if vals else 0.0  # noqa: E731
            return {"norm_frob_dist": mean(frob), "flattened_cosine": mean(cos)}

        same_relation = [_pair_stats(t_cells[:, k]) for k in range(num_relations)]
        same_factor = [_pair_stats(t_cells[f]) for f in range(num_factors)]

        delta_weighted_sum = 0.0
        gamma_total = 0.0
        for f in range(num_factors):
            for k in range(num_relations):
                g_cell = g_perm[:, f, k]
                base = w0(g_cell)
                trans = self._cell_transform(g_cell, f, k, w0)
                delta = (trans - base).norm(dim=-1) / (base.norm(dim=-1) + _EPS)
                w = gamma_graph[:, f, k]
                delta_weighted_sum += float((w * delta).sum().item())
                gamma_total += float(w.sum().item())

        return {
            "mode": self.mode,
            "w0_norm": w0_norm,
            "residual_norms": {
                "factor": [],
                "relation": [],
                "pair": r_c,
            },
            "usage": usage.cpu().tolist(),
            "pair_strength": pair_strength,
            "operator_distance": {
                "same_relation_across_factors": same_relation,
                "same_factor_across_relations": same_factor,
            },
            "message_deviation_usage_weighted": float(
                delta_weighted_sum / (gamma_total + _EPS)
            ),
            "extra_residual_params": self.extra_residual_params(),
            "interaction": _interaction_diagnostic(t_cells, w0_norm, usage),
        }
