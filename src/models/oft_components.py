from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# O1 slot order is permanently fixed: 0=C, 1=Pt, 2=Pv.
NUM_SLOTS = 3
SLOT_NAMES = ("c", "pt", "pv")

# O2-A off-diagonal cross ownership pairs, in fixed order:
#   c->pt, c->pv, pt->c, pt->pv, pv->c, pv->pt
# Diagonal pairs (a == b) are deliberately absent: the existing per-slot D_a
# already owns the diagonal path.
CROSS_PAIR_KEYS = tuple(
    f"{SLOT_NAMES[a]}_to_{SLOT_NAMES[b]}"
    for a in range(NUM_SLOTS)
    for b in range(NUM_SLOTS)
    if a != b
)
CROSS_PAIR_TARGET_SLOT = {
    f"{SLOT_NAMES[a]}_to_{SLOT_NAMES[b]}": b
    for a in range(NUM_SLOTS)
    for b in range(NUM_SLOTS)
    if a != b
}


def incoming_mean(X: Tensor, edge_index: Tensor | None, num_nodes: int) -> Tensor:
    """Per-node incoming-neighbor mean: messages flow source -> target, i.e.

        N_i = (1 / |{j : (j, i) in E}|) * sum_j X_j

    with ``edge_index[0] = source`` and ``edge_index[1] = target``. No
    self-loops are ever added, and no GCN-style normalization is applied.
    Nodes with no incoming edges get EXACTLY 0.0 (``0 / clamp_min(1) = 0``),
    so an isolated node's graph delta is strictly zero by construction.

    The aggregation uses one ``index_add_`` over gathered source rows
    (``X[src]``, shape ``[E, S, F]``); it never materializes an
    ``[E, S, S, ...]`` edge-pair tensor.

    Shapes: X ``[N, S, F]``, edge_index ``[2, E]`` -> ``[N, S, F]``.
    """
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros_like(X)
    src, dst = edge_index
    msg = X[src]  # [E, S, F] gathered source rows
    out = torch.zeros_like(X)
    out.index_add_(0, dst, msg)
    deg = torch.zeros(num_nodes, dtype=X.dtype, device=X.device)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=X.dtype))
    return out / deg.clamp_min(1.0).view(num_nodes, 1, 1)


class DiagIDLayer(nn.Module):
    """O1 OFT-DIAG-ID graph propagation layer (functional operator = Identity).

    Applied per factor slot ``a`` independently (diagonal, factor-preserving):

        X_i^a       = V_a H_i^a                 # Linear 128->128, no bias
        N_i^a       = incoming neighbor mean    # mean over in-neighbors j -> i
        delta_i^a   = D_a N_i^a                 # Linear 128->128, no bias
        H_i^{a,+}   = LayerNorm(H_i^a + delta_i^a)

    There is no self-loop (delta of an isolated node is exactly 0; the ego is
    kept through the residual), no GCN normalization, and no cross-factor
    transfer anywhere inside the layer: slot ``a`` only ever reads H[:, a].
    """

    def __init__(self, factor_dim: int = 128, eps: float = 1e-8) -> None:
        super().__init__()
        self.factor_dim = int(factor_dim)
        self.num_slots = NUM_SLOTS
        self.eps = float(eps)
        # One independent 128->128 bias-free map per slot.
        self.V = nn.ModuleList(
            [nn.Linear(self.factor_dim, self.factor_dim, bias=False) for _ in range(self.num_slots)]
        )
        self.D = nn.ModuleList(
            [nn.Linear(self.factor_dim, self.factor_dim, bias=False) for _ in range(self.num_slots)]
        )
        # Per-slot LayerNorm keeps the LN statistics factor-diagonal.
        self.norm = nn.ModuleList(
            [nn.LayerNorm(self.factor_dim) for _ in range(self.num_slots)]
        )

    def propagate(
        self, H: Tensor, edge_index: Tensor | None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """H ``[N, S, F]`` -> (H_next ``[N, S, F]``, stats).

        stats holds one detached scalar per slot:
            diag_update_ratio[a] = mean_i ||delta_i^a|| / (||H_i^a|| + eps)
            neighbor_norm[a]     = mean_i ||N_i^a||
        """
        num_nodes = int(H.size(0))
        X = torch.stack(
            [self.V[a](H[:, a]) for a in range(self.num_slots)], dim=1
        )  # [N, S, F]
        N = incoming_mean(X, edge_index, num_nodes)  # [N, S, F]
        delta = torch.stack(
            [self.D[a](N[:, a]) for a in range(self.num_slots)], dim=1
        )  # [N, S, F]
        H_next = torch.stack(
            [self.norm[a](H[:, a] + delta[:, a]) for a in range(self.num_slots)], dim=1
        )  # [N, S, F]

        with torch.no_grad():
            h_norm = H.norm(dim=-1)  # [N, S]
            d_norm = delta.norm(dim=-1)
            n_norm = N.norm(dim=-1)
            update_ratio = (d_norm / (h_norm + self.eps)).mean(dim=0)  # [S]
            neighbor_norm = n_norm.mean(dim=0)  # [S]
        stats = {
            "diag_update_ratio": update_ratio.detach(),
            "neighbor_norm": neighbor_norm.detach(),
        }
        return H_next, stats

    def forward(self, H: Tensor, edge_index: Tensor | None) -> Tensor:
        return self.propagate(H, edge_index)[0]


class StaticCrossLayer(DiagIDLayer):
    """O2-A STATIC-CROSS layer: DIAG-ID backbone + static off-diagonal branch.

    The diagonal path is inherited bit-for-bit from ``DiagIDLayer`` (same V_a /
    D_a / LN_a modules, same ``incoming_mean``, no self-loop, no GCN norm):

        X_i^a      = V_a H_i^a
        N_i^a      = incoming neighbor mean over X^a
        Delta_diag^b = D_b N^b
        Delta_cross^b = (1 / (S-1)) * sum_{a != b} C_{a->b}( . )

    with six bias-free 128->128 maps ``C_{a->b}`` (one per off-diagonal pair,
    zero-initialized so step 0 is exactly DIAG-ID):

        H_next^b = LN_b( H^b + Delta_diag^b + Delta_cross^b )

    The single behavioural switch is ``dup``:

    * ``dup=False`` (STATIC-CROSS):  ``C_{a->b}(N^a)`` — real other-ownership
      source states. ``Delta_cross^b`` therefore carries cross-factor graph
      evidence.
    * ``dup=True`` (STATIC-CROSS-DUP, capacity-matched control): every pair
      module still exists with the same shape/init/optimizer presence, but is
      fed ``C_{a->b}(N^b)`` — the target's OWN neighborhood state. The cross
      branch therefore adds identical extra capacity while receiving NO new
      ownership source information.

    The transfer function is unconditional/static: it sees no node context,
    factor embedding, degree or relation (O2-A scope; conditionality is O2-B).
    """

    def __init__(self, factor_dim: int = 128, eps: float = 1e-8, dup: bool = False) -> None:
        super().__init__(factor_dim, eps)
        self.dup = bool(dup)
        self.cross = nn.ModuleDict(
            {
                key: nn.Linear(self.factor_dim, self.factor_dim, bias=False)
                for key in CROSS_PAIR_KEYS
            }
        )
        for module in self.cross.values():
            nn.init.zeros_(module.weight)

    # ------------------------------------------------------------------
    # Cross branch
    # ------------------------------------------------------------------

    def cross_messages(self, N: Tensor) -> dict[str, Tensor]:
        """Raw per-pair cross messages ``delta_{a->b}`` from neighbor states.

        N ``[N, S, F]`` -> ``{pair_key: [N, F]}``. STATIC-CROSS reads the real
        source slot ``N[:, a]``; the DUP control reads the target slot
        ``N[:, b]`` through the same (identically shaped) module.
        """
        messages: dict[str, Tensor] = {}
        for key in CROSS_PAIR_KEYS:
            a_name, b_name = key.split("_to_")
            a = SLOT_NAMES.index(a_name)
            b = SLOT_NAMES.index(b_name)
            source = N[:, b] if self.dup else N[:, a]
            messages[key] = self.cross[key](source)
        return messages

    def combine_cross(self, messages: dict[str, Tensor]) -> Tensor:
        """``Delta_cross`` ``[N, S, F]``: mean of the two incoming pair messages."""
        denom = float(self.num_slots - 1)
        stacked = []
        for b in range(self.num_slots):
            total = None
            for key in CROSS_PAIR_KEYS:
                if CROSS_PAIR_TARGET_SLOT[key] != b:
                    continue
                message = messages[key]
                total = message if total is None else total + message
            stacked.append(total / denom)
        return torch.stack(stacked, dim=1)

    def cross_update(self, N: Tensor) -> Tensor:
        return self.combine_cross(self.cross_messages(N))

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(
        self, H: Tensor, edge_index: Tensor | None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """H ``[N, S, F]`` -> (H_next, stats).

        Same stats keys as ``DiagIDLayer`` plus, for the cross branch:
            cross_update_ratio[b]      = mean_i ||Delta_cross_i^b|| / (||H_i^b|| + eps)
            cross_to_diag_ratio[b]     = mean_i ||Delta_cross_i^b|| / (||Delta_diag_i^b|| + eps)
            cross_pair_ratio_{a_to_b}  = mean_i ||delta_{a->b,i}|| / (||H_i^b|| + eps)
        These are learned relative update magnitudes, not factor demand.
        """
        num_nodes = int(H.size(0))
        X = torch.stack(
            [self.V[a](H[:, a]) for a in range(self.num_slots)], dim=1
        )  # [N, S, F]
        N = incoming_mean(X, edge_index, num_nodes)  # [N, S, F]
        delta_diag = torch.stack(
            [self.D[a](N[:, a]) for a in range(self.num_slots)], dim=1
        )  # [N, S, F]
        messages = self.cross_messages(N)
        delta_cross = self.combine_cross(messages)  # [N, S, F]
        H_next = torch.stack(
            [
                self.norm[b](H[:, b] + delta_diag[:, b] + delta_cross[:, b])
                for b in range(self.num_slots)
            ],
            dim=1,
        )  # [N, S, F]

        with torch.no_grad():
            h_norm = H.norm(dim=-1)  # [N, S]
            d_norm = delta_diag.norm(dim=-1)
            c_norm = delta_cross.norm(dim=-1)
            n_norm = N.norm(dim=-1)
            stats = {
                "diag_update_ratio": (d_norm / (h_norm + self.eps)).mean(dim=0),
                "neighbor_norm": n_norm.mean(dim=0),
                "cross_update_ratio": (c_norm / (h_norm + self.eps)).mean(dim=0),
                "cross_to_diag_ratio": (c_norm / (d_norm + self.eps)).mean(dim=0),
            }
            for key in CROSS_PAIR_KEYS:
                target = CROSS_PAIR_TARGET_SLOT[key]
                pair_norm = messages[key].norm(dim=-1)
                stats[f"cross_pair_ratio_{key}"] = (
                    pair_norm / (h_norm[:, target] + self.eps)
                ).mean()
        return H_next, stats


def binary_entropy(p: Tensor, eps: float = 1e-8) -> Tensor:
    """Binary entropy H(p) = -p log p - (1-p) log(1-p), elementwise.

    Used as an O2-B1 router diagnostic: H = 0 means the router is fully
    committed to Null or Transfer for that node, H = log 2 means undecided.
    """
    q = p.clamp(min=eps, max=1.0 - eps)
    return -(q * q.log() + (1.0 - q) * (1.0 - q).log())


class ConditionalCrossLayer(StaticCrossLayer):
    """O2-B1 CONDITIONAL-CROSS layer: DIAG-ID backbone + target-conditioned
    Null-vs-Transfer routing over the six off-diagonal cross maps.

    Diagonal path is inherited unchanged (same V_a / D_a / LN_a modules, same
    ``incoming_mean``, no self-loop, no GCN norm):

        X_i^a         = V_a H_i^a
        N_i^a         = incoming neighbor mean over X^a
        Delta_diag_i^b = D_b N_i^b

    O2-A's six bias-free maps ``T_{a->b}`` (a != b, zero-initialized) are reused
    verbatim as the Transfer operator ``B_transfer``; the Null operator is
    ``B_null(x) = 0``. Instead of applying the transfer unconditionally, a
    single SHARED router decides per node and per pair:

        q_i^{ab}          = [ H_i^b | N_i^b | e_a | e_b ]        # TARGET-ONLY context
        [l_null, l_tr]    = Router(q_i^{ab})                     # Linear->GELU->Linear(2)
        p_i^{ab}          = softmax([l_null, l_tr])              # p_null + p_transfer = 1
        payload_i^{a->b}  = T_{a->b}(N_i^a)                      # CONDITIONAL-CROSS
                          = T_{a->b}(N_i^b)                      # CONDITIONAL-DUP (control)
        m_i^{a->b}        = p_transfer_i^{ab} * payload_i^{a->b}
        Delta_cross_i^b   = (1 / (S-1)) * sum_{a != b} m_i^{a->b}
        H_next_i^b        = LN_b( H_i^b + Delta_diag_i^b + Delta_cross_i^b )

    CRITICAL CONTROL DISCIPLINE: the router reads the TARGET's own state
    ``H^b`` / ``N^b`` plus the two factor *identity* embeddings ``e_a`` / ``e_b``
    (indices, not content). It never reads the source ownership state
    ``H^a`` / ``N^a``, any same-node [C|Pt|Pv] conditioner, or any cross-source
    statistic. Otherwise CONDITIONAL-DUP — whose payload deliberately ignores
    the real source — could still see source ownership content through the
    router, and the matched control would be contaminated. See
    ``test_router_reads_target_context_only``.

    ``dup`` is the ONLY behavioural switch between the two O2-B1 variants
    (identical modules, params, init, optimizer presence, router and target
    update): ``dup=True`` feeds every pair module the target's own ``N^b``, so
    the conditional machinery is identical and only the payload's ownership
    information differs.

    Initialization: the router's last Linear is zero-initialized (no Null
    prior bias) so step 0 gives exactly ``p_null = p_transfer = 0.5``, and the
    six ``T_{a->b}`` stay zero-initialized, so ``m = p * T(.) = 0`` exactly and
    both variants start strictly as DIAG-ID. Because ``dL/dp ∝ T(x) = 0``, the
    router receives exactly zero gradient on the first backward pass; the
    transfer maps move first, and the router starts learning afterwards. This
    is expected, not a bug.
    """

    def __init__(
        self,
        factor_dim: int = 128,
        eps: float = 1e-8,
        dup: bool = False,
        embed_dim: int = 16,
        router_hidden: int = 128,
    ) -> None:
        super().__init__(factor_dim, eps, dup)
        self.embed_dim = int(embed_dim)
        self.router_hidden = int(router_hidden)
        # One embedding table serves both source and target factor identities.
        self.factor_embed = nn.Embedding(NUM_SLOTS, self.embed_dim)
        # ONE shared router for all six pairs; pair identity comes from e_a/e_b.
        self.router = nn.Sequential(
            nn.Linear(2 * self.factor_dim + 2 * self.embed_dim, self.router_hidden),
            nn.GELU(),
            nn.Linear(self.router_hidden, 2),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    @property
    def router_input_dim(self) -> int:
        return 2 * self.factor_dim + 2 * self.embed_dim

    def pair_context(self, H: Tensor, N: Tensor, a: int, b: int) -> Tensor:
        """Target-only router input ``[H^b | N^b | e_a | e_b]`` -> ``[N, D]``.

        Only the TARGET slot ``b`` is read from the states; the source enters
        solely through the identity embedding ``e_a``.
        """
        n = int(H.size(0))
        idx = torch.tensor([a, b], device=H.device, dtype=torch.long)
        emb = self.factor_embed(idx)  # [2, E]
        e_a = emb[0].unsqueeze(0).expand(n, -1)
        e_b = emb[1].unsqueeze(0).expand(n, -1)
        return torch.cat([H[:, b], N[:, b], e_a, e_b], dim=-1)

    def pair_logits(self, H: Tensor, N: Tensor, a: int, b: int) -> Tensor:
        """Raw router logits ``[l_null, l_transfer]`` -> ``[N, 2]``."""
        return self.router(self.pair_context(H, N, a, b))

    def router_transfer(self, H: Tensor, N: Tensor) -> dict[str, Tensor]:
        """Per-node transfer probability for every off-diagonal pair.

        ``{pair_key: p_transfer_i^{ab} [N]}``, all from the same shared router.
        """
        transfer: dict[str, Tensor] = {}
        for key in CROSS_PAIR_KEYS:
            a_name, b_name = key.split("_to_")
            a, b = SLOT_NAMES.index(a_name), SLOT_NAMES.index(b_name)
            probs = torch.softmax(self.pair_logits(H, N, a, b), dim=-1)  # [N, 2]
            transfer[key] = probs[:, 1]
        return transfer

    def neighbor_states(self, H: Tensor, edge_index: Tensor | None) -> Tensor:
        """Recompute the target neighborhood states ``N`` from ``H``.

        Diagnostic helper (offline scripts); identical arithmetic to the
        ``V_a`` -> ``incoming_mean`` part of ``propagate``.
        """
        X = torch.stack(
            [self.V[a](H[:, a]) for a in range(self.num_slots)], dim=1
        )
        return incoming_mean(X, edge_index, int(H.size(0)))

    # ------------------------------------------------------------------
    # Conditional cross branch
    # ------------------------------------------------------------------

    def conditional_cross_messages(self, H: Tensor, N: Tensor) -> dict[str, Tensor]:
        """Routed pair messages ``m_i^{a->b} = p_transfer_i^{ab} * T_{a->b}(.)``.

        ``cross_messages(N)`` (inherited) supplies the raw un-routed payloads;
        the source slot it reads is set by ``dup``.
        """
        transfer = self.router_transfer(H, N)
        payloads = self.cross_messages(N)
        return {
            key: transfer[key].unsqueeze(-1) * payloads[key] for key in CROSS_PAIR_KEYS
        }

    def cross_update(self, N: Tensor) -> Tensor:  # pragma: no cover - guard
        raise NotImplementedError(
            "ConditionalCrossLayer routes with the target context H; use "
            "conditional_cross_messages(H, N) instead of the static cross_update(N)"
        )

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(
        self, H: Tensor, edge_index: Tensor | None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """H ``[N, S, F]`` -> (H_next, stats).

        The diagonal block is written out verbatim as in
        ``StaticCrossLayer.propagate`` (same ops, same order) so the O2-A /
        O1 paths stay untouched; the cross block is routed. Stats keys:
        the O2-A ones plus, per pair, the raw vs effective magnitude and the
        router's transfer mean/std/entropy (all detached scalars).
        """
        num_nodes = int(H.size(0))
        X = torch.stack(
            [self.V[a](H[:, a]) for a in range(self.num_slots)], dim=1
        )  # [N, S, F]
        N = incoming_mean(X, edge_index, num_nodes)  # [N, S, F]
        delta_diag = torch.stack(
            [self.D[a](N[:, a]) for a in range(self.num_slots)], dim=1
        )  # [N, S, F]

        transfer = self.router_transfer(H, N)  # {key: [N]}
        payloads = self.cross_messages(N)  # raw T_{a->b}(payload)
        messages = {
            key: transfer[key].unsqueeze(-1) * payloads[key] for key in CROSS_PAIR_KEYS
        }
        delta_cross = self.combine_cross(messages)  # [N, S, F]

        H_next = torch.stack(
            [
                self.norm[b](H[:, b] + delta_diag[:, b] + delta_cross[:, b])
                for b in range(self.num_slots)
            ],
            dim=1,
        )  # [N, S, F]

        with torch.no_grad():
            h_norm = H.norm(dim=-1)  # [N, S]
            d_norm = delta_diag.norm(dim=-1)
            c_norm = delta_cross.norm(dim=-1)
            n_norm = N.norm(dim=-1)
            stats = {
                "diag_update_ratio": (d_norm / (h_norm + self.eps)).mean(dim=0),
                "neighbor_norm": n_norm.mean(dim=0),
                "cross_update_ratio": (c_norm / (h_norm + self.eps)).mean(dim=0),
                "cross_to_diag_ratio": (c_norm / (d_norm + self.eps)).mean(dim=0),
            }
            for key in CROSS_PAIR_KEYS:
                target = CROSS_PAIR_TARGET_SLOT[key]
                denom = h_norm[:, target] + self.eps
                raw_norm = payloads[key].norm(dim=-1)
                eff_norm = messages[key].norm(dim=-1)
                p = transfer[key]
                stats[f"cross_pair_ratio_{key}"] = (raw_norm / denom).mean()
                stats[f"effective_cross_pair_ratio_{key}"] = (eff_norm / denom).mean()
                stats[f"transfer_mean_{key}"] = p.mean()
                stats[f"transfer_std_{key}"] = p.std(unbiased=False)
                stats[f"router_entropy_{key}"] = binary_entropy(p).mean()
        return H_next, stats


def post_transition_stats(H0: Tensor, H1: Tensor, eps: float = 1e-8) -> dict[str, Tensor]:
    """O2-A post-transition ownership diagnostics comparing H0 and H1.

    Returns detached scalars:
        pre_cos_{a}_{b}   = mean cosine(H0^a, H0^b)        (pre-transition)
        post_cos_{a}_{b}  = mean cosine(H1^a, H1^b)        (post-transition)
        post_norm_{b}     = mean_i ||H1_i^b||
        state_drift_{b}   = mean_i [1 - cosine(H1_i^b, H0_i^b)]

    The pre/post cosine pair answers whether cross-factor transfer collapses the
    ownership states back into similar representations; the drift answers how
    far each slot moved from its own pre-transition state.
    """
    with torch.no_grad():
        stats: dict[str, Tensor] = {}
        for b in range(NUM_SLOTS):
            stats[f"post_norm_{SLOT_NAMES[b]}"] = H1[:, b].norm(dim=-1).mean()
            drift = 1.0 - F.cosine_similarity(H1[:, b], H0[:, b], dim=-1, eps=eps)
            stats[f"state_drift_{SLOT_NAMES[b]}"] = drift.mean()
        for a in range(NUM_SLOTS):
            for b in range(a + 1, NUM_SLOTS):
                pre = F.cosine_similarity(H0[:, a], H0[:, b], dim=-1, eps=eps)
                post = F.cosine_similarity(H1[:, a], H1[:, b], dim=-1, eps=eps)
                stats[f"pre_cos_{SLOT_NAMES[a]}_{SLOT_NAMES[b]}"] = pre.mean()
                stats[f"post_cos_{SLOT_NAMES[a]}_{SLOT_NAMES[b]}"] = post.mean()
    return stats
