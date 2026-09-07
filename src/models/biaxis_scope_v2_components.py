"""Components for SCOPE-MAG v2.

The implementation deliberately keeps the two axes separate:

* OSC aggregates only same-factor neighbours.  Edge messages are processed in
  small chunks, so no ``[num_edges, factor_dim]`` tensor is retained.
* ORB is pointwise at a node.  Cross-factor exchange happens only after OSC
  has produced structural evidence and is routed through a small latent.

The modules in this file are shared by IND, FULL and ORB controls.  This is
important for the R4 attribution protocol: changing the reconciliation mode
must not silently change the graph context encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_activation, make_norm

NUM_FACTORS = 3
FACTOR_NAMES = ("c", "pt", "pv")


def chunked_factor_mean(
    edge_index: torch.Tensor,
    values: torch.Tensor,
    num_nodes: int,
    edge_chunk_size: int,
) -> torch.Tensor:
    """Mean aggregate ``values[src]`` into ``dst`` without a full edge tensor."""

    d = int(values.size(-1))
    out = values.new_zeros((int(num_nodes), d))
    if edge_index is None or edge_index.numel() == 0:
        return out

    src, dst = edge_index[0], edge_index[1]
    degree = values.new_zeros((int(num_nodes),))
    chunk = max(int(edge_chunk_size), 1)
    for start in range(0, int(src.numel()), chunk):
        end = min(start + chunk, int(src.numel()))
        # The only edge-sized allocation is this bounded chunk.
        out.index_add_(0, dst[start:end], values[src[start:end]])
        degree.index_add_(0, dst[start:end], torch.ones_like(dst[start:end], dtype=values.dtype))
    return out / degree.clamp_min(1.0).unsqueeze(-1)


def _add_remaining_self_loops_once(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Return edges plus exactly one self-loop per node.

    Existing self-loops are deduplicated; non-self edges are preserved in
    their original order and multiplicity.  This makes the normalization
    control explicit and avoids relying on whether a dataset loader already
    inserted loops.
    """
    num_nodes = int(num_nodes)
    if edge_index is None or edge_index.numel() == 0:
        device = edge_index.device if edge_index is not None else "cpu"
        nodes = torch.arange(num_nodes, device=device, dtype=torch.long)
        return torch.stack([nodes, nodes], dim=0)
    src, dst = edge_index[0], edge_index[1]
    non_loop = src != dst
    base = edge_index[:, non_loop]
    existing_nodes = src[~non_loop].unique()
    present = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    if existing_nodes.numel():
        present[existing_nodes] = True
    missing_nodes = torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype)[~present]
    loop_nodes = torch.cat([existing_nodes.to(dtype=edge_index.dtype), missing_nodes], dim=0)
    loops = torch.stack([loop_nodes, loop_nodes], dim=0)
    return torch.cat([base, loops], dim=1)


def chunked_factor_sym_norm(
    edge_index: torch.Tensor,
    values: torch.Tensor,
    num_nodes: int,
    edge_chunk_size: int,
) -> torch.Tensor:
    """GCN-style symmetric normalization with bounded edge message chunks.

    The project uses ``edge_index[0]`` as source and ``edge_index[1]`` as
    destination.  Degree is therefore accumulated on destinations, matching
    the row degree of the implicit destination-by-source adjacency.  The
    normalized payload is accumulated as
    ``values[src] / sqrt(deg[dst] * deg[src])``.
    """
    num_nodes = int(num_nodes)
    full_edges = _add_remaining_self_loops_once(edge_index, num_nodes)
    src, dst = full_edges[0], full_edges[1]
    out = values.new_zeros((num_nodes, int(values.size(-1))))
    degree = values.new_zeros((num_nodes,))
    chunk = max(int(edge_chunk_size), 1)
    for start in range(0, int(src.numel()), chunk):
        end = min(start + chunk, int(src.numel()))
        degree.index_add_(
            0,
            dst[start:end],
            torch.ones((end - start,), dtype=values.dtype, device=values.device),
        )
    inv_sqrt = degree.clamp_min(1.0).rsqrt()
    for start in range(0, int(src.numel()), chunk):
        end = min(start + chunk, int(src.numel()))
        norm = inv_sqrt[dst[start:end]] * inv_sqrt[src[start:end]]
        out.index_add_(0, dst[start:end], values[src[start:end]] * norm.unsqueeze(-1))
    return out


class OwnershipSeparatedContext(nn.Module):
    """Shared nonlinear target update for the three ownership factors."""

    def __init__(
        self,
        factor_dim: int,
        hidden_dim: int,
        factor_id_dim: int,
        dropout: float,
        activation: str,
        norm: str,
    ) -> None:
        super().__init__()
        d = int(factor_dim)
        self.factor_dim = d
        self.factor_id_dim = int(factor_id_dim)
        self.factor_emb = nn.Embedding(NUM_FACTORS, self.factor_id_dim)
        in_dim = 4 * d + self.factor_id_dim
        self.core = nn.Sequential(
            make_norm(norm, in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), d),
        )

    def forward(self, h: torch.Tensor, message: torch.Tensor, factor: int) -> torch.Tensor:
        # Product and difference make the target update depend on the content
        # relationship between ego state and structural evidence.
        features = torch.cat(
            [h, message, h * message, h - message], dim=-1
        )
        e = self.factor_emb.weight[int(factor)].to(dtype=h.dtype)
        e = e.view(1, -1).expand(h.size(0), -1)
        return self.core(torch.cat([features, e], dim=-1))


class OACCBlock(nn.Module):
    """Ownership-Adaptive Context Core (OACC).

    The block separates the mature graph response from the target update:

        same-factor mean -> shared payload + small factor adapter
        -> node x factor graph-mass beta
        -> initial-residual anchored propagation + pre-norm FFN

    No edge, relation, or cross-factor router is used here.  ``beta_mode`` is
    the pre-registered control switch: ``adaptive`` is proposed, ``one`` is
    the beta=1 control, and ``node_shared`` removes factor-specific demand.
    """

    BETA_MODES = ("adaptive", "one", "node_shared")

    def __init__(
        self,
        factor_dim: int,
        factor_id_dim: int,
        adapter_rank: int,
        ffn_hidden_dim: int,
        beta_hidden_dim: int,
        beta_mode: str,
        rho: float,
        gamma: float,
        activation: str,
        norm: str,
    ) -> None:
        super().__init__()
        d = int(factor_dim)
        self.factor_dim = d
        self.factor_id_dim = int(factor_id_dim)
        self.beta_mode = str(beta_mode)
        if self.beta_mode not in self.BETA_MODES:
            raise ValueError(f"unknown OACC beta_mode: {self.beta_mode!r}")
        self.rho = float(rho)
        self.gamma = nn.Parameter(torch.full((NUM_FACTORS,), float(gamma)))
        self.factor_emb = nn.Embedding(NUM_FACTORS, self.factor_id_dim)

        # W0 is shared across ownership factors; the low-rank residual is the
        # only factor-specific payload transform.
        self.shared_payload = nn.Linear(d, d)
        self.adapter_down = nn.ModuleList([
            nn.Linear(d, int(adapter_rank), bias=False) for _ in range(NUM_FACTORS)
        ])
        self.adapter_up = nn.ModuleList([
            nn.Linear(int(adapter_rank), d, bias=False) for _ in range(NUM_FACTORS)
        ])

        if self.beta_mode == "adaptive":
            beta_in = 2 * d + d + self.factor_id_dim
            self.beta = nn.Sequential(
                make_norm(norm, beta_in),
                nn.Linear(beta_in, int(beta_hidden_dim)),
                get_activation(activation),
                nn.Linear(int(beta_hidden_dim), 1),
            )
        elif self.beta_mode == "node_shared":
            beta_in = 2 * d + d
            self.beta = nn.Sequential(
                make_norm(norm, beta_in),
                nn.Linear(beta_in, int(beta_hidden_dim)),
                get_activation(activation),
                nn.Linear(int(beta_hidden_dim), 1),
            )

        self.ffn_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(NUM_FACTORS)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, int(ffn_hidden_dim)),
                get_activation(activation),
                nn.Linear(int(ffn_hidden_dim), d),
            )
            for _ in range(NUM_FACTORS)
        ])

    def forward(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        message: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        payload = []
        for factor in range(NUM_FACTORS):
            v = self.shared_payload(message[:, factor])
            v = v + self.adapter_up[factor](self.adapter_down[factor](message[:, factor]))
            payload.append(v)
        payload = torch.stack(payload, dim=1)
        if self.beta_mode == "one":
            beta = payload.new_ones((payload.size(0), NUM_FACTORS, 1))
        elif self.beta_mode == "node_shared":
            h_mean = h.mean(dim=1)
            v_mean = payload.mean(dim=1)
            beta_input = torch.cat([
                torch.layer_norm(h_mean, (self.factor_dim,)),
                torch.layer_norm(v_mean, (self.factor_dim,)),
                h_mean * v_mean,
            ], dim=-1)
            beta = torch.sigmoid(self.beta(beta_input)).unsqueeze(1).expand(-1, NUM_FACTORS, -1)
        else:
            betas = []
            for factor in range(NUM_FACTORS):
                h_f = h[:, factor]
                pieces = [
                    torch.layer_norm(h_f, (self.factor_dim,)),
                    torch.layer_norm(payload[:, factor], (self.factor_dim,)),
                    h_f * payload[:, factor],
                ]
                e = self.factor_emb.weight[factor].to(dtype=h.dtype).view(1, -1).expand(h.size(0), -1)
                beta_input = torch.cat([*pieces, e], dim=-1)
                betas.append(torch.sigmoid(self.beta(beta_input)))
            beta = torch.stack(betas, dim=1)

        anchored = (1.0 - self.rho) * h + self.rho * h0 + beta * payload
        structural = []
        for factor in range(NUM_FACTORS):
            f = self.ffn[factor](self.ffn_norm[factor](anchored[:, factor]))
            structural.append(anchored[:, factor] + self.gamma[factor] * f)
        structural = torch.stack(structural, dim=1)
        return structural, {
            "payload": payload,
            "beta": beta,
            "graph_update": beta * payload,
            "ffn_update": structural - anchored,
        }


class FactorGlobalRelay(nn.Module):
    """Small factor-separated node->latent->node global relay.

    This is intentionally a low-rank global context path, not a second
    cross-factor mixer.  Each factor owns its assignment/value/readout maps.
    """

    ASSIGNMENT_MODES = ("learned", "uniform")

    def __init__(
        self,
        factor_dim: int,
        num_slots: int,
        activation: str,
        assignment_mode: str = "learned",
    ) -> None:
        super().__init__()
        d = int(factor_dim)
        s = int(num_slots)
        self.num_slots = s
        self.assignment_mode = str(assignment_mode)
        if self.assignment_mode not in self.ASSIGNMENT_MODES:
            raise ValueError(
                "assignment_mode must be learned|uniform, "
                f"got {self.assignment_mode!r}"
            )
        self.assign = nn.ModuleList([nn.Linear(d, s) for _ in range(NUM_FACTORS)])
        self.value = nn.ModuleList([nn.Linear(d, d) for _ in range(NUM_FACTORS)])
        self.slot_update = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                get_activation(activation),
                nn.Linear(d, d),
            )
            for _ in range(NUM_FACTORS)
        ])
        self.readout = nn.ModuleList([nn.Linear(d, d) for _ in range(NUM_FACTORS)])

    def forward(
        self,
        h: torch.Tensor,
        factor: int,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assignment = self._assignment(h, factor)
        collect_assignment = assignment
        dispatch_assignment = assignment
        intervention_mode = getattr(self, "_intervention_mode", None)
        if intervention_mode in ("shuffle_collect", "shuffle_dispatch"):
            permutation = getattr(self, "_intervention_permutation", None)
            if permutation is None or int(permutation.numel()) != int(h.size(0)):
                raise ValueError("relay intervention permutation must cover all nodes")
            shuffled = assignment[permutation]
            if intervention_mode == "shuffle_collect":
                collect_assignment = shuffled
            else:
                dispatch_assignment = shuffled
        value = self.value[int(factor)](h)  # [N, d]
        # Uniform assignment with any slot count, and learned assignment with
        # S=1, have one mathematically identical pooled value.  Computing the
        # canonical mean once makes that equivalence bitwise exact instead of
        # exposing harmless matmul/normalization rounding differences between
        # S=1 and repeated uniform slots.
        if self.assignment_mode == "uniform" or self.num_slots == 1:
            pooled = value.mean(dim=0, keepdim=True)
            slots = self.slot_update[int(factor)](pooled)
            output = self.readout[int(factor)](slots).expand(h.size(0), -1)
            if not return_stats:
                return output
            entropy = -(assignment.clamp_min(1e-8) * assignment.clamp_min(1e-8).log()).sum(dim=-1).mean()
            usage = assignment.mean(dim=0)
            return output, {
                "slot_entropy": entropy.detach(),
                "effective_active_slots": entropy.detach().exp(),
                "active_slots": (usage > (1.0 / max(self.num_slots * 10, 1))).sum().to(dtype=h.dtype).detach(),
                "slot_usage": usage.detach(),
            }
        denom = collect_assignment.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        slots = collect_assignment.transpose(0, 1) @ value / denom  # [S, d]
        slots = self.slot_update[int(factor)](slots)
        output = self.readout[int(factor)](dispatch_assignment @ slots)
        if not return_stats:
            return output
        entropy = -(assignment.clamp_min(1e-8) * assignment.clamp_min(1e-8).log()).sum(dim=-1).mean()
        usage = assignment.mean(dim=0)
        return output, {
            "slot_entropy": entropy.detach(),
            "effective_active_slots": entropy.detach().exp(),
            "active_slots": (usage > (1.0 / max(self.num_slots * 10, 1))).sum().to(dtype=h.dtype).detach(),
            "slot_usage": usage.detach(),
        }

    def _assignment(self, h: torch.Tensor, factor: int) -> torch.Tensor:
        """Return the node-to-slot assignment used by the relay."""
        if self.assignment_mode == "uniform":
            return h.new_full(
                (h.size(0), self.num_slots), 1.0 / float(self.num_slots)
            )
        return F.softmax(self.assign[int(factor)](h), dim=-1)  # [N, S]


class OwnershipReconciliationBottleneck(nn.Module):
    """ORB and its two controls.

    ``mode=none`` is IND, ``mode=orb`` is the proposed low-rank bottleneck,
    and ``mode=full`` is the unrestricted 3d->3d capacity control.
    """

    MODES = ("none", "orb", "full")

    def __init__(
        self,
        factor_dim: int,
        rank: int,
        hidden_dim: int,
        mode: str,
        activation: str,
        norm: str,
        use_graph_condition: bool = True,
    ) -> None:
        super().__init__()
        d = int(factor_dim)
        r = int(rank)
        self.factor_dim = d
        self.rank = r
        self.mode = str(mode)
        self.use_graph_condition = bool(use_graph_condition)
        if self.mode not in self.MODES:
            raise ValueError(f"unknown reconciliation mode: {self.mode!r}")
        if self.mode == "none":
            # IND must remain a clean ownership-isolation parent rather than
            # carrying unused ORB parameters in its optimizer/state dict.
            return
        self.evidence = nn.Sequential(
            make_norm(norm, 4 * d if self.use_graph_condition else d),
            nn.Linear(4 * d if self.use_graph_condition else d, int(hidden_dim)),
            get_activation(activation),
            nn.Linear(int(hidden_dim), d),
        )
        if self.mode == "orb":
            self.to_rank = nn.ModuleList([nn.Linear(d, r) for _ in range(NUM_FACTORS)])
            self.reconcile = nn.Sequential(
                nn.LayerNorm(NUM_FACTORS * r),
                nn.Linear(NUM_FACTORS * r, int(hidden_dim)),
                get_activation(activation),
                nn.Linear(int(hidden_dim), r),
            )
            self.decode = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(3 * r),
                    nn.Linear(3 * r, int(hidden_dim)),
                    get_activation(activation),
                    nn.Linear(int(hidden_dim), d),
                )
                for _ in range(NUM_FACTORS)
            ])
        elif self.mode == "full":
            self.full_mix = nn.Sequential(
                nn.LayerNorm(NUM_FACTORS * d),
                nn.Linear(NUM_FACTORS * d, int(hidden_dim)),
                get_activation(activation),
                nn.Linear(int(hidden_dim), NUM_FACTORS * d),
            )

    def forward(self, h: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Return factor-specific deltas with shape ``[N, 3, d]``."""
        if self.mode == "none":
            return h.new_zeros(h.shape)
        if self.use_graph_condition:
            evidence_input = torch.cat([h, message, h * message, h - message], dim=-1)
        else:
            evidence_input = h
        evidence = torch.stack(
            [self.evidence(evidence_input[:, f]) for f in range(NUM_FACTORS)], dim=1
        )
        if self.mode == "full":
            return self.full_mix(evidence.reshape(h.size(0), -1)).reshape_as(h)

        projected = torch.stack(
            [self.to_rank[f](evidence[:, f]) for f in range(NUM_FACTORS)], dim=1
        )
        latent = self.reconcile(projected.reshape(h.size(0), -1))
        deltas = []
        for f in range(NUM_FACTORS):
            p = projected[:, f]
            deltas.append(self.decode[f](torch.cat([p, latent, p * latent], dim=-1)))
        return torch.stack(deltas, dim=1)
