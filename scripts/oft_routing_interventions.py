"""O2-B1.5 frozen routing interventions (pure tensor policies).

Every policy takes a routing-probability mapping
``{pair_key: p_transfer [N]}`` (the output of
``ConditionalCrossLayer.router_transfer``) and returns a new mapping of the
same shape. Nothing here touches the model: the O2-B1.5 audit script installs
the returned mapping by monkeypatching ``router_transfer``, so the layer's own
payload / ``combine_cross`` / LayerNorm / fusion / head path is reused
verbatim and ONLY the gate changes.

Interventions (execution plan §8-§15):

    ORIGINAL         — the checkpoint's own router probabilities (no policy).
    PAIR_CONSTANT    — per-pair TRAIN mean, constant over all nodes.
    TARGET_NODE      — per-node target receptivity: both sources of a target
                       share ``0.5 * (p^{a1->b} + p^{a2->b})``.
    TARGET_CONSTANT  — per-target TRAIN mean over both incoming pairs.
    SOURCE_SWAP      — keep each node's gates, exchange the two sources of
                       every target.
    NODE_SHUFFLE     — per-pair independent permutation of the VALIDATION
                       gates (distribution preserved, node alignment broken).
    ALL_ON / ALL_NULL— p_transfer = 1 / 0 everywhere (frozen diagnostics only).

Discipline: TRAIN-derived constants use the TRAIN split only and never look at
labels; NODE_SHUFFLE permutes within the validation split only; no policy reads
``y`` anywhere.
"""

from __future__ import annotations

import contextlib

import torch
from torch import Tensor

from src.models.oft_components import CROSS_PAIR_KEYS, CROSS_PAIR_TARGET_SLOT, NUM_SLOTS


def pair_sources(target_slot: int) -> tuple[str, str]:
    """The two off-diagonal pair keys whose target is ``target_slot``."""
    keys = [key for key in CROSS_PAIR_KEYS if CROSS_PAIR_TARGET_SLOT[key] == target_slot]
    if len(keys) != NUM_SLOTS - 1:
        raise ValueError(f"target slot {target_slot} has {len(keys)} incoming pairs")
    return keys[0], keys[1]


def pair_constant(transfer: dict[str, Tensor], train_mask: Tensor) -> dict[str, Tensor]:
    """§9: ``p_i^{ab} <- mean_{i in TRAIN} p_i^{ab}`` (six constants)."""
    out: dict[str, Tensor] = {}
    for key in CROSS_PAIR_KEYS:
        mean = float(transfer[key][train_mask].double().mean())
        out[key] = torch.full_like(transfer[key], mean)
    return out


def target_node(transfer: dict[str, Tensor]) -> dict[str, Tensor]:
    """§10: per-node target receptivity, source identity removed.

    ``p_i^b = 0.5 * (p_i^{a1->b} + p_i^{a2->b})`` is assigned to BOTH sources.
    Node variation and target identity survive; the two sources of a target
    become indistinguishable.
    """
    out: dict[str, Tensor] = {}
    for b in range(NUM_SLOTS):
        a1, a2 = pair_sources(b)
        mean = 0.5 * (transfer[a1] + transfer[a2])
        out[a1] = mean.clone()
        out[a2] = mean.clone()
    return out


def target_constant(transfer: dict[str, Tensor], train_mask: Tensor) -> dict[str, Tensor]:
    """§11: ``p_i^{ab} <- mean_{i in TRAIN, a != b} p_i^{a->b}`` (three constants)."""
    node_gate = target_node(transfer)
    out: dict[str, Tensor] = {}
    for key in CROSS_PAIR_KEYS:
        mean = float(node_gate[key][train_mask].double().mean())
        out[key] = torch.full_like(transfer[key], mean)
    return out


def source_swap(transfer: dict[str, Tensor]) -> dict[str, Tensor]:
    """§12: exchange the two sources of every target, per node.

    Keeps node variation, target identity and the probability distribution;
    only the source->gate assignment is broken.
    """
    out: dict[str, Tensor] = {}
    for b in range(NUM_SLOTS):
        a1, a2 = pair_sources(b)
        out[a1] = transfer[a2].clone()
        out[a2] = transfer[a1].clone()
    return out


def node_shuffle(transfer: dict[str, Tensor], val_idx: Tensor, seed: int) -> dict[str, Tensor]:
    """§13: per-pair independent permutation of the VALIDATION gates.

    The permutation is drawn on CPU with an explicit ``torch.Generator`` so it
    is deterministic under a fixed seed and independent of the device. Each
    pair draws its own permutation from the same generator in the fixed
    ``CROSS_PAIR_KEYS`` order, so different pairs never share a permutation.
    Marginal distribution / mean / std / quantiles are preserved exactly.
    """
    n = int(val_idx.numel())
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out: dict[str, Tensor] = {}
    for key in CROSS_PAIR_KEYS:
        perm = torch.randperm(n, generator=gen)
        p = transfer[key].clone()
        val = val_idx.to(p.device)
        p[val] = transfer[key][val][perm.to(p.device)]
        out[key] = p
    return out


def all_on(transfer: dict[str, Tensor]) -> dict[str, Tensor]:
    """§14: ``p_transfer = 1`` everywhere (frozen-checkpoint diagnostic only)."""
    return {key: torch.ones_like(value) for key, value in transfer.items()}


def all_null(transfer: dict[str, Tensor]) -> dict[str, Tensor]:
    """§15: ``p_transfer = 0`` everywhere (frozen-checkpoint diagnostic only)."""
    return {key: torch.zeros_like(value) for key, value in transfer.items()}


@contextlib.contextmanager
def patched_layer(layer, transfer: dict[str, Tensor] | None):
    """Temporarily replace a layer's ``router_transfer`` with a fixed mapping.

    ``transfer=None`` is a no-op (the checkpoint's own router runs). The
    replacement returns the SAME tensors for every call, so the only thing that
    changes inside ``propagate`` is the gate; payloads, ``delta_diag``,
    LayerNorm, fusion and head are untouched, and no parameter is modified.
    """
    if transfer is None:
        yield layer
        return
    original = layer.router_transfer

    def _fixed(H: Tensor, N: Tensor) -> dict[str, Tensor]:  # noqa: ARG001 - fixed gate
        return {key: transfer[key].to(H.device) for key in CROSS_PAIR_KEYS}

    layer.router_transfer = _fixed
    try:
        yield layer
    finally:
        layer.router_transfer = original


@contextlib.contextmanager
def patched_router(model, transfer: dict[str, Tensor] | None):
    """``patched_layer`` for a whole model (single-layer OFT scaffold)."""
    with patched_layer(model.diag_layers[0], transfer) as layer:
        yield layer
