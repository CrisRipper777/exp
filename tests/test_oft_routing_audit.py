"""O2-B1.5 Part A — frozen routing-intervention unit tests.

Covers execution plan §29: ORIGINAL replay identity, every intervention's
defining invariant, determinism of NODE_SHUFFLE, the "no payload / no diag /
no parameter mutation" guarantee, and the "never touches the held-out TEST
split" property of the audit script. Pure CPU, no training, no dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from oft_routing_interventions import (  # noqa: E402
    all_null,
    all_on,
    node_shuffle,
    pair_constant,
    pair_sources,
    patched_layer,
    patched_router,
    source_swap,
    target_constant,
    target_node,
)

from src.models.oft_components import (  # noqa: E402
    CROSS_PAIR_KEYS,
    NUM_SLOTS,
    ConditionalCrossLayer,
)

AUDIT_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "oft_o2b1_5_routing_causal_audit.py"


def _make_layer(n: int = 12, f: int = 8, dup: bool = False, seed: int = 0):
    """Randomized conditional layer + random H/N states + a small graph."""
    torch.manual_seed(seed)
    layer = ConditionalCrossLayer(factor_dim=f, dup=dup)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for param in layer.parameters():
            param.copy_(torch.randn(param.shape, generator=g) * 0.3)
    H = torch.randn(n, NUM_SLOTS, f, generator=g)
    N = torch.randn(n, NUM_SLOTS, f, generator=g)
    edge = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    return layer, H, N, edge


def _transfer(layer, H, N):
    with torch.no_grad():
        return layer.router_transfer(H, N)


def _clone(transfer):
    return {key: value.clone() for key, value in transfer.items()}


# ----------------------------------------------------------------------
# 1. ORIGINAL replay
# ----------------------------------------------------------------------


def test_original_replay_equals_unpatched_forward():
    """Installing the checkpoint's OWN probabilities as a fixed gate must
    reproduce the standard forward bitwise (the replay proof, unit level)."""
    layer, H, N, edge = _make_layer()
    N = layer.neighbor_states(H, edge)  # the layer's OWN neighbor states
    p = _transfer(layer, H, N)
    H_ref, stats_ref = layer.propagate(H, edge)
    with patched_layer(layer, p):
        H_pat, stats_pat = layer.propagate(H, edge)
    assert torch.equal(H_ref, H_pat)
    for key in stats_ref:
        assert torch.equal(stats_ref[key], stats_pat[key])


def test_patched_layer_restores_original_router():
    layer, H, N, edge = _make_layer()
    p_before = _transfer(layer, H, N)
    with patched_layer(layer, all_null(p_before)):
        pass
    p_after = _transfer(layer, H, N)
    for key in CROSS_PAIR_KEYS:
        assert torch.equal(p_before[key], p_after[key])


# ----------------------------------------------------------------------
# 2-3. PAIR_CONSTANT
# ----------------------------------------------------------------------


def test_pair_constant_same_gate_for_all_nodes_and_train_derived():
    layer, H, N, edge = _make_layer()
    p = _transfer(layer, H, N)
    train_mask = torch.zeros(H.size(0), dtype=torch.bool)
    train_mask[:7] = True
    out = pair_constant(p, train_mask)
    for key in CROSS_PAIR_KEYS:
        assert out[key].unique().numel() == 1, "pair gate must be constant over nodes"
        assert torch.equal(out[key], torch.full_like(p[key], float(p[key][train_mask].double().mean())))


def test_pair_constant_uses_train_split_not_val():
    layer, H, N, edge = _make_layer()
    p = _transfer(layer, H, N)
    train_mask = torch.zeros(H.size(0), dtype=torch.bool)
    train_mask[:7] = True
    base = pair_constant(p, train_mask)

    perturbed_val = _clone(p)
    for key in CROSS_PAIR_KEYS:
        perturbed_val[key][7:] += 5.0
    assert all(
        torch.equal(base[key], pair_constant(perturbed_val, train_mask)[key]) for key in CROSS_PAIR_KEYS
    ), "validation entries must not influence the train-derived constant"

    perturbed_train = _clone(p)
    for key in CROSS_PAIR_KEYS:
        perturbed_train[key][:7] += 5.0
    assert any(
        not torch.equal(base[key], pair_constant(perturbed_train, train_mask)[key])
        for key in CROSS_PAIR_KEYS
    ), "train entries must define the constant"


# ----------------------------------------------------------------------
# 4-5. TARGET_NODE
# ----------------------------------------------------------------------


def test_target_node_equalizes_sources_per_node():
    layer, H, N, edge = _make_layer()
    p = _transfer(layer, H, N)
    out = target_node(p)
    for b in range(NUM_SLOTS):
        a1, a2 = pair_sources(b)
        expected = 0.5 * (p[a1] + p[a2])
        assert torch.equal(out[a1], expected)
        assert torch.equal(out[a1], out[a2]), "both sources of a target must share one gate"


def test_target_node_preserves_node_variation():
    layer, H, N, edge = _make_layer()
    p = _transfer(layer, H, N)
    out = target_node(p)
    for b in range(NUM_SLOTS):
        a1, _ = pair_sources(b)
        assert out[a1].std(unbiased=False) > 0.0
        assert out[a1].unique().numel() > 1, "TARGET_NODE must stay node-dependent"


# ----------------------------------------------------------------------
# 6. TARGET_CONSTANT
# ----------------------------------------------------------------------


def test_target_constant_is_per_target_constant():
    layer, H, N, edge = _make_layer()
    p = _transfer(layer, H, N)
    train_mask = torch.zeros(H.size(0), dtype=torch.bool)
    train_mask[:7] = True
    out = target_constant(p, train_mask)
    for b in range(NUM_SLOTS):
        a1, a2 = pair_sources(b)
        assert out[a1].unique().numel() == 1
        assert torch.equal(out[a1], out[a2])
        expected = float((0.5 * (p[a1] + p[a2]))[train_mask].double().mean())
        assert abs(float(out[a1][0]) - expected) < 1e-6  # stored in float32
    # different targets keep different constants (not collapsed to one value)
    values = {float(out[pair_sources(b)[0]][0]) for b in range(NUM_SLOTS)}
    assert len(values) == NUM_SLOTS


# ----------------------------------------------------------------------
# 7. SOURCE_SWAP
# ----------------------------------------------------------------------


def test_source_swap_exchanges_sources_per_target():
    layer, H, N, edge = _make_layer()
    p = _transfer(layer, H, N)
    out = source_swap(p)
    for b in range(NUM_SLOTS):
        a1, a2 = pair_sources(b)
        assert torch.equal(out[a1], p[a2])
        assert torch.equal(out[a2], p[a1])
    # distribution preserved per target
    for b in range(NUM_SLOTS):
        a1, a2 = pair_sources(b)
        assert torch.equal(
            torch.sort(torch.cat([out[a1], out[a2]])).values,
            torch.sort(torch.cat([p[a1], p[a2]])).values,
        )


# ----------------------------------------------------------------------
# 8-9. NODE_SHUFFLE
# ----------------------------------------------------------------------


def test_node_shuffle_preserves_distribution_breaks_alignment():
    layer, H, N, edge = _make_layer(n=16)
    p = _transfer(layer, H, N)
    val_idx = torch.arange(6, 16)
    out = node_shuffle(p, val_idx, 1000)
    for key in CROSS_PAIR_KEYS:
        assert torch.equal(
            torch.sort(out[key][val_idx]).values, torch.sort(p[key][val_idx]).values
        ), "marginal distribution must be preserved exactly"
        mask = torch.ones(16, dtype=torch.bool)
        mask[val_idx] = False
        assert torch.equal(out[key][mask], p[key][mask]), "non-validation nodes untouched"
    assert any(
        not torch.equal(out[key][val_idx], p[key][val_idx]) for key in CROSS_PAIR_KEYS
    ), "alignment must actually change"
    # pairs draw independent permutations: identical inputs must give distinct outputs
    same = {key: p[CROSS_PAIR_KEYS[0]].clone() for key in CROSS_PAIR_KEYS}
    out_same = node_shuffle(same, val_idx, 1000)
    vectors = {tuple(out_same[key][val_idx].tolist()) for key in CROSS_PAIR_KEYS}
    assert len(vectors) == len(CROSS_PAIR_KEYS), "pairs must not share one permutation"


def test_node_shuffle_deterministic_under_fixed_seed():
    layer, H, N, edge = _make_layer(n=16)
    p = _transfer(layer, H, N)
    val_idx = torch.arange(6, 16)
    a = node_shuffle(p, val_idx, 1000)
    b = node_shuffle(p, val_idx, 1000)
    c = node_shuffle(p, val_idx, 1001)
    for key in CROSS_PAIR_KEYS:
        assert torch.equal(a[key], b[key])
    assert any(not torch.equal(a[key], c[key]) for key in CROSS_PAIR_KEYS)


# ----------------------------------------------------------------------
# 10-11. ALL_ON / ALL_NULL
# ----------------------------------------------------------------------


def test_all_on_and_all_null():
    layer, H, N, edge = _make_layer()
    p = _transfer(layer, H, N)
    on, null = all_on(p), all_null(p)
    for key in CROSS_PAIR_KEYS:
        assert torch.equal(on[key], torch.ones_like(p[key]))
        assert torch.equal(null[key], torch.zeros_like(p[key]))


# ----------------------------------------------------------------------
# 12. no payload / diag / parameter mutation + the gate really acts
# ----------------------------------------------------------------------


def test_interventions_do_not_change_payload_diag_or_parameters():
    layer, H, N, edge = _make_layer()
    p = _transfer(layer, H, N)
    payloads_ref = layer.cross_messages(N)
    diag_ref = torch.stack(
        [layer.D[a](N[:, a]) for a in range(NUM_SLOTS)], dim=1
    )
    params_ref = {name: param.detach().clone() for name, param in layer.named_parameters()}

    with patched_layer(layer, all_on(p)):
        H1_on, _ = layer.propagate(H, edge)
    payloads_after = layer.cross_messages(N)
    diag_after = torch.stack([layer.D[a](N[:, a]) for a in range(NUM_SLOTS)], dim=1)
    for key in CROSS_PAIR_KEYS:
        assert torch.equal(payloads_ref[key], payloads_after[key])
    assert torch.equal(diag_ref, diag_after)
    for name, param in layer.named_parameters():
        assert torch.equal(params_ref[name], param.detach()), f"{name} changed"

    with patched_layer(layer, all_null(p)):
        H1_null, _ = layer.propagate(H, edge)
    assert not torch.equal(H1_on, H1_null), "the gate must actually change the update"


def test_interventions_do_not_mutate_the_input_mapping():
    layer, H, N, edge = _make_layer()
    p = _transfer(layer, H, N)
    snapshot = _clone(p)
    val_idx = torch.arange(6, H.size(0))
    train_mask = torch.zeros(H.size(0), dtype=torch.bool)
    train_mask[:6] = True
    for policy in (
        lambda: pair_constant(p, train_mask),
        lambda: target_node(p),
        lambda: target_constant(p, train_mask),
        lambda: source_swap(p),
        lambda: node_shuffle(p, val_idx, 1000),
        lambda: all_on(p),
        lambda: all_null(p),
    ):
        policy()
        for key in CROSS_PAIR_KEYS:
            assert torch.equal(snapshot[key], p[key]), "policy mutated its input"


# ----------------------------------------------------------------------
# 13. the audit script never reads the held-out TEST split
# ----------------------------------------------------------------------


def test_audit_script_never_references_test_split():
    source = AUDIT_SCRIPT.read_text(encoding="utf-8")
    assert "test_idx" not in source, "audit script must never reference the TEST split"
    assert ".test_" not in source
    assert "evaluate_test" not in source


def test_patched_router_wrapper_targets_the_model_layer():
    layer, H, N, edge = _make_layer()
    N = layer.neighbor_states(H, edge)
    p = _transfer(layer, H, N)

    class _Stub(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.diag_layers = torch.nn.ModuleList([inner])

    model = _Stub(layer)
    H_ref, _ = layer.propagate(H, edge)
    with patched_router(model, p):
        H_pat, _ = layer.propagate(H, edge)
    assert torch.equal(H_ref, H_pat)
