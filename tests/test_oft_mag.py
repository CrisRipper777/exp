"""Unit tests for the OFT-MAG O1 scaffold (OFT-DIAG-ID).

Covers (O1 plan §10):
1. forward interface / output shapes (train + eval);
2. factor slot order (fixed 0=C, 1=Pt, 2=Pv);
3. incoming mean source->target direction;
4. no self-loop (isolated source of a directed edge gets exactly zero delta);
5. empty graph -> graph delta exactly 0;
6. no cross-factor leakage inside the graph block (slots C/Pv bitwise
   unchanged when only Pt messages are perturbed);
7. gradients reach the factorizer, V/D/LN maps and fusion;
8. eval forward vs inference equivalence;
9. no [E,3,3,...] giant edge-pair tensor (profiler shape audit).

O1.5 (causal cleanup) adds the strict matched no-topology control
``variant=diag_nograph`` (O1.5-A):
10. unknown variant / num_layers != 1 validation;
11. diag_nograph param count == diag_id and identical init values;
12. diag_nograph graph delta exactly 0 (routed path N=0 -> D(0)=0);
13. changing edge_index never changes diag_nograph outputs (bitwise);
14. diag_nograph output == LayerNorm(H0) then the original fusion.

O2-A (static cross-ownership utility) adds the capacity-matched pair
``static_cross`` / ``static_cross_dup`` (15-26) and O2-B1 (target-conditioned
Null-vs-Transfer routing) adds ``conditional_cross`` / ``conditional_cross_dup``
(27-46): shared-router structure, target-only router input (no source-content
leak), zero-init p = 0.5, init-state equivalence to DIAG-ID, payload causality,
the two-step router-gradient cascade, effective <= raw magnitudes, degenerate
graph behaviour and the aux_info key surface.

Style: synthetic tensors only, no data fixtures (matches tests/ conventions).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from omegaconf import OmegaConf

from src.models import oft_mag
from src.models.oft_components import (
    CROSS_PAIR_KEYS,
    CROSS_PAIR_TARGET_SLOT,
    ConditionalCrossLayer,
    DiagIDLayer,
    SLOT_NAMES,
    StaticCrossLayer,
    binary_entropy,
    incoming_mean,
    post_transition_stats,
)
from src.models.oft_mag import Model, stack_factor_slots


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def _make_cfg(**overrides) -> OmegaConf:
    base = {
        "hidden_dim": 16,
        "factor_dim": 8,
        "dropout": 0.0,
        "activation": "gelu",
        "norm": "layernorm",
        "num_layers": 1,
        "lambda_common": 0.1,
        "lambda_orth": 0.01,
        "lambda_recon": 0.1,
        "orth_fallback_batch": 4,
        "full_graph_training": True,
    }
    base.update(overrides)
    return OmegaConf.create({"model": base})


def _random_graph(num_nodes: int, n_edges: int, seed: int = 0, allow_self: bool = False):
    """Directed random edges (source -> target), no self-loops by default."""
    g = torch.Generator().manual_seed(seed)
    edges = set()
    while len(edges) < n_edges:
        s = int(torch.randint(num_nodes, (1,), generator=g).item())
        t = int(torch.randint(num_nodes, (1,), generator=g).item())
        if allow_self or s != t:
            edges.add((s, t))
    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    return edge_index


def _make_data(num_nodes: int = 20, feat_t: int = 5, feat_v: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, feat_t + feat_v, generator=g)
    data_info = {
        "text_dim": feat_t,
        "visual_dim": feat_v,
        "input_dim": x.size(1),
        "num_nodes": num_nodes,
        "num_classes": 5,
    }
    return x, data_info


def _make_model(**overrides) -> Model:
    x, data_info = _make_data()
    model = Model(_make_cfg(**overrides), data_info)
    return model, x


def _manual_incoming_mean(X, edge_index, num_nodes):
    """Independent recomputation (different summation order from index_add_)."""
    out = torch.zeros_like(X)
    deg = torch.zeros(num_nodes, dtype=X.dtype)
    for r in range(edge_index.size(1)):
        s = int(edge_index[0, r])
        t = int(edge_index[1, r])
        out[t] = out[t] + X[s]
        deg[t] = deg[t] + 1.0
    return out / deg.clamp_min(1.0).view(num_nodes, 1, 1)


# ----------------------------------------------------------------------
# 1. Interface / shapes
# ----------------------------------------------------------------------


def test_forward_interface_train_and_eval():
    model, x = _make_model()
    edge_index = _random_graph(x.size(0), 24, seed=1)
    n = x.size(0)

    # train mode: (z, None, None, scalar aux_loss, aux_info)
    model.train()
    z, emb, logits, aux_loss, aux_info = model(x, edge_index)
    assert z.shape == (n, model.out_dim)
    assert z.dtype == torch.float32
    assert emb is None and logits is None
    assert aux_loss.dim() == 0
    # P0 diagnostics preserved + the six new oft_l1_* diagnostics.
    for key in (
        "p0_common_loss", "p0_orth_loss", "p0_recon_loss",
        "p0_common_sim", "p0_private_sim",
        "p0_c_norm", "p0_pt_norm", "p0_pv_norm",
        "p0_cp_overlap_t", "p0_cp_overlap_v",
        "oft_l1_c_diag_update_ratio", "oft_l1_pt_diag_update_ratio",
        "oft_l1_pv_diag_update_ratio",
        "oft_l1_c_neighbor_norm", "oft_l1_pt_neighbor_norm",
        "oft_l1_pv_neighbor_norm",
    ):
        assert key in aux_info, f"missing aux_info key {key}"
    for key in (
        "oft_l1_c_diag_update_ratio", "oft_l1_pt_diag_update_ratio",
        "oft_l1_pv_diag_update_ratio", "oft_l1_c_neighbor_norm",
        "oft_l1_pt_neighbor_norm", "oft_l1_pv_neighbor_norm",
    ):
        assert torch.is_tensor(aux_info[key]) and aux_info[key].numel() == 1

    # eval mode: P0 convention -> zero aux_loss, empty aux_info.
    model.eval()
    z_eval, _, _, aux_loss_eval, aux_info_eval = model(x, edge_index)
    assert z_eval.shape == (n, model.out_dim)
    assert float(aux_loss_eval) == 0.0
    assert aux_info_eval == {}

    # no-edge forward is also valid (topology-free fallback)
    z_no_edge, _, _, _, _ = model(x, None)
    assert z_no_edge.shape == (n, model.out_dim)


def test_num_layers_must_be_one():
    x, data_info = _make_data()
    with pytest.raises(ValueError, match="num_layers=1"):
        Model(_make_cfg(num_layers=2), data_info)


def test_out_dim_is_hidden_dim():
    model, _ = _make_model(hidden_dim=16)
    assert model.out_dim == 16


# ----------------------------------------------------------------------
# 2. Factor slot order (fixed 0=C, 1=Pt, 2=Pv)
# ----------------------------------------------------------------------


def test_factor_slot_order_fixed_c_pt_pv():
    g = torch.Generator().manual_seed(7)
    n, f = 11, 8
    factors = {
        "c": torch.randn(n, f, generator=g),
        "p_t": torch.randn(n, f, generator=g),
        "p_v": torch.randn(n, f, generator=g),
    }
    H = stack_factor_slots(factors)
    assert H.shape == (n, 3, f)
    assert torch.equal(H[:, 0], factors["c"])
    assert torch.equal(H[:, 1], factors["p_t"])
    assert torch.equal(H[:, 2], factors["p_v"])
    # wrong pairing must NOT hold (guards a permutation regression)
    assert not torch.equal(H[:, 0], factors["p_t"])
    assert not torch.equal(H[:, 1], factors["p_v"])

    # factorizer output keys used by the model match the fixed slot map.
    assert list(oft_mag._FACTOR_KEYS) == ["c", "p_t", "p_v"]
    assert list(SLOT_NAMES) == ["c", "pt", "pv"]
    # the factorizer emits exactly the keys the stacker consumes
    model, x = _make_model()
    with torch.no_grad():
        x_t, x_v = model._split_modalities(x)
        out = model.factorizer(x_t, x_v)
    for key in ("c", "p_t", "p_v"):
        assert key in out


# ----------------------------------------------------------------------
# 3. Incoming mean source->target
# ----------------------------------------------------------------------


def test_incoming_mean_direction_and_magnitude():
    num_nodes = 16
    X = torch.randn(num_nodes, 3, 5)
    edge_index = _random_graph(num_nodes, 40, seed=3)
    N = incoming_mean(X, edge_index, num_nodes)
    N_manual = _manual_incoming_mean(X, edge_index, num_nodes)
    assert N.shape == X.shape
    assert torch.allclose(N, N_manual, atol=1e-5, rtol=1e-5)

    # strictly directed: perturbing a source changes targets, not the source
    X2 = X.clone()
    srcs = edge_index[0].unique()
    X2[srcs] = X2[srcs] + 1.0
    N2 = incoming_mean(X2, edge_index, num_nodes)
    tgt = edge_index[1]
    assert not torch.allclose(N2[tgt], N[tgt], atol=1e-5)
    # nodes with zero in-degree are untouched and exactly zero
    in_deg = torch.zeros(num_nodes, dtype=torch.long)
    in_deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long))
    isolated = (in_deg == 0).nonzero().flatten()
    if isolated.numel() > 0:
        assert torch.equal(N2[isolated], torch.zeros_like(N2[isolated]))
        assert torch.equal(N[isolated], torch.zeros_like(N[isolated]))


# ----------------------------------------------------------------------
# 4. No self-loop
# ----------------------------------------------------------------------


def test_no_self_loop_single_directed_edge():
    """With exactly one edge 0 -> 1 the source 0 is untouched (no implicit
    self-loop added): its H_next is exactly LayerNorm(H0), the target 1
    receives exactly the single-edge message."""
    torch.manual_seed(0)
    model, x = _make_model()
    n = x.size(0)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # 0 -> 1
    layer = model.diag_layers[0]

    with torch.no_grad():
        x_t, x_v = model._split_modalities(x)
        factors = model.factorizer(x_t, x_v)
        H0 = stack_factor_slots(factors)  # [n, 3, F]
        H1, stats = layer.propagate(H0, edge_index)

    for a in range(3):
        # source 0: no incoming edge -> delta exactly 0 -> H1 = LN(H0)
        h0_src = H0[0, a]
        expected_src = layer.norm[a](h0_src + torch.zeros_like(h0_src))
        assert torch.equal(H1[0, a], expected_src)
        # target 1: exactly one message from node 0 (fp-tolerance compare:
        # the model's delta goes through stacked/transposed matmul paths)
        X = layer.V[a](H0[:, a])
        delta1 = layer.D[a](X[0])
        assert torch.allclose(H1[1, a], layer.norm[a](H0[1, a] + delta1), atol=1e-5, rtol=1e-5)
        # other nodes (2..n): isolated -> exactly LN(H0)
        for i in range(2, n):
            assert torch.equal(
                H1[i, a], layer.norm[a](H0[i, a] + torch.zeros_like(H0[i, a]))
            )

    # stats: with one in-edge the update ratio / neighbor norm are positive
    # (node 1 receives a real message), while the isolated majority is zero.
    neighbor_norm = stats["neighbor_norm"]
    assert neighbor_norm.shape == (3,)
    assert (neighbor_norm > 0.0).all()
    assert (stats["diag_update_ratio"] > 0.0).all()


# ----------------------------------------------------------------------
# 5. Empty graph -> delta exactly 0
# ----------------------------------------------------------------------


def test_empty_graph_delta_exactly_zero():
    torch.manual_seed(0)
    model, x = _make_model()
    empty = torch.empty((2, 0), dtype=torch.long)
    layer = model.diag_layers[0]

    with torch.no_grad():
        x_t, x_v = model._split_modalities(x)
        factors = model.factorizer(x_t, x_v)
        H0 = stack_factor_slots(factors)

    # pure aggregation: no edges -> exact zeros
    N = incoming_mean(H0, empty, H0.size(0))
    assert torch.equal(N, torch.zeros_like(N))

    # propagation: delta = D(0) = 0 exactly, so H_next == LN(H + 0)
    H1, stats = layer.propagate(H0, empty)
    for a in range(3):
        assert torch.equal(H1[:, a], layer.norm[a](H0[:, a] + torch.zeros_like(H0[:, a])))
    assert torch.equal(stats["diag_update_ratio"], torch.zeros(3))
    assert torch.equal(stats["neighbor_norm"], torch.zeros(3))

    # model-level: None and an empty edge set are exactly equivalent
    model.eval()
    with torch.no_grad():
        z_none, _, _, _, _ = model(x, None)
        z_empty, _, _, _, _ = model(x, empty)
    assert torch.equal(z_none, z_empty)


# ----------------------------------------------------------------------
# 6. No cross-factor leakage inside the graph block
# ----------------------------------------------------------------------


def test_no_cross_factor_leakage_before_fusion():
    torch.manual_seed(0)
    model, x = _make_model()
    num_nodes = x.size(0)
    edge_index = _random_graph(num_nodes, 30, seed=11)
    layer = model.diag_layers[0]

    with torch.no_grad():
        x_t, x_v = model._split_modalities(x)
        factors = model.factorizer(x_t, x_v)
        H0 = stack_factor_slots(factors)
        H1, _ = layer.propagate(H0, edge_index)

        # perturb ONLY the Pt messages (slot 1 of the source nodes)
        H0b = H0.clone()
        srcs = edge_index[0].unique()
        H0b[srcs, 1] = H0b[srcs, 1] + 1.0
        H1b, _ = layer.propagate(H0b, edge_index)

    # slots 0 (C) and 2 (Pv) must be bitwise unchanged
    assert torch.equal(H1b[:, 0], H1[:, 0])
    assert torch.equal(H1b[:, 2], H1[:, 2])
    # slot 1 (Pt) must actually have changed somewhere
    assert not torch.equal(H1b[:, 1], H1[:, 1])


# ----------------------------------------------------------------------
# 7. Gradients
# ----------------------------------------------------------------------


def test_gradients_reach_all_trained_components():
    torch.manual_seed(0)
    model, x = _make_model()
    edge_index = _random_graph(x.size(0), 30, seed=5)
    model.train()
    z, _, _, aux_loss, _ = model(x, edge_index)
    assert float(aux_loss) > 0.0
    loss = z.mean() + aux_loss
    loss.backward()

    # every trainable block must receive a non-zero gradient
    checks = {
        "factorizer.projector": model.factorizer.text_projector.net[0].weight,
        "factorizer.common_encoder": model.factorizer.common_encoder[0].weight,
        "factorizer.private_text": model.factorizer.private_text_encoder[0].weight,
        "factorizer.private_visual": model.factorizer.private_visual_encoder[0].weight,
        "recon_text_head": model.recon_text_head.net[0].weight,
        "recon_visual_head": model.recon_visual_head.net[0].weight,
        "diag_layer.V[c]": model.diag_layers[0].V[0].weight,
        "diag_layer.V[pt]": model.diag_layers[0].V[1].weight,
        "diag_layer.V[pv]": model.diag_layers[0].V[2].weight,
        "diag_layer.D[c]": model.diag_layers[0].D[0].weight,
        "diag_layer.D[pt]": model.diag_layers[0].D[1].weight,
        "diag_layer.D[pv]": model.diag_layers[0].D[2].weight,
        "diag_layer.norm[c]": model.diag_layers[0].norm[0].weight,
        "fusion": model.fusion[0].weight,
    }
    for name, param in checks.items():
        assert param.grad is not None, f"{name} got no gradient"
        assert param.grad.abs().sum().item() > 0.0, f"{name} gradient is zero"


# ----------------------------------------------------------------------
# 8. Eval forward vs inference equivalence
# ----------------------------------------------------------------------


def test_eval_forward_equals_inference():
    torch.manual_seed(0)
    model, x = _make_model()
    edge_index = _random_graph(x.size(0), 24, seed=9)
    model.eval()
    with torch.no_grad():
        z_fwd, _, _, _, _ = model(x, edge_index)
        z_inf = model.inference(x, edge_index, device=torch.device("cpu"))
    assert z_inf.device.type == "cpu"
    assert z_fwd.shape == z_inf.shape
    assert torch.allclose(z_fwd, z_inf, atol=1e-5, rtol=1e-5)

    # inference also works on a graph-free input (interface compatibility)
    z_no_edge = model.inference(x, None, device=torch.device("cpu"))
    assert z_no_edge.shape == z_fwd.shape


# ----------------------------------------------------------------------
# 9. No giant [E,3,3,...] edge-pair tensor
# ----------------------------------------------------------------------


def test_no_giant_edge_pair_tensor():
    """Profiler shape audit: no op input may carry an [E, S, S, ...] shape
    (the signature of a pairwise edge tensor / dense adjacency build)."""
    torch.manual_seed(0)
    model, x = _make_model()
    num_nodes, n_edges = 512, 4096
    edge_index = _random_graph(num_nodes, n_edges, seed=21, allow_self=False)
    layer = model.diag_layers[0]
    with torch.no_grad():
        x_t, x_v = model._split_modalities(x)
        factors = model.factorizer(x_t, x_v)
        H = stack_factor_slots(factors)
        H_big = torch.cat([H, torch.zeros(num_nodes - H.size(0), 3, H.size(-1))], dim=0)

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
        ) as prof:
            layer.propagate(H_big, edge_index)
    except (torch.profiler.ProfilerError, RuntimeError):
        pytest.skip("torch profiler with record_shapes unavailable")
    S = 3
    offenders = []
    for evt in prof.events():
        for shape in getattr(evt, "input_shapes", None) or []:
            if len(shape) >= 3 and shape[0] == n_edges and shape[1] == S and shape[2] == S:
                offenders.append((evt.key, shape))
    assert not offenders, f"giant [E,S,S,...] tensor ops: {offenders[:5]}"


# ----------------------------------------------------------------------
# 10-14. O1.5: diag_nograph matched no-topology control
# ----------------------------------------------------------------------


def test_variant_must_be_known():
    x, data_info = _make_data()
    with pytest.raises(ValueError, match="variant"):
        Model(_make_cfg(variant="diag_banana"), data_info)
    # diag_nograph is the same single-layer scaffold as diag_id
    with pytest.raises(ValueError, match="num_layers=1"):
        Model(_make_cfg(variant="diag_nograph", num_layers=2), data_info)
    # variant defaults to diag_id when absent (backward compatible)
    m, _ = _make_model()
    assert m.variant == "diag_id"


def test_diag_nograph_parameters_match_diag_id_exactly():
    """O1.5 matched control: identical parameter count AND identical
    initialization (same RNG stream -> same values, state dict key-identical),
    so diag_nograph differs from diag_id only in the graph response."""
    x, data_info = _make_data()
    cfg_id = _make_cfg(variant="diag_id")
    cfg_no = _make_cfg(variant="diag_nograph")
    torch.manual_seed(123)
    m_id = Model(cfg_id, data_info)
    torch.manual_seed(123)
    m_no = Model(cfg_no, data_info)

    sd_id, sd_no = m_id.state_dict(), m_no.state_dict()
    assert set(sd_id) == set(sd_no)
    numel = 0
    for key in sd_id:
        assert sd_id[key].shape == sd_no[key].shape, key
        assert torch.equal(sd_id[key], sd_no[key]), f"{key} init differs"
        numel += sd_id[key].numel()
    n_id = sum(p.numel() for p in m_id.parameters())
    n_no = sum(p.numel() for p in m_no.parameters())
    assert n_id == n_no == numel
    # every graph-block map stays instantiated in the control
    layer = m_no.diag_layers[0]
    assert len(layer.V) == 3 and len(layer.D) == 3 and len(layer.norm) == 3


def test_diag_nograph_graph_delta_exactly_zero():
    """The path diag_nograph actually executes (edge input routed to None):
    N == 0 exactly, delta = D(0) == 0 exactly (D bias=False), stats exactly
    0, and H_next == per-slot LayerNorm(H0)."""
    torch.manual_seed(0)
    model, x = _make_model(variant="diag_nograph")
    layer = model.diag_layers[0]
    with torch.no_grad():
        x_t, x_v = model._split_modalities(x)
        factors = model.factorizer(x_t, x_v)
        H0 = stack_factor_slots(factors)
        H1, stats = layer.propagate(H0, None)
    assert torch.equal(stats["diag_update_ratio"], torch.zeros(3))
    assert torch.equal(stats["neighbor_norm"], torch.zeros(3))
    for a in range(3):
        assert torch.equal(
            H1[:, a], layer.norm[a](H0[:, a] + torch.zeros_like(H0[:, a]))
        )

    # training forward on a real graph: P0 aux losses intact, oft stats zero
    model.train()
    edge_index = _random_graph(x.size(0), 24, seed=1)
    z, _, _, aux_loss, aux_info = model(x, edge_index)
    assert z.shape == (x.size(0), model.out_dim)
    assert float(aux_loss) > 0.0
    assert "p0_cp_overlap_t" in aux_info  # P0 diagnostics still emitted
    for key in (
        "oft_l1_c_diag_update_ratio", "oft_l1_pt_diag_update_ratio",
        "oft_l1_pv_diag_update_ratio",
        "oft_l1_c_neighbor_norm", "oft_l1_pt_neighbor_norm",
        "oft_l1_pv_neighbor_norm",
    ):
        assert float(aux_info[key]) == 0.0, key


def test_diag_nograph_edge_index_invariance():
    """Changing edge_index must not change diag_nograph's H_next (hence z):
    any non-empty edge set and None give bitwise identical outputs."""
    torch.manual_seed(0)
    model, x = _make_model(variant="diag_nograph")
    model.eval()
    e1 = _random_graph(x.size(0), 24, seed=1)
    e2 = _random_graph(x.size(0), 40, seed=2)
    with torch.no_grad():
        z1, _, _, _, _ = model(x, e1)
        z2, _, _, _, _ = model(x, e2)
        z0, _, _, _, _ = model(x, None)
    assert torch.equal(z1, z2)
    assert torch.equal(z1, z0)


def test_diag_nograph_output_equals_layernorm_then_fusion():
    """diag_nograph(x, edge_index) must equal LayerNorm(H0) per slot followed
    by the original P0 fusion — the exact no-topology scaffold semantics."""
    torch.manual_seed(0)
    model, x = _make_model(variant="diag_nograph")
    layer = model.diag_layers[0]
    edge_index = _random_graph(x.size(0), 24, seed=5)
    model.eval()
    with torch.no_grad():
        x_t, x_v = model._split_modalities(x)
        factors = model.factorizer(x_t, x_v)
        H0 = stack_factor_slots(factors)
        H1_manual = torch.stack(
            [layer.norm[a](H0[:, a]) for a in range(3)], dim=1
        )
        z_manual = model.fusion(
            torch.cat([H1_manual[:, a] for a in range(3)], dim=-1)
        )
        z, _, _, _, _ = model(x, edge_index)
    assert torch.equal(z, z_manual)


# ----------------------------------------------------------------------
# 15-26. O2-A: static cross-ownership utility test
# ----------------------------------------------------------------------


def _randomize_cross(layer: StaticCrossLayer, seed: int = 0, scale: float = 0.1) -> None:
    """Break the zero-init for tests that need a live cross branch."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for key in CROSS_PAIR_KEYS:
            layer.cross[key].weight.copy_(
                torch.randn(layer.factor_dim, layer.factor_dim, generator=g) * scale
            )


def test_static_cross_and_dup_parameter_counts_match():
    """STATIC-CROSS and its capacity-matched control must have identical
    parameter counts, exactly diag_id + 6 * factor_dim^2 (the six C_{a->b})."""
    x, data_info = _make_data()
    fd = 8
    models = {}
    for variant in ("diag_id", "static_cross_dup", "static_cross"):
        torch.manual_seed(11)
        models[variant] = Model(_make_cfg(variant=variant, factor_dim=fd), data_info)

    n_id = sum(p.numel() for p in models["diag_id"].parameters())
    n_dup = sum(p.numel() for p in models["static_cross_dup"].parameters())
    n_cross = sum(p.numel() for p in models["static_cross"].parameters())
    assert n_dup == n_cross
    assert n_cross - n_id == len(CROSS_PAIR_KEYS) * fd * fd == 6 * fd * fd
    # the six maps are the only new parameters
    assert len(models["static_cross"].diag_layers[0].cross) == 6


def test_static_cross_and_dup_state_dict_identical_init():
    """Same RNG stream -> identical state dicts (keys, shapes, values): the two
    O2-A variants differ only in which slot the cross branch reads."""
    x, data_info = _make_data()
    torch.manual_seed(123)
    m_dup = Model(_make_cfg(variant="static_cross_dup"), data_info)
    torch.manual_seed(123)
    m_cross = Model(_make_cfg(variant="static_cross"), data_info)

    sd_dup, sd_cross = m_dup.state_dict(), m_cross.state_dict()
    assert set(sd_dup) == set(sd_cross)
    for key in sd_dup:
        assert sd_dup[key].shape == sd_cross[key].shape, key
        assert torch.equal(sd_dup[key], sd_cross[key]), f"{key} init differs"
    assert m_dup.diag_layers[0].dup is True
    assert m_cross.diag_layers[0].dup is False


def test_six_cross_matrices_exist_only_off_diagonal():
    """Exactly six off-diagonal pair modules; no diagonal (a == b) module, since
    the existing per-slot D_a already owns the diagonal path."""
    layer = StaticCrossLayer(factor_dim=8)
    assert tuple(layer.cross.keys()) == CROSS_PAIR_KEYS
    assert len(CROSS_PAIR_KEYS) == 6
    for key in CROSS_PAIR_KEYS:
        a_name, b_name = key.split("_to_")
        assert a_name != b_name
        assert layer.cross[key].weight.shape == (8, 8)
        assert layer.cross[key].bias is None
        assert CROSS_PAIR_TARGET_SLOT[key] == SLOT_NAMES.index(b_name)
    # no identity / diagonal module anywhere
    assert "c_to_c" not in layer.cross and "pt_to_pt" not in layer.cross and "pv_to_pv" not in layer.cross


def test_cross_matrices_zero_initialized():
    """Zero init: at step 0 Delta_cross == 0 for both variants, so both start
    strictly as DIAG-ID."""
    for dup in (False, True):
        layer = StaticCrossLayer(factor_dim=8, dup=dup)
        for key in CROSS_PAIR_KEYS:
            assert torch.count_nonzero(layer.cross[key].weight) == 0, key
        N = torch.randn(7, 3, 8, generator=torch.Generator().manual_seed(4))
        assert torch.equal(layer.cross_update(N), torch.zeros(7, 3, 8))


def test_init_forward_equals_diag_path():
    """With zero-init cross weights, both O2-A variants must reproduce the
    DIAG-ID forward exactly (same seed -> same backbone weights)."""
    torch.manual_seed(0)
    x, data_info = _make_data()
    edge_index = _random_graph(x.size(0), 30, seed=13)
    torch.manual_seed(5)
    m_id = Model(_make_cfg(variant="diag_id"), data_info).eval()
    torch.manual_seed(5)
    m_dup = Model(_make_cfg(variant="static_cross_dup"), data_info).eval()
    torch.manual_seed(5)
    m_cross = Model(_make_cfg(variant="static_cross"), data_info).eval()

    with torch.no_grad():
        z_id, _, _, _, _ = m_id(x, edge_index)
        z_dup, _, _, _, _ = m_dup(x, edge_index)
        z_cross, _, _, _, _ = m_cross(x, edge_index)
        H0_id, H1_id, _ = m_id.encode_states(x, edge_index, device=torch.device("cpu"))
        _, H1_dup, stats_dup = m_dup.encode_states(x, edge_index, device=torch.device("cpu"))
    assert torch.equal(z_id, z_dup)
    assert torch.equal(z_id, z_cross)
    assert torch.equal(H1_id, H1_dup)
    # the cross branch is live but exactly zero at init
    assert torch.equal(stats_dup["cross_update_ratio"], torch.zeros(3))
    assert torch.equal(stats_dup["cross_to_diag_ratio"], torch.zeros(3))
    for key in CROSS_PAIR_KEYS:
        assert float(stats_dup[f"cross_pair_ratio_{key}"]) == 0.0


def test_static_cross_messages_follow_source_slot():
    """STATIC-CROSS: perturbing only the C source state must change exactly the
    two C-sourced pair messages (C->Pt, C->Pv) and leave the other four
    bitwise unchanged."""
    layer = StaticCrossLayer(factor_dim=8, dup=False)
    _randomize_cross(layer, seed=1)
    N = torch.randn(12, 3, 8, generator=torch.Generator().manual_seed(2))
    base = layer.cross_messages(N)
    perturbed = N.clone()
    perturbed[:, 0] = perturbed[:, 0] + 1.0  # source slot C only

    msgs = layer.cross_messages(perturbed)
    assert not torch.equal(msgs["c_to_pt"], base["c_to_pt"])
    assert not torch.equal(msgs["c_to_pv"], base["c_to_pv"])
    for key in ("pt_to_c", "pt_to_pv", "pv_to_c", "pv_to_pt"):
        assert torch.equal(msgs[key], base[key]), f"{key} must not depend on N^C"

    # same causal structure at the target-update level: the two targets that
    # read N^C move, the target C (whose sources are Pt/Pv) does not.
    delta_base = layer.cross_update(N)
    delta = layer.cross_update(perturbed)
    assert not torch.equal(delta[:, 1], delta_base[:, 1])  # target Pt <- C
    assert not torch.equal(delta[:, 2], delta_base[:, 2])  # target Pv <- C
    assert torch.equal(delta[:, 0], delta_base[:, 0])      # target C reads Pt/Pv


def test_static_cross_dup_ignores_other_source_states():
    """STATIC-CROSS-DUP: the fake cross path reads only N^b, so changing another
    ownership source N^a (a != b) must leave target b's cross update bitwise
    unchanged — extra capacity without extra ownership information."""
    layer = StaticCrossLayer(factor_dim=8, dup=True)
    _randomize_cross(layer, seed=3)
    N = torch.randn(12, 3, 8, generator=torch.Generator().manual_seed(6))
    base = layer.cross_update(N)
    perturbed = N.clone()
    perturbed[:, 0] = perturbed[:, 0] + 1.0  # perturb source ownership C

    delta = layer.cross_update(perturbed)
    assert torch.equal(delta[:, 1], base[:, 1]), "target Pt must ignore N^C"
    assert torch.equal(delta[:, 2], base[:, 2]), "target Pv must ignore N^C"
    assert not torch.equal(delta[:, 0], base[:, 0])  # reads N^C as its own state

    # every pair message reads its target slot, never the other source slot
    msgs = layer.cross_messages(perturbed)
    msgs_base = layer.cross_messages(N)
    for key in CROSS_PAIR_KEYS:
        b = CROSS_PAIR_TARGET_SLOT[key]
        if b == 0:
            assert not torch.equal(msgs[key], msgs_base[key]), key
        else:
            assert torch.equal(msgs[key], msgs_base[key]), key


def test_empty_graph_diag_and_cross_updates_exactly_zero():
    """Empty graph -> N = 0 -> diag delta = D(0) = 0 and cross delta = C(0) = 0
    exactly, for both O2-A variants."""
    empty = torch.empty((2, 0), dtype=torch.long)
    for dup in (False, True):
        layer = StaticCrossLayer(factor_dim=8, dup=dup)
        _randomize_cross(layer, seed=7)  # live cross weights, still C(0) = 0
        H = torch.randn(10, 3, 8, generator=torch.Generator().manual_seed(8))
        H_next, stats = layer.propagate(H, empty)
        for a in range(3):
            assert torch.equal(H_next[:, a], layer.norm[a](H[:, a]))
        assert torch.equal(stats["diag_update_ratio"], torch.zeros(3))
        assert torch.equal(stats["neighbor_norm"], torch.zeros(3))
        assert torch.equal(stats["cross_update_ratio"], torch.zeros(3))
        assert torch.equal(stats["cross_to_diag_ratio"], torch.zeros(3))
        for key in CROSS_PAIR_KEYS:
            assert float(stats[f"cross_pair_ratio_{key}"]) == 0.0


def test_isolated_node_graph_updates_exactly_zero():
    """A node with no in-edges receives no diag and no cross update: its H_next
    is exactly LayerNorm(H), for both O2-A variants."""
    edge_index = torch.tensor([[1, 2, 3], [2, 3, 1]], dtype=torch.long)  # node 0 isolated
    for dup in (False, True):
        layer = StaticCrossLayer(factor_dim=8, dup=dup)
        _randomize_cross(layer, seed=9)
        H = torch.randn(6, 3, 8, generator=torch.Generator().manual_seed(10))
        H_next, _ = layer.propagate(H, edge_index)
        for a in range(3):
            assert torch.equal(
                H_next[0, a], layer.norm[a](H[0, a] + torch.zeros_like(H[0, a]))
            )


def test_cross_gradients_nonzero_after_one_step():
    """One optimizer step on a toy graph with enough edges: every one of the six
    C_{a->b} must receive a non-zero gradient."""
    torch.manual_seed(0)
    model, x = _make_model(variant="static_cross")
    edge_index = _random_graph(x.size(0), 40, seed=15)
    model.train()
    z, _, _, aux_loss, _ = model(x, edge_index)
    (z.mean() + aux_loss).backward()
    layer = model.diag_layers[0]
    for key in CROSS_PAIR_KEYS:
        grad = layer.cross[key].weight.grad
        assert grad is not None, f"{key} got no gradient"
        assert grad.abs().sum().item() > 0.0, f"{key} gradient is zero"


def test_diag_variants_aux_keys_unchanged():
    """Regression guard: diag_id / diag_nograph keep exactly their O1/O1.5
    aux_info key set — no O2-A cross or post-transition keys leak in."""
    edge_index = None
    for variant in ("diag_id", "diag_nograph"):
        torch.manual_seed(0)
        model, x = _make_model(variant=variant)
        edge_index = _random_graph(x.size(0), 24, seed=1)
        model.train()
        _, _, _, _, aux_info = model(x, edge_index)
        assert len(aux_info) == 16, f"{variant} key count changed: {sorted(aux_info)}"
        for key in aux_info:
            assert not key.startswith("oft_l1_c_to_"), key
            assert not key.startswith("oft_l1_pt_to_"), key
            assert not key.startswith("oft_l1_pv_to_"), key
            assert "post_cos" not in key and "state_drift" not in key, key


def test_cross_variants_aux_keys_present():
    """The O2-A diagnostics required by the plan are emitted in training mode."""
    for variant in ("static_cross", "static_cross_dup"):
        torch.manual_seed(0)
        model, x = _make_model(variant=variant)
        edge_index = _random_graph(x.size(0), 30, seed=17)
        model.train()
        _, _, _, aux_loss, aux_info = model(x, edge_index)
        assert float(aux_loss) > 0.0
        for slot in SLOT_NAMES:
            assert f"oft_l1_{slot}_cross_update_ratio" in aux_info
            assert f"oft_l1_{slot}_cross_to_diag_ratio" in aux_info
            assert f"oft_l1_post_norm_{slot}" in aux_info
            assert f"oft_l1_state_drift_{slot}" in aux_info
            assert f"oft_l1_{slot}_diag_update_ratio" in aux_info
        for key in CROSS_PAIR_KEYS:
            assert f"oft_l1_{key}_cross_pair_ratio" in aux_info
        for pair in ("c_pt", "c_pv", "pt_pv"):
            assert f"oft_l1_post_cos_{pair}" in aux_info
            assert f"oft_l1_pre_cos_{pair}" in aux_info


def test_post_transition_stats_semantics():
    """post_transition_stats: identical H0/H1 -> drift 0, post_cos == pre_cos."""
    torch.manual_seed(0)
    H = torch.randn(9, 3, 8, generator=torch.Generator().manual_seed(19))
    stats = post_transition_stats(H, H)
    for slot in SLOT_NAMES:
        assert float(stats[f"state_drift_{slot}"]) == pytest.approx(0.0, abs=1e-6)
    for pair in ("c_pt", "c_pv", "pt_pv"):
        assert float(stats[f"post_cos_{pair}"]) == pytest.approx(
            float(stats[f"pre_cos_{pair}"]), abs=1e-6
        )


def test_no_giant_edge_pair_tensor_static_cross():
    """Profiler shape audit for the cross layer: still no [E, S, S, ...] op."""
    torch.manual_seed(0)
    layer = StaticCrossLayer(factor_dim=8)
    _randomize_cross(layer, seed=21)
    num_nodes, n_edges = 512, 4096
    edge_index = _random_graph(num_nodes, n_edges, seed=23)
    H = torch.randn(num_nodes, 3, 8, generator=torch.Generator().manual_seed(25))
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
        ) as prof:
            layer.propagate(H, edge_index)
    except (torch.profiler.ProfilerError, RuntimeError):
        pytest.skip("torch profiler with record_shapes unavailable")
    S = 3
    offenders = []
    for evt in prof.events():
        for shape in getattr(evt, "input_shapes", None) or []:
            if len(shape) >= 3 and shape[0] == n_edges and shape[1] == S and shape[2] == S:
                offenders.append((evt.key, shape))
    assert not offenders, f"giant [E,S,S,...] tensor ops: {offenders[:5]}"


def test_cross_variants_enforce_single_layer():
    """O2-A is a single-layer scaffold like O1 / O1.5."""
    x, data_info = _make_data()
    for variant in ("static_cross", "static_cross_dup"):
        with pytest.raises(ValueError, match="num_layers=1"):
            Model(_make_cfg(variant=variant, num_layers=2), data_info)


def test_encode_states_matches_forward_h1():
    """The offline diagnostics hook must expose exactly the H1 the forward uses."""
    torch.manual_seed(0)
    model, x = _make_model(variant="static_cross")
    _randomize_cross(model.diag_layers[0], seed=27)
    edge_index = _random_graph(x.size(0), 26, seed=29)
    model.eval()
    with torch.no_grad():
        H0, H1, stats = model.encode_states(x, edge_index, device=torch.device("cpu"))
        layer = model.diag_layers[0]
        H1_manual, stats_manual = layer.propagate(H0, edge_index)
    assert torch.equal(H1, H1_manual)
    for key in ("cross_update_ratio", "cross_to_diag_ratio"):
        assert torch.equal(stats[key], stats_manual[key]), key
    # post-transition hook is consistent with the returned states
    post = post_transition_stats(H0, H1)
    assert float(post["post_cos_c_pt"]) == pytest.approx(
        float(post_transition_stats(H0, H1_manual)["post_cos_c_pt"]), abs=1e-6
    )


# ----------------------------------------------------------------------
# 27-46. O2-B1: target-conditioned Null-vs-Transfer routing
# ----------------------------------------------------------------------


def _randomize_router(layer: ConditionalCrossLayer, seed: int = 0, scale: float = 0.1) -> None:
    """Break the zero-init router head so routing becomes input-dependent."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in layer.router.parameters():
            param.copy_(torch.randn(param.shape, generator=g) * scale)
        layer.factor_embed.weight.copy_(
            torch.randn(layer.factor_embed.weight.shape, generator=g) * scale
        )


def _make_conditional(variant: str, seed: int = 0, **cfg_over):
    torch.manual_seed(seed)
    return _make_model(variant=variant, **cfg_over)


def test_conditional_variants_parameter_counts_match():
    """CONDITIONAL-CROSS and its capacity-matched control must have identical
    parameter counts: diag_id + 6*F^2 (O2-A maps) + router (embedding + 2 linears)."""
    x, data_info = _make_data()
    fd, ed = 8, 16
    models = {}
    for variant in ("diag_id", "conditional_cross_dup", "conditional_cross"):
        torch.manual_seed(11)
        models[variant] = Model(_make_cfg(variant=variant, factor_dim=fd), data_info)

    n_id = sum(p.numel() for p in models["diag_id"].parameters())
    n_dup = sum(p.numel() for p in models["conditional_cross_dup"].parameters())
    n_cross = sum(p.numel() for p in models["conditional_cross"].parameters())
    router_params = (
        3 * ed + (2 * fd + 2 * ed) * 128 + 128 + 128 * 2 + 2
    )
    assert n_dup == n_cross
    assert n_cross - n_id == 6 * fd * fd + router_params


def test_conditional_variants_state_dict_identical_init():
    """Same RNG stream -> identical state dicts: the ONLY difference between the
    two O2-B1 variants is which slot the routed payload reads."""
    x, data_info = _make_data()
    torch.manual_seed(123)
    m_dup = Model(_make_cfg(variant="conditional_cross_dup"), data_info)
    torch.manual_seed(123)
    m_cross = Model(_make_cfg(variant="conditional_cross"), data_info)

    sd_dup, sd_cross = m_dup.state_dict(), m_cross.state_dict()
    assert set(sd_dup) == set(sd_cross)
    for key in sd_dup:
        assert sd_dup[key].shape == sd_cross[key].shape, key
        assert torch.equal(sd_dup[key], sd_cross[key]), f"{key} init differs"
    assert m_dup.diag_layers[0].dup is True
    assert m_cross.diag_layers[0].dup is False


def test_conditional_router_is_shared_single_module():
    """ONE shared router (Linear->GELU->Linear) plus ONE factor embedding table
    for all six pairs; pair identity comes from e_a/e_b, not from six MLPs."""
    layer = ConditionalCrossLayer(factor_dim=8)
    assert isinstance(layer.router, nn.Sequential)
    assert len(layer.router) == 3
    assert isinstance(layer.router[0], nn.Linear) and isinstance(layer.router[2], nn.Linear)
    assert layer.router[0].in_features == layer.router_input_dim == 2 * 8 + 2 * 16
    assert layer.router[0].out_features == 128 and layer.router[2].out_features == 2
    assert layer.factor_embed.weight.shape == (3, 16)
    # exactly V(3) + D(3) + cross(6) + router(2) linear layers; no per-pair router
    linears = [name for name, mod in layer.named_modules() if isinstance(mod, nn.Linear)]
    assert len(linears) == 3 + 3 + 6 + 2
    assert sum(1 for name in linears if name.startswith("router")) == 2
    for key in CROSS_PAIR_KEYS:
        assert not any(isinstance(m, nn.Linear) for m in layer.cross[key].children())


def test_router_reads_target_context_only():
    """CONTROL DISCIPLINE: perturbing a SOURCE slot (H^a / N^a, a != b) must not
    change that pair's router logits bitwise — otherwise CONDITIONAL-DUP could
    still see source ownership content and the matched control is contaminated."""
    layer = ConditionalCrossLayer(factor_dim=8)
    _randomize_router(layer, seed=5)
    H = torch.randn(11, 3, 8, generator=torch.Generator().manual_seed(6))
    N = torch.randn(11, 3, 8, generator=torch.Generator().manual_seed(7))

    for key in CROSS_PAIR_KEYS:
        a_name, b_name = key.split("_to_")
        a, b = SLOT_NAMES.index(a_name), SLOT_NAMES.index(b_name)
        base = layer.pair_logits(H, N, a, b)

        H_src = H.clone()
        H_src[:, a] = H_src[:, a] + 3.0
        N_src = N.clone()
        N_src[:, a] = N_src[:, a] + 3.0
        assert torch.equal(layer.pair_logits(H_src, N, a, b), base), key
        assert torch.equal(layer.pair_logits(H, N_src, a, b), base), key

        # sanity: the test is not vacuous — the TARGET context does move logits
        H_tgt = H.clone()
        H_tgt[:, b] = H_tgt[:, b] + 3.0
        assert not torch.equal(layer.pair_logits(H_tgt, N, a, b), base), key


def test_router_probabilities_are_valid_distribution():
    """p is a proper 2-way softmax: shapes [N], in [0, 1], p_null + p_transfer = 1."""
    layer = ConditionalCrossLayer(factor_dim=8)
    _randomize_router(layer, seed=9)
    H = torch.randn(13, 3, 8, generator=torch.Generator().manual_seed(10))
    N = torch.randn(13, 3, 8, generator=torch.Generator().manual_seed(11))
    transfer = layer.router_transfer(H, N)
    assert set(transfer) == set(CROSS_PAIR_KEYS)

    for key in CROSS_PAIR_KEYS:
        a_name, b_name = key.split("_to_")
        a, b = SLOT_NAMES.index(a_name), SLOT_NAMES.index(b_name)
        probs = torch.softmax(layer.pair_logits(H, N, a, b), dim=-1)
        assert probs.shape == (13, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(13), atol=1e-6)
        assert torch.equal(transfer[key], probs[:, 1])
        assert bool((transfer[key] >= 0).all() and (transfer[key] <= 1).all())
        ent = binary_entropy(transfer[key])
        assert bool((ent >= 0).all() and (ent <= float(torch.log(torch.tensor(2.0))) + 1e-6).all())


def test_router_zero_init_gives_half_transfer():
    """Zero-initialized router head (weight and bias): step 0 logits are exactly
    0 and p_null = p_transfer = 0.5 for every node and pair."""
    layer = ConditionalCrossLayer(factor_dim=8)
    assert torch.count_nonzero(layer.router[2].weight) == 0
    assert torch.count_nonzero(layer.router[2].bias) == 0
    H = torch.randn(9, 3, 8, generator=torch.Generator().manual_seed(12))
    N = torch.randn(9, 3, 8, generator=torch.Generator().manual_seed(13))
    for key in CROSS_PAIR_KEYS:
        a_name, b_name = key.split("_to_")
        a, b = SLOT_NAMES.index(a_name), SLOT_NAMES.index(b_name)
        assert torch.equal(layer.pair_logits(H, N, a, b), torch.zeros(9, 2))
        assert torch.equal(layer.router_transfer(H, N)[key], torch.full((9,), 0.5))


def test_conditional_transfer_matrices_zero_initialized():
    """The six O2-A transfer maps keep their strict zero init in O2-B1."""
    for dup in (False, True):
        layer = ConditionalCrossLayer(factor_dim=8, dup=dup)
        for key in CROSS_PAIR_KEYS:
            assert torch.count_nonzero(layer.cross[key].weight) == 0, key
        H = torch.randn(7, 3, 8, generator=torch.Generator().manual_seed(14))
        N = torch.randn(7, 3, 8, generator=torch.Generator().manual_seed(15))
        assert torch.equal(layer.conditional_cross_messages(H, N)[CROSS_PAIR_KEYS[0]],
                           torch.zeros(7, 8))


def test_conditional_init_forward_equals_diag_path():
    """p = 0.5 and T = 0 at init, so m = p * T(.) = 0 exactly: both O2-B1
    variants must reproduce DIAG-ID forward bitwise."""
    torch.manual_seed(0)
    x, data_info = _make_data()
    edge_index = _random_graph(x.size(0), 30, seed=13)
    torch.manual_seed(5)
    m_id = Model(_make_cfg(variant="diag_id"), data_info).eval()
    torch.manual_seed(5)
    m_dup = Model(_make_cfg(variant="conditional_cross_dup"), data_info).eval()
    torch.manual_seed(5)
    m_cross = Model(_make_cfg(variant="conditional_cross"), data_info).eval()

    with torch.no_grad():
        z_id, _, _, _, _ = m_id(x, edge_index)
        z_dup, _, _, _, _ = m_dup(x, edge_index)
        z_cross, _, _, _, _ = m_cross(x, edge_index)
        H0_id, H1_id, _ = m_id.encode_states(x, edge_index, device=torch.device("cpu"))
        _, H1_dup, stats_dup = m_dup.encode_states(x, edge_index, device=torch.device("cpu"))
    assert torch.equal(z_id, z_dup)
    assert torch.equal(z_id, z_cross)
    assert torch.equal(H1_id, H1_dup)
    assert torch.equal(stats_dup["cross_update_ratio"], torch.zeros(3))
    assert torch.equal(stats_dup["cross_to_diag_ratio"], torch.zeros(3))
    for key in CROSS_PAIR_KEYS:
        assert float(stats_dup[f"cross_pair_ratio_{key}"]) == 0.0
        assert float(stats_dup[f"effective_cross_pair_ratio_{key}"]) == 0.0
        assert float(stats_dup[f"transfer_mean_{key}"]) == 0.5
        assert float(stats_dup[f"transfer_std_{key}"]) == 0.0


def test_conditional_cross_payload_follows_source_slot():
    """CONDITIONAL-CROSS causality when only the C ownership state moves.

    Raw payloads: exactly the two C-sourced payloads change. Router: only the
    pairs TARGETING C change (C is part of their target context) — the pairs
    sourced by C keep their probability bitwise, because the router never reads
    source content. Routed messages therefore move on four pairs (two via the
    payload, two via the target-C router context) and stay bitwise on the two
    pairs that neither source nor target C.
    """
    layer = ConditionalCrossLayer(factor_dim=8, dup=False)
    _randomize_cross(layer, seed=1)
    _randomize_router(layer, seed=2)
    H = torch.randn(12, 3, 8, generator=torch.Generator().manual_seed(3))
    N = torch.randn(12, 3, 8, generator=torch.Generator().manual_seed(4))
    perturbed = N.clone()
    perturbed[:, 0] = perturbed[:, 0] + 1.0  # source slot C only

    # raw payloads follow the source slot
    base_payloads = layer.cross_messages(N)
    payloads = layer.cross_messages(perturbed)
    assert not torch.equal(payloads["c_to_pt"], base_payloads["c_to_pt"])
    assert not torch.equal(payloads["c_to_pv"], base_payloads["c_to_pv"])
    for key in ("pt_to_c", "pt_to_pv", "pv_to_c", "pv_to_pt"):
        assert torch.equal(payloads[key], base_payloads[key]), f"{key} payload"

    # router probability moves only for pairs whose TARGET is the perturbed slot
    p_base = layer.router_transfer(H, N)
    p_pert = layer.router_transfer(H, perturbed)
    for key in CROSS_PAIR_KEYS:
        a_name, b_name = key.split("_to_")
        a, b = SLOT_NAMES.index(a_name), SLOT_NAMES.index(b_name)
        if b == 0:
            assert not torch.equal(p_pert[key], p_base[key]), key
        else:
            assert torch.equal(p_pert[key], p_base[key]), key

    # routed messages: the four pairs touching C move, the other two do not
    base = layer.conditional_cross_messages(H, N)
    msgs = layer.conditional_cross_messages(H, perturbed)
    for key in ("c_to_pt", "c_to_pv", "pt_to_c", "pv_to_c"):
        assert not torch.equal(msgs[key], base[key]), key
    for key in ("pt_to_pv", "pv_to_pt"):
        assert torch.equal(msgs[key], base[key]), key

    delta_base = layer.combine_cross(base)
    delta = layer.combine_cross(msgs)
    for b in range(3):  # every target is reachable from C through one of the two
        assert not torch.equal(delta[:, b], delta_base[:, b]), b


def test_conditional_dup_payload_ignores_other_source_states():
    """CONDITIONAL-DUP: the routed payload reads only N^b, so changing another
    ownership source N^a (a != b) leaves target b's routed message bitwise
    unchanged — same conditional machinery, no ownership information."""
    layer = ConditionalCrossLayer(factor_dim=8, dup=True)
    _randomize_cross(layer, seed=3)
    _randomize_router(layer, seed=4)
    H = torch.randn(12, 3, 8, generator=torch.Generator().manual_seed(5))
    N = torch.randn(12, 3, 8, generator=torch.Generator().manual_seed(6))
    base = layer.conditional_cross_messages(H, N)
    perturbed = N.clone()
    perturbed[:, 0] = perturbed[:, 0] + 1.0  # perturb source ownership C

    msgs = layer.conditional_cross_messages(H, perturbed)
    for key in CROSS_PAIR_KEYS:
        b = CROSS_PAIR_TARGET_SLOT[key]
        if b == 0:
            assert not torch.equal(msgs[key], base[key]), key
        else:
            assert torch.equal(msgs[key], base[key]), key

    delta_base = layer.combine_cross(base)
    delta = layer.combine_cross(msgs)
    assert torch.equal(delta[:, 1], delta_base[:, 1])
    assert torch.equal(delta[:, 2], delta_base[:, 2])
    assert not torch.equal(delta[:, 0], delta_base[:, 0])


def test_conditional_variants_router_logits_match():
    """Both variants share the same router definition: given the same target
    context they must return bitwise-identical logits and probabilities."""
    x, data_info = _make_data()
    torch.manual_seed(77)
    m_dup = Model(_make_cfg(variant="conditional_cross_dup"), data_info)
    torch.manual_seed(77)
    m_cross = Model(_make_cfg(variant="conditional_cross"), data_info)
    l_dup, l_cross = m_dup.diag_layers[0], m_cross.diag_layers[0]
    # same seed -> same perturbation on both, since their init is identical
    _randomize_router(l_dup, seed=8)
    _randomize_router(l_cross, seed=8)

    H = torch.randn(10, 3, 8, generator=torch.Generator().manual_seed(9))
    N = torch.randn(10, 3, 8, generator=torch.Generator().manual_seed(10))
    p_dup = l_dup.router_transfer(H, N)
    p_cross = l_cross.router_transfer(H, N)
    for key in CROSS_PAIR_KEYS:
        assert torch.equal(p_dup[key], p_cross[key]), key


def test_first_backward_transfer_grad_nonzero_router_grad_zero():
    """Step 1: dL/dp ∝ T(x) = 0, so the router gets EXACTLY zero gradient while
    all six transfer maps do receive gradient. Expected, not a bug."""
    torch.manual_seed(0)
    model, x = _make_model(variant="conditional_cross")
    edge_index = _random_graph(x.size(0), 40, seed=31)
    model.train()
    z, _, _, aux_loss, _ = model(x, edge_index)
    (z.mean() + aux_loss).backward()
    layer = model.diag_layers[0]
    for key in CROSS_PAIR_KEYS:
        grad = layer.cross[key].weight.grad
        assert grad is not None and grad.abs().sum().item() > 0.0, key
    assert layer.router[0].weight.grad.abs().sum().item() == 0.0
    assert layer.router[2].weight.grad.abs().sum().item() == 0.0
    assert layer.factor_embed.weight.grad.abs().sum().item() == 0.0


def test_second_backward_router_grad_nonzero():
    """After the transfer maps move once, the router head receives non-zero
    gradient (its trunk needs one more step because the head was zero-init)."""
    torch.manual_seed(0)
    model, x = _make_model(variant="conditional_cross")
    edge_index = _random_graph(x.size(0), 40, seed=31)
    model.train()
    z, _, _, _, _ = model(x, edge_index)
    z.mean().backward()
    layer = model.diag_layers[0]
    with torch.no_grad():
        g = torch.Generator().manual_seed(33)
        for key in CROSS_PAIR_KEYS:
            layer.cross[key].weight.add_(
                torch.randn(8, 8, generator=g) * 0.05
            )
    model.zero_grad(set_to_none=True)

    z, _, _, _, _ = model(x, edge_index)
    z.mean().backward()
    assert layer.router[2].weight.grad.abs().sum().item() > 0.0
    assert layer.router[2].bias.grad.abs().sum().item() > 0.0
    # trunk still waits: dL/dh = dL/dlogits @ W_head, and W_head was zero
    assert layer.router[0].weight.grad.abs().sum().item() == 0.0


def test_conditional_empty_graph_updates_exactly_zero():
    """Empty graph -> N = 0 -> payload = T(0) = 0 exactly, so diag and routed
    cross updates are zero for both variants (the router still emits p)."""
    empty = torch.empty((2, 0), dtype=torch.long)
    for dup in (False, True):
        layer = ConditionalCrossLayer(factor_dim=8, dup=dup)
        _randomize_cross(layer, seed=7)
        _randomize_router(layer, seed=8)
        H = torch.randn(10, 3, 8, generator=torch.Generator().manual_seed(9))
        H_next, stats = layer.propagate(H, empty)
        for a in range(3):
            assert torch.equal(H_next[:, a], layer.norm[a](H[:, a]))
        assert torch.equal(stats["diag_update_ratio"], torch.zeros(3))
        assert torch.equal(stats["neighbor_norm"], torch.zeros(3))
        assert torch.equal(stats["cross_update_ratio"], torch.zeros(3))
        assert torch.equal(stats["cross_to_diag_ratio"], torch.zeros(3))
        for key in CROSS_PAIR_KEYS:
            assert float(stats[f"cross_pair_ratio_{key}"]) == 0.0
            assert float(stats[f"effective_cross_pair_ratio_{key}"]) == 0.0


def test_conditional_isolated_node_graph_update_exactly_zero():
    """A node with no in-edges gets H_next == LayerNorm(H) exactly, both variants."""
    edge_index = torch.tensor([[1, 2, 3], [2, 3, 1]], dtype=torch.long)  # node 0 isolated
    for dup in (False, True):
        layer = ConditionalCrossLayer(factor_dim=8, dup=dup)
        _randomize_cross(layer, seed=10)
        _randomize_router(layer, seed=11)
        H = torch.randn(6, 3, 8, generator=torch.Generator().manual_seed(12))
        H_next, _ = layer.propagate(H, edge_index)
        for a in range(3):
            assert torch.equal(
                H_next[0, a], layer.norm[a](H[0, a] + torch.zeros_like(H[0, a]))
            )


def test_conditional_aux_keys_present_and_static_unchanged():
    """Key-surface regression: conditional variants emit the O2-A keys plus the
    router/effective keys (64 total with P0); static variants stay at 40 and
    diag variants at 16 — no conditional key leaks into an older variant."""
    counts = {}
    for variant in ("diag_id", "static_cross", "conditional_cross"):
        torch.manual_seed(0)
        model, x = _make_model(variant=variant)
        edge_index = _random_graph(x.size(0), 30, seed=17)
        model.train()
        _, _, _, aux_loss, aux_info = model(x, edge_index)
        assert float(aux_loss) > 0.0
        counts[variant] = len(aux_info)
        for slot in SLOT_NAMES:
            assert f"oft_l1_{slot}_diag_update_ratio" in aux_info
    assert counts == {"diag_id": 16, "static_cross": 40, "conditional_cross": 64}

    torch.manual_seed(0)
    model, x = _make_model(variant="conditional_cross")
    edge_index = _random_graph(x.size(0), 30, seed=17)
    model.train()
    _, _, _, _, aux_info = model(x, edge_index)
    for key in CROSS_PAIR_KEYS:
        assert f"oft_l1_{key}_transfer_mean" in aux_info
        assert f"oft_l1_{key}_transfer_std" in aux_info
        assert f"oft_l1_{key}_router_entropy" in aux_info
        assert f"oft_l1_{key}_effective_cross_pair_ratio" in aux_info
        assert f"oft_l1_{key}_cross_pair_ratio" in aux_info
    for key, value in aux_info.items():
        assert value.numel() == 1, key  # only scalars reach the log surface

    # the static variants must NOT grow the conditional keys
    torch.manual_seed(0)
    model, x = _make_model(variant="static_cross")
    edge_index = _random_graph(x.size(0), 30, seed=17)
    model.train()
    _, _, _, _, aux_info = model(x, edge_index)
    assert not any("transfer_mean" in key or "router_entropy" in key for key in aux_info)


def test_effective_pair_ratio_bounded_by_raw():
    """Routed magnitude is p * raw with p <= 1, so the effective ratio can never
    exceed the raw one, and neither is numerically anomalous."""
    layer = ConditionalCrossLayer(factor_dim=8)
    _randomize_cross(layer, seed=13)
    _randomize_router(layer, seed=14)
    H = torch.randn(64, 3, 8, generator=torch.Generator().manual_seed(15))
    edge_index = _random_graph(64, 400, seed=16)
    _, stats = layer.propagate(H, edge_index)
    for key in CROSS_PAIR_KEYS:
        raw = float(stats[f"cross_pair_ratio_{key}"])
        eff = float(stats[f"effective_cross_pair_ratio_{key}"])
        assert eff <= raw + 1e-6, (key, raw, eff)
        assert 0.0 <= eff <= 10.0 and torch.isfinite(torch.tensor([raw, eff])).all()
        p = float(stats[f"transfer_mean_{key}"])
        assert 0.0 <= p <= 1.0


def test_conditional_no_giant_edge_pair_tensor():
    """Profiler shape audit for the conditional layer: still no [E, S, S, ...] op."""
    torch.manual_seed(0)
    layer = ConditionalCrossLayer(factor_dim=8)
    _randomize_cross(layer, seed=21)
    _randomize_router(layer, seed=22)
    num_nodes, n_edges = 512, 4096
    edge_index = _random_graph(num_nodes, n_edges, seed=23)
    H = torch.randn(num_nodes, 3, 8, generator=torch.Generator().manual_seed(25))
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
        ) as prof:
            layer.propagate(H, edge_index)
    except (torch.profiler.ProfilerError, RuntimeError):
        pytest.skip("torch profiler with record_shapes unavailable")
    S = 3
    offenders = []
    for evt in prof.events():
        for shape in getattr(evt, "input_shapes", None) or []:
            if len(shape) >= 3 and shape[0] == n_edges and shape[1] == S and shape[2] == S:
                offenders.append((evt.key, shape))
    assert not offenders, f"giant [E,S,S,...] tensor ops: {offenders[:5]}"


def test_conditional_variants_enforce_single_layer():
    """O2-B1 is a single-layer scaffold like O1 / O1.5 / O2-A."""
    x, data_info = _make_data()
    for variant in ("conditional_cross", "conditional_cross_dup"):
        with pytest.raises(ValueError, match="num_layers=1"):
            Model(_make_cfg(variant=variant, num_layers=2), data_info)


def test_conditional_encode_states_matches_forward_h1():
    """The offline diagnostics hook exposes exactly the H1 the forward uses, and
    the diagnostic helper recomputes the same N the layer routes on."""
    torch.manual_seed(0)
    model, x = _make_model(variant="conditional_cross")
    layer = model.diag_layers[0]
    _randomize_cross(layer, seed=27)
    _randomize_router(layer, seed=28)
    edge_index = _random_graph(x.size(0), 26, seed=29)
    model.eval()
    with torch.no_grad():
        H0, H1, stats = model.encode_states(x, edge_index, device=torch.device("cpu"))
        H1_manual, stats_manual = layer.propagate(H0, edge_index)
        N = layer.neighbor_states(H0, edge_index)
        transfer = layer.router_transfer(H0, N)
    assert torch.equal(H1, H1_manual)
    for key in ("cross_update_ratio", "cross_to_diag_ratio"):
        assert torch.equal(stats[key], stats_manual[key]), key
    for key in CROSS_PAIR_KEYS:
        assert torch.equal(
            stats[f"transfer_mean_{key}"], transfer[key].mean()
        ), key
        assert torch.equal(
            stats[f"effective_cross_pair_ratio_{key}"],
            stats_manual[f"effective_cross_pair_ratio_{key}"],
        ), key
    post = post_transition_stats(H0, H1)
    assert float(post["post_cos_c_pt"]) == pytest.approx(
        float(post_transition_stats(H0, H1_manual)["post_cos_c_pt"]), abs=1e-6
    )
    assert torch.isfinite(H1).all() and torch.isfinite(H0).all()
