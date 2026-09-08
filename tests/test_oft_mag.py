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

Style: synthetic tensors only, no data fixtures (matches tests/ conventions).
"""

from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf

from src.models import oft_mag
from src.models.oft_components import DiagIDLayer, SLOT_NAMES, incoming_mean
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
