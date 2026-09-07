"""Unit tests for the R0 SimpleMAGProbe.

Verifies (R0 plan §3.5 + §21, invariant Check 1):
- output shapes;
- per-edge source/target alignment of edge_delta_* with edge_index[:, r];
- the additive decomposition invariant
      z_full[i] = h_self[i] + sum_{r: target(r)=i}(delta_text[r] + delta_visual[r])
  to <1e-5 (recomputed through an independent mask-sum path).

Style: synthetic tensors only, no data fixtures (matches tests/ conventions).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.simple_mag_probe import Model, _normalize_edges


def _make_cfg(**overrides) -> OmegaConf:
    base = {
        "hidden_dim": 8,
        "num_proj_layers": 2,
        "dropout": 0.0,
        "activation": "relu",
        "norm": "none",
        "normalize_modalities": True,
    }
    base.update(overrides)
    return OmegaConf.create({"model": base})


def _make_graph(num_nodes=14, feat_t=5, feat_v=4, n_pairs=9, seed=0):
    """Undirected random graph stored bidirectionally (loader convention):
    no self-loops, each undirected pair appears as both directed edges."""
    g = torch.Generator().manual_seed(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n_pairs:
        a = int(torch.randint(num_nodes, (1,), generator=g).item())
        b = int(torch.randint(num_nodes, (1,), generator=g).item())
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    pairs = sorted(pairs)
    edge_index = torch.tensor(
        [p for pair in pairs for p in (pair, (pair[1], pair[0]))],
        dtype=torch.long,
    ).t().contiguous()
    x = torch.randn(num_nodes, feat_t + feat_v, generator=g)
    return x, edge_index, feat_t, feat_v


def _independent_decomposition(diag, edge_index, num_nodes):
    """Recompute z_full[i] = h_self[i] + sum over incoming edges via boolean
    masking + torch.sum (different summation order than the model's
    index_add_); tolerances absorb float32 ordering noise."""
    h_self = diag["h_self"]
    dt = diag["edge_delta_text"]
    dv = diag["edge_delta_visual"]
    z_man = torch.zeros_like(h_self)
    for i in range(num_nodes):
        mask = edge_index[1] == i
        z_man[i] = h_self[i] + dt[mask].sum(dim=0) + dv[mask].sum(dim=0)
    return z_man


def _model_and_graph(**model_overrides):
    x, edge_index, feat_t, feat_v = _make_graph()
    data_info = {
        "text_dim": feat_t,
        "visual_dim": feat_v,
        "num_classes": 5,
        "input_dim": x.size(1),
        "num_nodes": x.size(0),
    }
    model = Model(_make_cfg(**model_overrides), data_info)
    model.eval()
    return model, x, edge_index, feat_t, feat_v


def test_shapes():
    model, x, edge_index, _, _ = _model_and_graph()
    diag = model.forward_with_deltas(x, edge_index)
    n, h = x.size(0), model.out_dim
    e = edge_index.size(1)
    assert diag["h_text"].shape == (n, h)
    assert diag["h_visual"].shape == (n, h)
    assert diag["h_self"].shape == (n, h)
    assert diag["z_full"].shape == (n, h)
    assert diag["edge_delta_text"].shape == (e, h)
    assert diag["edge_delta_visual"].shape == (e, h)
    assert diag["edge_norm"].shape == (e,)


def test_additive_decomposition_invariant():
    """Check 1: z_full[i] == h_self[i] + sum_{j->i}(delta_T + delta_V) < 1e-5."""
    model, x, edge_index, _, _ = _model_and_graph()
    diag = model.forward_with_deltas(x, edge_index)
    z_man = _independent_decomposition(diag, edge_index, x.size(0))
    max_err = (diag["z_full"] - z_man).abs().max().item()
    assert max_err < 1e-5, f"decomposition error {max_err} >= 1e-5"


def test_forward_equals_diag_z():
    """The representation the classifier sees (forward) is exactly z_full."""
    model, x, edge_index, _, _ = _model_and_graph()
    z_fwd, _, _, _, _ = model.forward(x, edge_index)
    z_diag = model.forward_with_deltas(x, edge_index)["z_full"]
    assert torch.equal(z_fwd, z_diag)


def test_edge_alignment_source_to_target():
    """delta_text[r] is alpha[r] * W_T h_text_{source(r)}, with
    source = edge_index[0, r], target = edge_index[1, r]."""
    model, x, edge_index, feat_t, feat_v = _model_and_graph()
    diag = model.forward_with_deltas(x, edge_index)
    h_t, h_v = diag["h_text"], diag["h_visual"]
    # Per-node messages from the SAME projection the model used:
    msg_t = model.w_msg_t(h_t)
    msg_v = model.w_msg_v(h_v)
    r = torch.randint(edge_index.size(1), (1,)).item()
    src = int(edge_index[0, r])
    tgt = int(edge_index[1, r])
    assert src != tgt  # loader convention: no self-loops
    expected_t = diag["edge_norm"][r] * msg_t[src]
    expected_v = diag["edge_norm"][r] * msg_v[src]
    assert torch.allclose(diag["edge_delta_text"][r], expected_t, atol=1e-5, rtol=1e-5)
    assert torch.allclose(diag["edge_delta_visual"][r], expected_v, atol=1e-5, rtol=1e-5)
    # And the message must depend on the SOURCE's features, not the target's:
    assert not torch.allclose(diag["edge_delta_text"][r], diag["edge_norm"][r] * msg_t[tgt], atol=1e-3)


def test_edge_norm_value():
    """alpha_ij == 1 / sqrt(d_i d_j) with degrees counted on the graph."""
    model, x, edge_index, _, _ = _model_and_graph()
    diag = model.forward_with_deltas(x, edge_index)
    deg = torch.zeros(x.size(0))
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
    r = torch.randint(edge_index.size(1), (1,)).item()
    src, tgt = int(edge_index[0, r]), int(edge_index[1, r])
    expected = 1.0 / torch.sqrt(deg[src] * deg[tgt])
    assert torch.allclose(diag["edge_norm"][r], expected, atol=1e-6)


def test_normalized_edges_helper():
    """_normalize_edges returns identical alpha for both directions."""
    x, edge_index, _, _ = _make_graph()
    edge_index, alpha = _normalize_edges(edge_index, num_nodes=x.size(0), dtype=torch.float32)
    assert edge_index.size(1) == alpha.numel()
    # Graph is stored as [.. (a,b), (b,a) ..] per pair: alphas must match
    # because deg is symmetric on a bidirected graph.
    assert alpha.numel() % 2 == 0
    assert torch.allclose(alpha[0::2], alpha[1::2], atol=1e-6)


def test_no_modality_raises():
    x = torch.randn(10, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data_info = {"text_dim": 0, "visual_dim": 0, "num_classes": 2, "input_dim": 4, "num_nodes": 10}
    try:
        Model(_make_cfg(), data_info)
    except ValueError:
        return
    raise AssertionError("Model must raise when no modality dimensions are provided")


def test_normalize_modalities_off_still_decomposes():
    """Config flip (normalize_modalities=false) keeps the decomposition exact."""
    model, x, edge_index, _, _ = _model_and_graph(normalize_modalities=False)
    diag = model.forward_with_deltas(x, edge_index)
    z_man = _independent_decomposition(diag, edge_index, x.size(0))
    assert (diag["z_full"] - z_man).abs().max().item() < 1e-5


def test_split_modalities_l2_normalizes_by_default():
    """normalize_modalities=true: each modality block is unit-norm per row
    before projection (guards the text/visual scale imbalance)."""
    x, edge_index, feat_t, feat_v = _make_graph()
    x[:, :feat_t] *= 100.0  # blow up text scale, as on MAGB disk
    data_info = {"text_dim": feat_t, "visual_dim": feat_v, "num_classes": 5,
                 "input_dim": x.size(1), "num_nodes": x.size(0)}
    model = Model(_make_cfg(), data_info)
    x_t_out, x_v_out = model._split_modalities(x)
    assert torch.allclose(x_t_out.norm(dim=-1), torch.ones(x.size(0)), atol=1e-4)
    assert torch.allclose(x_v_out.norm(dim=-1), torch.ones(x.size(0)), atol=1e-4)
    # and the raw text scale is ~100, so normalization was doing real work:
    assert x[:, :feat_t].abs().max().item() > 50.0


def test_training_gradient_flow():
    """One training step produces finite gradients on all parameters."""
    model, x, edge_index, feat_t, feat_v = _model_and_graph()
    model.train()
    y = torch.randint(5, (x.size(0),))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    z, _, _, aux_loss, _ = model.forward(x, edge_index)
    loss = F.cross_entropy(z, y) + aux_loss
    loss.backward()
    opt.step()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) == len(list(model.parameters()))
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
