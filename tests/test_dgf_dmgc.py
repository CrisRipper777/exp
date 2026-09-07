from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.dgf import Model as DGFModel
from src.models.dmgc import Model as DMGCModel

TEXT_DIM = 8
VISUAL_DIM = 12
NUM_NODES = 20


def _dgf_cfg(**overrides) -> object:
    base = {
        "name": "dgf",
        "hidden_dim": 16,
        "alpha": 1.0,
        "beta": 1.0,
        "num_layers": 10,
        "dropout": 0.0,
    }
    base.update(overrides)
    return OmegaConf.create({"model": base})


def _dmgc_cfg(**overrides) -> object:
    base = {
        "name": "dmgc",
        "hidden_dim": 16,
        "num_layers": 1,
        "dropout": 0.5,
        "tau": 1.0,
        "lambda_cr": 0.001,
        "lambda_cm": 1.0,
    }
    base.update(overrides)
    return OmegaConf.create({"model": base})


def _data_info() -> dict:
    return {
        "input_dim": TEXT_DIM + VISUAL_DIM,
        "num_nodes": NUM_NODES,
        "num_classes": 4,
        "text_dim": TEXT_DIM,
        "visual_dim": VISUAL_DIM,
    }


def _x(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_NODES, TEXT_DIM + VISUAL_DIM, generator=generator)


def _edge_index(num_edges: int = 30, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, NUM_NODES, (2, num_edges), generator=generator)


# ---------------------------------------------------------------------------
# DGF
# ---------------------------------------------------------------------------


def test_dgf_forward_tuple_and_shapes() -> None:
    model = DGFModel(_dgf_cfg(), _data_info())
    z, second, third, aux_loss, aux_info = model(_x(), _edge_index())
    assert z.shape == (NUM_NODES, 16)
    assert second is None and third is None
    assert aux_loss.ndim == 0
    assert torch.isfinite(z).all()
    assert aux_info == {}


def test_dgf_requires_edge_index() -> None:
    import pytest

    model = DGFModel(_dgf_cfg(), _data_info())
    with pytest.raises(ValueError):
        model(_x(), None)


def test_dgf_inference_matches_forward() -> None:
    model = DGFModel(_dgf_cfg(), _data_info())
    x, edge_index = _x(), _edge_index()
    model.eval()
    z_forward, *_ = model(x, edge_index)
    z_infer = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=7)
    assert torch.allclose(z_forward, z_infer, atol=1e-4)


def test_dgf_output_depends_on_graph() -> None:
    model = DGFModel(_dgf_cfg(), _data_info())
    model.eval()
    z1 = model(_x(), _edge_index(seed=1))[0]
    z2 = model(_x(), _edge_index(seed=2))[0]
    assert not torch.allclose(z1, z2)


def test_dgf_train_eval_identical() -> None:
    # The filtering core has no dropout: train/eval outputs must coincide.
    model = DGFModel(_dgf_cfg(), _data_info())
    x, edge_index = _x(), _edge_index()
    model.eval()
    z_eval = model(x, edge_index)[0]
    model.train()
    z_train = model(x, edge_index)[0]
    assert torch.allclose(z_eval, z_train, atol=1e-6)


def test_dgf_gradients_flow() -> None:
    model = DGFModel(_dgf_cfg(), _data_info())
    model.train()
    z, *_ = model(_x(), _edge_index())
    z.square().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient: {name}"


# ---------------------------------------------------------------------------
# DMGC
# ---------------------------------------------------------------------------


def test_dmgc_forward_tuple_and_shapes() -> None:
    model = DMGCModel(_dmgc_cfg(), _data_info())
    z, second, third, aux_loss, aux_info = model(_x(), _edge_index())
    assert z.shape == (NUM_NODES, 16)
    assert second is None and third is None
    assert aux_loss.ndim == 0
    assert torch.isfinite(z).all()
    assert aux_info == {}


def test_dmgc_requires_edge_index() -> None:
    import pytest

    model = DMGCModel(_dmgc_cfg(), _data_info())
    with pytest.raises(ValueError):
        model(_x(), None)


def test_dmgc_inference_matches_forward() -> None:
    model = DMGCModel(_dmgc_cfg(), _data_info())
    x, edge_index = _x(), _edge_index()
    model.eval()
    z_forward, *_ = model(x, edge_index)
    z_infer = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=7)
    assert torch.allclose(z_forward, z_infer, atol=1e-4)


def test_dmgc_output_depends_on_graph() -> None:
    model = DMGCModel(_dmgc_cfg(), _data_info())
    model.eval()
    z1 = model(_x(), _edge_index(seed=1))[0]
    z2 = model(_x(), _edge_index(seed=2))[0]
    assert not torch.allclose(z1, z2)


def test_dmgc_gradients_flow() -> None:
    model = DMGCModel(_dmgc_cfg(), _data_info())
    model.train()
    z, *_ = model(_x(), _edge_index())
    z.square().mean().backward()
    components = {
        "t_proj": model.t_proj,
        "v_proj": model.v_proj,
        "encoder": model.encoder,
        "att": model.att,
        "fusion_1": model.fusion_1,
        "fusion_2": model.fusion_2,
    }
    for name, component in components.items():
        params = list(component.parameters())
        assert params, f"{name} has no parameters"
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params), name


def test_dmgc_low_high_paths_contribute() -> None:
    # Both the low-pass and high-pass branches must influence the output:
    # zeroing the high-pass branch (L_h weight -> zero) changes z.
    model = DMGCModel(_dmgc_cfg(), _data_info())
    x, edge_index = _x(), _edge_index()
    model.eval()
    z_full = model(x, edge_index)[0]
    with torch.no_grad():
        model.encoder.gnn_encoder_layers[-1].linear.weight.zero_()
        model.encoder.gnn_encoder_layers[-1].linear.bias.zero_()
    z_zeroed = model(x, edge_index)[0]
    assert not torch.allclose(z_full, z_zeroed)
