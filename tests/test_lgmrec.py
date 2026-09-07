"""Unit tests for the LGMRec port (OpenMAG reference, unified plain-CE
protocol — no InfoNCE aux)."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.lgmrec import Model

N = 17
TEXT_DIM = 13
VISUAL_DIM = 19
HIDDEN = 32


def _make_cfg() -> object:
    return OmegaConf.create({
        "model": {
            "name": "lgmrec",
            "hidden_dim": HIDDEN,
            "num_layers": 3,
            "dropout": 0.2,
            "hyper_num": 16,
            "alpha": 0.1,
            "lr": 5e-3,
            "weight_decay": 1e-5,
        }
    })


def _make_info() -> dict:
    return {
        "input_dim": TEXT_DIM + VISUAL_DIM,
        "num_nodes": N,
        "num_classes": 5,
        "text_dim": TEXT_DIM,
        "visual_dim": VISUAL_DIM,
    }


def _make_x(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(N, TEXT_DIM + VISUAL_DIM, generator=generator)


def _make_edge(seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, N, (2, 60), generator=generator)


def test_lgmrec_forward_shapes_and_finite() -> None:
    model = Model(_make_cfg(), _make_info())
    model.train()
    z, second, third, aux, aux_info = model(_make_x(), _make_edge())
    assert z.shape == (N, HIDDEN)
    assert second is None and third is None
    assert torch.isfinite(z).all()
    assert aux.ndim == 0 and aux.item() == 0.0  # unified protocol: no aux
    assert aux_info == {}


def test_lgmrec_no_aux_modules() -> None:
    """The OpenMAG InfoNCE heads/decoders are dropped under the unified
    protocol (no dead params)."""
    model = Model(_make_cfg(), _make_info())
    for name in ("vision_head", "text_head", "decoder_v", "decoder_t"):
        assert not hasattr(model, name)


def test_lgmrec_gradient_flow() -> None:
    model = Model(_make_cfg(), _make_info())
    model.train()
    z, _, _, aux, _ = model(_make_x(), _make_edge())
    (z.square().mean() + aux).backward()
    for component in (model.feat_encoder, model.hgnn.hyper_projector):
        for p in component.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"{component} bad grad"


def test_lgmrec_inference_equivalence() -> None:
    model = Model(_make_cfg(), _make_info())
    model.eval()
    x = _make_x()
    edge_index = _make_edge()
    z_fwd, _, _, _, _ = model(x, edge_index)
    z_inf = model.inference(x, edge_index, device=torch.device("cpu"))
    assert z_inf.shape == (N, HIDDEN)
    assert torch.allclose(z_fwd, z_inf, atol=1e-6)


def test_lgmrec_out_dim() -> None:
    model = Model(_make_cfg(), _make_info())
    assert model.out_dim == HIDDEN


def test_lgmrec_no_nan_random_inputs() -> None:
    model = Model(_make_cfg(), _make_info())
    for seed in range(3):
        model.train()
        z, _, _, aux, _ = model(_make_x(seed=seed), _make_edge(seed=seed))
        assert torch.isfinite(z).all(), f"NaN at seed {seed}"
        assert torch.isfinite(aux)
        model.eval()
        z, _, _, _, _ = model(_make_x(seed=seed), _make_edge(seed=seed))
        assert torch.isfinite(z).all()
