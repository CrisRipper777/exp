"""Locks the R0 first/second-order estimator algebra (plan §8, Check 4):

    Q^(1)_T = -g^T a_T
    Q^(2)_T = -g^T a_T - 0.5 a_T^T C a_T
    S^(2)   = Q^(2)_TV - Q^(2)_T - Q^(2)_V
    S^(2)   = -a_T^T C a_V                          (cross term identity)

with C = diag(p) - p p^T the logit-space CE Hessian. Pure float64 algebra on
synthetic tensors; no model involved.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _q1(g: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return -(g * a).sum(dim=-1)


def _q2(g: torch.Tensor, c_mat: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    ca = torch.einsum("scd,sd->sc", c_mat, a)
    return _q1(g, a) - 0.5 * (a * ca).sum(dim=-1)


def _make_logit_space(s: int = 64, c: int = 6, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(s, c, dtype=torch.float64, generator=g)
    p = F.softmax(o, dim=-1)
    y = torch.randint(c, (s,), generator=g)
    y_onehot = F.one_hot(y, c).double()
    gvec = p - y_onehot
    c_mat = torch.diag_embed(p) - torch.einsum("sc,sd->scd", p, p)
    a_t = torch.randn(s, c, dtype=torch.float64, generator=g)
    a_v = torch.randn(s, c, dtype=torch.float64, generator=g)
    return gvec, c_mat, a_t, a_v


def test_hessian_structure():
    """The logit-space CE Hessian C = diag(p) - p p^T must have zero row sums
    (constant offsets in logits leave softmax/CE unchanged) and be symmetric."""
    g = torch.Generator().manual_seed(3)
    p = F.softmax(torch.randn(8, 5, dtype=torch.float64, generator=g), dim=-1)
    c = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
    row_sums = c.sum(dim=-1)
    assert row_sums.abs().max().item() < 1e-12
    assert torch.allclose(c, c.transpose(-1, -2), atol=1e-12)


def test_s_second_cross_identity():
    """Check 4: S2 computed as difference of Q2's equals -a_T^T C a_V."""
    gvec, c_mat, a_t, a_v = _make_logit_space()
    q_t2 = _q2(gvec, c_mat, a_t)
    q_v2 = _q2(gvec, c_mat, a_v)
    q_tv2 = _q2(gvec, c_mat, a_t + a_v)
    s_second = q_tv2 - q_t2 - q_v2
    s_cross = -torch.einsum("sc,scd,sd->s", a_t, c_mat, a_v)
    assert torch.allclose(s_second, s_cross, atol=1e-12)


def test_first_order_interaction_is_zero():
    """Plan §8.4: S^(1) = 0 is a theoretical result, not a bug."""
    gvec, _, a_t, a_v = _make_logit_space()
    q_t1 = _q1(gvec, a_t)
    q_v1 = _q1(gvec, a_v)
    q_tv1 = _q1(gvec, a_t + a_v)
    s_first = q_tv1 - q_t1 - q_v1
    assert s_first.abs().max().item() < 1e-12


def test_second_order_recovers_first_order_plus_curvature():
    """Q2 differs from Q1 exactly by the -0.5 a^T C a curvature term."""
    gvec, c_mat, a_t, _ = _make_logit_space()
    q_t1 = _q1(gvec, a_t)
    q_t2 = _q2(gvec, c_mat, a_t)
    ca = torch.einsum("scd,sd->sc", c_mat, a_t)
    curvature = -0.5 * (a_t * ca).sum(dim=-1)
    assert torch.allclose(q_t2, q_t1 + curvature, atol=1e-12)
