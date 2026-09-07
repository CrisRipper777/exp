"""Memory fix 1+2 (2026-09-04, user-authorized): activation checkpointing
of the context+scorer segment and the hoisted source-feature gather.

Discipline: both changes are NUMERICS-PRESERVING — recomputation produces
bitwise-identical activations (no dropout/randomness in the segment), and
the hoisted gather yields the identical tensor the per-relation gathers
did. Tests lock this in:

    1. hoisted gather == per-relation gather reference (bitwise)
    2. checkpoint-on vs checkpoint-off forward outputs (bitwise, train+eval)
    3. checkpoint-on vs checkpoint-off gradients (bitwise)
    4. checkpoint lowers the training peak (GPU-only, synthetic graph)
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.biaxis_final import Model as FinalModel
from src.models.biaxis_p1_components import relation_mass, relation_weighted_mean

N, E, K, D = 50, 200, 4, 32


def _make_cfg(memory_checkpoint: bool) -> object:
    return OmegaConf.create({
        "model": {
            "name": "biaxis_final",
            "hidden_dim": 256,
            "factor_dim": 128,
            "dropout": 0.2,
            "activation": "gelu",
            "norm": "layernorm",
            "lambda_common": 0.02,
            "lambda_orth": 0.01,
            "lambda_recon": 0.3,
            "full_graph_training": True,
            "p1": {
                "factor_aware": True, "num_relations": 4, "relation_dim": 32,
                "relation_temperature": 0.5, "selector_hidden_dim": 64,
                "selector_input_norm": None, "budget_hidden_dim": 64,
                "use_graph_budget": True, "budget_shared": False,
                "eps": 1.0e-8, "relation_balance_weight": 0.0,
                "alpha_entropy_weight": 0.0, "budget_reg_weight": 0.0,
                "edge_chunk_size": None,
            },
            "p2": {
                "mode": "null_softmax", "score_hidden_dim": 64, "epsilon": 0.2,
                "tau_base": 1.0, "sinkhorn_iters": 10, "null_prior": 0.5,
                "null_score_init": 0.0, "detach_capacity_prior": True,
                "detach_relation_confidence": True, "eps": 1.0e-8,
            },
            "p3": {
                "operator_mode": "full_interaction", "lowrank_rank": 16,
                "basis_num_bases": 8, "operator_reg_weight": 0.0,
                "interaction_reg_weight": 0.0,
                "memory_checkpoint": memory_checkpoint,
            },
        }
    })


def _make_info() -> dict:
    return {"input_dim": 32, "num_nodes": N, "num_classes": 5,
            "text_dim": 13, "visual_dim": 19}


def _make_x(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(N, 32, generator=generator)


def _make_edge(seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, N, (2, 60), generator=generator)


def _rand(shape, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator)


# ---------------------------------------------------------------------------
# 1. hoisted gather == per-relation gather reference (bitwise)
# ---------------------------------------------------------------------------


def test_hoisted_gather_bitwise_identical() -> None:
    edge_index = torch.randint(0, N, (2, E))
    r = torch.softmax(_rand((E, K), 0), dim=-1)
    features = _rand((N, D), 1)
    g, mass = relation_weighted_mean(edge_index, r, features, N, edge_chunk_size=None)
    # independent reference with the OLD per-relation gather pattern
    src, dst = edge_index[0], edge_index[1]
    acc = torch.zeros(N, K, D, dtype=features.dtype)
    for rel in range(K):
        weighted = r[:, rel].unsqueeze(-1) * features[src]
        acc[:, rel].index_add_(0, dst, weighted)
    g_ref = acc / (mass.unsqueeze(-1) + 1e-8)
    assert torch.equal(g, g_ref)
    # chunked path: hoist per chunk is also bitwise identical to the
    # unchunked implementation's accumulation? (accumulation order differs,
    # so only allclose) — but chunked vs itself with different sizes must
    # stay internally consistent; the per-chunk hoist must equal the same
    # chunk's old pattern:
    g_c, mass_c = relation_weighted_mean(edge_index, r, features, N, edge_chunk_size=77)
    src, dst = edge_index[0], edge_index[1]
    acc_c = torch.zeros(N, K, D, dtype=features.dtype)
    for start in range(0, E, 77):
        end = min(start + 77, E)
        src_c, dst_c = src[start:end], dst[start:end]
        r_c = r[start:end]
        f_src_c = features[src_c]
        for rel in range(K):
            weighted = r_c[:, rel].unsqueeze(-1) * f_src_c
            acc_c[:, rel].index_add_(0, dst_c, weighted)
    g_ref_c = acc_c / (mass_c.unsqueeze(-1) + 1e-8)
    assert torch.equal(g_c, g_ref_c)


# ---------------------------------------------------------------------------
# 2+3. checkpoint-on == checkpoint-off (bitwise forward + bitwise gradients)
# ---------------------------------------------------------------------------


def test_checkpoint_forward_and_gradients_bitwise() -> None:
    ckpt = FinalModel(_make_cfg(True), _make_info())
    plain = FinalModel(_make_cfg(False), _make_info())
    plain.load_state_dict(ckpt.state_dict())
    x, edge = _make_x(), _make_edge()

    # eval-mode forward
    ckpt.eval()
    plain.eval()
    assert torch.equal(ckpt(x, edge)[0], plain(x, edge)[0])

    # train-mode forward (seeded RNG -> identical dropout masks)
    ckpt.train()
    plain.train()
    torch.manual_seed(3)
    z_c, _, _, aux_c, _ = ckpt(x, edge)
    torch.manual_seed(3)
    z_p, _, _, aux_p, _ = plain(x, edge)
    assert torch.equal(z_c, z_p)
    assert torch.equal(aux_c, aux_p)

    # gradients bitwise equal
    (z_c.square().sum() + aux_c).backward()
    (z_p.square().sum() + aux_p).backward()
    for name, p in plain.named_parameters():
        q = dict(ckpt.named_parameters())[name]
        assert torch.equal(q.grad, p.grad), f"grad differs: {name}"


def test_checkpoint_gradient_symmetry_r_to_factorizer() -> None:
    """Gradients flow through BOTH checkpointed inputs (r and f_cat)."""
    ckpt = FinalModel(_make_cfg(True), _make_info())
    ckpt.train()
    x, edge = _make_x(), _make_edge()
    z, _, _, aux, _ = ckpt(x, edge)
    (z.square().sum() + aux).backward()
    # relation-side (M2) params get gradients through the checkpointed r
    rel_grads = [p.grad for p in ckpt.relation_prototypes.parameters()]
    assert all(g is not None for g in rel_grads)
    assert any(g.norm() > 1e-9 for g in rel_grads)
    assert any(p.grad is not None and p.grad.norm() > 1e-9
               for p in ckpt.edge_token_mlp.parameters())
    # factorizer gets gradients through the checkpointed f_cat
    assert any(p.grad is not None and p.grad.norm() > 1e-9
               for p in ckpt.factorizer.parameters())


# ---------------------------------------------------------------------------
# 4. checkpoint lowers the training peak (GPU-only, synthetic graph)
# ---------------------------------------------------------------------------


def test_checkpoint_reduces_training_peak() -> None:
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no GPU")
    big_n, big_e = 20_000, 200_000
    dev = torch.device("cuda:0")
    ckpt = FinalModel(_make_cfg(True), _make_info()).to(dev)
    plain = FinalModel(_make_cfg(False), _make_info()).to(dev)
    plain.load_state_dict(ckpt.state_dict())
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(big_n, 32, generator=generator).to(dev)
    edge = torch.randint(0, big_n, (2, big_e), generator=generator).to(dev)
    ckpt.train()
    plain.train()

    def step(model: FinalModel) -> float:
        model.zero_grad()
        torch.cuda.reset_peak_memory_stats(dev)
        z, _, _, aux, _ = model(x, edge)
        (z.square().sum() + aux).backward()
        return torch.cuda.max_memory_allocated(dev)

    peak_plain = step(plain)
    del plain
    torch.cuda.empty_cache()
    peak_ckpt = step(ckpt)
    assert peak_ckpt < peak_plain - 0.4e9, (
        f"checkpoint should save >400MB: plain={peak_plain/1e9:.2f}GB "
        f"ckpt={peak_ckpt/1e9:.2f}GB"
    )
