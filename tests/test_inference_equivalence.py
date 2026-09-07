from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.data import MAGData
from src.models import dip, gcn, mlp, mmgcn, sage
from src.tasks.inference import infer_all_embeddings, resolve_inference_mode


class _CfgNode(dict):
    def __getattr__(self, name: str):
        return self[name]


def _cfg() -> _CfgNode:
    return _CfgNode(
        model=_CfgNode(
            hidden_dim=5,
            num_layers=2,
            dropout=0.0,
            activation="relu",
            norm="batchnorm",
            aggr="mean",
        )
    )


def _small_graph() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5, 1.6],
            [1.7, 1.8, 1.9, 2.0],
            [2.1, 2.2, 2.3, 2.4],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0],
        ],
        dtype=torch.long,
    )
    return x, edge_index


def _build_model(model_cls):
    torch.manual_seed(123)
    model = model_cls(_cfg(), {"input_dim": 4})
    model.eval()
    return model


def _compare_full_and_inference(model, x: torch.Tensor, edge_index: torch.Tensor | None):
    graph_edge_index = edge_index if edge_index is not None else torch.empty((2, 0), dtype=torch.long)
    data = MAGData(
        name="small",
        source="test",
        task="test",
        x=x,
        edge_index=graph_edge_index,
        num_nodes=int(x.size(0)),
    )
    with torch.no_grad():
        model.eval()
        full = infer_all_embeddings(
            model,
            data,
            device=torch.device("cpu"),
            uses_graph=edge_index is not None,
            batch_size=2,
            inference_mode="full",
        )
        model.eval()
        inferred = infer_all_embeddings(
            model,
            data,
            device=torch.device("cpu"),
            uses_graph=edge_index is not None,
            batch_size=2,
            inference_mode="layerwise",
        )
    return full, inferred, float((full - inferred).abs().max().item())


def test_mlp_inference_matches_full_batch_forward() -> None:
    x, _ = _small_graph()
    model = _build_model(mlp.Model)

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, None)

    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4


def test_sage_layerwise_inference_matches_full_batch_forward() -> None:
    x, edge_index = _small_graph()
    model = _build_model(sage.Model)

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, edge_index)

    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4


def test_gcn_layerwise_inference_matches_full_batch_forward() -> None:
    x, edge_index = _small_graph()
    model = _build_model(gcn.Model)

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, edge_index)

    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4


def test_mmgcn_layerwise_inference_matches_full_batch_forward() -> None:
    _, edge_index = _small_graph()
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    torch.manual_seed(123)
    model = mmgcn.Model(_cfg(), {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6})
    model.eval()

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, edge_index)

    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4


def test_dip_layerwise_inference_matches_full_batch_forward() -> None:
    _, edge_index = _small_graph()
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    cfg = _cfg()
    cfg.model["d_model"] = 4
    cfg.model["q_dim"] = 4
    cfg.model["n_q"] = 2
    cfg.model["mp_hops"] = 2
    cfg.model["n_pnode_v"] = 3
    cfg.model["n_pnode_t"] = 2
    cfg.model["dropout"] = 0.0
    cfg.model["norm"] = True
    cfg.model["fusion_type"] = "path_integral"
    cfg.model["embedding_dim"] = None

    torch.manual_seed(123)
    model = dip.Model(cfg, {"input_dim": 10, "text_dim": 4, "visual_dim": 6})
    model.eval()

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, edge_index)

    assert full.shape == (6, 8)
    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4


def test_mmgcn_uses_global_batch_node_ids_for_id_embeddings() -> None:
    torch.manual_seed(123)
    model = mmgcn.Model(_cfg(), {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6})
    n_id = torch.tensor([3, 7, 1, 4], dtype=torch.long)

    assert model._batch_n_id is None
    model._batch_n_id = n_id

    assert torch.equal(model._get_id_emb(n_id.numel()), model.id_embedding[n_id])


def test_mmgcn_id_embedding_is_not_registered_parameter() -> None:
    torch.manual_seed(123)
    model = mmgcn.Model(_cfg(), {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6})

    assert isinstance(model.id_embedding, torch.Tensor)
    assert not isinstance(model.id_embedding, nn.Parameter)
    assert model.id_embedding.requires_grad
    assert "id_embedding" not in dict(model.named_parameters())


def test_mmgcn_honors_dropout_config() -> None:
    cfg = _cfg()
    cfg.model["dropout"] = 0.37

    model = mmgcn.Model(cfg, {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6})

    assert model.v_branch.dropout.p == 0.37
    assert model.t_branch.dropout.p == 0.37


def test_mmgcn_honors_norm_config() -> None:
    cfg = _cfg()
    cfg.model["norm"] = "batchnorm"
    model = mmgcn.Model(cfg, {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6})

    assert isinstance(model.v_branch.norms[0], nn.BatchNorm1d)
    assert isinstance(model.t_branch.norms[0], nn.BatchNorm1d)

    cfg.model["norm"] = "none"
    model = mmgcn.Model(cfg, {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6})

    assert isinstance(model.v_branch.norms[0], nn.Identity)
    assert isinstance(model.t_branch.norms[0], nn.Identity)


def test_inference_mode_validation_accepts_supported_modes() -> None:
    cfg = _CfgNode(task=_CfgNode(inference_mode="full"))
    assert resolve_inference_mode(cfg) == "full"

    cfg.task["inference_mode"] = "layerwise"
    assert resolve_inference_mode(cfg) == "layerwise"


def test_inference_mode_validation_rejects_invalid_mode() -> None:
    cfg = _CfgNode(task=_CfgNode(inference_mode="sampled"))

    with pytest.raises(ValueError, match="task.inference_mode"):
        resolve_inference_mode(cfg)
