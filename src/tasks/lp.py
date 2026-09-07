from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from src.data import EdgeSplit, MAGData
from src.data.graph_utils import edge_dict_to_index
from src.models import LinkPredictor, build_model
from src.tasks.common import (
    clone_state_dict,
    format_aux_info_stats,
    load_state_dict_cpu,
    resolve_num_neighbors,
    summarize_aux_info_stats,
    update_aux_info_stats,
)
from src.tasks.inference import infer_all_embeddings, resolve_inference_mode
from src.utils.metrics import format_pct
from src.utils.seeds import set_seed
from src.utils.summary import count_parameters, mean_std


def _uses_graph_encoder(cfg) -> bool:
    return str(cfg.model.name).lower() != "mlp"


def _uses_full_graph_training(model, cfg) -> bool:
    if "full_graph_training" in cfg.model:
        return bool(cfg.model.full_graph_training)
    return bool(getattr(model, "requires_full_graph_training", False))


def _edge_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return edge_index[0].long() * int(num_nodes) + edge_index[1].long()


def _all_positive_edge_index(edge_split: EdgeSplit) -> torch.Tensor:
    return torch.cat(
        [
            edge_dict_to_index(edge_split.train),
            edge_dict_to_index(edge_split.valid),
            edge_dict_to_index(edge_split.test),
        ],
        dim=1,
    ).cpu()


def _build_forbidden_edge_keys(edge_split: EdgeSplit, num_nodes: int, undirected: bool) -> torch.Tensor:
    edge_index = _all_positive_edge_index(edge_split)
    if undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return torch.unique(_edge_keys(edge_index, num_nodes).contiguous(), sorted=True)


def _is_forbidden_edge(src: torch.Tensor, dst: torch.Tensor, num_nodes: int, forbidden_keys: torch.Tensor) -> torch.Tensor:
    keys = src.long() * int(num_nodes) + dst.long()
    positions = torch.searchsorted(forbidden_keys, keys)
    in_bounds = positions < forbidden_keys.numel()
    matches = torch.zeros_like(in_bounds, dtype=torch.bool)
    if bool(in_bounds.any()):
        matches[in_bounds] = forbidden_keys[positions[in_bounds]] == keys[in_bounds]
    return matches


def _sample_filtered_negative_targets(
    src: torch.Tensor,
    num_nodes: int,
    num_neg: int,
    forbidden_keys: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    if num_neg <= 0:
        raise ValueError(f"num_neg must be positive, got {num_neg}")

    src = src.cpu().long().contiguous()
    forbidden_keys = forbidden_keys.cpu().long().contiguous()
    negatives = torch.empty((src.numel(), num_neg), dtype=torch.long)
    row_index = torch.arange(src.numel(), dtype=torch.long)

    for neg_col in range(num_neg):
        pending = row_index
        attempts = 0
        while pending.numel() > 0:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError(
                    "Unable to sample filtered train negatives after 1000 attempts; "
                    "the graph may be too dense for the requested num_train_neg."
                )
            pending_src = src[pending]
            candidate = torch.randint(0, num_nodes, (pending.numel(),), generator=generator)
            ok = candidate != pending_src
            ok &= ~_is_forbidden_edge(pending_src, candidate, num_nodes, forbidden_keys)
            if neg_col > 0:
                ok &= ~(negatives[pending, :neg_col] == candidate.view(-1, 1)).any(dim=1)

            accepted = pending[ok]
            if accepted.numel() > 0:
                negatives[accepted, neg_col] = candidate[ok]
            pending = pending[~ok]

    return negatives


def _build_epoch_train_labels(
    train_pos_edge_index: torch.Tensor | EdgeSplit,
    num_nodes: int,
    num_neg: int,
    forbidden_keys: torch.Tensor,
    generator: torch.Generator,
    train_pos_per_epoch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_edge_index = (
        edge_dict_to_index(train_pos_edge_index.train).cpu()
        if isinstance(train_pos_edge_index, EdgeSplit)
        else train_pos_edge_index
    )
    if train_pos_per_epoch is not None and train_pos_per_epoch < pos_edge_index.size(1):
        perm = torch.randperm(pos_edge_index.size(1), generator=generator)[:train_pos_per_epoch]
        pos_edge_index = pos_edge_index[:, perm]
    pos_src = pos_edge_index[0]
    neg_dst = _sample_filtered_negative_targets(pos_src, num_nodes, num_neg, forbidden_keys, generator)
    neg_edge_index = torch.stack(
        [pos_src.repeat_interleave(num_neg), neg_dst.reshape(-1)],
        dim=0,
    )
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1).contiguous()
    edge_label = torch.cat(
        [
            torch.ones(pos_edge_index.size(1), dtype=torch.float32),
            torch.zeros(neg_edge_index.size(1), dtype=torch.float32),
        ],
        dim=0,
    ).contiguous()
    return edge_label_index, edge_label


def _exclude_positive_label_edges_from_message_graph(
    edge_index: torch.Tensor,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    positive_mask = edge_label > 0.5
    if not bool(positive_mask.any()):
        return edge_index.contiguous()

    positive_edges = edge_label_index[:, positive_mask].long()
    forbidden_edges = torch.cat([positive_edges, positive_edges.flip(0)], dim=1)
    forbidden_keys = torch.unique(_edge_keys(forbidden_edges, num_nodes), sorted=True)
    edge_keys = _edge_keys(edge_index, num_nodes)
    positions = torch.searchsorted(forbidden_keys, edge_keys)
    in_bounds = positions < forbidden_keys.numel()
    drop_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
    if bool(in_bounds.any()):
        drop_mask[in_bounds] = forbidden_keys[positions[in_bounds]] == edge_keys[in_bounds]
    return edge_index[:, ~drop_mask].contiguous()


def _build_link_loader(
    cfg,
    pyg_data: Data,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
) -> LinkNeighborLoader:
    return LinkNeighborLoader(
        pyg_data,
        num_neighbors=resolve_num_neighbors(cfg),
        batch_size=int(cfg.task.batch_size),
        shuffle=True,
        subgraph_type=str(cfg.task.get("subgraph_type", "bidirectional")),
        num_workers=int(cfg.task.get("loader_num_workers", 4)),
        prefetch_factor=int(cfg.task.get("loader_prefetch_factor", 2)),
        edge_label_index=edge_label_index,
        edge_label=edge_label,
    )


def _build_edge_loader(cfg, edge_label_index: torch.Tensor, edge_label: torch.Tensor) -> TorchDataLoader:
    return TorchDataLoader(
        TensorDataset(edge_label_index.t().contiguous(), edge_label.contiguous()),
        batch_size=int(cfg.task.batch_size),
        shuffle=True,
    )


def _prepare_eval_embeddings(
    z: torch.Tensor,
    device: torch.device,
    preload: bool,
    logger: logging.Logger,
) -> torch.Tensor:
    z_eval = z.to(device, non_blocking=True) if preload else z
    logger.info("Eval node embeddings: preload=%s | device=%s | shape=%s", preload, z_eval.device, tuple(z_eval.shape))
    return z_eval


@torch.no_grad()
def _evaluate_split(
    z: torch.Tensor,
    predictor: LinkPredictor,
    split: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    predictor.eval()
    src_all = split["source_node"]
    dst_all = split["target_node"]
    neg_all = split["target_node_neg"]
    total = int(src_all.numel())
    mrr_sum = 0.0
    hits1_sum = 0.0
    hits3_sum = 0.0
    hits10_sum = 0.0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        if z.device.type == "cuda":
            src = src_all[start:end].to(z.device, non_blocking=True).long()
            dst = dst_all[start:end].to(z.device, non_blocking=True).long()
            neg = neg_all[start:end].to(z.device, non_blocking=True).long()
            pos_src = z[src]
            pos_dst = z[dst]
            src_neg = src.view(-1, 1).expand_as(neg).reshape(-1)
            neg_src = z[src_neg]
            neg_dst = z[neg.reshape(-1)]
        else:
            src = src_all[start:end].cpu().long()
            dst = dst_all[start:end].cpu().long()
            neg = neg_all[start:end].cpu().long()
            pos_src = z[src].to(device, non_blocking=True)
            pos_dst = z[dst].to(device, non_blocking=True)
            src_neg = src.view(-1, 1).expand_as(neg).reshape(-1)
            neg_src = z[src_neg].to(device, non_blocking=True)
            neg_dst = z[neg.reshape(-1)].to(device, non_blocking=True)

        pos_score = predictor.score_pairs(pos_src, pos_dst)
        neg_score = predictor.score_pairs(neg_src, neg_dst).view(neg.size(0), neg.size(1))
        # RPTA/OpenMAG pessimistic ranking: ties rank BEHIND the positive.
        ranks = 1.0 + (neg_score >= pos_score.view(-1, 1)).sum(dim=1).float()

        mrr_sum += float((1.0 / ranks).sum().item())
        hits1_sum += float((ranks <= 1).float().sum().item())
        hits3_sum += float((ranks <= 3).float().sum().item())
        hits10_sum += float((ranks <= 10).float().sum().item())

    denom = max(total, 1)
    return {
        "mrr": mrr_sum / denom,
        "hits@1": hits1_sum / denom,
        "hits@3": hits3_sum / denom,
        "hits@10": hits10_sum / denom,
    }


def _run_single_lp(cfg, data: MAGData, device: torch.device, logger: logging.Logger, run_id: int) -> dict[str, float]:
    seed = int(cfg.seed) + run_id
    set_seed(seed)
    inference_mode = resolve_inference_mode(cfg)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info).to(device)
    predictor = LinkPredictor(
        in_dim=model.out_dim,
        hidden_dim=int(cfg.task.decoder.hidden_dim),
        num_layers=int(cfg.task.decoder.num_layers),
        dropout=float(cfg.task.decoder.dropout),
    ).to(device)
    # RPTA-style shared embedding dimension: project every encoder output to a
    # common dim so the predictor sees identical capacity across models.
    proj_dim = int(cfg.task.decoder.get("proj_dim", 0) or 0)
    projection = nn.Linear(model.out_dim, proj_dim).to(device) if proj_dim > 0 else None
    if projection is not None:
        predictor = LinkPredictor(
            in_dim=proj_dim,
            hidden_dim=int(cfg.task.decoder.hidden_dim),
            num_layers=int(cfg.task.decoder.num_layers),
            dropout=float(cfg.task.decoder.dropout),
        ).to(device)
    optimizer_name = str(cfg.task.get("optimizer", "adam")).strip().lower()
    lr = float(cfg.model.get("lr", cfg.task.lr))
    weight_decay = float(cfg.model.get("weight_decay", cfg.task.weight_decay))
    optimizer_params = list(model.parameters()) + list(predictor.parameters())
    if projection is not None:
        optimizer_params += list(projection.parameters())
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(optimizer_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(optimizer_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"task.optimizer must be adam|adamw, got {optimizer_name!r}")
    criterion = torch.nn.BCEWithLogitsLoss()
    uses_graph = _uses_graph_encoder(cfg)
    full_graph_training = uses_graph and _uses_full_graph_training(model, cfg)
    if full_graph_training:
        loader_name = "FullGraphEdgeLabelDataLoader"
    else:
        loader_name = "LinkNeighborLoader" if uses_graph else "EdgeLabelDataLoader"
    x_all = data.x.to(device) if (not uses_graph or full_graph_training) else None
    edge_index_all = data.edge_index.to(device) if full_graph_training else None
    undirected_filter = bool(data.edge_split.metadata.get("undirected", cfg.dataset.get("make_undirected", True)))
    forbidden_keys = _build_forbidden_edge_keys(data.edge_split, data.num_nodes, undirected=undirected_filter)
    neg_generator = torch.Generator().manual_seed(seed)

    logger.info(
        "[Run %d/%d] seed=%d | model+decoder params=%d",
        run_id + 1,
        int(cfg.num_runs),
        seed,
        count_parameters(model) + count_parameters(predictor),
    )
    train_pos_per_epoch = cfg.task.get("train_pos_per_epoch")
    if train_pos_per_epoch is not None:
        train_pos_per_epoch = int(train_pos_per_epoch)
    logger.info("Loader: %s", loader_name)
    if full_graph_training:
        logger.info("Train full graph: enabled for global pseudo-node pathways")
        logger.info("Train edge label batches still exclude current positive labels from the full message graph")
    elif uses_graph:
        logger.info("Train neighbor sampling: %s", resolve_num_neighbors(cfg))
    logger.info("Inference mode: %s", inference_mode)
    logger.info("Train negative sampling: global filtered | num_neg=%d", int(cfg.task.num_train_neg))
    if train_pos_per_epoch is not None:
        logger.info("Train pos per epoch: %d", train_pos_per_epoch)
    logger.info("Training...")

    # v7-style data-pipeline optimization (RPTA LP_SPEED_OPTIMIZATION.md):
    # build the feature/edge Data object ONCE and reuse it across epochs —
    # only the per-epoch edge labels change.
    pyg_data = Data(x=data.x, edge_index=data.edge_index)

    best_val = -1.0
    best_test: dict[str, float] = {}
    best_model_state = None
    best_predictor_state = None
    best_proj_state = None
    # patience null = run the full epoch budget (RPTA-style), still checkpoint
    # the best val MRR along the way.
    patience_total = cfg.task.get("patience")
    patience_left = int(patience_total) if patience_total is not None else None
    early_stop_min_epoch = int(cfg.task.get("early_stop_min_epoch", 1))
    grad_clip = cfg.task.get("grad_clip")
    max_train_batches = cfg.task.get("max_train_batches")
    inference_batch_size = int(cfg.task.inference_batch_size)
    eval_preload_node_emb = bool(cfg.task.get("eval_preload_node_emb", False))
    logger.info("LP eval preload node embeddings: %s", eval_preload_node_emb)
    train_pos_edge_index_all = edge_dict_to_index(data.edge_split.train).cpu()

    for epoch in range(1, int(cfg.task.epochs) + 1):
        model.train()
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        predictor.train()
        total_loss = 0.0
        total_examples = 0
        aux_sums: dict[str, float] = {}
        aux_counts: dict[str, float] = {}
        edge_label_index, edge_label = _build_epoch_train_labels(
            train_pos_edge_index_all,
            data.num_nodes,
            int(cfg.task.num_train_neg),
            forbidden_keys,
            neg_generator,
            train_pos_per_epoch=train_pos_per_epoch,
        )
        if full_graph_training:
            loader = _build_edge_loader(cfg, edge_label_index, edge_label)
        elif uses_graph:
            loader = _build_link_loader(cfg, pyg_data, edge_label_index, edge_label)
        else:
            loader = _build_edge_loader(cfg, edge_label_index, edge_label)

        for step, batch in enumerate(loader):
            if max_train_batches is not None and step >= int(max_train_batches):
                break
            optimizer.zero_grad(set_to_none=True)
            if full_graph_training:
                edges, labels = batch
                edge_label_index_batch = edges.t().contiguous().to(device)
                labels = labels.to(device)
                message_edge_index = _exclude_positive_label_edges_from_message_graph(
                    edge_index_all,
                    edge_label_index_batch,
                    labels,
                    num_nodes=int(data.num_nodes),
                )
                z, _, _, aux_loss, aux_info = model(x_all, message_edge_index)
                if projection is not None:
                    z = projection(z)
                src, dst = edge_label_index_batch
                logits = predictor.score_pairs(z[src], z[dst])
            elif uses_graph:
                batch = batch.to(device)
                if hasattr(model, "_batch_n_id"):
                    model._batch_n_id = batch.n_id
                message_edge_index = _exclude_positive_label_edges_from_message_graph(
                    batch.edge_index,
                    batch.edge_label_index,
                    batch.edge_label,
                    num_nodes=int(batch.x.size(0)),
                )
                z, _, _, aux_loss, aux_info = model(batch.x, message_edge_index)
                if projection is not None:
                    z = projection(z)
                src, dst = batch.edge_label_index
                logits = predictor.score_pairs(z[src], z[dst])
                labels = batch.edge_label.float()
            else:
                edges, labels = batch
                edges = edges.to(device)
                labels = labels.to(device)
                z_all, _, _, aux_loss, aux_info = model(x_all[edges.reshape(-1)], None)
                if projection is not None:
                    z_all = projection(z_all)
                z_pairs = z_all.view(edges.size(0), 2, -1)
                logits = predictor.score_pairs(z_pairs[:, 0], z_pairs[:, 1])
            loss = criterion(logits, labels) + float(cfg.task.loss.aux_weight) * aux_loss
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=float(grad_clip))
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_examples += int(labels.numel())
            update_aux_info_stats(aux_sums, aux_counts, aux_info, weight=float(labels.numel()))
            if hasattr(model, "_batch_n_id"):
                model._batch_n_id = None

        train_loss = total_loss / max(total_examples, 1)
        aux_stats = summarize_aux_info_stats(aux_sums, aux_counts)
        if epoch % int(cfg.task.eval_every) != 0:
            logger.info("Epoch %05d | Train Loss %.4f", epoch, train_loss)
            if aux_stats:
                logger.info("Aux %s", format_aux_info_stats(aux_stats))
            continue

        z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
        if projection is not None:
            with torch.no_grad():
                z = projection(z.to(device)).detach().cpu()
        z_eval = _prepare_eval_embeddings(z, device, eval_preload_node_emb, logger)
        val_metrics = _evaluate_split(z_eval, predictor, data.edge_split.valid, device, int(cfg.task.eval_edge_batch_size))
        logger.info("Epoch %05d | Train Loss %.4f", epoch, train_loss)
        if aux_stats:
            logger.info("Aux %s", format_aux_info_stats(aux_stats))
        logger.info(
            "Val MRR %.2f | Val H@1 %.2f | Val H@3 %.2f | Val H@10 %.2f",
            format_pct(val_metrics["mrr"]),
            format_pct(val_metrics["hits@1"]),
            format_pct(val_metrics["hits@3"]),
            format_pct(val_metrics["hits@10"]),
        )

        stop_early = False
        if val_metrics["mrr"] > best_val:
            best_val = val_metrics["mrr"]
            best_test = {
                "val_mrr": val_metrics["mrr"],
            }
            best_model_state = clone_state_dict(model)
            best_predictor_state = clone_state_dict(predictor)
            best_proj_state = clone_state_dict(projection) if projection is not None else None
            patience_left = int(patience_total) if patience_total is not None else None
        elif patience_left is not None and epoch >= early_stop_min_epoch:
            patience_left -= 1
            patience_used = int(patience_total) - patience_left
            logger.info("Patience %d/%d | Best Val MRR %.2f", patience_used, int(patience_total), format_pct(best_val))
            if patience_left <= 0:
                logger.info("Early stopping at epoch %03d", epoch)
                stop_early = True

        del z_eval
        del z
        torch.cuda.empty_cache()
        if stop_early:
            break

    if best_model_state is not None and best_predictor_state is not None:
        load_state_dict_cpu(model, best_model_state)
        load_state_dict_cpu(predictor, best_predictor_state)
        if best_proj_state is not None and projection is not None:
            load_state_dict_cpu(projection, best_proj_state)
        z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
        if projection is not None:
            with torch.no_grad():
                z = projection(z.to(device)).detach().cpu()
        z_eval = _prepare_eval_embeddings(z, device, eval_preload_node_emb, logger)
        test_metrics = _evaluate_split(
            z_eval,
            predictor,
            data.edge_split.test,
            device,
            int(cfg.task.eval_edge_batch_size),
        )
        best_test["test_mrr"] = test_metrics["mrr"]
        best_test["test_hits@1"] = test_metrics["hits@1"]
        best_test["test_hits@3"] = test_metrics["hits@3"]
        best_test["test_hits@10"] = test_metrics["hits@10"]
        del z_eval
        del z

    if not best_test:
        z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
        if projection is not None:
            with torch.no_grad():
                z = projection(z.to(device)).detach().cpu()
        z_eval = _prepare_eval_embeddings(z, device, eval_preload_node_emb, logger)
        val_metrics = _evaluate_split(z_eval, predictor, data.edge_split.valid, device, int(cfg.task.eval_edge_batch_size))
        test_metrics = _evaluate_split(z_eval, predictor, data.edge_split.test, device, int(cfg.task.eval_edge_batch_size))
        best_test = {
            "val_mrr": val_metrics["mrr"],
            "test_mrr": test_metrics["mrr"],
            "test_hits@1": test_metrics["hits@1"],
            "test_hits@3": test_metrics["hits@3"],
            "test_hits@10": test_metrics["hits@10"],
        }
        del z_eval
        del z

    if hasattr(model, "_batch_n_id"):
        model._batch_n_id = None

    save_ckpt_path = cfg.task.get("save_ckpt_path")
    if save_ckpt_path:
        path = Path(str(save_ckpt_path))
        if int(cfg.num_runs) > 1:
            path = path.with_name(f"{path.stem}_run{run_id + 1}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "task": "lp",
                "seed": seed,
                "model_state": clone_state_dict(model),
                "head_state": clone_state_dict(predictor),
                "proj_state": clone_state_dict(projection) if projection is not None else None,
                "data_info": data_info,
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)

    logger.info(
        "[Run %d] Best Val MRR %.2f",
        run_id + 1,
        format_pct(best_test["val_mrr"]),
    )
    logger.info(
        "[Run %d] Final Test MRR %.2f | H@1 %.2f | H@3 %.2f | H@10 %.2f",
        run_id + 1,
        format_pct(best_test["test_mrr"]),
        format_pct(best_test["test_hits@1"]),
        format_pct(best_test["test_hits@3"]),
        format_pct(best_test["test_hits@10"]),
    )
    return best_test


def run_lp(cfg, data: MAGData, device: torch.device, logger: logging.Logger) -> dict[str, tuple[float, float]]:
    if data.edge_split is None:
        raise ValueError("LP data must contain edge_split")
    run_results = [_run_single_lp(cfg, data, device, logger, run_id) for run_id in range(int(cfg.num_runs))]
    output: dict[str, tuple[float, float]] = {}
    logger.info("============================================================")
    logger.info("Final Results over %d runs", int(cfg.num_runs))
    logger.info("============================================================")
    display_names = {
        "val_mrr": "Highest Valid MRR",
        "test_mrr": "Test MRR",
        "test_hits@1": "Test Hits@1",
        "test_hits@3": "Test Hits@3",
        "test_hits@10": "Test Hits@10",
    }
    for key in ["val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10"]:
        values = [item[key] for item in run_results]
        mean, std = mean_std(values)
        output[key] = (mean, std)
        logger.info("%s: %.2f ± %.2f", display_names[key], format_pct(mean), format_pct(std))
    logger.info("============================================================")
    return output
