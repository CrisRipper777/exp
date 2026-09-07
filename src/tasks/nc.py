from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score

from src.data import MAGData
from src.models import build_model
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


def _resolve_training_mode(cfg, model) -> str:
    """RPTA-style protocol: full-graph training by default (one forward on the
    complete transductive graph per epoch, CE on train nodes only). ``sampled``
    is an opt-out fallback for graphs too large for a full-graph forward."""
    if getattr(model, "requires_full_graph_training", False):
        return "full_graph"
    mode = str(cfg.task.get("training_mode", "full_graph")).strip().lower()
    if mode not in ("full_graph", "sampled"):
        raise ValueError(f"task.training_mode must be full_graph|sampled, got {mode!r}")
    return mode


def _build_optimizer(parameters, cfg) -> torch.optim.Optimizer:
    name = str(cfg.task.get("optimizer", "adamw")).strip().lower()
    # Official per-model presets may override the task-level lr / weight_decay.
    lr = float(cfg.model.get("lr", cfg.task.lr))
    weight_decay = float(cfg.model.get("weight_decay", cfg.task.weight_decay))
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"task.optimizer must be adamw|adam, got {name!r}")


def _scheduler_step(cfg, optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int) -> None:
    """R1.5 LR schedule (plan §R15-4): task.scheduler=warmup_cosine enables
    10-epoch linear warmup + cosine decay to scheduler_min_lr. Default null
    keeps the constant-LR benchmark behavior unchanged."""
    name = str(cfg.task.get("scheduler", "null")).strip().lower()
    # hydra may render the yaml null as the string "none".
    if name in ("null", "", "none"):
        return
    if name != "warmup_cosine":
        raise ValueError(f"task.scheduler must be null|warmup_cosine, got {name!r}")
    warmup = max(int(cfg.task.get("scheduler_warmup_epochs", 10)), 1)
    base_lr = float(cfg.model.get("lr", cfg.task.lr))
    final_lr = float(cfg.task.get("scheduler_min_lr", 1e-5))
    if epoch <= warmup:
        lr = base_lr * epoch / warmup
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        lr = final_lr + 0.5 * (base_lr - final_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


@torch.no_grad()
def _evaluate_split(
    classifier,
    z: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    classifier.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch_idx in TorchDataLoader(idx.cpu(), batch_size=batch_size, shuffle=False):
        logits = classifier(z[batch_idx].to(device))
        preds.append(logits.argmax(dim=-1).cpu())
        targets.append(labels[batch_idx].cpu())

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return {
        "acc": float((pred == target).float().mean().item()),
        "macro_f1": float(f1_score(target.numpy(), pred.numpy(), average="macro", zero_division=0)),
    }


def _run_single_nc(cfg, data: MAGData, device: torch.device, logger: logging.Logger, run_id: int) -> dict[str, float]:
    seed = int(cfg.seed) + run_id
    set_seed(seed)
    inference_mode = resolve_inference_mode(cfg)

    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
        # Additive keys for label-aware models (RPTA migration, 2026-09-03):
        # existing models ignore them; the training protocol is unchanged.
        "y": data.y,
        "train_idx": data.train_idx,
    }
    model = build_model(cfg, data_info).to(device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    optimizer = _build_optimizer(list(model.parameters()) + list(classifier.parameters()), cfg)
    criterion = nn.CrossEntropyLoss()
    uses_graph = _uses_graph_encoder(cfg)
    training_mode = _resolve_training_mode(cfg, model)

    num_neighbors = None
    loader = None
    if training_mode == "full_graph":
        loader_name = "FullGraph"
    elif uses_graph:
        pyg_data = Data(x=data.x, edge_index=data.edge_index, y=data.y)
        num_neighbors = resolve_num_neighbors(cfg)
        loader = NeighborLoader(
            pyg_data,
            input_nodes=data.train_idx,
            num_neighbors=num_neighbors,
            batch_size=int(cfg.task.batch_size),
            shuffle=True,
        )
        loader_name = "NeighborLoader"
    else:
        loader = TorchDataLoader(
            data.train_idx,
            batch_size=int(cfg.task.batch_size),
            shuffle=True,
        )
        loader_name = "NodeDataLoader"

    x_all = data.x.to(device) if (training_mode == "full_graph" or not uses_graph) else None
    y_all = data.y.to(device) if (training_mode == "full_graph" or not uses_graph) else None
    edge_index_all = data.edge_index.to(device) if (uses_graph and training_mode == "full_graph") else None
    train_idx_all = data.train_idx.to(device) if training_mode == "full_graph" else None

    grad_clip = cfg.task.get("grad_clip")
    early_stop_min_epoch = int(cfg.task.get("early_stop_min_epoch", 1))
    aux_weight = float(cfg.task.loss.aux_weight)
    # R1.5 val-only screen (plan §4): evaluate_test=false skips ALL test
    # access/metrics (backward-compatible, default true keeps the formal
    # benchmark behavior unchanged).
    evaluate_test = bool(cfg.task.get("evaluate_test", True))
    # R1.5 per-epoch history (plan §9): written when task.history_path is
    # set. Records train CE/aux split, train acc (full-graph mode), val
    # metrics, lr, patience and the P0 aux components.
    history_path = cfg.task.get("history_path")
    history_file = None
    history_writer = None
    scope_history_keys: list[str] = []
    if hasattr(model, "num_blocks"):
        for layer_idx in range(1, int(model.num_blocks) + 1):
            for factor in ("c", "pt", "pv"):
                for metric in (
                    "osc_ratio", "orb_ratio", "graph_update_ratio",
                    "beta_mean", "beta_std", "anchor_drift",
                    "local_message_ratio", "global_local_ratio", "global_scale",
                    "slot_entropy", "effective_active_slots", "active_slots",
                    "slot_usage_0", "slot_usage_1", "slot_usage_2", "slot_usage_3",
                    "cross_context_ratio", "cross_scale", "cross_update_ratio",
                ):
                    scope_history_keys.append(f"scope_l{layer_idx}_{factor}_{metric}")
    if history_path:
        import csv

        history_file = Path(str(history_path))
        if int(cfg.num_runs) > 1:
            history_file = history_file.with_name(f"{history_file.stem}_run{run_id + 1}{history_file.suffix}")
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history_file = history_file.open("w", newline="", encoding="utf-8")
        history_writer = csv.writer(history_file)
        history_writer.writerow([
            "epoch", "train_total_loss", "train_ce_loss", "train_aux_loss",
            "train_acc", "val_acc", "val_macro_f1", "lr", "patience_left",
            "common_loss", "orth_loss", "recon_loss", "common_sim",
            "private_sim", "c_norm", "pt_norm", "pv_norm",
            "cp_overlap_t", "cp_overlap_v",
            *scope_history_keys,
        ])

    logger.info(
        "[Run %d/%d] seed=%d | model+head params=%d",
        run_id + 1,
        int(cfg.num_runs),
        seed,
        count_parameters(model) + count_parameters(classifier),
    )
    logger.info("Loader: %s | optimizer: %s", loader_name, str(cfg.task.get("optimizer", "adamw")))
    if training_mode == "sampled" and uses_graph:
        logger.info("Train neighbor sampling: %s", num_neighbors)
    logger.info("Inference mode: %s", inference_mode)
    logger.info("Training...")

    best_val = -1.0
    best_test: dict[str, float] = {}
    best_model_state = None
    best_head_state = None
    patience_total = int(cfg.task.patience)
    patience_left = patience_total
    max_train_batches = cfg.task.get("max_train_batches")
    inference_batch_size = int(cfg.task.inference_batch_size)

    for epoch in range(1, int(cfg.task.epochs) + 1):
        model.train()
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        classifier.train()
        total_loss = 0.0
        total_ce = 0.0
        total_aux = 0.0
        total_examples = 0
        train_correct = 0
        aux_sums: dict[str, float] = {}
        aux_counts: dict[str, float] = {}
        if training_mode == "full_graph":
            optimizer.zero_grad(set_to_none=True)
            z, _, _, aux_loss, aux_info = model(x_all, edge_index_all)
            labels = y_all[train_idx_all]
            logits = classifier(z[train_idx_all])
            ce_loss = criterion(logits, labels)
            loss = ce_loss + aux_weight * aux_loss
            if history_writer is not None:
                # pre-step train accuracy from the same forward the CE uses
                train_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(classifier.parameters()), max_norm=float(grad_clip)
                )
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_ce += float(ce_loss.item()) * int(labels.numel())
            total_aux += float(aux_loss.item()) * int(labels.numel())
            total_examples += int(labels.numel())
            update_aux_info_stats(aux_sums, aux_counts, aux_info, weight=float(labels.numel()))
            del z
        else:
            for step, batch in enumerate(loader):
                if max_train_batches is not None and step >= int(max_train_batches):
                    break
                optimizer.zero_grad(set_to_none=True)
                if uses_graph:
                    batch = batch.to(device)
                    if hasattr(model, "_batch_n_id"):
                        model._batch_n_id = batch.n_id
                    z, _, _, aux_loss, aux_info = model(batch.x, batch.edge_index)
                    logits = classifier(z[: batch.batch_size])
                    labels = batch.y[: batch.batch_size]
                else:
                    batch_idx = batch.to(device)
                    x_batch = x_all[batch_idx]
                    labels = y_all[batch_idx]
                    z, _, _, aux_loss, aux_info = model(x_batch, None)
                    logits = classifier(z)
                loss = criterion(logits, labels) + aux_weight * aux_loss
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(classifier.parameters()), max_norm=float(grad_clip)
                    )
                optimizer.step()
                total_loss += float(loss.item()) * int(labels.numel())
                total_examples += int(labels.numel())
                update_aux_info_stats(aux_sums, aux_counts, aux_info, weight=float(labels.numel()))
                if hasattr(model, "_batch_n_id"):
                    model._batch_n_id = None

        train_loss = total_loss / max(total_examples, 1)
        aux_stats = summarize_aux_info_stats(aux_sums, aux_counts)
        _scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        if epoch % int(cfg.task.eval_every) != 0:
            logger.info("Epoch %05d | Train Loss %.4f", epoch, train_loss)
            if aux_stats:
                logger.info("Aux %s", format_aux_info_stats(aux_stats))
            continue

        if inference_mode == "full" and training_mode == "full_graph":
            # The training forward already used the full graph: reuse the
            # hosted tensors instead of re-hosting a copy for eval.
            if hasattr(model, "_batch_n_id"):
                model._batch_n_id = None
            model.eval()
            classifier.eval()
            with torch.no_grad():
                z, _, _, _, _ = model(x_all, edge_index_all)
            z = z.detach().cpu()
        else:
            z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
        val_metrics = _evaluate_split(classifier, z, data.y, data.val_idx, device, inference_batch_size)
        logger.info("Epoch %05d | Train Loss %.4f", epoch, train_loss)
        if aux_stats:
            logger.info("Aux %s", format_aux_info_stats(aux_stats))
        logger.info(
            "Val Acc %.2f | Val F1 %.2f",
            format_pct(val_metrics["acc"]),
            format_pct(val_metrics["macro_f1"]),
        )

        if history_writer is not None:
            train_acc = train_correct / max(total_examples, 1)
            aux_row = {
                "common_loss": aux_stats.get("p0_common_loss"),
                "orth_loss": aux_stats.get("p0_orth_loss"),
                "recon_loss": aux_stats.get("p0_recon_loss"),
                "common_sim": aux_stats.get("p0_common_sim"),
                "private_sim": aux_stats.get("p0_private_sim"),
                "c_norm": aux_stats.get("p0_c_norm"),
                "pt_norm": aux_stats.get("p0_pt_norm"),
                "pv_norm": aux_stats.get("p0_pv_norm"),
                "cp_overlap_t": aux_stats.get("p0_cp_overlap_t"),
                "cp_overlap_v": aux_stats.get("p0_cp_overlap_v"),
            }

            def _f(v):
                return f"{v:.6f}" if v is not None else ""

            history_writer.writerow([
                epoch,
                _f(total_loss / max(total_examples, 1)),
                _f(total_ce / max(total_examples, 1)),
                _f(total_aux / max(total_examples, 1)),
                _f(train_acc),
                _f(val_metrics["acc"]),
                _f(val_metrics["macro_f1"]),
                _f(optimizer.param_groups[0]["lr"]),
                patience_left,
                *[_f(aux_row[k]) for k in (
                    "common_loss", "orth_loss", "recon_loss", "common_sim",
                    "private_sim", "c_norm", "pt_norm", "pv_norm",
                    "cp_overlap_t", "cp_overlap_v",
                )],
                *[_f(aux_stats.get(k)) for k in scope_history_keys],
            ])
            history_file.flush()

        stop_early = False
        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            best_test = {
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            best_model_state = clone_state_dict(model)
            best_head_state = clone_state_dict(classifier)
            patience_left = patience_total
        elif epoch >= early_stop_min_epoch:
            patience_left -= 1
            patience_used = patience_total - patience_left
            logger.info("Patience %d/%d | Best Val Acc %.2f", patience_used, patience_total, format_pct(best_val))
            if patience_left <= 0:
                logger.info("Early stopping at epoch %03d", epoch)
                stop_early = True
        del z
        torch.cuda.empty_cache()
        if stop_early:
            break

    if best_model_state is not None and best_head_state is not None:
        load_state_dict_cpu(model, best_model_state)
        load_state_dict_cpu(classifier, best_head_state)
        if evaluate_test:
            z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
            test_metrics = _evaluate_split(classifier, z, data.y, data.test_idx, device, inference_batch_size)
            best_test["test_acc"] = test_metrics["acc"]
            best_test["test_macro_f1"] = test_metrics["macro_f1"]
            del z

    if not best_test:
        z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
        val_metrics = _evaluate_split(classifier, z, data.y, data.val_idx, device, inference_batch_size)
        best_test = {
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        if evaluate_test:
            test_metrics = _evaluate_split(classifier, z, data.y, data.test_idx, device, inference_batch_size)
            best_test["test_acc"] = test_metrics["acc"]
            best_test["test_macro_f1"] = test_metrics["macro_f1"]
        del z

    if hasattr(model, "_batch_n_id"):
        model._batch_n_id = None

    if history_file is not None:
        history_file.close()

    save_ckpt_path = cfg.task.get("save_ckpt_path")
    if save_ckpt_path:
        path = Path(str(save_ckpt_path))
        if int(cfg.num_runs) > 1:
            path = path.with_name(f"{path.stem}_run{run_id + 1}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "task": "nc",
                "seed": seed,
                "model_state": clone_state_dict(model),
                "head_state": clone_state_dict(classifier),
                "data_info": data_info,
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)

    logger.info(
        "[Run %d] Best Val Acc %.2f",
        run_id + 1,
        format_pct(best_test["val_acc"]),
    )
    if evaluate_test:
        logger.info(
            "[Run %d] Final Test Acc %.2f | Final Test F1 %.2f",
            run_id + 1,
            format_pct(best_test["test_acc"]),
            format_pct(best_test["test_macro_f1"]),
        )
    return best_test


def run_nc(cfg, data: MAGData, device: torch.device, logger: logging.Logger) -> dict[str, tuple[float, float]]:
    if data.y is None or data.train_idx is None or data.val_idx is None:
        raise ValueError("NC data must contain y/train_idx/val_idx")
    if bool(cfg.task.get("evaluate_test", True)) and data.test_idx is None:
        raise ValueError("NC data must contain test_idx when task.evaluate_test=true")

    run_results = [_run_single_nc(cfg, data, device, logger, run_id) for run_id in range(int(cfg.num_runs))]
    val_acc = [item["val_acc"] for item in run_results]
    val_mean, val_std = mean_std(val_acc)
    logger.info("============================================================")
    logger.info("Final Results over %d runs", int(cfg.num_runs))
    logger.info("============================================================")
    logger.info("Highest Valid Acc: %.2f ± %.2f", format_pct(val_mean), format_pct(val_std))
    evaluate_test = bool(cfg.task.get("evaluate_test", True))
    if evaluate_test:
        test_acc = [item["test_acc"] for item in run_results]
        test_f1 = [item["test_macro_f1"] for item in run_results]
        acc_mean, acc_std = mean_std(test_acc)
        f1_mean, f1_std = mean_std(test_f1)
        logger.info("Test Acc: %.2f ± %.2f", format_pct(acc_mean), format_pct(acc_std))
        logger.info("Test Macro-F1: %.2f ± %.2f", format_pct(f1_mean), format_pct(f1_std))
        logger.info("============================================================")
        return {
            "val_acc": (val_mean, val_std),
            "test_acc": (acc_mean, acc_std),
            "test_macro_f1": (f1_mean, f1_std),
        }
    logger.info("(evaluate_test=false: no test access / metrics recorded)")
    logger.info("============================================================")
    return {"val_acc": (val_mean, val_std)}
