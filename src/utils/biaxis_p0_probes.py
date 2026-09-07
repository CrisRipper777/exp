"""P0-D factor-wise propagation utility probes (supervised).

NC probe: for each factor in {C, Pt, Pv}, train an identical Linear
classifier on the LOCAL representation and on the fixed-GCN GRAPH
representation (LayerNorm(F + A_norm F)), then compare per-node
cross-entropy deltas. delta = CE_local - CE_graph; delta > 0 means the
graph propagation helps that node for that factor.

Discipline (plan §11 / §26):
  - probe trains ONLY on train nodes; validation drives early stopping;
  - test evaluation happens only when include_test=True (final confirm);
  - identical capacity/optimizer/protocol across all factor/mode probes;
  - frozen factorizer; no test-based hyperparameter selection.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score

from src.data import EdgeSplit, MAGData
from src.data.graph_utils import edge_dict_to_index
from src.models import LinkPredictor
from src.tasks.lp import _build_epoch_train_labels, _build_forbidden_edge_keys, _edge_keys
from src.utils.biaxis_p0_diagnostics import compute_conflict_statistics, propagate_fixed_gcn
from src.utils.seeds import set_seed

FACTOR_NAMES = ("C", "Pt", "Pv")


def _factor_matrix(factors: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name == "C":
        return factors["c"]
    if name == "Pt":
        return factors["p_t"]
    if name == "Pv":
        return factors["p_v"]
    raise ValueError(f"unknown factor {name!r}")


@torch.no_grad()
def _evaluate_classifier(
    classifier: nn.Module,
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


@torch.no_grad()
def _per_node_cross_entropy(
    classifier: nn.Module,
    z: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    classifier.eval()
    ces: list[torch.Tensor] = []
    for batch_idx in TorchDataLoader(idx.cpu(), batch_size=batch_size, shuffle=False):
        logits = classifier(z[batch_idx].to(device))
        ces.append(nn.functional.cross_entropy(logits, labels[batch_idx].to(device), reduction="none").cpu())
    return torch.cat(ces, dim=0)


def _train_linear_probe(
    z: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    num_classes: int,
    device: torch.device,
    probe_cfg,
    seed: int,
    batch_size: int,
) -> tuple[nn.Linear, dict[str, float]]:
    """Identical protocol for every factor/mode probe."""
    set_seed(seed)
    classifier = nn.Linear(int(z.size(1)), int(num_classes)).to(device)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=float(probe_cfg.lr),
        weight_decay=float(probe_cfg.weight_decay),
    )
    criterion = nn.CrossEntropyLoss()
    patience_total = int(probe_cfg.patience)
    patience_left = patience_total
    best_val_acc = -1.0
    best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}

    for epoch in range(1, int(probe_cfg.epochs) + 1):
        classifier.train()
        for batch_idx in TorchDataLoader(train_idx.cpu(), batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(z[batch_idx].to(device))
            loss = criterion(logits, labels[batch_idx].to(device))
            loss.backward()
            optimizer.step()
        val_metrics = _evaluate_classifier(classifier, z, labels, val_idx, device, batch_size)
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
            patience_left = patience_total
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    classifier.load_state_dict(best_state)
    val_metrics = _evaluate_classifier(classifier, z, labels, val_idx, device, batch_size)
    return classifier, val_metrics


def run_nc_factor_probes(
    factors: dict[str, torch.Tensor],
    data: MAGData,
    device: torch.device,
    probe_cfg,
    output_dir: str | Path | None = None,
    include_test: bool = False,
    seed: int = 42,
    batch_size: int = 4096,
) -> dict:
    """P0-D NC: factor-wise local vs fixed-GCN propagation utility.

    factors: output of biaxis_p0 Model.encode_factors() (CPU tensors).
    probe_cfg: cfg.model.p0.probe (epochs/patience/lr/weight_decay).
    """
    if data.y is None or data.train_idx is None or data.val_idx is None or data.test_idx is None:
        raise ValueError("NC probe requires y/train_idx/val_idx/test_idx")
    num_nodes = int(factors["c"].size(0))
    if int(data.edge_index.max()) >= num_nodes or data.edge_index.size(1) == 0:
        raise ValueError("edge_index out of range or empty")

    rows: list[dict] = []
    node_deltas: dict[str, dict[str, torch.Tensor]] = {}
    conflict_input: dict[str, torch.Tensor] = {}
    label_tensor = data.y
    train_idx = data.train_idx
    val_idx = data.val_idx
    test_idx = data.test_idx

    for name in FACTOR_NAMES:
        f_local = _factor_matrix(factors, name)
        f_graph = propagate_fixed_gcn(f_local, data.edge_index, num_nodes=num_nodes)
        f_local = f_local.to(device)
        f_graph = f_graph.to(device)

        classifier_local, val_local = _train_linear_probe(
            f_local, label_tensor, train_idx, val_idx, int(data.num_classes),
            device, probe_cfg, seed, batch_size,
        )
        classifier_graph, val_graph = _train_linear_probe(
            f_graph, label_tensor, train_idx, val_idx, int(data.num_classes),
            device, probe_cfg, seed, batch_size,
        )

        row = {
            "factor": name,
            "mode": "local",
            "val_acc": val_local["acc"],
            "val_macro_f1": val_local["macro_f1"],
            "test_acc": None,
            "test_macro_f1": None,
        }
        rows.append(row)
        row = {
            "factor": name,
            "mode": "graph",
            "val_acc": val_graph["acc"],
            "val_macro_f1": val_graph["macro_f1"],
            "test_acc": None,
            "test_macro_f1": None,
        }
        rows.append(row)

        ce_local = _per_node_cross_entropy(classifier_local, f_local, label_tensor, val_idx, device, batch_size)
        ce_graph = _per_node_cross_entropy(classifier_graph, f_graph, label_tensor, val_idx, device, batch_size)
        delta = ce_local - ce_graph
        node_deltas[name] = {"ce_local": ce_local, "ce_graph": ce_graph, "delta": delta}
        conflict_input[name] = delta

        if include_test:
            test_local = _evaluate_classifier(classifier_local, f_local, label_tensor, test_idx, device, batch_size)
            test_graph = _evaluate_classifier(classifier_graph, f_graph, label_tensor, test_idx, device, batch_size)
            rows[-2]["test_acc"] = test_local["acc"]
            rows[-2]["test_macro_f1"] = test_local["macro_f1"]
            rows[-1]["test_acc"] = test_graph["acc"]
            rows[-1]["test_macro_f1"] = test_graph["macro_f1"]

        del classifier_local, classifier_graph, f_local, f_graph
        torch.cuda.empty_cache()

    conflict_stats = compute_conflict_statistics(conflict_input, names=FACTOR_NAMES)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(node_deltas, output_dir / "nc_node_delta.pt")
        _write_conflict_json(conflict_stats, output_dir / "conflict_stats.json")

        with (output_dir / "nc_probe.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["factor", "mode", "val_acc", "val_macro_f1", "test_acc", "test_macro_f1"],
            )
            writer.writeheader()
            writer.writerows(rows)

    return {"nc_probe": rows, "nc_conflict": conflict_stats, "nc_node_delta": node_deltas}


# ---------------------------------------------------------------------------
# P0-D LP probe
# ---------------------------------------------------------------------------


def _canonical_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row, col = edge_index[0].long(), edge_index[1].long()
    a = torch.minimum(row, col)
    b = torch.maximum(row, col)
    return torch.unique(a * int(num_nodes) + b)


def _count_intersection(sorted_a: torch.Tensor, sorted_b: torch.Tensor) -> int:
    positions = torch.searchsorted(sorted_b, sorted_a)
    in_bounds = positions < sorted_b.numel()
    if not bool(in_bounds.any()):
        return 0
    clamped = positions.clamp(max=sorted_b.numel() - 1)
    return int((in_bounds & (sorted_b[clamped] == sorted_a)).sum().item())


def assert_no_edge_leakage(edge_index: torch.Tensor, edge_split: EdgeSplit, num_nodes: int) -> dict[str, int]:
    """Guard against real edge leakage (plan §13).

    The LP message graph is train-only by construction. The published
    DIRECTED splits additionally have valid/test queries whose reverse
    direction is a train edge (valid (u,v) with (v,u) in train) — that
    overlap is inherent to the public-split semantics and is shared by the
    main LP protocol, so it is REPORTED, not flagged.

    Real leakage = a valid/test edge whose BOTH directions are absent from
    the train set appearing in the propagation graph (i.e. someone built
    A_full = train + valid + test).
    """
    train_keys = _canonical_keys(edge_dict_to_index(edge_split.train), int(num_nodes))
    message_keys = _canonical_keys(edge_index, int(num_nodes))
    extra = message_keys
    positions = torch.searchsorted(train_keys, message_keys)
    in_bounds = positions < train_keys.numel()
    if bool(in_bounds.any()):
        keep = ~(in_bounds & (train_keys[positions[in_bounds].clamp(max=train_keys.numel() - 1)] == message_keys[in_bounds]))
        extra = message_keys[in_bounds][keep]
    overlap_report: dict[str, int] = {}
    for split_name in ("valid", "test"):
        pos = edge_dict_to_index(getattr(edge_split, split_name))
        pos_keys = _canonical_keys(pos, int(num_nodes))
        overlap_report[split_name] = _count_intersection(pos_keys, train_keys)
        leaked = _count_intersection(pos_keys, extra)
        if leaked:
            raise ValueError(
                f"edge leakage: propagation graph contains {leaked} {split_name} positive "
                f"edges whose both directions are absent from the train set"
            )
    return overlap_report


def _subset_split(split: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[idx] for key, value in split.items()}


@torch.no_grad()
def _evaluate_lp_split_with_rr(
    z: torch.Tensor,
    predictor: nn.Module,
    split: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], torch.Tensor]:
    """MRR/Hits + per-positive-edge reciprocal rank (same ranking rule as lp.py).

    z may live on GPU: all row gathers then happen on-device (no CPU
    gather / PCIe copy per batch).
    """
    predictor.eval()
    src_all = split["source_node"]
    dst_all = split["target_node"]
    neg_all = split["target_node_neg"]
    total = int(src_all.numel())
    rr = torch.empty(total, dtype=torch.float32)
    mrr_sum = 0.0
    hits1_sum = 0.0
    hits3_sum = 0.0
    hits10_sum = 0.0
    z_device = z.device
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        src = src_all[start:end].to(z_device).long()
        dst = dst_all[start:end].to(z_device).long()
        neg = neg_all[start:end].to(z_device).long()
        pos_score = predictor.score_pairs(z[src], z[dst])
        neg_score = predictor.score_pairs(
            z[src.view(-1, 1).expand_as(neg).reshape(-1)], z[neg.reshape(-1)]
        ).view(neg.size(0), neg.size(1))
        # Pessimistic ties (RPTA/OpenMAG rule, same as the LP task runner).
        ranks = 1.0 + (neg_score >= pos_score.view(-1, 1)).sum(dim=1).float()
        rr[start:end] = (1.0 / ranks).cpu()
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
    }, rr


def _train_lp_probe(
    z: torch.Tensor,
    edge_split: EdgeSplit,
    num_nodes: int,
    device: torch.device,
    probe_cfg,
    seed: int,
    batch_size: int,
    eval_batch_size: int,
) -> tuple[LinkPredictor, dict[str, float], torch.Tensor]:
    """Identical protocol for every factor/mode probe (plan §12)."""
    set_seed(seed)
    predictor = LinkPredictor(
        in_dim=int(z.size(1)),
        hidden_dim=int(probe_cfg.get("decoder_hidden_dim", 256)),
        num_layers=int(probe_cfg.get("decoder_num_layers", 3)),
        dropout=float(probe_cfg.get("decoder_dropout", 0.02)),
    ).to(device)
    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=float(probe_cfg.lr),
        weight_decay=float(probe_cfg.weight_decay),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    undirected = bool(edge_split.metadata.get("undirected", True))
    forbidden_keys = _build_forbidden_edge_keys(edge_split, num_nodes, undirected=undirected)
    train_pos = edge_dict_to_index(edge_split.train).cpu()
    neg_generator = torch.Generator().manual_seed(seed)
    num_train_neg = int(probe_cfg.get("num_train_neg", 1))
    train_pos_per_epoch = probe_cfg.get("train_pos_per_epoch")
    if train_pos_per_epoch is not None:
        train_pos_per_epoch = int(train_pos_per_epoch)

    # Fixed validation subset for early stopping (same subset for all probes).
    num_val = int(edge_split.valid["source_node"].numel())
    val_sub_idx = None
    eval_subset_size = probe_cfg.get("eval_subset_size")
    if eval_subset_size and int(eval_subset_size) < num_val:
        generator = torch.Generator().manual_seed(seed)
        val_sub_idx = torch.randperm(num_val, generator=generator)[: int(eval_subset_size)]

    patience_total = int(probe_cfg.patience)
    patience_left = patience_total
    best_val_mrr = -1.0
    best_state = {key: value.detach().cpu().clone() for key, value in predictor.state_dict().items()}

    for epoch in range(1, int(probe_cfg.epochs) + 1):
        predictor.train()
        edge_label_index, edge_label = _build_epoch_train_labels(
            train_pos,
            num_nodes,
            num_train_neg,
            forbidden_keys,
            neg_generator,
            train_pos_per_epoch=train_pos_per_epoch,
        )
        loader = TorchDataLoader(
            TensorDataset(edge_label_index.t().contiguous(), edge_label.contiguous()),
            batch_size=batch_size,
            shuffle=True,
        )
        for edges, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            src, dst = edges.t().contiguous()
            logits = predictor.score_pairs(z[src].to(device), z[dst].to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
        val_split = _subset_split(edge_split.valid, val_sub_idx) if val_sub_idx is not None else edge_split.valid
        val_metrics, _ = _evaluate_lp_split_with_rr(z, predictor, val_split, device, eval_batch_size)
        if val_metrics["mrr"] > best_val_mrr:
            best_val_mrr = val_metrics["mrr"]
            best_state = {key: value.detach().cpu().clone() for key, value in predictor.state_dict().items()}
            patience_left = patience_total
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    predictor.load_state_dict(best_state)
    full_val_metrics, rr_val = _evaluate_lp_split_with_rr(z, predictor, edge_split.valid, device, eval_batch_size)
    return predictor, full_val_metrics, rr_val


def run_lp_factor_probes(
    factors: dict[str, torch.Tensor],
    data: MAGData,
    device: torch.device,
    probe_cfg,
    output_dir: str | Path | None = None,
    include_test: bool = False,
    seed: int = 42,
    batch_size: int = 2048,
    eval_batch_size: int = 2048,
) -> dict:
    """P0-D LP: factor-wise local vs fixed-GCN LinkPredictor utility.

    Propagation graph is data.edge_index (train-edge-only by construction).
    delta_RR = RR_graph - RR_local per validation positive edge.
    """
    if data.edge_split is None:
        raise ValueError("LP probe requires edge_split")
    num_nodes = int(factors["c"].size(0))
    if int(data.edge_index.max()) >= num_nodes or data.edge_index.size(1) == 0:
        raise ValueError("edge_index out of range or empty")
    inherent_overlap = assert_no_edge_leakage(data.edge_index, data.edge_split, num_nodes)

    rows: list[dict] = []
    edge_deltas: dict[str, dict[str, torch.Tensor]] = {}
    conflict_input: dict[str, torch.Tensor] = {}

    for name in FACTOR_NAMES:
        f_local = _factor_matrix(factors, name)
        f_graph = propagate_fixed_gcn(f_local, data.edge_index, num_nodes=num_nodes)
        f_local = f_local.to(device)
        f_graph = f_graph.to(device)

        predictor_local, val_local, rr_local = _train_lp_probe(
            f_local, data.edge_split, num_nodes, device, probe_cfg, seed, batch_size, eval_batch_size
        )
        predictor_graph, val_graph, rr_graph = _train_lp_probe(
            f_graph, data.edge_split, num_nodes, device, probe_cfg, seed, batch_size, eval_batch_size
        )
        edge_deltas[name] = {"rr_local": rr_local, "rr_graph": rr_graph, "delta": rr_graph - rr_local}
        conflict_input[name] = edge_deltas[name]["delta"]

        row_local = {"factor": name, "mode": "local"}
        row_graph = {"factor": name, "mode": "graph"}
        for key in ("mrr", "hits@1", "hits@3", "hits@10"):
            row_local[f"val_{key}"] = val_local[key]
            row_graph[f"val_{key}"] = val_graph[key]
            row_local[f"test_{key}"] = None
            row_graph[f"test_{key}"] = None
        rows.append(row_local)
        rows.append(row_graph)

        if include_test:
            test_local, _ = _evaluate_lp_split_with_rr(f_local, predictor_local, data.edge_split.test, device, eval_batch_size)
            test_graph, _ = _evaluate_lp_split_with_rr(f_graph, predictor_graph, data.edge_split.test, device, eval_batch_size)
            for key in ("mrr", "hits@1", "hits@3", "hits@10"):
                rows[-2][f"test_{key}"] = test_local[key]
                rows[-1][f"test_{key}"] = test_graph[key]

        del predictor_local, predictor_graph, f_local, f_graph
        torch.cuda.empty_cache()

    conflict_stats = compute_conflict_statistics(conflict_input, names=FACTOR_NAMES)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(edge_deltas, output_dir / "lp_edge_delta.pt")
        _write_conflict_json(conflict_stats, output_dir / "conflict_stats.json")
        fieldnames = [
            "factor",
            "mode",
            "val_mrr",
            "val_hits@1",
            "val_hits@3",
            "val_hits@10",
            "test_mrr",
            "test_hits@1",
            "test_hits@3",
            "test_hits@10",
        ]
        with (output_dir / "lp_probe.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return {
        "lp_probe": rows,
        "lp_conflict": conflict_stats,
        "lp_edge_delta": edge_deltas,
        "lp_split_overlap": inherent_overlap,
    }


def _write_conflict_json(stats: dict, path: Path) -> None:
    def _clean(value):
        if isinstance(value, dict):
            return {key: _clean(item) for key, item in value.items()}
        if isinstance(value, float) and value != value:  # NaN -> None
            return None
        return value

    with path.open("w", encoding="utf-8") as f:
        json.dump(_clean(stats), f, indent=2)
