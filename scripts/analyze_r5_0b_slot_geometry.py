"""Extract slot-geometry diagnostics from frozen R5-0b checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_mag_data
from src.models import build_model
from src.models.biaxis_scope_v2_components import chunked_factor_sym_norm

DATASETS = ["Movies", "Toys", "Grocery"]
SEEDS = [42, 43, 44]
VARIANTS = {
    "B2-FSEP4": (4, "learned"),
    "B3-FSEP1": (1, "learned"),
    "B4-UNIFORM4": (4, "uniform"),
}
FACTORS = ("c", "pt", "pv")
OUT_ROOT = PROJECT_ROOT / "outputs" / "r5_0b_global_relay_audit"


def _config(dataset: str, seed: int, num_slots: int, assignment_mode: str):
    base = OmegaConf.load(PROJECT_ROOT / "configs/config.yaml")
    base.paths.data_root = "/hdd1/DataInHere/YHF/data"
    base.paths.split_root = "/hdd1/DataInHere/YHF/data/MAGB_split"
    dataset_cfg = OmegaConf.load(PROJECT_ROOT / f"configs/dataset/{dataset}.yaml")
    model_cfg = OmegaConf.load(PROJECT_ROOT / "configs/model/biaxis_scope_v2.yaml")
    model_cfg.scope.graph_mode = "local_global"
    model_cfg.scope.local_aggregation = "sym_norm"
    model_cfg.scope.post_context_norm = False
    model_cfg.scope.context_core = "oacc"
    model_cfg.scope.beta_mode = "one"
    model_cfg.scope.num_blocks = 2
    model_cfg.scope.reconciliation.mode = "none"
    model_cfg.scope.use_source_projection = False
    model_cfg.scope.global_num_slots = int(num_slots)
    model_cfg.scope.global_assignment_mode = assignment_mode
    model_cfg.scope.global_scale_init = 0.05
    task_cfg = OmegaConf.load(PROJECT_ROOT / "configs/task/nc.yaml")
    cfg = OmegaConf.create({
        "seed": int(seed), "num_runs": 1, "device": "cpu",
        "paths": base.paths, "logging": {"level": "INFO"},
        "dataset": dataset_cfg, "model": model_cfg, "task": task_cfg,
    })
    OmegaConf.resolve(cfg)
    return cfg


def _load_variant(dataset: str, seed: int, variant: str, device: torch.device):
    num_slots, assignment_mode = VARIANTS[variant]
    cfg = _config(dataset, seed, num_slots, assignment_mode)
    data = load_mag_data(cfg, "nc", seed)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
        "y": data.y,
        "train_idx": data.train_idx,
    }
    model = build_model(cfg, data_info).to(device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    checkpoint_path = OUT_ROOT / "formal" / variant / dataset / f"seed_{seed}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    classifier.load_state_dict(checkpoint["head_state"])
    model.eval()
    classifier.eval()
    return data, model, classifier, checkpoint_path


def _slot_metrics(
    assignment: torch.Tensor,
    slots: torch.Tensor,
    global_output: torch.Tensor,
    local: torch.Tensor,
    scale: torch.Tensor,
    num_slots: int,
) -> dict[str, float]:
    entropy = -(assignment.clamp_min(1e-8) * assignment.clamp_min(1e-8).log()).sum(dim=-1)
    quantiles = torch.quantile(entropy, torch.tensor([0.1, 0.5, 0.9], device=entropy.device))
    usage = assignment.mean(dim=0)
    assignment_std = assignment.std(dim=0, unbiased=False)
    gram = assignment.transpose(0, 1) @ assignment
    singular = torch.linalg.svdvals(gram)
    stable_rank = float((singular.square().sum() / singular.square().max().clamp_min(1e-12)).item())
    if num_slots > 1:
        normalized_slots = F.normalize(slots, dim=-1)
        cosine_matrix = normalized_slots @ normalized_slots.transpose(0, 1)
        distances = torch.cdist(slots, slots)
        mask = ~torch.eye(num_slots, dtype=torch.bool, device=slots.device)
        cosine_mean = float(cosine_matrix[mask].mean().item())
        distance_mean = float(distances[mask].mean().item())
        pairwise_defined = 1
    else:
        cosine_mean = 0.0
        distance_mean = 0.0
        pairwise_defined = 0
    global_local_ratio = global_output.norm(dim=-1).mean() / local.norm(dim=-1).mean().clamp_min(1e-8)
    scale_value = float(scale.detach().item())
    values = {
        "assignment_entropy_mean": float(entropy.mean().item()),
        "assignment_entropy_p10": float(quantiles[0].item()),
        "assignment_entropy_p50": float(quantiles[1].item()),
        "assignment_entropy_p90": float(quantiles[2].item()),
        "normalized_entropy_mean": float(entropy.mean().item() / torch.log(torch.tensor(float(num_slots))).item()) if num_slots > 1 else 0.0,
        "normalized_entropy_p10": float(quantiles[0].item() / torch.log(torch.tensor(float(num_slots))).item()) if num_slots > 1 else 0.0,
        "normalized_entropy_p50": float(quantiles[1].item() / torch.log(torch.tensor(float(num_slots))).item()) if num_slots > 1 else 0.0,
        "normalized_entropy_p90": float(quantiles[2].item() / torch.log(torch.tensor(float(num_slots))).item()) if num_slots > 1 else 0.0,
        "assignment_std_mean": float(assignment_std.mean().item()),
        "assignment_std_max": float(assignment_std.max().item()),
        "assignment_stable_rank": stable_rank,
        "slot_cosine_mean_offdiag": cosine_mean,
        "slot_distance_mean_offdiag": distance_mean,
        "pairwise_geometry_defined": pairwise_defined,
        "global_output_variance": float(global_output.var(dim=0, unbiased=False).mean().item()),
        "global_local_norm_ratio": float(global_local_ratio.item()),
        "global_scale": scale_value,
        "actual_injection_ratio": float(scale_value * global_local_ratio.item()),
    }
    for slot_index, value in enumerate(usage):
        values[f"slot_usage_{slot_index}"] = float(value.item())
    return values


@torch.no_grad()
def _analyze_checkpoint(dataset: str, seed: int, variant: str, device: torch.device) -> list[dict]:
    data, model, classifier, checkpoint_path = _load_variant(dataset, seed, variant, device)
    del classifier
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    rows: list[dict] = []
    handles = []

    def make_hook(layer: int):
        def hook(module, inputs, output):
            h = inputs[0]
            factor = int(inputs[1])
            assignment = module._assignment(h, factor)
            value = module.value[factor](h)
            denom = assignment.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
            slots = assignment.transpose(0, 1) @ value / denom
            slots = module.slot_update[factor](slots)
            global_output = module.readout[factor](assignment @ slots)
            local = chunked_factor_sym_norm(
                edge_index, h, int(h.size(0)), int(model.edge_chunk_size)
            )
            metrics = _slot_metrics(
                assignment, slots, global_output, local,
                model.global_scales[layer][factor], module.num_slots,
            )
            rows.append({
                "variant": variant,
                "dataset": dataset,
                "seed": seed,
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "layer": layer + 1,
                "factor": FACTORS[factor],
                "num_slots": module.num_slots,
                "assignment_mode": module.assignment_mode,
                **metrics,
            })
        return hook

    for layer, relay in enumerate(model.global_relays):
        handles.append(relay.register_forward_hook(make_hook(layer)))
    model(x, edge_index)
    for handle in handles:
        handle.remove()
    del x, edge_index
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    device = torch.device(args.device)
    datasets = [value for value in args.datasets.split(",") if value in DATASETS]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    rows = []
    for variant in VARIANTS:
        for dataset in datasets:
            for seed in seeds:
                print(f"[geometry] {variant} {dataset} seed={seed}", flush=True)
                rows.extend(_analyze_checkpoint(dataset, seed, variant, device))
    summary_dir = OUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / "R5_0B_SLOT_GEOMETRY.csv"
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] wrote {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()

