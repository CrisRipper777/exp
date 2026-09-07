"""Conditional validation-only interventions for the nine C3 checkpoints.

The script refuses to run unless the R5-1 decision JSON says
``OSCI_SUPPORTED=true``.  It evaluates validation nodes only and restores the
model/intervention state after every intervention.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_mag_data
from src.models import build_model

DATASETS = ["Movies", "Toys", "Grocery"]
SEEDS = [42, 43, 44]
INTERVENTIONS = (
    "I0-NORMAL",
    "I1-ZERO-CROSS",
    "I2-SHUFFLE-SOURCES",
    "I3-SHUFFLE-Q",
    "I4-DUP",
)
OUT_ROOT = PROJECT_ROOT / "outputs" / "r5_1_osci"


def _config(dataset: str, seed: int):
    base = OmegaConf.load(PROJECT_ROOT / "configs/config.yaml")
    base.paths.data_root = "/hdd1/DataInHere/YHF/data"
    base.paths.split_root = "/hdd1/DataInHere/YHF/data/MAGB_split"
    dataset_cfg = OmegaConf.load(PROJECT_ROOT / f"configs/dataset/{dataset}.yaml")
    model_cfg = OmegaConf.load(PROJECT_ROOT / "configs/model/biaxis_osci.yaml")
    task_cfg = OmegaConf.load(PROJECT_ROOT / "configs/task/nc.yaml")
    model_cfg.scope.graph_mode = "local"
    model_cfg.scope.local_aggregation = "sym_norm"
    model_cfg.scope.post_context_norm = False
    model_cfg.scope.context_core = "oacc"
    model_cfg.scope.beta_mode = "one"
    model_cfg.scope.num_blocks = 2
    model_cfg.scope.reconciliation.mode = "none"
    model_cfg.scope.use_source_projection = False
    model_cfg.osci.mode = "c3-osci"
    model_cfg.osci.composer_hidden_dim = 256
    model_cfg.osci.cross_scale_init = 0.05
    cfg = OmegaConf.create({
        "seed": int(seed), "num_runs": 1, "device": "cpu",
        "paths": base.paths, "logging": {"level": "INFO"},
        "dataset": dataset_cfg, "model": model_cfg, "task": task_cfg,
    })
    cfg.task.evaluate_test = False
    OmegaConf.resolve(cfg)
    return cfg


def _build_model_and_data(dataset: str, seed: int, device: torch.device):
    cfg = _config(dataset, seed)
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
    return data, model, classifier


def _load_checkpoint(dataset: str, seed: int, device: torch.device):
    data, model, classifier = _build_model_and_data(dataset, seed, device)
    checkpoint_path = OUT_ROOT / "formal" / "C3-OSCI" / dataset / f"seed_{seed}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    classifier.load_state_dict(checkpoint["head_state"])
    model.eval()
    classifier.eval()
    return data, model, classifier, checkpoint_path


@torch.no_grad()
def _evaluate(data, model, classifier, device: torch.device) -> tuple[float, float]:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    z, _, _, _, _ = model(x, edge_index)
    val_idx = data.val_idx.to(device)
    logits = classifier(z[val_idx])
    pred = logits.argmax(dim=-1).cpu()
    target = data.y[data.val_idx].cpu()
    acc = float((pred == target).float().mean().item())
    f1 = float(f1_score(target.numpy(), pred.numpy(), average="macro", zero_division=0))
    return acc, f1


def _snapshot(model, classifier) -> dict[str, torch.Tensor]:
    state = {}
    for prefix, module in (("model.", model), ("classifier.", classifier)):
        for key, value in module.state_dict().items():
            state[prefix + key] = value.detach().cpu().clone()
    return state


def _state_equal(before: dict[str, torch.Tensor], model, classifier) -> bool:
    after = _snapshot(model, classifier)
    return before.keys() == after.keys() and all(torch.equal(before[key], after[key]) for key in before)


def _permutation(dataset: str, seed: int, layer: int, target: int, channel: int, num_nodes: int, device):
    dataset_index = DATASETS.index(dataset)
    permutation_seed = (
        20260907 + dataset_index * 1_000_000 + seed * 10_000
        + layer * 100 + target * 10 + channel
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(permutation_seed)
    return permutation_seed, torch.randperm(num_nodes, generator=generator).to(device)


def _apply_intervention(intervention: str, model, dataset: str, seed: int, num_nodes: int, device):
    if intervention == "I0-NORMAL":
        return None
    if intervention == "I1-ZERO-CROSS":
        model._osci_intervention = {"mode": "zero_cross"}
        return None
    if intervention == "I4-DUP":
        model._osci_intervention = {"mode": "duplicate"}
        return None
    if intervention == "I3-SHUFFLE-Q":
        permutations = {}
        seeds = []
        for layer in range(model.num_blocks):
            for target in range(3):
                permutation_seed, permutation = _permutation(
                    dataset, seed, layer, target, 9, num_nodes, device
                )
                permutations[(layer, target)] = permutation
                seeds.append({"layer": layer + 1, "target": target, "seed": permutation_seed})
        model._osci_intervention = {
            "mode": "shuffle_target_q", "target_permutations": permutations,
        }
        return seeds
    if intervention == "I2-SHUFFLE-SOURCES":
        permutations = {}
        seeds = []
        for layer in range(model.num_blocks):
            for target in range(3):
                for channel in range(2):
                    permutation_seed, permutation = _permutation(
                        dataset, seed, layer, target, channel, num_nodes, device
                    )
                    permutations[(layer, target, channel)] = permutation
                    seeds.append({
                        "layer": layer + 1, "target": target,
                        "channel": channel, "seed": permutation_seed,
                    })
        model._osci_intervention = {
            "mode": "shuffle_sources", "source_permutations": permutations,
        }
        return seeds
    raise ValueError(f"unknown intervention: {intervention}")


def _restore(model) -> None:
    if hasattr(model, "_osci_intervention"):
        del model._osci_intervention


def _run_checkpoint(dataset: str, seed: int, device: torch.device) -> list[dict]:
    data, model, classifier, checkpoint_path = _load_checkpoint(dataset, seed, device)
    history_path = OUT_ROOT / "formal" / "C3-OSCI" / dataset / f"seed_{seed}" / "history.csv"
    with history_path.open(encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    best_history = max(history, key=lambda row: float(row["val_acc"]))
    normal_acc, normal_f1 = _evaluate(data, model, classifier, device)
    expected_acc = float(best_history["val_acc"])
    expected_f1 = float(best_history["val_macro_f1"])
    i0_consistent = abs(normal_acc - expected_acc) <= 2e-5 and abs(normal_f1 - expected_f1) <= 2e-5
    if not i0_consistent:
        raise RuntimeError(f"I0 mismatch for {dataset}/seed{seed}")

    rows = []
    for intervention in INTERVENTIONS:
        before = _snapshot(model, classifier)
        permutation_seeds = _apply_intervention(
            intervention, model, dataset, seed, data.num_nodes, device
        ) or []
        try:
            acc, f1 = _evaluate(data, model, classifier, device)
        finally:
            _restore(model)
        restored = _state_equal(before, model, classifier) and not hasattr(model, "_osci_intervention")
        rows.append({
            "dataset": dataset, "seed": seed,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "intervention": intervention,
            "val_acc_pct": 100.0 * acc, "val_f1_pct": 100.0 * f1,
            "delta_acc_vs_I0_pp": 100.0 * (acc - normal_acc),
            "delta_f1_vs_I0_pp": 100.0 * (f1 - normal_f1),
            "i0_consistent": i0_consistent,
            "state_restored": restored,
            "permutation_seeds": json.dumps(permutation_seeds, separators=(",", ":")),
        })
        if not restored:
            raise RuntimeError(f"intervention state did not restore: {dataset}/seed{seed}/{intervention}")
    checkpoint_dir = OUT_ROOT / "interventions" / dataset / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    del data, model, classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    decision_path = OUT_ROOT / "summary/R5_1_DECISION.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not bool(decision["decision_rules"]["OSCI_SUPPORTED"]["value"]):
        print("[intervention] OSCI_SUPPORTED=false; nine-checkpoint intervention audit not triggered")
        return
    device = torch.device(args.device)
    datasets = [value for value in args.datasets.split(",") if value in DATASETS]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    all_rows = []
    for dataset in datasets:
        for seed in seeds:
            print(f"[intervention] {dataset} seed={seed}", flush=True)
            all_rows.extend(_run_checkpoint(dataset, seed, device))
    summary_dir = OUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "seed", "checkpoint", "intervention", "val_acc_pct", "val_f1_pct",
        "delta_acc_vs_I0_pp", "delta_f1_vs_I0_pp", "i0_consistent", "state_restored",
        "permutation_seeds",
    ]
    with (summary_dir / "R5_1_INTERVENTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[summary] wrote {summary_dir / 'R5_1_INTERVENTIONS.csv'} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
