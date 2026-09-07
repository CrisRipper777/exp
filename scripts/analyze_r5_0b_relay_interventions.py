"""Validation-only causal interventions for the nine B2-FSEP4 checkpoints."""

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
FACTORS = ("c", "pt", "pv")
INTERVENTIONS = (
    "I0-NORMAL",
    "I1-ZERO-ALL",
    "I2-ZERO-C",
    "I3-ZERO-PT",
    "I4-ZERO-PV",
    "I5-UNIFORM-ASSIGN",
    "I6-SHUFFLE-COLLECT",
    "I7-SHUFFLE-DISPATCH",
    "I8-EQUAL-SCALE",
)
OUT_ROOT = PROJECT_ROOT / "outputs" / "r5_0b_global_relay_audit"


def _config(dataset: str, seed: int):
    base = OmegaConf.load(PROJECT_ROOT / "configs/config.yaml")
    base.paths.data_root = "/hdd1/DataInHere/YHF/data"
    base.paths.split_root = "/hdd1/DataInHere/YHF/data/MAGB_split"
    dataset_cfg = OmegaConf.load(PROJECT_ROOT / f"configs/dataset/{dataset}.yaml")
    model_cfg = OmegaConf.load(PROJECT_ROOT / "configs/model/biaxis_scope_v2.yaml")
    task_cfg = OmegaConf.load(PROJECT_ROOT / "configs/task/nc.yaml")
    model_cfg.scope.graph_mode = "local_global"
    model_cfg.scope.local_aggregation = "sym_norm"
    model_cfg.scope.post_context_norm = False
    model_cfg.scope.context_core = "oacc"
    model_cfg.scope.beta_mode = "one"
    model_cfg.scope.num_blocks = 2
    model_cfg.scope.reconciliation.mode = "none"
    model_cfg.scope.use_source_projection = False
    model_cfg.scope.global_num_slots = 4
    model_cfg.scope.global_assignment_mode = "learned"
    model_cfg.scope.global_scale_init = 0.05
    cfg = OmegaConf.create({
        "seed": int(seed),
        "num_runs": 1,
        "device": "cpu",
        "paths": base.paths,
        "logging": {"level": "INFO"},
        "dataset": dataset_cfg,
        "model": model_cfg,
        "task": task_cfg,
    })
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
    return cfg, data, model, classifier


def _load_checkpoint(dataset: str, seed: int, device: torch.device):
    _, data, model, classifier = _build_model_and_data(dataset, seed, device)
    checkpoint_path = OUT_ROOT / "formal" / "B2-FSEP4" / dataset / f"seed_{seed}" / "best.pt"
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
    del x, edge_index, z, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return acc, f1


def _state_snapshot(model, classifier) -> dict[str, torch.Tensor]:
    state = {}
    for prefix, module in (("model.", model), ("classifier.", classifier)):
        for key, value in module.state_dict().items():
            state[prefix + key] = value.detach().cpu().clone()
    return state


def _state_equal(before: dict[str, torch.Tensor], model, classifier) -> bool:
    after = _state_snapshot(model, classifier)
    return before.keys() == after.keys() and all(torch.equal(before[key], after[key]) for key in before)


def _relays(model):
    return list(model.global_relays)


def _permutation(dataset: str, seed: int, layer: int, factor: int, intervention: str, num_nodes: int, device):
    dataset_index = DATASETS.index(dataset)
    intervention_offset = 0 if intervention == "I6-SHUFFLE-COLLECT" else 1
    perm_seed = 20260907 + dataset_index * 1_000_000 + seed * 1_000 + layer * 10 + factor * 2 + intervention_offset
    generator = torch.Generator(device="cpu")
    generator.manual_seed(perm_seed)
    permutation = torch.randperm(num_nodes, generator=generator, device="cpu").to(device)
    return perm_seed, permutation


def _apply_intervention_direct(intervention: str, model, dataset: str, seed: int, num_nodes: int, device):
    """Apply interventions without replacing forward methods.

    Shuffle interventions use a small per-factor wrapper because the relay's
    public forward receives the factor id at call time.
    """
    relays = _relays(model)
    saved = [
        (
            relay,
            relay.assignment_mode,
            getattr(relay, "_intervention_mode", None),
            getattr(relay, "_intervention_permutation", None),
        )
        for relay in relays
    ]
    saved_scales = [scale.detach().clone() for scale in model.global_scales]
    permutation_seeds = []
    wrappers = []
    if intervention in {"I1-ZERO-ALL", "I2-ZERO-C", "I3-ZERO-PT", "I4-ZERO-PV"}:
        target = None if intervention == "I1-ZERO-ALL" else {"I2-ZERO-C": 0, "I3-ZERO-PT": 1, "I4-ZERO-PV": 2}[intervention]
        for scale in model.global_scales:
            with torch.no_grad():
                if target is None:
                    scale.zero_()
                else:
                    scale[target] = 0.0
    elif intervention == "I5-UNIFORM-ASSIGN":
        for relay in relays:
            relay.assignment_mode = "uniform"
    elif intervention in {"I6-SHUFFLE-COLLECT", "I7-SHUFFLE-DISPATCH"}:
        mode = "shuffle_collect" if intervention == "I6-SHUFFLE-COLLECT" else "shuffle_dispatch"
        for layer, relay in enumerate(relays):
            original_forward = relay.forward
            permutation_map = {}
            for factor in range(3):
                perm_seed, permutation = _permutation(dataset, seed, layer, factor, intervention, num_nodes, device)
                permutation_map[factor] = permutation
                permutation_seeds.append({"layer": layer + 1, "factor": FACTORS[factor], "seed": perm_seed})

            def factor_forward(h, factor, return_stats=False, _original=original_forward, _relay=relay, _map=permutation_map, _mode=mode):
                _relay._intervention_mode = _mode
                _relay._intervention_permutation = _map[int(factor)]
                return _original(h, factor, return_stats=return_stats)

            relay.forward = factor_forward
            wrappers.append((relay, original_forward))
    elif intervention == "I8-EQUAL-SCALE":
        for scale in model.global_scales:
            with torch.no_grad():
                scale.fill_(float(scale.detach().mean().item()))
    return saved, saved_scales, wrappers, permutation_seeds


def _restore_intervention(model, saved, saved_scales, wrappers):
    for scale, original in zip(model.global_scales, saved_scales):
        with torch.no_grad():
            scale.copy_(original)
    for relay, mode, old_mode, old_perm in saved:
        relay.assignment_mode = mode
        if old_mode is None:
            if hasattr(relay, "_intervention_mode"):
                del relay._intervention_mode
        else:
            relay._intervention_mode = old_mode
        if old_perm is None:
            if hasattr(relay, "_intervention_permutation"):
                del relay._intervention_permutation
        else:
            relay._intervention_permutation = old_perm
    for relay, original_forward in wrappers:
        relay.forward = original_forward


def _run_checkpoint(dataset: str, seed: int, device: torch.device) -> list[dict]:
    data, model, classifier, checkpoint_path = _load_checkpoint(dataset, seed, device)
    formal_history = OUT_ROOT / "formal" / "B2-FSEP4" / dataset / f"seed_{seed}" / "history.csv"
    with formal_history.open(encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    best_history = max(history, key=lambda row: float(row["val_acc"]))
    baseline_acc, baseline_f1 = _evaluate(data, model, classifier, device)
    expected_acc = float(best_history["val_acc"])
    expected_f1 = float(best_history["val_macro_f1"])
    i0_consistent = abs(baseline_acc - expected_acc) <= 2e-5 and abs(baseline_f1 - expected_f1) <= 2e-5
    if not i0_consistent:
        raise RuntimeError(
            f"I0 mismatch for {dataset}/seed{seed}: evaluator={baseline_acc}/{baseline_f1}, "
            f"history={expected_acc}/{expected_f1}"
        )
    rows = []
    for intervention in INTERVENTIONS:
        state_before = _state_snapshot(model, classifier)
        saved, saved_scales, wrappers, permutation_seeds = _apply_intervention_direct(
            intervention, model, dataset, seed, data.num_nodes, device
        )
        try:
            acc, f1 = _evaluate(data, model, classifier, device)
        finally:
            _restore_intervention(model, saved, saved_scales, wrappers)
        restored = _state_equal(state_before, model, classifier)
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "intervention": intervention,
            "val_acc_pct": 100.0 * acc,
            "val_f1_pct": 100.0 * f1,
            "delta_acc_vs_I0_pp": 100.0 * (acc - baseline_acc),
            "delta_f1_vs_I0_pp": 100.0 * (f1 - baseline_f1),
            "i0_consistent": i0_consistent,
            "state_restored": restored,
            "permutation_seeds": json.dumps(permutation_seeds, separators=(",", ":")),
        })
        if not restored:
            raise RuntimeError(f"intervention failed to restore state: {dataset}/seed{seed}/{intervention}")
    checkpoint_dir = OUT_ROOT / "interventions" / dataset / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
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
    all_rows = []
    for dataset in datasets:
        for seed in seeds:
            print(f"[intervention] {dataset} seed={seed}", flush=True)
            all_rows.extend(_run_checkpoint(dataset, seed, device))
    summary_dir = OUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / "R5_0B_INTERVENTIONS.csv"
    fields = [
        "dataset", "seed", "checkpoint", "intervention", "val_acc_pct", "val_f1_pct",
        "delta_acc_vs_I0_pp", "delta_f1_vs_I0_pp", "i0_consistent", "state_restored",
        "permutation_seeds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[summary] wrote {path} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
