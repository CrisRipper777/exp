"""R2-Design-1 shared layer for checkpoint analysis and summarization.

Same discipline as perf_r1_utils: frozen-model-read-only, no test access.
The mechanism diagnostics are entirely model-internal
(Model.compute_r2_diagnostics), so this layer only resolves configs, loads
checkpoints and provides the frozen A0 reference results.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from hydra import compose, initialize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
TARGET_DATASETS = ["Movies", "Toys", "Grocery"]  # D1-1..D1-4 screen set
GUARD_DATASETS = ["ele-fashion", "Reddit-S"]  # D1-5 step A
SEEDS = [42, 43, 44]
FACTOR_NAMES = ["C", "Pt", "Pv"]

# Variant label -> (hydra model config, output root under outputs/perf_r2d1/).
VARIANTS = ("B0", "F", "S", "J")
VARIANT_YAMLS = {
    "B0": "biaxis_r2_b0",
    "F": "biaxis_r2_f",
    "S": "biaxis_r2_s",
    "J": "biaxis_r2_j",
}
VARIANT_ROOTS = {
    "B0": "b0",
    "F": "functional",
    "S": "semantic",
    "J": "joint",
}
CKPT_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d1"

# Frozen A0 (biaxis_final) formal per-seed reference: datasets x seeds
# 42/43/44, Val accuracy. NEVER retrained for R2 comparisons (plan §25/§31).
A0_REFERENCE_CSV = (
    PROJECT_ROOT / "outputs" / "final_nc_benchmark" / "tables" / "nc_main_per_seed.csv"
)


@dataclass
class R2Setup:
    dataset: str
    seed: int
    variant: str  # "B0" | "F" | "S" | "J"
    cfg: object
    data: object
    model: object
    head: torch.nn.Module
    device: torch.device


def resolve_cfg(dataset: str, seed: int, variant: str) -> object:
    # NOTE: hydra resolves config_path relative to THIS file (src/analysis/).
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        f"model={VARIANT_YAMLS[variant]}",
        f"seed={int(seed)}",
    ]
    with initialize(config_path="../../configs", version_base=None):
        return compose(config_name="config", overrides=overrides)


def load_r2_setup(
    dataset: str, seed: int, variant: str, device: torch.device, root: Path | None = None
) -> R2Setup:
    """Load an R2 checkpoint model + head + data. NEVER reads test labels.

    root = per-variant base directory (default outputs/perf_r2d1/<variant_root>);
    the checkpoint lives at root/<dataset>/<variant>/seed_<seed>/model.pt."""
    from src.data import load_mag_data
    from src.models.biaxis_r2 import Model

    cfg = resolve_cfg(dataset, seed, variant)
    data = load_mag_data(cfg, "nc", int(seed))
    base = Path(root) if root is not None else CKPT_ROOT / VARIANT_ROOTS[variant]
    ckpt_path = base / dataset / variant / f"seed_{seed}" / "model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Model(cfg, ckpt["data_info"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(ckpt["head_state"])
    return R2Setup(dataset, seed, variant, cfg, data, model, head, device)


def assert_no_test_access(data: object) -> None:
    """Guard: R2 scripts must only use train/val supervision."""
    assert data.train_idx is not None and data.val_idx is not None


def load_a0_reference() -> dict[tuple[str, int], float]:
    """Frozen A0 (biaxis_final) per-(dataset, seed) Val accuracy."""
    reference: dict[tuple[str, int], float] = {}
    with A0_REFERENCE_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] != "biaxis_final":
                continue
            reference[(row["dataset"], int(row["seed"]))] = float(row["val_acc"])
    return reference


def a0_val_acc(reference: dict[tuple[str, int], float], dataset: str, seed: int) -> float:
    key = (dataset, seed)
    if key not in reference:
        raise KeyError(f"A0 reference missing for {key}")
    return reference[key]
