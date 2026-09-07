"""R2D29 shared utilities
(docs/BiAxis_R2D29_System_Level_Performance_Advancement_Plan.md).

Shared across G0-G6 scripts: root paths, hydra cfg resolution for
model=biaxis_cort, model construction, the G2 factorial cell tables, and
run-log parsing helpers. Val-only discipline throughout: nothing here reads
Test metrics for selection.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
R2D29_ROOT = PROJECT_ROOT / "outputs" / "r2d29"
G0_ROOT = R2D29_ROOT / "g0_reference"
G1_ROOT = R2D29_ROOT / "g1_audit"
G2_ROOT = R2D29_ROOT / "g2_synergy"

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
LARGE_DATASETS = {"ele-fashion"}

# ---------------------------------------------------------------------------
# Config resolution / model construction
# ---------------------------------------------------------------------------


def resolve_cort_cfg(dataset: str, seed: int, overrides: dict | None = None) -> object:
    from hydra import compose, initialize_config_dir

    ov = [f"dataset={dataset}", "task=nc", "model=biaxis_cort", f"seed={int(seed)}"]
    for k, v in (overrides or {}).items():
        if isinstance(v, bool):
            ov.append(f"model.cort.{k}={'true' if v else 'false'}")
        else:
            ov.append(f"model.cort.{k}={v}")
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"),
                               version_base=None):
        return compose(config_name="config", overrides=ov)


def build_cort_model(cfg: object, data, device):
    from src.models.biaxis_cort import Model

    info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    return Model(cfg, info).to(device)


def load_r2d29_data(dataset: str, seed: int, device):
    """Load the MAGData for the dataset/seed (no model involved). The cfg is
    a biaxis_cort compose so build_cort_model can consume it directly."""
    from src.data import load_mag_data

    cfg = resolve_cort_cfg(dataset, seed)
    data = load_mag_data(cfg, "nc", int(seed))
    return cfg, data


# ---------------------------------------------------------------------------
# G2 full-system synergy matrix (plan §7.2)
# ---------------------------------------------------------------------------

# 2x2x2x2 factorial: R router, S source, W writeback, F fusion
# (fixed: backbone_mode=a0_augment, num_blocks=1)
G2_FIXED = {"backbone_mode": "a0_augment", "num_blocks": 1}

G2_CELLS: dict[str, dict] = {}
for _r, _rv in (("R0", "uniform"), ("R1", "pair_null")):
    for _s, _sv in (("S0", "mean"), ("S1", "preserve_concat")):
        for _w, _wv in (("W0", "late"), ("W1", "factor")):
            for _f, _fv in (("F0", "legacy"), ("F1", "oif")):
                G2_CELLS[f"{_r}{_s}{_w}{_f}"] = {
                    "router_mode": _rv,
                    "source_mode": _sv,
                    "writeback_mode": _wv,
                    "fusion_mode": _fv,
                }

# Matched control (plan §7.5): the mean message duplicated into the 3
# channels and run through the exact preserve_concat path. Only for the
# source-preserving top variants (driver selects which cells).
G2_MATCHED_CONTROLS: dict[str, dict] = {
    f"{cell}+MEAN_DUP": {**G2_CELLS[cell], "mean_dup": True}
    for cell in (
        "R0S1W0F0", "R0S1W1F0", "R0S1W0F1", "R0S1W1F1",
        "R1S1W0F0", "R1S1W1F0", "R1S1W0F1", "R1S1W1F1",
    )
}


# ---------------------------------------------------------------------------
# Run-log parsing
# ---------------------------------------------------------------------------

def parse_train_log(log_path: Path) -> dict:
    """best val acc / val f1 at best epoch / params from a train.log."""
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    params = None
    match = re.search(r"model\+head params=(\d+)", text)
    if match:
        params = int(match.group(1))
    current_epoch = None
    best_acc, best_f1, best_epoch = -1.0, None, None
    for line in text.splitlines():
        em = re.search(r"Epoch (\d+)", line)
        if em:
            current_epoch = int(em.group(1))
            continue
        vm = re.search(r"Val Acc ([\d.]+) \| Val F1 ([\d.]+)", line)
        if vm and current_epoch is not None:
            acc = float(vm.group(1))
            if acc > best_acc:
                best_acc, best_f1, best_epoch = acc, float(vm.group(2)), current_epoch
    return {"val_acc": best_acc if best_epoch is not None else None,
            "val_f1": best_f1, "best_epoch": best_epoch, "params": params}


def load_results_json(path: Path) -> dict:
    """results.json -> {'val_acc', 'test_acc', 'test_f1'} in percent."""
    res = json.loads(path.read_text(encoding="utf-8"))
    out = {"val_acc": res["val_acc"]["mean"] * 100.0}
    if "test_acc" in res:
        out["test_acc"] = res["test_acc"]["mean"] * 100.0
        out["test_f1"] = res["test_macro_f1"]["mean"] * 100.0
    return out


def _f(value, nd: int = 4) -> str:
    return f"{value:.{nd}f}" if value is not None else ""
