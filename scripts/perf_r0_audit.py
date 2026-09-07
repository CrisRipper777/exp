"""R0-D0: repository / checkpoint audit (plan §3/§31 Prompt 1).

Verifies the 15 P3 OFR checkpoints (5 datasets x seeds 42/43/44) are a
sound diagnostic base: existence, field completeness, frozen-config match,
model/head loading, and Val-Acc reproduction from the saved head_state
(no training, no test access). Outputs checkpoint_audit.csv + R0_AUDIT.md.

Usage:
    python scripts/perf_r0_audit.py --gpus 0,1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CKPT_ROOT = PROJECT_ROOT / "outputs" / "p3" / "operator"
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "perf_r0" / "audit"
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]


def _saved_config(dataset: str, seed: int) -> dict | None:
    path = CKPT_ROOT / dataset / "OFR" / f"seed_{seed}" / "hydra" / ".hydra" / "config.yaml"
    if not path.exists():
        return None
    import yaml

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _checkpoint_recorded_val(dataset: str, seed: int) -> float | None:
    summary = CKPT_ROOT / dataset / "OFR" / f"seed_{seed}" / "summary.json"
    if summary.exists():
        with summary.open(encoding="utf-8") as f:
            s = json.load(f)
        results = s.get("results") or {}
        entry = results.get("val_acc")
        if isinstance(entry, dict):
            return float(entry["mean"])
    results_json = CKPT_ROOT / dataset / "OFR" / f"seed_{seed}" / "hydra" / "results.json"
    if results_json.exists():
        with results_json.open(encoding="utf-8") as f:
            r = json.load(f)
        entry = r.get("val_acc")
        if isinstance(entry, dict):
            return float(entry["mean"])
    return None


def _recompute_val(dataset: str, seed: int, device: torch.device) -> float | None:
    """Load model+head from checkpoint, rerun current forward on the full
    graph, evaluate val with the saved head. No test access."""
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}", "task=nc", "model=biaxis_p3",
                "model.p3.operator_mode=full_interaction", f"seed={int(seed)}",
            ],
        )
    from src.data import load_mag_data
    from src.models.biaxis_p3 import Model

    data = load_mag_data(cfg, "nc", int(seed))
    ckpt = torch.load(
        CKPT_ROOT / dataset / "OFR" / f"seed_{seed}" / "model.pt",
        map_location="cpu", weights_only=False,
    )
    model = Model(cfg, ckpt["data_info"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(ckpt["head_state"])

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    model.eval()
    with torch.no_grad():
        z, _, _, _, _ = model(x, edge_index)
    logits = head(z)
    pred = logits.argmax(dim=-1)
    val_idx = data.val_idx.to(device)
    y = data.y.to(device)
    acc = float((pred[val_idx] == y[val_idx]).float().mean().item())
    del z
    torch.cuda.empty_cache()
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="R0 checkpoint audit")
    parser.add_argument("--gpus", default="0")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    device = torch.device(f"cuda:{gpus[0]}")

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            row = {"dataset": dataset, "seed": seed}
            ckpt_path = CKPT_ROOT / dataset / "OFR" / f"seed_{seed}" / "model.pt"
            row["checkpoint_exists"] = bool(ckpt_path.exists())
            if not ckpt_path.exists():
                rows.append(row)
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            row["fields_ok"] = all(k in ckpt for k in ("model_state", "head_state", "data_info", "seed", "task"))
            row["task"] = ckpt.get("task")
            row["ckpt_seed"] = ckpt.get("seed")
            saved = _saved_config(dataset, seed)
            if saved is not None:
                p2, p3, p1 = saved["model"]["p2"], saved["model"]["p3"], saved["model"]["p1"]
                row["config_match"] = (
                    p2.get("mode") == "null_softmax"
                    and p2.get("deterministic") is False
                    and p3.get("operator_mode") == "full_interaction"
                    and int(p1.get("num_relations")) == 4
                    and int(saved["model"].get("factor_dim")) == 128
                    and saved["task"].get("epochs") == 300
                    and saved["task"].get("patience") == 30
                )
                row["saved_p2_mode"] = p2.get("mode")
                row["saved_p2_deterministic"] = p2.get("deterministic")
                row["saved_operator"] = p3.get("operator_mode")
            else:
                row["config_match"] = False
                row["saved_p2_mode"] = None
                row["saved_p2_deterministic"] = None
                row["saved_operator"] = None
            row["model_load_ok"] = False
            row["head_load_ok"] = False
            row["val_reproduce_ok"] = False
            row["checkpoint_val_acc"] = _checkpoint_recorded_val(dataset, seed)
            row["recomputed_val_acc"] = None
            row["absolute_diff"] = None
            try:
                repro = _recompute_val(dataset, seed, device)
                row["recomputed_val_acc"] = repro
                row["model_load_ok"] = True
                row["head_load_ok"] = True
                recorded = row["checkpoint_val_acc"]
                if recorded is not None and repro is not None:
                    row["absolute_diff"] = abs(recorded - repro)
                    # same-weights re-eval: only float32 aggregation noise
                    # expected (~1e-6); 0.1pp tolerance is conservative.
                    row["val_reproduce_ok"] = row["absolute_diff"] < 0.001
            except Exception as exc:  # noqa: BLE001
                row["load_error"] = str(exc)[:200]
            rows.append(row)
            print(
                f"[audit] {dataset:12s} s{seed} exists={row['checkpoint_exists']} "
                f"fields={row.get('fields_ok')} cfg={row.get('config_match')} "
                f"repro={row.get('val_reproduce_ok')} diff={row.get('absolute_diff')}",
                flush=True,
            )

    with (AUDIT_ROOT / "checkpoint_audit.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines: list[str] = []
    lines.append("# R0-AUDIT — Repository / Checkpoint Audit")
    lines.append("")
    lines.append("> 计划 §3/§31。15 OFR checkpoints 审计：存在性、字段、配置匹配、加载、Val 复现。")
    lines.append("")
    lines.append("| dataset | seed | exists | fields | cfg | load | repro | recorded | recomputed | diff |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['checkpoint_exists']} | {row.get('fields_ok')} | "
            f"{row.get('config_match')} | {row.get('model_load_ok')} | {row.get('val_reproduce_ok')} | "
            f"{row.get('checkpoint_val_acc')} | {row.get('recomputed_val_acc')} | {row.get('absolute_diff')} |"
        )
    lines.append("")
    ok = all(
        row.get("checkpoint_exists") and row.get("fields_ok") and row.get("config_match")
        and row.get("model_load_ok") and row.get("val_reproduce_ok")
        for row in rows
    )
    lines.append(f"**Audit verdict: {'PASS' if ok else 'FAIL'}**")
    lines.append("")
    (AUDIT_ROOT / "R0_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] done -> {AUDIT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
