"""R2-D2.5-D: optimization-accessibility interventions
(docs/BiAxis_R2_Design_2_5_Structured_Capacity_Utilization_Audit.md).

Only for at most two top candidates chosen from D2.5-A/D2.5-C
("representation evidence positive but training usage weak"). SINGLE-
FACTOR interventions, never stacked:

    D1 expert-specific LR  : H2-expert LR in {1e-3, 5e-3, 1e-2} (all 3 seeds)
    D2 deep supervision    : lambda 0 vs 0.1 (aux heads on hop experts,
                             removed at inference)
    D3 path/expert dropout : drop the H1 expert with p=0.2 during training;
                             all experts used at inference

Recorded per run: per-group grad norm / update ratio, per-expert output
norm, classifier sensitivity (||dCE/de_f||), H2-off ablation, train/val CE,
Acc/F1. Verdicts: Gradient Starvation SUPPORTED/NOT SUPPORTED,
Objective mismatch SUPPORTED/NOT SUPPORTED (summarizer).

Outputs:
    outputs/perf_r2d25/optimization/<dataset>/<variant>/<intervention>/seed_<s>/
    (aggregation: scripts/summarize_perf_r2d25.py --stage optimization)

Usage:
    python scripts/perf_r2d25_optimization.py --variants sep_concat \
        --interventions expert_lr --gpus 0,1
    python scripts/perf_r2d25_optimization.py --variants sep_concat \
        --interventions deep_sup --datasets Movies --seeds 42 --epochs 5  # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d25_utils import (  # noqa: E402
    R2D25_ROOT,
    TARGET_DATASETS,
)

OPTIMIZATION_ROOT = R2D25_ROOT / "optimization"
HEAD_INIT_ROOT = OPTIMIZATION_ROOT / "head_init"

# intervention -> list of configurations (each a single-factor setting).
# The BASE setting of every intervention (expert LR 1e-3 / lambda 0 / p 0)
# IS the D2.5-C capacity run for the same (dataset, variant, seed) — the
# summarizer reuses those rows; the driver only runs NEW settings.
EXPERT_LR_VALUES = (5e-3, 1e-2)
DEEP_SUP_LAMBDAS = (0.1,)
PATH_DROPOUT_VALUES = (0.2,)


class _Semaphore:
    def __init__(self, value: int) -> None:
        self._cond = threading.Condition()
        self._value = int(value)

    def acquire(self) -> None:
        with self._cond:
            while self._value < 1:
                self._cond.wait()
            self._value -= 1

    def release(self) -> None:
        with self._cond:
            self._value += 1
            self._cond.notify_all()


def _expert_output_norms(experts: dict[str, object]) -> dict[str, float]:
    """Per-expert mean output norm (eval-mode forward)."""
    out = {}
    for key, e in experts.items():
        out[key] = float(e.norm(dim=-1).mean().item())
    return out


def _expert_usage_stats(model, head, x, ei, train_idx, y_train) -> tuple[dict, dict]:
    """One CE backward on the FULL z at the best checkpoint:
      - classifier sensitivity per expert = ||dCE/de|| per factor (autograd);
      - per-expert PARAM grad norm + update ratio (from .grad)."""
    import torch.nn.functional as F

    model.eval()
    with torch.enable_grad():
        z, experts = model.forward_with_experts(x, ei)
        loss = F.cross_entropy(head(z[train_idx]), y_train)
        sens: dict[str, list[float]] = {}
        for key, e in experts.items():  # e: [N, 3, d]
            grad = torch.autograd.grad(loss, e, retain_graph=True, allow_unused=True)[0]
            sens[key] = [float(grad[:, f].norm().item()) for f in range(e.size(1))]
        loss.backward()
    param_stats: dict[str, dict[str, float]] = {}
    for key in experts:
        params = [p for n, p in model.named_parameters()
                  if n.startswith(f"hop_experts.{key}.") and p.grad is not None]
        gn = sum(p.grad.square().sum().item() for p in params) ** 0.5
        ur = []
        for p in params:
            w = p.detach().norm().item()
            if w > 0:
                ur.append(1e-3 * p.grad.norm().item() / w)
        param_stats[key] = {
            "param_grad_norm": gn,
            "update_ratio": float(sum(ur) / len(ur)) if ur else 0.0,
        }
    sens_out = {k: float(sum(v) / len(v)) for k, v in sens.items()}
    sens_per_factor = {k: v for k, v in sens.items()}
    return sens_out, {"param_stats": param_stats, "sens_per_factor": sens_per_factor}


def run_worker(dataset: str, variant: str, intervention: str, setting: str,
               seed: int, outdir: Path, epochs: int | None, force: bool) -> None:
    import torch

    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} {variant} {intervention}/{setting} s{seed}] SKIP", flush=True)
        return
    from src.analysis.perf_r2d25_utils import (
        ablation_metrics, load_mag_data_wrap, load_or_make_head_init,
        resolve_capacity_cfg, train_capacity_variant,
    )
    from src.models.biaxis_r2_capacity import EXPERT_KEYS, Model

    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    cfg = resolve_capacity_cfg(dataset, seed, variant)
    expert_keys = EXPERT_KEYS[variant]
    assert expert_keys, f"variant={variant} has no hop experts; D2.5-D not applicable"

    expert_lr_group = None
    expert_lr = 1e-3
    deep_sup_lambda = 0.0
    path_dropout_p = 0.0
    if intervention == "expert_lr":
        expert_lr = float(setting)
        expert_lr_group = "hop_experts.e2"
        if "e2" not in expert_keys:
            expert_lr_group = f"hop_experts.{expert_keys[-1]}"
    elif intervention == "deep_sup":
        deep_sup_lambda = float(setting)
        cfg.model.deep_supervision.enabled = True
        # NOTE: "lambda" is a reserved word — dict-style access required.
        cfg.model.deep_supervision["lambda"] = deep_sup_lambda
    elif intervention == "path_dropout":
        path_dropout_p = float(setting)
    else:
        raise ValueError(f"unknown intervention {intervention!r}")

    data = load_mag_data_wrap(cfg, seed)
    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
    }
    t0 = time.monotonic()
    model = Model(cfg, info).to(device)
    head = load_or_make_head_init(
        HEAD_INIT_ROOT / f"{dataset}_seed{seed}.pt",
        model.out_dim, int(data.num_classes), device,
    )
    history_path = outdir / "history.csv"
    history_file = history_path.open("w", encoding="utf-8", newline="")
    history_writer = csv.DictWriter(
        history_file, fieldnames=[
            "epoch", "lr_graph", "lr_p0", "train_ce", "val_acc", "p0_unfrozen",
        ])
    history_writer.writeheader()
    total_epochs = 300 if epochs is None else int(epochs)
    res = train_capacity_variant(
        cfg, data, model, head, device, total_epochs=total_epochs,
        expert_lr_group=expert_lr_group, expert_lr=expert_lr,
        deep_sup_lambda=deep_sup_lambda, path_dropout_p=path_dropout_p,
        history_callback=history_writer.writerow,
    )
    history_file.close()

    x = data.x.to(device)
    ei = data.edge_index.to(device)
    model.eval()
    with torch.no_grad():
        z_full, experts = model.forward_with_experts(x, ei)
        expert_norms = _expert_output_norms(experts)
    head.eval()
    from torch.nn import CrossEntropyLoss
    criterion = CrossEntropyLoss()
    y_train = data.y[data.train_idx].to(device)
    y_val = data.y[data.val_idx].to(device)
    with torch.no_grad():
        train_ce = float(criterion(head(z_full[data.train_idx.to(device)]), y_train).item())
        val_ce = float(criterion(head(z_full[data.val_idx.to(device)]), y_val).item())
    sens, usage = _expert_usage_stats(
        model, head, x, ei, data.train_idx.to(device), y_train)
    abl = ablation_metrics(model, head, x, ei, data, device)
    runtime_sec = time.monotonic() - t0
    summary = {
        "dataset": dataset, "variant": variant, "intervention": intervention,
        "setting": setting, "seed": seed,
        "expert_lr_group": expert_lr_group, "expert_lr": expert_lr,
        "deep_sup_lambda": deep_sup_lambda, "path_dropout_p": path_dropout_p,
        "best_val_acc": res["best_val_acc"],
        "best_val_macro_f1": res["best_val_macro_f1"],
        "per_class_f1": res["per_class_f1"],
        "best_epoch": res["best_epoch"], "stop_epoch": res["stop_epoch"],
        "train_ce_at_best": train_ce, "val_ce_at_best": val_ce,
        "expert_output_norm": expert_norms,
        "classifier_sensitivity": sens,
        "classifier_sensitivity_per_factor": usage["sens_per_factor"],
        "expert_param_stats": usage["param_stats"],
        "ablations": abl,
        "parameter_count": int(model.parameter_count),
        "runtime_sec": round(runtime_sec, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (outdir / "grad_samples.json").open("w", encoding="utf-8") as f:
        json.dump(res["grad_samples"], f, indent=2)
    print(
        f"[run] {dataset} {variant} {intervention}/{setting} s{seed} "
        f"best_acc={res['best_val_acc']:.5f} f1={res['best_val_macro_f1']:.5f} "
        f"ep={res['best_epoch']}/{res['stop_epoch']} ({runtime_sec:.0f}s)", flush=True,
    )


def _run_one(dataset: str, variant: str, intervention: str, setting: str,
             seed: int, gpu: int, force: bool, epochs: int | None) -> None:
    outdir = OPTIMIZATION_ROOT / dataset / variant / intervention / f"setting_{setting}" / f"seed_{seed}"
    tag = f"[{gpu}] {dataset} {variant} {intervention}/{setting} seed={seed}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--dataset", dataset, "--variant", variant,
        "--intervention", intervention, "--setting", str(setting),
        "--seed", str(seed), "--outdir", str(outdir),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(int(epochs))]
    if force:
        cmd += ["--force"]
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "run.log"
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"{tag} FAILED rc={proc.returncode}", flush=True)
        print(log.read_text(encoding="utf-8")[-3000:], flush=True)
        return
    print(f"{tag} OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-D2.5-D optimization interventions")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None, help="comma-separated capacity modes")
    parser.add_argument("--interventions", default=None, help="expert_lr,deep_sup,path_dropout")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--intervention", default=None)
    parser.add_argument("--setting", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.dataset, args.variant, args.intervention, args.setting,
                   args.seed, Path(args.outdir), args.epochs, args.force)
        return

    datasets = TARGET_DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    variants = ["sep_concat"] if not args.variants else [v for v in args.variants.split(",")]
    interventions = ["expert_lr", "deep_sup", "path_dropout"] if not args.interventions \
        else [i for i in args.interventions.split(",")]
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    settings_by_intervention = {
        "expert_lr": [str(v) for v in EXPERT_LR_VALUES],
        "deep_sup": [str(v) for v in DEEP_SUP_LAMBDAS],
        "path_dropout": [str(v) for v in PATH_DROPOUT_VALUES],
    }
    jobs = [
        (d, v, i, s, seed_)
        for d in datasets for v in variants for i in interventions
        for s in settings_by_intervention[i] for seed_ in seeds
    ]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out=outputs/perf_r2d25/optimization", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for idx, (d, v, i, s, seed_) in enumerate(jobs):
            gpu = gpus[idx % len(gpus)]
            futures[executor.submit(_run_one, d, v, i, s, seed_, gpu, args.force, args.epochs)] = \
                (d, v, i, s, seed_)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
