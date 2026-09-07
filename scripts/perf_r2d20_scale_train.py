"""R2-Design-2.0 frozen scale trainer (plan §11-§23): M1 / M2 screens,
guards and formal confirmation on the FROZEN B0 parent.

Per (dataset, seed): load the B0 best checkpoint (b0_confirm) into the
biaxis_r2_scale model, freeze every parent parameter, train ONLY the scale
parameters (alpha for M1 / hop logits for M2; wd=0) + a fresh classifier
(shared exact init; wd=1e-4). AdamW lr=1e-3, 300 epochs, patience 30, best
Val Acc. HEAD = the same frozen B0 with the scale params fixed at init
(alpha=0 for M1 => exact B0 output) + the same classifier init.

Saves per run: summary.json, scale trajectory CSV (per-epoch coefficients +
val acc), smoothing diagnostics (sim(H0,H1)/sim(H0,H2)/rel gap), per-class
F1. Val only, never test.

Usage:
    python scripts/perf_r2d20_scale_train.py --mode m1 --phase screen \
        --gpus 0,1                                  # Movies/Toys/Grocery seed42
    python scripts/perf_r2d20_scale_train.py --mode m1 --phase guards --gpus 0,1
    python scripts/perf_r2d20_scale_train.py --mode m1 --phase confirm \
        --seeds 43,44 --gpus 0,1
    python scripts/perf_r2d20_scale_train.py --mode m2 --phase screen --gpus 0,1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d16_utils import R2D16_ROOT, TARGET_DATASETS  # noqa: E402

B0_CONFIRM = PROJECT_ROOT / "outputs" / "perf_r2d15" / "b0_confirm"
R2D20_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d20"
GUARD_DATASETS = ["ele-fashion", "Reddit-S"]
CLASSIFIER_SEED = 20260904
PHASES = {
    "screen": (TARGET_DATASETS, [42]),
    "guards": (GUARD_DATASETS, [42]),
    "confirm": (TARGET_DATASETS, [42, 43, 44]),
}


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


def _run_one(mode: str, dataset: str, seed: int, variant: str, gpu: int,
             force: bool, epochs: int | None, out_root: Path) -> None:
    outdir = out_root / dataset / variant / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"[{gpu}] {mode} {dataset} {variant} s{seed}"
    if (outdir / "summary.json").exists() and not force:
        print(f"{tag} SKIP", flush=True)
        return
    code = f"""
import csv, json, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.biaxis_r2_scale import Model as ScaleModel, load_b0_checkpoint_into
from src.analysis.perf_r2d16_utils import make_classifier_init, save_state, load_state_into
from src.analysis.perf_r2d15_utils import val_metrics_with_head
from hydra import compose, initialize_config_dir

device = torch.device("cuda:0")
mode, dataset, seed, variant = "{mode}", "{dataset}", {seed}, "{variant}"
epochs_override = {epochs if epochs is not None else 'None'}
outdir = Path(r"{outdir}")

with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
    cfg = compose(config_name="config", overrides=[
        f"dataset={{dataset}}", "task=nc", "model=biaxis_r2_scale_{mode}",
        f"seed={{seed}}",
    ])
from src.data import load_mag_data
data = load_mag_data(cfg, "nc", int(seed))
info = {{
    "input_dim": data.input_dim, "num_nodes": data.num_nodes,
    "num_classes": data.num_classes,
    "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
}}
model = ScaleModel(cfg, info).to(device).eval()
ckpt = str(Path(r"{B0_CONFIRM}") / dataset / "B0" / f"seed_{{seed}}" / "model.pt")
report = load_b0_checkpoint_into(model, ckpt)
scale_params = [p for name, p in model.named_parameters() if name.startswith("mixer.")]
for name, p in model.named_parameters():
    if not name.startswith("mixer."):
        p.requires_grad_(False)
if variant == "HEAD":
    for p in scale_params:
        p.requires_grad_(False)  # alpha fixed at init (0) => exact B0 output

head_init_path = outdir.parent.parent / "head_init.pt"
if not head_init_path.exists():
    torch.manual_seed({CLASSIFIER_SEED})
    init_head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    save_state(head_init_path, init_head)
head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
load_state_into(head_init_path, head)

opt = torch.optim.AdamW([
    {{"params": [p for p in scale_params if p.requires_grad], "lr": 1e-3,
      "weight_decay": 0.0}},
    {{"params": head.parameters(), "lr": 1e-3, "weight_decay": 1e-4}},
])
criterion = torch.nn.CrossEntropyLoss()
x = data.x.to(device)
ei = data.edge_index.to(device)
train_idx = data.train_idx.to(device)
y_train = data.y[data.train_idx].to(device)
val_idx = data.val_idx.to(device)
y_val = data.y[data.val_idx].to(device)

# smoothing diagnostics (parent frozen => constant; computed once, plan §30)
with torch.no_grad():
    smoothing = model.compute_scale_diagnostics(x, ei)["smoothing"]

def val_metrics():
    head.eval()
    with torch.no_grad():
        pred = head(model(x, ei)[0][val_idx]).argmax(dim=-1)
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    y = y_val.cpu().numpy(); p = pred.cpu().numpy()
    return float(accuracy_score(y, p)), float(f1_score(y, p, average="macro", zero_division=0))

trajectory = []
best_acc, best_f1, best_state = -1.0, None, None
patience_left = 30
total_epochs = 300 if epochs_override is None else epochs_override
t0 = time.monotonic()
stop_epoch = total_epochs
best_epoch = None
for epoch in range(1, total_epochs + 1):
    head.train()
    model.eval()  # parent stays eval (dropout off); mixer has no dropout
    opt.zero_grad(set_to_none=True)
    z, _, _, _, _ = model(x, ei)
    loss = criterion(head(z[train_idx]), y_train)
    loss.backward()
    opt.step()
    acc, f1 = val_metrics()
    scale = model.mixer.scale_diagnostics()
    row = {{"epoch": epoch, "val_acc": acc}}
    if mode == "m1":
        for i, v in enumerate(scale["alpha"]):
            row[f"alpha_{{i}}"] = v
    else:
        for i, g in enumerate(scale["gamma"]):
            for k in range(3):
                row[f"gamma_{{i}}_{{k}}"] = g[k]
    trajectory.append(row)
    if acc > best_acc:
        best_acc, best_epoch, best_f1 = acc, epoch, f1
        best_state = {{
            "head": {{k: v.detach().clone() for k, v in head.state_dict().items()}},
            "scale": {{k: v.detach().clone() for k, v in
                       {{n: p for n, p in model.named_parameters() if n.startswith("mixer.")}}.items()}},
        }}
        patience_left = 30
    else:
        patience_left -= 1
        if patience_left <= 0:
            stop_epoch = epoch
            break
runtime_sec = time.monotonic() - t0
head.load_state_dict(best_state["head"])
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.startswith("mixer.") and name in best_state["scale"]:
            p.copy_(best_state["scale"][name])
head.eval()
with torch.no_grad():
    z_best = model(x, ei)[0]
m_full = val_metrics_with_head(head, z_best, data, device)
scale_best = model.mixer.scale_diagnostics()
instability = False
if mode == "m1" and max(abs(v) for v in scale_best["alpha"]) > 2.0:
    instability = True
summary = {{
    "mode": mode, "dataset": dataset, "variant": variant, "seed": seed,
    "best_val_acc": best_acc, "best_val_macro_f1": best_f1,
    "best_epoch": best_epoch, "stop_epoch": stop_epoch,
    "runtime_sec": round(runtime_sec, 1),
    "scale": scale_best, "smoothing": smoothing,
    "per_class_f1": m_full["per_class_f1"],
    "confusion": m_full["confusion"],
    "alpha_instability_warning": instability,
    "missing_scale_keys_on_load": report["missing_scale_keys"],
    "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    "adapter_params": sum(p.numel() for p in scale_params),
}}
with (outdir / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)
with (outdir / "scale_trajectory.csv").open("w", newline="") as f:
    if trajectory:
        writer = csv.DictWriter(f, fieldnames=list(trajectory[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory)
print(f"[run] {{mode}} {{dataset}} {{variant}} s{{seed}} best_acc={{best_acc:.5f}} "
      f"f1={{best_f1:.5f}} ep={{best_epoch}}/{{stop_epoch}}", flush=True)
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = outdir / "run.log"
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"{tag} FAILED rc={proc.returncode}", flush=True)
        print(log.read_text(encoding="utf-8")[-3000:], flush=True)
        return
    print(f"{tag} OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-Design-2.0 frozen scale trainer")
    parser.add_argument("--mode", default="m1", choices=["m1", "m2"])
    parser.add_argument("--phase", default="screen", choices=["screen", "guards", "confirm"])
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    args = parser.parse_args()
    datasets, seeds = PHASES[args.phase]
    if args.datasets:
        datasets = [d for d in args.datasets.split(",")]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    variants = ["HEAD", "M1"] if args.mode == "m1" else ["HEAD", "M2"]
    gpus = [int(g) for g in args.gpus.split(",")]
    out_root = R2D20_ROOT / (f"{args.mode}_screen" if args.phase == "screen" else
                             (f"{args.mode}_guards" if args.phase == "guards" else
                              f"{args.mode}_confirm"))
    jobs = [(d, s, v) for d in datasets for s in seeds for v in variants]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] mode={args.mode} phase={args.phase} jobs={len(jobs)} "
          f"gpus={gpus} out={out_root.relative_to(PROJECT_ROOT)}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (d, s, v) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(
                _run_one, args.mode, d, s, v, gpu, args.force, args.epochs, out_root
            )] = (d, s, v)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
