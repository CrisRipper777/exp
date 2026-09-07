"""R2-D2.0.5-B: controlled warm-start adaptation for M1 (user-directed).

Same M1 architecture, matched initialization (same B0 checkpoint, alpha
init = 0, same classifier init seed 20260904):

    S0 FROZEN           : scale + classifier only (reuses the M1 screen runs)
    S1 GRAPH-UNFREEZE   : ep 1-30 frozen; ep 31+ unfreeze source_transforms
                          + msg_norm_base + raw_rho_base (P0 & fusion frozen)
    S2 GRAPH+FUSION     : ep 1-30 frozen; ep 31+ additionally unfreeze fusion
                          (P0 always frozen)

AdamW: alpha wd=0 lr1e-3; classifier wd1e-4 lr1e-3; unfrozen parent groups
lr1e-3 wd1e-4 (added to the existing optimizer at epoch 31 via
add_param_group, preserving Adam state of the warm groups). 300ep/patience30,
best Val Acc. Grad-norm + parameter-drift sampling at ep 1/10/30/31/best.
Val only.

Verdict (user rule): max(S1,S2) - HEAD >= +0.30pp & >=2/3 positive & F1
safe -> scale route REOPENS (formal confirm next); if still < +0.15~0.20pp
-> scale route exits the mainline.

Usage:
    python scripts/perf_r2d205_schedule.py --gpus 0,1
"""

from __future__ import annotations

import argparse
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

from src.analysis.perf_r2d16_utils import R2D16_ROOT  # noqa: E402

B0_CONFIRM = PROJECT_ROOT / "outputs" / "perf_r2d15" / "b0_confirm"
R2D205_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d20_5"
SCHEDULE_ROOT = R2D205_ROOT / "schedule"
TARGET_DATASETS = ["Movies", "Toys", "Grocery"]
CLASSIFIER_SEED = 20260904
SCHEDULES = ("S1", "S2")
SAMPLE_EPOCHS = (1, 10, 30, 31)
# groups for gradient sampling / drift (plan D2.0 §28)
GROUPS = ("source_transforms", "msg_norm_base", "fusion")


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


def _run_one(dataset: str, schedule: str, gpu: int, force: bool) -> None:
    outdir = SCHEDULE_ROOT / dataset / schedule
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"[{gpu}] {dataset} {schedule}"
    if (outdir / "summary.json").exists() and not force:
        print(f"{tag} SKIP", flush=True)
        return
    code = f"""
import json, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.biaxis_r2_scale import Model as ScaleModel, load_b0_checkpoint_into
from src.analysis.perf_r2d16_utils import make_classifier_init, save_state, load_state_into
from src.analysis.perf_r2d15_utils import val_metrics_with_head
from hydra import compose, initialize_config_dir

device = torch.device("cuda:0")
dataset, schedule = "{dataset}", "{schedule}"
outdir = Path(r"{outdir}")

with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
    cfg = compose(config_name="config", overrides=[
        f"dataset={{dataset}}", "task=nc", "model=biaxis_r2_scale_m1", "seed=42",
    ])
from src.data import load_mag_data
data = load_mag_data(cfg, "nc", 42)
info = {{
    "input_dim": data.input_dim, "num_nodes": data.num_nodes,
    "num_classes": data.num_classes,
    "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
}}
model = ScaleModel(cfg, info).to(device).eval()
ckpt = str(Path(r"{B0_CONFIRM}") / dataset / "B0" / "seed_42" / "model.pt")
load_b0_checkpoint_into(model, ckpt)

scale_names = [n for n, _ in model.named_parameters() if n.startswith("mixer.")]
unfreeze_names = ({{"source_transforms", "msg_norm_base", "raw_rho_base"}}
                  if schedule == "S1"
                  else {{"source_transforms", "msg_norm_base", "raw_rho_base", "fusion"}})
for n, p in model.named_parameters():
    p.requires_grad_(n in scale_names)  # frozen parent at t=0

head_init_path = outdir.parent / "head_init.pt"
if not head_init_path.exists():
    torch.manual_seed({CLASSIFIER_SEED})
    init_head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    save_state(head_init_path, init_head)
head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
load_state_into(head_init_path, head)

opt = torch.optim.AdamW([
    {{"params": [p for n, p in model.named_parameters() if n in scale_names],
      "lr": 1e-3, "weight_decay": 0.0}},
    {{"params": head.parameters(), "lr": 1e-3, "weight_decay": 1e-4}},
])
criterion = torch.nn.CrossEntropyLoss()
x = data.x.to(device)
ei = data.edge_index.to(device)
train_idx = data.train_idx.to(device)
y_train = data.y[data.train_idx].to(device)
val_idx = data.val_idx.to(device)
y_val = data.y[data.val_idx].to(device)

theta0 = {{k: v.detach().clone() for k, v in model.state_dict().items()}}
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

def group_norms():
    out = {{}}
    for g in ({GROUPS}):
        sq = 0.0
        for n, p in model.named_parameters():
            if n.startswith(g + ".") and p.grad is not None:
                sq += float(p.grad.square().sum().item())
        out[g + "_grad_norm"] = sq ** 0.5
    return out

def group_drift():
    out = {{}}
    for g in ({GROUPS}):
        ref = sum(float(theta0[k].square().sum().item())
                  for k in theta0 if k.startswith(g + "."))
        diff = sum(float((model.state_dict()[k] - theta0[k]).square().sum().item())
                   for k in theta0 if k.startswith(g + "."))
        out[g + "_drift"] = (diff ** 0.5) / (ref ** 0.5 + 1e-12)
    return out

trajectory = []
grad_samples = []
best_acc, best_f1, best_state = -1.0, None, None
patience_left = 30
unfrozen = False
t0 = time.monotonic()
stop_epoch = 300
best_epoch = None
for epoch in range(1, 301):
    head.train()
    model.eval()
    if schedule != "S0" and epoch == 31:
        for n, p in model.named_parameters():
            if n == "raw_rho_base" or any(n.startswith(g + ".") for g in unfreeze_names):
                p.requires_grad_(True)
        opt.add_param_group({{"params": [p for n, p in model.named_parameters()
                                         if n == "raw_rho_base"
                                         or any(n.startswith(g + ".") for g in unfreeze_names)],
                              "lr": 1e-3, "weight_decay": 1e-4}})
        unfrozen = True
    opt.zero_grad(set_to_none=True)
    z, _, _, _, _ = model(x, ei)
    loss = criterion(head(z[train_idx]), y_train)
    loss.backward()
    if epoch in ({SAMPLE_EPOCHS}):
        grad_samples.append({{"epoch": epoch, "schedule": schedule, "dataset": dataset,
                              **group_norms(), **group_drift()}})
    opt.step()
    acc, f1 = val_metrics()
    a = model.mixer.alpha.detach().cpu().tolist()
    trajectory.append({{"epoch": epoch, "val_acc": acc,
                        "alpha_0": a[0], "alpha_1": a[1], "alpha_2": a[2],
                        "unfrozen": int(unfrozen)}})
    if acc > best_acc:
        best_acc, best_epoch, best_f1 = acc, epoch, f1
        best_state = {{
            "head": {{k: v.detach().clone() for k, v in head.state_dict().items()}},
            "model": {{k: v.detach().clone() for k, v in model.state_dict().items()}},
        }}
        patience_left = 30
    else:
        patience_left -= 1
        if patience_left <= 0:
            stop_epoch = epoch
            break
runtime_sec = time.monotonic() - t0
head.load_state_dict(best_state["head"])
model.load_state_dict(best_state["model"])
head.eval()
model.eval()
# best-checkpoint sample
z_best = model(x, ei)[0]
with torch.enable_grad():
    z_g, _, _, _, _ = model(x, ei)
    loss_g = criterion(head(z_g[train_idx]), y_train)
    loss_g.backward()
grad_samples.append({{"epoch": "best", "schedule": schedule, "dataset": dataset,
                      **group_norms(), **group_drift()}})
with torch.no_grad():
    z_best = model(x, ei)[0]
m_full = val_metrics_with_head(head, z_best, data, device)
summary = {{
    "dataset": dataset, "schedule": schedule, "seed": 42,
    "best_val_acc": best_acc, "best_val_macro_f1": best_f1,
    "best_epoch": best_epoch, "stop_epoch": stop_epoch,
    "runtime_sec": round(runtime_sec, 1),
    "scale": {{"mode": "m1", "alpha": model.mixer.alpha.detach().cpu().tolist()}},
    "smoothing": smoothing, "per_class_f1": m_full["per_class_f1"],
    "confusion": m_full["confusion"],
    "unfrozen_at_31": unfrozen,
    "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
}}
with (outdir / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)
with (outdir / "grad_samples.json").open("w") as f:
    json.dump(grad_samples, f, indent=2)
with (outdir / "trajectory.json").open("w") as f:
    json.dump(trajectory, f)
print(f"[run] {{dataset}} {{schedule}} best_acc={{best_acc:.5f}} "
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
    parser = argparse.ArgumentParser(description="D2.0.5-B warm-start schedule study")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = TARGET_DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, s) for d in datasets for s in SCHEDULES]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out=outputs/perf_r2d20_5/schedule", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (d, s) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(_run_one, d, s, gpu, args.force)] = (d, s)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
