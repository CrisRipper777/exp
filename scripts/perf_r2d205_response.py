"""R2-D2.0.5-A: alpha_Pt fixed response curve (user-directed short control).

For each (dataset, alpha_Pt): load the B0 best checkpoint, build the M1
scale model with alpha = [0, alpha_Pt, 0] FIXED (frozen, not trained), and
train ONLY a fresh classifier (same exact init across all alpha values and
datasets, seed 20260904). 300ep/patience30, best Val Acc, AdamW lr1e-3
wd1e-4. Val only.

Question: is there a fixed scale region with real value that SGD did not
find in the M1 screen? (M1 learned alphas were all |alpha| < 0.07.)

Usage:
    python scripts/perf_r2d205_response.py --gpus 0,1
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
RESPONSE_ROOT = R2D205_ROOT / "response"
ALPHA_PT_VALUES = (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0)
TARGET_DATASETS = ["Movies", "Toys", "Grocery"]
CLASSIFIER_SEED = 20260904


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


def _run_one(dataset: str, alpha_pt: float, gpu: int, force: bool) -> None:
    tag_key = f"{dataset}_a{alpha_pt:g}".replace("-", "m")
    outdir = RESPONSE_ROOT / dataset / tag_key
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"[{gpu}] {dataset} alpha_Pt={alpha_pt:g}"
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
dataset, alpha_pt = "{dataset}", {alpha_pt}
outdir = Path(r"{outdir}")

with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
    cfg = compose(config_name="config", overrides=[
        f"dataset={{dataset}}", "task=nc", "model=biaxis_r2_scale_m1",
        "seed=42",
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
with torch.no_grad():
    model.mixer.alpha.copy_(torch.tensor([0.0, alpha_pt, 0.0]))
for p in model.parameters():
    p.requires_grad_(False)  # alpha FIXED at the sweep value

# same exact classifier init as the M1 screen HEAD (same seed)
head_init_path = outdir.parent / "head_init.pt"
if not head_init_path.exists():
    torch.manual_seed({CLASSIFIER_SEED})
    init_head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    save_state(head_init_path, init_head)
head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
load_state_into(head_init_path, head)

opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
x = data.x.to(device)
ei = data.edge_index.to(device)
train_idx = data.train_idx.to(device)
y_train = data.y[data.train_idx].to(device)
val_idx = data.val_idx.to(device)
y_val = data.y[data.val_idx].to(device)

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

best_acc, best_f1, best_state = -1.0, None, None
patience_left = 30
t0 = time.monotonic()
stop_epoch = 300
best_epoch = None
for epoch in range(1, 301):
    head.train()
    opt.zero_grad(set_to_none=True)
    z, _, _, _, _ = model(x, ei)
    loss = criterion(head(z[train_idx]), y_train)
    loss.backward()
    opt.step()
    acc, f1 = val_metrics()
    if acc > best_acc:
        best_acc, best_epoch, best_f1 = acc, epoch, f1
        best_state = {{k: v.detach().clone() for k, v in head.state_dict().items()}}
        patience_left = 30
    else:
        patience_left -= 1
        if patience_left <= 0:
            stop_epoch = epoch
            break
runtime_sec = time.monotonic() - t0
head.load_state_dict(best_state)
head.eval()
with torch.no_grad():
    z_best = model(x, ei)[0]
m_full = val_metrics_with_head(head, z_best, data, device)
summary = {{
    "dataset": dataset, "alpha_pt": alpha_pt, "alpha": [0.0, alpha_pt, 0.0],
    "best_val_acc": best_acc, "best_val_macro_f1": best_f1,
    "best_epoch": best_epoch, "stop_epoch": stop_epoch,
    "runtime_sec": round(runtime_sec, 1),
    "smoothing": smoothing, "per_class_f1": m_full["per_class_f1"],
    "confusion": m_full["confusion"],
    "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
}}
with (outdir / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)
print(f"[run] {{dataset}} alpha_Pt={{alpha_pt:g}} best_acc={{best_acc:.5f}} "
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
    parser = argparse.ArgumentParser(description="D2.0.5-A alpha_Pt response curve")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = TARGET_DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, a) for d in datasets for a in ALPHA_PT_VALUES]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out=outputs/perf_r2d20_5/response", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (d, a) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(_run_one, d, a, gpu, args.force)] = (d, a)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
