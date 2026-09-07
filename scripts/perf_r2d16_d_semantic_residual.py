"""R2-Design-1.6 D1.6-D: Semantic Residual-Only warm-start screen
(plan §34-§40).

Strictly removes the Adaptive Common gate: C = 0.5*(c_t+c_v) FIXED. The
factor interaction residual is inserted BEFORE the parent graph path
(plan §36):

    F0 = [C, Pt, Pv]
    I  = [C*Pt, C*Pv, Pt*Pv, |C-Pt|, |C-Pv|, |Pt-Pv|]
    Delta = Linear(6d,128) LN GELU Linear(128,3d)  (last layer zero-init)
    F* = F0 + Delta
    -> frozen parent graph path -> frozen parent fusion -> fresh classifier

Parents: B0 mandatory; A0 FEASIBLE (audit §D: P3._graph_update takes f_block
as input — strict factor override, no parent weights touched).
HEAD = same parent without the residual, same exact classifier init.
Mismatch control (plan §40): keep C_i, use P_{pi(i)} partners in the
interaction (fixed perm 20260904), best checkpoint only, no training.

Usage:
    python scripts/perf_r2d16_d_semantic_residual.py --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d16_utils import PARENTS, R2D16_ROOT, TARGET_DATASETS  # noqa: E402

SEM_ROOT = R2D16_ROOT / "semantic"
VARIANTS = ("HEAD", "SEM")
CLASSIFIER_SEED = 20260904
MISMATCH_SEED = 20260904


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


def _run_one(parent: str, dataset: str, variant: str, seed: int, gpu: int,
             force: bool, epochs: int | None) -> None:
    outdir = SEM_ROOT / parent / dataset / variant / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"[{gpu}] {parent} {dataset} {variant}"
    if (outdir / "summary.json").exists() and not force:
        print(f"{tag} SKIP", flush=True)
        return
    code = f"""
import json, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d16_utils import (
    load_parent_setup, make_classifier_init, save_state, load_state_into,
)
from src.analysis.perf_r2d15_utils import fixed_node_permutation, val_metrics_with_head
from src.models.biaxis_r2d16_adapters import SemanticResidualAdapter

device = torch.device("cuda:0")
parent, dataset, variant, seed = "{parent}", "{dataset}", "{variant}", {seed}
epochs_override = {epochs if epochs is not None else 'None'}
outdir = Path(r"{outdir}")
factor_dim = {128}

setup = load_parent_setup(parent, dataset, seed, device)
model = setup.model.eval()
for p in model.parameters():
    p.requires_grad_(False)
x = setup.data.x.to(device)
ei = setup.data.edge_index.to(device)
num_nodes = int(x.size(0))

# fixed common: F0 = [C, Pt, Pv] with C = 0.5*(c_t+c_v) (plan §34)
x_t, x_v = model._split_modalities(x)
factors = model.factorizer(x_t, x_v)
f0 = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)

def parent_graph(f_star):
    if parent == "A0":
        graph_out = model._graph_update(f_star, ei, num_nodes)
        f_out = graph_out["f_tilde"]
    else:
        f_out, _n, _b, _f = model._graph_update(f_star, ei, num_nodes)
    return model.fusion(torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1))

z_parent = parent_graph(f0).detach()  # HEAD embedding

head_init_path = outdir.parent.parent / "head_init.pt"
if not head_init_path.exists():
    torch.manual_seed({CLASSIFIER_SEED})
    init_head = torch.nn.Linear(model.out_dim, int(setup.data.num_classes)).to(device)
    save_state(head_init_path, init_head)
head = torch.nn.Linear(model.out_dim, int(setup.data.num_classes)).to(device)
load_state_into(head_init_path, head)

adapter = None
if variant == "SEM":
    adapter = SemanticResidualAdapter(factor_dim).to(device)

params = list(head.parameters()) + (list(adapter.parameters()) if adapter else [])
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
train_idx = setup.data.train_idx.to(device)
y_train = setup.data.y[setup.data.train_idx].to(device)
val_idx = setup.data.val_idx.to(device)
y_val = setup.data.y[setup.data.val_idx].to(device)

def forward_z(f0_in=None):
    f_star = f0 if adapter is None else adapter(f0_in if f0_in is not None else f0)
    return parent_graph(f_star)

def val_metrics(z=None):
    head.eval()
    with torch.no_grad():
        pred = head((forward_z() if z is None else z)[val_idx]).argmax(dim=-1)
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    y = y_val.cpu().numpy(); p = pred.cpu().numpy()
    return float(accuracy_score(y, p)), float(f1_score(y, p, average="macro", zero_division=0))

best_acc = -1.0
best_state = None
patience_left = 30
total_epochs = 300 if epochs_override is None else epochs_override
t0 = time.monotonic()
stop_epoch = total_epochs
best_epoch = None
for epoch in range(1, total_epochs + 1):
    head.train()
    if adapter is not None:
        adapter.train()
    opt.zero_grad(set_to_none=True)
    z = forward_z()
    loss = criterion(head(z[train_idx]), y_train)
    loss.backward()
    opt.step()
    acc, f1 = val_metrics(z)
    if acc > best_acc:
        best_acc, best_epoch, best_f1 = acc, epoch, f1
        best_state = {{
            "head": {{k: v.detach().clone() for k, v in head.state_dict().items()}},
            "adapter": ({{k: v.detach().clone() for k, v in adapter.state_dict().items()}}
                        if adapter else None),
        }}
        patience_left = 30
    else:
        patience_left -= 1
        if patience_left <= 0:
            stop_epoch = epoch
            break
runtime_sec = time.monotonic() - t0
head.load_state_dict(best_state["head"])
if adapter is not None:
    adapter.load_state_dict(best_state["adapter"])
head.eval()
if adapter is not None:
    adapter.eval()

diag = {{"val_acc": best_acc, "val_macro_f1": best_f1, "best_epoch": best_epoch,
        "stop_epoch": stop_epoch, "runtime_sec": round(runtime_sec, 1)}}
if adapter is not None:
    with torch.no_grad():
        f_star = adapter(f0)
        delta = f_star - f0
        ratio = delta.norm(dim=-1) / (f0.norm(dim=-1) + 1e-8)
        diag["residual_ratio"] = {{
            "C": {{"mean": float(ratio[:,0].mean()), "std": float(ratio[:,0].std(unbiased=False))}},
            "Pt": {{"mean": float(ratio[:,1].mean()), "std": float(ratio[:,1].std(unbiased=False))}},
            "Pv": {{"mean": float(ratio[:,2].mean()), "std": float(ratio[:,2].std(unbiased=False))}},
        }}
        def pair_cos(u, v):
            un, vn = torch.nn.functional.normalize(u, dim=-1), torch.nn.functional.normalize(v, dim=-1)
            return float((un * vn).sum(dim=-1).mean())
        diag["refined_pair_cosine"] = {{
            "C_Pt": pair_cos(f_star[:, 0], f_star[:, 1]),
            "C_Pv": pair_cos(f_star[:, 0], f_star[:, 2]),
            "Pt_Pv": pair_cos(f_star[:, 1], f_star[:, 2]),
        }}
        diag["base_pair_cosine"] = {{
            "C_Pt": pair_cos(f0[:, 0], f0[:, 1]),
            "C_Pv": pair_cos(f0[:, 0], f0[:, 2]),
            "Pt_Pv": pair_cos(f0[:, 1], f0[:, 2]),
        }}
        # mismatch: keep C_i, permute the PARTNER factor rows in the
        # interaction (plan §40), best checkpoint only
        perm = fixed_node_permutation(num_nodes)
        f0_perm = torch.stack([f0[:, 0], f0[:, 1][perm], f0[:, 2][perm]], dim=1)
        delta_m = adapter.residual(f0_perm)
        z_m = parent_graph(f0 + delta_m)
        m_mis = val_metrics_with_head(head, z_m, setup.data, device)
        diag["mismatch_val_acc"] = m_mis["val_acc"]
        diag["mismatch_val_macro_f1"] = m_mis["val_macro_f1"]
        diag["adapter_params"] = sum(p.numel() for p in adapter.parameters())
        del z_m, delta_m
        torch.cuda.empty_cache()
else:
    diag["adapter_params"] = 0
m_full = val_metrics_with_head(head, forward_z(), setup.data, device)
diag["per_class_f1"] = m_full["per_class_f1"]
diag["classifier_params"] = sum(p.numel() for p in head.parameters())
diag["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
with (outdir / "summary.json").open("w") as f:
    json.dump({{"parent": parent, "dataset": dataset, "variant": variant,
                "seed": seed, **diag}}, f, indent=2)
print(f"[run] {{parent}} {{dataset}} {{variant}} best_acc={{best_acc:.5f}} "
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
    parser = argparse.ArgumentParser(description="D1.6-D semantic residual screen")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--parents", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    args = parser.parse_args()
    parents = list(PARENTS) if not args.parents else [p for p in args.parents.split(",")]
    datasets = TARGET_DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(p, d, v) for p in parents for d in datasets for v in VARIANTS]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out=outputs/perf_r2d16/semantic", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (p, d, v) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(_run_one, p, d, v, 42, gpu, args.force, args.epochs)] = (p, d, v)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
