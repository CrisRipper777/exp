"""R2-Design-1.6 D1.6-C: dual-parent frozen interaction adapter screen
(plan §19-§32).

Per (parent, dataset): load the parent best checkpoint, FREEZE everything,
extract {f_pre, n, f_out, z}; train adapter + fresh classifier only.
    HEAD (no adapter) / CONCAT / PRODDIFF / FiLM  — same exact classifier
    init (saved per parent/dataset), AdamW 1e-3 wd1e-4, 300 ep, patience 30,
    best Val Acc. Best checkpoint -> fixed-permutation mismatch
    (seed=20260904, n rows only) + cell-level novelty/specialization
    diagnostics. Macro-F1 safety: delta < -0.50pp vs HEAD -> WARNING.

Usage:
    python scripts/perf_r2d16_c_interaction_dual_parent.py --gpus 0,1
    python scripts/perf_r2d16_c_interaction_dual_parent.py --parents A0 \
        --datasets Movies --variants HEAD,PRODDIFF --gpu 0 --epochs 5   # smoke
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

from run_p1_screen import _poll_peak_mem  # noqa: E402
from src.analysis.perf_r2d16_utils import (  # noqa: E402
    PARENTS,
    R2D16_ROOT,
    TARGET_DATASETS,
)

VARIANTS = ("HEAD", "CONCAT", "PRODDIFF", "FiLM")
INTERACTION_ROOT = R2D16_ROOT / "interaction"
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
    outdir = INTERACTION_ROOT / parent / dataset / variant / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"[{gpu}] {parent} {dataset} {variant}"
    summary_path = outdir / "summary.json"
    if summary_path.exists() and not force:
        print(f"{tag} SKIP", flush=True)
        return
    code = f"""
import json, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d16_utils import (
    extract_parent_states, load_parent_setup, make_classifier_init,
    save_state, load_state_into,
)
from src.analysis.perf_r2d15_utils import (
    fixed_node_permutation, val_metrics_with_head,
)
from src.models.biaxis_r2d16_adapters import build_interaction_adapter

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
states = extract_parent_states(setup, x, ei)
f_pre, n, f_out, z_parent = states["f_pre"], states["n"], states["f_out"], states["z"]

# exact shared classifier init (plan §24)
head_init_path = outdir.parent.parent / "head_init.pt"
if not head_init_path.exists():
    torch.manual_seed({CLASSIFIER_SEED})
    init_head = torch.nn.Linear(model.out_dim, int(setup.data.num_classes)).to(device)
    save_state(head_init_path, init_head)
head = torch.nn.Linear(model.out_dim, int(setup.data.num_classes)).to(device)
load_state_into(head_init_path, head)

adapter = None
if variant != "HEAD":
    adapter = build_interaction_adapter(variant, factor_dim).to(device)

params = list(head.parameters()) + (list(adapter.parameters()) if adapter else [])
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
train_idx = setup.data.train_idx.to(device)
y_train = setup.data.y[setup.data.train_idx].to(device)
val_idx = setup.data.val_idx.to(device)
y_val = setup.data.y[setup.data.val_idx].to(device)

def forward_z():
    if adapter is None:
        return z_parent
    delta = adapter(f_pre, n)
    return model.fusion(torch.cat([f_out[:, 0] + delta[:, 0], f_out[:, 1] + delta[:, 1],
                                   f_out[:, 2] + delta[:, 2]], dim=-1))

def val_metrics():
    head.eval()
    with torch.no_grad():
        pred = head(forward_z()[val_idx]).argmax(dim=-1)
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    y = y_val.cpu().numpy(); p = pred.cpu().numpy()
    return float(accuracy_score(y, p)), float(f1_score(y, p, average="macro", zero_division=0))

best_acc = -1.0
best_state = None
patience = 30
patience_left = patience
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
    acc, f1 = val_metrics()
    if acc > best_acc:
        best_acc, best_epoch = acc, epoch
        best_state = {{
            "head": {{k: v.detach().clone() for k, v in head.state_dict().items()}},
            "adapter": ({{k: v.detach().clone() for k, v in adapter.state_dict().items()}}
                        if adapter else None),
        }}
        best_f1 = f1
        patience_left = patience
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

# ---- diagnostics at best checkpoint ----
diag = {{"val_acc": best_acc, "val_macro_f1": best_f1, "best_epoch": best_epoch,
        "stop_epoch": stop_epoch, "runtime_sec": round(runtime_sec, 1)}}
m_full = val_metrics_with_head(head, forward_z(), setup.data, device)
diag["per_class_f1"] = m_full["per_class_f1"]
params_count = 0
if adapter is not None:
    with torch.no_grad():
        delta = adapter(f_pre, n)
        cells = adapter.cell_deltas(f_pre, n)
        base = states["base_update"]
        # residual ratio per factor
        ratio = delta.norm(dim=-1) / (f_out.norm(dim=-1) + 1e-8)  # [N,3]
        diag["residual_ratio"] = {{
            "C": {{"mean": float(ratio[:,0].mean()), "std": float(ratio[:,0].std(unbiased=False))}},
            "Pt": {{"mean": float(ratio[:,1].mean()), "std": float(ratio[:,1].std(unbiased=False))}},
            "Pv": {{"mean": float(ratio[:,2].mean()), "std": float(ratio[:,2].std(unbiased=False))}},
        }}
        # 9-cell norm matrix + cosine to parent update + orthogonal novelty
        cell_norm = torch.zeros(3, 3)
        cell_cos = torch.zeros(3, 3)
        cell_novel = torch.zeros(3, 3)
        cell_means = []
        for b in range(3):
            base_b = base[:, b]
            base_norm = base_b.norm(dim=-1, keepdim=True) + 1e-8
            for a in range(3):
                d_ab = cells[b][a]
                cell_norm[a, b] = d_ab.norm(dim=-1).mean()
                cell_means.append(d_ab.mean(dim=0))
                cos = (d_ab * base_b).sum(dim=-1) / (d_ab.norm(dim=-1) + 1e-8) / base_norm.squeeze(-1)
                cell_cos[a, b] = cos.mean()
                proj = (d_ab * base_b).sum(dim=-1, keepdim=True) / base_norm * base_b
                novelty = (d_ab - proj).norm(dim=-1) / (d_ab.norm(dim=-1) + 1e-8)
                cell_novel[a, b] = novelty.mean()
        diag["cell_norm"] = cell_norm.tolist()
        diag["cell_cosine_to_parent_update"] = cell_cos.tolist()
        diag["cell_orthogonal_novelty"] = cell_novel.tolist()
        # 9x9 specialization
        cm = torch.stack(cell_means)  # [9, d]
        cm_n = torch.nn.functional.normalize(cm, dim=-1)
        pairwise = cm_n @ cm_n.t()
        off_diag = pairwise[~torch.eye(9, dtype=torch.bool)]
        sv = torch.linalg.svdvals(cm)
        sv = sv / (sv.sum() + 1e-12)
        eff_rank = float(torch.exp(-(sv * torch.log(sv + 1e-12)).sum()))
        diag["cell_pairwise_cosine"] = pairwise.tolist()
        diag["cell_mean_offdiag_cosine"] = float(off_diag.mean())
        diag["cell_effective_rank"] = eff_rank
        # mismatch control (fixed perm, n rows only, no training)
        perm = fixed_node_permutation(int(f_pre.size(0)))
        n_perm = n[perm]
        with torch.no_grad():
            delta_m = adapter(f_pre, n_perm)
            z_m = model.fusion(torch.cat([f_out[:, 0] + delta_m[:, 0], f_out[:, 1] + delta_m[:, 1],
                                          f_out[:, 2] + delta_m[:, 2]], dim=-1))
        m_mis = val_metrics_with_head(head, z_m, setup.data, device)
        diag["mismatch_val_acc"] = m_mis["val_acc"]
        diag["mismatch_val_macro_f1"] = m_mis["val_macro_f1"]
        del z_m, delta_m
        torch.cuda.empty_cache()
    params_count = sum(p.numel() for p in adapter.parameters())
diag["adapter_params"] = params_count
diag["classifier_params"] = sum(p.numel() for p in head.parameters())
peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
diag["peak_allocated_mb"] = round(peak_mb, 1)
with (outdir / "summary.json").open("w") as f:
    json.dump({{
        "parent": parent, "dataset": dataset, "variant": variant, "seed": seed,
        **diag,
    }}, f, indent=2)
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
    parser = argparse.ArgumentParser(description="D1.6-C dual-parent interaction screen")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--parents", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    args = parser.parse_args()
    parents = list(PARENTS) if not args.parents else [p for p in args.parents.split(",")]
    datasets = TARGET_DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    variants = list(VARIANTS) if not args.variants else [v for v in args.variants.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(p, d, v, s) for p in parents for d in datasets for v in variants for s in seeds]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out=outputs/perf_r2d16/interaction", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (p, d, v, s) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(
                _run_one, p, d, v, s, gpu, args.force, args.epochs
            )] = (p, d, v, s)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
