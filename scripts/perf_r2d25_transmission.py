"""R2-D2.5-B: layer-wise utility transmission audit
(docs/BiAxis_R2_Design_2_5_Structured_Capacity_Utilization_Audit.md).

For each (dataset, seed): trace Pt-factor H1/H2 information through the
actual frozen-B0 computation and probe each stage with the SAME
StandardScaler + RidgeClassifier(alpha=1.0), TRAIN-fit / VAL-eval:

    S0 raw            : [Pt | H1]  vs  [Pt | H2]
    S1 source transform: [Pt | V_Pt(H1)] vs [Pt | V_Pt(H2)]
    S2 after LayerNorm : [Pt | LN(V(H1))] vs [Pt | LN(V(H2))]
    S3 factor residual : Pt + rho*LN(V(H1)) vs Pt + rho*LN(V(H2))
    S4 after fusion    : z (Pt ctx = H1) vs z_cf (Pt ctx = H2)
                         fixed parent classifier + retrained same-init head

Per stage: Acc / Macro-F1, H2-H1 utility delta, retention ratio relative
to S0, cosine, CKA, relative norm, effective rank.

Secondary PRODDIFF audit (Movies/Toys/Grocery only): retrain the D1.6-C
PRODDIFF adapter on frozen B0 states (~3s), then track the strongest cell
through raw 9 cells -> source mean -> factor add -> fusion: probe utility,
rank and pairwise cosine at each stage.

Outputs:
    outputs/perf_r2d25/transmission/<dataset>/seed_<s>/summary.json
    outputs/perf_r2d25/transmission/<dataset>/seed_<s>/proddiff.json  (M/T/G)
    outputs/perf_r2d25/transmission/scale_transmission.csv
    outputs/perf_r2d25/transmission/interaction_transmission.csv
    (report: scripts/summarize_perf_r2d25.py --stage transmission)

Usage:
    python scripts/perf_r2d25_transmission.py --gpus 0,1
    python scripts/perf_r2d25_transmission.py --datasets Movies --seeds 42 --gpu 0  # smoke
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

from src.analysis.perf_r2d25_utils import (  # noqa: E402
    DATASETS,
    R2D25_ROOT,
    SEEDS,
    TARGET_DATASETS,
)

TRANSMISSION_ROOT = R2D25_ROOT / "transmission"
PRODDIFF_EPOCHS = 300
PRODDIFF_PATIENCE = 30


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


def _run_one(dataset: str, seed: int, gpu: int, force: bool) -> None:
    outdir = TRANSMISSION_ROOT / dataset / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"[{gpu}] {dataset} seed={seed}"
    if (outdir / "summary.json").exists() and not force:
        print(f"{tag} SKIP", flush=True)
        return
    code = f"""
import json, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d25_utils import (
    effective_rank, load_or_make_head_init, load_r2d25_b0_setup,
    pt_transmission_features, ridge_probe, train_head_on_frozen_z,
)
from src.analysis.perf_r2d15_utils import (
    linear_cka, mean_cosine, mean_relative_l2, val_metrics_with_head,
)

device = torch.device("cuda:0")
dataset, seed = "{dataset}", {seed}
do_proddiff = dataset in {TARGET_DATASETS}
outdir = Path(r"{outdir}")
CLASSIFIER_SEED = 20260904

setup = load_r2d25_b0_setup(dataset, seed, device)
model = setup.model.eval()
for p in model.parameters():
    p.requires_grad_(False)
x = setup.data.x.to(device)
ei = setup.data.edge_index.to(device)
feats = pt_transmission_features(setup, x, ei)
train_idx = setup.data.train_idx
val_idx = setup.data.val_idx
y = setup.data.y

stages = [
    ("s0_raw", "s0_h1", "s0_h2"),
    ("s1_src_transform", "s1_h1", "s1_h2"),
    ("s2_after_ln", "s2_h1", "s2_h2"),
    ("s3_factor_residual", "s3_h1", "s3_h2"),
]
summary_rows = []
for name, key_h1, key_h2 in stages:
    f_h1, f_h2 = feats[key_h1], feats[key_h2]
    probe_h1 = ridge_probe(f_h1[train_idx], y[train_idx], f_h1[val_idx], y[val_idx])
    probe_h2 = ridge_probe(f_h2[train_idx], y[train_idx], f_h2[val_idx], y[val_idx])
    cka = linear_cka(f_h1[train_idx], f_h2[train_idx])
    summary_rows.append({{
        "dataset": dataset, "seed": seed, "stage": name, "branch": "h1",
        "acc": probe_h1["acc"], "macro_f1": probe_h1["macro_f1"],
        "cosine": mean_cosine(f_h1[train_idx], f_h2[train_idx]),
        "cka": cka,
        "rel_norm": mean_relative_l2(f_h1[train_idx], f_h2[train_idx]),
        "eff_rank": effective_rank(torch.cat([f_h1[train_idx], f_h2[train_idx]], dim=-1)),
    }})
    summary_rows.append({{
        "dataset": dataset, "seed": seed, "stage": name, "branch": "h2",
        "acc": probe_h2["acc"], "macro_f1": probe_h2["macro_f1"],
        "cosine": None, "cka": None, "rel_norm": None, "eff_rank": None,
    }})

# ---- S4: fusion z vs z_cf ------------------------------------------------
z, z_cf = feats["z"], feats["z_cf"]
m_fixed_h1 = val_metrics_with_head(setup.head, z, setup.data, device)
m_fixed_h2 = val_metrics_with_head(setup.head, z_cf, setup.data, device)
head_init_path = outdir / "head_init.pt"
num_classes = int(setup.data.num_classes)
head_a = load_or_make_head_init(head_init_path, model.out_dim, num_classes, device)
head_b = load_or_make_head_init(head_init_path, model.out_dim, num_classes, device)
res_a = train_head_on_frozen_z(z, head_a, setup.data, device)
res_b = train_head_on_frozen_z(z_cf, head_b, setup.data, device)
summary_rows.append({{
    "dataset": dataset, "seed": seed, "stage": "s4_fusion", "branch": "h1",
    "acc": res_a["best_val_acc"], "macro_f1": res_a["best_val_macro_f1"],
    "cosine": mean_cosine(z[train_idx], z_cf[train_idx]),
    "cka": linear_cka(z[train_idx], z_cf[train_idx]),
    "rel_norm": mean_relative_l2(z[train_idx], z_cf[train_idx]),
    "eff_rank": effective_rank(torch.cat([z[train_idx], z_cf[train_idx]], dim=-1)),
    "fixed_parent_acc": m_fixed_h1["val_acc"], "fixed_parent_f1": m_fixed_h1["val_macro_f1"],
    "retrained_acc": res_a["best_val_acc"], "retrained_f1": res_a["best_val_macro_f1"],
}})
summary_rows.append({{
    "dataset": dataset, "seed": seed, "stage": "s4_fusion", "branch": "h2",
    "acc": res_b["best_val_acc"], "macro_f1": res_b["best_val_macro_f1"],
    "cosine": None, "cka": None, "rel_norm": None, "eff_rank": None,
    "fixed_parent_acc": m_fixed_h2["val_acc"], "fixed_parent_f1": m_fixed_h2["val_macro_f1"],
    "retrained_acc": res_b["best_val_acc"], "retrained_f1": res_b["best_val_macro_f1"],
}})

summary = {{
    "dataset": dataset, "seed": seed, "rows": summary_rows,
    "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    "proddiff": None,
}}

# ---- secondary PRODDIFF transmission (M/T/G only) -------------------------
if do_proddiff:
    from src.analysis.perf_r2d15_utils import extract_b0_states
    from src.models.biaxis_r2d16_adapters import build_interaction_adapter

    states = extract_b0_states(model, x, ei)
    f_pre, n, f_out = states["f_pre"], states["n"], states["f_out"]
    adapter = build_interaction_adapter("PRODDIFF", 128).to(device)
    head = load_or_make_head_init(head_init_path, model.out_dim, num_classes, device)
    opt = torch.optim.AdamW(list(head.parameters()) + list(adapter.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    y_train = y[train_idx].to(device)
    y_val = y[val_idx].to(device)
    def z_hat():
        delta = adapter(f_pre, n)
        return model.fusion(torch.cat([f_out[:, 0] + delta[:, 0], f_out[:, 1] + delta[:, 1],
                                       f_out[:, 2] + delta[:, 2]], dim=-1))
    best_acc, best_state = -1.0, None
    patience_left = {PRODDIFF_PATIENCE}
    for epoch in range(1, {PRODDIFF_EPOCHS} + 1):
        head.train(); adapter.train()
        opt.zero_grad(set_to_none=True)
        zz = z_hat()
        loss = criterion(head(zz[train_idx]), y_train)
        loss.backward()
        opt.step()
        with torch.no_grad():
            pred = head(z_hat()[val_idx]).argmax(-1)
            acc = float((pred == y_val).float().mean().item())
        if acc > best_acc:
            best_acc = acc
            best_state = {{"head": {{k: v.detach().clone() for k, v in head.state_dict().items()}},
                           "adapter": {{k: v.detach().clone() for k, v in adapter.state_dict().items()}}}}
            patience_left = {PRODDIFF_PATIENCE}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    head.load_state_dict(best_state["head"]); adapter.load_state_dict(best_state["adapter"])
    head.eval(); adapter.eval()
    with torch.no_grad():
        delta = adapter(f_pre, n)
        cells = adapter.cell_deltas(f_pre, n)  # [b][a] -> [N, d]
        zz = z_hat()
    factor_names = ("C", "Pt", "Pv")
    pd_rows = []
    # raw 9 cells: probe utility [F_b | D_ab] vs [F_b]
    base_probes = {{}}
    for b in range(3):
        f_b = f_pre[:, b]
        base_probes[b] = ridge_probe(
            f_b[train_idx], y[train_idx], f_b[val_idx], y[val_idx])["acc"]
    cell_util = [[None] * 3 for _ in range(3)]
    cell_norm = [[None] * 3 for _ in range(3)]
    cell_means = []
    for b in range(3):
        for a in range(3):
            d_ab = cells[b][a]
            f_b = f_pre[:, b]
            cell_util[b][a] = ridge_probe(
                torch.cat([f_b, d_ab], dim=-1)[train_idx], y[train_idx],
                torch.cat([f_b, d_ab], dim=-1)[val_idx], y[val_idx])["acc"] - base_probes[b]
            cell_norm[b][a] = float(d_ab.norm(dim=-1).mean().item())
            cell_means.append(d_ab.mean(dim=0))
            pd_rows.append({{
                "dataset": dataset, "seed": seed, "stage": "raw_9_cells",
                "cell": f"{{factor_names[a]}}->{{factor_names[b]}}",
                "utility_delta": cell_util[b][a], "norm": cell_norm[b][a],
            }})
    cm = torch.stack(cell_means)
    cm_n = torch.nn.functional.normalize(cm, dim=-1)
    pd_rows.append({{
        "dataset": dataset, "seed": seed, "stage": "raw_9_cells", "cell": "PAIRWISE",
        "utility_delta": (cm_n @ cm_n.t()).detach().cpu().tolist(), "norm": None,
    }})
    # source mean: Delta^b = (1/3) sum_a D_ab
    src_mean = {{}}
    for b in range(3):
        src_mean[b] = sum(cells[b][a] for a in range(3)) / 3.0
        f_b = f_pre[:, b]
        util = ridge_probe(
            torch.cat([f_b, src_mean[b]], dim=-1)[train_idx], y[train_idx],
            torch.cat([f_b, src_mean[b]], dim=-1)[val_idx], y[val_idx])["acc"] - base_probes[b]
        pd_rows.append({{"dataset": dataset, "seed": seed, "stage": "source_mean",
                        "cell": factor_names[b], "utility_delta": util,
                        "norm": float(src_mean[b].norm(dim=-1).mean().item())}})
    sm = torch.stack([src_mean[b].mean(dim=0) for b in range(3)])
    sm_n = torch.nn.functional.normalize(sm, dim=-1)
    pd_rows.append({{"dataset": dataset, "seed": seed, "stage": "source_mean",
                    "cell": "PAIRWISE", "utility_delta": (sm_n @ sm_n.t()).detach().cpu().tolist(),
                    "norm": None}})
    # factor add: z_hat with fixed head + retrained head
    m_pd_fixed = val_metrics_with_head(head, zz, setup.data, device)
    head_rt = load_or_make_head_init(head_init_path, model.out_dim, num_classes, device)
    res_rt = train_head_on_frozen_z(zz, head_rt, setup.data, device)
    # strongest cell (by utility, fallback norm) tracked into fusion
    flat = [(cell_util[b][a], cell_norm[b][a], a, b) for b in range(3) for a in range(3)]
    best_cell = max(flat, key=lambda t: (t[0] if t[0] is not None else -1e9, t[1]))
    summary["proddiff"] = {{
        "best_val_acc": best_acc,
        "fixed_parent_acc": m_pd_fixed["val_acc"], "fixed_parent_f1": m_pd_fixed["val_macro_f1"],
        "retrained_acc": res_rt["best_val_acc"], "retrained_f1": res_rt["best_val_macro_f1"],
        "strongest_cell": f"{{factor_names[best_cell[2]]}}->{{factor_names[best_cell[3]]}}",
        "strongest_cell_utility": best_cell[0], "strongest_cell_norm": best_cell[1],
        "rows": pd_rows,
    }}

with (outdir / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)
print(f"[done] {{dataset}} s{{seed}}", flush=True)
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
    parser = argparse.ArgumentParser(description="R2-D2.5-B transmission audit")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    seeds = SEEDS if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, s) for d in datasets for s in seeds]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out=outputs/perf_r2d25/transmission", flush=True)
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
