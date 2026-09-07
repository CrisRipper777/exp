"""R2-D2.6-A/B: strong-parent integration driver
(docs/BiAxis_R2_Design_2_6_Strong_Parent_Readout_Integration.md).

Variants (VARIANTS in perf_r2d26_utils):
    A0_BASE      frozen A0 z_base + fresh matched classifier
    NC_HOP/NC_H1 [z_base | 9 tokens], no projection back to 256
    FHC_HOP/H1   [z_base | s_C | s_Pt | s_Pv]
    RSF_HOP/H1   z_base + R_side(ResidualFusion([s]))
    HIER_HOP/H1  z_base + W_o(T_final[0] - z_base)  (base-anchored 2-block attn)
    READOUT_ONLY z_base + param-matched residual MLP

A0 fully frozen; side/aux heads/classifier lr 1e-3 wd 1e-4, warmup10+
cosine, 300 ep / patience 30 / best Val Acc; expert deep supervision
lambda=0.1 (plan §23). Val only — this driver NEVER touches test.

Outputs: outputs/perf_r2d26/integration/<ds>/<variant>/seed_<s>/
    {summary.json, history.csv, best.pt, run.log}
(D2.6-A reuses this driver via scripts/perf_r2d26_no_compression.py with
the NC/A0_BASE variant set and the no_compression/ root.)

Usage:
    python scripts/perf_r2d26_integration.py --gpus 0,1
    python scripts/perf_r2d26_integration.py --datasets Movies --variants FHC_HOP,FHC_H1 --seeds 42 --epochs 5  # smoke
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

from src.analysis.perf_r2d26_utils import (  # noqa: E402
    DATASETS,
    R2D26_ROOT,
    VARIANTS,
    causal_metrics,
    load_a0_parent,
    load_or_make_head_init,
    scheduled_lr,
    train_strong_parent,
)

INTEGRATION_ROOT = R2D26_ROOT / "integration"
HEAD_INIT_ROOT = R2D26_ROOT / "head_init"


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


def resolve_cfg(dataset: str, seed: int, variant: str):
    from hydra import compose, initialize_config_dir

    readout_type, token_source = VARIANTS[variant]
    overrides = [
        f"dataset={dataset}", "task=nc", "model=biaxis_r2_strong_parent",
        f"model.readout_type={readout_type}", f"model.token_source={token_source}",
        f"seed={int(seed)}",
    ]
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        return compose(config_name="config", overrides=overrides)


def _train_a0_base_head(z_base, head, data, device, total_epochs, patience):
    """A0_BASE: fresh classifier on the FROZEN z_base under the candidate
    protocol (warmup10+cosine, AdamW 1e-3 wd 1e-4, best Val Acc)."""
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train_idx = data.train_idx.to(device)
    y_train = data.y[data.train_idx].to(device)
    val_idx = data.val_idx.to(device)
    y_val = data.y[data.val_idx].to(device)
    z_tr, z_va = z_base[train_idx], z_base[val_idx]
    best_acc, best_epoch, best_state = -1.0, None, None
    patience_left = patience
    stop_epoch = total_epochs
    for epoch in range(1, total_epochs + 1):
        for pg in opt.param_groups:
            pg["lr"] = scheduled_lr(epoch, total_epochs, 1e-3)
        head.train()
        opt.zero_grad(set_to_none=True)
        logits = head(z_tr)
        loss = criterion(logits, y_train)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            pred_v = head(z_va).argmax(-1)
            acc = float((pred_v == y_val).float().mean().item())
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                stop_epoch = epoch
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        z_best = z_base
    from src.analysis.perf_r2d15_utils import val_metrics_with_head

    m_full = val_metrics_with_head(head, z_best, data, device)
    return {
        "best_val_acc": best_acc, "best_val_macro_f1": m_full["val_macro_f1"],
        "per_class_f1": m_full["per_class_f1"],
        "best_epoch": best_epoch, "stop_epoch": stop_epoch,
    }


def run_worker(dataset: str, variant: str, seed: int, outdir: Path,
               out_root: Path, epochs: int | None, force: bool,
               deep_sup_lambda: float | None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} {variant} s{seed}] SKIP", flush=True)
        return
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    setup = load_a0_parent(dataset, seed, device)
    cfg = resolve_cfg(dataset, seed, variant)
    if deep_sup_lambda is not None:
        cfg.model.deep_supervision.enabled = True
        cfg.model.deep_supervision["lambda"] = deep_sup_lambda
    info = {
        "input_dim": setup.data.input_dim, "num_nodes": setup.data.num_nodes,
        "num_classes": setup.data.num_classes,
        "text_dim": int(setup.data.x_t.shape[1]), "visual_dim": int(setup.data.x_i.shape[1]),
    }
    data = setup.data
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    total_epochs = 300 if epochs is None else int(epochs)
    t0 = time.monotonic()

    if variant == "A0_BASE":
        num_nodes = int(x.size(0))
        with torch.no_grad():
            factors, _ = setup.parent._encode(x)
            f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
            graph_out = setup.parent._graph_update(f_block, ei, num_nodes)
            f_tilde = graph_out["f_tilde"]
            z_base = setup.parent.fusion(
                torch.cat([f_tilde[:, 0], f_tilde[:, 1], f_tilde[:, 2]], dim=-1))
        head = load_or_make_head_init(
            HEAD_INIT_ROOT / f"{dataset}_seed{seed}_d{setup.parent.hidden_dim}.pt",
            setup.parent.hidden_dim, int(data.num_classes), device)
        res = _train_a0_base_head(z_base, head, data, device, total_epochs, 30)
        summary = {
            "dataset": dataset, "variant": variant, "seed": seed,
            **res,
            "side_params": 0, "parent_params": 0,
            "runtime_sec": round(time.monotonic() - t0, 1),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
            "ablations": None, "diagnostics": None,
        }
        with (outdir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[run] {dataset} {variant} s{seed} best_acc={res['best_val_acc']:.5f} "
              f"f1={res['best_val_macro_f1']:.5f} ep={res['best_epoch']}/{res['stop_epoch']}",
              flush=True)
        return

    from src.models.biaxis_r2_strong_parent import Model

    model = Model(cfg, info, setup.parent).to(device)
    head = load_or_make_head_init(
        HEAD_INIT_ROOT / f"{dataset}_seed{seed}_d{model.out_dim}.pt",
        model.out_dim, int(data.num_classes), device)
    history_path = outdir / "history.csv"
    history_file = history_path.open("w", encoding="utf-8", newline="")
    history_writer = csv.DictWriter(
        history_file, fieldnames=["epoch", "lr", "train_ce", "val_acc"])
    history_writer.writeheader()
    res = train_strong_parent(
        data, model, head, device, total_epochs=total_epochs,
        deep_sup_lambda=(0.1 if deep_sup_lambda is None else deep_sup_lambda),
        history_callback=history_writer.writerow,
    )
    history_file.close()

    # quick causal metrics at the best checkpoint (full D2.6-C runs later).
    # side_off returns z_base (h-dim): only residual readouts (out_dim == h)
    # can evaluate it through the candidate head; base preservation is
    # reported bitwise in diagnostics regardless.
    causal_keys = ["full"]
    if model.out_dim == model.hidden_dim:
        causal_keys.append("side_off")
    if model.token_source == "hop":
        causal_keys += ["h2_zero", "h2_to_h1"]
    abl = causal_metrics(model, head, x, ei, data, device, causal_keys=tuple(causal_keys))
    with torch.no_grad():
        diag = model.compute_diagnostics(x, ei)
    torch.save({"head_state": head.state_dict(), "model_state": model.state_dict()},
               outdir / "best.pt")
    summary = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "readout_type": model.readout_type, "token_source": model.token_source,
        "best_val_acc": res["best_val_acc"],
        "best_val_macro_f1": res["best_val_macro_f1"],
        "per_class_f1": res["per_class_f1"],
        "best_epoch": res["best_epoch"], "stop_epoch": res["stop_epoch"],
        "side_params": int(model.side_parameter_count),
        "parent_params": int(sum(p.numel() for p in setup.parent.parameters())),
        "out_dim": int(model.out_dim),
        "ablations": abl,
        "diagnostics": diag,
        "runtime_sec": round(time.monotonic() - t0, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] {dataset} {variant} s{seed} best_acc={res['best_val_acc']:.5f} "
          f"f1={res['best_val_macro_f1']:.5f} ep={res['best_epoch']}/{res['stop_epoch']} "
          f"side_params={model.side_parameter_count} ({summary['runtime_sec']:.0f}s)",
          flush=True)


def _run_one(dataset, variant, seed, gpu, force, epochs, out_root, deep_sup_lambda):
    outdir = out_root / dataset / variant / f"seed_{seed}"
    tag = f"[{gpu}] {dataset} {variant} seed={seed}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--dataset", dataset, "--variant", variant, "--seed", str(seed),
        "--outdir", str(outdir), "--out-root", str(out_root),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(int(epochs))]
    if deep_sup_lambda is not None:
        cmd += ["--deep-sup", str(deep_sup_lambda)]
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="R2-D2.6 strong-parent integration")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--deep-sup", type=float, default=None,
                        help="override deep supervision lambda (D2.6-E)")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args(argv)

    out_root = Path(args.out_root) if args.out_root else INTEGRATION_ROOT
    if args.worker:
        run_worker(args.dataset, args.variant, args.seed, Path(args.outdir),
                   out_root, args.epochs, args.force, args.deep_sup)
        return

    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    variants = list(VARIANTS) if not args.variants else [v for v in args.variants.split(",")]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {unknown}")
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, v, s) for d in datasets for v in variants for s in seeds]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out={out_root}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (d, v, s) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(_run_one, d, v, s, gpu, args.force, args.epochs,
                                    out_root, args.deep_sup)] = (d, v, s)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
