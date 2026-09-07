"""R2-D2.5-C: structured-capacity model matrix
(docs/BiAxis_R2_Design_2_5_Structured_Capacity_Utilization_Audit.md).

Variants: EARLY_MIX / SEP_SUM / SEP_CONCAT / INCEPTION_012 / CAP_H1_DUP /
WIDE_B0 / DEEP_FUSION on Movies/Toys/Grocery x seeds 42/43/44.

Unified schedule (hard-coded in train_capacity_variant):
    epoch 1-20  P0 factorizer frozen
    epoch 21+   P0 unfrozen, lr 1e-4
    graph/readout/fusion/classifier lr 1e-3
    AdamW wd 1e-4; warmup10 + cosine; 300 epochs; patience 30; best Val Acc.
Val only — this driver NEVER touches test.

Per run: history.csv, grad_samples.json, summary.json (best metrics,
ablation FULL/H2-OFF/H1-OFF(/H0-OFF), expert effective rank / pairwise
cosine / CKA, readout weight norms, parameter accounting incl. the
WIDE_B0 <-> SEP_CONCAT +/-5% match).

Outputs:
    outputs/perf_r2d25/capacity/<dataset>/<variant>/seed_<s>/
    (aggregation: scripts/summarize_perf_r2d25.py --stage capacity)

Usage:
    python scripts/perf_r2d25_capacity_train.py --gpus 0,1
    python scripts/perf_r2d25_capacity_train.py --datasets Movies \
        --variants early_mix,sep_concat --seeds 42 --epochs 5     # smoke
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d25_utils import (  # noqa: E402
    CAPACITY_MODES,
    R2D25_ROOT,
    TARGET_DATASETS,
)

CAPACITY_ROOT = R2D25_ROOT / "capacity"
HEAD_INIT_ROOT = CAPACITY_ROOT / "head_init"


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


# ---------------------------------------------------------------------------
# Worker (also runnable directly: --worker)
# ---------------------------------------------------------------------------


def run_worker(dataset: str, variant: str, seed: int, outdir: Path,
               epochs: int | None = None, force: bool = False) -> None:
    import torch

    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.json"
    if summary_path.exists() and not force:
        print(f"[{dataset} {variant} s{seed}] SKIP", flush=True)
        return
    from src.analysis.perf_r2d25_utils import (
        ablation_metrics, load_mag_data_wrap, load_or_make_head_init,
        resolve_capacity_cfg, train_capacity_variant,
    )
    from src.models.biaxis_r2_capacity import Model

    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    cfg = resolve_capacity_cfg(dataset, seed, variant)
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
        history_callback=history_writer.writerow,
    )
    history_file.close()

    with torch.no_grad():
        diag = model.compute_capacity_diagnostics(data.x.to(device), data.edge_index.to(device))
    abl = ablation_metrics(model, head, data.x.to(device), data.edge_index.to(device),
                           data, device)
    runtime_sec = time.monotonic() - t0

    # parameter accounting (plan: C4 == C2 exact; C5 vs C2 within +/-5%)
    param_count = int(model.parameter_count)
    ref_params = None
    wide_match = diag.get("wide_match")
    if variant == "wide_b0" and wide_match is not None:
        ref_params = int(wide_match["target_sep_concat_params"])
    elif variant == "cap_h1_dup":
        ref_cfg = resolve_capacity_cfg(dataset, seed, "sep_concat")
        ref = Model(ref_cfg, info)
        ref_params = int(ref.parameter_count)
        del ref

    summary = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "capacity_mode": variant,
        "best_val_acc": res["best_val_acc"],
        "best_val_macro_f1": res["best_val_macro_f1"],
        "per_class_f1": res["per_class_f1"],
        "best_epoch": res["best_epoch"],
        "stop_epoch": res["stop_epoch"],
        "p0_unfrozen": res["p0_unfrozen"],
        "parameter_count": param_count,
        "reference_sep_concat_params": ref_params,
        "param_delta_pct": (
            round(100.0 * (param_count - ref_params) / ref_params, 3)
            if ref_params else None
        ),
        "ablations": abl,
        "diagnostics": diag,
        "runtime_sec": round(runtime_sec, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (outdir / "grad_samples.json").open("w", encoding="utf-8") as f:
        json.dump(res["grad_samples"], f, indent=2)
    print(
        f"[run] {dataset} {variant} s{seed} best_acc={res['best_val_acc']:.5f} "
        f"f1={res['best_val_macro_f1']:.5f} ep={res['best_epoch']}/{res['stop_epoch']} "
        f"params={param_count} ({runtime_sec:.0f}s)", flush=True,
    )


def _run_one(dataset: str, variant: str, seed: int, gpu: int, force: bool,
             epochs: int | None) -> None:
    outdir = CAPACITY_ROOT / dataset / variant / f"seed_{seed}"
    tag = f"[{gpu}] {dataset} {variant} seed={seed}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--dataset", dataset, "--variant", variant, "--seed", str(seed),
        "--outdir", str(outdir),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(int(epochs))]
    if force:
        cmd += ["--force"]
    log = outdir / "run.log"
    outdir.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"{tag} FAILED rc={proc.returncode}", flush=True)
        print(log.read_text(encoding="utf-8")[-3000:], flush=True)
        return
    print(f"{tag} OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-D2.5-C capacity matrix")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--out-root", default=None,
                        help="override output root (default outputs/perf_r2d25/capacity)")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.dataset, args.variant, args.seed, Path(args.outdir),
                   epochs=args.epochs, force=args.force)
        return

    global CAPACITY_ROOT
    if args.out_root:
        CAPACITY_ROOT = Path(args.out_root)

    datasets = TARGET_DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    variants = list(CAPACITY_MODES) if not args.variants else [v for v in args.variants.split(",")]
    unknown = [v for v in variants if v not in CAPACITY_MODES]
    if unknown:
        parser.error(f"unknown variants: {unknown}")
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, v, s) for d in datasets for v in variants for s in seeds]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out=outputs/perf_r2d25/capacity", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (d, v, s) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(_run_one, d, v, s, gpu, args.force, args.epochs)] = (d, v, s)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
