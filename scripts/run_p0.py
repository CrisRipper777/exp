"""P0 job: train biaxis_p0 (NC or LP) for one dataset across MULTIPLE seeds,
then run factor sanity, edge statistics, and factor-wise propagation probes
for each seed.

Outputs (plan §15):
    outputs/p0/<dataset>/seed_<seed>/
        model.pt            best checkpoint (saved by the task runner)
        factor_sanity.json  P0-B
        edge_statistics.json P0-C
        nc_probe.csv / lp_probe.csv        P0-D
        nc_node_delta.pt / lp_edge_delta.pt
        conflict_stats.json
        summary.json

Usage:
    python scripts/run_p0.py --dataset Movies --task nc --seeds 42,43,44
    python scripts/run_p0.py --dataset sports-copurchase --task lp --include-test --force

Rules:
  - training is skipped when model.pt already exists (--retrain to override);
  - a seed's diagnostics are skipped when summary.json exists (--force to override);
  - each seed still gets its OWN training pass: MAGB split files are per-seed;
  - probe seed is fixed at 42 across all dataset seeds for comparability;
  - test metrics only enter outputs when --include-test is given.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and value != value:  # NaN -> None
        return None
    return value


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)


def _resolve_cfg(dataset: str, task: str, model_overrides: list[str] | None = None):
    # hydra resolves relative config_path against THIS file's directory
    # (scripts/), so use ../configs to reach the project config dir.
    overrides = [f"dataset={dataset}", f"task={task}", "model=biaxis_p0"] + list(model_overrides or [])
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="config", overrides=overrides)


def _train(
    dataset: str,
    task: str,
    seed: int,
    device: str,
    outdir: Path,
    train_epochs: int | None,
    model_overrides: list[str] | None,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        f"task={task}",
        "model=biaxis_p0",
        "num_runs=1",
        f"seed={seed}",
        f"device={device}",
        f"task.save_ckpt_path={outdir / 'model.pt'}",
        f"hydra.run.dir={outdir / 'hydra'}",
    ] + list(model_overrides or [])
    if train_epochs is not None:
        cmd.append(f"task.epochs={int(train_epochs)}")
    train_log = outdir / "train.log"
    print(f"[train] {' '.join(cmd[2:])}", flush=True)
    with train_log.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def _run_diagnostics(
    dataset: str,
    task: str,
    seed: int,
    device: str,
    outdir: Path,
    include_test: bool,
    probe_epochs: int | None,
    encode_batch: int,
    model_overrides: list[str] | None,
) -> None:
    from src.data import load_mag_data
    from src.models.biaxis_p0 import Model
    from src.utils.biaxis_p0_diagnostics import compute_edge_factor_statistics, compute_factor_sanity
    from src.utils.biaxis_p0_probes import run_lp_factor_probes, run_nc_factor_probes

    cfg = _resolve_cfg(dataset, task, model_overrides)
    data = load_mag_data(cfg, task, seed)
    ckpt = torch.load(outdir / "model.pt", map_location="cpu", weights_only=False)
    model = Model(cfg, ckpt["data_info"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(torch.device(device))

    print(f"[encode] {dataset} {task} seed={seed} nodes={data.num_nodes}", flush=True)
    factors = model.encode_factors(data.x, edge_index=data.edge_index, batch_size=encode_batch)

    p0 = cfg.model.p0
    sanity = compute_factor_sanity(factors, max_nodes=int(p0.max_diag_nodes))
    _write_json(outdir / "factor_sanity.json", sanity)
    edge_stats = compute_edge_factor_statistics(
        factors,
        data.edge_index,
        max_edges=int(p0.max_diag_edges),
        seed=42,
        top_ratios=tuple(float(r) for r in p0.edge_top_ratios),
        gap_threshold=float(p0.gap_threshold),
    )
    _write_json(outdir / "edge_statistics.json", edge_stats)

    print(f"[probe] {dataset} {task} seed={seed}", flush=True)
    probe_cfg = OmegaConf.create(dict(p0.probe))
    if probe_epochs is not None:
        probe_cfg.epochs = int(probe_epochs)
    if task == "nc":
        probe_result = run_nc_factor_probes(
            factors, data, torch.device(device), probe_cfg, output_dir=outdir, include_test=include_test, seed=42
        )
    else:
        probe_cfg = OmegaConf.merge(probe_cfg, OmegaConf.create(dict(p0.probe.lp)))
        probe_result = run_lp_factor_probes(
            factors, data, torch.device(device), probe_cfg, output_dir=outdir, include_test=include_test, seed=42
        )

    train_results = None
    results_json = outdir / "hydra" / "results.json"
    if results_json.exists():
        with results_json.open(encoding="utf-8") as f:
            train_results = json.load(f)

    summary = {
        "dataset": dataset,
        "task": task,
        "seed": seed,
        "device": device,
        "include_test": bool(include_test),
        "factor_sanity": sanity,
        "edge_statistics": edge_stats,
        "probe": probe_result["nc_probe"] if task == "nc" else probe_result["lp_probe"],
        "conflict": probe_result["nc_conflict"] if task == "nc" else probe_result["lp_conflict"],
        "fused_train_results": train_results,
        "config_snapshot": OmegaConf.to_container(cfg, resolve=True),
    }
    if task == "lp":
        summary["split_overlap"] = probe_result["lp_split_overlap"]
    _write_json(outdir / "summary.json", summary)
    print(f"[done] {dataset} {task} seed={seed} -> {outdir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 biaxis job (train + diagnostics + probes) for one dataset x multiple seeds")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True, choices=["nc", "lp"])
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="comma-separated seeds, one full train+diagnostics+probes pass each",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", default="outputs/p0")
    parser.add_argument("--include-test", action="store_true", help="run test evaluation in probes (final confirm only)")
    parser.add_argument("--retrain", action="store_true", help="retrain even if model.pt exists")
    parser.add_argument("--force", action="store_true", help="rerun diagnostics even if summary.json exists")
    parser.add_argument("--probe-epochs", type=int, default=None, help="override probe epochs (debug only)")
    parser.add_argument("--train-epochs", type=int, default=None, help="override train epochs (debug only)")
    parser.add_argument("--encode-batch", type=int, default=32768)
    parser.add_argument(
        "--model-overrides",
        default="",
        help="comma-separated model config overrides, e.g. lambda_common=0.02,lambda_recon=0.3",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="output subdir suffix when using model overrides, e.g. lc002_lr03",
    )
    args = parser.parse_args()

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    if not seeds:
        parser.error("--seeds must contain at least one seed")
    model_overrides = [item.strip() for item in args.model_overrides.split(",") if item.strip()] or None

    outdirs = {}
    for seed in seeds:
        seed_dir = f"seed_{seed}" + (f"_{args.variant}" if args.variant else "")
        outdirs[seed] = Path(args.out_root) / args.dataset / seed_dir
        outdirs[seed].mkdir(parents=True, exist_ok=True)

    # Phase 1: train every seed (each seed has its own split file).
    for seed in seeds:
        outdir = outdirs[seed]
        if (outdir / "summary.json").exists() and not args.force:
            print(f"[skip] {args.dataset} {args.task} seed={seed} (summary exists, use --force to rerun)", flush=True)
            continue
        ckpt_path = outdir / "model.pt"
        if not ckpt_path.exists() or args.retrain:
            _train(args.dataset, args.task, seed, args.device, outdir, args.train_epochs, model_overrides)
        else:
            print(f"[skip-train] {args.dataset} {args.task} seed={seed} (checkpoint exists: {ckpt_path})", flush=True)

    # Phase 2: diagnostics + probes per seed.
    for seed in seeds:
        outdir = outdirs[seed]
        if (outdir / "summary.json").exists() and not args.force:
            continue
        _run_diagnostics(
            args.dataset,
            args.task,
            seed,
            args.device,
            outdir,
            include_test=args.include_test,
            probe_epochs=args.probe_epochs,
            encode_batch=args.encode_batch,
            model_overrides=model_overrides,
        )


if __name__ == "__main__":
    main()
