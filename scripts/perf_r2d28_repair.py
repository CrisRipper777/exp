"""R2-Design-2.8 v2 D2.8-A: repaired neighbor-identity attribution (v2 §6).

No retraining. Loads the existing D2.7 best checkpoints for
    PAIR_EDGE            (outputs/perf_r2d27/matrix/<ds>/PAIR_EDGE/seed_<s>)
    TARGET_FACTOR_ONLY   (outputs/perf_r2d27/ownership/<ds>/TARGET_FACTOR_ONLY/seed_<s>)
for all five datasets x seeds 42/43/44 and evaluates

    FULL
    WITHIN_TARGET_SHUFFLE_FIXED
    REMOVE_TOP/RANDOM/BOTTOM_PER_TARGET_{10,25,50}
    KEEP_TOP_PER_TARGET_{25,50}

with the repaired causal machinery (exact integer-segment permutation +
per-target removal). Also exports per-pair shuffle validation stats
(fraction_score_changed, fraction_nonidentity_targets).

Verdict (v2 §6):
    IDENTITY SUPPORTED:  FULL - SHUFFLE >= +0.30pp Acc macro (M/T/G)
                         or DROP_top - DROP_random >= +0.20pp Acc,
                         with F1 nonnegative and >=2/3 dataset means positive.
    IDENTITY WEAK:       consistent +0.10..+0.30pp.
    IDENTITY NOT SUPPORTED: corrected interventions remain ~zero.
The old shuffle=0 conclusion is NOT carried forward.

Outputs: outputs/perf_r2d28/repair/
    repair_results.csv, repair_shuffle_validation.csv, repair_removal.csv,
    per-run <ds>/<model>/seed_<s>/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d27_utils import (  # noqa: E402
    DATASETS,
    load_a0_parent,
)
from src.analysis.perf_r2d28_utils import (  # noqa: E402
    D27_CKPT_ROOTS,
    REPAIR_CAUSAL_KEYS,
    R2D28_ROOT,
    launch_jobs,
)

REPAIR_ROOT = R2D28_ROOT / "repair"
MODELS = ("PAIR_EDGE", "TARGET_FACTOR_ONLY")

OLD_MODES = {"PAIR_EDGE": "pair_edge", "TARGET_FACTOR_ONLY": "target_factor_only"}


def resolve_old_cfg(dataset: str, seed: int, mode: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"),
                               version_base=None):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=biaxis_r2_neighbor_utility",
            f"model.mode={mode}", f"seed={int(seed)}"])


def run_worker(dataset: str, variant: str, seed: int, outdir: Path,
               force: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} {variant} s{seed}] SKIP", flush=True)
        return
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    setup = load_a0_parent(dataset, seed, device)
    data = setup.data
    mode = OLD_MODES[variant]
    cfg = resolve_old_cfg(dataset, seed, mode)
    info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
            "num_classes": data.num_classes,
            "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1])}

    from src.models.biaxis_r2_neighbor_utility import Model

    model = Model(cfg, info, setup.parent).to(device)
    ckpt_path = D27_CKPT_ROOTS[variant] / dataset / variant / f"seed_{seed}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(ckpt["head_state"])
    model.eval()
    head.eval()

    x = data.x.to(device)
    ei = data.edge_index.to(device)
    num_nodes = int(x.size(0))
    t0 = time.monotonic()

    from src.analysis.perf_r2d15_utils import val_metrics_with_head
    from src.models.biaxis_r2_relfunc_components import (
        shuffle_scores_within_target,
        validate_shuffle,
    )

    results = {}
    with torch.no_grad():
        for key in REPAIR_CAUSAL_KEYS:
            z, _, _, _, _ = model(x, ei, causal=key)
            m = val_metrics_with_head(head, z, data, device)
            results[key] = {"val_acc": m["val_acc"],
                            "val_macro_f1": m["val_macro_f1"]}
            del z

    # per-pair shuffle validation on the trained scores (real graph)
    f_block, _ = model._parent_ctx(x, ei, num_nodes)
    shuffle_validation = {}
    with torch.no_grad():
        for (a, b) in [(0, 1), (1, 1), (2, 2)]:
            s = model._pair_scores_chunked(f_block, ei, a, b, int(ei.size(1)))
            s_perm = shuffle_scores_within_target(s, ei)
            st = validate_shuffle(s, s_perm, ei, num_nodes)
            shuffle_validation[f"pair_{a}{b}"] = st
    # per-target removal count validation on real scores (pair 0->1)
    from src.models.biaxis_r2_relfunc_components import per_target_edge_mask

    deg = torch.bincount(ei[1], minlength=num_nodes)
    removal_validation = {}
    with torch.no_grad():
        s01 = model._pair_scores_chunked(f_block, ei, 0, 1, int(ei.size(1)))
        for op, pct in (("remove_top", 0.10), ("remove_random", 0.10),
                        ("remove_bottom", 0.25), ("keep_top", 0.25)):
            mask = per_target_edge_mask(s01, ei, num_nodes, op, pct)
            n_sel = int(sum(int(deg[i] * pct) for i in range(num_nodes)))
            expected = int(ei.size(1)) - n_sel if op == "keep_top" else n_sel
            removal_validation[f"{op}_{pct}"] = {
                "expected_removed": expected, "actual_removed": int((~mask).sum().item())}

    summary = {
        "dataset": dataset, "model": variant, "seed": seed, "mode": mode,
        "causal": results,
        "shuffle_validation": shuffle_validation,
        "removal_validation": removal_validation,
        "runtime_sec": round(time.monotonic() - t0, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[run] {dataset} {variant} s{seed} full_acc={results['full']['val_acc']:.5f} "
          f"({summary['runtime_sec']:.0f}s)", flush=True)


def summarize() -> None:
    """Aggregate per-run summaries into the three required CSVs."""
    rows = []
    for p in sorted(REPAIR_ROOT.rglob("summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for key, m in d["causal"].items():
            rows.append({
                "dataset": d["dataset"], "model": d["model"], "seed": d["seed"],
                "causal": key, "val_acc": m["val_acc"],
                "val_macro_f1": m["val_macro_f1"], "_dir": str(p.parent),
            })
    with (REPAIR_ROOT / "repair_results.csv").open("w", encoding="utf-8",
                                                   newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "seed", "causal",
                                          "val_acc", "val_macro_f1"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    shuffle_rows = []
    for p in sorted(REPAIR_ROOT.rglob("summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for pair, st in d["shuffle_validation"].items():
            shuffle_rows.append({
                "dataset": d["dataset"], "model": d["model"], "seed": d["seed"],
                "pair": pair,
                "frac_score_changed": st["frac_score_changed"],
                "frac_nonidentity_targets": st["frac_nonidentity_targets"],
                "sums_preserved": st["sums_preserved"],
            })
    with (REPAIR_ROOT / "repair_shuffle_validation.csv").open(
            "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "seed", "pair",
                                          "frac_score_changed",
                                          "frac_nonidentity_targets",
                                          "sums_preserved"])
        w.writeheader()
        for r in shuffle_rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    removal_rows = []
    for p in sorted(REPAIR_ROOT.rglob("summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for key, v in d["removal_validation"].items():
            removal_rows.append({
                "dataset": d["dataset"], "model": d["model"], "seed": d["seed"],
                "override": key, "expected_removed": v["expected_removed"],
                "actual_removed": v["actual_removed"],
            })
    with (REPAIR_ROOT / "repair_removal.csv").open("w", encoding="utf-8",
                                                   newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "seed", "override",
                                          "expected_removed", "actual_removed"])
        w.writeheader()
        for r in removal_rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    # verdicts (v2 §6 thresholds)
    import statistics

    def _papermodel(model):
        # FULL - SHUFFLE and DROP_top10 - DROP_random10 over M/T/G, 3 seeds
        rows_m = {r["dataset"]: r for r in rows}
        full = {(d, s): rows_m.get((d, s)) for d in DATASETS for s in (42, 43, 44)}
        acc_deltas, f1_deltas = [], []
        for d in ("Movies", "Toys", "Grocery"):
            ds = []
            for s in (42, 43, 44):
                f_row = next((r for r in rows if r["dataset"] == d and r["seed"] == s
                              and r["model"] == model and r["causal"] == "full"), None)
                s_row = next((r for r in rows if r["dataset"] == d and r["seed"] == s
                              and r["model"] == model
                              and r["causal"] == "within_target_shuffle_fixed"), None)
                t_row = next((r for r in rows if r["dataset"] == d and r["seed"] == s
                              and r["model"] == model
                              and r["causal"] == "remove_top_per_target_10"), None)
                r_row = next((r for r in rows if r["dataset"] == d and r["seed"] == s
                              and r["model"] == model
                              and r["causal"] == "remove_random_per_target_10"), None)
                if f_row and s_row:
                    ds.append((100 * (f_row["val_acc"] - s_row["val_acc"]),
                               100 * (f_row["val_macro_f1"] - s_row["val_macro_f1"])))
                if f_row and t_row and r_row:
                    acc_deltas.append((d, s,
                                       100 * (r_row["val_acc"] - t_row["val_acc"]),
                                       100 * (r_row["val_macro_f1"] - t_row["val_macro_f1"])))
            if ds:
                acc_deltas_mean = statistics.fmean([v[0] for v in ds])
                f1_deltas.append((d, acc_deltas_mean,
                                  statistics.fmean([v[1] for v in ds])))
        # acc_deltas: (d, s, drop_top - drop_random acc, f1)
        drop = {}
        for d in ("Movies", "Toys", "Grocery"):
            vals = [v for v in acc_deltas if v[0] == d]
            if vals:
                drop[d] = {"acc": statistics.fmean(v[2] for v in vals),
                           "f1": statistics.fmean(v[3] for v in vals)}
        full_minus_shuffle_acc = statistics.fmean(v[1] for v in f1_deltas) if f1_deltas else 0.0
        full_minus_shuffle_f1 = statistics.fmean(v[2] for v in f1_deltas) if f1_deltas else 0.0
        drop_acc = statistics.fmean(v["acc"] for v in drop.values()) if drop else 0.0
        drop_f1 = statistics.fmean(v["f1"] for v in drop.values()) if drop else 0.0
        n_pos = sum(1 for v in f1_deltas if v[1] > 0)
        return {"full_minus_shuffle_acc": full_minus_shuffle_acc,
                "full_minus_shuffle_f1": full_minus_shuffle_f1,
                "drop_top_minus_random_acc": drop_acc,
                "drop_top_minus_random_f1": drop_f1,
                "n_pos_ds": n_pos}

    def _verdict(st):
        cond1 = st["full_minus_shuffle_acc"] >= 0.30 and st["full_minus_shuffle_f1"] >= 0.0
        cond2 = st["drop_top_minus_random_acc"] >= 0.20 and st["drop_top_minus_random_f1"] >= 0.0 \
            and st["n_pos_ds"] >= 2
        if cond1 or cond2:
            return "IDENTITY SUPPORTED"
        cond_w = (st["full_minus_shuffle_acc"] >= 0.10
                  or st["drop_top_minus_random_acc"] >= 0.10)
        return "IDENTITY WEAK" if cond_w else "IDENTITY NOT SUPPORTED"

    v_pair, v_tfo = _verdict(_papermodel("PAIR_EDGE")), _verdict(_papermodel("TARGET_FACTOR_ONLY"))
    lines = [
        "# R2D28_REPAIR_REPORT — D2.8-A repaired neighbor-identity attribution",
        "",
        f"- PAIR_EDGE: **{v_pair}**",
        f"- TARGET_FACTOR_ONLY: **{v_tfo}**",
        "",
        "Verdict criteria (v2 §6): SUPPORTED if FULL-SHUFFLE >= +0.30pp Acc"
        " macro (M/T/G) or DROP_top10-DROP_random10 >= +0.20pp Acc with F1"
        " nonnegative and >=2/3 dataset means positive; WEAK if consistent"
        " +0.10..+0.30pp; otherwise NOT SUPPORTED. The old shuffle=0"
        " conclusion is not carried forward.",
        "",
        "| model | FULL-SHUFFLE Acc (pp) | FULL-SHUFFLE F1 (pp) |"
        " DROP_top-random Acc (pp) | DROP_top-random F1 (pp) | pos ds | verdict |",
        "|---|---|---|---|---|---|---|",
        f"| PAIR_EDGE | {_papermodel('PAIR_EDGE')['full_minus_shuffle_acc']:+.3f} |"
        f" {_papermodel('PAIR_EDGE')['full_minus_shuffle_f1']:+.3f} |"
        f" {_papermodel('PAIR_EDGE')['drop_top_minus_random_acc']:+.3f} |"
        f" {_papermodel('PAIR_EDGE')['drop_top_minus_random_f1']:+.3f} |"
        f" {_papermodel('PAIR_EDGE')['n_pos_ds']}/3 | {v_pair} |",
        f"| TARGET_FACTOR_ONLY | {_papermodel('TARGET_FACTOR_ONLY')['full_minus_shuffle_acc']:+.3f} |"
        f" {_papermodel('TARGET_FACTOR_ONLY')['full_minus_shuffle_f1']:+.3f} |"
        f" {_papermodel('TARGET_FACTOR_ONLY')['drop_top_minus_random_acc']:+.3f} |"
        f" {_papermodel('TARGET_FACTOR_ONLY')['drop_top_minus_random_f1']:+.3f} |"
        f" {_papermodel('TARGET_FACTOR_ONLY')['n_pos_ds']}/3 | {v_tfo} |",
        "",
        "Raw per-run rows: `repair_results.csv`; shuffle validation:"
        " `repair_shuffle_validation.csv`; removal counts: `repair_removal.csv`.",
        "",
    ]
    (REPAIR_ROOT / "R2D28_REPAIR_REPORT.md").write_text("\n".join(lines),
                                                        encoding="utf-8")
    print(f"[summarize] {v_pair} / {v_tfo}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2D2.8 v2 D2.8-A repair")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.dataset, args.variant, args.seed, Path(args.outdir),
                   args.force)
        return
    if args.summarize:
        summarize()
        return

    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    models = list(MODELS) if not args.models else [m for m in args.models.split(",")]
    seeds = [42, 43, 44] if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, m, s) for d in datasets for m in models for s in seeds]
    launch_jobs(Path(__file__).resolve(), jobs, REPAIR_ROOT, gpus, args.force)


if __name__ == "__main__":
    main()
