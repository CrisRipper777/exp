"""R2-0 audit (plan §3): reproduce R0 fixed-probe results from frozen OFR
checkpoints and verify the no-test discipline on current main.

Stage smoke (Movies seed42, plan §3.1):
    Probe([C|Pt|Pv]), Probe(z_local), Probe(z_final)
    vs outputs/perf_r0/factor/factor_probe_per_seed.csv.

Stage full (plan §3.1, M/T/G x 42/43/44):
    all 13 R0 factor representations (same fixed Ridge probe) +
    head val-acc on z_final (vs R0 checkpoint_audit.csv recorded val).

Verdict per row: PASS (abs diff <= 1e-5), PASS_TOL (<= 1e-4, plan §3.1
environment tolerance), FAIL otherwise. Any FAIL in smoke halts the run
(plan §3.2: Audit FAIL -> stop, no R2-0A/B/C).

Usage:
    python scripts/perf_r20_audit.py --gpus 0,1            # smoke then full
    python scripts/perf_r20_audit.py --gpus 0,1 --stage smoke
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r0_utils import val_metrics_with_head  # noqa: E402
from src.analysis.perf_r20_utils import (  # noqa: E402
    DATASETS,
    SEEDS,
    extract_forward,
    factor_tensor,
    load_setup,
    ridge_probe,
    write_csv,
)

OUT_ROOT = PROJECT_ROOT / "outputs" / "perf_r20" / "audit"
R0_FACTOR_CSV = PROJECT_ROOT / "outputs" / "perf_r0" / "factor" / "factor_probe_per_seed.csv"
R0_AUDIT_CSV = PROJECT_ROOT / "outputs" / "perf_r0" / "audit" / "checkpoint_audit.csv"

SMOKE_REPS = ["C|Pt|Pv", "z_local", "z_final"]
# The 13 representations probed by R0-D2 (perf_r0_factor.REPRESENTATIONS).
FULL_REPS = [
    "h_t", "h_v", "h_t|h_v",
    "C", "Pt", "Pv", "C|Pt|Pv",
    "z_local",
    "C'", "Pt'", "Pv'", "C'|Pt'|Pv'",
    "z_final",
]

STRICT_TOL = 1e-5
SOFT_TOL = 1e-4


def _representation_tensor(fex: dict, name: str) -> torch.Tensor:
    factors, f_tilde = fex["factors"], fex["f_tilde"]
    if name == "h_t":
        return factors["h_t"]
    if name == "h_v":
        return factors["h_v"]
    if name == "h_t|h_v":
        return torch.cat([factors["h_t"], factors["h_v"]], dim=-1)
    if name in ("C", "Pt", "Pv"):
        return factor_tensor(fex, name)
    if name == "C|Pt|Pv":
        return torch.cat([factor_tensor(fex, n) for n in ("C", "Pt", "Pv")], dim=-1)
    if name == "z_local":
        return fex["z_local"]
    if name in ("C'", "Pt'", "Pv'"):
        return f_tilde[:, ("C'", "Pt'", "Pv'").index(name)]
    if name == "C'|Pt'|Pv'":
        return f_tilde.reshape(f_tilde.size(0), -1)
    if name == "z_final":
        return fex["z_final"]
    raise KeyError(name)


def _load_references() -> dict:
    """(dataset, seed, representation) -> {val_acc, val_macro_f1} from R0."""
    refs: dict = {}
    with R0_FACTOR_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            refs[(row["dataset"], int(row["seed"]), row["representation"])] = {
                "val_acc": float(row["val_acc"]),
                "val_macro_f1": float(row["val_macro_f1"]),
            }
    return refs


def _load_recorded_val() -> dict:
    """(dataset, seed) -> recorded checkpoint val acc (R0 audit table)."""
    rec: dict = {}
    with R0_AUDIT_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec[(row["dataset"], int(row["seed"]))] = float(row["checkpoint_val_acc"])
    return rec


def _verdict(max_abs: float) -> str:
    if max_abs <= STRICT_TOL:
        return "PASS"
    if max_abs <= SOFT_TOL:
        return "PASS_TOL"
    return "FAIL"


def _probe_row(setup, fex, dataset: str, seed: int, name: str,
               ref_val: float | None, ref_f1: float | None, stage: str) -> dict:
    tensor = _representation_tensor(fex, name)
    probe = ridge_probe(tensor, setup)
    del tensor
    diffs = []
    row = {
        "stage": stage, "dataset": dataset, "seed": seed, "representation": name,
        "val_acc": probe["val_acc"], "val_macro_f1": probe["val_macro_f1"],
    }
    if ref_val is not None:
        row["ref_val_acc"] = ref_val
        row["abs_diff_acc"] = abs(probe["val_acc"] - ref_val)
        diffs.append(row["abs_diff_acc"])
    else:
        row["ref_val_acc"] = ""
        row["abs_diff_acc"] = ""
    if ref_f1 is not None:
        row["ref_val_f1"] = ref_f1
        row["abs_diff_f1"] = abs(probe["val_macro_f1"] - ref_f1)
        diffs.append(row["abs_diff_f1"])
    else:
        row["ref_val_f1"] = ""
        row["abs_diff_f1"] = ""
    row["verdict"] = _verdict(max(diffs)) if diffs else "NO_REF"
    print(
        f"[audit] {stage:5s} {dataset:12s} s{seed} {name:10s} "
        f"val={probe['val_acc']:.6f} "
        f"diff={row['abs_diff_acc'] if row['abs_diff_acc'] != '' else 'n/a'} "
        f"-> {row['verdict']}",
        flush=True,
    )
    return row


def _head_row(setup, fex, dataset: str, seed: int, recorded_val: float, stage: str) -> dict:
    """Head val-acc on z_final vs the checkpoint's recorded training val."""
    metric = val_metrics_with_head(setup, fex["z_final"])
    diff = abs(metric["val_acc"] - recorded_val)
    row = {
        "stage": stage, "dataset": dataset, "seed": seed, "representation": "head_z_final",
        "val_acc": metric["val_acc"], "val_macro_f1": "",
        "ref_val_acc": recorded_val, "abs_diff_acc": diff,
        "ref_val_f1": "", "abs_diff_f1": "", "verdict": _verdict(diff),
    }
    print(
        f"[audit] {stage:5s} {dataset:12s} s{seed} head_z_final "
        f"val={metric['val_acc']:.6f} diff={diff:.3e} -> {row['verdict']}",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-0 audit reproduction (plan §3)")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--stage", default="all", choices=["smoke", "full", "all"])
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    refs = _load_references()
    recorded = _load_recorded_val()
    rows: list[dict] = []
    smoke_failed = False

    if args.stage in ("smoke", "all"):
        device = torch.device(f"cuda:{gpus[0]}")
        setup = load_setup("Movies", 42, device)
        fex = extract_forward(setup)
        for name in SMOKE_REPS:
            ref = refs[("Movies", 42, name)]
            row = _probe_row(setup, fex, "Movies", 42, name, ref["val_acc"], ref["val_macro_f1"], "smoke")
            rows.append(row)
            smoke_failed = smoke_failed or row["verdict"] == "FAIL"
        del fex
        torch.cuda.empty_cache()

    if args.stage == "all" and smoke_failed:
        write_csv(OUT_ROOT / "reproduction.csv", rows)
        print("[audit] SMOKE FAIL — halting, no full stage (plan §3.2).", flush=True)
        sys.exit(1)

    if args.stage in ("full", "all"):
        lifecycles = [(ds, s) for ds in DATASETS for s in SEEDS]
        for idx, (ds, s) in enumerate(lifecycles):
            device = torch.device(f"cuda:{gpus[idx % len(gpus)]}")
            setup = load_setup(ds, s, device)
            fex = extract_forward(setup)
            for name in FULL_REPS:
                ref = refs.get((ds, s, name))
                row = _probe_row(
                    setup, fex, ds, s, name,
                    ref["val_acc"] if ref else None,
                    ref["val_macro_f1"] if ref else None,
                    "full",
                )
                rows.append(row)
            rows.append(_head_row(setup, fex, ds, s, recorded[(ds, s)], "full"))
            del fex
            torch.cuda.empty_cache()

    write_csv(OUT_ROOT / "reproduction.csv", rows)
    fails = [r for r in rows if r["verdict"] == "FAIL"]
    tols = [r for r in rows if r["verdict"] == "PASS_TOL"]
    print(
        f"[audit] done -> {OUT_ROOT / 'reproduction.csv'} "
        f"(rows={len(rows)}, FAIL={len(fails)}, PASS_TOL={len(tols)})",
        flush=True,
    )
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
