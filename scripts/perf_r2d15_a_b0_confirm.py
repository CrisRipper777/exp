"""R2-Design-1.5 D1.5-A: B0 formal confirmation (plan §5).

1. Copies the existing seed42 B0 runs (same commit/config — verified in the
   D1.5-0 audit) into outputs/perf_r2d15/b0_confirm/ instead of retraining.
2. Runs all missing runs via run_perf_r2d1.py (--out-root) on both GPUs:
   5 datasets x seeds 42/43/44, Val only, evaluate_test=false.

Usage:
    python scripts/perf_r2d15_a_b0_confirm.py --gpus 0,1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d15_utils import DATASETS, R2D15_ROOT, R2D1_ROOT  # noqa: E402

SEEDS = [42, 43, 44]
B0_CONFIRM = R2D15_ROOT / "b0_confirm"

# Files that make a run reusable; seed42 B0 runs must carry all of them.
RUN_FILES = ("summary.json", "model.pt", "history.csv", "train.log", "r2_diagnostics.json", "diag.log")


def _copy_seed42() -> list[str]:
    copied = []
    for dataset in DATASETS:
        src = R2D1_ROOT / "b0" / dataset / "B0" / "seed_42"
        dst = B0_CONFIRM / dataset / "B0" / "seed_42"
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for name in RUN_FILES:
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        copied.append(dataset)
        print(f"[copy] seed42 B0 {dataset} -> {dst}", flush=True)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="D1.5-A B0 formal confirmation driver")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()

    copied = _copy_seed42()

    # Missing runs: everything not already present (seed42 kept where copied).
    missing: list[tuple[str, int]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            if dataset in copied and seed == 42:
                continue
            if (B0_CONFIRM / dataset / "B0" / f"seed_{seed}" / "summary.json").exists():
                continue
            missing.append((dataset, seed))
    if not missing:
        print("[b0-confirm] nothing to run — all 15 runs present", flush=True)
        return

    datasets_csv = ",".join(dict.fromkeys(d for d, _ in missing))
    seeds_csv = ",".join(sorted({str(s) for _, s in missing}))
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_perf_r2d1.py"),
        "--datasets", datasets_csv,
        "--variants", "B0",
        "--seeds", seeds_csv,
        "--gpus", args.gpus,
        "--out-root", str(B0_CONFIRM),
    ]
    print(f"[b0-confirm] running {len(missing)} missing runs: {missing}", flush=True)
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"driver failed rc={proc.returncode}")
    print("[b0-confirm] done", flush=True)


if __name__ == "__main__":
    main()
