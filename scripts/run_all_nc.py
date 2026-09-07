from __future__ import annotations

import subprocess
import sys

DATASETS = ["Movies", "Toys", "Grocery", "Reddit-S", "ele-fashion", "books-nc"]
# RPTA-style full-graph NC protocol; per-model official presets live in
# configs/model/*.yaml (lr / weight_decay / architecture).
MODELS = ["mlp", "gcn", "sage", "mmgcn", "mgat", "dgf", "dmgc", "dip"]
# For books-nc (685K nodes) use sampled training to avoid OOM:
#   task.training_mode=sampled
SAMPLED_DATASETS = {"books-nc"}


def main() -> None:
    for dataset in DATASETS:
        for model in MODELS:
            cmd = [sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc", f"model={model}"]
            if dataset in SAMPLED_DATASETS:
                cmd.append("task.training_mode=sampled")
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
