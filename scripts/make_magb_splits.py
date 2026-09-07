from __future__ import annotations

import subprocess
import sys


DATASETS = ["Movies", "Toys", "Grocery", "Reddit-S"]


def main() -> None:
    for dataset in DATASETS:
        for task in ["nc", "lp"]:
            cmd = [
                sys.executable,
                "-m",
                "src.main",
                f"dataset={dataset}",
                f"task={task}",
                "model=mlp",
                "num_runs=1",
                "task.epochs=0",
                "device=cpu",
            ]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
