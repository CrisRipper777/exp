from __future__ import annotations

import subprocess
import sys

DATASETS = ["Movies", "Toys", "Grocery", "Reddit-S", "sports-copurchase", "cloth-copurchase", "books-lp"]
# RPTA-style sampled LP protocol. LP-side official presets that differ from
# the NC-side model configs are passed as CLI overrides (mirrors RPTA's
# per-runner preset dicts in run_nc_main_table.py).
MODELS = ["gcn", "sage", "mmgcn", "mgat", "dgf", "dmgc"]
LP_MODEL_OVERRIDES = {
    # RPTA LP presets (link_prediction.py _encoder_preset): the LP encoders
    # have no dropout.
    "mmgcn": ["model.dropout=0.0"],
    "mgat": ["model.dropout=0.0"],
}


def main() -> None:
    for dataset in DATASETS:
        for model in MODELS:
            cmd = [sys.executable, "-m", "src.main", f"dataset={dataset}", "task=lp", f"model={model}"]
            cmd += LP_MODEL_OVERRIDES.get(model, [])
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
