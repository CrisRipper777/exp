"""R2-D2.6-E: deep-supervision confirmation
(docs/BiAxis_R2_Design_2_6_Strong_Parent_Readout_Integration.md §40).

For the final top-1 candidate: lambda_aux = 0 vs 0.1 on Movies/Toys/
Grocery x seeds 42/43/44. The lambda=0.1 runs are the D2.6-B integration
runs (reused); only lambda=0 is trained here (into deep_supervision/).

Usage:
    python scripts/perf_r2d26_deepsup.py --variant FHC_HOP --gpus 0,1
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perf_r2d26_integration import main  # noqa: E402
from src.analysis.perf_r2d26_utils import R2D26_ROOT  # noqa: E402


def _main() -> None:
    argv = sys.argv[1:]
    if not any(a.startswith("--variants") for a in argv):
        argv = ["--variants", "FHC_HOP"] + argv
    if not any(a.startswith("--out-root") for a in argv):
        argv = argv + ["--out-root", str(R2D26_ROOT / "deep_supervision")]
    if not any(a.startswith("--deep-sup") for a in argv):
        argv = argv + ["--deep-sup", "0"]
    main(argv)


if __name__ == "__main__":
    _main()
