"""R2-D2.6-A: no-compression strong-parent diagnosis
(docs/BiAxis_R2_Design_2_6_Strong_Parent_Readout_Integration.md §8-§13).

A0_BASE / NC_HOP / NC_H1 on all 5 datasets x seeds 42/43/44. A0 frozen;
[z_base | 9 expert tokens] fed to the classifier WITHOUT projection back
to 256. NC_H1 = architecture-identical H1-only control.

Reuses the D2.6-B driver (scripts/perf_r2d26_integration.py) with the
NC variant set and the no_compression/ output root.

Usage:
    python scripts/perf_r2d26_no_compression.py --gpus 0,1
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perf_r2d26_integration import main  # noqa: E402
from src.analysis.perf_r2d26_utils import R2D26_ROOT  # noqa: E402

NC_VARIANTS = ("A0_BASE", "NC_HOP", "NC_H1")


def _main() -> None:
    argv = sys.argv[1:]
    if not any(a.startswith("--variants") for a in argv):
        argv = ["--variants", ",".join(NC_VARIANTS)] + argv
    if not any(a.startswith("--out-root") for a in argv):
        argv = argv + ["--out-root", str(R2D26_ROOT / "no_compression")]
    main(argv)


if __name__ == "__main__":
    _main()
