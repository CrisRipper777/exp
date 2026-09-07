"""P1 confirm driver: 5 NC datasets x 4 variants x seeds 42/43/44 = 60 runs
(plan §31). Thin wrapper around the screen driver with --stage confirm.

Usage:
    python scripts/run_p1_confirm.py --gpus 0,1
    python scripts/run_p1_confirm.py --datasets Movies --force
"""

from __future__ import annotations

import sys

from run_p1_screen import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--stage", "confirm"]
    main()
