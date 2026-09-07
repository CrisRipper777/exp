"""P2 confirm driver (plan §48). Thin wrapper around run_p2_screen.

Usage:
    python scripts/run_p2_confirm.py --gpus 0,1 --modes null_softmax,adaptive_uot
"""

from __future__ import annotations

import sys

from run_p2_screen import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--stage", "confirm"]
    main()
