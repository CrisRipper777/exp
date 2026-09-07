"""R1.5 selection helper (plan §14/§18): rank variants by the M/T/G mean
Val-Acc delta vs the fresh A0 anchor, pick Top-2, decide the stage winner.

Usage:
    python scripts/perf_r15_select.py --family opt --top 2 --out outputs/perf_r15/opt/OPT_SELECTION.json
    python scripts/perf_r15_select.py --family capacity --top 2 --out ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEAK = ["Movies", "Toys", "Grocery"]
ANCHOR_ROOT = PROJECT_ROOT / "outputs" / "perf_r15" / "anchor"
SEED = 42


def _val_acc(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    with summary_path.open(encoding="utf-8") as f:
        s = json.load(f)
    return (s.get("results") or {}).get("val_acc", {}).get("mean")


def main() -> None:
    parser = argparse.ArgumentParser(description="R1.5 variant selection")
    parser.add_argument("--family", required=True)
    parser.add_argument("--variants", required=True,
                        help="comma-separated variant labels (dataset dirs live under the family root)")
    parser.add_argument("--top", type=int, default=2)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-gain", type=float, default=0.15)
    args = parser.parse_args()

    family_root = PROJECT_ROOT / "outputs" / "perf_r15" / args.family
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    anchor = {d: _val_acc(ANCHOR_ROOT / d / "A0" / f"seed_{SEED}" / "summary.json") for d in WEAK}
    scored = []
    for variant in variants:
        deltas = []
        ok = True
        for d in WEAK:
            a0 = anchor[d]
            v = _val_acc(family_root / d / variant / f"seed_{SEED}" / "summary.json")
            if a0 is None or v is None:
                ok = False
                break
            deltas.append(100.0 * (v - a0))
        if not ok:
            continue
        scored.append({
            "variant": variant,
            "score_MTG": sum(deltas) / len(deltas),
            "deltas": {d: round(dt, 4) for d, dt in zip(WEAK, deltas)},
            "positive_count": sum(1 for dt in deltas if dt > 0),
        })
    scored.sort(key=lambda r: -r["score_MTG"])
    top = [r["variant"] for r in scored[: args.top]]
    winner = None
    if scored and scored[0]["score_MTG"] >= args.min_gain:
        winner = scored[0]["variant"]
    out = {
        "family": args.family,
        "ranking": scored,
        "top_k": top,
        "winner": winner,
        "note": (f"winner requires mean M/T/G gain >= +{args.min_gain}pp; "
                 f"None => keep A0 baseline"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for r in scored:
        print(f"[select] {r['variant']:16s} score_MTG={r['score_MTG']:+.3f}pp "
              f"pos={r['positive_count']}/3 deltas={r['deltas']}", flush=True)
    print(f"[select] top_k={top} winner={winner} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
