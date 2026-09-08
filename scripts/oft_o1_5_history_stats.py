"""O1.5 offline history diagnostics over the per-epoch NC history CSVs.

For each CSV prints:
  * best-Val-Acc epoch row (epoch / val_acc / val_macro_f1)
  * best-Val-Macro-F1 epoch row (epoch / val_macro_f1 / val_acc)
  * first / mean / min / max / last of the P0 factor diagnostics columns
    (common_sim, private_sim, c_norm, pt_norm, pv_norm, cp_overlap_t,
    cp_overlap_v)

Usage:
    python scripts/oft_o1_5_history_stats.py CSV [CSV ...]
"""

from __future__ import annotations

import csv
import sys


def stats(col: list[float]) -> tuple[float, float, float, float, float]:
    vals = [v for v in col if v is not None]
    return vals[0], sum(vals) / len(vals), min(vals), max(vals), vals[-1]


def main() -> None:
    diag_cols = ["common_sim", "private_sim", "c_norm", "pt_norm", "pv_norm",
                 "cp_overlap_t", "cp_overlap_v"]
    for path in sys.argv[1:]:
        rows = []
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append({k: (float(row[k]) if row.get(k) not in ("", None) else None)
                             for k in row})
        print(f"\n===== {path} ({len(rows)} epochs) =====")
        best_acc = max(rows, key=lambda r: r["val_acc"])
        best_f1 = max(rows, key=lambda r: r["val_macro_f1"])
        print(f"best-Acc epoch {int(best_acc['epoch'])} | acc {best_acc['val_acc']:.6f} "
              f"| F1 {best_acc['val_macro_f1']:.6f}")
        print(f"best-F1 epoch {int(best_f1['epoch'])} | F1 {best_f1['val_macro_f1']:.6f} "
              f"| acc {best_f1['val_acc']:.6f}")
        for key in diag_cols:
            col = [r[key] for r in rows]
            first, mean, lo, hi, last = stats(col)
            print(f"{key:>14}: first {first:8.4f} | mean {mean:8.4f} | "
                  f"min {lo:8.4f} | max {hi:8.4f} | last {last:8.4f}")


if __name__ == "__main__":
    main()
