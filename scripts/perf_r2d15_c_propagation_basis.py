"""R2-Design-1.5 D1.5-C: Frozen-B0 propagation-basis audit (plan §11-§17).

Frozen Ridge probes (StandardScaler + RidgeClassifier(alpha=1.0), fit TRAIN
/ eval VAL) over the B0 formal checkpoints:

    per-factor matched 2d : [F^a|H1^a] / [F^a|H2^a] / [F^a|HP^a]
    joint matched 6d      : [L|H1_all] / [L|H2_all] / [L|HP_all]
    final-residual matched: [Z|H1_all] / [Z|H2_all] / [Z|HP_all]   (Z = z_B0)
    multi-scale upper bound: [L|H1_all|H2_all|HP_all]  (15d, upper bound only)
    shuffle control        : fixed perm seed=20260904 on H2/HP rows

H1 = P·H0 (== B0 neighbor context), H2 = P·H1, HP = H0 − H1 (plan §12).
GO rules (plan §17): 2-hop factor GO / 2-hop final GO / high-pass GO.
Primary verdict on Movies/Toys/Grocery x seeds 42/43/44 only. Val only.

Usage:
    python scripts/perf_r2d15_c_propagation_basis.py --gpu 0 [--include-guards]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d15_utils import (  # noqa: E402
    FACTOR_NAMES,
    GUARD_DATASETS,
    R2D15_ROOT,
    SEEDS,
    TARGET_DATASETS,
    assert_no_test_access,
    extract_b0_states,
    fixed_node_permutation,
    load_frozen_r2_checkpoint,
    propagation_signals,
)
from src.analysis.perf_r0_utils import ridge_probe, write_csv  # noqa: E402

B0_CONFIRM = R2D15_ROOT / "b0_confirm"
OUTDIR = R2D15_ROOT / "propagation"


def _probe(features: torch.Tensor, data: object, device: torch.device) -> dict:
    return ridge_probe(features, SimpleNamespace(data=data, device=device))


def main() -> None:
    parser = argparse.ArgumentParser(description="D1.5-C frozen-B0 propagation basis audit")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--include-guards", action="store_true")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    datasets = list(TARGET_DATASETS)
    if args.include_guards:
        datasets += GUARD_DATASETS

    factor_rows: list[dict] = []
    joint_rows: list[dict] = []
    final_rows: list[dict] = []
    upper_rows: list[dict] = []
    shuffle_rows: list[dict] = []

    for ds in datasets:
        for seed in SEEDS:
            setup = load_frozen_r2_checkpoint(ds, seed, "B0", device, root=B0_CONFIRM)
            assert_no_test_access(setup.data)
            x = setup.data.x.to(device)
            ei = setup.data.edge_index.to(device)
            states = extract_b0_states(setup.model, x, ei)
            f_pre, z = states["f_pre"], states["z"]
            h1, h2, hp = propagation_signals(setup.model, f_pre, ei, int(x.size(0)))

            l_cat = f_pre.reshape(f_pre.size(0), -1)  # L = [C|Pt|Pv], 3d
            h1_cat = h1.reshape(h1.size(0), -1)
            h2_cat = h2.reshape(h2.size(0), -1)
            hp_cat = hp.reshape(hp.size(0), -1)

            # --- per-factor matched 2d (plan §13) -------------------------
            for a, name in enumerate(FACTOR_NAMES):
                f_a = f_pre[:, a]
                probes = {
                    "H1": _probe(torch.cat([f_a, h1[:, a]], dim=-1), setup.data, device),
                    "H2": _probe(torch.cat([f_a, h2[:, a]], dim=-1), setup.data, device),
                    "HP": _probe(torch.cat([f_a, hp[:, a]], dim=-1), setup.data, device),
                }
                for variant, p in probes.items():
                    factor_rows.append({
                        "dataset": ds, "seed": seed, "factor": name, "variant": variant,
                        "val_acc": p["val_acc"], "val_macro_f1": p["val_macro_f1"],
                    })
                print(f"[factor] {ds} s{seed} {name}: "
                      f"H1={probes['H1']['val_acc']:.4f} H2={probes['H2']['val_acc']:.4f} "
                      f"HP={probes['HP']['val_acc']:.4f}", flush=True)

            # --- joint matched 6d (plan §14) ------------------------------
            for label, cat in (("H1", h1_cat), ("H2", h2_cat), ("HP", hp_cat)):
                p = _probe(torch.cat([l_cat, cat], dim=-1), setup.data, device)
                joint_rows.append({
                    "dataset": ds, "seed": seed, "variant": label,
                    "val_acc": p["val_acc"], "val_macro_f1": p["val_macro_f1"],
                })

            # --- final-residual matched (plan §15) ------------------------
            for label, cat in (("H1", h1_cat), ("H2", h2_cat), ("HP", hp_cat)):
                p = _probe(torch.cat([z, cat], dim=-1), setup.data, device)
                final_rows.append({
                    "dataset": ds, "seed": seed, "variant": label,
                    "val_acc": p["val_acc"], "val_macro_f1": p["val_macro_f1"],
                })
                print(f"[final] {ds} s{seed} {label}: acc={p['val_acc']:.4f}", flush=True)

            # --- multi-scale upper bound (plan §16, upper bound only) ------
            p_ms = _probe(torch.cat([l_cat, h1_cat, h2_cat, hp_cat], dim=-1), setup.data, device)
            upper_rows.append({
                "dataset": ds, "seed": seed,
                "val_acc": p_ms["val_acc"], "val_macro_f1": p_ms["val_macro_f1"],
            })

            # --- fixed permutation control (plan §16): shuffle H2/HP rows --
            perm = fixed_node_permutation(int(x.size(0)))
            for label, cat in (("H2", h2_cat), ("HP", hp_cat)):
                p_shuf = _probe(torch.cat([l_cat, cat[perm]], dim=-1), setup.data, device)
                shuffle_rows.append({
                    "dataset": ds, "seed": seed, "variant": label,
                    "val_acc": p_shuf["val_acc"], "val_macro_f1": p_shuf["val_macro_f1"],
                })
            del setup, x, ei, f_pre, z, h1, h2, hp
            torch.cuda.empty_cache()

    write_csv(OUTDIR / "propagation_factor_probe.csv", factor_rows)
    write_csv(OUTDIR / "propagation_joint_probe.csv", joint_rows)
    write_csv(OUTDIR / "propagation_final_probe.csv", final_rows)
    write_csv(OUTDIR / "propagation_upper_bound.csv", upper_rows)
    write_csv(OUTDIR / "propagation_shuffle.csv", shuffle_rows)

    # ---------------- verdicts (plan §17) ---------------------------------
    def seed_mean_delta(rows: list[dict], variant_a: str, variant_b: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in rows:
            key = (row["dataset"], row["seed"])
            if row["variant"] == variant_a:
                out.setdefault(key, [0.0, 0.0])[0] = row["val_acc"]
            elif row["variant"] == variant_b:
                out.setdefault(key, [0.0, 0.0])[1] = row["val_acc"]
        return {
            ds: statistics.mean(a - b for (d, s), (a, b) in out.items() if d == ds and a is not None)
            for ds in TARGET_DATASETS
        }

    def seed_positivity(rows: list[dict], variant_a: str, variant_b: str) -> dict[str, int]:
        deltas: dict[str, list[float]] = {}
        for row in rows:
            if row["variant"] not in (variant_a, variant_b):
                continue
            key = (row["dataset"], row["seed"], row["variant"])
            deltas.setdefault((row["dataset"], row["seed"]), [None, None])
            idx = 0 if row["variant"] == variant_a else 1
            deltas[(row["dataset"], row["seed"])][idx] = row["val_acc"]
        out: dict[str, int] = {}
        for ds in TARGET_DATASETS:
            pos = 0
            for (d, s), (a, b) in deltas.items():
                if d == ds and a is not None and b is not None and a - b > 0:
                    pos += 1
            out[ds] = pos
        return out

    # factor-level 2-hop / HP (plan §17)
    factor_go_2hop = {}
    factor_go_hp = {}
    for a in FACTOR_NAMES:
        sub = [r for r in factor_rows if r["factor"] == a]
        d21 = seed_mean_delta(sub, "H2", "H1")
        dhp1 = seed_mean_delta(sub, "HP", "H1")
        pos21 = seed_positivity(sub, "H2", "H1")
        poshp = seed_positivity(sub, "HP", "H1")
        ds21 = [ds for ds, v in d21.items() if v >= 0.30 / 100 and pos21[ds] >= 2]
        dshp = [ds for ds, v in dhp1.items() if v >= 0.30 / 100 and poshp[ds] >= 2]
        factor_go_2hop[a] = (d21, len(ds21) >= 2)
        factor_go_hp[a] = (dhp1, len(dshp) >= 2)

    # final-level (plan §17)
    f21 = seed_mean_delta(final_rows, "H2", "H1")
    fhp1 = seed_mean_delta(final_rows, "HP", "H1")
    f21_pos = seed_positivity(final_rows, "H2", "H1")
    fhp1_pos = seed_positivity(final_rows, "HP", "H1")
    final_go_2hop = (
        statistics.mean(f21.values()) >= 0.20 / 100
        and sum(1 for ds in TARGET_DATASETS if f21[ds] > 0) >= 2
    )
    final_go_hp = (
        statistics.mean(fhp1.values()) >= 0.20 / 100
        and sum(1 for ds in TARGET_DATASETS if fhp1[ds] > 0) >= 2
    )

    lines = [
        "# R2D15_PROPAGATION_REPORT — D1.5-C Frozen-B0 Propagation Basis Audit",
        "",
        "Frozen Ridge probes on B0 formal checkpoints; fit TRAIN / eval VAL; "
        "primary verdict Movies/Toys/Grocery x seeds 42/43/44. Val only.",
        "",
        "## Per-factor Δ(2−1) / Δ(HP−1) seed-mean (pp)",
        "",
        "| factor | Movies | Toys | Grocery | 2-hop GO? | HP GO? |",
        "|---|---:|---:|---:|---|---|",
    ]
    for a in FACTOR_NAMES:
        d21, go21 = factor_go_2hop[a]
        dhp1, gohp = factor_go_hp[a]
        lines.append(
            f"| {a} | {d21['Movies']*100:+.3f} / {dhp1['Movies']*100:+.3f} "
            f"| {d21['Toys']*100:+.3f} / {dhp1['Toys']*100:+.3f} "
            f"| {d21['Grocery']*100:+.3f} / {dhp1['Grocery']*100:+.3f} "
            f"| {'YES' if go21 else 'no'} | {'YES' if gohp else 'no'} |"
        )
    lines += [
        "",
        f"## Final-residual Δ (pp): 2-hop M/T/G mean = "
        f"{statistics.mean(f21.values())*100:+.3f} ({ {k: round(v*100,2) for k,v in f21.items()} }) "
        f"-> {'**2-hop FINAL GO**' if final_go_2hop else 'no'}",
        f"## Final-residual Δ (pp): high-pass M/T/G mean = "
        f"{statistics.mean(fhp1.values())*100:+.3f} ({ {k: round(v*100,2) for k,v in fhp1.items()} }) "
        f"-> {'**HP FINAL GO**' if final_go_hp else 'no'}",
        "",
        "## Verdict summary",
        "",
        f"- 2-hop FACTOR GO: {any(go for _, go in factor_go_2hop.values())} "
        f"({ {a: go for a, (_, go) in factor_go_2hop.items()} })",
        f"- 2-hop FINAL GO: {final_go_2hop}",
        f"- high-pass FACTOR GO: {any(go for _, go in factor_go_hp.values())} "
        f"({ {a: go for a, (_, go) in factor_go_hp.items()} })",
        f"- high-pass FINAL GO: {final_go_hp}",
        "",
        "Upper bound and shuffle controls are recorded in "
        "propagation_upper_bound.csv / propagation_shuffle.csv (upper bound "
        "never counts as GO evidence, plan §16).",
    ]
    (OUTDIR / "R2D15_PROPAGATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[propagation] saved -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
