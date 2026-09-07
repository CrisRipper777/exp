"""R2-Design-1.6 D1.6-B: dual-parent frozen propagation audit (plan §12-§18).

Frozen Ridge probes on BOTH parents' own trained states:
    per-factor matched 2d : [H0|H1] / [H0|H2] / [H0|HP]
    joint matched 6d      : [L|H1_all] / [L|H2_all] / [L|HP_all]
    final residual        : [z_parent|H1_all] / [H2] / [HP]
    shuffle control       : fixed perm seed=20260904 on H2/HP rows

Within-parent deltas ONLY (never compare coordinates across parents, §13).
Verdicts (plan §18): cross-parent / parent-specific 2-hop & high-pass
SUPPORT; FINAL-RESIDUAL SUPPORT; INDUCTIVE-BIAS SUPPORT ONLY.
Primary = Movies/Toys/Grocery x seeds 42/43/44; guards secondary. Val only.

Usage:
    python scripts/perf_r2d16_b_propagation_dual_parent.py --gpu 1 [--guards]
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

from src.analysis.perf_r0_utils import ridge_probe, write_csv  # noqa: E402
from src.analysis.perf_r2d15_utils import (  # noqa: E402
    FACTOR_NAMES,
    fixed_node_permutation,
    propagation_signals,
)
from src.analysis.perf_r2d16_utils import (  # noqa: E402
    GUARD_DATASETS,
    PARENTS,
    R2D16_ROOT,
    SEEDS,
    TARGET_DATASETS,
    extract_parent_states,
    load_parent_setup,
)

OUTDIR = R2D16_ROOT / "propagation"


def _probe(features: torch.Tensor, data: object, device: torch.device) -> dict:
    return ridge_probe(features, SimpleNamespace(data=data, device=device))


def main() -> None:
    parser = argparse.ArgumentParser(description="D1.6-B dual-parent propagation audit")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--guards", action="store_true")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    datasets = list(TARGET_DATASETS) + (GUARD_DATASETS if args.guards else [])

    factor_rows: list[dict] = []
    joint_rows: list[dict] = []
    final_rows: list[dict] = []
    shuffle_rows: list[dict] = []

    for parent in PARENTS:
        for ds in datasets:
            for seed in SEEDS:
                setup = load_parent_setup(parent, ds, seed, device)
                x = setup.data.x.to(device)
                ei = setup.data.edge_index.to(device)
                states = extract_parent_states(setup, x, ei)
                f_pre, z = states["f_pre"], states["z"]
                h1, h2, hp = propagation_signals(setup.model, f_pre, ei, int(x.size(0)))

                l_cat = f_pre.reshape(f_pre.size(0), -1)
                h1_cat, h2_cat, hp_cat = (t.reshape(t.size(0), -1) for t in (h1, h2, hp))

                for a, name in enumerate(FACTOR_NAMES):
                    f_a = f_pre[:, a]
                    probes = {
                        "H1": _probe(torch.cat([f_a, h1[:, a]], dim=-1), setup.data, device),
                        "H2": _probe(torch.cat([f_a, h2[:, a]], dim=-1), setup.data, device),
                        "HP": _probe(torch.cat([f_a, hp[:, a]], dim=-1), setup.data, device),
                    }
                    for variant, p in probes.items():
                        factor_rows.append({
                            "parent": parent, "dataset": ds, "seed": seed,
                            "factor": name, "variant": variant,
                            "val_acc": p["val_acc"], "val_macro_f1": p["val_macro_f1"],
                        })

                for label, cat in (("H1", h1_cat), ("H2", h2_cat), ("HP", hp_cat)):
                    p = _probe(torch.cat([l_cat, cat], dim=-1), setup.data, device)
                    joint_rows.append({
                        "parent": parent, "dataset": ds, "seed": seed, "variant": label,
                        "val_acc": p["val_acc"], "val_macro_f1": p["val_macro_f1"],
                    })
                    pf = _probe(torch.cat([z, cat], dim=-1), setup.data, device)
                    final_rows.append({
                        "parent": parent, "dataset": ds, "seed": seed, "variant": label,
                        "val_acc": pf["val_acc"], "val_macro_f1": pf["val_macro_f1"],
                    })

                perm = fixed_node_permutation(int(x.size(0)))
                for label, cat in (("H2", h2_cat), ("HP", hp_cat)):
                    p = _probe(torch.cat([l_cat, cat[perm]], dim=-1), setup.data, device)
                    shuffle_rows.append({
                        "parent": parent, "dataset": ds, "seed": seed, "variant": label,
                        "val_acc": p["val_acc"], "val_macro_f1": p["val_macro_f1"],
                    })
                print(f"[prop] {parent} {ds} s{seed} done", flush=True)
                del setup, x, ei, states, h1, h2, hp
                torch.cuda.empty_cache()

    write_csv(OUTDIR / "propagation_factor.csv", factor_rows)
    write_csv(OUTDIR / "propagation_joint.csv", joint_rows)
    write_csv(OUTDIR / "propagation_final.csv", final_rows)
    write_csv(OUTDIR / "propagation_shuffle.csv", shuffle_rows)

    # ---------------- verdicts (plan §18) ---------------------------------
    def seed_mean_delta(rows, parent, variant_a, variant_b, factor=None) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in rows:
            if row["parent"] != parent or (factor and row.get("factor") != factor):
                continue
            key = (row["dataset"], row["seed"])
            if row["variant"] == variant_a:
                out.setdefault(key, [None, None])[0] = row["val_acc"]
            elif row["variant"] == variant_b:
                out.setdefault(key, [None, None])[1] = row["val_acc"]
        return {
            ds: statistics.mean(a - b for (d, s), (a, b) in out.items()
                                if d == ds and a is not None and b is not None)
            for ds in TARGET_DATASETS
        }

    def seed_positivity(rows, parent, variant_a, variant_b, factor=None) -> dict[str, int]:
        deltas: dict[str, list] = {}
        for row in rows:
            if row["parent"] != parent or (factor and row.get("factor") != factor):
                continue
            if row["variant"] not in (variant_a, variant_b):
                continue
            key = (row["dataset"], row["seed"])
            idx = 0 if row["variant"] == variant_a else 1
            deltas.setdefault(key, [None, None])[idx] = row["val_acc"]
        out: dict[str, int] = {}
        for ds in TARGET_DATASETS:
            out[ds] = sum(1 for (d, s), (a, b) in deltas.items()
                          if d == ds and a is not None and b is not None and a - b > 0)
        return out

    lines = [
        "# R2D16_PROPAGATION_REPORT — D1.6-B Dual-Parent Frozen Propagation Audit",
        "",
        "Frozen Ridge probes (StandardScaler + RidgeClassifier(alpha=1.0), TRAIN fit "
        "/ VAL eval) on both parents' own trained states. Within-parent deltas only. "
        "Primary verdict Movies/Toys/Grocery x seeds 42/43/44. Val only.",
        "",
        "## Per-factor Δ(2−1) / Δ(HP−1) seed-mean (pp) — factor GO rule: "
        "≥2/3 datasets with mean ≥ +0.20pp AND ≥2/3 seeds positive",
        "",
        "| parent | factor | Movies | Toys | Grocery | 2-hop? | HP? |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    factor_go = {}
    for parent in PARENTS:
        for a in FACTOR_NAMES:
            d21 = seed_mean_delta(factor_rows, parent, "H2", "H1", factor=a)
            dhp1 = seed_mean_delta(factor_rows, parent, "HP", "H1", factor=a)
            pos21 = seed_positivity(factor_rows, parent, "H2", "H1", factor=a)
            poshp = seed_positivity(factor_rows, parent, "HP", "H1", factor=a)
            go21 = sum(1 for ds in TARGET_DATASETS if d21[ds] >= 0.20 / 100 and pos21[ds] >= 2) >= 2
            gohp = sum(1 for ds in TARGET_DATASETS if dhp1[ds] >= 0.20 / 100 and poshp[ds] >= 2) >= 2
            factor_go[(parent, a, "2hop")] = go21
            factor_go[(parent, a, "hp")] = gohp
            lines.append(
                f"| {parent} | {a} | {d21['Movies']*100:+.3f} / {dhp1['Movies']*100:+.3f} "
                f"| {d21['Toys']*100:+.3f} / {dhp1['Toys']*100:+.3f} "
                f"| {d21['Grocery']*100:+.3f} / {dhp1['Grocery']*100:+.3f} "
                f"| {'YES' if go21 else 'no'} | {'YES' if gohp else 'no'} |"
            )
    lines += [
        "",
        "## Final-residual Δ (pp) — FINAL-RESIDUAL SUPPORT rule: M/T/G macro "
        "≥ +0.20pp AND ≥2/3 datasets positive",
        "",
    ]
    final_go = {}
    for parent in PARENTS:
        f21 = seed_mean_delta(final_rows, parent, "H2", "H1")
        fhp1 = seed_mean_delta(final_rows, parent, "HP", "H1")
        macro21 = statistics.mean(f21.values())
        macrohp = statistics.mean(fhp1.values())
        go21 = macro21 >= 0.20 / 100 and sum(1 for ds in TARGET_DATASETS if f21[ds] > 0) >= 2
        gohp = macrohp >= 0.20 / 100 and sum(1 for ds in TARGET_DATASETS if fhp1[ds] > 0) >= 2
        final_go[(parent, "2hop")] = go21
        final_go[(parent, "hp")] = gohp
        lines.append(
            f"- {parent}: 2-hop final Δ macro = {macro21*100:+.3f}pp "
            f"({ {k: round(v*100,2) for k,v in f21.items()} }) → "
            f"{'**FINAL-RESIDUAL SUPPORT**' if go21 else 'no'}"
        )
        lines.append(
            f"- {parent}: high-pass final Δ macro = {macrohp*100:+.3f}pp "
            f"({ {k: round(v*100,2) for k,v in fhp1.items()} }) → "
            f"{'**FINAL-RESIDUAL SUPPORT**' if gohp else 'no'}"
        )
    lines += [
        "",
        "## Cross-parent verdict (plan §18)",
        "",
    ]
    for kind, label in (("2hop", "2-hop"), ("hp", "high-pass")):
        for a in FACTOR_NAMES:
            a0 = factor_go[("A0", a, kind)]
            b0 = factor_go[("B0", a, kind)]
            status = (
                "CROSS-PARENT SUPPORT" if a0 and b0 else
                ("PARENT-SPECIFIC (A0)" if a0 else
                 ("PARENT-SPECIFIC (B0)" if b0 else "no"))
            )
            lines.append(f"- {label} factor {a}: A0={'YES' if a0 else 'no'} / "
                         f"B0={'YES' if b0 else 'no'} → {status}")
        a0f, b0f = final_go[("A0", kind)], final_go[("B0", kind)]
        lines.append(f"- {label} final-residual: A0={'YES' if a0f else 'no'} / "
                     f"B0={'YES' if b0f else 'no'} → "
                     + ("CROSS-PARENT FINAL-RESIDUAL" if a0f and b0f else
                        ("PARENT-SPECIFIC FINAL-RESIDUAL" if a0f or b0f else "no")))
    lines += [
        "",
        "If only per-factor strong and final weak → INDUCTIVE-BIAS SUPPORT ONLY "
        "(plan §18). Shuffle controls in propagation_shuffle.csv.",
    ]
    (OUTDIR / "R2D16_PROPAGATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[propagation] saved -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
