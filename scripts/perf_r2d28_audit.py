"""R2-Design-2.8 v2 D2.8-0 audit (v2 plan §5, Prompt 1).

Repair validation on a real graph (Movies seed 42, no training):
  A. repaired within-target shuffle: per-target histograms / change
     fractions / non-identity fractions on real edge scores;
  B. per-target removal: per-target counts exact vs floor(deg*pct);
  C. COUPLED_EQUIV: explicit r*pi factorization reproduces the old
     PAIR_EDGE message to < 1e-6 on the trained D2.7 checkpoint;
  D. new model family smoke: every stage-representative config runs one
     training epoch on Movies (forward + backward + val finite), the
     exposure stats export and the operator stats export run.

Writes outputs/perf_r2d28/audit/{R2D28_AUDIT_V2.md, audit.json}.
No formal experiments — smoke only (v2 Prompt 1: "不要跑正式实验").
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d27_utils import (  # noqa: E402
    R2D27_ROOT,
    load_a0_parent,
)
from src.analysis.perf_r2d28_utils import (  # noqa: E402
    R2D28_ROOT,
    build_model,
    resolve_cfg,
)

AUDIT_ROOT = R2D28_ROOT / "audit"
COUPLED_TOL = 1e-6

SMOKE_CONFIGS = [
    # D2.8-B exposure variants (comp uniform, operator linear, channel mean)
    ("E0", dict(exposure="fixed_full")),
    ("E1", dict(exposure="node")),
    ("E2", dict(exposure="target")),
    ("E3", dict(exposure="source")),
    ("E4", dict(exposure="pair")),
    # D2.8-C composition variants (exposure frozen at pair granularity)
    ("C0", dict(exposure="pair", composition="uniform", freeze_exposure=True)),
    ("C1", dict(exposure="pair", composition="generic", freeze_exposure=True)),
    ("C2", dict(exposure="pair", composition="target", freeze_exposure=True)),
    ("C3", dict(exposure="pair", composition="source", freeze_exposure=True)),
    ("C4", dict(exposure="pair", composition="pair", freeze_exposure=True)),
    # D2.8-D channel variants (exposure + composition frozen)
    ("M0", dict(exposure="pair", composition="target", channel="mean",
                freeze_exposure=True, freeze_composition=True)),
    ("M1", dict(exposure="pair", composition="target", channel="softmax",
                freeze_exposure=True, freeze_composition=True)),
    ("M2", dict(exposure="pair", composition="target", channel="concat",
                freeze_exposure=True, freeze_composition=True)),
    ("M2_MEAN_DUP", dict(exposure="pair", composition="target", channel="concat",
                         mean_dup=True, freeze_exposure=True, freeze_composition=True)),
    ("M3", dict(exposure="pair", composition="target", channel="attn",
                freeze_exposure=True, freeze_composition=True)),
    ("M3_MEAN_DUP", dict(exposure="pair", composition="target", channel="attn",
                         mean_dup=True, freeze_exposure=True, freeze_composition=True)),
    # D2.8-E operator variants (E + C + M frozen; NormMatch on)
    ("O0", dict(exposure="pair", composition="target", channel="mean",
                operator="linear", freeze_exposure=True, freeze_composition=True,
                freeze_channel=True)),
    ("O1", dict(exposure="pair", composition="target", channel="mean",
                operator="static_pair", freeze_exposure=True, freeze_composition=True,
                freeze_channel=True)),
    ("O2", dict(exposure="pair", composition="target", channel="mean",
                operator="target_film", freeze_exposure=True, freeze_composition=True,
                freeze_channel=True)),
    ("O3", dict(exposure="pair", composition="target", channel="mean",
                operator="edge_film", freeze_exposure=True, freeze_composition=True,
                freeze_channel=True)),
    ("O4", dict(exposure="pair", composition="target", channel="mean",
                operator="basis", freeze_exposure=True, freeze_composition=True,
                freeze_channel=True)),
    ("O4_UNIFORM", dict(exposure="pair", composition="target", channel="mean",
                        operator="basis", uniform_router=True,
                        freeze_exposure=True, freeze_composition=True,
                        freeze_channel=True)),
    ("O4_TARGET", dict(exposure="pair", composition="target", channel="mean",
                       operator="basis", target_router=True,
                       freeze_exposure=True, freeze_composition=True,
                       freeze_channel=True)),
]


def repair_validation(device: torch.device, ds: str = "Movies", seed: int = 42) -> dict:
    """A/B/C: repaired machinery on the trained D2.7 PAIR_EDGE checkpoint."""
    from src.models.biaxis_r2_neighbor_utility import Model as OldModel
    from src.models.biaxis_r2_neighbor_utility_components import (
        chunked_pair_message,
    )
    from src.models.biaxis_r2_relfunc_components import (
        chunked_coupled_message,
        per_target_edge_mask,
        shuffle_scores_within_target,
        validate_shuffle,
    )

    setup = load_a0_parent(ds, seed, device)
    data = setup.data
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"),
                               version_base=None):
        cfg = compose(config_name="config", overrides=[
            f"dataset={ds}", "task=nc", "model=biaxis_r2_neighbor_utility",
            "model.mode=pair_edge", f"seed={seed}"])
    info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
            "num_classes": data.num_classes,
            "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1])}
    ckpt = torch.load(
        R2D27_ROOT / "matrix" / ds / "PAIR_EDGE" / f"seed_{seed}" / "best.pt",
        map_location="cpu", weights_only=False)
    m_old = OldModel(cfg, info, setup.parent).to(device)
    m_old.load_state_dict(ckpt["model_state"])
    m_old.eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    num_nodes = int(x.size(0))
    with torch.no_grad():
        f_block, _ = m_old._parent_ctx(x, ei, num_nodes)

    # A: repaired shuffle on real scores (pairs 01 and 11)
    shuffle_stats = {}
    for (a, b) in [(0, 1), (1, 1)]:
        with torch.no_grad():
            s = m_old._pair_scores_chunked(f_block, ei, a, b, int(ei.size(1)))
            s_perm = shuffle_scores_within_target(s, ei)
            shuffle_stats[f"pair_{a}{b}"] = validate_shuffle(
                s, s_perm, ei, num_nodes)
    shuffle_ok = all(
        st["frac_score_changed"] >= 0.80 and st["frac_nonidentity_targets"] >= 0.95
        and st["sums_preserved"] for st in shuffle_stats.values())

    # B: per-target removal counts on real scores
    removal = {}
    with torch.no_grad():
        s = m_old._pair_scores_chunked(f_block, ei, 0, 1, int(ei.size(1)))
        deg = torch.bincount(ei[1], minlength=num_nodes)
        for op, pct in (("remove_top", 0.10), ("remove_random", 0.10),
                        ("remove_bottom", 0.25), ("keep_top", 0.25)):
            mask = per_target_edge_mask(s, ei, num_nodes, op, pct)
            n_sel = int(sum(int(deg[i] * pct) for i in range(num_nodes)))
            # remove_* drop n_sel edges; keep_top keeps n_sel and drops the rest
            expected = int(ei.size(1)) - n_sel if op == "keep_top" else n_sel
            actual = int((~mask).sum().item())
            removal[f"{op}_{pct}"] = {"expected_removed": expected,
                                      "actual_removed": actual}
    removal_ok = all(r["expected_removed"] == r["actual_removed"]
                     for r in removal.values())

    # C: COUPLED_EQUIV vs old message, same weights
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"),
                               version_base=None):
        cfg2 = compose(config_name="config", overrides=[
            f"dataset={ds}", "task=nc", "model=biaxis_r2_neighbor_utility",
            "model.mode=coupled_equiv", f"seed={seed}"])
    m_coupled = OldModel(cfg2, info, setup.parent).to(device)
    m_coupled.load_state_dict(m_old.state_dict())
    m_coupled.eval()
    max_diff = 0.0
    with torch.no_grad():
        payloads = [m_old.payload[a](f_block[:, a]) for a in range(3)]
        for (a, b) in [(0, 1), (1, 1), (2, 2)]:
            s_ab = m_old._pair_scores_chunked(f_block, ei, a, b, int(ei.size(1)))
            null_ab = m_old._null_scores(f_block, a, b)
            m_old_msg = chunked_pair_message(
                f_block, ei, num_nodes, s_ab, null_ab, payloads[a],
                edge_chunk_size=m_old.edge_chunk_size)
            m_new_msg = chunked_coupled_message(
                f_block, ei, num_nodes, s_ab, null_ab, payloads[a],
                edge_chunk_size=m_old.edge_chunk_size)
            diff = float((m_old_msg - m_new_msg).abs().max().item())
            max_diff = max(max_diff, diff)
    coupled_ok = max_diff < COUPLED_TOL

    # also compare whole-model forward outputs (z) of both modes
    with torch.no_grad():
        z_old, _, _, _, _ = m_old(x, ei)
        z_new, _, _, _, _ = m_coupled(x, ei)
    z_diff = float((z_old - z_new).abs().max().item())

    return {"shuffle_stats": shuffle_stats, "shuffle_ok": bool(shuffle_ok),
            "removal": removal, "removal_ok": bool(removal_ok),
            "coupled_max_msg_diff": max_diff,
            "coupled_max_z_diff": z_diff, "coupled_ok": bool(coupled_ok),
            "num_nodes": num_nodes, "num_edges": int(ei.size(1))}


def new_model_smoke(device: torch.device, ds: str = "Movies",
                    seed: int = 42, epochs: int = 1) -> list[dict]:
    """D: one training epoch per stage-representative config (forward +
    backward + val), plus exposure/operator stat exports on E4/O4."""
    setup = load_a0_parent(ds, seed, device)
    data = setup.data
    results = []
    for name, overrides in SMOKE_CONFIGS:
        t0 = time.monotonic()
        cfg = resolve_cfg(ds, seed, overrides)
        model = build_model(cfg, data, setup.parent, device)
        from src.analysis.perf_r2d27_utils import load_or_make_head_init
        from src.analysis.perf_r2d28_utils import HEAD_INIT_ROOT, train_relfunc_model

        head = load_or_make_head_init(
            HEAD_INIT_ROOT / f"{ds}_seed{seed}_d{model.out_dim}.pt",
            model.out_dim, int(data.num_classes), device)
        res = train_relfunc_model(data, model, head, device, total_epochs=int(epochs))
        # export smoke: exposure stats on E4, operator stats on O4
        exports = {}
        x = data.x.to(device)
        ei = data.edge_index.to(device)
        if name == "E4":
            exports["exposure_stats"] = model.export_exposure_stats(
                x, ei, data.train_idx, data.y[data.train_idx])
        if name == "O4":
            exports["operator_stats"] = model.export_operator_stats(x, ei)
        finite = math.isfinite(res["best_val_acc"])
        peak_mb = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
        results.append({"variant": name, "finite": bool(finite),
                        "val_acc": res["best_val_acc"],
                        "runtime_sec": round(time.monotonic() - t0, 1),
                        "peak_allocated_mb": peak_mb,
                        "side_params": int(model.side_parameter_count),
                        "exports_ok": all(
                            isinstance(v, dict) and v for v in exports.values()),
                        })
        del model, head, res
        torch.cuda.empty_cache()
        print(f"[smoke] {name} finite={finite} acc={results[-1]['val_acc']:.4f} "
              f"{results[-1]['runtime_sec']}s", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="R2D2.8 v2 D2.8-0 audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    device = torch.device(args.device)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    print("[audit] repair validation on trained D2.7 PAIR_EDGE checkpoint ...",
          flush=True)
    repair = repair_validation(device)
    print(f"[audit] shuffle_ok={repair['shuffle_ok']} "
          f"removal_ok={repair['removal_ok']} "
          f"coupled_max_diff={repair['coupled_max_msg_diff']:.3e} "
          f"(tol {COUPLED_TOL:.0e})", flush=True)
    print("[audit] new-model smoke (1 epoch each) ...", flush=True)
    smoke = new_model_smoke(device, epochs=args.epochs)

    audit = {"repair": repair, "smoke": smoke}
    with (AUDIT_ROOT / "audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    lines = [
        "# R2D28_AUDIT_V2 — D2.8-0 causal repair + factorization infrastructure",
        "",
        "Repository: `CrisRipper777/0901` — R2-Design-2.8 v2",
        "(docs/BiAxis_R2_Design_2_8_v2_Identifiable_Relational_Function_Decomposition.md)",
        "",
        "Scope: implementation audit + smoke only. **No formal experiments.**",
        "",
        "## 1. Repaired within-target shuffle (v2 §5.1)",
        "",
        "The old float32 composite key (`dst.float() * 1e7 + tie_break`) quantizes"
        " `dst*1e7` to float32, whose ULP reaches ~1e5 at large `dst`; distinct"
        " targets then share one key value and the shuffle degenerates toward"
        " identity on exactly the large graphs that matter.",
        "The repair groups edges by exact integer `dst` and permutes inside each"
        " segment with the integer key `(dst << 32) | r32` (seed 20260904).",
        "",
        "Real-graph validation (Movies seed 42, trained PAIR_EDGE checkpoint,"
        f" {repair['num_nodes']} nodes / {repair['num_edges']} edges):",
        "",
        "| pair | frac score changed (deg>1) | frac non-identity targets | sums preserved |",
        "|---|---|---|---|",
    ]
    for key, st in repair["shuffle_stats"].items():
        lines.append(
            f"| {key} | {st['frac_score_changed']:.4f} "
            f"(>=0.80) | {st['frac_nonidentity_targets']:.4f} (>=0.95) "
            f"| {st['sums_preserved']} |")
    lines += [
        f"",
        f"**Verdict: {'PASS' if repair['shuffle_ok'] else 'FAIL'}** — the repaired"
        f" shuffle meets both mandatory thresholds on real scores.",
        "",
        "## 2. Per-target removal (v2 §5.2)",
        "",
        "`REMOVE_TOP/RANDOM/BOTTOM_PER_TARGET_{10,25,50}` and"
        " `KEEP_TOP_PER_TARGET_{25,50}` select inside each target's own"
        " neighborhood (floor(deg*pct) per target); random removes the same"
        " per-target count as top/bottom; the null is preserved and the"
        " remaining real-neighbor composition is renormalized by the softmax.",
        "",
        "Count validation on real scores (pair 0->1):",
        "",
        "| override | expected removed | actual removed |",
        "|---|---|---|",
    ]
    for k, r in repair["removal"].items():
        lines.append(f"| {k} | {r['expected_removed']} | {r['actual_removed']} |")
    lines += [
        "",
        f"**Verdict: {'PASS' if repair['removal_ok'] else 'FAIL'}**",
        "",
        "## 3. COUPLED_EQUIV exact factorization (v2 §5.3)",
        "",
        "`alpha_ji = r_i * pi_ji` with `Z_i = sum_j exp(s_ji)`,"
        " `r_i = Z_i/(exp(s_null)+Z_i)`, `pi_ji = exp(s_ji)/Z_i`, using the"
        " **same** edge/null logits as the old PAIR_EDGE model.",
        "",
        f"- max |m_coupled - m_old| over pairs 0->1, 1->1, 2->2: "
        f"**{repair['coupled_max_msg_diff']:.3e}** (require < {COUPLED_TOL:.0e})",
        f"- max |z_coupled - z_pair_edge| full forward: {repair['coupled_max_z_diff']:.3e}",
        "",
        f"**Verdict: {'PASS' if repair['coupled_ok'] else 'FAIL'}** — the D2.7 ->"
        f" D2.8 bridge is exact.",
        "",
        "## 4. New model family smoke (1 epoch each, Movies seed 42)",
        "",
        "| variant | finite | val acc | params | time s | peak MB |",
        "|---|---|---|---|---|---|",
    ]
    for r in smoke:
        lines.append(
            f"| {r['variant']} | {r['finite']} | {r['val_acc']:.4f} "
            f"| {r['side_params']} | {r['runtime_sec']} | {r['peak_allocated_mb']} |")
    smoke_ok = all(r["finite"] for r in smoke) and len(smoke) == len(SMOKE_CONFIGS)
    lines += [
        "",
        f"**Verdict: {'PASS' if smoke_ok else 'FAIL'}**",
        "",
        "## 5. v2 identifiability controls implemented",
        "",
        "- Rule I — exposure tests fix pi uniform / O = U_a / lambda = 1/3; only"
        " `r` varies (`E0..E4`, capacity-matched predictors, sigmoid outputs).",
        "- Rule II — composition softmax over **real neighbors only** (no null"
        " inside pi); E* is loaded and frozen (staged loading via"
        " `load_frozen_components`).",
        "- Rule III — channel lambda is `Softmax_a` (simplex; cannot become a"
        " second exposure gate).",
        "- Rule IV — NormMatch on all primary operator diagnostics"
        " (O1-O4 step 0 == O0 by zero/small-init).",
        "- Rule V — staged freezing flags `freeze_exposure/…/freeze_operator`;"
        " frozen groups excluded from the optimizer; joint co-training deferred"
        " to D2.8-F as secondary confirmation.",
        "- No free node/edge tables; A0 parent frozen and bitwise-reproduced by"
        " `side_off`; No Test access; RoleMAG/TMTE collision guardrails.",
        "",
        "## 6. Discipline",
        "",
        "- Seeds 42/43/44, Val-only, **No Test** for all formal experiments.",
        "- A0 accepted strong parent (`biaxis_final`), frozen in D2.8-B..E.",
        "- Classifier init: shared D2.7 per-(dataset, seed) head-init files.",
        "",
    ]
    (AUDIT_ROOT / "R2D28_AUDIT_V2.md").write_text("\n".join(lines),
                                                  encoding="utf-8")
    print("[audit] wrote outputs/perf_r2d28/audit/", flush=True)
    ok = (repair["shuffle_ok"] and repair["removal_ok"] and repair["coupled_ok"]
          and smoke_ok)
    print(f"[audit] OVERALL {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
