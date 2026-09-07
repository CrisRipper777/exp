"""R1 best-checkpoint mechanism diagnostics (plan §39 Prompt 5 / §11).

For one (dataset, seed, mode) checkpoint:
  1. Model.compute_r1_diagnostics -> r1_diagnostics.json
     (P3 plan/operator payload + eta/CV/neighbor/context_change/D_ctx/
     effective_mass; baseline mode reports None for the reliability sections)
  2. Fixed Ridge context probes (train fit / val eval): f, g_bar, g_all,
     f|g_bar, f|g_all -> Delta_relctx = probe(f|g_all) - probe(f|g_bar)
  3. Gamma counterfactuals CF0/CF1/CF2 (frozen weights, VAL acc only):
     Current / Uniform / Availability (R0 D5 definitions, plan §11)
Prints the analysis peak allocation. NEVER reads test.

Usage:
    python scripts/analyze_perf_r1_checkpoint.py --dataset Movies --seed 42 \
        --mode A1 --ckpt outputs/perf_r1/reliability/Movies/A1/seed_42/model.pt \
        --out outputs/perf_r1/reliability/Movies/A1/seed_42 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r0_utils import ridge_probe, write_csv  # noqa: E402
from src.analysis.perf_r1_utils import (  # noqa: E402
    COUNTERFACTUALS,
    FACTOR_NAMES,
    R1Setup,
    apply_plan,
    assert_no_test_access,
    load_r1_setup,
    r1_pipeline,
    val_acc_with_head,
)
from src.models.biaxis_p2_components import (  # noqa: E402
    build_augmented_scores,
    null_augmented_softmax,
)

# Modes carrying the dynamic Local residual (user §10 diagnostics).
LOCAL_RESIDUAL_MODES = {"BL", "BLR"}
# Modes carrying the support relation residual (user §11 + J-series).
RELATION_RESIDUAL_MODES = {"BR", "BLR"}


def _run_probes(setup: R1Setup, internals: dict) -> list[dict]:
    """Fixed Ridge probes per factor (R0 protocol): f, g_bar, g_all,
    f|g_bar, f|g_all. Delta_relctx = f|g_all - f|g_bar (plan §11)."""
    from src.models.biaxis_p1_components import neighbor_mean

    f_block = internals["f_block"]
    g_perm = internals["graph_out"]["g_perm"]  # [N, F, K, d]
    n, fnum, k, d = g_perm.shape
    g_bar = neighbor_mean(
        internals["edge_index"], f_block.reshape(n, fnum * d), n,
        edge_chunk_size=setup.model.edge_chunk_size,
    ).reshape(n, fnum, d)
    rows: list[dict] = []
    for fi, fname in enumerate(FACTOR_NAMES):
        f_f = f_block[:, fi]
        gbar_f = g_bar[:, fi]
        g = g_perm[:, fi]  # [N, K, d]
        variants = {
            "f": f_f,
            "g_bar": gbar_f,
            "g_all": g.reshape(n, k * d),
            "f|g_bar": torch.cat([f_f, gbar_f], dim=-1),
            "f|g_all": torch.cat([f_f, g.reshape(n, k * d)], dim=-1),
        }
        for name, tensor in variants.items():
            probe = ridge_probe(tensor, setup)
            rows.append({
                "dataset": setup.dataset, "seed": setup.seed, "mode": setup.mode,
                "factor": fname, "variant": name,
                "val_acc": probe["val_acc"], "val_macro_f1": probe["val_macro_f1"],
            })
            print(
                f"[probe] {setup.dataset:12s} s{setup.seed} {setup.mode} {fname} {name:8s} "
                f"val={probe['val_acc']:.4f}", flush=True,
            )
    return rows


def _cf_dynlocal_off(setup: R1Setup, internals: dict) -> torch.Tensor:
    """Frozen counterfactual (user §10): the SAME trained checkpoint with the
    dynamic Local residual forced to 0 (Local column back to the global z_f).
    Isolates Delta_dyn-local = val(CF0) - val(CF_dynlocal_off)."""
    model = setup.model
    s_aug0 = build_augmented_scores(internals["scores"]["s_rel"], model.null_score)
    gamma = null_augmented_softmax(s_aug0, model.p2_epsilon)
    deg = internals["deg"]
    isolated = deg <= 0
    if bool(isolated.any()):
        local_plan = torch.zeros_like(gamma)
        local_plan[..., 0] = 1.0
        gamma = torch.where(isolated[:, None, None], local_plan, gamma)
    return gamma


def _run_j_counterfactuals(setup: R1Setup, internals: dict) -> list[dict]:
    """BLR J-series (user ruling): frozen same-checkpoint counterfactuals.
    J0 full BLR / J1 delta_local=0 / J2 delta_relation=0 / J3 both off.
    Weights and head untouched; Val only."""
    from src.models.biaxis_p1_components import neighbor_mean

    model = setup.model
    f_block = internals["f_block"]
    g_perm = internals["graph_out"]["g_perm"]
    availability = internals["graph_out"]["availability"]
    deg = internals["deg"]
    num_nodes, num_factors = f_block.shape[0], f_block.shape[1]
    s_rel_base = model.transport_scorer(f_block, g_perm)  # [N, F, K]

    mass = availability * deg.unsqueeze(-1)  # [N, K]
    rel_res = model.relation_score_residual(
        torch.log1p(mass), availability
    )  # [N, K, 1]
    s_rel_br = s_rel_base + rel_res.permute(0, 2, 1).expand(num_nodes, num_factors, model.num_relations)

    f_cat = f_block.reshape(num_nodes, num_factors * model.factor_dim)
    g_bar = neighbor_mean(
        internals["edge_index"], f_cat, num_nodes, edge_chunk_size=model.edge_chunk_size
    ).reshape(num_nodes, num_factors, model.factor_dim)
    loc_res = model.local_score_residual(f_block, g_bar).squeeze(-1)  # [N, F]

    def make_gamma(s_rel: torch.Tensor, add_local: bool) -> torch.Tensor:
        s_aug = build_augmented_scores(s_rel, model.null_score)
        if add_local:
            s_aug[..., 0] = s_aug[..., 0] + loc_res
        gamma = null_augmented_softmax(s_aug, model.p2_epsilon)
        isolated = deg <= 0
        if bool(isolated.any()):
            local_plan = torch.zeros_like(gamma)
            local_plan[..., 0] = 1.0
            gamma = torch.where(isolated[:, None, None], local_plan, gamma)
        return gamma

    rows: list[dict] = []
    for name, gamma in (
        ("J0_full", make_gamma(s_rel_br, True)),
        ("J1_local_off", make_gamma(s_rel_br, False)),
        ("J2_relation_off", make_gamma(s_rel_base, True)),
        ("J3_both_off", make_gamma(s_rel_base, False)),
    ):
        acc = val_acc_with_head(setup, apply_plan(setup, internals, gamma))
        rows.append({"dataset": setup.dataset, "seed": setup.seed, "mode": setup.mode,
                     "j": name, "val_acc": acc})
        print(f"[j] {setup.dataset:12s} s{setup.seed} {setup.mode} {name} val={acc:.4f}", flush=True)
        del gamma
        torch.cuda.empty_cache()
    return rows


def _run_hop_counterfactual(setup: R1Setup) -> list[dict]:
    """C1SG (user ruling): same-checkpoint learned-lambda vs lambda=0, Val
    only. Both sides come from the FULL two-hop forward (the trajectory
    readout is part of the model), differing only in lam."""
    model = setup.model
    data, device = setup.data, setup.device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    num_nodes = int(x.size(0))
    rows: list[dict] = []
    zs: dict[str, torch.Tensor] = {}

    def _full_hop_forward(use_lambda: bool) -> torch.Tensor:
        factors, _ = model._encode(x)
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        graph_out1 = model._graph_update(f_block, edge_index, num_nodes)
        f1 = graph_out1["f_tilde"]
        with torch.no_grad():
            graph_out2 = model._graph_update(f1.detach(), edge_index, num_nodes)
        f2 = graph_out2["f_tilde"]
        _f_out, lam, d = model._hop_readout(f_block, f1, f2)
        if not use_lambda:
            lam = torch.zeros_like(lam)
        f_out = f1 + lam.unsqueeze(-1) * d
        return model.fusion(torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1))

    # Evaluate each variant immediately and free its full autograd graph
    # before the next forward (ele-fashion: holding both graphs at once
    # exceeds a shared 24GB card).
    for name, use_lam in (("CF8_hop_on", True), ("CF9_hop_off", False)):
        z = _full_hop_forward(use_lam)
        acc = val_acc_with_head(setup, z)
        rows.append({"dataset": setup.dataset, "seed": setup.seed, "mode": setup.mode,
                     "cf": name, "val_acc": acc})
        print(f"[hop-cf] {setup.dataset:12s} s{setup.seed} {setup.mode} {name} val={acc:.4f}", flush=True)
        del z
        torch.cuda.empty_cache()
    return rows


def _run_router_stats(setup: R1Setup, internals: dict) -> dict:
    """BLR router diagnostics (user ruling):
    - Local-vs-Graph margin M_if = LSE_k(s_ifk/eps) - s_if0/eps
      (margin > 0 <=> relation side dominates the Local column)
    - relation residual decomposition: common shift delta_bar_if (mean over k,
      softmax-shift-invariant inside the relation block -> acts on the graph
      mass only) vs centered residual delta_c_ifk (changes conditional alpha)
    """
    model = setup.model
    f_block = internals["f_block"]
    g_perm = internals["graph_out"]["g_perm"]
    availability = internals["graph_out"]["availability"]
    deg = internals["deg"]
    num_nodes, num_factors = f_block.shape[0], f_block.shape[1]

    s_rel_base = model.transport_scorer(f_block, g_perm)
    mass = availability * deg.unsqueeze(-1)
    rel_res = model.relation_score_residual(
        torch.log1p(mass), availability
    )  # [N, K, 1]
    s_rel = s_rel_base + rel_res.permute(0, 2, 1).expand(num_nodes, num_factors, model.num_relations)
    s_local = internals["scores"]["s_aug"][..., 0]  # z_f + delta_local
    margin = torch.logsumexp(s_rel / model.p2_epsilon, dim=-1) - s_local / model.p2_epsilon  # [N, F]

    delta = rel_res.squeeze(-1).unsqueeze(1).expand(num_nodes, num_factors, model.num_relations)  # [N, F, K]
    delta_bar = delta.mean(dim=-1)  # [N, F] common shift
    delta_c = delta - delta_bar.unsqueeze(-1)  # [N, F, K] centered

    qs = torch.tensor([0.1, 0.5, 0.9], dtype=s_rel.dtype, device=s_rel.device)
    out: dict[str, dict[str, float]] = {}
    for fi, fname in enumerate(FACTOR_NAMES):
        m = margin[:, fi]
        q = torch.quantile(m, qs)
        db = delta_bar[:, fi]
        dc = delta_c[:, fi].reshape(-1)
        qb = torch.quantile(db, qs)
        qc = torch.quantile(dc, qs)
        out[fname] = {
            "margin_mean": float(m.mean().item()),
            "margin_std": float(m.std(unbiased=False).item()),
            "margin_p10": float(q[0].item()),
            "margin_p50": float(q[1].item()),
            "margin_p90": float(q[2].item()),
            "shift_mean": float(db.mean().item()),
            "shift_std": float(db.std(unbiased=False).item()),
            "shift_p50": float(qb[1].item()),
            "centered_std": float(dc.std(unbiased=False).item()),
            "centered_p50": float(qc[1].item()),
        }
    return out


def _run_local_diagnostics(setup: R1Setup, internals: dict) -> dict:
    """BL diagnostics (user §10): delta_if0 mean/std/quantiles, std_i(s_if0),
    std_i(beta_if) per factor, from the trained checkpoint's own forward."""
    model = setup.model
    s_aug = internals["scores"]["s_aug"]  # [N, F, K+1]
    beta = internals["beta"]  # [N, F]
    null_score = model.null_score  # [F]
    delta_local = s_aug[..., 0] - null_score.reshape(1, -1)  # [N, F]
    qs = torch.tensor([0.1, 0.5, 0.9], dtype=s_aug.dtype, device=s_aug.device)
    out: dict[str, dict[str, float]] = {}
    for fi, fname in enumerate(FACTOR_NAMES):
        d = delta_local[:, fi]
        q = torch.quantile(d, qs)
        out[fname] = {
            "delta_mean": float(d.mean().item()),
            "delta_std": float(d.std(unbiased=False).item()),
            "delta_p10": float(q[0].item()),
            "delta_p50": float(q[1].item()),
            "delta_p90": float(q[2].item()),
            "s_local_std": float(s_aug[:, fi, 0].std(unbiased=False).item()),
            "beta_std": float(beta[:, fi].std(unbiased=False).item()),
        }
    return out


def _run_counterfactuals(setup: R1Setup, internals: dict) -> list[dict]:
    """CF0 current / CF1 uniform / CF2 availability / CF4 no-local /
    CF5 mean-beta (frozen weights) + CF_dynlocal_off for BL/BLR modes."""
    gamma = internals["graph_out"]["gamma"]
    beta = internals["beta"]
    alpha = internals["alpha"]
    availability = internals["graph_out"]["availability"]
    rows: list[dict] = []
    for name, fn in COUNTERFACTUALS.items():
        z = internals["z_final"] if fn is None else apply_plan(
            setup, internals, fn(gamma, beta, alpha, availability)
        )
        acc = val_acc_with_head(setup, z)
        rows.append({"dataset": setup.dataset, "seed": setup.seed, "mode": setup.mode,
                     "cf": name, "val_acc": acc})
        print(f"[cf] {setup.dataset:12s} s{setup.seed} {setup.mode} {name} val={acc:.4f}", flush=True)
        if z is not internals["z_final"]:
            del z
        torch.cuda.empty_cache()
    if setup.mode in LOCAL_RESIDUAL_MODES:
        z_off = apply_plan(setup, internals, _cf_dynlocal_off(setup, internals))
        acc_off = val_acc_with_head(setup, z_off)
        rows.append({"dataset": setup.dataset, "seed": setup.seed, "mode": setup.mode,
                     "cf": "CF7_dynlocal_off", "val_acc": acc_off})
        print(
            f"[cf] {setup.dataset:12s} s{setup.seed} {setup.mode} CF7_dynlocal_off val={acc_off:.4f}",
            flush=True,
        )
        del z_off
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="R1 checkpoint mechanism diagnostics")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", default="nc")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", required=True, choices=["A0", "A1", "A1R1", "A1R2", "A1R3", "A1R4", "A2", "BL", "BR", "BLR", "C1SG"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    setup = load_r1_setup(args.dataset, args.seed, args.mode, device)
    assert_no_test_access(setup.data)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. model-level diagnostics (P3 payload + R1-A reliability payload).
    x = setup.data.x.to(device)
    edge_index = setup.data.edge_index.to(device)
    diag = setup.model.compute_r1_diagnostics(x, edge_index)
    with (outdir / "r1_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    if diag.get("reliability"):
        eta = diag["reliability"]["eta"]
        print(
            f"[diag] {args.dataset} s{args.seed} {args.mode} "
            + " ".join(f"{k} mean={v['mean']:.4f} cv={v['cv']:.4f}" for k, v in eta.items()),
            flush=True,
        )

    # 2. context probes + Delta_relctx.
    internals = r1_pipeline(setup)
    probe_rows = _run_probes(setup, internals)
    write_csv(outdir / "context_probes.csv", probe_rows)

    # 3. routing counterfactuals (Current / Uniform / Availability /
    # no-local / mean-beta; + dyn-local OFF for BL/BLR). For C1SG the
    # Gamma-level CFs do not compose with the two-hop readout — replaced
    # by the learned-lambda vs lambda=0 counterfactual (user ruling).
    if setup.mode == "C1SG":
        write_csv(outdir / "routing_counterfactual.csv", [])
        write_csv(outdir / "hop_counterfactual.csv", _run_hop_counterfactual(setup))
    else:
        cf_rows = _run_counterfactuals(setup, internals)
        write_csv(outdir / "routing_counterfactual.csv", cf_rows)

    # 4. BL local-score diagnostics (user §10), BL/BLR modes only.
    if setup.mode in LOCAL_RESIDUAL_MODES:
        local_stats = _run_local_diagnostics(setup, internals)
        with (outdir / "local_stats.json").open("w", encoding="utf-8") as f:
            json.dump(local_stats, f, indent=2)
        print(
            f"[local] {args.dataset} s{args.seed} {args.mode} "
            f"C delta_std={local_stats['C']['delta_std']:.4f} "
            f"s_local_std={local_stats['C']['s_local_std']:.4f}", flush=True,
        )

    # 5. BLR J-series + router stats (user ruling), BLR only.
    if setup.mode == "BLR":
        j_rows = _run_j_counterfactuals(setup, internals)
        write_csv(outdir / "j_counterfactual.csv", j_rows)
        router_stats = _run_router_stats(setup, internals)
        with (outdir / "router_stats.json").open("w", encoding="utf-8") as f:
            json.dump(router_stats, f, indent=2)
        print(
            f"[router] {args.dataset} s{args.seed} {args.mode} "
            f"C margin_mean={router_stats['C']['margin_mean']:.3f} "
            f"shift_std={router_stats['C']['shift_std']:.4f} "
            f"centered_std={router_stats['C']['centered_std']:.4f}", flush=True,
        )
    del internals
    torch.cuda.empty_cache()

    if x.is_cuda:
        print(f"[mem] peak_allocated_mb={torch.cuda.max_memory_allocated(device) / 1e6:.1f}", flush=True)
    print(f"[r1-diag] saved -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
