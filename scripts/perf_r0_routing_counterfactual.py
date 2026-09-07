"""R0-D5: Γ routing / local-graph utility counterfactual (plan §17-§21/§35).

Frozen-weight counterfactuals on each OFR checkpoint (model_state +
head_state): the pipeline is recomputed with the FROZEN components and only
the plan Gamma is replaced. VAL evaluation only; CF0 (current Gamma) is the
per-checkpoint sanity baseline.

    CF0 current                Gamma^cur
    CF1 uniform relations      gamma0 = cur gamma0; gamma_k = beta / K
    CF2 availability relations gamma0 = cur gamma0; gamma_k = beta * a_ik
    CF3 hard top-1 relation    gamma_k* = beta, others 0
    CF4 no local               gamma0 = 0; gamma_k = alpha
    CF5 fixed factor-mean beta gamma0 = 1 - beta_bar_f; gamma_k = beta_bar_f * alpha
    CF6 local only             gamma0 = 1; gamma_k = 0

Plus score-level diagnostics (null scores, local margin, top1-top2 margin,
local win fraction) and routing-quality alignment (corr(gamma,Q),
corr(alpha,Q), corr(gamma,availability)).

No training, no test access, no val-label routing.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r0_utils import (  # noqa: E402
    DATASETS,
    SEEDS,
    load_setup,
    write_csv,
)
from src.models.biaxis_p1_components import relation_weighted_mean  # noqa: E402
from src.models.biaxis_p2_components import (  # noqa: E402
    build_augmented_scores,
    null_augmented_softmax,
)

OUT_ROOT = PROJECT_ROOT / "outputs" / "perf_r0" / "routing"


@torch.no_grad()
def _pipeline(setup, gamma_override=None) -> tuple[torch.Tensor, dict]:
    """Recompute the frozen pipeline with an optional Gamma override.
    Returns (z, internals). internals includes gamma, g_perm, f_block,
    s_rel, s_aug, alpha, beta, availability, deg, f_f."""
    model, data, device = setup.model, setup.data, setup.device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    num_nodes = int(x.size(0))
    factors, z_local = model._encode(x)
    f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
    r, availability, deg = model._decompose_relations(edge_index, num_nodes)
    f_cat = f_block.reshape(num_nodes, 3 * model.factor_dim)
    g_cat, _mass = relation_weighted_mean(
        edge_index, r, f_cat, num_nodes, edge_chunk_size=model.edge_chunk_size
    )
    g_perm = g_cat.reshape(num_nodes, model.num_relations, 3, model.factor_dim).permute(0, 2, 1, 3)
    s_rel = model.transport_scorer(f_block, g_perm)
    s_aug = build_augmented_scores(s_rel, model.null_score)
    gamma = null_augmented_softmax(s_aug, model.p2_epsilon)
    isolated = deg <= 0
    if bool(isolated.any()):
        local_plan = torch.zeros_like(gamma)
        local_plan[..., 0] = 1.0
        gamma = torch.where(isolated[:, None, None], local_plan, gamma)
    beta = 1.0 - gamma[..., 0]
    alpha = gamma[..., 1:] / (beta.unsqueeze(-1) + model.eps)

    if gamma_override is not None:
        gamma = gamma_override(gamma=gamma, beta=beta, alpha=alpha, availability=availability)

    m_f = model.operator(g_perm, gamma[..., 1:], model.graph_w0)
    f_tilde = model.graph_norm((f_block + m_f).reshape(num_nodes * 3, model.factor_dim))
    f_tilde = f_tilde.reshape(num_nodes, 3, model.factor_dim)
    z = model.fusion(torch.cat([f_tilde[:, 0], f_tilde[:, 1], f_tilde[:, 2]], dim=-1))
    return z, {
        "gamma": gamma, "g_perm": g_perm, "f_block": f_block,
        "s_rel": s_rel, "s_aug": s_aug, "alpha": alpha, "beta": beta,
        "availability": availability, "deg": deg, "z_local": z_local,
    }


def _val_acc(setup, z: torch.Tensor) -> float:
    data, device = setup.data, setup.device
    y = data.y.to(device)
    pred = setup.head(z).argmax(dim=-1)
    val_idx = data.val_idx.to(device)
    return float((pred[val_idx] == y[val_idx]).float().mean().item())


def _cf1(gamma, beta, alpha, availability):
    out = gamma.clone()
    k = alpha.size(-1)
    out[..., 1:] = (beta.unsqueeze(-1) / k).expand_as(alpha)
    return out


def _cf2(gamma, beta, alpha, availability):
    out = gamma.clone()
    out[..., 1:] = beta.unsqueeze(-1) * availability.unsqueeze(1)
    return out


def _cf3(gamma, beta, alpha, availability):
    out = gamma.clone()
    out[..., 1:] = 0.0
    top = alpha.argmax(dim=-1)  # [N, F]
    out.scatter_(-1, top.unsqueeze(-1) + 1, beta.unsqueeze(-1))
    return out


def _cf4(gamma, beta, alpha, availability):
    out = gamma.clone()
    out[..., 0] = 0.0
    out[..., 1:] = alpha
    return out


def _cf5(gamma, beta, alpha, availability):
    out = gamma.clone()
    beta_bar = beta.mean(dim=0, keepdim=True)  # [1, F]
    out[..., 0] = 1.0 - beta_bar
    out[..., 1:] = beta_bar.unsqueeze(-1) * alpha
    return out


def _cf6(gamma, beta, alpha, availability):
    out = gamma.clone()
    out[..., 0] = 1.0
    out[..., 1:] = 0.0
    return out


CFS = {"CF0": None, "CF1": _cf1, "CF2": _cf2, "CF3": _cf3, "CF4": _cf4, "CF5": _cf5, "CF6": _cf6}


def _score_diagnostics(internals: dict, setup) -> dict:
    gamma, s_rel = internals["gamma"], internals["s_rel"]
    s_local = internals["s_aug"][..., 0]
    margin = s_rel.max(dim=-1).values - s_local  # [N, F]
    top2 = gamma.topk(2, dim=-1).values
    top_margin = top2[..., 0] - top2[..., 1]
    local_win = (gamma.argmax(dim=-1) == 0).float().mean()
    qs = torch.quantile(margin.flatten(), torch.tensor([0.1, 0.5, 0.9], device=margin.device))
    return {
        "null_score_C": float(setup.model.null_score[0].item()),
        "null_score_Pt": float(setup.model.null_score[1].item()),
        "null_score_Pv": float(setup.model.null_score[2].item()),
        "local_margin_mean": float(margin.mean().item()),
        "local_margin_p10": float(qs[0].item()),
        "local_margin_p50": float(qs[1].item()),
        "local_margin_p90": float(qs[2].item()),
        "gamma_top_margin_mean": float(top_margin.mean().item()),
        "local_win_fraction": float(local_win.item()),
    }


def _routing_alignment(internals: dict, setup) -> dict:
    import numpy as np

    gamma, alpha, g_perm, f_block, availability = (
        internals["gamma"], internals["alpha"], internals["g_perm"],
        internals["f_block"], internals["availability"],
    )
    q = torch.nn.functional.cosine_similarity(g_perm, f_block.unsqueeze(2), dim=-1)  # [N,F,K]
    gam = gamma[..., 1:]
    out: dict = {}
    flat_g, flat_q, flat_a, flat_alpha = [], [], [], []
    for start in range(0, gam.size(0), 200_000):
        flat_g.append(gam[start : start + 200_000].reshape(-1).cpu().numpy())
        flat_q.append(q[start : start + 200_000].reshape(-1).cpu().numpy())
        flat_a.append(availability[start : start + 200_000].unsqueeze(1).expand(-1, gam.size(1), -1).reshape(-1).cpu().numpy())
        flat_alpha.append(alpha[start : start + 200_000].reshape(-1).cpu().numpy())
    g_all = np.concatenate(flat_g)
    q_all = np.concatenate(flat_q)
    a_all = np.concatenate(flat_a)
    alpha_all = np.concatenate(flat_alpha)
    out["corr_gamma_Q"] = float(np.corrcoef(g_all, q_all)[0, 1])
    out["corr_gamma_avail"] = float(np.corrcoef(g_all, a_all)[0, 1])
    out["corr_alpha_Q"] = float(np.corrcoef(alpha_all, q_all)[0, 1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="R0-D5 routing counterfactual")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    datasets = [d for d in (args.datasets or ",".join(DATASETS)).split(",") if d in DATASETS]
    seeds = [int(s) for s in (args.seeds or ",".join(map(str, SEEDS))).split(",") if s]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cf_rows: list[dict] = []
    score_rows: list[dict] = []
    align_rows: list[dict] = []

    for di, dataset in enumerate(datasets):
        device = torch.device(f"cuda:{gpus[di % len(gpus)]}")
        for seed in seeds:
            setup = load_setup(dataset, seed, device)
            z_cur, internals = _pipeline(setup, gamma_override=None)
            vals = {}
            for name, fn in CFS.items():
                if name == "CF0":
                    acc = _val_acc(setup, z_cur)
                else:
                    z_cf, _ = _pipeline(setup, gamma_override=fn)
                    acc = _val_acc(setup, z_cf)
                    del z_cf
                vals[name] = acc
                cf_rows.append({"dataset": dataset, "seed": seed, "cf": name, "val_acc": acc})
                print(f"[routing] {dataset:12s} s{seed} {name} val={acc:.4f}", flush=True)
                torch.cuda.empty_cache()
            score_rows.append({"dataset": dataset, "seed": seed, **_score_diagnostics(internals, setup)})
            align_rows.append({"dataset": dataset, "seed": seed, **_routing_alignment(internals, setup)})
            del z_cur, internals
            torch.cuda.empty_cache()

    write_csv(OUT_ROOT / "routing_counterfactual_per_seed.csv", cf_rows)
    write_csv(OUT_ROOT / "routing_scores.csv", score_rows)
    write_csv(OUT_ROOT / "routing_alignment.csv", align_rows)

    # summary over seeds
    summary_rows: list[dict] = []
    for dataset in datasets:
        for name in CFS:
            vals = [r["val_acc"] for r in cf_rows if r["dataset"] == dataset and r["cf"] == name]
            summary_rows.append({
                "dataset": dataset, "cf": name,
                "val_acc": statistics.mean(vals) if vals else float("nan"),
                "val_acc_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            })
    write_csv(OUT_ROOT / "routing_counterfactual_summary.csv", summary_rows)

    lines = ["# R0-ROUTING-REPORT — Γ Routing / Local-Graph Utility", ""]
    lines.append("> 冻结权重 counterfactual（val only）。Δ 全部相对各自 checkpoint 的 CF0 current-Gamma。")
    lines.append("")
    lines.append("| dataset | CF0 | CF1 uniform | CF2 avail | CF3 top1 | CF4 no-local | CF5 mean-β | CF6 local-only |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        def v(name):
            rows = [r for r in summary_rows if r["dataset"] == dataset and r["cf"] == name]
            return rows[0] if rows else None
        cells = [v(n) for n in CFS]
        lines.append(
            f"| {dataset} | " + " | ".join(
                f"{c['val_acc']:.4f}±{c['val_acc_std']:.4f}" if c else "" for c in cells
            ) + " |"
        )
    lines.append("")
    lines.append("| dataset | Δ_select (CF0−CF1) | Δ_sem-select (CF0−CF2) | Δ_demand (CF0−CF5) | Δ_local (CF0−CF4) | corr(α,Q) | local_win |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        def v(name):
            rows = [r for r in summary_rows if r["dataset"] == dataset and r["cf"] == name]
            return rows[0] if rows else None
        c0 = v("CF0")
        d_sel = 100 * (c0["val_acc"] - v("CF1")["val_acc"]) if c0 and v("CF1") else float("nan")
        d_sem = 100 * (c0["val_acc"] - v("CF2")["val_acc"]) if c0 and v("CF2") else float("nan")
        d_dem = 100 * (c0["val_acc"] - v("CF5")["val_acc"]) if c0 and v("CF5") else float("nan")
        d_loc = 100 * (c0["val_acc"] - v("CF4")["val_acc"]) if c0 and v("CF4") else float("nan")
        corr_aq = statistics.mean([r["corr_alpha_Q"] for r in align_rows if r["dataset"] == dataset])
        lw = statistics.mean([r["local_win_fraction"] for r in score_rows if r["dataset"] == dataset])
        lines.append(
            f"| {dataset} | {d_sel:+.2f} | {d_sem:+.2f} | {d_dem:+.2f} | {d_loc:+.2f} | {corr_aq:.3f} | {lw:.3f} |"
        )
    lines.append("")
    (OUT_ROOT / "R0_ROUTING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[routing] done -> {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
