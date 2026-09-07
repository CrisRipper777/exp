"""R0-D3/D4: structural relation + relation-conditioned context quality
(plan §6-§16/§34 Prompt 4).

Relation (per seed):
    occupancy / K_eff / normalized entropy / max confidence
    relation mass & availability statistics (support)
    r-weighted structural profiles (src/dst log-degree, degree gap, u1/u2)
    r-weighted semantic edge cosine Sim_{f,k} for C/Pt/Pv
    train-train edge weighted homophily Hom_k (TRAIN labels only)
    cross-seed behavioral signature stability (Hungarian matching)

Context (per seed):
    D_ctx^f diversity (mask m_ik >= 0.5)
    K x K cosine redundancy matrix
    g_fk vs plain neighbor mean distance
    Q_ifk = cos(f_i, g_ifk) agreement + corr(m,Q) / corr(a,Q)
    fixed Ridge context probes: f, g_bar, [g1..gK], [f|g_bar], [f|g1..gK]
    Delta_relctx^f = Probe([f|g1..gK]) - Probe([f|g_bar])

No test access; frozen model untouched; big tensors chunked over E / N.
Outputs under outputs/perf_r0/{relation,context}/.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r0_utils import (  # noqa: E402
    DATASETS,
    FACTOR_NAMES,
    SEEDS,
    chunked_mean_cos,
    extract_forward,
    load_setup,
    ridge_probe,
    write_csv,
)

OUT_REL = PROJECT_ROOT / "outputs" / "perf_r0" / "relation"
OUT_CTX = PROJECT_ROOT / "outputs" / "perf_r0" / "context"
SUPPORT_MASS = 0.5


# ---------------------------------------------------------------------------
# Relation part
# ---------------------------------------------------------------------------


def _relation_stats(fex: dict, setup) -> dict:
    graph_out, deg = fex["graph_out"], fex["deg"]
    r = graph_out["r"]  # [E, K]
    edge_index = fex["edge_index"]
    src, dst = edge_index[0], edge_index[1]
    k = int(r.size(1))
    n = int(deg.size(0))
    device = r.device
    log_k = float(torch.log(torch.tensor(float(k))).item())

    occ = r.mean(dim=0)  # [K]
    occ = occ / (occ.sum() + 1e-8)
    k_eff = float(torch.exp(-(occ * torch.log(occ + 1e-8)).sum()).item())
    max_conf = float(r.max(dim=-1).values.mean().item())
    h_norm = float((-(r * torch.log(r + 1e-8)).sum(dim=-1)).mean().item()) / log_k

    # support: mass = availability * deg (a = m/(d+eps))
    mass = graph_out["availability"] * deg.unsqueeze(-1)  # [N, K]
    avail = graph_out["availability"]
    out: dict = {
        "k_eff": k_eff, "max_conf": max_conf, "h_norm": h_norm,
        "occ_std": float(occ.std(unbiased=False).item()),
        "mass_mean": float(mass.mean().item()),
        "mass_frac_below_0.5": float((mass < SUPPORT_MASS).float().mean().item()),
        "avail_frac_below_0.05": float((avail < 0.05).float().mean().item()),
    }
    for j in range(k):
        out[f"occ_r{j+1}"] = float(occ[j].item())

    # structural profiles (chunked r-weighted means over E)
    deg_src = deg[src]
    deg_dst = deg[dst]
    logd_src = torch.log1p(deg_src)
    logd_dst = torch.log1p(deg_dst)
    gap = (logd_src - logd_dst).abs()
    raw_sig = setup.model._get_raw_signature(edge_index, n)  # [N, 3] u0,u1,u2
    u_src = raw_sig[src]  # [E, 3]
    u_dst = raw_sig[dst]

    def rw_mean(values: torch.Tensor, weights: torch.Tensor, chunk: int = 500_000) -> float:
        total = weights.sum()
        s = (weights * values).sum() if values.numel() <= chunk * 4 else torch.zeros(())
        if values.numel() > chunk * 4:
            s = torch.zeros((), device=device, dtype=torch.float64)
            for start in range(0, values.numel(), chunk):
                s = s + (weights[start : start + chunk] * values[start : start + chunk]).sum()
        return float((s / (total + 1e-8)).item())

    prof_rows: list[dict] = []
    for j in range(k):
        w = r[:, j]
        row = {"dataset": setup.dataset, "seed": setup.seed, "relation": f"R{j+1}"}
        row["src_logdeg"] = rw_mean(logd_src, w)
        row["dst_logdeg"] = rw_mean(logd_dst, w)
        row["deg_gap"] = rw_mean(gap, w)
        row["u1_src"] = rw_mean(u_src[:, 1], w)
        row["u2_src"] = rw_mean(u_src[:, 2], w)
        row["u1_dst"] = rw_mean(u_dst[:, 1], w)
        row["u2_dst"] = rw_mean(u_dst[:, 2], w)
        row["occupancy"] = float(occ[j].item())
        row["mean_availability"] = float(avail[:, j].mean().item())
        prof_rows.append(row)

    # semantic coherence Sim_{f,k} (chunked)
    factors = fex["factors"]
    sem_rows: list[dict] = []
    for fi, fname in enumerate(FACTOR_NAMES):
        f = factors["c" if fname == "C" else ("p_t" if fname == "Pt" else "p_v")]
        f_src = f[src]
        f_dst = f[dst]
        num = torch.zeros(k, device=device, dtype=torch.float64)
        den = torch.zeros(k, device=device, dtype=torch.float64)
        chunk = 500_000
        for start in range(0, f_src.size(0), chunk):
            a = f_src[start : start + chunk]
            b = f_dst[start : start + chunk]
            cos = (a * b).sum(dim=-1) / ((a.norm(dim=-1) * b.norm(dim=-1)) + 1e-8)
            wchunk = r[start : start + chunk]
            num = num + (wchunk * cos.unsqueeze(-1)).sum(dim=0)
            den = den + wchunk.sum(dim=0)
        sim = num / (den + 1e-8)
        row = {"dataset": setup.dataset, "seed": setup.seed, "factor": fname}
        for j in range(k):
            row[f"sim_R{j+1}"] = float(sim[j].item())
        row["sim_range"] = float((sim.max() - sim.min()).item())
        row["sim_std"] = float(sim.std(unbiased=False).item())
        sem_rows.append(row)

    # train-train homophily (TRAIN labels only)
    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[setup.data.train_idx.to(device)] = True
    y = setup.data.y.to(device)
    tt_mask = train_mask[src] & train_mask[dst]
    hom_rows = []
    if bool(tt_mask.any()):
        same = (y[src] == y[dst]).float()
        hom_row = {"dataset": setup.dataset, "seed": setup.seed}
        hom_vals = []
        for j in range(k):
            w = r[:, j][tt_mask]
            val = float(((w * same[tt_mask]).sum() / (w.sum() + 1e-8)).item())
            hom_row[f"hom_R{j+1}"] = val
            hom_vals.append(val)
        hom_row["hom_range"] = max(hom_vals) - min(hom_vals)
        hom_row["hom_std"] = statistics.pstdev(hom_vals)
        hom_rows.append(hom_row)

    return {"summary": out, "profiles": prof_rows, "semantic": sem_rows, "homophily": hom_rows}


def _hungarian_stability(all_profiles: dict, datasets: list[str]) -> list[dict]:
    """Cross-seed behavioral matching on label-free signatures."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    rows: list[dict] = []
    for dataset in datasets:
        seeds = sorted(all_profiles.get(dataset, {}))
        for ai in range(len(seeds)):
            for bi in range(ai + 1, len(seeds)):
                sa, sb = seeds[ai], seeds[bi]
                pa = {p["relation"]: p for p in all_profiles[dataset][sa]}
                pb = {p["relation"]: p for p in all_profiles[dataset][sb]}
                keys = ["src_logdeg", "dst_logdeg", "deg_gap", "u1_src", "u2_src",
                        "u1_dst", "u2_dst", "occupancy", "mean_availability"]
                rels = sorted(pa)
                a = np.array([[pa[r][key] for key in keys] for r in rels])
                b = np.array([[pb[r][key] for key in keys] for r in rels])
                a = (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-8)
                b = (b - b.mean(axis=0)) / (b.std(axis=0) + 1e-8)
                cos = a @ b.T
                # normalize to a proper cosine similarity
                cos = cos / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :] + 1e-8)
                ri, ci = linear_sum_assignment(-cos)
                matched = [float(cos[i, j]) for i, j in zip(ri, ci)]
                l2 = [float(np.linalg.norm(a[i] - b[j])) for i, j in zip(ri, ci)]
                rows.append({
                    "dataset": dataset, "seed_pair": f"{sa}-{sb}",
                    "matched_cos_mean": statistics.mean(matched),
                    "matched_cos_min": min(matched),
                    "matched_l2_mean": statistics.mean(l2),
                    "matched_l2_max": max(l2),
                })
    return rows


# ---------------------------------------------------------------------------
# Context part
# ---------------------------------------------------------------------------


def _context_stats(fex: dict, setup) -> tuple[dict, list[dict]]:
    """Diversity, redundancy matrix, agreement, per-cell distance-to-mean."""
    from src.models.biaxis_p1_components import neighbor_mean

    graph_out, deg = fex["graph_out"], fex["deg"]
    g_perm = graph_out["g_perm"]  # [N, F, K, d]
    f_block = fex["f_block"]
    n, fnum, k, d = g_perm.shape
    device = g_perm.device
    mass = graph_out["availability"] * deg.unsqueeze(-1)  # [N, K]

    # plain neighbor mean per factor
    g_bar = neighbor_mean(
        fex["edge_index"], f_block.reshape(n, fnum * d), n, edge_chunk_size=setup.model.edge_chunk_size
    ).reshape(n, fnum, d)

    out: dict = {}
    redundancy_rows: list[dict] = []
    for fi, fname in enumerate(FACTOR_NAMES):
        valid = mass >= SUPPORT_MASS  # [N, K]
        g = g_perm[:, fi]  # [N, K, d]
        gbar_f = g_bar[:, fi]  # [N, d]

        # D_ctx^f over nodes with >=2 valid cells
        pairwise_dists = 1.0 - torch.nn.functional.cosine_similarity(
            g.unsqueeze(2), g.unsqueeze(1), dim=-1
        )  # [N, K, K]
        n_valid = valid.sum(dim=-1)  # [N]
        usable = n_valid >= 2
        if bool(usable.any()):
            sel = pairwise_dists[usable]
            vsel = valid[usable]
            tri = torch.triu(sel, diagonal=1)
            mask = torch.triu(
                vsel.unsqueeze(1) & vsel.unsqueeze(2), diagonal=1
            )
            num = (tri * mask.float()).sum(dim=(1, 2))
            den = mask.float().sum(dim=(1, 2))
            d_ctx = float((num / (den + 1e-8)).mean().item())
        else:
            d_ctx = float("nan")
        out[f"D_ctx_{fname}"] = d_ctx

        # K x K redundancy (mean cosine, masked valid pairs)
        cos_mat = torch.nn.functional.cosine_similarity(g.unsqueeze(2), g.unsqueeze(1), dim=-1)
        for k1 in range(k):
            row = {"dataset": setup.dataset, "seed": setup.seed, "factor": fname, "k1": f"R{k1+1}"}
            for k2 in range(k):
                pair_mask = valid[:, k1] & valid[:, k2] & (~torch.eye(k, dtype=torch.bool, device=device)[k1, k2])
                if k1 == k2:
                    row[f"R{k2+1}"] = 1.0
                elif bool(pair_mask.any()):
                    row[f"R{k2+1}"] = float(cos_mat[:, k1, k2][pair_mask].mean().item())
                else:
                    row[f"R{k2+1}"] = float("nan")
            redundancy_rows.append(row)

        # Q_ifk = cos(f_i, g_ifk)
        f_f = f_block[:, fi]  # [N, d]
        q = torch.nn.functional.cosine_similarity(g, f_f.unsqueeze(1), dim=-1)  # [N, K]
        q_row = {"dataset": setup.dataset, "seed": setup.seed, "factor": fname}
        q_vals = []
        for kk in range(k):
            qk = q[:, kk][valid[:, kk]]
            if qk.numel():
                q_row[f"Q_R{kk+1}"] = float(qk.mean().item())
                q_vals.append(float(qk.mean().item()))
        q_row["Q_range"] = (max(q_vals) - min(q_vals)) if q_vals else float("nan")
        # corr(m, Q) and corr(a, Q) over valid cells (flattened)
        m_flat, q_flat = [], []
        a_flat = []
        for kk in range(k):
            v = valid[:, kk]
            if bool(v.any()):
                m_flat.append(mass[v, kk].cpu())
                a_flat.append(graph_out["availability"][v, kk].cpu())
                q_flat.append(q[v, kk].cpu())
        if m_flat:
            import numpy as np

            m_all = torch.cat(m_flat).numpy()
            a_all = torch.cat(a_flat).numpy()
            q_all = torch.cat(q_flat).numpy()
            q_row["corr_mass_Q"] = float(np.corrcoef(m_all, q_all)[0, 1])
            q_row["corr_avail_Q"] = float(np.corrcoef(a_all, q_all)[0, 1])
        out[f"Q_mean_{fname}"] = float(q[valid].mean().item()) if bool(valid.any()) else float("nan")
    return out, redundancy_rows


def _context_probes(fex: dict, setup) -> list[dict]:
    """Fixed Ridge probes per factor: f, g_bar, [g1..gK], [f|g_bar], [f|g1..gK]."""
    from src.models.biaxis_p1_components import neighbor_mean

    graph_out, deg = fex["graph_out"], fex["deg"]
    g_perm = graph_out["g_perm"]
    f_block = fex["f_block"]
    n, fnum, k, d = g_perm.shape
    mass = graph_out["availability"] * deg.unsqueeze(-1)
    g_bar = neighbor_mean(
        fex["edge_index"], f_block.reshape(n, fnum * d), n, edge_chunk_size=setup.model.edge_chunk_size
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
                "dataset": setup.dataset, "seed": setup.seed, "factor": fname,
                "variant": name, "val_acc": probe["val_acc"], "val_macro_f1": probe["val_macro_f1"],
            })
            print(f"[ctx-probe] {setup.dataset:12s} s{setup.seed} {fname} {name:8s} val={probe['val_acc']:.4f}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="R0-D3/D4 relation + context diagnostics")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    datasets = [d for d in (args.datasets or ",".join(DATASETS)).split(",") if d in DATASETS]
    seeds = [int(s) for s in (args.seeds or ",".join(map(str, SEEDS))).split(",") if s]

    OUT_REL.mkdir(parents=True, exist_ok=True)
    OUT_CTX.mkdir(parents=True, exist_ok=True)
    (OUT_CTX / "context_redundancy_matrices").mkdir(parents=True, exist_ok=True)

    rel_summary: list[dict] = []
    rel_profiles: dict = {}
    rel_semantic: list[dict] = []
    rel_hom: list[dict] = []
    ctx_summary: list[dict] = []
    ctx_probe_rows: list[dict] = []

    for di, dataset in enumerate(datasets):
        device = torch.device(f"cuda:{gpus[di % len(gpus)]}")
        rel_profiles[dataset] = {}
        for seed in seeds:
            setup = load_setup(dataset, seed, device)
            fex = extract_forward(setup)
            rel = _relation_stats(fex, setup)
            row = {"dataset": dataset, "seed": seed, **rel["summary"]}
            rel_summary.append(row)
            rel_profiles[dataset][seed] = rel["profiles"]
            rel_semantic.extend(rel["semantic"])
            rel_hom.extend(rel["homophily"])
            ctx, redundancy_rows = _context_stats(fex, setup)
            ctx_summary.append({"dataset": dataset, "seed": seed, **ctx})
            write_csv(
                OUT_CTX / "context_redundancy_matrices" / f"{dataset}_seed{seed}.csv",
                redundancy_rows,
            )
            ctx_probe_rows.extend(_context_probes(fex, setup))
            del fex
            torch.cuda.empty_cache()
            print(f"[rel/ctx] {dataset:12s} s{seed} done", flush=True)

    write_csv(OUT_REL / "relation_assignment_per_seed.csv", rel_summary)
    profile_rows = [
        {**p, "dataset": ds, "seed": seed}
        for ds, seeds_dict in rel_profiles.items() for seed, profs in seeds_dict.items() for p in profs
    ]
    write_csv(OUT_REL / "relation_structural_profiles.csv", profile_rows)
    write_csv(OUT_REL / "relation_semantic_profiles.csv", rel_semantic)
    write_csv(OUT_REL / "relation_train_homophily.csv", rel_hom)
    stability = _hungarian_stability(rel_profiles, datasets)
    write_csv(OUT_REL / "relation_seed_stability.csv", stability)
    write_csv(OUT_CTX / "context_diversity.csv", ctx_summary)
    write_csv(OUT_CTX / "context_probe_per_seed.csv", ctx_probe_rows)

    # context probe summary
    ctx_probe_summary: list[dict] = []
    for dataset in datasets:
        for fname in FACTOR_NAMES:
            for variant in ("f", "g_bar", "g_all", "f|g_bar", "f|g_all"):
                vals = [r["val_acc"] for r in ctx_probe_rows
                        if r["dataset"] == dataset and r["factor"] == fname and r["variant"] == variant]
                if vals:
                    ctx_probe_summary.append({
                        "dataset": dataset, "factor": fname, "variant": variant,
                        "val_acc": statistics.mean(vals),
                        "val_acc_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    })
    write_csv(OUT_CTX / "context_probe_summary.csv", ctx_probe_summary)

    # relation report
    lines = ["# R0-RELATION-REPORT — Structural Relation Quality", ""]
    lines.append("> per-seed detail: relation_*.csv；以下为 3-seed 均值。")
    lines.append("")
    lines.append("| dataset | K_eff | Conf_R | H_norm | occ_std | mass<0.5 frac | sim_C range | sim_Pt range | sim_Pv range | hom range | seed stab cos |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    stab_mean = {}
    for r in stability:
        stab_mean.setdefault(r["dataset"], []).append(r["matched_cos_mean"])
    for dataset in datasets:
        rs = [r for r in rel_summary if r["dataset"] == dataset]
        def m(key):
            vals = [r[key] for r in rs if r[key] is not None]
            return statistics.mean(vals) if vals else float("nan")
        sem = [r for r in rel_semantic if r["dataset"] == dataset]
        hom = [r for r in rel_hom if r["dataset"] == dataset]
        def sem_range(fname):
            vals = [r["sim_range"] for r in sem if r["factor"] == fname]
            return statistics.mean(vals) if vals else float("nan")
        hom_range = statistics.mean([r["hom_range"] for r in hom]) if hom else float("nan")
        stab = statistics.mean(stab_mean.get(dataset, [float("nan")])) if stab_mean.get(dataset) else float("nan")
        lines.append(
            f"| {dataset} | {m('k_eff'):.2f} | {m('max_conf'):.3f} | {m('h_norm'):.3f} | "
            f"{m('occ_std'):.3f} | {m('mass_frac_below_0.5'):.3f} | {sem_range('C'):.3f} | "
            f"{sem_range('Pt'):.3f} | {sem_range('Pv'):.3f} | {hom_range:.3f} | {stab:.3f} |"
        )
    lines.append("")
    (OUT_REL / "R0_RELATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # context report
    lines = ["# R0-CONTEXT-REPORT — Relation-conditioned Context Quality", ""]
    lines.append("> Δ_relctx^f = Probe([f|g1..gK]) − Probe([f|g_bar])（3-seed 均值，pp）。")
    lines.append("")
    lines.append("| dataset | D_ctx^C | D_ctx^Pt | D_ctx^Pv | Δ_relctx^C | Δ_relctx^Pt | Δ_relctx^Pv |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        cs = [r for r in ctx_summary if r["dataset"] == dataset]
        def m(key):
            vals = [r[key] for r in cs if r[key] is not None and r[key] == r[key]]
            return statistics.mean(vals) if vals else float("nan")
        def delta(fname):
            a = [r["val_acc"] for r in ctx_probe_summary if r["dataset"] == dataset and r["factor"] == fname and r["variant"] == "f|g_all"]
            b = [r["val_acc"] for r in ctx_probe_summary if r["dataset"] == dataset and r["factor"] == fname and r["variant"] == "f|g_bar"]
            if a and b:
                return 100 * (a[0] - b[0])
            return float("nan")
        lines.append(
            f"| {dataset} | {m('D_ctx_C'):.4f} | {m('D_ctx_Pt'):.4f} | {m('D_ctx_Pv'):.4f} | "
            f"{delta('C'):+.2f} | {delta('Pt'):+.2f} | {delta('Pv'):+.2f} |"
        )
    lines.append("")
    (OUT_CTX / "R0_CONTEXT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[rel/ctx] done -> {OUT_REL}, {OUT_CTX}", flush=True)


if __name__ == "__main__":
    main()
