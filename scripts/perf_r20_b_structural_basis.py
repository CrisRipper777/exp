"""R2-0B: Explicit Structural Function Basis Probe (user Prompt B).

Frozen OFR checkpoints only; no training; no Test (r20 wrapper masks test
labels / cuts test_idx). Fixed probe = R0 ridge_probe (StandardScaler +
RidgeClassifier(alpha=1.0), fit TRAIN / eval VAL).

Explicit basis (deterministic, topology-only; Splus reused unchanged):
    G1    = P F           1-hop ordinary        (== neighbor_mean)
    G2    = P G1          2-hop diffusion       (no explicit 2-hop edge list)
    Gsim  = weighted_neighbor_mean(w_sim, F),  w_sim  = (1+c_ji)/2 + 1e-8
    Gdiff = weighted_neighbor_mean(w_diff, F), w_diff = (1-c_ji)/2 + 1e-8
    c_ji = cos(Splus_j, Splus_i) on observed edges j->i.

Levels:
    Level-I   single-channel decomposition (per-factor + joint), all dim-matched
    Level-II  current K=4 latent contexts vs explicit K=4 basis
              (per-factor 5d, joint 15d — strictly matched)
    Level-III z_final residual structural headroom ([Z|B_x], h+3d / h+12d)
    Level-IV  fixed shuffle negative control (perm seed 20260904)
    secondary D_ctx / 4x4 redundancy (NOT used for GO/NO-GO)

Outputs: outputs/perf_r20/structural_basis/{csv x9, R20_B_STRUCTURAL_BASIS_REPORT.md}
Usage:
    python scripts/perf_r20_b_structural_basis.py --gpus 0,1
    python scripts/perf_r20_b_structural_basis.py --gpus 0 --stage smoke
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r20_utils import (  # noqa: E402
    DATASETS,
    SEEDS,
    assert_feature_dim,
    compute_splus,
    context_concat,
    explicit_channels,
    extract_forward,
    factor_tensor,
    load_setup,
    raw_splus,
    ridge_probe,
    structural_edge_weights,
    write_csv,
)

OUT_ROOT = PROJECT_ROOT / "outputs" / "perf_r20" / "structural_basis"
FACTORS = ["C", "Pt", "Pv"]
CHANNELS = ["G1", "G2", "Gsim", "Gdiff"]
SHUFFLE_SEED = 20260904

# ---------------------------------------------------------------------------
# Pre-registered thresholds (user §十二)
# ---------------------------------------------------------------------------
B_STRONG_PP = 0.50
B_GO_PP = 0.30
B_WEAK_PP = 0.15
POS_SEEDS_MIN = 2
H_EXPLICIT_STABLE_PP = 0.10      # "H_explicit_final 有稳定正增益" operationalization
ATTRIB_PP = 0.10                 # channel attribution signal floor (joint level)


def _stat(values: list[float]) -> tuple[float, float, int]:
    mean = statistics.mean(values) if values else float("nan")
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    pos = sum(1 for v in values if v > 0)
    return mean, std, pos


def _probe(tensor: torch.Tensor, setup, tag: str) -> tuple[float, float]:
    probe = ridge_probe(tensor, setup)
    print(f"[B] {setup.dataset:12s} s{setup.seed} {tag:26s} acc={probe['val_acc']:.4f}", flush=True)
    return probe["val_acc"], probe["val_macro_f1"]


def _d_ctx(channels: list[torch.Tensor], valid: torch.Tensor) -> tuple[float, int]:
    """Mean pairwise 1-cos over nodes with >=2 valid channels (upper triangle).

    channels: list of [N, d]; valid: [N, K] bool (False -> cell excluded).
    """
    g = torch.stack(channels, dim=1)  # [N, K, d]
    k = g.size(1)
    pairwise = 1.0 - F.cosine_similarity(g.unsqueeze(2), g.unsqueeze(1), dim=-1)  # [N,K,K]
    n_valid = valid.sum(dim=-1)
    usable = n_valid >= 2
    if not bool(usable.any()):
        return float("nan"), 0
    sel = pairwise[usable]
    vsel = valid[usable]
    tri = torch.triu(sel, diagonal=1)
    mask = torch.triu(vsel.unsqueeze(1) & vsel.unsqueeze(2), diagonal=1)
    num = (tri * mask.float()).sum(dim=(1, 2))
    den = mask.float().sum(dim=(1, 2))
    return float((num / (den + 1e-8)).mean().item()), int(usable.sum().item())


def _redundancy(channels: list[torch.Tensor], valid: torch.Tensor, names: list[str]) -> list[dict]:
    """4x4 mean-cosine matrix (diag = 1) under the both-valid mask."""
    g = torch.stack(channels, dim=1)  # [N, K, d]
    k = g.size(1)
    cos_mat = F.cosine_similarity(g.unsqueeze(2), g.unsqueeze(1), dim=-1)  # [N,K,K]
    rows = []
    for k1 in range(k):
        for k2 in range(k):
            if k1 == k2:
                rows.append((names[k1], names[k2], 1.0))
                continue
            pair = valid[:, k1] & valid[:, k2]
            rows.append((
                names[k1], names[k2],
                float(cos_mat[:, k1, k2][pair].mean().item()) if bool(pair.any()) else float("nan"),
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-0B structural function basis probe")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--stage", default="all", choices=["smoke", "full", "all"])
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    datasets = [d for d in (args.datasets or ",".join(DATASETS)).split(",") if d in DATASETS]
    seeds = [int(s) for s in (args.seeds or ",".join(map(str, SEEDS))).split(",") if s]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    sig_rows: list[dict] = []          # structural_signature_stats
    sc_pf: list[dict] = []             # single_channel_per_factor
    sc_joint: list[dict] = []          # single_channel_joint
    ctx_pf: list[dict] = []            # context_probe_per_factor
    ctx_joint: list[dict] = []         # context_probe_joint
    final_rows: list[dict] = []        # final_residual
    shuf_rows: list[dict] = []         # shuffle_control
    div_rows: list[dict] = []          # diversity
    red_rows: list[dict] = []          # redundancy

    if args.stage == "all":
        lifecycles = [(ds, s) for ds in datasets for s in seeds]
    elif args.stage == "smoke":
        lifecycles = [("Movies", 42)]
    else:
        lifecycles = [(ds, s) for ds in datasets for s in seeds]
    # all reporting / verdict tables only see the datasets actually run
    datasets = sorted({ds for ds, _ in lifecycles})

    for idx, (ds, s) in enumerate(lifecycles):
        device = torch.device(f"cuda:{gpus[idx % len(gpus)]}")
        setup = load_setup(ds, s, device)
        fex = extract_forward(setup)
        n = int(fex["f_block"].size(0))
        d = int(fex["f_block"].size(2))
        hidden = int(fex["z_final"].size(1))
        k = int(fex["graph_out"]["g_perm"].size(2))
        assert k == 4 and d == 128, f"frozen M2 K=4 / d_f=128 expected, got K={k} d={d}"
        edge_index = fex["edge_index"]
        chunk = setup.model.edge_chunk_size

        F_dict = {f: factor_tensor(fex, f) for f in FACTORS}
        L = context_concat([F_dict[f] for f in FACTORS])
        Z = fex["z_final"]
        g_perm = fex["graph_out"]["g_perm"]
        deg = fex["deg"]

        raw = raw_splus(edge_index, n, edge_chunk_size=chunk)
        splus = compute_splus(edge_index, n, edge_chunk_size=chunk)
        ch = {f: explicit_channels(edge_index, F_dict[f], splus, n, edge_chunk_size=chunk) for f in FACTORS}
        g_cur = {f: g_perm[:, i].reshape(n, -1) for i, f in enumerate(FACTORS)}
        w_sim, w_diff = structural_edge_weights(edge_index, splus)

        perm = torch.randperm(n, generator=torch.Generator().manual_seed(SHUFFLE_SEED)).to(device)

        # --- signature stats --------------------------------------------------
        row = {"dataset": ds, "seed": s}
        col_names = ["u0", "u1", "u2", "u3", "mu_d", "std_d", "mu_gap", "std_gap"]
        for ci, cname in enumerate(col_names):
            col = raw[:, ci]
            row[f"{cname}_mean"] = float(col.mean().item())
            row[f"{cname}_std"] = float(col.std(unbiased=False).item())
            row[f"{cname}_min"] = float(col.min().item())
            row[f"{cname}_max"] = float(col.max().item())
        norms = splus.norm(dim=-1)
        row["norm_mean"] = float(norms.mean().item())
        row["norm_std"] = float(norms.std(unbiased=False).item())
        row["norm_min"] = float(norms.min().item())
        row["norm_max"] = float(norms.max().item())
        qs = torch.quantile(norms, torch.tensor([0.1, 0.5, 0.9], device=norms.device))
        row["norm_p10"] = float(qs[0].item())
        row["norm_p50"] = float(qs[1].item())
        row["norm_p90"] = float(qs[2].item())
        row["norm_frac_zero"] = float((norms < 1e-6).float().mean().item())
        row["n_isolated"] = int((deg <= 0).sum().item())
        row["finite_splus"] = bool(torch.isfinite(splus).all())
        row["finite_w_sim"] = bool(torch.isfinite(w_sim).all())
        row["finite_w_diff"] = bool(torch.isfinite(w_diff).all())
        row["finite_channels"] = bool(
            all(torch.isfinite(ch[f][c]).all() for f in FACTORS for c in CHANNELS)
        )
        sig_rows.append(row)

        # --- Level-I per-factor single channels --------------------------------
        local_acc, local_f1 = {}, {}
        g1_acc, g1_f1 = {}, {}
        for f in FACTORS:
            a, b = _probe(F_dict[f], setup, f"F[{f}]")
            local_acc[f], local_f1[f] = a, b
            sc_pf.append({
                "dataset": ds, "seed": s, "factor": f, "channel": "local",
                "val_acc": a, "val_macro_f1": b, "U_acc": "", "U_f1": "",
                "delta_vs_G1_acc": "", "delta_vs_G1_f1": "",
            })
        for f in FACTORS:
            for c in CHANNELS:
                block = context_concat([F_dict[f], ch[f][c]])
                assert_feature_dim(block, n, 2 * d, f"single {f} {c}")
                a, b = _probe(block, setup, f"[F[{f}]|{c}]")
                u_a = a - local_acc[f]
                u_b = b - local_f1[f]
                d_a = a - g1_acc[f] if f in g1_acc else ""
                d_b = b - g1_f1[f] if f in g1_acc else ""
                sc_pf.append({
                    "dataset": ds, "seed": s, "factor": f, "channel": c,
                    "val_acc": a, "val_macro_f1": b,
                    "U_acc": u_a, "U_f1": u_b,
                    "delta_vs_G1_acc": d_a, "delta_vs_G1_f1": d_b,
                })
                if c == "G1":
                    g1_acc[f], g1_f1[f] = a, b
        # fix G1 rows' delta (empty at write time above)
        for r in sc_pf:
            if r["dataset"] == ds and r["seed"] == s and r["channel"] == "G1":
                r["delta_vs_G1_acc"], r["delta_vs_G1_f1"] = 0.0, 0.0

        # --- Level-I joint single channels -------------------------------------
        j_acc, j_f1 = {}, {}
        for c in CHANNELS:
            b_c = context_concat([ch[f][c] for f in FACTORS])
            assert_feature_dim(b_c, n, 3 * d, f"B_{c}")
            block = context_concat([L, b_c])
            assert_feature_dim(block, n, 6 * d, f"joint {c}")
            a, b = _probe(block, setup, f"[L|B_{c}]")
            j_acc[c], j_f1[c] = a, b
            d_a = a - j_acc["G1"] if c != "G1" else ""
            d_b = b - j_f1["G1"] if c != "G1" else ""
            sc_joint.append({
                "dataset": ds, "seed": s, "channel": c,
                "val_acc": a, "val_macro_f1": b,
                "delta_vs_G1_acc": d_a, "delta_vs_G1_f1": d_b,
            })
        for r in sc_joint:
            if r["dataset"] == ds and r["seed"] == s and r["channel"] == "G1":
                r["delta_vs_G1_acc"], r["delta_vs_G1_f1"] = 0.0, 0.0

        # --- Level-II per-factor current vs explicit ---------------------------
        for f in FACTORS:
            cur = context_concat([F_dict[f], g_cur[f]])
            assert_feature_dim(cur, n, (1 + k) * d, f"per-factor current {f}")
            cur_a, cur_b = _probe(cur, setup, f"current[{f}] 5d")
            exp = context_concat([F_dict[f]] + [ch[f][c] for c in CHANNELS])
            assert_feature_dim(exp, n, 5 * d, f"per-factor explicit {f}")
            exp_a, exp_b = _probe(exp, setup, f"explicit[{f}] 5d")
            ctx_pf.append({
                "dataset": ds, "seed": s, "factor": f,
                "current_acc": cur_a, "current_f1": cur_b,
                "explicit_acc": exp_a, "explicit_f1": exp_b,
                "delta_B_acc": exp_a - cur_a, "delta_B_f1": exp_b - cur_b,
            })

        # --- Level-II joint (+ intermediate shuffle) ----------------------------
        B_current = context_concat([g_cur[f] for f in FACTORS])
        B_explicit = context_concat(
            [context_concat([ch[f][c] for c in CHANNELS]) for f in FACTORS]
        )
        assert_feature_dim(B_current, n, 12 * d, "B_current")
        assert_feature_dim(B_explicit, n, 12 * d, "B_explicit")
        joint_cur = context_concat([L, B_current])
        joint_exp = context_concat([L, B_explicit])
        assert_feature_dim(joint_cur, n, 15 * d, "joint current")
        assert_feature_dim(joint_exp, n, 15 * d, "joint explicit")
        cur_a, cur_b = _probe(joint_cur, setup, "current_joint 15d")
        exp_a, exp_b = _probe(joint_exp, setup, "explicit_joint 15d")
        ctx_joint.append({
            "dataset": ds, "seed": s,
            "current_acc": cur_a, "current_f1": cur_b,
            "explicit_acc": exp_a, "explicit_f1": exp_b,
            "delta_B_intermediate_acc": exp_a - cur_a,
            "delta_B_intermediate_f1": exp_b - cur_b,
        })
        for basis, block_sh, tag in (
            ("current", context_concat([L, B_current[perm]]), "inter_sh[current]"),
            ("explicit", context_concat([L, B_explicit[perm]]), "inter_sh[explicit]"),
        ):
            sa, sb = _probe(block_sh, setup, tag)
            shuf_rows.append({
                "dataset": ds, "seed": s, "level": "intermediate", "basis": basis,
                "real_acc": cur_a if basis == "current" else exp_a,
                "shuf_acc": sa,
                "diff_acc": (cur_a if basis == "current" else exp_a) - sa,
                "real_f1": cur_b if basis == "current" else exp_b,
                "shuf_f1": sb,
                "diff_f1": (cur_b if basis == "current" else exp_b) - sb,
            })

        # --- Level-III z_final residual ----------------------------------------
        z_acc, z_f1 = _probe(Z, setup, "z_final")
        h_c, h_f = {}, {}
        for c in CHANNELS:
            b_c = context_concat([ch[f][c] for f in FACTORS])
            block = context_concat([Z, b_c])
            assert_feature_dim(block, n, hidden + 3 * d, f"final {c}")
            a, b = _probe(block, setup, f"[Z|B_{c}]")
            h_c[c], h_f[c] = a, b
        z_cur = context_concat([Z, B_current])
        z_exp = context_concat([Z, B_explicit])
        assert_feature_dim(z_cur, n, hidden + 12 * d, "final current")
        assert_feature_dim(z_exp, n, hidden + 12 * d, "final explicit")
        h_cur_acc, h_cur_f1 = _probe(z_cur, setup, "[Z|B_current]")
        h_exp_acc, h_exp_f1 = _probe(z_exp, setup, "[Z|B_explicit]")
        final_rows.append({
            "dataset": ds, "seed": s,
            "z_acc": z_acc, "z_f1": z_f1,
            **{f"{c}_acc": h_c[c] for c in CHANNELS},
            **{f"{c}_f1": h_f[c] for c in CHANNELS},
            "current_acc": h_cur_acc, "current_f1": h_cur_f1,
            "explicit_acc": h_exp_acc, "explicit_f1": h_exp_f1,
            **{f"H_{c}_acc": h_c[c] - z_acc for c in CHANNELS},
            **{f"H_{c}_f1": h_f[c] - z_f1 for c in CHANNELS},
            "H_current_acc": h_cur_acc - z_acc, "H_current_f1": h_cur_f1 - z_f1,
            "H_explicit_acc": h_exp_acc - z_acc, "H_explicit_f1": h_exp_f1 - z_f1,
            "delta_hop_acc": h_c["G2"] - h_c["G1"], "delta_hop_f1": h_f["G2"] - h_f["G1"],
            "delta_sim_acc": h_c["Gsim"] - h_c["G1"], "delta_sim_f1": h_f["Gsim"] - h_f["G1"],
            "delta_diff_acc": h_c["Gdiff"] - h_c["G1"], "delta_diff_f1": h_f["Gdiff"] - h_f["G1"],
            "delta_B_final_acc": h_exp_acc - h_cur_acc,
            "delta_B_final_f1": h_exp_f1 - h_cur_f1,
        })
        for basis, block_sh, tag in (
            ("current", context_concat([Z, B_current[perm]]), "final_sh[current]"),
            ("explicit", context_concat([Z, B_explicit[perm]]), "final_sh[explicit]"),
        ):
            sa, sb = _probe(block_sh, setup, tag)
            shuf_rows.append({
                "dataset": ds, "seed": s, "level": "final", "basis": basis,
                "real_acc": h_cur_acc if basis == "current" else h_exp_acc,
                "shuf_acc": sa,
                "diff_acc": (h_cur_acc if basis == "current" else h_exp_acc) - sa,
                "real_f1": h_cur_f1 if basis == "current" else h_exp_f1,
                "shuf_f1": sb,
                "diff_f1": (h_cur_f1 if basis == "current" else h_exp_f1) - sb,
            })

        # --- secondary diversity / redundancy -----------------------------------
        availability = fex["graph_out"]["availability"]
        mass = availability * deg.unsqueeze(-1)  # [N, K] (R0 convention)
        for basis in ("current", "explicit"):
            for fi, f in enumerate(FACTORS):
                if basis == "current":
                    chans = [g_perm[:, fi, kk] for kk in range(k)]
                    valid = mass >= 0.5  # R0 support convention
                    names = [f"R{kk + 1}" for kk in range(k)]
                else:
                    chans = [ch[f][c] for c in CHANNELS]
                    valid = torch.stack(
                        [ch[f][c].norm(dim=-1) > 1e-6 for c in CHANNELS], dim=1
                    )
                    names = CHANNELS
                d_ctx, n_usable = _d_ctx(chans, valid)
                div_rows.append({
                    "dataset": ds, "seed": s, "basis": basis, "factor": f,
                    "d_ctx": d_ctx, "n_usable": n_usable,
                })
                for k1_name, k2_name, cos_val in _redundancy(chans, valid, names):
                    red_rows.append({
                        "dataset": ds, "seed": s, "basis": basis, "factor": f,
                        "k1": k1_name, "k2": k2_name, "mean_cos": cos_val,
                    })

        del fex
        torch.cuda.empty_cache()
        print(f"[B] {ds:12s} s{s} done ({idx + 1}/{len(lifecycles)})", flush=True)

    # ------------------------------------------------------------------ CSVs
    write_csv(OUT_ROOT / "structural_signature_stats.csv", sig_rows)
    write_csv(OUT_ROOT / "structural_single_channel_per_factor.csv", sc_pf)
    write_csv(OUT_ROOT / "structural_single_channel_joint.csv", sc_joint)
    write_csv(OUT_ROOT / "structural_context_probe_per_factor.csv", ctx_pf)
    write_csv(OUT_ROOT / "structural_context_probe_joint.csv", ctx_joint)
    write_csv(OUT_ROOT / "structural_final_residual.csv", final_rows)
    write_csv(OUT_ROOT / "structural_shuffle_control.csv", shuf_rows)
    write_csv(OUT_ROOT / "structural_context_diversity.csv", div_rows)
    write_csv(OUT_ROOT / "structural_context_redundancy.csv", red_rows)
    print(f"[B] CSVs written -> {OUT_ROOT}", flush=True)

    # ------------------------------------------------------------------ report
    pp = lambda v: 100.0 * v

    def ds_stats(values_by_seed: dict[tuple, float]) -> dict[str, tuple]:
        out = {}
        for ds in datasets:
            vals = [values_by_seed[(ds, s)] for s in seeds if (ds, s) in values_by_seed]
            if vals:
                out[ds] = _stat(vals)
        return out

    def macro_mean(ds_dict: dict[str, tuple]) -> float:
        means = [v[0] for v in ds_dict.values() if v[0] == v[0]]
        return statistics.mean(means) if means else float("nan")

    def fmt(stat):
        m, s, p = stat
        return f"{pp(m):+.2f}±{pp(s):.2f} ({p})"

    def fmt_macro(ds_dict: dict[str, tuple]) -> str:
        """M/T/G macro: mean of dataset means ± std of dataset means."""
        means = [v[0] for v in ds_dict.values() if v[0] == v[0]]
        m = statistics.mean(means) if means else float("nan")
        s = statistics.pstdev(means) if len(means) > 1 else 0.0
        return f"{pp(m):+.2f}±{pp(s):.2f}"

    # Level-I per factor: U and delta_vs_G1 per (factor, channel)
    u_pf, d_pf = {}, {}
    for f in FACTORS:
        for c in CHANNELS:
            u_vals = {(ds, s): next(r["U_acc"] for r in sc_pf if r["dataset"] == ds and r["seed"] == s and r["factor"] == f and r["channel"] == c)
                      for ds, s in lifecycles}
            d_vals = {(ds, s): next(r["delta_vs_G1_acc"] for r in sc_pf if r["dataset"] == ds and r["seed"] == s and r["factor"] == f and r["channel"] == c)
                      for ds, s in lifecycles}
            u_pf[(f, c)] = ds_stats(u_vals)
            d_pf[(f, c)] = ds_stats(d_vals)

    # Level-I joint deltas
    d_joint = {}
    for c in ("G2", "Gsim", "Gdiff"):
        d_vals = {(ds, s): next(r["delta_vs_G1_acc"] for r in sc_joint if r["dataset"] == ds and r["seed"] == s and r["channel"] == c)
                  for ds, s in lifecycles}
        d_joint[c] = ds_stats(d_vals)
    g1_joint = {(ds, s): next(r["val_acc"] for r in sc_joint if r["dataset"] == ds and r["seed"] == s and r["channel"] == "G1")
                for ds, s in lifecycles}

    # Level-II
    db_pf = {}
    for f in FACTORS:
        vals = {(ds, s): next(r["delta_B_acc"] for r in ctx_pf if r["dataset"] == ds and r["seed"] == s and r["factor"] == f)
                for ds, s in lifecycles}
        db_pf[f] = ds_stats(vals)
    db_inter = ds_stats({(ds, s): next(r["delta_B_intermediate_acc"] for r in ctx_joint if r["dataset"] == ds and r["seed"] == s)
                         for ds, s in lifecycles})

    # Level-III
    h_fin = {c: ds_stats({(ds, s): next(r[f"H_{c}_acc"] for r in final_rows if r["dataset"] == ds and r["seed"] == s)
                          for ds, s in lifecycles}) for c in CHANNELS}
    h_cur = ds_stats({(ds, s): next(r["H_current_acc"] for r in final_rows if r["dataset"] == ds and r["seed"] == s)
                      for ds, s in lifecycles})
    h_exp = ds_stats({(ds, s): next(r["H_explicit_acc"] for r in final_rows if r["dataset"] == ds and r["seed"] == s)
                      for ds, s in lifecycles})
    d_hop_fin = ds_stats({(ds, s): next(r["delta_hop_acc"] for r in final_rows if r["dataset"] == ds and r["seed"] == s)
                          for ds, s in lifecycles})
    d_sim_fin = ds_stats({(ds, s): next(r["delta_sim_acc"] for r in final_rows if r["dataset"] == ds and r["seed"] == s)
                          for ds, s in lifecycles})
    d_diff_fin = ds_stats({(ds, s): next(r["delta_diff_acc"] for r in final_rows if r["dataset"] == ds and r["seed"] == s)
                           for ds, s in lifecycles})
    db_final = ds_stats({(ds, s): next(r["delta_B_final_acc"] for r in final_rows if r["dataset"] == ds and r["seed"] == s)
                         for ds, s in lifecycles})

    # Level-IV shuffle
    shuf_stats = {}
    for level in ("intermediate", "final"):
        for basis in ("current", "explicit"):
            vals = {(ds, s): next(r["diff_acc"] for r in shuf_rows if r["dataset"] == ds and r["seed"] == s and r["level"] == level and r["basis"] == basis)
                    for ds, s in lifecycles}
            shuf_stats[(level, basis)] = ds_stats(vals)

    # ------------------------------------------------------------------ verdicts
    mean_MTG_inter = macro_mean(db_inter)
    inter_ds_pos = sum(1 for ds in datasets if db_inter[ds][0] > 0)
    inter_ds_qual = sum(1 for ds in datasets if db_inter[ds][0] > 0 and db_inter[ds][2] >= POS_SEEDS_MIN)
    if mean_MTG_inter >= B_STRONG_PP / 100 and inter_ds_qual >= 2:
        verdict_inter = "STRONG"
    elif mean_MTG_inter >= B_GO_PP / 100 and inter_ds_pos >= 2:
        verdict_inter = "GO"
    elif mean_MTG_inter >= B_WEAK_PP / 100:
        verdict_inter = "WEAK"
    else:
        verdict_inter = "NO-GO"

    mean_MTG_final = macro_mean(db_final)
    final_ds_pos = sum(1 for ds in datasets if db_final[ds][0] > 0)
    final_ds_qual = sum(1 for ds in datasets if db_final[ds][0] > 0 and db_final[ds][2] >= POS_SEEDS_MIN)
    h_exp_stable = (
        macro_mean(h_exp) >= H_EXPLICIT_STABLE_PP / 100
        and sum(1 for ds in datasets if h_exp[ds][0] > 0) >= 2
    )
    shuf_final_exp = shuf_stats[("final", "explicit")]
    shuf_stable = macro_mean(shuf_final_exp) > 0 and sum(1 for ds in datasets if shuf_final_exp[ds][0] > 0) >= 2
    if mean_MTG_final >= B_STRONG_PP / 100 and final_ds_qual >= 2 and h_exp_stable and shuf_stable:
        verdict_final = "STRONG FINAL"
    elif mean_MTG_final >= B_GO_PP / 100 and final_ds_pos >= 2 and (h_exp_stable or shuf_stable):
        verdict_final = "GO FINAL"
    elif mean_MTG_final >= B_WEAK_PP / 100:
        verdict_final = "WEAK FINAL"
    else:
        verdict_final = "NO FINAL HEADROOM"

    inter_pos = verdict_inter in ("STRONG", "GO")
    final_pos = verdict_final in ("STRONG FINAL", "GO FINAL")
    if inter_pos and final_pos:
        case = "B-1（Intermediate STRONG + Final STRONG/GO：current M2 / structural evidence 是明确的 final architecture bottleneck）"
    elif inter_pos and not final_pos:
        case = "B-2（Intermediate STRONG + Final NO-GO：A0 最终表示已基本吸收；若重构必须改变 end-to-end inductive bias，而非 feature concat）"
    elif (not inter_pos) and final_pos:
        case = "B-3（Intermediate WEAK + Final STRONG：explicit structure 暴露 z_final 遗漏信息，仍支持 R2 structural redesign）"
    else:
        case = "B-4（两者都弱：hand-designed structural basis 未证明优于 M2；R2 不应仅靠扩充 structural channels）"

    # channel attribution (joint-level, pre-registered ATTRIB_PP / POS_SEEDS_MIN)
    signals = [c for c in ("G2", "Gsim", "Gdiff")
               if macro_mean(d_joint[c]) >= ATTRIB_PP / 100
               and sum(1 for ds in datasets if d_joint[c][ds][0] > 0 and d_joint[c][ds][2] >= POS_SEEDS_MIN) >= 1]
    if signals:
        role = {"G2": "multi-hop / receptive-field", "Gsim": "structurally-similar neighbor selection",
                "Gdiff": "structural contrast / role diversity"}
        attribution = " + ".join(f"{c}（{role[c]}）" for c in signals)
    else:
        full_vs_g1 = {(ds, s): next(r["explicit_acc"] for r in ctx_joint if r["dataset"] == ds and r["seed"] == s) - g1_joint[(ds, s)]
                      for ds, s in lifecycles}
        fvg = ds_stats(full_vs_g1)
        if macro_mean(fvg) >= ATTRIB_PP / 100 and sum(1 for ds in datasets if fvg[ds][0] > 0 and fvg[ds][2] >= POS_SEEDS_MIN) >= 1:
            attribution = "complementary combination（full 4-channel basis 优于 G1，但单 channel 无稳定信号）"
        else:
            attribution = "1-hop only（无 channel 在 G1 之上提供稳定增益）"

    # ------------------------------------------------------------------ report
    lines = ["# R20-B STRUCTURAL BASIS REPORT", ""]
    lines.append("## 结论（Top）")
    lines.append("")
    lines.append(f"- **Intermediate verdict：{verdict_inter}**（Δ_B_intermediate M/T/G macro = {pp(mean_MTG_inter):+.2f}pp）")
    lines.append(f"- **Final-residual verdict：{verdict_final}**（Δ_B_final M/T/G macro = {pp(mean_MTG_final):+.2f}pp）")
    lines.append(f"- **Gain channel：{attribution}**")
    lines.append(f"- **Case：{case}**")
    lines.append("")
    lines.append("> 协议：frozen OFR checkpoints（M/T/G × 42/43/44）；固定 Ridge probe（StandardScaler +")
    lines.append("> RidgeClassifier(alpha=1.0)，TRAIN fit / VAL eval）；无 Test（test labels masked + test_idx 切断）。")
    lines.append("> 数值为 Val Acc 的 pp（×100），3-seed mean±population std(ddof=0)，括号 = positive seeds。")
    lines.append("> R2-0 审计：Audit PASS（129/129，max diff 5.96e-8）。")
    lines.append("")

    lines.append("## Level-I — Single-channel decomposition（全部 2·d_f 维匹配）")
    lines.append("")
    lines.append("### per-factor U = Probe([F|c]) − Probe(F)")
    lines.append("")
    lines.append("| dataset | factor | G1 | G2 | Gsim | Gdiff |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for ds in datasets:
        for f in FACTORS:
            lines.append(f"| {ds} | {f} | " + " | ".join(fmt(u_pf[(f, c)][ds]) for c in CHANNELS) + " |")
    lines.append("")
    lines.append("### per-factor Delta vs G1 = Probe([F|c]) − Probe([F|G1])")
    lines.append("")
    lines.append("| dataset | factor | G2 | Gsim | Gdiff |")
    lines.append("|---|---|---:|---:|---:|")
    for ds in datasets:
        for f in FACTORS:
            lines.append(f"| {ds} | {f} | " + " | ".join(fmt(d_pf[(f, c)][ds]) for c in ("G2", "Gsim", "Gdiff")) + " |")
    lines.append("")
    lines.append("### joint Delta vs G1（6·d_f 维匹配）")
    lines.append("")
    lines.append("| dataset | Δ_hop_joint (G2−G1) | Δ_sim_joint (Gsim−G1) | Δ_diff_joint (Gdiff−G1) |")
    lines.append("|---|---:|---:|---:|")
    for ds in datasets:
        lines.append(f"| {ds} | {fmt(d_joint['G2'][ds])} | {fmt(d_joint['Gsim'][ds])} | {fmt(d_joint['Gdiff'][ds])} |")
    lines.append(f"| **M/T/G macro** | {fmt_macro(d_joint['G2'])} | {fmt_macro(d_joint['Gsim'])} | {fmt_macro(d_joint['Gdiff'])} |")
    lines.append("")

    lines.append("## Level-II — Current K4 vs Explicit K4（intermediate）")
    lines.append("")
    lines.append("### per-factor Δ_B^f = Probe(explicit 5d) − Probe(current 5d)")
    lines.append("")
    lines.append("| dataset | Δ_B^C | Δ_B^Pt | Δ_B^Pv |")
    lines.append("|---|---:|---:|---:|")
    for ds in datasets:
        lines.append(f"| {ds} | {fmt(db_pf['C'][ds])} | {fmt(db_pf['Pt'][ds])} | {fmt(db_pf['Pv'][ds])} |")
    lines.append("")
    lines.append("### joint Δ_B_intermediate = Probe(explicit_joint 15d) − Probe(current_joint 15d)")
    lines.append("")
    lines.append("| dataset | Δ_B_intermediate | positive seeds |")
    lines.append("|---|---:|---:|")
    for ds in datasets:
        m, s, p = db_inter[ds]
        lines.append(f"| {ds} | {fmt((m, s, p))} | {p}/3 |")
    lines.append(f"| **M/T/G macro** | **{fmt_macro(db_inter)}**（positive datasets {inter_ds_pos}/3） | |")
    lines.append("")

    lines.append("## Level-III — z_final residual structural headroom")
    lines.append("")
    lines.append("### H_x_final = Probe([Z|B_x]) − Probe(Z)（h+3d；all 三 factor 拼接）")
    lines.append("")
    lines.append("| dataset | H_G1 | H_G2 | H_Gsim | H_Gdiff | H_current | H_explicit |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ds in datasets:
        lines.append(
            f"| {ds} | {fmt(h_fin['G1'][ds])} | {fmt(h_fin['G2'][ds])} | {fmt(h_fin['Gsim'][ds])} "
            f"| {fmt(h_fin['Gdiff'][ds])} | {fmt(h_cur[ds])} | {fmt(h_exp[ds])} |"
        )
    lines.append("")
    lines.append("### Delta vs [Z|B_G1] 与 Δ_B_final（h+12d 严格同维）")
    lines.append("")
    lines.append("| dataset | Δ_hop_final | Δ_sim_final | Δ_diff_final | Δ_B_final = explicit−current |")
    lines.append("|---|---:|---:|---:|---:|")
    for ds in datasets:
        lines.append(f"| {ds} | {fmt(d_hop_fin[ds])} | {fmt(d_sim_fin[ds])} | {fmt(d_diff_fin[ds])} | {fmt(db_final[ds])} |")
    lines.append(f"| **M/T/G macro** | {fmt_macro(d_hop_fin)} | {fmt_macro(d_sim_fin)} | {fmt_macro(d_diff_fin)} | **{fmt_macro(db_final)}** |")
    lines.append("")

    lines.append("## Level-IV — Shuffle negative control（perm seed 20260904，real − shuffled）")
    lines.append("")
    lines.append("| dataset | inter current | inter explicit | final current | final explicit |")
    lines.append("|---|---:|---:|---:|---:|")
    for ds in datasets:
        lines.append(
            f"| {ds} | {fmt(shuf_stats[('intermediate', 'current')][ds])} | {fmt(shuf_stats[('intermediate', 'explicit')][ds])} "
            f"| {fmt(shuf_stats[('final', 'current')][ds])} | {fmt(shuf_stats[('final', 'explicit')][ds])} |"
        )
    lines.append("")

    lines.append("## Secondary — D_ctx / redundancy（不作 GO/NO-GO 依据）")
    lines.append("")
    lines.append("| dataset | basis | D_ctx^C | D_ctx^Pt | D_ctx^Pv | mean off-diag cos (C/Pt/Pv) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for basis in ("current", "explicit"):
        for ds in datasets:
            dctx = {}
            offd = {}
            for f in FACTORS:
                vals = [r["d_ctx"] for r in div_rows if r["dataset"] == ds and r["basis"] == basis and r["factor"] == f]
                dctx[f] = statistics.mean([v for v in vals if v == v]) if any(v == v for v in vals) else float("nan")
                coses = [r["mean_cos"] for r in red_rows if r["dataset"] == ds and r["basis"] == basis and r["factor"] == f
                         and r["k1"] != r["k2"] and r["mean_cos"] == r["mean_cos"]]
                offd[f] = statistics.mean(coses) if coses else float("nan")
            lines.append(
                f"| {ds} | {basis} | {dctx['C']:.4f} | {dctx['Pt']:.4f} | {dctx['Pv']:.4f} "
                f"| {offd['C']:.4f} / {offd['Pt']:.4f} / {offd['Pv']:.4f} |"
            )
    lines.append("")
    lines.append("> D_ctx 增大 ≠ task utility 增大；仅作机制背景，绝不用于 GO/NO-GO。")
    lines.append("")

    lines.append("## Verdict 细则（预注册）")
    lines.append("")
    lines.append(f"- Verdict-B1（Intermediate）：Δ_B_intermediate M/T/G macro = {pp(mean_MTG_inter):+.2f}pp；"
                 f"positive datasets = {inter_ds_pos}/3（其中 ≥{POS_SEEDS_MIN}/3 seeds positive 的 = {inter_ds_qual}）→ **{verdict_inter}**")
    lines.append(f"- Verdict-B2（Final）：Δ_B_final M/T/G macro = {pp(mean_MTG_final):+.2f}pp；"
                 f"positive datasets = {final_ds_pos}/3（qual = {final_ds_qual}）；"
                 f"H_explicit_final macro = {pp(macro_mean(h_exp)):+.2f}pp（stable：{h_exp_stable}）；"
                 f"final explicit real−shuffled macro = {pp(macro_mean(shuf_final_exp)):+.2f}pp（stable：{shuf_stable}）→ **{verdict_final}**")
    lines.append(f"- 最终分类：**{case}**")
    lines.append("")

    (OUT_ROOT / "R20_B_STRUCTURAL_BASIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[B] report done -> {OUT_ROOT / 'R20_B_STRUCTURAL_BASIS_REPORT.md'} "
          f"(inter={verdict_inter}, final={verdict_final}, case={case.split('（')[0]})", flush=True)


if __name__ == "__main__":
    main()
