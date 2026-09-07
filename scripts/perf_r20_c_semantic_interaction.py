"""R2-0C: Semantic Interaction Headroom Probe (user Prompt C).

Frozen OFR checkpoints only; no training; no Test (r20 wrapper masks test
labels / cuts test_idx). Fixed probe = R0 ridge_probe (StandardScaler +
RidgeClassifier(alpha=1.0), fit TRAIN / eval VAL).

Levels:
    Level-I   common averaging headroom: C_avg / [c_t|c_v] / + interaction
    Level-Ib  full semantic representation (S_base vs branch/inter vs z_local)
    Level-II  factor interaction [C|Pt|Pv|I_factor] + same-node shuffle control
    Level-III projected-modality interaction [h_t|h_v|I_modal] + mismatch control
    Level-IV  z_final residual: [Z|I_factor] / [Z|I_modal] / [Z|B_common|I_common]
    Level-V   graph-conditioned cross-factor interaction, 9 cells a->b:
              X_linear=[F^b|N^a] vs X_inter=[F|N|F*N|abs(F-N)] + N-mismatch
    Level-Vb  L-conditioned: [L|N^a] vs [L|N^a|F^b*N^a|abs(F^b-N^a)]
    Level-VI  z_final conditional-interaction residual [Z|I_ab] + mismatch

One fixed permutation (torch seed 20260904) for every shuffle/mismatch
control; no second permutation. Gsim/Gdiff are NOT used in this round.

Outputs: outputs/perf_r20/semantic_interaction/{csv x8, R20_C_SEMANTIC_INTERACTION_REPORT.md}
Usage:
    python scripts/perf_r20_c_semantic_interaction.py --gpus 0,1
    python scripts/perf_r20_c_semantic_interaction.py --gpus 0 --stage smoke
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

from src.analysis.perf_r20_utils import (  # noqa: E402
    DATASETS,
    SEEDS,
    assert_feature_dim,
    cond_interaction_block,
    context_concat,
    extract_forward,
    factor_interaction_block,
    factor_tensor,
    load_setup,
    modal_interaction_block,
    ridge_probe,
    write_csv,
)
from src.models.biaxis_p1_components import neighbor_mean  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / "perf_r20" / "semantic_interaction"
FACTORS = ["C", "Pt", "Pv"]
SHUFFLE_SEED = 20260904

# ---------------------------------------------------------------------------
# Pre-registered thresholds (user §十一)
# ---------------------------------------------------------------------------
C_STRONG_PP = 0.50
C_GO_PP = 0.30
C_WEAK_PP = 0.15
POS_SEEDS_MIN = 2
L3_STRONG_PP = 0.30   # C3: Delta_L_cond_inter stable threshold
L3_GO_PP = 0.20
H3_FINAL_PP = 0.20    # C3 soft criterion: H_cond_inter_final


def _stat(values: list[float]) -> tuple[float, float, int]:
    mean = statistics.mean(values) if values else float("nan")
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    pos = sum(1 for v in values if v > 0)
    return mean, std, pos


def _probe(tensor: torch.Tensor, setup, tag: str) -> tuple[float, float]:
    probe = ridge_probe(tensor, setup)
    print(f"[C] {setup.dataset:12s} s{setup.seed} {tag:30s} acc={probe['val_acc']:.4f}", flush=True)
    return probe["val_acc"], probe["val_macro_f1"]


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-0C semantic interaction probe")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--stage", default="all", choices=["smoke", "full", "all"])
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    datasets = [d for d in (args.datasets or ",".join(DATASETS)).split(",") if d in DATASETS]
    seeds = [int(s) for s in (args.seeds or ",".join(map(str, SEEDS))).split(",") if s]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    common_rows: list[dict] = []        # common_representation_probe
    factor_rows: list[dict] = []        # semantic_factor_interaction_probe
    modal_rows: list[dict] = []         # semantic_modal_interaction_probe
    ctrl_rows: list[dict] = []          # semantic_interaction_shuffle_control
    final_rows: list[dict] = []         # semantic_final_residual
    cells_rows: list[dict] = []         # graph_conditioned_interaction_cells
    local_rows: list[dict] = []         # graph_conditioned_local_complete
    z_cells_rows: list[dict] = []       # graph_conditioned_final_residual

    if args.stage == "all":
        lifecycles = [(ds, s) for ds in datasets for s in seeds]
    elif args.stage == "smoke":
        lifecycles = [("Movies", 42)]
    else:
        lifecycles = [(ds, s) for ds in datasets for s in seeds]
    datasets = sorted({ds for ds, _ in lifecycles})

    for idx, (ds, s) in enumerate(lifecycles):
        device = torch.device(f"cuda:{gpus[idx % len(gpus)]}")
        setup = load_setup(ds, s, device)
        fex = extract_forward(setup)
        n = int(fex["f_block"].size(0))
        d = int(fex["f_block"].size(2))
        hidden = int(fex["z_final"].size(1))
        assert d == 128, f"frozen d_f=128 expected, got {d}"
        edge_index = fex["edge_index"]
        chunk = setup.model.edge_chunk_size

        factors = fex["factors"]
        F = {f: factor_tensor(fex, f) for f in FACTORS}
        c_t, c_v = factors["c_t"], factors["c_v"]
        h_t, h_v = factors["h_t"], factors["h_v"]
        L = context_concat([F[f] for f in FACTORS])
        Z = fex["z_final"]
        z_local = fex["z_local"]

        I_factor = factor_interaction_block(F)
        I_modal = modal_interaction_block(h_t, h_v)
        I_common = context_concat([c_t * c_v, (c_t - c_v).abs()])
        B_common = context_concat([c_t, c_v])

        N = {f: neighbor_mean(edge_index, F[f], n, edge_chunk_size=chunk) for f in FACTORS}
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(SHUFFLE_SEED)).to(device)
        I_factor_shuf = I_factor[perm]
        h_v_perm = h_v[perm]
        I_modal_mismatch = modal_interaction_block(h_t, h_v_perm)
        N_perm = {f: N[f][perm] for f in FACTORS}

        # --- Level-I common averaging -----------------------------------------
        c_avg_a, c_avg_b = _probe(F["C"], setup, "C_avg=C")
        common_rows.append({"dataset": ds, "seed": s, "variant": "C_avg",
                            "val_acc": c_avg_a, "val_macro_f1": c_avg_b})
        br_a, br_b = _probe(B_common, setup, "C_branches=[c_t|c_v]")
        assert_feature_dim(B_common, n, 2 * d, "C_branches")
        common_rows.append({"dataset": ds, "seed": s, "variant": "C_branches",
                            "val_acc": br_a, "val_macro_f1": br_b})
        inter_a, inter_b = _probe(context_concat([B_common, I_common]), setup, "C_inter")
        assert_feature_dim(context_concat([B_common, I_common]), n, 4 * d, "C_inter")
        common_rows.append({"dataset": ds, "seed": s, "variant": "C_inter",
                            "val_acc": inter_a, "val_macro_f1": inter_b})

        # --- Level-Ib full semantic representation -----------------------------
        s_base_a, s_base_b = _probe(L, setup, "S_base=[C|Pt|Pv]")
        common_rows.append({"dataset": ds, "seed": s, "variant": "S_base",
                            "val_acc": s_base_a, "val_macro_f1": s_base_b})
        scb = context_concat([c_t, c_v, F["Pt"], F["Pv"]])
        assert_feature_dim(scb, n, 4 * d, "S_common_branches")
        scb_a, scb_b = _probe(scb, setup, "S_common_branches")
        common_rows.append({"dataset": ds, "seed": s, "variant": "S_common_branches",
                            "val_acc": scb_a, "val_macro_f1": scb_b})
        sci = context_concat([scb, I_common])
        assert_feature_dim(sci, n, 6 * d, "S_common_inter")
        sci_a, sci_b = _probe(sci, setup, "S_common_inter")
        common_rows.append({"dataset": ds, "seed": s, "variant": "S_common_inter",
                            "val_acc": sci_a, "val_macro_f1": sci_b})
        zl_a, zl_b = _probe(z_local, setup, "z_local")
        common_rows.append({"dataset": ds, "seed": s, "variant": "z_local",
                            "val_acc": zl_a, "val_macro_f1": zl_b})

        # --- Level-II factor interaction ----------------------------------------
        factor_rows.append({"dataset": ds, "seed": s, "variant": "base",
                            "val_acc": s_base_a, "val_macro_f1": s_base_b})
        fi = context_concat([L, I_factor])
        assert_feature_dim(fi, n, 9 * d, "X_factor_inter")
        fi_a, fi_b = _probe(fi, setup, "X_factor_inter")
        factor_rows.append({"dataset": ds, "seed": s, "variant": "inter",
                            "val_acc": fi_a, "val_macro_f1": fi_b})
        fi_shuf = context_concat([L, I_factor_shuf])
        assert_feature_dim(fi_shuf, n, 9 * d, "X_factor_shuf")
        fs_a, fs_b = _probe(fi_shuf, setup, "X_factor_shuf")
        factor_rows.append({"dataset": ds, "seed": s, "variant": "shuf",
                            "val_acc": fs_a, "val_macro_f1": fs_b})
        factor_rows.append({"dataset": ds, "seed": s, "variant": "z_local",
                            "val_acc": zl_a, "val_macro_f1": zl_b})
        ctrl_rows.append({"dataset": ds, "seed": s, "block": "factor",
                          "real_acc": fi_a, "control_acc": fs_a,
                          "diff_acc": fi_a - fs_a, "real_f1": fi_b,
                          "control_f1": fs_b, "diff_f1": fi_b - fs_b})

        # --- Level-III projected modality ---------------------------------------
        mb = context_concat([h_t, h_v])
        assert_feature_dim(mb, n, 2 * hidden, "X_modal_base")
        mb_a, mb_b = _probe(mb, setup, "X_modal_base")
        modal_rows.append({"dataset": ds, "seed": s, "variant": "base",
                           "val_acc": mb_a, "val_macro_f1": mb_b})
        mi = context_concat([mb, I_modal])
        assert_feature_dim(mi, n, 4 * hidden, "X_modal_inter")
        mi_a, mi_b = _probe(mi, setup, "X_modal_inter")
        modal_rows.append({"dataset": ds, "seed": s, "variant": "inter",
                           "val_acc": mi_a, "val_macro_f1": mi_b})
        mm = context_concat([mb, I_modal_mismatch])
        assert_feature_dim(mm, n, 4 * hidden, "X_modal_mismatch")
        mm_a, mm_b = _probe(mm, setup, "X_modal_mismatch")
        modal_rows.append({"dataset": ds, "seed": s, "variant": "mismatch",
                           "val_acc": mm_a, "val_macro_f1": mm_b})
        ctrl_rows.append({"dataset": ds, "seed": s, "block": "modal",
                          "real_acc": mi_a, "control_acc": mm_a,
                          "diff_acc": mi_a - mm_a, "real_f1": mi_b,
                          "control_f1": mm_b, "diff_f1": mi_b - mm_b})

        # --- Level-IV final residual --------------------------------------------
        z_acc, z_f1 = _probe(Z, setup, "z_final")
        z_if = context_concat([Z, I_factor])
        assert_feature_dim(z_if, n, hidden + 6 * d, "[Z|I_factor]")
        zif_a, zif_b = _probe(z_if, setup, "[Z|I_factor]")
        z_ifs = context_concat([Z, I_factor_shuf])
        zifs_a, zifs_b = _probe(z_ifs, setup, "[Z|I_factor_shuf]")
        z_im = context_concat([Z, I_modal])
        assert_feature_dim(z_im, n, hidden + 2 * hidden, "[Z|I_modal]")
        zim_a, zim_b = _probe(z_im, setup, "[Z|I_modal]")
        z_imm = context_concat([Z, I_modal_mismatch])
        zimm_a, zimm_b = _probe(z_imm, setup, "[Z|I_modal_mismatch]")
        z_bc = context_concat([Z, B_common])
        assert_feature_dim(z_bc, n, hidden + 2 * d, "[Z|B_common]")
        zbc_a, zbc_b = _probe(z_bc, setup, "[Z|B_common]")
        z_bci = context_concat([z_bc, I_common])
        assert_feature_dim(z_bci, n, hidden + 4 * d, "[Z|B_common|I_common]")
        zbci_a, zbci_b = _probe(z_bci, setup, "[Z|B_common|I_common]")
        final_rows.append({
            "dataset": ds, "seed": s, "z_acc": z_acc, "z_f1": z_f1,
            "zif_acc": zif_a, "zif_f1": zif_b, "zifs_acc": zifs_a, "zifs_f1": zifs_b,
            "zim_acc": zim_a, "zim_f1": zim_b, "zimm_acc": zimm_a, "zimm_f1": zimm_b,
            "zbc_acc": zbc_a, "zbc_f1": zbc_b, "zbci_acc": zbci_a, "zbci_f1": zbci_b,
            "H_factor_acc": zif_a - z_acc, "H_factor_f1": zif_b - z_f1,
            "Specific_factor_acc": zif_a - zifs_a, "Specific_factor_f1": zif_b - zifs_b,
            "H_modal_acc": zim_a - z_acc, "H_modal_f1": zim_b - z_f1,
            "Specific_modal_acc": zim_a - zimm_a, "Specific_modal_f1": zim_b - zimm_b,
            "H_common_branch_acc": zbc_a - z_acc, "H_common_branch_f1": zbc_b - z_f1,
            "H_common_inter_acc": zbci_a - z_acc, "H_common_inter_f1": zbci_b - z_f1,
        })

        # --- Level-V graph-conditioned cells ------------------------------------
        for a in FACTORS:
            for b in FACTORS:
                Fb = F[b]
                Na = N[a]
                lin = context_concat([Fb, Na])
                assert_feature_dim(lin, n, 2 * d, f"linear {a}->{b}")
                lin_a, lin_b = _probe(lin, setup, f"lin {a}->{b}")
                inter = context_concat([lin, cond_interaction_block(Fb, Na)])
                assert_feature_dim(inter, n, 4 * d, f"inter {a}->{b}")
                i_a, i_b = _probe(inter, setup, f"inter {a}->{b}")
                mism = context_concat([lin, cond_interaction_block(Fb, N_perm[a])])
                assert_feature_dim(mism, n, 4 * d, f"mismatch {a}->{b}")
                m_a, m_b = _probe(mism, setup, f"mismatch {a}->{b}")
                cells_rows.append({
                    "dataset": ds, "seed": s, "source": a, "target": b,
                    "linear_acc": lin_a, "linear_f1": lin_b,
                    "inter_acc": i_a, "inter_f1": i_b,
                    "delta_acc": i_a - lin_a, "delta_f1": i_b - lin_b,
                    "mismatch_acc": m_a, "mismatch_f1": m_b,
                    "specific_acc": i_a - m_a, "specific_f1": i_b - m_b,
                })

        # --- Level-Vb L-conditioned ---------------------------------------------
        for a in FACTORS:
            lin = context_concat([L, N[a]])
            assert_feature_dim(lin, n, 4 * d, f"L lin {a}")
            lin_a, lin_b = _probe(lin, setup, f"L lin {a}")
            for b in FACTORS:
                i_ab = cond_interaction_block(F[b], N[a])
                inter = context_concat([L, N[a], i_ab])
                assert_feature_dim(inter, n, 6 * d, f"L inter {a}->{b}")
                i_a, i_b = _probe(inter, setup, f"L inter {a}->{b}")
                mism = context_concat([L, N[a], cond_interaction_block(F[b], N_perm[a])])
                assert_feature_dim(mism, n, 6 * d, f"L mismatch {a}->{b}")
                m_a, m_b = _probe(mism, setup, f"L mismatch {a}->{b}")
                local_rows.append({
                    "dataset": ds, "seed": s, "source": a, "target": b,
                    "linear_acc": lin_a, "linear_f1": lin_b,
                    "inter_acc": i_a, "inter_f1": i_b,
                    "delta_acc": i_a - lin_a, "delta_f1": i_b - lin_b,
                    "mismatch_acc": m_a, "mismatch_f1": m_b,
                    "specific_acc": i_a - m_a, "specific_f1": i_b - m_b,
                })

        # --- Level-VI z_final conditional interaction residual ------------------
        for a in FACTORS:
            for b in FACTORS:
                i_ab = cond_interaction_block(F[b], N[a])
                inter = context_concat([Z, i_ab])
                assert_feature_dim(inter, n, hidden + 2 * d, f"Z inter {a}->{b}")
                i_a, i_b = _probe(inter, setup, f"Z inter {a}->{b}")
                mism = context_concat([Z, cond_interaction_block(F[b], N_perm[a])])
                m_a, m_b = _probe(mism, setup, f"Z mismatch {a}->{b}")
                z_cells_rows.append({
                    "dataset": ds, "seed": s, "source": a, "target": b,
                    "z_acc": z_acc, "z_f1": z_f1,
                    "inter_acc": i_a, "inter_f1": i_b,
                    "H_acc": i_a - z_acc, "H_f1": i_b - z_f1,
                    "mismatch_acc": m_a, "mismatch_f1": m_b,
                    "specific_acc": i_a - m_a, "specific_f1": i_b - m_b,
                })

        del fex
        torch.cuda.empty_cache()
        print(f"[C] {ds:12s} s{s} done ({idx + 1}/{len(lifecycles)})", flush=True)

    # ------------------------------------------------------------------ CSVs
    write_csv(OUT_ROOT / "common_representation_probe.csv", common_rows)
    write_csv(OUT_ROOT / "semantic_factor_interaction_probe.csv", factor_rows)
    write_csv(OUT_ROOT / "semantic_modal_interaction_probe.csv", modal_rows)
    write_csv(OUT_ROOT / "semantic_interaction_shuffle_control.csv", ctrl_rows)
    write_csv(OUT_ROOT / "semantic_final_residual.csv", final_rows)
    write_csv(OUT_ROOT / "graph_conditioned_interaction_cells.csv", cells_rows)
    write_csv(OUT_ROOT / "graph_conditioned_local_complete.csv", local_rows)
    write_csv(OUT_ROOT / "graph_conditioned_final_residual.csv", z_cells_rows)
    print(f"[C] CSVs written -> {OUT_ROOT}", flush=True)

    # ------------------------------------------------------------------ report
    pp = lambda v: 100.0 * v

    def pick(rows: list[dict], key: str) -> dict[tuple, float]:
        return {(r["dataset"], r["seed"]): r[key] for r in rows}

    def ds_stats(vals: dict[tuple, float]) -> dict[str, tuple]:
        out = {}
        for ds in datasets:
            v = [vals[(ds, s)] for s in seeds if (ds, s) in vals]
            if v:
                out[ds] = _stat(v)
        return out

    def macro_mean(ds_dict: dict[str, tuple]) -> float:
        means = [v[0] for v in ds_dict.values() if v[0] == v[0]]
        return statistics.mean(means) if means else float("nan")

    def fmt(stat):
        m, s, p = stat
        return f"{pp(m):+.2f}±{pp(s):.2f} ({p})"

    def fmt_macro(ds_dict: dict[str, tuple]) -> str:
        means = [v[0] for v in ds_dict.values() if v[0] == v[0]]
        m = statistics.mean(means) if means else float("nan")
        s = statistics.pstdev(means) if len(means) > 1 else 0.0
        return f"{pp(m):+.2f}±{pp(s):.2f}"

    def delta_of(a_rows, a_key, b_rows, b_key) -> dict[tuple, float]:
        va, vb = pick(a_rows, a_key), pick(b_rows, b_key)
        return {(ds, s): va[(ds, s)] - vb[(ds, s)] for (ds, s) in va}

    # Level-I / Ib
    def rows_of(rows: list[dict], key: str) -> dict[tuple, float]:
        return {(r["dataset"], r["seed"]): r["val_acc"] for r in rows if r["variant"] == key}

    variant = lambda key: rows_of(common_rows, key)

    d_cb = ds_stats({(ds, s): variant("C_branches")[(ds, s)] - variant("C_avg")[(ds, s)] for ds, s in lifecycles})
    d_ci = ds_stats({(ds, s): variant("C_inter")[(ds, s)] - variant("C_branches")[(ds, s)] for ds, s in lifecycles})
    d_fb = ds_stats({(ds, s): variant("S_common_branches")[(ds, s)] - variant("S_base")[(ds, s)] for ds, s in lifecycles})
    d_fi = ds_stats({(ds, s): variant("S_common_inter")[(ds, s)] - variant("S_common_branches")[(ds, s)] for ds, s in lifecycles})
    d_sci_zl = ds_stats({(ds, s): variant("S_common_inter")[(ds, s)] - variant("z_local")[(ds, s)] for ds, s in lifecycles})

    # Level-II
    fac_inter = rows_of(factor_rows, "inter")
    fac_base = rows_of(factor_rows, "base")
    fac_shuf = rows_of(factor_rows, "shuf")
    fac_zl = rows_of(factor_rows, "z_local")
    d_fi_vs_base = ds_stats({(ds, s): fac_inter[(ds, s)] - fac_base[(ds, s)] for ds, s in lifecycles})
    d_fi_vs_zl = ds_stats({(ds, s): fac_inter[(ds, s)] - fac_zl[(ds, s)] for ds, s in lifecycles})
    node_spec = ds_stats({(ds, s): fac_inter[(ds, s)] - fac_shuf[(ds, s)] for ds, s in lifecycles})

    # Level-III
    def modal(key):
        return {(r["dataset"], r["seed"]): r["val_acc"] for r in modal_rows if r["variant"] == key}

    d_modal_inter = ds_stats({(ds, s): modal("inter")[(ds, s)] - modal("base")[(ds, s)] for ds, s in lifecycles})
    d_modal_real_mm = ds_stats({(ds, s): modal("inter")[(ds, s)] - modal("mismatch")[(ds, s)] for ds, s in lifecycles})

    # Level-IV
    h_factor = ds_stats(pick(final_rows, "H_factor_acc"))
    spec_factor = ds_stats(pick(final_rows, "Specific_factor_acc"))
    h_modal = ds_stats(pick(final_rows, "H_modal_acc"))
    spec_modal = ds_stats(pick(final_rows, "Specific_modal_acc"))
    h_common_b = ds_stats(pick(final_rows, "H_common_branch_acc"))
    h_common_i = ds_stats(pick(final_rows, "H_common_inter_acc"))

    # Level-V / Vb / VI per cell
    def cell_stats(rows: list[dict], key: str) -> dict[tuple, dict[str, tuple]]:
        out = {}
        for a in FACTORS:
            for b in FACTORS:
                vals = {(r["dataset"], r["seed"]): r[key] for r in rows if r["source"] == a and r["target"] == b}
                out[(a, b)] = ds_stats(vals)
        return out

    d_l5 = cell_stats(cells_rows, "delta_acc")
    spec_l5 = cell_stats(cells_rows, "specific_acc")
    d_l5b = cell_stats(local_rows, "delta_acc")
    spec_l5b = cell_stats(local_rows, "specific_acc")
    h_l6 = cell_stats(z_cells_rows, "H_acc")
    spec_l6 = cell_stats(z_cells_rows, "specific_acc")

    # ------------------------------------------------------------------ verdicts
    # --- C1 node-local semantic refinement ---
    m_c1 = macro_mean(d_fi_vs_zl)
    c1_ds_pos = sum(1 for ds in datasets if d_fi_vs_zl[ds][0] > 0)
    ctrl_m = macro_mean(node_spec)
    ctrl_ds_pos = sum(1 for ds in datasets if node_spec[ds][0] > 0)
    if m_c1 >= C_STRONG_PP / 100 and c1_ds_pos >= 2 and ctrl_m > 0 and ctrl_ds_pos >= 2:
        verdict_c1 = "STRONG"
    elif m_c1 >= C_GO_PP / 100 and c1_ds_pos >= 2 and ctrl_m > 0:
        verdict_c1 = "GO"
    elif m_c1 >= C_WEAK_PP / 100:
        verdict_c1 = "WEAK"
    else:
        verdict_c1 = "NO-GO"

    # --- C2 final semantic residual ---
    fams = {
        "factor": (h_factor, spec_factor),
        "modal": (h_modal, spec_modal),
        "common": (h_common_i, None),
    }
    best_tier = 0  # 0 none, 1 WEAK, 2 GO, 3 STRONG
    best_fam = None
    fam_details = {}
    for fam, (h, ctrl) in fams.items():
        m = macro_mean(h)
        pos = sum(1 for ds in datasets if h[ds][0] > 0)
        tier = 0
        if m >= C_STRONG_PP / 100 and pos >= 2:
            if ctrl is not None:
                cm = macro_mean(ctrl)
                cpos = sum(1 for ds in datasets if ctrl[ds][0] > 0)
                tier = 3 if (cm > 0 and cpos >= 2) else 2
            else:
                tier = 2  # common family has no pre-registered control
        elif m >= C_GO_PP / 100 and pos >= 2:
            tier = 2
        elif m >= C_WEAK_PP / 100:
            tier = 1
        fam_details[fam] = (m, pos, tier)
        if tier > best_tier:
            best_tier, best_fam = tier, fam
    verdict_c2 = {0: "NO-GO", 1: "WEAK", 2: "GO", 3: "STRONG"}[best_tier]
    if best_tier == 3 and best_fam == "common":
        verdict_c2 = "GO"  # common has no control; STRONG needs a controlled family

    # --- C3 graph-conditioned cross-factor interaction ---
    stable_cells = []   # (a, b) with >=2 datasets qualifying at 0.30pp
    qual_pairs = []     # (a, b, ds) at 0.30pp
    go_pairs = []       # (a, b, ds) at 0.20pp
    for (a, b), st in d_l5b.items():
        for ds in datasets:
            m, s, p = st[ds]
            if m >= L3_STRONG_PP / 100 and p >= POS_SEEDS_MIN:
                qual_pairs.append((a, b, ds))
            if m >= L3_GO_PP / 100 and p >= POS_SEEDS_MIN:
                go_pairs.append((a, b, ds))
    for a in FACTORS:
        for b in FACTORS:
            qds = {ds for (aa, bb, ds) in qual_pairs if aa == a and bb == b}
            if len(qds) >= 2:
                stable_cells.append((a, b))
    spec_ok_cells = []
    for (a, b) in stable_cells:
        means = [spec_l5b[(a, b)][ds][0] for ds in datasets if d_l5b[(a, b)][ds][0] >= L3_STRONG_PP / 100
                 and d_l5b[(a, b)][ds][2] >= POS_SEEDS_MIN]
        if means and statistics.mean(means) > 0 and sum(1 for v in means if v > 0) >= len(means) / 3 * 2:
            spec_ok_cells.append((a, b))
    # soft H_cond_inter_final criterion
    h3_patterns = []
    for (a, b), st in h_l6.items():
        for ds in datasets:
            m, s, p = st[ds]
            if m >= H3_FINAL_PP / 100 and p >= POS_SEEDS_MIN:
                h3_patterns.append((a, b, ds))
    go_ds = {ds for (_, _, ds) in go_pairs}
    go_ok = len(go_pairs) >= 2 and len(go_ds) >= 2 and all(
        spec_l5b[(a, b)][ds][0] > 0 for (a, b, ds) in go_pairs
    )
    if len(stable_cells) >= 2 and len(spec_ok_cells) >= 2:
        verdict_c3 = "STRONG"
    elif go_ok:
        verdict_c3 = "GO"
    elif len(go_pairs) >= 1 or any(st[ds][0] >= C_WEAK_PP / 100 for (a, b), st in d_l5b.items() for ds in datasets):
        verdict_c3 = "WEAK"
    else:
        verdict_c3 = "NO-GO"

    # --- case ---
    c1_pos = verdict_c1 in ("STRONG", "GO")
    c3_pos = verdict_c3 in ("STRONG", "GO")
    if c1_pos and not c3_pos:
        case = "C-1（node-local semantic refinement 是主瓶颈：Semantic Ownership → Semantic Refinement → simple graph propagation）"
    elif (not c1_pos) and c3_pos:
        case = "C-2（factors 够用、缺失的是 target semantic state × neighbor source 条件交互：Semantic Ownership → Target-Conditioned Cross-Factor Functional Transfer → simple topology）"
    elif c1_pos and c3_pos:
        case = "C-3（lightweight Semantic Refinement + Functional Cross-Factor Transfer，第一版保持简洁）"
    else:
        case = "C-4（frozen probe 无证据支持 local interaction / conditional transfer；需重新审查 backbone-level learning dynamics）"

    # ------------------------------------------------------------------ report
    lines = ["# R20-C SEMANTIC INTERACTION REPORT", ""]
    lines.append("## 结论（Top）")
    lines.append("")
    lines.append(f"- **Verdict-C1（Node-local Semantic Refinement）：{verdict_c1}**（inter − z_local M/T/G macro = {pp(m_c1):+.2f}pp；node-correspondence control = {pp(ctrl_m):+.2f}pp）")
    lines.append(f"- **Verdict-C2（Final Semantic Residual）：{verdict_c2}**（best family = {best_fam}，tier {best_tier}）")
    lines.append(f"- **Verdict-C3（Graph-Conditioned Cross-Factor Interaction）：{verdict_c3}**（stable cells = {len(stable_cells)}，qualifying (cell,dataset) = {len(qual_pairs)}，H_final patterns = {len(h3_patterns)}）")
    lines.append(f"- **Case：{case}**")
    lines.append("")
    lines.append("> 协议：frozen OFR checkpoints（M/T/G × 42/43/44）；固定 Ridge probe（StandardScaler +")
    lines.append("> RidgeClassifier(alpha=1.0)，TRAIN fit / VAL eval）；无 Test（test labels masked + test_idx 切断）。")
    lines.append("> 数值为 Val Acc 的 pp（×100），3-seed mean±population std(ddof=0)，括号 = positive seeds；")
    lines.append("> Macro-F1 全程记录于 CSV，表内 F1 为 secondary evidence。")
    lines.append("> 固定 permutation seed = 20260904（单 permutation）；本轮不使用 Gsim/Gdiff。")
    lines.append("> R2-0 审计：Audit PASS（129/129，max diff 5.96e-8）。")
    lines.append("")

    lines.append("## Level-I — Common averaging headroom")
    lines.append("")
    lines.append("| dataset | Δ_common_branches (C_branches−C_avg) | Δ_common_interaction (C_inter−C_branches) | Δ_full_common_branch (S_cb−S_base) | Δ_full_common_inter (S_ci−S_cb) | S_common_inter − z_local |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for ds in datasets:
        lines.append(f"| {ds} | {fmt(d_cb[ds])} | {fmt(d_ci[ds])} | {fmt(d_fb[ds])} | {fmt(d_fi[ds])} | {fmt(d_sci_zl[ds])} |")
    lines.append(f"| **M/T/G macro** | {fmt_macro(d_cb)} | {fmt_macro(d_ci)} | {fmt_macro(d_fb)} | {fmt_macro(d_fi)} | {fmt_macro(d_sci_zl)} |")
    lines.append("")

    lines.append("## Level-II — Factor interaction（Primary = inter − z_local）")
    lines.append("")
    lines.append("| dataset | Δ_factor_inter_vs_base | **Δ_factor_inter_vs_zlocal** | node-correspondence (inter−shuf) |")
    lines.append("|---|---:|---:|---:|")
    for ds in datasets:
        lines.append(f"| {ds} | {fmt(d_fi_vs_base[ds])} | {fmt(d_fi_vs_zl[ds])} | {fmt(node_spec[ds])} |")
    lines.append(f"| **M/T/G macro** | {fmt_macro(d_fi_vs_base)} | **{fmt_macro(d_fi_vs_zl)}** | {fmt_macro(node_spec)} |")
    lines.append("")

    lines.append("## Level-III — Projected-modality interaction")
    lines.append("")
    lines.append("| dataset | Δ_modal_inter (inter−base) | real − mismatched-pair |")
    lines.append("|---|---:|---:|")
    for ds in datasets:
        lines.append(f"| {ds} | {fmt(d_modal_inter[ds])} | {fmt(d_modal_real_mm[ds])} |")
    lines.append(f"| **M/T/G macro** | {fmt_macro(d_modal_inter)} | {fmt_macro(d_modal_real_mm)} |")
    lines.append("")

    lines.append("## Level-IV — z_final residual semantic headroom")
    lines.append("")
    lines.append("| dataset | H_factor | Specific_factor | H_modal | Specific_modal | H_common_branch | H_common_inter |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ds in datasets:
        lines.append(
            f"| {ds} | {fmt(h_factor[ds])} | {fmt(spec_factor[ds])} | {fmt(h_modal[ds])} | {fmt(spec_modal[ds])} "
            f"| {fmt(h_common_b[ds])} | {fmt(h_common_i[ds])} |"
        )
    lines.append(f"| **M/T/G macro** | {fmt_macro(h_factor)} | {fmt_macro(spec_factor)} | {fmt_macro(h_modal)} | {fmt_macro(spec_modal)} | {fmt_macro(h_common_b)} | {fmt_macro(h_common_i)} |")
    lines.append("")

    lines.append("## Level-V — Graph-conditioned cells：Δ_cond_inter = Probe([F|N|F*N|abs(F-N)]) − Probe([F|N])")
    lines.append("")
    lines.append("### matrix（source rows × target columns；Specific = inter − N-mismatch）")
    lines.append("")
    for a in FACTORS:
        lines.append(f"### source = {a}")
        lines.append("")
        lines.append("| dataset | C (Δ / Spec) | Pt (Δ / Spec) | Pv (Δ / Spec) |")
        lines.append("|---|---:|---:|---:|")
        for ds in datasets:
            cells = " | ".join(f"{fmt(d_l5[(a, b)][ds])} / {fmt(spec_l5[(a, b)][ds])}" for b in FACTORS)
            lines.append(f"| {ds} | {cells} |")
        lines.append("")
    lines.append("> 预注册重点 cells：C→Pt、Pv→Pt、C→Pv（Toys/Grocery）、Pv→C（Movies）；不得临时挑选新的 best cell。")
    lines.append("")

    lines.append("## Level-Vb — L-conditioned interaction：Δ_L_cond_inter（[L|N^a|I_ab] − [L|N^a]）")
    lines.append("")
    lines.append("### matrix（source rows × target columns；Specific = inter − mismatch）")
    lines.append("")
    for a in FACTORS:
        lines.append(f"### source = {a}")
        lines.append("")
        lines.append("| dataset | C (Δ / Spec) | Pt (Δ / Spec) | Pv (Δ / Spec) |")
        lines.append("|---|---:|---:|---:|")
        for ds in datasets:
            cells = " | ".join(f"{fmt(d_l5b[(a, b)][ds])} / {fmt(spec_l5b[(a, b)][ds])}" for b in FACTORS)
            lines.append(f"| {ds} | {cells} |")
        lines.append("")

    lines.append("## Level-VI — z_final conditional-interaction residual：H = Probe([Z|I_ab]) − Probe(Z)")
    lines.append("")
    lines.append("### matrix（source rows × target columns；Specific = inter − mismatch）")
    lines.append("")
    for a in FACTORS:
        lines.append(f"### source = {a}")
        lines.append("")
        lines.append("| dataset | C (H / Spec) | Pt (H / Spec) | Pv (H / Spec) |")
        lines.append("|---|---:|---:|---:|")
        for ds in datasets:
            cells = " | ".join(f"{fmt(h_l6[(a, b)][ds])} / {fmt(spec_l6[(a, b)][ds])}" for b in FACTORS)
            lines.append(f"| {ds} | {cells} |")
        lines.append("")

    lines.append("## Verdict 细则（预注册）")
    lines.append("")
    lines.append(f"- C1：Δ_factor_inter_vs_zlocal macro = {pp(m_c1):+.2f}pp（positive datasets {c1_ds_pos}/3）；node-correspondence macro = {pp(ctrl_m):+.2f}pp（positive {ctrl_ds_pos}/3）→ **{verdict_c1}**")
    for fam, (m, pos, tier) in fam_details.items():
        lines.append(f"- C2 family {fam}：macro = {pp(m):+.2f}pp（positive {pos}/3，tier {tier}）")
    lines.append(f"- C2 综合 → **{verdict_c2}**")
    lines.append(f"- C3：stable cells（Δ_L ≥ +0.30pp 且 ≥2/3 seeds positive，跨 ≥2 datasets）= {len(stable_cells)}；qualifying (cell,dataset) = {len(qual_pairs)}；GO 级 pairs = {len(go_pairs)}（{go_ds} datasets）；Spec-ok stable cells = {len(spec_ok_cells)}；H_cond_inter_final ≥ +0.20pp patterns = {len(h3_patterns)} → **{verdict_c3}**")
    lines.append(f"- 最终分类：**{case}**")
    lines.append("")

    (OUT_ROOT / "R20_C_SEMANTIC_INTERACTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[C] report done -> {OUT_ROOT / 'R20_C_SEMANTIC_INTERACTION_REPORT.md'} "
          f"(C1={verdict_c1}, C2={verdict_c2}, C3={verdict_c3}, case={case.split('（')[0]})", flush=True)


if __name__ == "__main__":
    main()
