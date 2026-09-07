"""R2-0A: Cross-Factor Transfer Probe (plan §4, user-augmented A4/A5/A6).

Frozen OFR checkpoints only; no training; no Test (the r20 wrapper masks
test labels and cuts test_idx). Fixed probe = R0 ridge_probe
(StandardScaler + RidgeClassifier(alpha=1.0), fit TRAIN / eval VAL).

Layers:
    A1 target-specific plain 3x3:      X = [F^b | N^a],  N^a = P F^a
    A2 target-specific relation 3x3:   X = [F^b | G^a],  G^a = g_perm[:,a].flat
    A3 all-source upper bound:         [F^b | N^all] / [F^b | G^all]
    A4 all-local conditioned:          [L|N^a] / [L|G^a] vs Probe(L), L=[C|Pt|Pv]
    A5 current-A0 incremental:         [z_final|N^a] / [z_final|G^a] vs Probe(z_final)
    A6 fixed shuffled-neighborhood negative control: ONE permutation
       (torch seed 20260904), node-row shuffle of N^a / G^a (G^a's K/d
       internals untouched). No second permutation.

R1..R4 prototype indices are NOT interpreted across seeds (plan: prototype
order is arbitrary; cells are read by position only).

Outputs: outputs/perf_r20/cross_factor/{csv x6, R20_A_CROSS_FACTOR_REPORT.md}
Usage:
    python scripts/perf_r20_a_cross_factor.py --gpus 0,1
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
    context_concat,
    extract_forward,
    factor_tensor,
    load_setup,
    ridge_probe,
    write_csv,
)
from src.models.biaxis_p1_components import neighbor_mean  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / "perf_r20" / "cross_factor"
FACTORS = ["C", "Pt", "Pv"]
SHUFFLE_SEED = 20260904

# ---------------------------------------------------------------------------
# Pre-registered thresholds (plan §4.5 + user Prompt 2 amendments)
# ---------------------------------------------------------------------------
ADV_STABLE_PP = 0.20       # off-diagonal Adv 3-seed mean >= +0.20pp
ADV_MODERATE_PP = 0.15     # partial-signal floor
POS_SEEDS_MIN = 2          # >= 2/3 seeds positive
LEVEL23_PP = 0.20          # Level-II/III gain >= ~+0.20pp (cross-seed)
N_STABLE_MIN = 2           # at least 2 stable off-diagonal cells
ALLSOURCE_STRONG_PP = 0.30  # original §4.5 all-source headroom (context only)
ALLSOURCE_MODERATE_PP = 0.15


def _stat(values: list[float]) -> tuple[float, float, int]:
    mean = statistics.mean(values) if values else float("nan")
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    pos = sum(1 for v in values if v > 0)
    return mean, std, pos


def _probe(tensor: torch.Tensor, setup, tag: str) -> tuple[float, float]:
    probe = ridge_probe(tensor, setup)
    print(f"[A] {setup.dataset:12s} s{setup.seed} {tag:28s} acc={probe['val_acc']:.4f}", flush=True)
    return probe["val_acc"], probe["val_macro_f1"]


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-0A cross-factor transfer probe")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    datasets = [d for d in (args.datasets or ",".join(DATASETS)).split(",") if d in DATASETS]
    seeds = [int(s) for s in (args.seeds or ",".join(map(str, SEEDS))).split(",") if s]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # per-seed storage
    local_rows: dict[tuple, dict] = {}              # (ds, seed, target) -> (acc, f1)
    cell_rows: dict[tuple, dict] = {}               # (kind, ds, seed, a, b) -> (acc, f1)
    allsrc_rows: dict[tuple, dict] = {}             # (kind, ds, seed, target) -> (acc, f1)
    l4_rows: dict[tuple, dict] = {}                 # (kind, ds, seed, src) -> (acc, f1); kind "base" = Probe(L)
    l5_rows: dict[tuple, dict] = {}                 # same for z_final layer
    shuf_rows: dict[tuple, dict] = {}               # (block, kind, ds, seed, cell) -> real/shuf (acc, f1)

    lifecycles = [(ds, s) for ds in datasets for s in seeds]
    for idx, (ds, s) in enumerate(lifecycles):
        device = torch.device(f"cuda:{gpus[idx % len(gpus)]}")
        setup = load_setup(ds, s, device)
        fex = extract_forward(setup)
        n_nodes = int(fex["f_block"].size(0))
        edge_index = fex["edge_index"]
        chunk = setup.model.edge_chunk_size

        F = {f: factor_tensor(fex, f) for f in FACTORS}
        L = context_concat([F[f] for f in FACTORS])
        Z = fex["z_final"]
        g_perm = fex["graph_out"]["g_perm"]  # [N, 3, K, d]

        N = {f: neighbor_mean(edge_index, F[f], n_nodes, edge_chunk_size=chunk) for f in FACTORS}
        G = {f: g_perm[:, i].reshape(n_nodes, -1) for i, f in enumerate(FACTORS)}

        perm = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(SHUFFLE_SEED)).to(device)
        N_sh = {f: N[f][perm] for f in FACTORS}
        G_sh = {f: G[f][perm] for f in FACTORS}
        N_all = context_concat([N[f] for f in FACTORS])
        G_all = context_concat([G[f] for f in FACTORS])
        N_all_sh = context_concat([N_sh[f] for f in FACTORS])
        G_all_sh = context_concat([G_sh[f] for f in FACTORS])

        # --- local baselines -------------------------------------------------
        for b in FACTORS:
            acc, f1 = _probe(F[b], setup, f"local_F[{b}]")
            local_rows[(ds, s, b)] = {"val_acc": acc, "val_macro_f1": f1}
            cell_rows[("plain", ds, s, "local", b)] = {"val_acc": acc, "val_macro_f1": f1}
            cell_rows[("relation", ds, s, "local", b)] = {"val_acc": acc, "val_macro_f1": f1}

        # --- A1 / A2 3x3 cells (real + shuffled) ----------------------------
        for kind, ctx, ctx_sh, tag in (
            ("plain", N, N_sh, "N"),
            ("relation", G, G_sh, "G"),
        ):
            for a in FACTORS:
                for b in FACTORS:
                    acc, f1 = _probe(
                        context_concat([F[b], ctx[a]]), setup, f"{tag}[{a}]->F[{b}]"
                    )
                    cell_rows[(kind, ds, s, a, b)] = {"val_acc": acc, "val_macro_f1": f1}
                    sacc, sf1 = _probe(
                        context_concat([F[b], ctx_sh[a]]), setup, f"{tag}_sh[{a}]->F[{b}]"
                    )
                    shuf_rows[("A1", kind, ds, s, f"{a}->{b}")] = {
                        "real_acc": acc, "shuf_acc": sacc, "real_f1": f1, "shuf_f1": sf1,
                    }

        # --- A3 all-source ---------------------------------------------------
        for kind, all_ctx, tag in (("plain", N_all, "N_all"), ("relation", G_all, "G_all")):
            for b in FACTORS:
                acc, f1 = _probe(context_concat([F[b], all_ctx]), setup, f"{tag}->F[{b}]")
                allsrc_rows[(kind, ds, s, b)] = {"val_acc": acc, "val_macro_f1": f1}

        # --- A4 all-local conditioned ----------------------------------------
        l_acc, l_f1 = _probe(L, setup, "L=[C|Pt|Pv]")
        l4_rows[("base", ds, s, "base")] = {"val_acc": l_acc, "val_macro_f1": l_f1}
        for kind, ctx, ctx_sh, all_ctx, all_ctx_sh, tag in (
            ("plain", N, N_sh, N_all, N_all_sh, "L|N"),
            ("relation", G, G_sh, G_all, G_all_sh, "L|G"),
        ):
            for a in FACTORS:
                acc, f1 = _probe(context_concat([L, ctx[a]]), setup, f"{tag}^[{a}]")
                l4_rows[(kind, ds, s, a)] = {"val_acc": acc, "val_macro_f1": f1}
                sacc, sf1 = _probe(context_concat([L, ctx_sh[a]]), setup, f"{tag}_sh[{a}]")
                shuf_rows[("A4", kind, ds, s, a)] = {
                    "real_acc": acc, "shuf_acc": sacc, "real_f1": f1, "shuf_f1": sf1,
                }
            acc, f1 = _probe(context_concat([L, all_ctx]), setup, f"{tag}^[all]")
            l4_rows[(kind, ds, s, "all")] = {"val_acc": acc, "val_macro_f1": f1}
            sacc, sf1 = _probe(context_concat([L, all_ctx_sh]), setup, f"{tag}_sh[all]")
            shuf_rows[("A4", kind, ds, s, "all")] = {
                "real_acc": acc, "shuf_acc": sacc, "real_f1": f1, "shuf_f1": sf1,
            }

        # --- A5 current-A0 incremental ---------------------------------------
        z_acc, z_f1 = _probe(Z, setup, "z_final")
        l5_rows[("base", ds, s, "base")] = {"val_acc": z_acc, "val_macro_f1": z_f1}
        for kind, ctx, ctx_sh, all_ctx, all_ctx_sh, tag in (
            ("plain", N, N_sh, N_all, N_all_sh, "Z|N"),
            ("relation", G, G_sh, G_all, G_all_sh, "Z|G"),
        ):
            for a in FACTORS:
                acc, f1 = _probe(context_concat([Z, ctx[a]]), setup, f"{tag}^[{a}]")
                l5_rows[(kind, ds, s, a)] = {"val_acc": acc, "val_macro_f1": f1}
                sacc, sf1 = _probe(context_concat([Z, ctx_sh[a]]), setup, f"{tag}_sh[{a}]")
                shuf_rows[("A5", kind, ds, s, a)] = {
                    "real_acc": acc, "shuf_acc": sacc, "real_f1": f1, "shuf_f1": sf1,
                }
            acc, f1 = _probe(context_concat([Z, all_ctx]), setup, f"{tag}^[all]")
            l5_rows[(kind, ds, s, "all")] = {"val_acc": acc, "val_macro_f1": f1}
            sacc, sf1 = _probe(context_concat([Z, all_ctx_sh]), setup, f"{tag}_sh[all]")
            shuf_rows[("A5", kind, ds, s, "all")] = {
                "real_acc": acc, "shuf_acc": sacc, "real_f1": f1, "shuf_f1": sf1,
            }

        del fex
        torch.cuda.empty_cache()
        print(f"[A] {ds:12s} s{s} done ({idx + 1}/{len(lifecycles)})", flush=True)

    # ------------------------------------------------------------------ CSVs
    def dump(path: Path, key_fn, row_fn, key_iter):
        rows = [row_fn(key) for key in key_iter if key in key_fn]
        write_csv(path, rows)

    dump(OUT_ROOT / "cross_factor_plain_cells.csv", cell_rows,
         lambda k: {"dataset": k[1], "seed": k[2], "source": k[3], "target": k[4], **cell_rows[k]},
         [(kk) for kk in cell_rows if kk[0] == "plain"])
    dump(OUT_ROOT / "cross_factor_relation_cells.csv", cell_rows,
         lambda k: {"dataset": k[1], "seed": k[2], "source": k[3], "target": k[4], **cell_rows[k]},
         [(kk) for kk in cell_rows if kk[0] == "relation"])
    dump(OUT_ROOT / "cross_factor_all_source.csv", allsrc_rows,
         lambda k: {"dataset": k[1], "seed": k[2], "kind": k[0], "target": k[3], **allsrc_rows[k]},
         list(allsrc_rows))
    dump(OUT_ROOT / "cross_factor_local_conditioned.csv", l4_rows,
         lambda k: {"dataset": k[1], "seed": k[2], "kind": k[0], "source": k[3], **l4_rows[k]},
         list(l4_rows))
    dump(OUT_ROOT / "cross_factor_final_incremental.csv", l5_rows,
         lambda k: {"dataset": k[1], "seed": k[2], "kind": k[0], "source": k[3], **l5_rows[k]},
         list(l5_rows))
    dump(OUT_ROOT / "cross_factor_shuffle_control.csv", shuf_rows,
         lambda k: {"dataset": k[2], "seed": k[3], "block": k[0], "kind": k[1], "cell": k[4],
                    "real_acc": shuf_rows[k]["real_acc"], "shuf_acc": shuf_rows[k]["shuf_acc"],
                    "diff_acc": shuf_rows[k]["real_acc"] - shuf_rows[k]["shuf_acc"],
                    "real_f1": shuf_rows[k]["real_f1"], "shuf_f1": shuf_rows[k]["shuf_f1"],
                    "diff_f1": shuf_rows[k]["real_f1"] - shuf_rows[k]["shuf_f1"]},
         list(shuf_rows))
    print(f"[A] CSVs written -> {OUT_ROOT}", flush=True)

    # ------------------------------------------------------------------ summaries
    def cells_acc(kind, ds, s, a, b):
        return cell_rows[(kind, ds, s, a, b)]["val_acc"]

    # Level-I: U / Adv per (kind, ds, a, b) over seeds
    u_stats, adv_stats = {}, {}
    for kind in ("plain", "relation"):
        for ds in datasets:
            for a in FACTORS:
                for b in FACTORS:
                    u_vals = [cells_acc(kind, ds, s, a, b) - cell_rows[(kind, ds, s, "local", b)]["val_acc"] for s in seeds]
                    adv_vals = [cells_acc(kind, ds, s, a, b) - cells_acc(kind, ds, s, b, b) for s in seeds]
                    u_stats[(kind, ds, a, b)] = _stat(u_vals)
                    adv_stats[(kind, ds, a, b)] = _stat(adv_vals)

    # A3 all-source delta vs same-factor diagonal
    asrc_stats = {}
    for kind in ("plain", "relation"):
        for ds in datasets:
            for b in FACTORS:
                vals = [allsrc_rows[(kind, ds, s, b)]["val_acc"] - cells_acc(kind, ds, s, b, b) for s in seeds]
                asrc_stats[(kind, ds, b)] = _stat(vals)

    # A4 / A5 gains vs base
    def layer_gains(rows, kind, ds, src):
        vals = [rows[(kind, ds, s, src)]["val_acc"] - rows[("base", ds, s, "base")]["val_acc"] for s in seeds]
        return _stat(vals)

    l4_stats = {(kind, ds, src): layer_gains(l4_rows, kind, ds, src)
                for kind in ("plain", "relation") for ds in datasets for src in FACTORS + ["all"]}
    l5_stats = {(kind, ds, src): layer_gains(l5_rows, kind, ds, src)
                for kind in ("plain", "relation") for ds in datasets for src in FACTORS + ["all"]}

    # A6 real - shuffled per cell
    shuf_stats = {}
    for key, row in shuf_rows.items():
        block, kind, ds, s, cell = key
        vals = [shuf_rows[(block, kind, ds, ss, cell)]["real_acc"] - shuf_rows[(block, kind, ds, ss, cell)]["shuf_acc"] for ss in seeds]
        shuf_stats[(block, kind, ds, cell)] = _stat(vals)

    # ------------------------------------------------------------------ report
    pp = lambda v: 100.0 * v
    lines = ["# R20-A CROSS-FACTOR TRANSFER REPORT", ""]
    lines.append("> R2-0 审计状态：**Audit PASS（129/129，max diff 5.96e-8）**。")
    lines.append("> 协议：frozen OFR checkpoints（M/T/G × 42/43/44）；固定 Ridge probe")
    lines.append("> （StandardScaler + RidgeClassifier(alpha=1.0)，TRAIN fit / VAL eval）；")
    lines.append("> 全程无 Test（r20 wrapper 已 mask test labels 并切断 test_idx）。")
    lines.append("> 数值均为 Val Acc 的 pp 变化（×100）；3-seed mean±population std(ddof=0)，")
    lines.append("> 括号内 = positive seed count。R1..R4 原型索引不跨 seed 对齐，cell 只按位置读。")
    lines.append("")

    def fmt(stat):
        m, s, p = stat
        return f"{pp(m):+.2f}±{pp(s):.2f} ({p})"

    # Level-I matrices
    for kind, label in (("plain", "A1 plain N^a=P F^a"), ("relation", "A2 relation G^a=g_perm flat")):
        lines.append(f"## Level-I — {label}：U（vs local F^b）与 Adv（vs 同 factor 对角 cell）")
        lines.append("")
        for metric, stats in (("U = Probe([F^b|ctx^a]) − Probe(F^b)", u_stats), ("Adv = Probe([F^b|ctx^a]) − Probe([F^b|ctx^b])", adv_stats)):
            lines.append(f"### {metric}")
            lines.append("")
            lines.append("| dataset | source\\target | C | Pt | Pv |")
            lines.append("|---|---|---:|---:|---:|")
            for ds in datasets:
                for a in FACTORS:
                    cells = " | ".join(fmt(stats[(kind, ds, a, b)]) for b in FACTORS)
                    lines.append(f"| {ds} | {a} | {cells} |")
                lines.append("")
            lines.append("")

    # off-diagonal per-seed detail (acc + f1)
    lines.append("## Level-I 附：off-diagonal Adv per-seed 明细")
    lines.append("")
    lines.append("| dataset | kind | cell | Adv_acc s42/s43/s44 (pp) | Adv_f1 s42/s43/s44 (pp) |")
    lines.append("|---|---|---|---|---|")
    for kind in ("plain", "relation"):
        for ds in datasets:
            for a in FACTORS:
                for b in FACTORS:
                    if a == b:
                        continue
                    accs = [pp(cells_acc(kind, ds, s, a, b) - cells_acc(kind, ds, s, b, b)) for s in seeds]
                    f1s = [pp(cell_rows[(kind, ds, s, a, b)]["val_macro_f1"] - cell_rows[(kind, ds, s, b, b)]["val_macro_f1"]) for s in seeds]
                    lines.append(f"| {ds} | {kind} | {a}->{b} | " + " / ".join(f"{v:+.2f}" for v in accs) + " | " + " / ".join(f"{v:+.2f}" for v in f1s) + " |")
    lines.append("")

    # A3 all-source
    lines.append("## A3 — All-source upper bound（vs 同 factor 对角 cell）")
    lines.append("")
    lines.append("| dataset | kind | Δ_all C | Δ_all Pt | Δ_all Pv |")
    lines.append("|---|---|---:|---:|---:|")
    for kind in ("plain", "relation"):
        for ds in datasets:
            cells = " | ".join(fmt(asrc_stats[(kind, ds, b)]) for b in FACTORS)
            lines.append(f"| {ds} | {kind} | {cells} |")
    lines.append("")

    # Level-II A4
    lines.append("## Level-II — A4 all-local conditioned：Probe([L|ctx]) − Probe(L)，L=[C|Pt|Pv]")
    lines.append("")
    lines.append("| dataset | kind | N^C | N^Pt | N^Pv | all |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for kind in ("plain", "relation"):
        for ds in datasets:
            cells = " | ".join(fmt(l4_stats[(kind, ds, src)]) for src in FACTORS + ["all"])
            lines.append(f"| {ds} | {kind} | {cells} |")
    lines.append("")

    # Level-III A5
    lines.append("## Level-III — A5 current-A0 incremental：Probe([z_final|ctx]) − Probe(z_final)")
    lines.append("")
    lines.append("| dataset | kind | N^C | N^Pt | N^Pv | all |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for kind in ("plain", "relation"):
        for ds in datasets:
            cells = " | ".join(fmt(l5_stats[(kind, ds, src)]) for src in FACTORS + ["all"])
            lines.append(f"| {ds} | {kind} | {cells} |")
    lines.append("")

    # A6 shuffle control
    lines.append("## A6 — Shuffled-neighborhood negative control（perm seed 20260904，单 permutation）")
    lines.append("")
    lines.append("| block | kind | cell | real−shuffled per dataset (M/T/G, pp) |")
    lines.append("|---|---|---|---|")
    for block in ("A1", "A4", "A5"):
        for kind in ("plain", "relation"):
            cells_sorted = sorted({k[3] for k in shuf_stats if k[0] == block and k[1] == kind and k[2] in datasets})
            for cell in cells_sorted:
                vals = []
                for ds in datasets:
                    key = (block, kind, ds, cell)
                    if key in shuf_stats:
                        m, s, p = shuf_stats[key]
                        vals.append(f"{ds} {pp(m):+.2f}±{pp(s):.2f}({p})")
                lines.append(f"| {block} | {kind} | {cell} | " + "；".join(vals) + " |")
    lines.append("")

    # ------------------------------------------------------------------ verdict
    stable = [(kind, ds, a, b) for (kind, ds, a, b), (m, s, p) in adv_stats.items()
              if a != b and m >= ADV_STABLE_PP / 100.0 and p >= POS_SEEDS_MIN]
    stable_ds = {ds for (_, ds, _, _) in stable}
    level23 = [(level, kind, ds, src) for (level, stats) in (("A4", l4_stats), ("A5", l5_stats))
               for (kind, ds, src), (m, s, p) in stats.items()
               if m >= LEVEL23_PP / 100.0 and p >= POS_SEEDS_MIN]
    # shuffle gate on the qualifying level-II/III cells
    shuf_gate_ok = True
    for level, kind, ds, src in level23:
        key = (level, kind, ds, src)
        m, s, p = shuf_stats.get(key, (float("nan"), 0.0, 0))
        shuf_gate_ok = shuf_gate_ok and (m > 0)
    level1_strong = len(stable) >= N_STABLE_MIN and len(stable_ds) >= 2
    level23_strong = len(level23) > 0 and shuf_gate_ok

    any_partial = (any(a != b and m >= ADV_MODERATE_PP / 100.0 and p >= POS_SEEDS_MIN
                       for (kind, ds, a, b), (m, s, p) in adv_stats.items()) or len(level23) > 0)

    lines.append("## Verdict（预注册判定）")
    lines.append("")
    lines.append(f"- Level-I stable off-diagonal cells（mean Adv ≥ +{ADV_STABLE_PP:.2f}pp 且 ≥{POS_SEEDS_MIN}/3 seeds positive）：**{len(stable)}**；覆盖数据集：{sorted(stable_ds)}")
    if level23:
        lines.append(f"- Level-II/III 信号（mean gain ≥ +{LEVEL23_PP:.2f}pp 且 ≥{POS_SEEDS_MIN}/3 seeds positive）：**{len(level23)}** 个 qualifying cells，其中 mean real−shuffled > 0 的比例：{sum(1 for level, kind, ds, src in level23 if shuf_stats.get((level, kind, ds, src), (float('nan'), 0, 0))[0] > 0)}/{len(level23)}（shuffle gate pass：{shuf_gate_ok}）")
    else:
        lines.append(f"- Level-II/III 信号（mean gain ≥ +{LEVEL23_PP:.2f}pp 且 ≥{POS_SEEDS_MIN}/3 seeds positive）：**0**（shuffle gate 无 qualifying cells，vacuous）")
    lines.append("")
    if level1_strong and level23_strong:
        verdict = "STRONG"
        lines.append(f"**Verdict: STRONG** — Level-I 稳定 off-diagonal + Level-II/III ≥+{LEVEL23_PP:.2f}pp 跨 seed 正信号 + real>shuffled。")
    elif any_partial:
        verdict = "MODERATE"
        lines.append("**Verdict: MODERATE** — 存在部分正信号但未同时满足三层条件（明细见上表）。")
    else:
        verdict = "NO EVIDENCE"
        lines.append("**Verdict: NO EVIDENCE** — 无稳定 off-diagonal，且 Level-II/III 无 ≥+0.20pp 跨 seed 正信号。")
    lines.append("")
    # original §4.5 all-source context numbers
    lines.append("### 原始 §4.5 all-source headroom（仅作上下文，不作 GO 依据）")
    lines.append("")
    for kind in ("plain", "relation"):
        macro = statistics.mean(statistics.mean(asrc_stats[(kind, ds, b)][0] for b in FACTORS) for ds in datasets)
        lines.append(f"- {kind} all-source Δ macro mean(M/T/G, 9 cells)：**{pp(macro):+.2f}pp**")
    lines.append("")
    lines.append("> 注意：all-source 维度更高，只作 upper bound；GO 依据 = 三层条件。")
    lines.append("")

    (OUT_ROOT / "R20_A_CROSS_FACTOR_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[A] report done -> {OUT_ROOT / 'R20_A_CROSS_FACTOR_REPORT.md'} (verdict={verdict})", flush=True)


if __name__ == "__main__":
    main()
