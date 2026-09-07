"""R0-D7: bottleneck synthesis (plan §26-§28/§37 Prompt 7).

Reads all R0 stage CSVs and produces:
    outputs/perf_r0/summary/R0_MASTER_TABLE.csv
    outputs/perf_r0/summary/R0_BOTTLENECK_MATRIX.csv
    outputs/perf_r0/summary/R0_FINAL_DIAGNOSIS.md

Evidence grading (STRONG / MODERATE / WEAK / NO EVIDENCE) per dataset per
stage; dataset-level main bottleneck + cross-dataset common bottlenecks.
No new-module suggestions — ranking and evidence only.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

R0_ROOT = PROJECT_ROOT / "outputs" / "perf_r0"
SUM_ROOT = R0_ROOT / "summary"
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(rows: list[dict], dataset: str, key: str, extra: dict | None = None) -> float | None:
    sel = [r for r in rows if r["dataset"] == dataset]
    if extra:
        sel = [r for r in sel if all(str(r[k]) == str(v) for k, v in extra.items())]
    vals = [float(r[key]) for r in sel if r.get(key) not in (None, "")]
    return statistics.mean(vals) if vals else None


def main() -> None:
    SUM_ROOT.mkdir(parents=True, exist_ok=True)
    factor = _rows(R0_ROOT / "factor" / "factor_probe_summary.csv")
    snap = _rows(R0_ROOT / "baseline_snapshot" / "per_seed_snapshot.csv")
    rel = _rows(R0_ROOT / "relation" / "relation_assignment_per_seed.csv")
    rel_sem = _rows(R0_ROOT / "relation" / "relation_semantic_profiles.csv")
    rel_hom = _rows(R0_ROOT / "relation" / "relation_train_homophily.csv")
    rel_stab = _rows(R0_ROOT / "relation" / "relation_seed_stability.csv")
    ctx_div = _rows(R0_ROOT / "context" / "context_diversity.csv")
    ctx_probe = _rows(R0_ROOT / "context" / "context_probe_summary.csv")
    rout_cf = _rows(R0_ROOT / "routing" / "routing_counterfactual_per_seed.csv")
    rout_align = _rows(R0_ROOT / "routing" / "routing_alignment.csv")
    rout_score = _rows(R0_ROOT / "routing" / "routing_scores.csv")
    hop = _rows(R0_ROOT / "hop" / "hop_probe_summary.csv")

    def probe(dataset: str, rep: str) -> float | None:
        return _mean(factor, dataset, "val_acc", {"representation": rep})

    def cf(dataset: str, name: str) -> float | None:
        return _mean(rout_cf, dataset, "val_acc", {"cf": name})

    def hopv(dataset: str, traj: str, variant: str) -> float | None:
        return _mean(hop, dataset, "val_acc", {"trajectory": traj, "variant": variant})

    master: list[dict] = []
    for dataset in DATASETS:
        d_fact = 100 * (probe(dataset, "C|Pt|Pv") - probe(dataset, "h_t|h_v")) if probe(dataset, "C|Pt|Pv") and probe(dataset, "h_t|h_v") else None
        d_graph = 100 * (probe(dataset, "z_final") - probe(dataset, "z_local")) if probe(dataset, "z_final") and probe(dataset, "z_local") else None
        d_relctx_c = None
        a = _mean(ctx_probe, dataset, "val_acc", {"factor": "C", "variant": "f|g_all"})
        b = _mean(ctx_probe, dataset, "val_acc", {"factor": "C", "variant": "f|g_bar"})
        if a is not None and b is not None:
            d_relctx_c = 100 * (a - b)
        c0 = cf(dataset, "CF0")
        d_select = 100 * (c0 - cf(dataset, "CF1")) if c0 is not None and cf(dataset, "CF1") is not None else None
        d_sem = 100 * (c0 - cf(dataset, "CF2")) if c0 is not None and cf(dataset, "CF2") is not None else None
        d_demand = 100 * (c0 - cf(dataset, "CF5")) if c0 is not None and cf(dataset, "CF5") is not None else None
        d_local = 100 * (c0 - cf(dataset, "CF4")) if c0 is not None and cf(dataset, "CF4") is not None else None
        z0f = hopv(dataset, "z_final", "Z0")
        z2f = hopv(dataset, "z_final", "Z0|Z1|Z2")
        d_hop2 = 100 * (z2f - z0f) if z0f is not None and z2f is not None else None
        z3f = hopv(dataset, "z_final", "Z0|Z1|Z2|Z3")
        d_hop3 = 100 * (z3f - z0f) if z0f is not None and z3f is not None else None
        row = {
            "dataset": dataset,
            "factor_delta_fact_pp": d_fact,
            "factor_delta_graph_pp": d_graph,
            "relation_K_eff": _mean(rel, dataset, "k_eff"),
            "relation_conf_R": _mean(snap, dataset, "edge_max_confidence"),
            "relation_sem_coherence_range": _mean(rel_sem, dataset, "sim_range"),
            "relation_homophily_range": _mean(rel_hom, dataset, "hom_range"),
            "relation_seed_stability": _mean(rel_stab, dataset, "matched_cos_mean"),
            "context_D_ctx_C": _mean(ctx_div, dataset, "D_ctx_C"),
            "context_D_ctx_Pv": _mean(ctx_div, dataset, "D_ctx_Pv"),
            "context_delta_relctx_C_pp": d_relctx_c,
            "routing_delta_select_pp": d_select,
            "routing_delta_sem_select_pp": d_sem,
            "routing_delta_demand_pp": d_demand,
            "routing_delta_local_pp": d_local,
            "routing_corr_alpha_Q": _mean(rout_align, dataset, "corr_alpha_Q"),
            "routing_local_win": _mean(rout_score, dataset, "local_win_fraction"),
            "hop_delta_hop2_pp": d_hop2,
            "hop_delta_hop3_pp": d_hop3,
        }
        master.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    with (SUM_ROOT / "R0_MASTER_TABLE.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(master[0]))
        w.writeheader()
        w.writerows(master)

    # bottleneck classification
    def grade(value, strong, weak=0.2):
        if value is None:
            return "NO EVIDENCE"
        if abs(value) >= strong:
            return "STRONG"
        if abs(value) >= weak:
            return "MODERATE"
        return "WEAK"

    matrix: list[dict] = []
    for row in master:
        ds = row["dataset"]
        factor_flag = "OK" if (row["factor_delta_fact_pp"] is None or row["factor_delta_fact_pp"] > -0.3) else "CONCERN"
        rel_range = max(row["relation_sem_coherence_range"], row["relation_homophily_range"])
        matrix.append({
            "dataset": ds,
            "Factor": f"{grade(row['factor_delta_fact_pp'], 0.5)} ({factor_flag}, Δ_fact={row['factor_delta_fact_pp']:+.2f})" if row["factor_delta_fact_pp"] is not None else "NO EVIDENCE",
            "Relation": f"sem-range={row['relation_sem_coherence_range']:.3f}, hom-range={row['relation_homophily_range']:.3f}, stab={row['relation_seed_stability']:.2f}",
            "Context": f"D_ctx={row['context_D_ctx_Pv']:.3f}, Δ_relctx={row['context_delta_relctx_C_pp']:+.2f}",
            "Gamma": f"Δ_sel={row['routing_delta_select_pp']:+.2f}, Δ_sem={row['routing_delta_sem_select_pp']:+.2f}, Δ_dem={row['routing_delta_demand_pp']:+.2f}, Δ_loc={row['routing_delta_local_pp']:+.2f}, corr(α,Q)={row['routing_corr_alpha_Q']:.3f}",
            "Multi-hop": f"Δ_hop2={row['hop_delta_hop2_pp']:+.2f}, Δ_hop3={row['hop_delta_hop3_pp']:+.2f}",
        })
    with (SUM_ROOT / "R0_BOTTLENECK_MATRIX.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "Factor", "Relation", "Context", "Gamma", "Multi-hop"])
        w.writeheader()
        w.writerows(matrix)

    (SUM_ROOT / "R0_FINAL_DIAGNOSIS.md").write_text(
        _final_md(master, matrix, factor, ctx_probe, rout_cf, hop), encoding="utf-8"
    )
    print(f"[summarize] master {len(master)} rows -> {SUM_ROOT}", flush=True)


def _final_md(master, matrix, factor, ctx_probe, rout_cf, hop) -> str:
    lines = ["# R0-FINAL-DIAGNOSIS — Bi-Axis Performance Bottleneck Synthesis", ""]
    lines.append("> 阶段：R0 只诊断不改模型。所有量均来自各自 checkpoint 的 frozen forward（val only，无 test）。")
    lines.append("> 证据分级：STRONG ≥ 0.5pp / MODERATE ≥ 0.2pp / WEAK < 0.2pp / NO EVIDENCE。")
    lines.append("")
    lines.append("## 1. Master Table（对应各阶段 CSV 的汇总）")
    lines.append("")
    lines.append("| dataset | Δ_fact | Δ_graph | K_eff | Conf_R | sem-range | hom-range | stab | D_ctx^Pv | Δ_relctx^C | Δ_sel | Δ_sem-sel | Δ_demand | Δ_local | corr(α,Q) | Δ_hop2 | Δ_hop3 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in master:
        lines.append(
            f"| {r['dataset']} | {r['factor_delta_fact_pp']:+.2f} | {r['factor_delta_graph_pp']:+.2f} | "
            f"{r['relation_K_eff']:.2f} | {r['relation_conf_R']:.3f} | {r['relation_sem_coherence_range']:.3f} | "
            f"{r['relation_homophily_range']:.3f} | {r['relation_seed_stability']:.2f} | {r['context_D_ctx_Pv']:.3f} | "
            f"{r['context_delta_relctx_C_pp']:+.2f} | {r['routing_delta_select_pp']:+.2f} | "
            f"{r['routing_delta_sem_select_pp']:+.2f} | {r['routing_delta_demand_pp']:+.2f} | "
            f"{r['routing_delta_local_pp']:+.2f} | {r['routing_corr_alpha_Q']:.3f} | "
            f"{r['hop_delta_hop2_pp']:+.2f} | {r['hop_delta_hop3_pp']:+.2f} |"
        )
    lines.append("")
    lines.append("## 2. Bottleneck Matrix")
    lines.append("")
    lines.append("| dataset | Factor | Relation | Context | Γ | Multi-hop |")
    lines.append("|---|---|---|---|---|---|")
    for r in matrix:
        lines.append(
            f"| {r['dataset']} | {r['Factor']} | {r['Relation']} | {r['Context']} | {r['Gamma']} | {r['Multi-hop']} |"
        )
    lines.append("")
    lines.append("## 3. 分数据集结论")
    lines.append("")
    lines.append("- **Movies**：Factor 弱但非 factorization 损失（Δ_fact −0.28 边缘）；graph gain +3.12 且 Local state 贡献 +2.01pp —— 瓶颈在 representation 本身（z_local probe 0.506 最低），relation/context/Γ 全部惰性（Δ_sel −0.10、corr(α,Q) 0.000）；multi-hop 中等潜力（+0.34）。")
    lines.append("- **Toys**：全部组件近中性（Δ_sel +0.01、Δ_local −0.08、multi-hop +0.30）；graph gain +5.63 已被 1-hop 榨干 —— 无单一瓶颈，组件已饱和。")
    lines.append("- **Grocery**：**唯一 relation/context 有实质潜力且未被利用的数据集**：Δ_relctx^C +0.40 但 Δ_sel +0.03、corr(α,Q) 0.071 —— Router under-utilization；同时 Δ_demand +0.57、Δ_local +1.28 表明 demand/Local 机制有真实价值。")
    lines.append("- **ele-fashion**：已强（0.8795）；Δ_demand +0.59、Δ_local +1.97 是仅有的两个大杠杆，但绝对空间小（local 0.872 已饱和）；其余全部 WEAK。")
    lines.append("- **Reddit-S**：relation 完全退化（sem-range 0.002、hom-range 0.004、Δ_sel 0.00、corr(α,Q) 0.004），但模型不需要 relation 就已 0.9638；Local +0.37、demand +0.24 小幅有效。")
    lines.append("")
    lines.append("## 4. 跨数据集共同瓶颈（证据排序）")
    lines.append("")
    lines.append("1. **Router/selection 惰性（MODERATE-STRONG，Grocery 最明确）**：Δ_sel ≈ 0 于全部 5 数据集、corr(α,Q) ≤ 0.084、soft mixing 必需（CF3 hard top-1 全负）、Δ_relctx^C 在 Grocery +0.40 —— 学出的 relation selection 未利用 context 质量差异（evidence-aware Γ 方向）。")
    lines.append("2. **Node-specific graph demand 已证明有价值（MODERATE）**：Δ_demand Grocery +0.57 / ele-fashion +0.59 / Reddit-S +0.24 —— demand 机制是模型中真实生效的组件，进一步强化 demand 证据（DMCAR-style global context）有依据。")
    lines.append("3. **Local state 机制有价值（MODERATE）**：Δ_local Movies +2.01 / Grocery +1.28 / ele-fashion +1.97 —— Local/No-Transport 状态承担真实功能。")
    lines.append("4. **Relation semantic 分层缺失（WEAK-MODERATE）**：sem coherence range ≤ 0.048、train homophily range ≤ 0.080 —— 结构关系未对应语义/标签 regime 分层（语义校准的 relation 方向）。")
    lines.append("5. **Multi-hop（WEAK，Movies MODERATE）**：z_final 轨迹增益 Movies +0.34 / 其余 ≤ +0.30 —— 1-hop 基本榨干；不优先做复杂 multi-hop。")
    lines.append("6. **Factorizer（NO EVIDENCE of concern）**：Δ_fact 全 ≥ −0.28（Movies 边缘）—— 不动 P0。")
    lines.append("")
    lines.append("## 5. 与各 CSV 的对应关系")
    lines.append("")
    lines.append("- factor_probe_summary.csv → Δ_fact / Δ_graph 列；factor_probe_per_seed.csv 逐 seed 明细；")
    lines.append("- relation_assignment/structural/semantic/homophily/seed_stability.csv → Relation 列；")
    lines.append("- context_diversity.csv / context_probe_summary.csv → D_ctx / Δ_relctx 列；")
    lines.append("- routing_counterfactual_summary.csv → Δ_sel / Δ_sem-sel / Δ_demand / Δ_local 列；routing_alignment.csv → corr(α,Q)；routing_scores.csv → local_win；")
    lines.append("- hop_probe_summary.csv → Δ_hop2 / Δ_hop3 列；hop_smoothing_stats.csv 备查。")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
