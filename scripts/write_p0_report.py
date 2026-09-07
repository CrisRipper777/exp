"""Generate the final P0_REPORT.md from aggregated CSVs."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs" / "p0"
NC_D = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
LP_D = ["sports-copurchase", "cloth-copurchase"]


def read(name: str) -> list[dict]:
    with (OUT / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ms(rows: list[dict], key: str) -> str:
    vals = [float(r[key]) for r in rows]
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f}±{statistics.pstdev(vals):.3f}"


def main() -> None:
    nc = read("p0_nc_summary.csv")
    lp = read("p0_lp_summary.csv")
    cmp = read("p0_protocol_comparison.csv")

    lines: list[str] = []
    add = lines.append
    add("# P0_REPORT — Factor-Dependent Neighborhood Utility（最终版）")
    add("")
    add("> 协议：RPTA 式（NC 全图训练 / LP 采样训练，2026-09-02 改造后）；模型 biaxis_p0；")
    add("> λ_common=0.02 / λ_orth=0.01 / λ_recon=0.3（D1 后修复 Common 坍塌）；seeds 42/43/44；probe seed 固定 42")
    add("> 新旧协议对比见 `p0_protocol_comparison.csv`（旧协议结果归档于 `old_protocol/`）")
    add("")

    add("## 1. NC：5 数据集 × 3 seeds（mean±population std）")
    add("")
    add("| Dataset | S_C | S_P | effrank C | rho C/T | rho C/V | Jacc20 C/T | ΔAcc C | ΔAcc Pt | ΔAcc Pv | Conflict C/T | Conflict C/V | Fused Test Acc |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in NC_D:
        rows = [r for r in nc if r["dataset"] == d]
        add(
            f"| {d} | {ms(rows, 'common_sim')} | {ms(rows, 'private_sim')} | {ms(rows, 'effrank_c')} | "
            f"{ms(rows, 'rho_C_Pt')} | {ms(rows, 'rho_C_Pv')} | {ms(rows, 'jaccard_top20_C_Pt')} | "
            f"{ms(rows, 'delta_acc_C')} | {ms(rows, 'delta_acc_Pt')} | {ms(rows, 'delta_acc_Pv')} | "
            f"{ms(rows, 'conflict_C_Pt')} | {ms(rows, 'conflict_C_Pv')} | {ms(rows, 'fused_test_acc')} |"
        )
    add("")

    add("## 2. LP：2 数据集 × 3 seeds")
    add("")
    add("| Dataset | rho C/T | rho C/V | ΔMRR C | ΔMRR Pt | ΔMRR Pv | RR Conflict C/T | RR Conflict C/V | Fused Test MRR |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in LP_D:
        rows = [r for r in lp if r["dataset"] == d]
        add(
            f"| {d} | {ms(rows, 'rho_C_Pt')} | {ms(rows, 'rho_C_Pv')} | {ms(rows, 'delta_mrr_C')} | "
            f"{ms(rows, 'delta_mrr_Pt')} | {ms(rows, 'delta_mrr_Pv')} | {ms(rows, 'conflict_C_Pt')} | "
            f"{ms(rows, 'conflict_C_Pv')} | {ms(rows, 'fused_test_mrr')} |"
        )
    add("")

    add("## 3. 新旧协议对比（关键指标 old → new）")
    add("")
    add("| Task | Dataset | Seed | fused test | S_C | effrank C | rho C/T | rho C/V | Conflict C/T | Conflict C/V |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in cmp:
        task = r["task"]
        add(
            f"| {task} | {r['dataset']} | {r['seed']} | "
            f"{float(r['old_fused_test_' + task]):.4f}→{float(r['new_fused_test_' + task]):.4f} | "
            f"{float(r['old_common_sim']):.3f}→{float(r['new_common_sim']):.3f} | "
            f"{float(r['old_effrank_c']):.1f}→{float(r['new_effrank_c']):.1f} | "
            f"{float(r['old_rho_C_Pt']):.2f}→{float(r['new_rho_C_Pt']):.2f} | "
            f"{float(r['old_rho_C_Pv']):.2f}→{float(r['new_rho_C_Pv']):.2f} | "
            f"{float(r['old_conflict_C_Pt']):.2f}→{float(r['new_conflict_C_Pt']):.2f} | "
            f"{float(r['old_conflict_C_Pv']):.2f}→{float(r['new_conflict_C_Pv']):.2f} |"
        )
    add("")

    add("## 4. GO / NO-GO 判据（计划 §19）")
    add("")
    add("| 判据 | 结论 | 证据 |")
    add("|---|---|---|")
    add("| S_C > S_P 且无 rank 坍塌 | ✅ GO | 15/15 NC run：S_C 0.61–0.94 vs S_P ±0.03；effrank_c 46–67 |")
    add("| Spearman < 0.7 或 Top20 Jaccard < 0.6 | ✅ GO | Movies rho_C/T 0.46–0.62、Reddit-S 0.44–0.60；LP rho_C/V 0.05–0.34 |")
    add("| Conflict > 15% 于 ≥2 NC 数据集且 seed 稳定 | ✅ GO | Movies 0.24–0.35、ele-fashion 0.28–0.46、Toys 0.25–0.36、Reddit-S C/V 0.20–0.73 |")
    add("| LP：factor-wise ΔMRR 不一致 + 非平凡 RR 冲突 | ✅ GO | sports ΔMRR C +0.3~1.5 vs Pt/Pv +6.9~8.5；cloth Pt/Pv +5.3~6.1；RR 冲突 0.31–0.43 |")
    add("")
    add("**决策：STRONG GO → 进入 P1**（Semantic Factor × Structural Relation 耦合）")
    add("")

    add("## 5. 五个核心问题（计划 §28）的回答")
    add("")
    add("1. **Common 与 Private 是否形成不同语义空间？** 是。S_C 0.61–0.94 显著高于 S_P（±0.03），Common–Private 协方差重叠 0.01–0.15，effrank 46–67 无坍塌。")
    add("2. **同一条边在不同因子空间的相关性是否不同？** 是。NC 上 rho_C/Pt 0.44–0.74、rho_Pt/Pv 低至 −0.08~0.42；LP 上 rho 低至 0.05——同一条边的有用性排序因因子而异。")
    add("3. **同一邻域对不同因子的传播收益是否不同？** 是，且模式跨任务一致：Private 因子从图传播中获益远大于 Common（NC：Reddit-S Pt +14.8~17.1% vs C +1.3~3.6%；LP：sports Pt/Pv +6.9~8.5 MRR vs C +0.3~1.5）。")
    add("4. **差异是否体现在 node/edge 级冲突而非仅平均？** 是。NC 冲突率最高 73%（Reddit-S s42 C/V），LP RR 冲突 31–43%，mixed 模式占 61–75%。")
    add("5. **是否跨数据集、跨 seed、跨 NC/LP 稳定？** 是。5 NC + 2 LP 数据集 × 3 seeds 全部复现；唯一弱点是 Movies 的 ΔAcc 因子差异较小、cloth 的 C 因子接近随机（local MRR 0.047≈随机 0.037，其 Δ 噪声大，s43 出现 +0.74 离群点，已记录）。")
    add("")

    add("## 6. 说明与遗留")
    add("")
    add("- cloth s43 的 ΔMRR C=+0.737 为离群点：该数据集的 C 因子在 local probe 上接近随机水平（0.047–0.050），对图传播极度敏感；Pt/Pv 在所有 seed 上稳定。")
    add("- 公开 LP split 的反向边固有重叠（sports valid 8264 / test 6621 条）已记录在 summary 的 split_overlap 字段，与主协议同语义，非泄漏。")
    add("- LP 的 fused test MRR（sports ~0.273 / cloth ~0.207）低于 RPTA 基线表（GCN 0.340 等）——biaxis_p0 是纯语义因子模型（无任务化图编码器），P0 目的只是验证假设，不追 LP 性能。")

    (OUT / "P0_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"P0_REPORT.md written ({len(lines)} lines)")


if __name__ == "__main__":
    main()
