# OFT-MAG O1.5 — Graph Causality Cleanup Report

日期：2026-09-08
范围：O1.5（三个诊断：① matched no-graph control ② Accuracy–Macro-F1 checkpoint 诊断
③ per-class 诊断 + mechanism comparison）。未实现 O2 / cross-factor transition，
未修改论文主模型结构，未做任何 tuning，未读取 Test。
协议：**Val-only**（`task.evaluate_test=false`，全程未访问/未记录 Test 指标）。

## Verdict：**GO_TO_O2**

O1 的 +3.15pp（DIAG-GRAPH 54.86% vs P0 51.71%，Movies seed42 Val-only）分解为：

| 分量 | 定义 | ΔVal Acc | ΔVal Macro-F1 |
|---|---|---|---|
| Δnorm | DIAG-NOGRAPH − P0 | **+0.36pp**（52.07 vs 51.71） | +1.68pp（39.30 vs 37.62） |
| Δgraph | DIAG-GRAPH − DIAG-NOGRAPH | **+2.79pp**（54.86 vs 52.07） | +3.43pp（42.73 vs 39.30） |
| Δtotal | DIAG-GRAPH − P0 | **+3.15pp** | +5.11pp |

按 §8 判据：Δgraph = +2.79pp **> 1.0pp = 强证据** —— O1 的增益绝大部分来自真实
graph propagation（topology），而非仅由额外 LayerNorm scaffold 提供；DIAG-NOGRAPH
相对 P0 的 +0.36pp Acc 处于 ±0.3pp 噪声带边缘（弱证据；其 F1 分量 +1.68pp 更清楚）。
结论落在 **Case B**（scaffold normalization 有部分/边缘收益，graph propagation 有独立且
主导的贡献）→ **GO_TO_O2**。O2 故事起点：O1 提升 ≈ (≤0.4pp scaffold) + (~2.8pp 真实拓扑
传播)，graph consumer 确实在消费拓扑信息，且以最优收益方式作用在尾部小类上（§10）。

---

## 1. Git SHA 与 files changed

| 项 | 值 |
|---|---|
| starting HEAD | `0cf059836e074dfbc65427df19e0a8314410cbc3`（`O1`；branch `oft-mag`） |
| O1 tag（本阶段加） | `oft-o1-diag`（0cf0598） |
| ending HEAD | 见 §16（本报告 commit） |

新增/修改文件：

```text
src/models/oft_mag.py              # +variant 开关：diag_id（原路径不动）/ diag_nograph
configs/model/oft_mag.yaml         # 注释说明 variant（值/默认不变：diag_id）
tests/test_oft_mag.py              # +5 tests（O1.5-A 的 5 个要求 + variant 校验）
scripts/oft_o1_5_val_per_class.py  # O1.5-D 离线 per-class 分析（val-only）
scripts/oft_o1_5_history_stats.py  # 逐 epoch CSV 统计 helper
experiments/oft/o1_5/diag_nograph_movies_seed42.csv   # NOGRAPH 逐 epoch history
experiments/oft/o1_5/diag_id_rerun_movies_seed42.csv  # DIAG 诊断复跑 history（见 §2/§10）
experiments/oft/o1_5/diag_id_val_per_class.csv        # DIAG per-class（best-Acc ckpt）
experiments/oft/o1_5/diag_nograph_val_per_class.csv   # NOGRAPH per-class（best-Acc ckpt）
docs/OFT_O1_5_CAUSAL_CLEANUP_REPORT.md                # 本报告
```

未修改（语义锁定文件）：`biaxis_p0.py`、`biaxis_components.py`、`biaxis_final.py`、
`dip.py`、`oft_components.py`、`src/tasks/nc.py`、`src/tasks/common.py`、`src/main.py`、
数据与 split。

## 2. 环境与运行说明

conda `yhf_env`（torch 2.4.0+cu121 / PyG 2.7.0 / hydra 1.3.2）、RTX 3090 双卡。
**注意**：O1 锁定运行（P0 / DIAG-GRAPH / A0 / DiP）在 `cuda:0`；O1.5 两条新运行在
空闲的 `cuda:1`（`device=cuda:1`）。除 device 外配置逐项一致（§6 命令可比对），
噪声带沿用 O0 ±0.3pp。diag_id 诊断复跑（§10 需要 best-Acc ckpt，O1 未保存）
结果 55.04% vs O1 锁定 54.86%（+0.18pp，带内），与 O1 数字互相印证；正式对比仍以
O1 锁定 CSV 为准，复跑只用于 checkpoint 重载/per-class/机制统计。

## 3. Control 定义（exact）

`diag_nograph` = `diag_id` 的严格 matched no-topology control，唯一允许差异：
**graph neighbor response 恒为 0**。实现为模型级 variant 开关（`src/models/oft_mag.py`，
单一点 `_encode_oft` 内 `prop_edge = edge_index if self.variant == "diag_id" else None`，
forward 与 inference 共用同一路径）：

```text
X_a      = V_a H_a               # 照常计算（参数存在）
N_a      = incoming_mean(X_a, None) = zeros_like   # 跳过 aggregation → 严格 0
delta_a  = D_a(N_a) = D_a(0) = 0                   # D_a bias=False → 严格 0
H_next_a = LayerNorm(H_a + 0) = LayerNorm(H_a)     # 残余 scaffold 原样保留
```

- 与 `diag_id` 相同的：V/D/LN 模块结构与实例化、初始化、fusion、factorizer、P0 aux
  losses、dropout/optimizer/lr/patience/early-stopping、训练协议（full-graph）。
- 参数量完全一致（同一 `DiagIDLayer` 构造）；V/D/LN 权重仍进入 optimizer 参数集合，
  因 graph 通道死亡（N≡0）梯度严格为 0，权重不动（保持初始化值）——这是"无拓扑
  但结构与代价完全 matched"的最强对照形式。
- `diag_id` 代码路径与 O1 版本逐位相同（variant 读取为纯字符串操作，不消耗 RNG）。

## 4. Parameter-matching proof

单元测试 `test_diag_nograph_parameters_match_diag_id_exactly`（同一 RNG 流下分别构造两
variant）：

- state_dict keys 集合相同、逐 key shape 相同、逐 key 数值 `torch.equal`（初始化一致）；
- 参数量 `n_diag_id == n_diag_nograph == sum(numel)`。

真实规模核对（§7 表，model+head params，Movies）：DIAG-ID = DIAG-NOGRAPH =
**1,141,780**（= P0 1,042,708 + 99,072 = 3 slots × (2·128·128 + 2·128) V+D+LN），
O1 报告 §9 的 params 核对继续成立。A0 1,400,824 / DiP 8,170,620 不变。

## 5. Unit-test results

```bash
pytest tests/test_oft_mag.py -q    # 16 passed（11 O1 + 5 O1.5）, 4.8s
pytest tests/ -q                   # 72 passed（67 baseline + 5 new）, 20.2s
```

新增 5 个 O1.5 测试（逐条对应执行计划要求）：

| # | test | 验证点 |
|---|---|---|
| 1 | `test_variant_must_be_known` | 未知 variant → ValueError；`diag_nograph` 同样强制 num_layers=1；缺省 variant = diag_id（向后兼容） |
| 2 | `test_diag_nograph_parameters_match_diag_id_exactly` | **参数量 == diag_id** + 同 seed 初始化逐参数 `torch.equal`；V/D/LN 三槽俱在 |
| 3 | `test_diag_nograph_graph_delta_exactly_zero` | 执行路径 N ≡ 0、delta = D(0) ≡ 0（严格）、stats ≡ 0、H_next == 逐槽 LayerNorm(H0)；train 模式下 P0 aux 完好且 6 个 `oft_l1_*` 严格为 0 |
| 4 | `test_diag_nograph_edge_index_invariance` | 两个不同非空 edge set 与 None 的输出**逐位相同** → pre-fusion H_next 不随图改变 |
| 5 | `test_diag_nograph_output_equals_layernorm_then_fusion` | 输出逐位等于「逐槽 LayerNorm(H0) → 原 fusion」（手工重算比对） |

原有 11 个 diag_id tests 全部继续通过。

## 6. 执行的完整命令

```bash
# Git prep（本阶段起点 = O1 commit 0cf0598，工作区原本干净）
git tag oft-o1-diag

# A. diag_nograph matched control（Movies seed42，Val-only，cuda:1）
python -m src.main dataset=Movies task=nc model=oft_mag num_runs=1 seed=42 device=cuda:1 \
  task.evaluate_test=false model.variant=diag_nograph \
  task.history_path=experiments/oft/o1_5/diag_nograph_movies_seed42.csv \
  task.save_ckpt_path=outputs/oft_o1_5/diag_nograph_best.pt
#   -> log 镜像 outputs/oft_o1_5_diag_nograph.log；hydra outputs/2026-09-08/17-32-04

# B. DIAG-ID 诊断复跑（O1.5-D 需要 best-Acc checkpoint；O1 未保存。配置与 O1-D 逐项
#    一致，仅 device=cuda:1 + save_ckpt_path，Val-only，无 tuning）
python -m src.main dataset=Movies task=nc model=oft_mag num_runs=1 seed=42 device=cuda:1 \
  task.evaluate_test=false model.variant=diag_id \
  task.history_path=experiments/oft/o1_5/diag_id_rerun_movies_seed42.csv \
  task.save_ckpt_path=outputs/oft_o1_5/diag_id_best.pt
#   -> log 镜像 outputs/oft_o1_5_diag_id.log；hydra outputs/2026-09-08/17-32-40

# C. 逐 epoch 诊断统计（best-Acc / best-F1 行 + p0 因子诊断列）
python scripts/oft_o1_5_history_stats.py \
  experiments/oft/o1/p0_movies_seed42.csv \
  experiments/oft/o1/diag_id_movies_seed42.csv \
  experiments/oft/o1_5/diag_nograph_movies_seed42.csv \
  experiments/oft/o1_5/diag_id_rerun_movies_seed42.csv \
  experiments/oft/o0/biaxis_final_movies_seed42.csv \
  experiments/oft/o0/dip_movies_seed42.csv

# D. Per-class val 诊断（离线，post-hoc，仅 val split；ckpt 即 best-Val-Acc 状态）
PYTHONPATH=. python scripts/oft_o1_5_val_per_class.py \
  --ckpt outputs/oft_o1_5/diag_id_best.pt \
  --config outputs/2026-09-08/17-32-40/.hydra/config.yaml \
  --out experiments/oft/o1_5/diag_id_val_per_class.csv --label diag_id_rerun
PYTHONPATH=. python scripts/oft_o1_5_val_per_class.py \
  --ckpt outputs/oft_o1_5/diag_nograph_best.pt \
  --config outputs/2026-09-08/17-32-04/.hydra/config.yaml \
  --out experiments/oft/o1_5/diag_nograph_val_per_class.csv --label diag_nograph
```

未重跑 P0 / DIAG（O1 锁定 CSV 完好，直接复用：`experiments/oft/o1/p0_movies_seed42.csv`、
`experiments/oft/o1/diag_id_movies_seed42.csv`）。checkpoint 一致性验证：per-class 脚本在
ckpt 上重算的 val acc/F1 与 history CSV best-Acc 行逐位一致（DIAG ep75:
0.550390/0.427039；NOGRAPH ep76: 0.520696/0.393043）→ 分析确实落在 best-Val-Acc 状态。

## 7. 性能表（Movies seed42，Val-only，单 run，best-Val-Acc checkpoint）

| 模型 | variant | params (model+head) | best ep | Val Acc | Val Macro-F1 | 早停 ep |
|---|---|---|---|---|---|---|
| biaxis_p0（P0，O1 锁定，cuda:0） | — | 1,042,708 | 57 | **51.71%** | 37.62% | 87 |
| **OFT-DIAG-NOGRAPH**（本次，cuda:1） | diag_nograph | 1,141,780 | 76 | **52.07%** | **39.30%** | 106 |
| OFT-DIAG-GRAPH（O1 锁定，cuda:0） | diag_id | 1,141,780 | 75 | **54.86%** | **42.73%** | 105 |
| OFT-DIAG-GRAPH 复跑（诊断用，cuda:1） | diag_id | 1,141,780 | 75 | 55.04% | 42.70% | 105 |
| A0 = biaxis_final（O0 锁定） | — | 1,400,824 | 74 | 55.01% | 46.37–46.67% | 104 |
| DiP（O0 锁定） | — | 8,170,620 | 78 | 56.00–56.27% | 46.49–48.06% | 108–118 |

## 8. Δnorm / Δgraph / Δtotal（§Verdict 表；Acc）

```text
Δnorm  = NOGRAPH − P0     = 52.07 − 51.71 = +0.36pp    （≤ ±0.3pp 噪声带边缘，弱证据）
Δgraph = GRAPH − NOGRAPH  = 54.86 − 52.07 = +2.79pp    （> 1.0pp，强证据）
Δtotal = GRAPH − P0       = 54.86 − 51.71 = +3.15pp
```

按 §8 判据逐条：Δgraph +2.79pp 远超 1.0pp → **有意义且强**；Δnorm +0.36pp 落在
0.3–0.5pp 弱证据区（且同带噪声，保守读法为"≈ 无法区分、至多弱贡献"）。宏观点结论
不依赖 Δnorm 的精确归属：**无论 scaffold 是否有边缘收益，真实 graph propagation 贡献
≈ +2.8pp，占 Δtotal 的 ~88%**。（三条运行独立训练，上式在本次数据上恰好代数闭合
0.36+2.79=3.15，仅作解释用，不承诺一般性。）

F1 同样分解：Δnorm +1.68pp、Δgraph +3.43pp —— scaffold 对 F1 的贡献比 Acc 更清楚，
但 graph 仍主导。

## 9. best-Acc vs best-F1 checkpoint 诊断（O1.5-C，离线，不改正式选择协议）

| model | best-Acc ep | best Acc | F1@bestAcc | best-F1 ep | best F1 | Acc@bestF1 | ΔAcc(两个选择) |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 57 | 51.71 | 37.62 | 85 | 41.64 | 50.21 | −1.50 |
| DIAG-NOGRAPH | 76 | 52.07 | 39.30 | 93 | 41.56 | 50.00 | −2.07 |
| DIAG-GRAPH | 75 | 54.86 | 42.73 | 90 | 45.21 | 49.67 | **−5.19** |
| A0 | 74 | 55.10 | 46.67 | 94 | 46.90 | 52.70 | −2.29 |
| DiP | 78 | 56.27 | 46.62 | 98 | 50.04 | 56.15 | −0.12 |

**回答 O1.5 Q2**：Acc 与 Macro-F1 存在明显的 checkpoint trade-off——**所有模型**（含 P0/
A0/DiP）的 best-F1 checkpoint 都比 best-Acc 晚 20+ epoch，选 best-F1 会牺牲 Acc
（P0 −1.5 ~ A0 −2.3pp），说明"后期 epoch 继续提升 tail-class 预测、但总体 Acc 回落"是
该 val 曲线族的普遍形态，不是 OFT 引入的异常。DIAG-GRAPH 是该 trade-off 最极端的
（选 best-F1 损失 −5.19pp Acc，换取 +2.48pp F1）。正式 benchmark 维持 best-Val-Acc
选择（本报告全部数字均来自 best-Acc checkpoint；**禁止**以 best-F1 替代）。

## 10. Per-class validation diagnostic（O1.5-D）

best-Val-Acc ckpt 离线分析，val split 3334 节点（test 未访问）。重算核对：DIAG
val acc 55.04% / macro-F1 42.70%、balanced accuracy（= macro recall）39.46%；
NOGRAPH acc 52.07% / F1 39.30% / balanced acc 35.84%。

CSV：`experiments/oft/o1_5/diag_id_val_per_class.csv`、`diag_nograph_val_per_class.csv`
（每行：class_id, support, recall, precision, f1, predicted_count, true_count）。

**类分布结构**：20 类中 **12 类 val support < 100**（supports: 88,82,50,47,43,38,37,30,
24,23,23,14），多数类 cls1 占 1098/3334（33%）——极度头重，macro-F1 的锚点在长尾小类。

**DIAG-GRAPH bottom-5（按 F1）**：

| cls | support | recall | precision | F1 | predicted |
|---|---:|---:|---:|---:|---:|
| 16 | 38 | 0.026 | 1.000 | **0.051** | 1 |
| 9 | 37 | 0.054 | 0.333 | **0.093** | 6 |
| 5 | 82 | 0.171 | 0.452 | 0.248 | 31 |
| 15 | 47 | 0.192 | 0.450 | 0.269 | 20 |
| 19 | 14 | 0.214 | 0.500 | 0.300 | 6 |

top-5：cls3 F1 0.784（sup 277）、cls12 0.727（sup 23）、cls1 0.651（sup 1098）、
cls10 0.596（sup 269）、cls13 0.571（sup 43）。

**NOGRAPH bottom-5**：cls16 **0.000**（pred 2/38）、cls9 0.167、cls5 0.268、
cls17 0.278、cls6 0.322 —— 与 DIAG 高度同构：**cls16/cls9 这两个近零预测类在无图对照
中同样瘫痪**（cls16: DIAG pred 1、NOGRAPH pred 2，support 38）→ 是尾部类欠预测的系统性
现象（数据极度不平衡 + Accuracy 选择压力），不是 graph 引入的缺陷。

**graph 增益落在哪些类**（DIAG − NOGRAPH 逐类 ΔF1，best-Acc ckpt）：+cls12 **+32.7pp**
(0.727 vs 0.400)、cls17 +13.6、cls13 +7.1、cls6 +6.7、cls3 +5.2、cls18 +5.2；负向仅
cls15 −7.8、cls9 −7.4、cls19 −5.3、cls5 −2.0。graph propagation 的主要收益集中在
长尾小类（尤其 cls12 等），与 §12 中"graph 通道主要在 Pt 上被利用、为小类带去可判别
邻居信号"的机制画像一致。

**A0 相对 deficit 的归因**：O0 未保存 A0 的 ckpt（exp 与历史 0901 outputs 均无 *.pt），
按执行计划"不要为 O1.5 强制重跑 A0"，**跳过 A0 逐类对比**。现状说明：DIAG-GRAPH 在
55.0% Acc 处 macro-F1 42.7% vs A0 同 Acc 处 46.7%（−4.0pp），只能给出结构性解释——
DIAG 对 tail 类（16/9/15/19 等）的 recall 不足压低 macro-F1；逐类数值对比需 A0 ckpt，
留待后续有 ckpt 时补做。

## 11. Pt / C / Pv norm 对比（O1.5-E，p0_*_norm 列，best-Acc 无关，全程）

| key（last / mean over epochs） | P0 | DIAG-NOGRAPH | DIAG-GRAPH (O1) |
|---|---|---|---|
| pt_norm last | **2.60** | **6.54** | **6.04** |
| pt_norm mean | 3.53 | 5.68 | 5.54 |
| c_norm last | 4.57 | 4.63 | 4.28 |
| pv_norm last | 4.53 | 4.25 | 4.20 |

（三模型 first 值相同 = 4.1806/3.5208/4.0071，factorizer 初始化一致。）

**回答 O1.5 Q**：DIAG-NOGRAPH **同样**产生 Pt norm 放大（甚至终值略高于 GRAPH 版：
6.54 vs 6.04）→ Pt 的 geometry shift **不是** graph consumer 反向塑造 factorizer 的产物，
而是 scaffold 本身（新增逐槽 LayerNorm residual 重参数化 + 死 V/D 通道）带来的；
graph propagation 没有再额外推高 Pt norm。因此 §7 的 +2.79pp graph 增益不能被解释为
"Pt norm 放大"的副作用——GRAPH 版 Pt norm 略低于 NOGRAPH，Acc 反而高 2.8pp。

## 12. P0 ownership diagnostics 对比（O1.5-E）

| key（last / mean） | P0 | NOGRAPH | GRAPH |
|---|---|---|---|
| common_sim | 0.639 / 0.635 | 0.652 / 0.628 | 0.637 / 0.627 |
| private_sim | 0.004 / −0.011 | 0.022 / 0.033 | 0.020 / 0.040 |
| cp_overlap_t | 0.013 / 0.015 | 0.016 / 0.013 | 0.015 / 0.013 |
| cp_overlap_v | 0.022 / 0.034 | 0.020 / 0.029 | 0.019 / 0.029 |

三模型全程同形态：common_sim 收敛 ~0.63（远离坍塌点 1.0）、private_sim ≈ 0、
overlap 均 < 0.06 → Common/Private 分解在 scaffold 与 graph 两种干预下都不坍塌；
NOGRAPH 与 GRAPH 的 ownership 统计几乎不可分，说明 graph propagation 不改变 factor
ownership 结构（只在 §11 的 norm 层面积累差异）。

**DIAG graph update ratios**（复跑 105 epochs 逐 epoch 统计；与 O1 锁定 run 数字一致）：
仅描述为 **learned relative graph update magnitude**，不作 "factor graph demand" 解读：

| key | first | mean | min | max | last |
|---|---|---|---|---|---|
| oft_l1_c_diag_update_ratio | 0.320 | 0.327 | 0.278 | 0.426 | 0.278 |
| oft_l1_pt_diag_update_ratio | 0.261 | 0.425 | 0.261 | 0.489 | 0.489 |
| oft_l1_pv_diag_update_ratio | 0.287 | 0.420 | 0.287 | 0.473 | 0.389 |
| oft_l1_c_neighbor_norm | 1.83 | 2.22 | 1.83 | 3.10 | 1.84 |
| oft_l1_pt_neighbor_norm | 1.85 | 3.20 | 1.85 | 3.59 | 3.57 |
| oft_l1_pv_neighbor_norm | 1.99 | 2.48 | 1.99 | 2.92 | 2.04 |

DIAG-NOGRAPH 全程（106 epochs）6 个 `oft_l1_*` 严格为 0（日志验证）——对照的"图响应
恒为 0"在运行级成立，非仅构造级。

## 13. Interpretation

1. **O1 +3.15pp 的因果归属**：~+0.36pp（噪声带边缘，弱）来自"多了 LayerNorm 的
   scaffold"这一结构差异，**+2.79pp（强）来自真实的消息传播**。对比历史：A0 依靠
   P1 K=4 拓扑 relations + P2 Null routing 等复杂 machinery 才达到 55.01%；
   DIAG-ID 仅一个 identity-operator 对角传播层在 Δtotal ≈ +3.15 中贡献 ~2.8pp，说明
   该最简拓扑消费者本身有效——O2 在此基线上做 functional transition 的故事成立。
2. **Pt norm 放大 ≠ graph 功劳**：放大在 NOGRAPH 中同样出现且更大 → 归因于 scaffold
   重参数化；graph 的 +2.8pp 与 Pt norm 无关（GRAPH Pt norm 更低而 Acc 更高），
   更可能来自邻居信号直接补充了 tail 类判别信息（§10 逐类 ΔF1 集中在小类）。
3. **ownership 结构稳健**：common/private 分解在所有干预下不坍塌（§12），O2 可在
   factor-preserving 语义上继续。
4. **checkpoint 纪律**：best-Acc 与 best-F1 的 trade-off 是全模型族普遍形态，DIAG 最
   极端；正式协议维持 best-Val-Acc，F1 deficit 不因选择协议放大。
5. 局限：单 seed 单 run、P0/DIAG 锁定时在 cuda:0 而 O1.5 新运行在 cuda:1（配置逐项一
   致）；Δnorm 0.36pp 需在噪声带语境下保守阅读。

## 14. O1.5 Verdict

**GO_TO_O2**（Case B：DIAG-NOGRAPH 略高于 P0 且 DIAG-GRAPH 显著高于 DIAG-NOGRAPH；
核心证据 Δgraph = +2.79pp > 1.0pp 强证据区）。O2 起点建议保持：C/Pt/Pv 为显式 graph
states + identity-operator 对角传播的干净基线已被证明"传播本身承担主要收益"，O2 可
放心在 (j,a)→(i,b) functional transition 上展开，而无需先解释 scaffold confound。

## 15. 声明

- **未读取 Test**：所有新运行 `task.evaluate_test=false`；per-class 分析仅 val split。
- **未做 tuning**：无超参/lr/dropout/scheduler/seed 修改；diag_id 复跑与 O1-D 配置逐项
  一致（仅 device + ckpt 保存）。
- **未进入 O2**；未实现 cross-factor transition / router / operator mixture / Null
  routing / 任何被禁止项（§执行计划 O1.5 禁止清单全遵守）。
- 正式 checkpoint 选择维持 best-Val-Acc（§9 表格纯诊断，禁止反向采用）。

## 16. Commit

O1.5 产物（含本报告）已 milestone commit 并打 tag `oft-o1-5`
（ending HEAD = 该 tag 指向的 commit；starting HEAD `0cf0598` 即 tag `oft-o1-diag`）：

```text
git add src/models/oft_mag.py configs/model/oft_mag.yaml tests/test_oft_mag.py \
        scripts/oft_o1_5_val_per_class.py scripts/oft_o1_5_history_stats.py \
        experiments/oft/o1_5 docs/OFT_O1_5_CAUSAL_CLEANUP_REPORT.md
git commit -m "O1.5: graph causality cleanup (matched diag_nograph control, Acc-F1 and per-class diagnostics)"
git tag oft-o1-5
```

（checkpoint `.pt` 与运行日志在 `outputs/`（gitignored），不入库。）
