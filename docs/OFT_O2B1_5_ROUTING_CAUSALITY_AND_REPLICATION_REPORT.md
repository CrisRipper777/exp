# OFT-MAG O2-B1.5 — Routing Granularity Causality Audit + 3-Seed Replication

Movies / NC / Val-only / best-Val-Acc / `cuda:1` / seed42 locked + seed43/44 replication。

本阶段是 **causality audit + replication**，不是 architecture expansion。全程：

- **no Test**：`task.evaluate_test=false`，10 个新 run 的日志均打印
  `(evaluate_test=false: no test access / metrics recorded)`；Part A 审计脚本被单测
  静态检查为不含 `test_idx` / `.test_` / `evaluate_test`（`test_audit_script_never_references_test_split`）。
- **no tuning**：所有 10 个新 run 的 resolved config 与对应 seed42 config 逐键比对，
  **除 `seed`（及由 seed 派生的 split/输出路径）外 0 处差异**（§22 合规检查，见 §29）。
- **no O2-B2 implementation**：本报告只给 recommendation，未实现任何 Operator Bank /
  expert / FiLM / source-aware router / channel group / edge router / attention /
  multi-scale / 第二层 / global context / PPR / 新 loss / entropy 正则 / 温度或
  router hidden 或 LR/dropout tuning / class-balanced loss。

---

## 1. Git starting / ending SHA（条目 1）

| 项 | 值 |
|---|---|
| branch | `oft-mag` |
| starting HEAD | `93015ebede9e9424146873bad138e35b94a217a7`（`O2-B1: record final commit SHA in conditional-routing report`，文档补记 commit；其父为 O2-B1 代码 commit `36fec95780638814778ffccdea59417fa8a2f32e`） |
| ancestor 检查 | `git merge-base --is-ancestor oft-o2b1-conditional-routing HEAD` → **通过**（未 reset） |
| 起始工作区 | clean（`git status --short` 为空） |
| ending SHA | `32455971259ea04a050c8203291bcb0db4069211`（本阶段主 commit `O2-B1.5: audit routing granularity and replicate across seeds`） |
| tag | `oft-o2b1-5-routing-audit`（打在本阶段主 commit 上） |

## 2. Files changed（条目 2）

新增（代码 / 测试）：

| 文件 | 作用 |
|---|---|
| `scripts/oft_routing_interventions.py` | 8 种 frozen gate 干预的纯张量策略 + `patched_layer` / `patched_router` 上下文管理器 |
| `scripts/oft_o2b1_5_routing_causal_audit.py` | Part A：frozen causal audit 主脚本 |
| `scripts/oft_o2b1_5_seed_summary.py` | Part B：3-seed 汇总 + paired delta + router stability |
| `tests/test_oft_routing_audit.py` | 15 个新单测（覆盖 §29 全部 14 条要求） |

新增（实验产物，`experiments/oft/o2b1_5/`）：

```
routing_causal_summary.csv              # Part A 主表（8 行）
routing_shuffle_results.csv             # Part A 20 个 shuffle seed 逐 seed 表
routing_granularity_stats.json          # Part A granularity stats + replay 证明
routing_original_val_per_class.csv      # ORIGINAL per-class（val）
<V>_movies_seed<S>.csv                  # 10 个 run 的逐 epoch history（V∈5 variants, S∈{43,44}）
<V>_val_per_class_seed<S>.csv           # 10 个 run 的 per-class（val）
conditional_cross_stats_seed<S>.json    # 4 个 conditional run 的 router stats（43/44）
conditional_cross_router_nodes_val_seed<S>.csv  # 4 个 conditional run 的逐 val node 路由概率
conditional_cross_dup_*.{json,csv}      # 同上（dup）
three_seed_model_summary.csv            # 3-seed 每模型指标 + mean/std
paired_seed_summary.csv                 # 5 个 paired contrast × 3 seeds + mean/std/pos
three_seed_router_stability.json        # router 跨 seed 稳定性全部量
three_seed_per_class_f1.csv             # 每类 F1（5 模型 × 3 seeds）
```

文档：`docs/OFT_O2B1_5_ROUTING_CAUSALITY_AND_REPLICATION_REPORT.md`（本文件）。
未修改任何训练代码：`src/models/oft_mag.py`、`src/models/oft_components.py`、
`src/tasks/nc.py`、`configs/model/oft_mag.yaml` 均 **0 改动**（`git status` 可验证）。

## 3. O2-B1 locked facts（条目 3）

Movies / seed42 / Val-only / best-Val-Acc（**未重跑，直接复用锁定 artifacts**）：

| model | Acc | Macro-F1 | BalAcc |
|---|---:|---:|---:|
| DIAG-ID | 55.0390 | 42.7039 | 39.4594 |
| STATIC-DUP | 55.2490 | 44.4705 | 42.4651 |
| STATIC-CROSS | 54.5591 | 44.0328 | 42.8308 |
| CONDITIONAL-DUP | 55.1290 | 43.4314 | 39.1127 |
| CONDITIONAL-CROSS | 55.1290 | 45.7487 | 42.7895 |

锁定关键量：CC−CD = Acc 0.00 / F1 +2.32 / BalAcc +3.68 pp；CC−SC = +0.57 / +1.72 / −0.04 pp；
CC−STATIC-DUP = −0.12 / +1.28 / +0.32 pp。O2-B1 的结论边界是
“selective transfer 有正信号”，**不是**“node-specific dynamic routing 已被证明”。

## 4. Frozen audit exact implementation（条目 4）

- 唯一被干预的对象是 **`ConditionalCrossLayer.router_transfer` 的输出 `p_transfer_i^{ab}`**。
  干预方式是把该实例方法临时替换为“返回固定 gate 的闭包”
  （`scripts/oft_routing_interventions.py::patched_layer`），`finally` 恢复原方法。
- 因此 `propagate()` 内部其余全部走**模型自己的代码**：`neighbor_states` → `cross_messages`
  （payload）→ `delta_diag` → `combine_cross` → LayerNorm → fusion → head。
  **不存在第二套模型数学**；无任何参数被修改（实测 `parameter drift = 0.000e+00`）。
- 结构上不可能影响非 val node：cross 更新是**逐节点**的
  `delta_cross_i^b = (1/2) Σ_{a≠b} p_i^{ab} T_{a→b}(N_i^a)`，val node 的 gate 只进入自己的更新。
- 常数全部由 **TRAIN split** 的节点概率求均值（`train_mask` 来自 `data.train_idx`），
  **不看任何 label**；NODE_SHUFFLE 只在 val node 内部做置换。无 temperature / threshold /
  手工选择 / 按 val 性能优化。
- 单测进一步固定了“干预不改变 payload / diag / 参数”和“策略不改写输入映射”。

## 5. ORIGINAL replay proof（条目 5）

冻结路径必须逐位复现 O2-B1 锁定值，否则 `HOLD_IMPLEMENTATION`。实测：

| 指标 | measured | locked | abs diff |
|---|---:|---:|---:|
| Val Acc | 0.5512897372245789 | 0.5512897372245789 | **0.000e+00** |
| Val Macro-F1 | 0.45748732111520135 | 0.45748732111520135 | **0.000e+00** |
| Balanced Acc | 0.42789518276301874 | 0.42789518276301874 | **0.000e+00** |

per-class CSV 对 `experiments/oft/o2b1/conditional_cross_val_per_class.csv` 的
**逐类 recall/precision/f1/support/predicted_count/true_count 最大绝对差 = 0.000e+00**。
→ 不触发 HOLD，counterfactual 继续。

## 6–12. Part A 主结果（条目 6–12）

同一 checkpoint、frozen inference，只改 `p_transfer`（Δ 相对 ORIGINAL，单位 pp）：

| intervention | Acc | Macro-F1 | BalAcc | ΔAcc | ΔF1 | ΔBalAcc |
|---|---:|---:|---:|---:|---:|---:|
| ORIGINAL | 0.551290 | 0.457487 | 0.427895 | 0.000 | 0.000 | 0.000 |
| PAIR_CONSTANT | 0.550990 | 0.456213 | 0.425904 | −0.030 | −0.127 | −0.199 |
| TARGET_NODE | 0.551890 | 0.457851 | 0.428212 | +0.060 | +0.036 | +0.032 |
| TARGET_CONSTANT | 0.551290 | 0.455972 | 0.425992 | 0.000 | −0.152 | −0.190 |
| SOURCE_SWAP | 0.552190 | 0.457994 | 0.428493 | +0.090 | +0.051 | +0.060 |
| NODE_SHUFFLE (20 seeds mean) | 0.551095 | 0.455796 | 0.425566 | −0.019 | −0.169 | −0.233 |
| ALL_ON（frozen diagnostic） | 0.547690 | 0.463491 | 0.445053 | −0.360 | +0.600 | +1.716 |
| ALL_NULL（frozen diagnostic） | 0.549490 | 0.417581 | 0.383103 | −0.180 | −3.991 | −4.479 |

- **条目 7 PAIR_CONSTANT**：六个 train-mean 常数（`c_to_pt` 0.9400、`c_to_pv` 0.1280、
  `pt_to_c` 0.9816、`pt_to_pv` 0.2493、`pv_to_c` 0.9966、`pv_to_pt` 0.9965，由 TRAIN 节点求均值）
  把 node 维度完全抹平后，F1 仅 −0.13pp、BalAcc 仅 −0.20pp。
- **条目 8 TARGET_NODE**：`p_i^b = 0.5(p_i^{a1→b}+p_i^{a2→b})` 赋给两个 source（保留 node 变化、
  抹掉 source 身份）→ 与 ORIGINAL 在 0.06pp 内，**甚至略高**。
- **条目 9 TARGET_CONSTANT**：三个 target 常数（train：C 0.9891、Pt 0.9683、Pv 0.1886）
  → Acc 完全相同，F1 −0.15pp、BalAcc −0.19pp。
- **条目 10 SOURCE_SWAP**：逐节点交换每个 target 的两个 source gate（保留 node 变化 / target 身份 /
  分布）→ 与 ORIGINAL 在 0.09pp 内，**也是略高**。即 source 身份对决策几乎无因果贡献。
- **条目 11 NODE_SHUFFLE 20-seed**：acc 0.551095 ± 0.000332（min 0.550690 / max 0.551590）、
  f1 0.455796 ± 0.000657、balacc 0.425566 ± 0.000697（均为总体 std，20 个固定 seed 1000–1019）。
  逐 seed 表见 `routing_shuffle_results.csv`。相对 ORIGINAL 的逐 seed 位移：
  **Acc −0.060 … +0.030pp（11 低 / 5 平 / 4 高）、F1 −0.308 … −0.086pp（20/20 全部更低）、
  BalAcc −0.417 … −0.161pp（20/20 全部更低）**。即“打乱 val 节点的 gate 对齐”只会让 F1/BalAcc
  一致地变差一点点（≤0.42pp），Acc 则是双向抖动——**破坏 node 对齐的代价与量测噪声同量级**。
- **条目 12 ALL_ON / ALL_NULL（frozen diagnostic，不是正式 baseline）**：ALL_ON（全 transfer）
  把 BalAcc 抬 +1.72pp、F1 +0.60pp 但 Acc −0.36pp；ALL_NULL（全 null）BalAcc −4.48pp、
  F1 −3.99pp。说明**这个 checkpoint 的 cross 通路确实在做事**（全部关掉会明显变差），
  但**开多大 / 给谁开**的节点级细节几乎不影响结果。ALL_NULL ≠ DIAG-ID（DIAG-ID 是另一套
  只保留 diag 更新的结构），二者不可等同。

## 13. same-target Pearson / Spearman（条目 13）

同一 target 的两个 incoming pair 在 val 节点上的相关性（ORIGINAL 路由概率）：

| target | pair A | pair B | Pearson | Spearman |
|---|---|---:|---:|---:|
| C | `pt→c` | `pv→c` | **+0.9993** | +0.9992 |
| Pt | `c→pt` | `pv→pt` | **+0.9803** | +0.9980 |
| Pv | `c→pv` | `pt→pv` | **+0.9902** | +0.9949 |

两个 source 的 gate 在节点维度上几乎共线 → 路由实际上编码的是
“**这个节点对目标 slot b 的接受度**”，而不是“某个 source 该不该投递”。

## 14. source difference statistics（条目 14）

| target | pair mean (val) | pair std (val) | target gate (val) | meanᵢ\|p^{a1→b}−p^{a2→b}\| |
|---|---|---:|---:|---:|
| C | `pt→c` 0.9815 / `pv→c` 0.9966 | 0.0154 / 0.0029 | 0.9890 | 0.0151 |
| Pt | `c→pt` 0.9404 / `pv→pt` 0.9965 | 0.0473 / 0.0035 | 0.9684 | 0.0562 |
| Pv | `c→pv` 0.1262 / `pt→pv` 0.2467 | 0.0589 / 0.0963 | 0.1864 | 0.1205 |

- **target C / Pt 的 gate ≈ 0.97–0.99**（几乎全量 transfer）；**target Pv 的 gate ≈ 0.186**
  （主要走 null）。source 间平均绝对差最大只有 0.12（Pv），C 只有 0.015。
- node 维度 std 也很小（最大 0.096，`pt→pv`），即**节点级波动远小于 target 间差异**。

## 15. Part A verdict（条目 15）

按 §18 的阈值逐条判定（同一 checkpoint，无重训噪声）：

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| NODE_SPECIFIC_SUPPORTED | ORIGINAL−PAIR_CONSTANT F1/BalAcc ≥ +0.5pp 且 shuffle 至少低 0.5pp | +0.13 / +0.20pp；shuffle −0.17 / −0.23pp | ✗ |
| **NODE_SPECIFIC_WEAK** | ORIGINAL vs PAIR_CONSTANT < 0.5pp 且 NODE_SHUFFLE ≈ ORIGINAL | 0.13/0.20pp；shuffle 差 0.17/0.23pp | **✓** |
| **TARGET_RECEPTIVITY_SUFFICIENT** | TARGET_NODE ≈ ORIGINAL（\|ΔAcc\|<0.2、\|ΔF1\|<0.5、\|ΔBalAcc\|<0.5pp）且 SOURCE_SWAP ≈ ORIGINAL | +0.06/+0.04/+0.03pp；SOURCE_SWAP +0.09/+0.05/+0.06pp | **✓** |
| **STATIC_TARGET_POLICY_SUFFICIENT** | TARGET_CONSTANT ≈ ORIGINAL | 0.00/−0.15/−0.19pp | **✓** |
| PAIR_POLICY_SUFFICIENT | PAIR_CONSTANT ≈ ORIGINAL 且 TARGET_CONSTANT 明显下降 | 两者都 ≈ ORIGINAL，无差分 | ✗ |
| SOURCE_IDENTITY_REQUIRED | TARGET_NODE 或 SOURCE_SWAP 明显下降 | 二者都不降 | ✗ |

**Part A 结论（组合）**：

> **NODE_SPECIFIC_WEAK + TARGET_RECEPTIVITY_SUFFICIENT + STATIC_TARGET_POLICY_SUFFICIENT**

即：在当前 checkpoint 上，**真正必要的粒度是“static target-level ownership receptivity”**
（三个常数就够）；node 级动态、source 身份、甚至 pair 级身份都**没有可测量的因果贡献**
（所有非诊断干预的位移 ≤ 0.24pp，20 个 shuffle seed 的 std 仅 0.07pp）。
**唯一在结构上有意义的自由度是 target**：C/Pt 几乎全量接收 cross（gate≈0.97–0.99），
Pv 主要拒绝（gate≈0.19）。

## 16. seed42 / seed43 / seed44 每模型指标（条目 16）

Val-only / best-Val-Acc（Balanced Acc 与 Macro-F1 由 per-class CSV 统一口径导出）：

| model | seed | Acc | Macro-F1 | BalAcc | best epoch | epochs |
|---|---:|---:|---:|---:|---:|---:|
| DIAG-ID | 42 | 55.0390 | 42.7039 | 39.4594 | 75 | 105 |
| DIAG-ID | 43 | 54.8890 | 43.2982 | 40.5944 | 70 | 100 |
| DIAG-ID | 44 | 55.7289 | 47.8102 | 47.3056 | 99 | 129 |
| STATIC-DUP | 42 | 55.2490 | 44.4705 | 42.4651 | 70 | 100 |
| STATIC-DUP | 43 | 55.1590 | 40.2618 | 38.5987 | 54 | 84 |
| STATIC-DUP | 44 | 55.3089 | 44.5632 | 43.2997 | 60 | 90 |
| STATIC-CROSS | 42 | 54.5591 | 44.0328 | 42.8308 | 71 | 101 |
| STATIC-CROSS | 43 | 55.6089 | 45.6543 | 44.4263 | 77 | 107 |
| STATIC-CROSS | 44 | 55.5489 | 43.6370 | 42.7205 | 73 | 103 |
| COND-DUP | 42 | 55.1290 | 43.4314 | 39.1127 | 84 | 114 |
| COND-DUP | 43 | 55.0390 | 40.9494 | 38.0902 | 68 | 98 |
| COND-DUP | 44 | 55.0990 | 44.9363 | 44.0276 | 78 | 108 |
| COND-CROSS | 42 | 55.1290 | 45.7487 | 42.7895 | 71 | 101 |
| COND-CROSS | 43 | 54.9490 | 42.5514 | 39.4597 | 78 | 108 |
| COND-CROSS | 44 | 55.4289 | 42.9355 | 40.7162 | 64 | 94 |

一致性检查：per-class 导出的 Acc 与各 run history 的 best `val_acc` 差均 < 1e-4 pp
（`acc_ckpt_vs_history_pp` 列），10 个 run 全部 exit=0，**0 个 NaN/Inf 单元**。

## 17. 3-seed mean ± std（条目 17）

| model | Acc mean ± std | Macro-F1 mean ± std | BalAcc mean ± std |
|---|---:|---:|---:|
| DIAG-ID | 55.219 ± **0.366** | 44.604 ± **2.280** | 42.453 ± **3.462** |
| STATIC-DUP | 55.239 ± 0.062 | 43.099 ± 2.006 | 41.455 ± 2.048 |
| STATIC-CROSS | 55.239 ± 0.481 | 44.441 ± 0.873 | **43.326 ± 0.779** |
| COND-DUP | 55.089 ± 0.037 | 43.106 ± 1.644 | 40.410 ± 2.592 |
| COND-CROSS | 55.169 ± 0.198 | 43.745 ± 1.425 | 40.989 ± 1.373 |

- Acc：五个模型全部落在 **55.09–55.24** 的 0.15pp 带内，**没有任何模型在 Acc 上分开**。
- Macro-F1 的跨 seed std（0.87–2.28pp）**大于**模型间 mean 差（最大 1.51pp）——
  seed 噪声与模型差异同量级甚至更大。
- DIAG 的 F1 mean 最高（44.60）但 std 也最大（2.28），且完全由 **seed44 的 47.81 离群**撑起
  （seed42/43 为 42.70 / 43.30）。
- STATIC-CROSS 是 3-seed 下**最稳**的：BalAcc mean 最高且 std 最小（43.33 ± 0.78），
  F1 std 最小（0.87）。

## 18–19. paired seed deltas 与 positive-seed count（条目 18–19）

| contrast | 指标 | s42 | s43 | s44 | mean Δ | std Δ | positive /3 |
|---|---|---:|---:|---:|---:|---:|---:|
| **A. CC−CD** | ΔAcc | +0.00 | −0.09 | +0.33 | +0.08 | 0.21 | 1/3 |
| | ΔF1 | **+2.32** | **+1.60** | **−2.00** | +0.64 | 2.19 | 2/3 |
| | ΔBalAcc | **+3.68** | **+1.37** | **−3.31** | +0.58 | 3.51 | 2/3 |
| **B. CC−STATIC-DUP** | ΔAcc | −0.12 | −0.21 | +0.12 | −0.07 | 0.14 | 1/3 |
| | ΔF1 | +1.28 | +2.29 | −1.63 | +0.65 | 1.96 | 2/3 |
| | ΔBalAcc | +0.32 | +0.86 | −2.58 | −0.47 | 1.49 | 2/3 |
| **C. CC−STATIC-CROSS** | ΔAcc | +0.57 | −0.66 | −0.12 | −0.07 | 0.51 | 1/3 |
| | ΔF1 | +1.72 | −3.10 | −0.70 | −0.70 | 1.99 | 1/3 |
| | ΔBalAcc | −0.04 | −4.97 | −2.00 | **−2.34** | 2.03 | **0/3** |
| **D. SC−STATIC-DUP** | ΔAcc | −0.69 | +0.45 | +0.24 | +0.00 | 0.49 | 2/3 |
| | ΔF1 | −0.44 | +5.39 | −0.93 | +1.34 | 2.86 | 1/3 |
| | ΔBalAcc | +0.37 | +5.83 | −0.58 | +1.87 | 2.85 | 2/3 |
| **E. CC−DIAG** | ΔAcc | +0.09 | +0.06 | −0.30 | −0.05 | 0.18 | 2/3 |
| | ΔF1 | +3.04 | −0.75 | −4.87 | −0.86 | 3.25 | 1/3 |
| | ΔBalAcc | +3.33 | −1.13 | −6.59 | −1.46 | 4.12 | 1/3 |

（**不做显著性检验包装**：3 个 seed 太少，只报告 paired consistency。）

**读法**：

- A（matched control）：seed42 的 +2.32/+3.68pp 在 seed43 只留下 +1.60/+1.37pp，
  在 seed44 **反转为 −2.00/−3.31pp**。mean +0.64/+0.58pp 但 paired std 2.19/3.51pp。
- B（相对 STATIC-DUP）：同样 seed42/43 正、seed44 负，BalAcc mean 甚至为负（−0.47pp）。
- C（相对 STATIC-CROSS）：**BalAcc 0/3 为正**，mean −2.34pp —— 条件路由相对静态 cross
  在 3 seed 平均上**没有优势**。
- E（相对 DIAG）：F1/BalAcc 各只有 1/3 为正，mean 为负。
- 每一个 contrast 都**至少有一个 seed 符号反转**；所有 mean Δ 的绝对值（≤0.86pp）
  **小于**对应的 paired std（1.4–4.1pp）。

## 20–22. Router stability across seeds（条目 20–22）

### 20. 六个 pair mean（val，逐 seed）

| variant | seed | `c→pt` | `c→pv` | `pt→c` | `pt→pv` | `pv→c` | `pv→pt` |
|---|---:|---:|---:|---:|---:|---:|---:|
| COND-CROSS | 42 | 0.9404 | 0.1262 | 0.9815 | 0.2467 | 0.9966 | 0.9965 |
| COND-CROSS | 43 | 0.9980 | 0.3722 | 0.8314 | 0.1925 | 0.9832 | 0.9994 |
| COND-CROSS | 44 | 0.9916 | 0.4172 | 0.8827 | 0.4308 | 0.9390 | 0.9964 |
| COND-DUP | 42 | 0.6084 | 0.7211 | 0.3234 | 0.6772 | 0.2553 | 0.4297 |
| COND-DUP | 43 | 0.2098 | 0.8845 | 0.6539 | 0.9084 | 0.4630 | 0.1389 |
| COND-DUP | 44 | 0.5092 | 0.7681 | 0.3713 | 0.7235 | 0.3329 | 0.4005 |

**target 层面（条目 20 的判读）**：

| variant | seed | target C gate | target Pt gate | target Pv gate |
|---|---:|---:|---:|---:|
| COND-CROSS | 42 | 0.9890 | 0.9684 | **0.1864** |
| COND-CROSS | 43 | 0.9073 | 0.9987 | **0.2824** |
| COND-CROSS | 44 | 0.9109 | 0.9940 | **0.4240** |
| COND-DUP | 42 | 0.2893 | 0.5190 | 0.6992 |
| COND-DUP | 43 | 0.5585 | **0.1744** | 0.8964 |
| COND-DUP | 44 | 0.3521 | 0.4549 | 0.7458 |

- **① “每个 seed 都出现 target Pv 接收 cross 较少”**：COND-CROSS **是**（0.186 / 0.282 / 0.424，
  三个 seed 都远低于同 seed 的 C 与 Pt）；COND-DUP **不是**（Pv 反而是三个 target 中最高）。
- **② “target C / Pt 的 transfer 较高”**：COND-CROSS **是**（C 0.91–0.99，Pt 0.97–1.00）；
  COND-DUP 不稳定（seed43 的 Pt 只有 0.174）。
- **③ same-target correlation 跨 seed 是否都高**：**是**，且两个 variant 都是
  （Pearson 0.980–0.9999，Spearman 0.985–0.9999；逐 seed 见
  `three_seed_router_stability.json::router.same_target_corr`）。**两个 source 的 gate 共线是
  跨 seed 稳定的结构性事实**。
- **④ pair 排序是否稳定**：COND-CROSS 部分稳定 —— 两端稳定、中间交换：
  - 最高两名：seed42 {`pv→c`,`pv→pt`}，seed43/44 {`pv→pt`,`c→pt`}；
  - **最低两名三个 seed 恒为 {`c→pv`, `pt→pv`}**（即“Pv 少收 cross”）；
  - `pt→c` 与 `c→pt` 在中间交换位置。
  COND-DUP 的排序**不稳定**（seed43 整体顺序与 42/44 明显不同）。
- **22. inter-seed node-level p 相关**（val node IDs 取交集；每 seed val=3334，
  两两交集 634–690 个节点）：

| variant | pair | n_common | Pearson (all pairs) | Spearman (all pairs) | 逐 pair Pearson 范围 |
|---|---|---:|---:|---:|---|
| COND-CROSS | 42 vs 43 | 690 | +0.937 | +0.727 | +0.343 (`pt→pv`) … +0.730 (`pv→pt`) |
| COND-CROSS | 42 vs 44 | 634 | +0.947 | +0.749 | +0.337 (`pt→pv`) … +0.622 (`pv→pt`) |
| COND-CROSS | 43 vs 44 | 687 | +0.948 | +0.955 | +0.264 (`pt→c`) … +0.672 (`c→pt`) |
| COND-DUP | 42 vs 43 | 690 | +0.364 | +0.328 | +0.401 (`pv→pt`) … +0.669 (`pv→c`) |
| COND-DUP | 42 vs 44 | 634 | +0.588 | +0.602 | +0.040 (`c→pv`) … +0.352 (`pv→c`) |
| COND-DUP | 43 vs 44 | 687 | +0.579 | +0.555 | +0.062 (`pt→c`) … +0.532 (`pv→pt`) |

→ COND-CROSS 的节点级 gate 在**不同 seed 之间是可迁移的**（整体 r≈0.94，主导 pair
`c→pt` / `pv→pt` r≈0.6–0.7；低 gate 的 pair 迁移性弱，r≈0.26–0.40）；
COND-DUP 的节点级 gate **跨 seed 基本不可迁移**（r≈0.36–0.59）。

### 21. 其它 router diagnostics（§24 要求，逐 seed 保存在 `experiments/oft/o2b1_5/`）

| variant | seed | effective cross `c→pt`/`c→pv`/`pt→c`/`pt→pv`/`pv→c`/`pv→pt` | cross/diag C/Pt/Pv | post-cos `c-pt`/`c-pv`/`pt-pv` | state drift C/Pt/Pv |
|---|---:|---|---|---|---|
| COND-CROSS | 42 | .105/.020/.152/.037/.165/.180 | .117/.113/**.027** | −.035/−.026/−.022 | .093/.071/.074 |
| COND-CROSS | 43 | .101/.052/.139/.039/.149/.151 | .107/.102/**.042** | +.039/+.024/−.021 | .088/.065/.068 |
| COND-CROSS | 44 | .117/.064/.084/.053/.163/.218 | .097/.142/**.053** | −.053/−.028/+.000 | .065/.052/.082 |
| COND-DUP | 42 | .071/.123/.053/.115/.040/.045 | .047/.058/**.119** | −.049/−.030/−.003 | .080/.069/.087 |
| COND-DUP | 43 | .033/.131/.088/.136/.061/.021 | .075/.027/**.134** | +.044/+.029/−.009 | .087/.072/.089 |
| COND-DUP | 44 | .065/.134/.053/.125/.047/.049 | .050/.057/**.129** | −.081/−.014/+.001 | .070/.057/.094 |

- COND-CROSS 三个 seed 一致：**Pv 的 cross/diag 最小**（0.027/0.042/0.053），C 与 Pt 较大；
  COND-DUP 则完全相反（Pv 最大 0.119–0.134）。这是两个 variant 学到的**相反的先验**。
- ownership cosine 全部在 ±0.09 内（几乎没有 slot 间对齐/反对齐），state drift 0.05–0.09。
- 完整 per-class / correct-vs-incorrect / entropy / p10-p50-p90 见各 `*_stats_seed*.json`。

## 23. Part B verdict（条目 23）

| 判据（§27） | 要求 | 实测 | 结论 |
|---|---|---|---|
| REPLICATED_POSITIVE | CC−SD F1 mean>0 且 ≥2/3 seeds 正，Acc 不显著负，**且 CC−SC 也大多为正** | CC−SD F1 mean +0.65（2/3）✓；CC−SC F1 mean **−0.70**（1/3）✗ | **✗** |
| MATCHED_CONTROL_POSITIVE | CC−CD 的 F1 / BalAcc 至少 2/3 seeds 为正 | F1 2/3、BalAcc 2/3（**但 seed44 反转 −2.00/−3.31**） | 名义成立、**不稳健** |
| **SEED_UNSTABLE** | +F1 只存在 seed42、seed43/44 反转 | 每个 contrast 都有 ≥1 个 seed 符号反转；mean Δ ≤0.86pp < paired std 1.4–4.1pp；CC−SC BalAcc **0/3** 为正 | **✓（主判定）** |
| ACCURACY_NEGATIVE | CC 相对 DIAG/STATIC-DUP 平均 Acc 明显为负 | −0.05 / −0.07pp | ✗（Acc 基本持平） |

**Part B 结论：SEED_UNSTABLE（主）+ 名义 MATCHED_CONTROL_POSITIVE（不稳健）**。

- O2-B1 seed42 的正信号（CC−CD +2.32pp F1 / +3.68pp BalAcc）**没有被 seed43/44 复现为
  一个稳定效应**：均值只剩 +0.64/+0.58pp，且标准差是均值的 3–6 倍。
- 同时必须报告的对照：CC−STATIC-DUP（mean F1 +0.65 / BalAcc −0.47）与
  CC−STATIC-CROSS（mean F1 −0.70 / **BalAcc −2.34，0/3 为正**）—— 条件路由相对
  两个静态 cross 对照都没有稳定优势。
- **不是 ACCURACY_NEGATIVE**（Acc 没有明显损失），但也**没有任何 Acc 增益**：
  五个模型 3-seed Acc 全部在 55.09–55.24。

## 24. Final combined decision（条目 24）

- Part A = `NODE_SPECIFIC_WEAK + TARGET_RECEPTIVITY_SUFFICIENT + STATIC_TARGET_POLICY_SUFFICIENT`
- Part B = `SEED_UNSTABLE`（名义 `MATCHED_CONTROL_POSITIVE`，不稳健）

对照 §28 矩阵：

- **CASE 1**（NODE_SPECIFIC_SUPPORTED + REPLICATED_POSITIVE）→ **不成立**。
- **CASE 2**（TARGET_RECEPTIVITY_SUFFICIENT + REPLICATED_POSITIVE）→ **不成立**
  （Part B 不是 REPLICATED_POSITIVE）。
- **CASE 3**（STATIC_TARGET_POLICY_SUFFICIENT + Part B 稳定）→ **不成立**
  （Part A 一侧成立，但 Part B 信号不稳定）。
- **CASE 4**（SEED_UNSTABLE）→ **成立**：**HOLD**，不进入 Operator Bank，
  先重新审查 conditionality hypothesis。
- **CASE 5**（Part A node-specific 很弱 + Part B 无稳定 cross-info 增益）→ **同样成立**：
  3-seed 下 cross 变体相对 DIAG 无稳定增益（CC−DIAG F1 mean −0.86 / BalAcc −1.46，
  各 1/3 为正），**当前 dynamic cross routing 路线应关闭**，回到 DIAG strong parent
  重新选择增量方向。

**最终判定：CASE 4（HOLD）+ CASE 5（关闭当前 dynamic cross routing 路线）。**

## 25–26. 是否推荐进入 O2-B2 / 推荐哪种 granularity（条目 25–26）

**不推荐进入 O2-B2（Node-conditioned Functional Operator Bank）。**

理由（三重）：

1. **因果层面（Part A）**：node 级与 source 级条件性在冻结 checkpoint 上没有可测量的因果贡献
   （所有干预 ≤0.24pp，20-seed shuffle std 0.07pp）。在一个“节点级条件性不承重”的架构上加
   Operator Bank，是在给无效自由度扩容。
2. **复现层面（Part B）**：唯一曾支持“选择性 transfer 有用”的 seed42 信号未跨 seed 复现
   （SEED_UNSTABLE，每个 contrast 至少一次符号反转）。
3. **性能层面**：3-seed Acc 五个模型完全打平（0.15pp 带内），cross 变体相对 DIAG 的
   F1/BalAcc 均值均为负；STATIC-CROSS 反而是最稳的（BalAcc 43.33 ± 0.78）。

**如果将来重启**：唯一有跨 seed 结构支撑的粒度是 **target-level**（target C/Pt 高接收、
target Pv 低接收，在 COND-CROSS 三个 seed 上一致，且 same-target 相关 0.98–0.9999），
**但它在 3-seed 性能上并未兑现**，因此不能作为“下一阶段直接实现 target-conditioned
operator bank”的依据。**本阶段不做、不实现、不设计**任何 granularity 的 Operator Bank。

## 27. Warnings / anomalies（条目 27）

1. **DIAG-ID seed44 是离群点**：F1 47.81 / BalAcc 47.31，显著高于其 seed42/43
   （42.70/43.30、39.46/40.59），best epoch 99、总 epoch 129（其余 ~70–84 / 100–114）。
   DIAG 的 3-seed F1 mean 与巨大 std（44.60 ± 2.28）**完全由该 seed 驱动**，
   不能据此宣称 DIAG 的 F1 优于其他模型。
2. **Macro-F1 的跨 seed 噪声（0.87–2.28pp）与模型间差异同量级**：O0 的 ±0.3pp 是 Acc 噪声，
   **不能外推到 F1/BalAcc**；seed42 上 +1.28pp 的 F1 差异小于 seed 间 std。
3. **val split 每 seed 不同**（`Movies_nc_seed<S>_train0.6_val0.2.pt`），因此 seed 间
   per-node 相关性只能在 val 交集（634–690 节点）上计算，样本约为 val 的 1/5，
   该相关性是**保守下界**。
4. **COND-DUP 的 router 在 seed43 与 seed42/44 结构上不一致**（pair 排序、target gate
   均明显不同，inter-seed node r 仅 0.36）；其 matched-control 解释力因此受限。
5. **COND-CROSS 的 Pv gate 从 0.186 漂到 0.424（42→44）**：定性模式（Pv 最低）保持，
   但绝对幅度跨 seed 漂移 2.3×，说明该“先验”本身没有被数据强约束。
6. **ALL_ON 不是免费午餐**：全量 transfer 提高 F1/BalAcc 但掉 Acc 0.36pp，
   是典型的 precision/recall 重分配；它只是 frozen diagnostic，**不是 baseline**。
7. **Part A 的结论只对 seed42 这唯一 checkpoint 成立**；本阶段没有对 seed43/44 的
   checkpoint 重跑 Part A 干预（spec 只要求对 seed42 锁定 checkpoint 做因果审计），
   因此“node 级不承重”这一因果判断的跨 seed 外推是**有条件的**（有 Part B 的
   router 稳定性结果作为间接支持）。

## 28. Tests（条目 28）

`tests/test_oft_routing_audit.py`（**15 个新测试**，覆盖 §29 全部 14 条要求）：

| # | 测试 | 覆盖 |
|---|---|---|
| 1 | `test_original_replay_equals_unpatched_forward` | ① ORIGINAL replay ≡ 标准 forward |
| 2 | `test_patched_layer_restores_original_router` | 补丁可逆性（审计安全） |
| 3 | `test_pair_constant_same_gate_for_all_nodes_and_train_derived` | ② PAIR_CONSTANT 常数性 |
| 4 | `test_pair_constant_uses_train_split_not_val` | ③ train-derived（扰动 val 不改变常数） |
| 5 | `test_target_node_equalizes_sources_per_node` | ④ 同 target 两 source 逐位相同 |
| 6 | `test_target_node_preserves_node_variation` | ⑤ 保留 node 变化 |
| 7 | `test_target_constant_is_per_target_constant` | ⑥ 每 target 常数（且三 target 不塌缩） |
| 8 | `test_source_swap_exchanges_sources_per_target` | ⑦ gate 正确交换 + 分布保持 |
| 9 | `test_node_shuffle_preserves_distribution_breaks_alignment` | ⑧ 分布完全保持、对齐改变、pair 间独立置换 |
| 10 | `test_node_shuffle_deterministic_under_fixed_seed` | ⑨ 固定 seed 确定性 |
| 11 | `test_all_on_and_all_null` | ⑩⑪ ALL_ON=1 / ALL_NULL=0 |
| 12 | `test_interventions_do_not_change_payload_diag_or_parameters` | ⑫ payload/diag/参数不变 + gate 真的起作用 |
| 13 | `test_interventions_do_not_mutate_the_input_mapping` | ⑫ 策略不改写输入 |
| 14 | `test_audit_script_never_references_test_split` | ⑬ 脚本绝不访问 test labels（静态检查） |
| 15 | `test_patched_router_wrapper_targets_the_model_layer` | 补丁作用到正确的层 |

运行结果：

```
pytest tests/test_oft_mag.py -q   → 52 passed
pytest tests/ -q                  → 123 passed, 4 warnings（108 old + 15 new，⑭ old 全部通过）
```

## 29. Exact commands（条目 29）

环境：`/home/m3/miniconda3/envs/yhf_env/bin/python`（`PYTHONPATH=.`），device **`cuda:1`**（未换）。

**Part A（frozen causal audit，无训练）**

```bash
PYTHONPATH=. python scripts/oft_o2b1_5_routing_causal_audit.py \
  --ckpt outputs/oft_o2b1/conditional_cross_best.pt \
  --config outputs/2026-09-08/18-44-22/.hydra/config.yaml \
  --out-dir experiments/oft/o2b1_5 --device cuda:1
```

**Part B（10 个 replication run，仅 seed 变化）**

```bash
for SEED in 43 44; do
  for V in diag_id static_cross_dup static_cross conditional_cross_dup conditional_cross; do
    python -m src.main dataset=Movies task=nc model=oft_mag num_runs=1 seed=$SEED device=cuda:1 \
      task.evaluate_test=false model.variant=$V \
      task.history_path=experiments/oft/o2b1_5/${V}_movies_seed${SEED}.csv \
      task.save_ckpt_path=outputs/oft_o2b1_5/${V}_seed${SEED}_best.pt
  done
done
```

**Part B offline diagnostics（每 run per-class；conditional run 另做 router diagnostics）**

```bash
# 非 conditional（6 run）
PYTHONPATH=. python scripts/oft_o1_5_val_per_class.py \
  --ckpt outputs/oft_o2b1_5/<V>_seed<S>_best.pt --config <run>/.hydra/config.yaml --device cuda:1 \
  --out experiments/oft/o2b1_5/<V>_val_per_class_seed<S>.csv

# conditional（4 run）
PYTHONPATH=. python scripts/oft_o2b1_router_diagnostics.py \
  --ckpt outputs/oft_o2b1_5/<V>_seed<S>_best.pt --config <run>/.hydra/config.yaml --device cuda:1 \
  --out-stats experiments/oft/o2b1_5/<V>_stats_seed<S>.json \
  --out-per-class experiments/oft/o2b1_5/<V>_val_per_class_seed<S>.csv \
  --out-nodes experiments/oft/o2b1_5/<V>_router_nodes_val_seed<S>.csv
```

**3-seed 汇总**

```bash
PYTHONPATH=. python scripts/oft_o2b1_5_seed_summary.py --out-dir experiments/oft/o2b1_5
```

**§22 合规检查（resolved config 逐键比对）**：把每个 seed43/44 run 的
`.hydra/config.yaml` 与对应 seed42 run 逐键比较，忽略 `seed` / split 路径 / 输出路径后
**0 处差异**（10/10 run 全部 IDENTICAL）。

**Tests**

```bash
pytest tests/test_oft_mag.py -q
pytest tests/ -q
```

## 30. 明确声明（条目 30）

- **no Test**：全程 `task.evaluate_test=false`；Part A 脚本经单测静态检查不含
  `test_idx` / `.test_` / `evaluate_test`；未读取任何 test label 或 test metric。
- **no tuning**：未改任何训练超参（lr / optimizer / dropout / patience / epochs / aux loss /
  factor_dim / hidden_dim / router hidden / embedding dim / init / scheduler / training mode /
  inference mode），唯一变化是 seed；未做 temperature / threshold / 手工 gate 选择 /
  按 val 性能优化常数 / 用 val label 决定 gate。
- **no O2-B2 implementation**：未实现 Operator Bank、B2/B3 experts、FiLM、source-aware router、
  channel groups、edge router、attention、multi-scale、第二层、global context、PPR、新 loss、
  entropy 正则，未跑 Toys/Grocery、seed45+。

## 36. Commit（条目 33）

```bash
git add scripts/oft_routing_interventions.py scripts/oft_o2b1_5_routing_causal_audit.py \
        scripts/oft_o2b1_5_seed_summary.py tests/test_oft_routing_audit.py \
        experiments/oft/o2b1_5/ docs/OFT_O2B1_5_ROUTING_CAUSALITY_AND_REPLICATION_REPORT.md
git commit -m "O2-B1.5: audit routing granularity and replicate across seeds"
git tag oft-o2b1-5-routing-audit
git push origin oft-mag --tags
```

| 项 | 值 |
|---|---|
| starting HEAD | `93015ebede9e9424146873bad138e35b94a217a7` |
| O2-B1.5 commit | `32455971259ea04a050c8203291bcb0db4069211` |
| 本次补记 commit | 仅把上述 SHA 写入本报告（文档改动，无代码/实验变化） |
| tag | `oft-o2b1-5-routing-audit`（打在 `3245597`） |

## 附：一句话总结

**在冻结的 seed42 checkpoint 上，路由的有效自由度只有 target 一级（C/Pt 高接收、Pv 低接收，
node/source/pair 级均无可测因果贡献）；而 seed42 那个 +2.32pp F1 / +3.68pp BalAcc 的
“条件路由优势”没有跨 seed 复现（SEED_UNSTABLE），3-seed 下五个模型 Acc 完全打平。
→ HOLD，不进入 O2-B2，关闭当前 dynamic cross routing 路线。**
