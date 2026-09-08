# OFT-MAG O2-A — Cross-Ownership Utility Test Report

日期：2026-09-08
范围：O2-A（STATIC-CROSS + capacity-matched STATIC-CROSS-DUP）。**未实现 O2-B**，
未实现 dynamic router / operator bank / Null routing / channel-wise routing /
multi-scale，未加任何新 loss / 正则 / gate，未做超参搜索，未跑 Toys/Grocery、
未跑 seed43/44。
协议：**Val-only**（`task.evaluate_test=false`，全程未访问/未记录 Test 指标）。
正式 checkpoint 选择：**best-Val-Acc**（未改成 F1 selection）。

## Verdict：**CASE D — STATIC_HARMFUL**（按预注册判据），但 CASE-D 机制签名**缺席**

核心因果量：

```text
Δcross_info = STATIC-CROSS − STATIC-CROSS-DUP
            = 54.56 − 55.25 = −0.69pp Val Acc   （Macro-F1 −0.44pp，Balanced Acc +0.37pp）
```

按 §15 预注册判据（STATIC-CROSS 比 DUP 低 >0.5pp → CASE D），本结果落入
**STATIC_HARMFUL**。**但**该判据同时预设的机制签名（post-state cosine 明显上升 /
cross_update_ratio 很大 / ownership drift 明显）**一条都不成立**：

| CASE-D 预设签名 | 实测 | 是否出现 |
|---|---|---|
| post-state cosine 大幅上升 | CROSS 与 DUP 几乎相同，且相对 DIAG 反而**更负**（更分离） | ✗ |
| cross_update_ratio 很大 | CROSS 0.083–0.122 vs DUP 0.083–0.155（同量级、受控） | ✗ |
| ownership drift 明显 | CROSS 0.061–0.091 < DUP 0.074–0.097 | ✗ |

因此正确的读法是：**static unconditional cross transfer 没有带来可用的增量信息，
也没有造成 semantic interference / ownership 坍塌**；它只是把一部分本已由 diagonal
路径 + 额外容量完成的工作重新分配了一遍（Acc 略降、Balanced Acc 略升、尾部类互相
置换）。真正的失败模式是"无信息"，不是"信息有害"。这一区分直接决定 O2-B 的设计。

**本轮最重要的单条结果不是 CASE D 本身，而是 Δcapacity：**

```text
Δcapacity = STATIC-CROSS-DUP − DIAG = 55.25 − 55.04 = +0.21pp Acc
                                     Macro-F1 +1.77pp，Balanced Acc +3.01pp
```

即：一个**完全看不到其他 ownership 状态**的静态 cross 分支（只读 target 自己的
N^b），仅凭"多了一条 128→128 residual 通路"就吃掉了大部分 F1 / balanced-accuracy
增益。这说明 O1.5 之后任何 cross-factor 方案的收益都必须相对 capacity-matched
control 来读 —— 这正是 O2-A 存在的理由。

---

## 1. Git SHA 与 files changed

| 项 | 值 |
|---|---|
| starting HEAD | `67bb0ab9f46924bf517428101fc159becb3c26d9`（tag **`oft-o1-5`**，branch `oft-mag`；与计划给定 commit 一致） |
| O2-A commit | `09c039f26c3dbbbe8985d48025035299f3862643` |
| O2-A tag | **`oft-o2a-static-cross`** → `09c039f`（已 `git push origin oft-mag --tags`） |
| ending HEAD | `09c039f` + 本报告补记 SHA 的文档 commit（见 §21） |

新增：

```text
docs/OFT_O2A_CROSS_UTILITY_REPORT.md              # 本报告
scripts/oft_o2a_diagnostics.py                    # O2-A 离线诊断（post-transition / cross pair / per-class）
experiments/oft/o2a/static_cross_dup_movies_seed42.csv   # DUP 逐 epoch history（100 ep）
experiments/oft/o2a/static_cross_movies_seed42.csv       # CROSS 逐 epoch history（101 ep）
experiments/oft/o2a/static_cross_dup_val_per_class.csv   # DUP per-class（best-Acc ckpt）
experiments/oft/o2a/static_cross_val_per_class.csv       # CROSS per-class（best-Acc ckpt）
experiments/oft/o2a/diag_id_val_per_class.csv            # DIAG parent per-class（对照，重算）
experiments/oft/o2a/diag_id_stats.json                   # DIAG parent ownership 诊断
experiments/oft/o2a/static_cross_dup_stats.json          # DUP 全量诊断标量
experiments/oft/o2a/static_cross_stats.json              # CROSS 全量诊断标量
```

修改：

```text
src/models/oft_components.py      # +StaticCrossLayer + post_transition_stats + CROSS_PAIR_* （DiagIDLayer 逐字未动）
src/models/oft_mag.py             # +static_cross / static_cross_dup dispatch、+encode_states、+O2-A aux_info
configs/model/oft_mag.yaml        # 注释说明两个新 variant（值/默认不变：diag_id）
tests/test_oft_mag.py             # +16 tests（O2-A 12 项要求 + 4 项回归/一致性）
scripts/oft_o1_5_val_per_class.py # 抽出 build_model_and_head / val_per_class_rows（行为不变，见 §5）
```

未修改（语义锁定文件）：`biaxis_p0.py`、`biaxis_components.py`、`biaxis_final.py`、
`dip.py`、`src/tasks/nc.py`、`src/tasks/common.py`、`src/main.py`、数据与 split。

## 2. 环境与运行

conda `yhf_env`（torch 2.4.0+cu121 / PyG 2.7.0 / hydra 1.3.2）、RTX 3090 双卡、
cwd `/hdd1/DataInHere/YHF/exp`。两条新运行均在 **`device=cuda:1`**（与 O1.5 parent
DIAG rerun 同一 GPU；运行时该卡空闲 12MiB/0% 占用）。`cuda:1` 可用，**未换 GPU**。
单 run 训练墙钟约 9–13s（100 epoch，full-graph）。

## 3. 精确数学：STATIC-CROSS

保留 O1 DIAG 主路径**逐字不变**（`DiagIDLayer` 未修改，diag_id / diag_nograph 仍走原
代码路径）：

```text
X_i^a        = V_a H_i^a                                   # per-slot Linear 128->128, bias=False
N_i^a        = incoming_mean_j->i ( X_j^a )                # no self-loop, no GCN norm
Delta_diag_i^b = D_b N_i^b                                 # per-slot Linear 128->128, bias=False
```

新增 **additive** static cross branch，六个 bias-free 128→128 映射
`C_{a->b}`（a ≠ b，仅 off-diagonal）：

```text
delta_{a->b}   = C_{a->b}( N^a )                           # STATIC-CROSS: 真实 source ownership 状态
Delta_cross^b  = (1 / (S-1)) * sum_{a != b} delta_{a->b}   # S = 3 -> 除以 2
H_next_i^b     = LN_b( H_i^b + Delta_diag_i^b + Delta_cross_i^b )
```

六个 pair（固定顺序，无 diagonal）：

```text
c->pt, c->pv, pt->c, pt->pv, pv->c, pv->pt
```

不创建 `c->c / pt->pt / pv->pv`：diagonal 已由现有 `D_b` 承担。transfer function 是
**静态、无条件**的 —— 不读 target node context、source node context、same-node
semantic context、factor embedding、degree、relation、channel group（conditionality
属 O2-B 范围）。

## 4. 精确数学：STATIC-CROSS-DUP（capacity-matched control）

结构、参数量、模块数、initialization、optimizer 参数集合、diagonal 路径、cross
branch 数量、target update 形式**全部相同**，唯一差异是 cross branch 的输入：

```text
delta_dup_{a->b}  = C_{a->b}( N^b )                        # 只读 target 自己的 neighborhood state
Delta_cross_dup^b = (1 / 2) * sum_{a != b} C_{a->b}( N^b )
H_next_i^b        = LN_b( H_i^b + Delta_diag_i^b + Delta_cross_dup_i^b )
```

source label `a` 对应的模块仍然存在并被训练，只是收到的是 `N^b`。
因此 **STATIC-CROSS 与 DUP 的唯一变量是"是否真的输入了其他 ownership 状态的
graph information"**。

**训练后观察（已验证，§17）**：DUP 的六个矩阵按 target 塌成三组完全相同
（`C_{c->pt} == C_{pv->pt}`、`C_{c->pv} == C_{pt->pv}`、`C_{pt->c} == C_{pv->c}`，
逐位相等）。这是零初始化 + 相同输入 + 相同梯度的解析必然（`dL/dC_{a->b}` 与 `a` 无关），
实测 frobenius 范数逐对相等（0.8708 / 1.2730 / 0.9148）。**这不削弱对照有效性**：
参数计数、可训练张量数、前向 FLOPs 与 CROSS 完全一致，DUP 仍然拿满了全部额外容量，
只是这份容量在函数上冗余。

## 5. Parameter-matching proof

单元测试 `test_static_cross_and_dup_parameter_counts_match` + `..._state_dict_identical_init`：

- `n_dup == n_cross`，且 `n_cross − n_diag == 6 × factor_dim²`；
- 同一 RNG 流下两 variant 的 state_dict **key 集合相同、逐 key shape 相同、逐 key 数值
  `torch.equal`**（初始化完全一致）。

真实规模核对（run 日志 `model+head params=`，Movies）：

| model | model params | +head (20×256+20) | model+head |
|---|---:|---:|---:|
| P0（参考） | 1,037,568 | 5,140 | 1,042,708 |
| DIAG-ID / NOGRAPH | 1,136,640 | 5,140 | 1,141,780 |
| **STATIC-CROSS-DUP** | 1,234,944 | 5,140 | **1,240,084** |
| **STATIC-CROSS** | 1,234,944 | 5,140 | **1,240,084** |

增量 `1,240,084 − 1,141,780 = 98,304 = 6 × 128 × 128` ✓（与计划 §8 预期一致）。
DUP 与 CROSS 参数**严格相同**。

## 6. Zero-init proof

`StaticCrossLayer.__init__` 对六个 `C_{a->b}` 执行 `nn.init.zeros_(weight)`。单元测试
`test_cross_matrices_zero_initialized` 断言六个权重 `count_nonzero == 0` 且
`cross_update(N) == 0`（逐位）；`test_init_forward_equals_diag_path` 断言同 seed 下
`static_cross` / `static_cross_dup` 的 eval forward 输出与 `diag_id` **逐位相等**
（`torch.equal`，非仅 tolerance），且 `cross_update_ratio` / `cross_to_diag_ratio` /
六个 pair ratio 全为 0。即 **step 0 严格退化为 DIAG-ID**，新分支不会在训练一开始
rewrite ownership state。

## 7. Unit-test results

```bash
pytest tests/test_oft_mag.py -q    # 32 passed（16 O1/O1.5 + 16 O2-A）, 1.5s
pytest tests/ -q                   # 88 passed（72 baseline + 16 new）, 5.4s
```

新增 16 个 O2-A 测试（逐条对应计划 §12 的 12 项要求，括号内为要求编号）：

| # | test | 验证点 |
|---|---|---|
| 1 | `test_static_cross_and_dup_parameter_counts_match` | ① 两 variant 参数量严格一致，且 == diag_id + 6·F² |
| 2 | `test_static_cross_and_dup_state_dict_identical_init` | ② 同 seed 下所有参数 key/shape/数值一致 |
| 3 | `test_six_cross_matrices_exist_only_off_diagonal` | ③ 六个 off-diagonal 矩阵存在，无 a==b |
| 4 | `test_cross_matrices_zero_initialized` | ④ 初始权重严格为 0，cross_update ≡ 0 |
| 5 | `test_init_forward_equals_diag_path` | ⑤ 初始化态 forward 等价于 DIAG path（逐位） |
| 6 | `test_static_cross_messages_follow_source_slot` | ⑥ 只改 N^C → 仅 C→Pt / C→Pv 改变，其余四个 pair 逐位不变；target C 的 cross update 不变 |
| 7 | `test_static_cross_dup_ignores_other_source_states` | ⑦ 改 N^a（a≠b）→ target b 的 cross update 逐位不变 |
| 8 | `test_empty_graph_diag_and_cross_updates_exactly_zero` | ⑧ 空图：diag / cross update 与全部 stats 严格 0 |
| 9 | `test_isolated_node_graph_updates_exactly_zero` | ⑨ isolated node：H_next == LN(H)，两 variant 均成立 |
| 10 | `test_cross_gradients_nonzero_after_one_step` | ⑩ 一步 backward 后六个 C_{a->b} 梯度均非零 |
| 11 | `test_diag_variants_aux_keys_unchanged` | ⑪ 回归：diag_id / diag_nograph 的 aux_info 仍是原 16 个 key，无 O2-A key 泄漏 |
| 12 | `test_no_giant_edge_pair_tensor_static_cross` | ⑫ profiler 审计：无 `[E,S,S,...]` 巨型 edge-pair tensor |
| 13 | `test_cross_variants_aux_keys_present` | 计划 §10 要求的全部 cross / post-transition 诊断 key 确实产出 |
| 14 | `test_post_transition_stats_semantics` | H0==H1 → drift 0、post_cos == pre_cos |
| 15 | `test_cross_variants_enforce_single_layer` | 两新 variant 同样强制 num_layers=1 |
| 16 | `test_encode_states_matches_forward_h1` | 离线 hook 返回的 H1/stats 与 forward 逐位一致 |

原有 16 个 diag_id / diag_nograph tests 全部继续通过。

**diag path 数值回归（checkpoint 级）**：用改造后的 per-class 脚本重放 O1.5 的
`diag_id_best.pt`，得到 `val_acc=0.550390 / macro_f1=0.427039 / balanced_acc=0.394594`，
与 O1.5 报告 §10 及已提交的 `experiments/oft/o1_5/diag_id_val_per_class.csv`
**逐行 diff 完全相同** → diag 路径数值未被 O2-A 改动。

## 8. 执行的完整命令

```bash
# Git prep（起点 = tag oft-o1-5, 67bb0ab；工作区干净）

# A. Unit tests
pytest tests/test_oft_mag.py -q
pytest tests/ -q

# B. diag path 回归检查（复用 O1.5 ckpt，不重跑 DIAG）
PYTHONPATH=. python scripts/oft_o1_5_val_per_class.py \
  --ckpt outputs/oft_o1_5/diag_id_best.pt \
  --config outputs/2026-09-08/17-32-40/.hydra/config.yaml \
  --out /tmp/diag_id_regression.csv --label diag_id_regression

# C. Run 1 — STATIC-CROSS-DUP（Movies seed42, Val-only, cuda:1）
python -m src.main dataset=Movies task=nc model=oft_mag num_runs=1 seed=42 device=cuda:1 \
  task.evaluate_test=false model.variant=static_cross_dup \
  task.history_path=experiments/oft/o2a/static_cross_dup_movies_seed42.csv \
  task.save_ckpt_path=outputs/oft_o2a/static_cross_dup_best.pt
#   -> hydra outputs/2026-09-08/18-03-36；log 镜像 outputs/oft_o2a_static_cross_dup.log

# D. Run 2 — STATIC-CROSS（同上）
python -m src.main dataset=Movies task=nc model=oft_mag num_runs=1 seed=42 device=cuda:1 \
  task.evaluate_test=false model.variant=static_cross \
  task.history_path=experiments/oft/o2a/static_cross_movies_seed42.csv \
  task.save_ckpt_path=outputs/oft_o2a/static_cross_best.pt
#   -> hydra outputs/2026-09-08/18-03-49；log 镜像 outputs/oft_o2a_static_cross.log

# E. 离线诊断（best-Val-Acc ckpt；val split only）
PYTHONPATH=. python scripts/oft_o2a_diagnostics.py \
  --ckpt outputs/oft_o1_5/diag_id_best.pt \
  --config outputs/2026-09-08/17-32-40/.hydra/config.yaml \
  --out-stats experiments/oft/o2a/diag_id_stats.json \
  --out-per-class experiments/oft/o2a/diag_id_val_per_class.csv --label diag_id_parent
PYTHONPATH=. python scripts/oft_o2a_diagnostics.py \
  --ckpt outputs/oft_o2a/static_cross_dup_best.pt \
  --config outputs/2026-09-08/18-03-36/.hydra/config.yaml \
  --out-stats experiments/oft/o2a/static_cross_dup_stats.json \
  --out-per-class experiments/oft/o2a/static_cross_dup_val_per_class.csv --label static_cross_dup
PYTHONPATH=. python scripts/oft_o2a_diagnostics.py \
  --ckpt outputs/oft_o2a/static_cross_best.pt \
  --config outputs/2026-09-08/18-03-49/.hydra/config.yaml \
  --out-stats experiments/oft/o2a/static_cross_stats.json \
  --out-per-class experiments/oft/o2a/static_cross_val_per_class.csv --label static_cross
```

未重跑 DIAG：O1.5 cuda:1 rerun 的 history/ckpt 完好，直接作为 parent reference；
diag path 的数值正确性由 §7 的 checkpoint 重放 diff 保证。

## 9. DIAG locked parent reference

| 来源 | device | best-Acc ep | Val Acc | Val Macro-F1 | Balanced Acc |
|---|---|---:|---:|---:|---:|
| O1 锁定 run（正式对比基准） | cuda:0 | 75 | 54.86% | 42.73% | — |
| **O1.5 rerun（本轮 parent，优先复用）** | cuda:1 | 75 | **55.04%** | **42.70%** | **39.46%** |
| P0 no-graph（O1 锁定） | cuda:0 | 57 | 51.71% | 37.62% | — |
| A0（O0 锁定） | — | 74 | 55.01% | 46.37–46.67% | — |
| DiP（O0 锁定） | — | 78 | 56.00–56.27% | 46.49–48.06% | — |

## 10. 性能表（Movies seed42，Val-only，单 run，best-Val-Acc checkpoint）

| variant | params (model+head) | best-Acc ep | 早停 ep | **Val Acc** | **Val Macro-F1** | **Balanced Acc** | F1@bestAcc |
|---|---|---:|---:|---:|---:|---:|---:|
| DIAG-ID（parent, cuda:1） | 1,141,780 | 75 | 105 | **55.04%** | **42.70%** | **39.46%** | 42.70% |
| STATIC-CROSS-DUP | 1,240,084 | 70 | 100 | **55.25%** | **44.47%** | **42.47%** | 44.47% |
| STATIC-CROSS | 1,240,084 | 71 | 101 | **54.56%** | **44.03%** | **42.83%** | 44.03% |

（best-F1 epoch 仅作诊断、**不用于 selection**：DUP ep77 F1 47.63% / Acc 53.87%；
CROSS ep84 F1 48.14% / Acc 53.57%。两条新 run 的 Acc–F1 trade-off 形态与 O1.5 §9
记录的整个模型族一致 —— best-F1 比 best-Acc 晚 6–13 epoch，选 F1 会牺牲 Acc。）

## 11. Δ 表

| 量 | 定义 | ΔVal Acc | ΔVal Macro-F1 | ΔBalanced Acc |
|---|---|---:|---:|---:|
| **Δcapacity** | DUP − DIAG | **+0.21pp** | **+1.77pp** | **+3.01pp** |
| **Δcross_info** | CROSS − DUP | **−0.69pp** | **−0.44pp** | **+0.37pp** |
| **Δtotal_cross** | CROSS − DIAG | **−0.48pp** | **+1.33pp** | **+3.37pp** |

以 O1 锁定 run（54.86%）为 parent 时：Δtotal_cross = **−0.30pp** Acc（Δcross_info 不变，
仍 −0.69pp）。两套 parent 的结论一致。

逐项读法：

- **Δcross_info（唯一真正回答 O2-A 科学问题的量）**：Acc −0.69pp、F1 −0.44pp、
  Balanced Acc +0.37pp。Acc/F1 方向为负且 Acc 超出 ±0.3pp 噪声带约 2×。
- **Δcapacity**：一个不含跨 ownership 信息的静态分支就能拿到 +1.77pp F1 /
  +3.01pp Balanced Acc —— 这是本轮最强的单项信号，也是 O2-B 必须继续带
  capacity-matched control 的直接证据。
- **Δtotal_cross**：相对 DIAG 的 F1 / Balanced Acc 增益（+1.33 / +3.37pp）**不能**
  归给 cross 信息，其中 Acc 还倒退了 0.48pp。

## 12. Cross pair update ratios（best-Acc ckpt，val split）

定义 `cross_pair_ratio_{a->b} = mean_i ||delta_{a->b,i}|| / (||H_i^b|| + eps)`
（仅描述为 learned relative update magnitude，**不作 factor demand 解读**）：

| pair | DUP | CROSS |
|---|---:|---:|
| c→pt | 0.0975 | 0.0892 |
| c→pv | 0.1453 | 0.1214 |
| pt→c | 0.1124 | 0.1394 |
| pt→pv | 0.1453 | 0.1583 |
| pv→c | 0.1124 | 0.1390 |
| pv→pt | 0.0975 | 0.1566 |

DUP 的三组逐位相等（§4）是结构性必然；CROSS 六个互不相同，且 **source 侧不对称**
（Pt/Pv → C 的贡献 0.139 > C → Pt 的 0.089），说明真实 cross 通道确实被差异化了 ——
只是这种差异化没有换来 Acc。

## 13. Per-target diag / cross / cross-to-diag（best-Acc ckpt）

`diag_update_ratio_b = mean ||Delta_diag^b|| / (||H^b||+eps)`；
`cross_update_ratio_b = mean ||Delta_cross^b|| / (||H^b||+eps)`；
`cross_to_diag_ratio_b = mean [ ||Delta_cross^b|| / (||Delta_diag^b||+eps) ]`。

| target | DUP diag | DUP cross | DUP cross/diag | CROSS diag | CROSS cross | CROSS cross/diag |
|---|---:|---:|---:|---:|---:|---:|
| C | 0.3168 | 0.1124 | 0.3547 | 0.3122 | 0.1059 | 0.3440 |
| Pt | 0.4156 | 0.0975 | 0.2333 | 0.3950 | 0.1006 | 0.2552 |
| Pv | 0.4007 | 0.1453 | 0.3598 | 0.4358 | 0.1267 | 0.3085 |

逐 epoch 轨迹（log `Aux` 行解析，100/101 epoch 全程）：

| key | DUP first → last (min, max) | CROSS first → last (min, max) |
|---|---|---|
| c_cross_update_ratio | 0.000 → 0.121 (0.000, 0.122) | 0.000 → 0.105 (0.000, 0.112) |
| pt_cross_update_ratio | 0.000 → 0.124 (0.000, 0.124) | 0.000 → 0.083 (0.000, 0.110) |
| pv_cross_update_ratio | 0.000 → 0.125 (0.000, 0.155) | 0.000 → 0.122 (0.000, 0.127) |
| c_cross_to_diag_ratio | 0.000 → 0.391 (0.000, 0.402) | 0.000 → 0.350 (0.000, 0.380) |
| pt_cross_to_diag_ratio | 0.000 → 0.281 (0.000, 0.281) | 0.000 → 0.219 (0.000, 0.291) |
| pv_cross_to_diag_ratio | 0.000 → 0.366 (0.000, 0.370) | 0.000 → 0.363 (0.000, 0.364) |

cross 通道从严格 0 单调长到 ~0.08–0.16（cross/diag 0.22–0.39），**全程有界、不爆炸、
不归零**；没有任何 HOLD 判据触发。CROSS 与 DUP 的更新幅度处于同一量级 → 两者消耗的
"计算预算"相当，差异不在幅度而在方向。

## 14. Post-transition ownership 诊断（best-Acc ckpt）

`post_cos_{a}_{b} = mean_i cos(H1_i^a, H1_i^b)`；`state_drift_b = mean_i [1 − cos(H1_i^b, H0_i^b)]`。

| 量 | DIAG (parent) | DUP | CROSS |
|---|---:|---:|---:|
| pre_cos c_pt → post_cos c_pt | 0.0369 → **0.0383** | −0.0836 → **−0.0607** | −0.0848 → **−0.0605** |
| pre_cos c_pv → post_cos c_pv | −0.0514 → **−0.0309** | −0.0295 → **−0.0122** | −0.0315 → **−0.0036** |
| pre_cos pt_pv → post_cos pt_pv | 0.0258 → **0.0245** | −0.0277 → **−0.0372** | −0.0240 → **−0.0280** |
| post_norm c / pt / pv | 11.31 / 11.25 / 11.41 | 11.31 / 11.26 / 11.41 | 11.32 / 11.26 / 11.41 |
| state_drift c / pt / pv | 0.048 / 0.056 / 0.083 | 0.074 / 0.075 / 0.097 | 0.064 / 0.068 / 0.091 |

结论（对 §18 判读问题的直接回答）：

1. **没有 ownership 坍塌**。三个模型的 post-transition 槽间 cosine 全部在 |0.061| 以内，
   与 pre-transition 同量级；CROSS 相对 DIAG 甚至**更分离**（c_pt −0.061 vs +0.038）。
2. **没有 semantic interference 迹象**。CROSS 的 state drift（0.064–0.091）**低于**
   DUP（0.074–0.097），post_norm 三槽几乎与 DIAG 完全重合。
3. 因此 STATIC-CROSS 的性能略降**不能**用"cross transfer 把 ownership states 揉成
   相似表示"来解释；cross 信息是被"读了但没用上"。

P0 原始 common/private 诊断继续保留在 history CSV（`common_sim` / `private_sim` /
`cp_overlap_t` / `cp_overlap_v`），本轮未见异常（与 O1.5 同形态）。

## 15. Per-class 对比（best-Acc ckpt，val split，3334 节点 / 20 类）

CSV：`experiments/oft/o2a/{diag_id,static_cross_dup,static_cross}_val_per_class.csv`
（schema：class_id, support, recall, precision, f1, predicted_count, true_count）。
ckpt 重算与 history best-Acc 行逐位一致（DUP ep70 0.552490/0.444705；
CROSS ep71 0.545591/0.440328）。

ΔF1（pp）逐类：

| cls | support | DIAG F1 | DUP F1 | CROSS F1 | **CROSS−DUP** | DUP−DIAG | CROSS−DIAG |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 355 | 0.4583 | 0.4510 | 0.4588 | +0.77 | −0.73 | +0.04 |
| 1 | 1098 | 0.6514 | 0.6473 | 0.6465 | −0.08 | −0.41 | −0.49 |
| 2 | 363 | 0.4552 | 0.4334 | 0.3627 | **−7.07** | −2.18 | −9.25 |
| 3 | 277 | 0.7838 | 0.7628 | 0.7812 | +1.85 | −2.11 | −0.26 |
| 4 | 24 | 0.4500 | 0.6818 | 0.6222 | −5.96 | **+23.18** | **+17.22** |
| 5 | 82 | 0.2478 | 0.3111 | 0.2901 | −2.10 | +6.33 | +4.23 |
| 6 | 122 | 0.3889 | 0.4372 | 0.4115 | −2.57 | +4.83 | +2.26 |
| 7 | 210 | 0.3689 | 0.3889 | 0.3767 | −1.22 | +2.00 | +0.78 |
| 8 | 30 | 0.5000 | 0.5000 | 0.4706 | −2.94 | 0.00 | −2.94 |
| 9 | 37 | 0.0930 | 0.2308 | 0.2128 | −1.80 | **+13.77** | **+11.97** |
| 10 | 269 | 0.5958 | 0.6167 | 0.6012 | −1.55 | +2.09 | +0.54 |
| 11 | 88 | 0.3688 | 0.3622 | 0.4151 | **+5.29** | −0.66 | +4.63 |
| 12 | 23 | 0.7273 | 0.7600 | 0.7500 | −1.00 | +3.27 | +2.27 |
| 13 | 43 | 0.5714 | 0.5909 | 0.5652 | −2.57 | +1.95 | −0.62 |
| 14 | 50 | 0.4138 | 0.3571 | 0.4407 | **+8.35** | −5.67 | +2.69 |
| 15 | 47 | 0.2687 | 0.3656 | 0.3855 | +2.00 | +9.69 | **+11.69** |
| 16 | 38 | 0.0513 | 0.0000 | 0.0000 | 0.00 | −5.13 | −5.13 |
| 17 | 141 | 0.4140 | 0.3796 | 0.3188 | **−6.08** | −3.44 | −9.52 |
| 18 | 23 | 0.4324 | 0.5000 | 0.3636 | **−13.64** | +6.76 | −6.88 |
| 19 | 14 | 0.3000 | 0.1176 | 0.3333 | **+21.57** | **−18.24** | +3.33 |

**最大正/负变化（CROSS − DUP）**：正 = cls19 **+21.57pp**（support 仅 14）、cls14 +8.35、
cls11 +5.29；负 = cls18 **−13.64pp**（support 仅 23）、cls2 −7.07（support 363）、
cls17 −6.08（support 141）、cls4 −5.96（support 24）。

**读法（不做预设）**：

- CROSS−DUP 的正负两端都落在 **support ≤ 50 的极小类**（cls19 sup14、cls18 sup23、
  cls14 sup50、cls4 sup24）—— 单类只有 3–8 个样本被改判，F1 就会摆动 10–20pp。
  这些是**噪声主导**的逐类差异，不能当作机制证据。
- 唯一 support 较大且变化显著的是 cls2（sup 363，−7.07pp）与 cls17（sup 141，−6.08pp），
  它们构成 CROSS 相对 DUP 的 Acc 损失来源；CROSS 在这些类上预测更分散。
- 相比之下 **DUP−DIAG 的增益更"真"**：cls4 +23.18、cls9 +13.77、cls15 +9.69 与
  cls5 +6.33、cls18 +6.76 —— 集中在 O1.5 §10 已识别的尾部瘫痪类（cls16/9/5/15/19），
  说明"额外一条静态 residual 通路"确实改善了尾部召回（Balanced Acc +3.01pp 一致）。
- 因此 cross 信息的净效果是**在尾部类之间做置换**（把 DUP 已经挣到的部分收益
  从 cls18/17/2 挪给 cls19/14/11），而不是新增可用信息。

## 16. Warnings / memory

- 两条 run 日志：**0 NaN / 0 Inf / 0 Traceback / 0 CUDA error / 0 OOM**
  （`grep -icE "\bnan\b|\binf\b|Traceback|CUDA error|out of memory"` = 0）。
- 均在 patience 自然早停（100 / 101 epoch，patience 30）。
- 显存：Movies full-graph，O1 已测 DIAG-ID train peak 1.07GB / eval 0.58GB；
  O2-A 仅增加 6 个 128×128 无 bias 线性（cross 分支张量规模与 diag 同阶），
  运行期间 `nvidia-smi` 未出现异常占用，无 memory pathology。
- `task.evaluate_test=false` 全程生效，日志无 Test 指标行。

## 17. 实现说明（最小侵入）

- `DiagIDLayer` **逐字未动**；`diag_id` / `diag_nograph` 的 `_encode_oft` 路由逻辑
  改写为等价形式（`None if variant == "diag_nograph" else edge_index`，两 variant 取值不变），
  其 aux_info key 集合被单元测试钉死为原 16 个（`test_diag_variants_aux_keys_unchanged`）。
- `StaticCrossLayer` 继承 `DiagIDLayer`，复用 V/D/LN/`incoming_mean`，只覆写 `propagate`
  并在其上叠加 cross 分支；`cross_messages` / `cross_update` 供单元测试直接验证因果性。
- `scripts/oft_o1_5_val_per_class.py` 抽出了 `build_model_and_head` / `val_per_class_rows`
  两个函数（**行为不变**：§7 的 diff 为证），`scripts/oft_o2a_diagnostics.py` 直接
  导入复用，没有第二套 per-class 实现。
- `src/tasks/nc.py` / `src/tasks/common.py` **未修改**：新诊断 key 全部以 `oft_` 前缀
  产出，走已有的 `_aux_iteration_keys` 放行路径进入 epoch 日志（§13 的轨迹即由
  log `Aux` 行解析得到）。

## 18. Interpretation category

**CASE D — STATIC_HARMFUL**（按 §15 预注册判据的字面定义）。

同时必须记录与判据预期不符的两点（否则会误导 O2-B）：

1. **机制签名缺席**：CASE D 预设的"post-state cosine 上升 / cross_update_ratio 很大 /
   ownership drift 明显"三条全部不成立（§14）。ownership 结构未被破坏，
   CROSS 的 drift 甚至低于 DUP。
2. **Balanced Acc 方向相反**：Δcross_info 在 Acc/F1 上为负，但在 Balanced Acc 上
   为 **+0.37pp**，即 cross 信息把预测推向尾部类，代价是头部/中频类的 Acc。

正确的科学结论是：**static unconditional cross-ownership transfer 无法从真实
cross-ownership evidence 中得到可靠增量**（这正是计划 CASE C 的语义），它同时
**也不是**通过 semantic interference 造成伤害（CASE D 的机制）。按字面判据落 D、
按机制实质近 C —— 报告两者，避免把"无信息"误读成"有害"。

## 19. 对 O2-B 的建议

1. **主判据继续用 Δcross_info（CROSS − DUP），不要用 Δtotal_cross**。本轮 Δcapacity
   = +1.77pp F1 / +3.01pp Balanced Acc 全部由"多一条静态 residual 通路"贡献；
   任何 O2-B 变体若不以 capacity-matched control 为基线，增益将不可归因。
2. **O2-B 的第一性问题应改成"conditional selection 能否把 cross 证据用起来"**，
   而不是"cross 证据是否存在"。本轮已证明：静态无条件映射下，六个矩阵确实学到
   了差异化的 source→target 变换（§12 的不对称性），但差异化 ≠ 有用。
3. **需要 Null / Identity 对照**：既然 DUP 表明"任意静态 residual 都有效"，O2-B
   必须能区分"选择性地使用 cross 证据"与"再多一条通路"。建议 Null routing 作为
   DUP 之上的第二层控制。
4. **不要在本轮结果上加 repair**：没有 ownership 坍塌需要修（§14）。若 O2-B 引入
   条件化后仍无增益，再考虑 ownership-preserving 约束。
5. **单 seed 局限**：−0.69pp 虽超 ±0.3pp 噪声带约 2×，仍属单 seed 单 run；
   结论表述应保持"static 形式下无可靠增量"，而非"CROSS 有害"。

## 20. 声明

- **未读取 Test**：两条新 run 均 `task.evaluate_test=false`；全部诊断/ per-class 分析
  仅使用 val split（3334 节点）。
- **未做 tuning**：无超参 / lr / dropout / scheduler / seed / patience 修改；两条新 run
  的配置与 O1.5 parent 逐项一致（仅 `model.variant` + history/ckpt 路径）。
- **未进入 O2-B**；未实现 dynamic router / operator bank / Null routing / channel-wise
  routing / multi-scale / 任何新 loss（计划 §19 禁止清单全遵守）。
- 正式 checkpoint 选择维持 **best-Val-Acc**；§10 的 best-F1 行纯属诊断，禁止反向采用。

## 21. Commit

```text
git add src/models/oft_components.py src/models/oft_mag.py configs/model/oft_mag.yaml \
        tests/test_oft_mag.py scripts/oft_o2a_diagnostics.py scripts/oft_o1_5_val_per_class.py \
        experiments/oft/o2a docs/OFT_O2A_CROSS_UTILITY_REPORT.md
git commit -m "O2-A: test static cross-ownership graph utility"
git tag oft-o2a-static-cross
git push origin oft-mag --tags
```

| 项 | 值 |
|---|---|
| starting HEAD | `67bb0ab9f46924bf517428101fc159becb3c26d9`（tag `oft-o1-5`） |
| O2-A commit | `09c039f26c3dbbbe8985d48025035299f3862643` |
| O2-A tag | `oft-o2a-static-cross` → `09c039f` |
| push | `67bb0ab..09c039f  oft-mag -> oft-mag`；tags 已推送 |
| 本次补记 commit | 仅把上述 SHA 写入本报告（文档改动，无代码/实验变化） |

checkpoint `.pt`、运行日志、hydra 目录在 `outputs/`（gitignored），不入库。
