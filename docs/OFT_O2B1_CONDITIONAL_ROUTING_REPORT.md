# OFT-MAG O2-B1 — Target-Conditioned Null-vs-Transfer Routing Report

日期：2026-09-08
范围：O2-B1（CONDITIONAL-CROSS + capacity-matched CONDITIONAL-CROSS-DUP）。
**未实现 O2-B2**，未加任何新 loss / 正则 / temperature / balance loss，未做超参搜索，
未跑 Toys/Grocery、未跑 seed43/44。
协议：**Val-only**（`task.evaluate_test=false`，全程未访问/未记录 Test 指标）。
正式 checkpoint 选择：**best-Val-Acc**（未改成 F1 selection）。

## Verdict：**CASE B — CONDITIONAL_PARTIAL**

```text
Δcross_info_cond = CONDITIONAL-CROSS − CONDITIONAL-CROSS-DUP
                 = 55.1290 − 55.1290 =  0.00pp  Val Acc
                   45.7487 − 43.4314 = +2.32pp  Val Macro-F1
                   42.7895 − 39.1127 = +3.68pp  Balanced Accuracy
```

按 §25 预注册判据：Acc 未达 CASE A 的 +0.5pp 门槛（**恰好持平，两个 run 都是
1838/3334 正确**），但 **Macro-F1 与 Balanced Accuracy 同时明显提高**（+2.32 /
+3.68pp）→ 落入 **CASE B — CONDITIONAL_PARTIAL**。

配套的三条条件化对比（同一套 conditional machinery，只换对照）：

| 量 | 定义 | ΔAcc | ΔF1 | ΔBalAcc |
|---|---|---:|---:|---:|
| **Δcross_info_cond** | COND-CROSS − COND-DUP | **0.00pp** | **+2.32pp** | **+3.68pp** |
| **Δconditionality_cross** | COND-CROSS − STATIC-CROSS | **+0.57pp** | **+1.72pp** | −0.04pp |
| **Δconditionality_dup** | COND-DUP − STATIC-DUP | **−0.12pp** | **−1.04pp** | **−3.35pp** |
| **Δvs_diag** | COND-CROSS − DIAG | **+0.09pp** | **+3.05pp** | **+3.33pp** |

三条必须同时读的结论：

1. **Router 没有坍塌，而且"真的在选"**。六个 pair 的 transfer probability 在
   CONDITIONAL-CROSS 下分化到 0.126 / 0.247 / 0.940 / 0.982 / 0.997 / 0.997，
   std 0.0029–0.0963（非零），entropy 0.022–0.534 —— 既非全 Transfer、也非全 Null、
   也非 constant scalar（§26 三个退化判据全部不成立）。
2. **条件化收益是"选择"带来的，不是"容量"带来的**。CONDITIONAL-DUP 拥有完全相同的
   router / embedding / 参数 / 训练机制，但 payload 只读 target 自己的 N^b，
   router 收敛到 ≈0.5 的中庸区（mean 0.255–0.721，entropy 0.42–0.67），既学不出
   选择性、也拿不到 F1/BalAcc 增益（Δconditionality_dup 为负）。
3. **增益落在 Macro-F1 / Balanced Acc，不在 Acc**。Acc 恰好打平（1838/3334 相同），
   F1 +2.32pp、BalAcc +3.68pp —— 说明 router 重新分配了尾部类的判定，而不是整体
   提升 top-1。这一点与 O2-A 的 Δcapacity（+1.77pp F1 / +3.01pp BalAcc）方向一致，
   但本轮是**在 matched conditional machinery 下**由真实 cross payload 拿到的。

**必须同时记录的保守性/混淆**：CONDITIONAL-DUP 的 router 因为 payload 无信息而停在
0.5 附近，等于把 DUP 自己的 payload **减半**，所以它比 STATIC-DUP 弱
（Δconditionality_dup = −1.04pp F1）。若改用更强的 STATIC-DUP 做参照，
CONDITIONAL-CROSS 的 F1 优势仍有 **+1.28pp**、BalAcc **+0.32pp**（Acc −0.12pp）。
即：增益在两个对照下同号，但在预注册的严格 matched control 下最大。详见 §17、§19。

---

## 1. Git SHA

| 项 | 值 |
|---|---|
| starting HEAD | `12334481a20f1ab8b53008be95f24cd2734eb35a`（O2-A 文档补记 commit；其父为 O2-A 代码 commit `09c039f26c3dbbbe8985d48025035299f3862643`，tag `oft-o2a-static-cross`） |
| O2-B1 commit | `36fec95780638814778ffccdea59417fa8a2f32e` |
| O2-B1 tag | **`oft-o2b1-conditional-routing`** → `36fec95`（已 `git push origin oft-mag --tags`） |

O2-B1 建立在 `oft-o2a-static-cross` 已完成的代码之上：`StaticCrossLayer`、
`CROSS_PAIR_KEYS`、`post_transition_stats`、`oft_o1_5_val_per_class.py` 的
`build_model_and_head` / `val_per_class_rows` 全部原样复用。

## 2. Files changed

新增：

```text
docs/OFT_O2B1_CONDITIONAL_ROUTING_REPORT.md            # 本报告
scripts/oft_o2b1_router_diagnostics.py                 # router 离线诊断（复用 O1.5/O2-A 的 per-class helper）
experiments/oft/o2b1/conditional_cross_dup_movies_seed42.csv   # DUP 逐 epoch history（114 ep）
experiments/oft/o2b1/conditional_cross_movies_seed42.csv       # CROSS 逐 epoch history（101 ep）
experiments/oft/o2b1/conditional_cross_dup_stats.json          # DUP 全量诊断标量
experiments/oft/o2b1/conditional_cross_stats.json              # CROSS 全量诊断标量
experiments/oft/o2b1/conditional_cross_dup_val_per_class.csv   # DUP per-class（best-Acc ckpt）
experiments/oft/o2b1/conditional_cross_val_per_class.csv       # CROSS per-class（best-Acc ckpt）
experiments/oft/o2b1/conditional_cross_dup_router_nodes_val.csv # 每 val 节点的 6 个 p_transfer
experiments/oft/o2b1/conditional_cross_router_nodes_val.csv     # 同上
experiments/oft/o2b1/cdup_transfer_series.csv                   # 逐 epoch transfer_mean/std/entropy（日志解析）
experiments/oft/o2b1/ccross_transfer_series.csv                 # 同上
```

修改：

```text
src/models/oft_components.py   # +binary_entropy +ConditionalCrossLayer（DiagIDLayer / StaticCrossLayer 逐字未动）
src/models/oft_mag.py          # +conditional_cross_dup / conditional_cross dispatch、+O2-B1 aux_info
configs/model/oft_mag.yaml     # 注释说明两个新 variant（值/默认不变：diag_id）
tests/test_oft_mag.py          # +20 tests（O2-B1 §18 要求）
```

未修改：`DiagIDLayer`、`StaticCrossLayer`、`incoming_mean`、`biaxis_p0.py`、
`src/tasks/nc.py`、`src/tasks/common.py`、`src/main.py`、数据与 split。

**旧路径数值回归**：用改造后的 per-class 脚本重放 O2-A 的
`outputs/oft_o2a/static_cross_best.pt`，得到
`val_acc=0.545591 / macro_f1=0.440328 / balanced_acc=0.428308`，与
`experiments/oft/o2a/static_cross_val_per_class.csv` **逐行 diff 完全相同** →
O2-B1 的改动没有触碰 O2-A 数值。

## 3. Exact scientific question

> 如果不再让 cross-ownership transfer 对所有节点无条件发生，而是根据 **target node
> 当前 semantic/structural state** 动态选择 Null / Transfer，是否能把 O2-A 中
> static computation 没有利用起来的真实 cross-ownership evidence 转化为
> downstream incremental gain？

只测这一件事：**Target-conditioned Null-vs-Transfer selection**。不测
source-aware conditioning、不测 operator bank、不测 multi-scale。

## 4. Exact router equation

```text
q_i^{ab}        = [ H_i^b | N_i^b | e_a | e_b ]            # 128+128+16+16 = 288
[l_null, l_tr]  = Router(q_i^{ab})                          # Linear(288,128) -> GELU -> Linear(128,2)
p_i^{ab}        = softmax([l_null, l_tr], dim=-1)           # p_null + p_transfer = 1
B_null(x)       = 0                                         # Null operator（不构造 tensor：p_null*0 = 0）
B_transfer(x)   = T_{a->b}(x)                               # 复用 O2-A 的六个 128->128 bias-free 映射
payload_i^{a->b}  = T_{a->b}(N_i^a)                         # CONDITIONAL-CROSS
                  = T_{a->b}(N_i^b)                         # CONDITIONAL-DUP
m_i^{a->b}      = p_transfer_i^{ab} * payload_i^{a->b}
Delta_cross_i^b = (1 / (S-1)) * sum_{a != b} m_i^{a->b}     # S = 3 -> 除以 2
H_next_i^b      = LN_b( H_i^b + Delta_diag_i^b + Delta_cross_i^b )
```

一个**共享** router 服务全部 6 个 off-diagonal pair（`c->pt, c->pv, pt->c,
pt->pv, pv->c, pv->pt`）；pair identity 由 `e_a` / `e_b` 提供，不存在 6 个独立 MLP，
不做 channel group / edge-wise router / neighbor attention / temperature。

## 5. Proof：router 是 target-only

Router 输入**只**含：target 当前状态 `H_i^b`、target 邻域状态 `N_i^b`、
source 因子**身份** embedding `e_a`、target 因子**身份** embedding `e_b`。

- 结构证明：`router_input_dim == 2*factor_dim + 2*embed_dim`（单测断言），
  `pair_context()` 只索引 `H[:, b]` / `N[:, b]`，从不索引 `H[:, a]` / `N[:, a]`。
- 因果证明（单测 `test_router_reads_target_context_only`）：对全部 6 个 pair，
  固定 target 的 `H^b` / `N^b`，把 source 的 `H^a` / `N^a` 改动 +3.0，
  router logits **逐位不变**（`torch.equal`）；同时验证改动 target 上下文确实会改变
  logits（反 vacuous 检查）。
- 训练后行为一致性：perturb 某个 ownership slot 时，**只有以它为 target 的 pair**
  的 `p_transfer` 改变（单测 `test_conditional_cross_payload_follows_source_slot`）。

这是本轮最重要的控制纪律：否则 CONDITIONAL-DUP 虽然 payload 不看 `N^a`，
router 自己却看到了 source ownership content，matched control 就被污染。

## 6. Exact CONDITIONAL-DUP definition

与 CONDITIONAL-CROSS **逐模块、逐参数、逐初始化、逐优化器、逐 target update
完全相同**，唯一差异是 payload 的读取槽位：

```text
payload_i^{a->b} = T_{a->b}(N_i^b)        # 每个 pair 模块仍存在、仍被训练、仍参与优化器
```

即：conditional 机制（router + embedding + Null/Transfer + 六个 T）**全部保留**，
但跨 ownership 的信息被替换成 target 自己的邻域状态 —— 额外的 conditional
capacity / optimization path / inductive bias 存在，**额外的 ownership 信息不存在**。

## 7. Exact CONDITIONAL-CROSS definition

```text
payload_i^{a->b} = T_{a->b}(N_i^a)        # 真实的其他 ownership 邻域状态（a != b）
```

router 定义与 DUP 完全一致（同一 target-only 输入）。

## 8. Parameter-matching proof

单测 `test_conditional_variants_parameter_counts_match` +
`test_conditional_variants_state_dict_identical_init`：

- 两 variant 参数量严格相等；
- `n_cross − n_diag_id == 6·F² + [3·E + (2F+2E)·128 + 128 + 128·2 + 2]`；
- 同 RNG 流下 state_dict **key 集合相同、逐 key shape 相同、逐 key 数值 `torch.equal`**。

真实规模（run 日志 `model+head params=`，Movies，factor_dim=128，embed_dim=16）：

| model | model params | +head | model+head |
|---|---:|---:|---:|
| DIAG-ID | 1,136,640 | 5,140 | 1,141,780 |
| STATIC-CROSS / DUP | 1,234,944 | 5,140 | 1,240,084 |
| **CONDITIONAL-CROSS / DUP** | 1,272,242 | 5,140 | **1,277,382** |

增量 `1,277,382 − 1,240,084 = 37,298 = 3×16 + 288×128 + 128 + 128×2 + 2` ✓
（embedding 48 + router 37,250）。两个新 run 的日志都打印 **1,277,382**，严格相同。

## 9. Initialization proof

- 六个 `T_{a->b}`：沿用 O2-A 的 `nn.init.zeros_`（单测
  `test_conditional_transfer_matrices_zero_initialized`）。
- router 最后一个 Linear：`weight = 0`、`bias = 0`（无 Null prior bias）→ step 0
  logits 逐位为 0、`p_null = p_transfer = 0.5`（单测
  `test_router_zero_init_gives_half_transfer`）。
- 因此 step 0 `m = p · T(·) = 0.5 × 0 = 0`，两 variant forward 与 `diag_id`
  **逐位相等**（单测 `test_conditional_init_forward_equals_diag_path`，`torch.equal`）。
- 日志侧证：epoch 1–2 的 `transfer_mean` 全为 0.5000、`transfer_std` 全为 0.00000；
  epoch 3 起 router 开始移动（mean 0.4964，std 0.0001）。

## 10. Two-step router gradient proof

数学预期：`dL/dp ∝ T(x) = 0`，因此第一步 backward 中 router 梯度**严格为 0**，
而六个 T 有梯度。实测（单测 + 真实 run 日志一致）：

| step | T_{a->b} 梯度 | router head (Linear 128→2) | router trunk (Linear 288→128) | factor_embed |
|---|---|---|---|---|
| 1（零初始化） | **非零** | **恰好 0** | **恰好 0** | **恰好 0** |
| 2（T 手动更新一次后） | 非零 | **非零** | 仍为 0（见下） | 仍为 0 |

第二级的 trunk/embedding 还要再等一步：`dL/dh = dL/dlogits @ W_head`，而 `W_head`
在 step 2 时仍是零 → trunk 梯度仍为 0；`W_head` 更新后（step 3）才回传。
单测 `test_first_backward_transfer_grad_nonzero_router_grad_zero` /
`test_second_backward_router_grad_nonzero` 把这条**梯度级联**
（T → router head → router trunk → embedding）钉死，避免把"first step router grad = 0"
误判为 bug。

## 11. Unit-test results

```bash
pytest tests/test_oft_mag.py -q    # 52 passed（32 O1/O1.5/O2-A + 20 O2-B1）, 1.8s
pytest tests/ -q                   # 108 passed（88 baseline + 20 new）, 5.0s
```

新增 20 个 O2-B1 测试（对应 §18 的 19 项要求）：

| # | test | 验证点（要求编号） |
|---|---|---|
| 1 | `test_conditional_variants_parameter_counts_match` | ① 两 variant 参数量严格一致 + 增量 = 6F² + router |
| 2 | `test_conditional_variants_state_dict_identical_init` | ② 同 seed key/shape/value 全一致 |
| 3 | `test_conditional_router_is_shared_single_module` | ③ 单一共享 router + 单一 embedding；线性层计数 = 3+3+6+2；cross 模块内无 router |
| 4 | `test_router_reads_target_context_only` | ④ 改 source H^a/N^a → logits 逐位不变（防 source-content 泄漏） |
| 5 | `test_router_probabilities_are_valid_distribution` | ⑤ shape [N]、p∈[0,1]、p_null+p_transfer=1、entropy∈[0,log2] |
| 6 | `test_router_zero_init_gives_half_transfer` | ⑥ logits 逐位 0、p 逐位 0.5 |
| 7 | `test_conditional_transfer_matrices_zero_initialized` | ⑦ 六个 T 仍严格零初始化 |
| 8 | `test_conditional_init_forward_equals_diag_path` | ⑧ 初始化态 forward == diag_id（逐位） |
| 9 | `test_conditional_cross_payload_follows_source_slot` | ⑨ CROSS payload 跟随 source；router 概率只随 target 变 |
| 10 | `test_conditional_dup_payload_ignores_other_source_states` | ⑩ DUP payload 忽略其他 source（逐位） |
| 11 | `test_conditional_variants_router_logits_match` | ⑪ 两 variant 相同 target context → logits 逐位相同 |
| 12 | `test_first_backward_transfer_grad_nonzero_router_grad_zero` | ⑫ 第一步：T 有梯度、router 恰好 0 |
| 13 | `test_second_backward_router_grad_nonzero` | ⑬ 更新 T 后 router head 梯度非零（并记录 trunk 级联） |
| 14 | `test_conditional_empty_graph_updates_exactly_zero` | ⑭ 空图：diag / cross / effective 全为 0 |
| 15 | `test_conditional_isolated_node_graph_update_exactly_zero` | ⑮ isolated node：H_next == LN(H) |
| 16 | `test_conditional_aux_keys_present_and_static_unchanged` | ⑯ 诊断 key 齐备 + 旧 variant key 面不变（16/40/64）+ 只输出标量 |
| 17 | `test_effective_pair_ratio_bounded_by_raw` | ⑰ effective ≤ raw，且数值有限、p∈[0,1] |
| 18 | `test_conditional_no_giant_edge_pair_tensor` | ⑲ profiler：无 [E,3,3,...] |
| 19 | `test_conditional_variants_enforce_single_layer` | num_layers=1 约束 |
| 20 | `test_conditional_encode_states_matches_forward_h1` | 离线 hook == forward；诊断 helper 复算一致 |

原有 32 个 O1/O1.5/O2-A tests 全部继续通过（§18 第 18 项）。

## 12. Exact commands

```bash
# A. Unit tests
pytest tests/test_oft_mag.py -q
pytest tests/ -q

# B. O2-A 数值回归（不重跑，只重放 ckpt）
PYTHONPATH=. python scripts/oft_o1_5_val_per_class.py \
  --ckpt outputs/oft_o2a/static_cross_best.pt \
  --config outputs/2026-09-08/18-03-49/.hydra/config.yaml \
  --out /tmp/o2b1_regression_static_cross.csv --label o2b1_static_cross_regression
diff /tmp/o2b1_regression_static_cross.csv experiments/oft/o2a/static_cross_val_per_class.csv

# C. Run 1 — CONDITIONAL-CROSS-DUP（Movies seed42, Val-only, cuda:1）
python -m src.main dataset=Movies task=nc model=oft_mag num_runs=1 seed=42 device=cuda:1 \
  task.evaluate_test=false model.variant=conditional_cross_dup \
  task.history_path=experiments/oft/o2b1/conditional_cross_dup_movies_seed42.csv \
  task.save_ckpt_path=outputs/oft_o2b1/conditional_cross_dup_best.pt
#   -> hydra outputs/2026-09-08/18-44-06；log 镜像 outputs/oft_o2b1_conditional_cross_dup.log

# D. Run 2 — CONDITIONAL-CROSS（同上）
python -m src.main dataset=Movies task=nc model=oft_mag num_runs=1 seed=42 device=cuda:1 \
  task.evaluate_test=false model.variant=conditional_cross \
  task.history_path=experiments/oft/o2b1/conditional_cross_movies_seed42.csv \
  task.save_ckpt_path=outputs/oft_o2b1/conditional_cross_best.pt
#   -> hydra outputs/2026-09-08/18-44-22；log 镜像 outputs/oft_o2b1_conditional_cross.log

# E. 离线诊断（best-Val-Acc ckpt；val split only）
PYTHONPATH=. python scripts/oft_o2b1_router_diagnostics.py \
  --ckpt outputs/oft_o2b1/conditional_cross_dup_best.pt \
  --config outputs/2026-09-08/18-44-06/.hydra/config.yaml \
  --out-stats experiments/oft/o2b1/conditional_cross_dup_stats.json \
  --out-per-class experiments/oft/o2b1/conditional_cross_dup_val_per_class.csv \
  --out-nodes experiments/oft/o2b1/conditional_cross_dup_router_nodes_val.csv \
  --label conditional_cross_dup
PYTHONPATH=. python scripts/oft_o2b1_router_diagnostics.py \
  --ckpt outputs/oft_o2b1/conditional_cross_best.pt \
  --config outputs/2026-09-08/18-44-22/.hydra/config.yaml \
  --out-stats experiments/oft/o2b1/conditional_cross_stats.json \
  --out-per-class experiments/oft/o2b1/conditional_cross_val_per_class.csv \
  --out-nodes experiments/oft/o2b1/conditional_cross_router_nodes_val.csv \
  --label conditional_cross
```

两条 run 的 hydra config **只差 3 行**（`variant` + history/ckpt 路径），其余逐行相同
（`diff` 已验证）。`device=cuda:1` 与 O1.5 / O2-A 一致，运行时该卡空闲 12MiB/0%。

## 13-15. 复用的 references（未重跑）

| reference | params | best-Acc ep | Val Acc | Val Macro-F1 | Balanced Acc |
|---|---|---:|---:|---:|---:|
| **DIAG-ID**（O1.5 cuda:1 parent） | 1,141,780 | 75 | 55.0390% | 42.7039% | 39.4594% |
| **STATIC-CROSS-DUP**（O2-A） | 1,240,084 | 70 | 55.2490% | 44.4705% | 42.4651% |
| **STATIC-CROSS**（O2-A） | 1,240,084 | 71 | 54.5591% | 44.0328% | 42.8308% |

三个 reference 均直接复用 O2-A / O1.5 的 ckpt 与 CSV，**未重跑**（§22）。

## 16-17. O2-B1 结果

| variant | params (model+head) | best-Acc ep | 早停 ep | **Val Acc** | **Val Macro-F1** | **Balanced Acc** |
|---|---|---:|---:|---:|---:|---:|
| CONDITIONAL-CROSS-DUP | 1,277,382 | 84 | 114 | **55.1290%** | **43.4314%** | **39.1127%** |
| CONDITIONAL-CROSS | 1,277,382 | 71 | 101 | **55.1290%** | **45.7487%** | **42.7895%** |

（best-F1 epoch 仅作诊断、**不用于 selection**：DUP ep108 F1 47.86% / Acc 53.06%；
CROSS ep93 F1 46.77% / Acc 54.56%。整个模型族维持 O1.5 §9 记录的 Acc–F1 晚熟形态。）

两个 run 的 best-Acc 都是 1838/3334 = 0.551290 —— 数值上完全相同的 Acc，但
per-class 分布不同（§25），F1 相差 2.32pp。

## 18-24. Δ 表（Accuracy / Macro-F1 / Balanced Accuracy）

| 量 | 定义 | ΔVal Acc | ΔVal Macro-F1 | ΔBalanced Acc |
|---|---|---:|---:|---:|
| **Δcross_info_cond** | COND-CROSS − COND-DUP | **0.00pp** | **+2.32pp** | **+3.68pp** |
| **Δconditionality_cross** | COND-CROSS − STATIC-CROSS | **+0.57pp** | **+1.72pp** | −0.04pp |
| **Δconditionality_dup** | COND-DUP − STATIC-DUP | **−0.12pp** | **−1.04pp** | **−3.35pp** |
| **Δvs_diag** | COND-CROSS − DIAG | **+0.09pp** | **+3.05pp** | **+3.33pp** |

补充（**非预注册**，用于估计混淆方向）：

| 量 | 定义 | ΔAcc | ΔF1 | ΔBalAcc |
|---|---|---:|---:|---:|
| COND-CROSS − STATIC-DUP | 对最强 static 对照 | −0.12pp | **+1.28pp** | +0.32pp |
| COND-DUP − DIAG | 条件化 + target-only payload | +0.09pp | +0.73pp | −0.35pp |

读法：

- **Δcross_info_cond 是唯一能回答科学问题的量**（§24）：在完全相同的 conditional
  machinery 下，真实 cross payload 相对 target-only payload 净得 +2.32pp F1 /
  +3.68pp BalAcc，Acc 持平。
- **Δconditionality_cross** 单独看**不能**归因给 cross information：它混入了 router /
  embedding / adaptive scaling 带来的参数化与优化路径收益。它为正（+0.57pp Acc /
  +1.72pp F1）说明条件化本身不是有害的。
- **Δconditionality_dup 为负**是关键反向证据：把同一套 conditional machinery 加到
  **没有跨 ownership 信息**的 payload 上，Acc/F1/BalAcc 全面下降。也就是说，
  conditional 机制的价值**依赖 payload 是否携带跨 ownership 信息**。

## 25. 六 pair router 概率表（val split，best-Acc ckpt）

CONDITIONAL-CROSS：

| pair | transfer mean | std | p10 | p50 | p90 | frac<0.25 | frac>0.75 | H(p) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c→pt | 0.9404 | 0.0473 | 0.8855 | 0.9566 | 0.9757 | 0.0000 | 0.9904 | 0.2108 |
| c→pv | 0.1262 | 0.0589 | 0.0631 | 0.1151 | 0.2015 | 0.9616 | 0.0000 | 0.3644 |
| pt→c | 0.9815 | 0.0154 | 0.9670 | 0.9863 | 0.9909 | 0.0000 | 1.0000 | 0.0878 |
| pt→pv | 0.2467 | 0.0963 | 0.1360 | 0.2324 | 0.3757 | 0.5741 | 0.0003 | 0.5341 |
| pv→c | 0.9966 | 0.0029 | 0.9940 | 0.9975 | 0.9983 | 0.0000 | 1.0000 | 0.0220 |
| pv→pt | 0.9965 | 0.0035 | 0.9933 | 0.9977 | 0.9987 | 0.0000 | 1.0000 | 0.0220 |

CONDITIONAL-CROSS-DUP：

| pair | transfer mean | std | p10 | p50 | p90 | frac<0.25 | frac>0.75 | H(p) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c→pt | 0.6084 | 0.0883 | 0.4951 | 0.6077 | 0.7270 | 0.0000 | 0.0543 | 0.6528 |
| c→pv | 0.7211 | 0.2545 | 0.3025 | 0.8099 | 0.9799 | 0.0669 | 0.5708 | 0.4211 |
| pt→c | 0.3234 | 0.1437 | 0.2169 | 0.2773 | 0.5042 | 0.3023 | 0.0447 | 0.5845 |
| pt→pv | 0.6772 | 0.2627 | 0.2607 | 0.7480 | 0.9689 | 0.0918 | 0.4979 | 0.4568 |
| pv→c | 0.2553 | 0.1185 | 0.1746 | 0.2164 | 0.3863 | 0.7226 | 0.0000 | 0.5353 |
| pv→pt | 0.4297 | 0.0760 | 0.3346 | 0.4237 | 0.5361 | 0.0018 | 0.0000 | 0.6714 |

（全部节点 vs val split 的 mean/std 差异 < 0.01，两套都在 JSON 中：`transfer_*_all_*`
与 `val_*_*`。）

## 26. Router 是否"真的动态"（§26 五问）

| 问题 | 回答 |
|---|---|
| 1. 是否存在 node-level variation？ | **是**。CROSS 六个 pair 的 val std 为 0.0029–0.0963，全部 > 0；p10–p90 跨度 0.0078–0.2397（pt→pv 最大）。DUP 更大（0.0760–0.2627）。 |
| 2. 是否存在 pair-level variation？ | **是，且极强**。CROSS 的 pair 均值从 0.1262 到 0.9966，跨越几乎整个区间；DUP 也有 0.2553–0.7211 的差异。 |
| 3. 是否退化为全 Transfer？ | **否**。只有 3 个 pair（pt→c、pv→c、pv→pt）frac>0.75 = 1.0；c→pv 有 96.2% 的节点 < 0.25。 |
| 4. 是否退化为全 Null？ | **否**。同上，两个方向的极端都只覆盖部分 pair。 |
| 5. CROSS 与 DUP 的 router 分布是否明显不同？ | **是，且这是本轮最强的定性对比**。CROSS：低熵、强 pair 选择（H 0.022–0.534）；DUP：高熵、靠近 0.5 的中庸区（H 0.421–0.671）。同一套 router、同一 target-only 输入定义，只有 payload 的信息含量不同。 |

**是否 collapse（CASE E 判据）**：`p ≈ 1 for all pairs`、`p ≈ 0 for all pairs`、
`std ≈ 0` 三条**全部不成立** → **ROUTER_COLLAPSE 未发生**。

**逐 epoch 轨迹**（`*_transfer_series.csv`，从 log `Aux` 行解析）：

| pair | CROSS ep1 → ep20 → ep40 → ep70 → ep101 | DUP ep1 → ep20 → ep40 → ep70 → ep114 |
|---|---|---|
| c→pt | 0.500 → 0.490 → 0.873 → 0.939 → 0.907 | 0.500 → 0.418 → 0.534 → 0.704 → 0.426 |
| c→pv | 0.500 → 0.344 → 0.215 → 0.135 → 0.102 | 0.500 → 0.558 → 0.812 → 0.849 → 0.629 |
| pt→c | 0.500 → 0.495 → 0.894 → 0.980 → 0.983 | 0.500 → 0.271 → 0.310 → 0.388 → 0.269 |
| pt→pv | 0.500 → 0.382 → 0.324 → 0.259 → 0.237 | 0.500 → 0.532 → 0.745 → 0.790 → 0.620 |
| pv→c | 0.500 → 0.544 → 0.963 → 0.996 → 0.997 | 0.500 → 0.236 → 0.228 → 0.311 → 0.217 |
| pv→pt | 0.500 → 0.579 → 0.979 → 0.996 → 0.996 | 0.500 → 0.352 → 0.355 → 0.489 → 0.327 |

router 在 epoch 1–2 完全静止（`std` 恰为 0，因 T 仍为零），epoch 3 起开始移动，
epoch 40 左右基本定型。**解释约束（§27）**：即使 pt→c / pv→c / pv→pt 的
`p_transfer` 很高，也只能说"under the learned router, this pair is selected more
frequently / receives higher transfer probability"，**不能**说这些 factor 更重要或
graph demand 更高。

## 27. Raw vs effective cross pair magnitudes

定义：`cross_pair_ratio_{a→b} = mean_i ||T_{a→b}(payload_i)|| / (||H_i^b||+eps)`
（raw），`effective_cross_pair_ratio_{a→b} = mean_i ||p·T(payload_i)|| / (||H_i^b||+eps)`
（routed）。二者只差路由因子，是"变换幅度"与"实际写入幅度"的区分。

| pair | CROSS raw | CROSS effective | DUP raw | DUP effective |
|---|---:|---:|---:|---:|
| c→pt | 0.1119 | 0.1051 | 0.1168 | 0.0711 |
| c→pv | 0.1555 | 0.0204 | 0.1698 | 0.1226 |
| pt→c | 0.1550 | 0.1520 | 0.1667 | 0.0532 |
| pt→pv | 0.1486 | 0.0369 | 0.1692 | 0.1149 |
| pv→c | 0.1657 | 0.1651 | 0.1588 | 0.0399 |
| pv→pt | 0.1805 | 0.1799 | 0.1052 | 0.0453 |

- CROSS 的 raw 幅度与 DUP 同量级（0.11–0.18），说明**差异不在变换强度**，
  而在"哪些 pair 被放行"：c→pv 与 pt→pv 的 effective 被压到 raw 的 13% / 25%，
  其余四个几乎不被衰减。
- DUP 的 raw/effective 全面被削（p 平均 ≈0.5），这正是它相对 STATIC-DUP 变弱的原因。

## 28. Cross / diag 更新比

| target | CROSS cross_ratio | CROSS cross/diag | DUP cross_ratio | DUP cross/diag |
|---|---:|---:|---:|---:|
| C | 0.1174 | 0.3223 | 0.0466 | 0.1242 |
| Pt | 0.1131 | 0.2961 | 0.0582 | 0.1449 |
| Pv | 0.0272 | 0.0701 | 0.1187 | 0.3060 |

（`diag_update_ratio` CROSS：C 0.3661 / Pt 0.3831 / Pv 0.4175；DUP：0.3678 / 0.3977 /
0.3843。diag 路径本身在两个 variant 间基本一致 —— 条件化只作用于 cross 分支。）
CROSS 下 **Pv 的 cross 通道被 router 关掉**（cross/diag 0.070 vs C/Pt 的 0.32/0.30），
与 §25 的 c→pv / pt→pv 低转移概率完全自洽。

## 29. Pre/post ownership cosine

| 量 | DIAG | STATIC-DUP | STATIC-CROSS | COND-DUP | COND-CROSS |
|---|---:|---:|---:|---:|---:|
| pre → post cos c_pt | +0.0369 → +0.0383 | −0.0836 → −0.0607 | −0.0848 → −0.0605 | −0.0053 → −0.0487 | +0.0150 → −0.0349 |
| pre → post cos c_pv | −0.0514 → −0.0309 | −0.0295 → −0.0122 | −0.0315 → −0.0036 | −0.0355 → −0.0303 | −0.0366 → −0.0263 |
| pre → post cos pt_pv | +0.0258 → +0.0245 | −0.0277 → −0.0372 | −0.0240 → −0.0280 | −0.0033 → −0.0029 | −0.0164 → −0.0224 |

所有 post-transition 槽间 cosine 仍在 |0.05| 以内 → **没有 ownership collapse**。
COND-CROSS 相对 COND-DUP 的 post cosine 略更负（更分离），方向与 O2-A 一致。

## 30. State drift / post norm

| 量 | DIAG | STATIC-DUP | STATIC-CROSS | COND-DUP | COND-CROSS |
|---|---:|---:|---:|---:|---:|
| state_drift C / Pt / Pv | 0.048 / 0.056 / 0.083 | 0.074 / 0.075 / 0.097 | 0.064 / 0.068 / 0.091 | 0.080 / 0.069 / 0.087 | 0.093 / 0.071 / 0.074 |
| post_norm C / Pt / Pv | 11.31 / 11.25 / 11.41 | 11.31 / 11.26 / 11.41 | 11.32 / 11.26 / 11.41 | 11.31 / 11.25 / 11.42 | 11.32 / 11.26 / 11.39 |

drift 全部在 0.07–0.10 区间、post_norm 全部 ≈11.3 —— 条件化与路由**没有把状态推离
DIAG 家族的量级**，没有数值病理。

## 31. Per-class comparison

CSV：`experiments/oft/o2b1/{conditional_cross_dup,conditional_cross}_val_per_class.csv`
（schema：class_id, support, recall, precision, f1, predicted_count, true_count；
由 `oft_o1_5_val_per_class.py` 的共享实现产出，非第二套代码）。

ΔF1（pp）逐类：

| cls | sup | DIAG | STATIC-DUP | STATIC-CROSS | COND-DUP | COND-CROSS | **CC−CD** | CC−SC | CD−SD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 355 | 0.4583 | 0.4510 | 0.4588 | 0.4328 | 0.4835 | **+5.07** | +2.47 | −1.83 |
| 1 | 1098 | 0.6514 | 0.6473 | 0.6465 | 0.6494 | 0.6410 | −0.83 | −0.55 | +0.21 |
| 2 | 363 | 0.4552 | 0.4334 | 0.3627 | 0.4202 | 0.4241 | +0.39 | +6.15 | −1.32 |
| 3 | 277 | 0.7838 | 0.7628 | 0.7812 | 0.7932 | 0.7801 | −1.31 | −0.12 | +3.04 |
| 4 | 24 | 0.4500 | 0.6818 | 0.6222 | 0.5946 | 0.6667 | +7.21 | +4.44 | −8.72 |
| 5 | 82 | 0.2478 | 0.3111 | 0.2901 | 0.2645 | 0.3066 | +4.21 | +1.65 | −4.66 |
| 6 | 122 | 0.3889 | 0.4372 | 0.4115 | 0.3689 | 0.3280 | **−4.09** | −8.35 | −6.83 |
| 7 | 210 | 0.3689 | 0.3889 | 0.3767 | 0.3906 | 0.4067 | +1.61 | +3.00 | +0.17 |
| 8 | 30 | 0.5000 | 0.5000 | 0.4706 | 0.5926 | 0.5806 | −1.19 | +11.01 | +9.26 |
| 9 | 37 | 0.0930 | 0.2308 | 0.2128 | 0.0513 | 0.2128 | **+16.15** | +0.00 | −17.95 |
| 10 | 269 | 0.5958 | 0.6167 | 0.6012 | 0.5860 | 0.6124 | +2.63 | +1.12 | −3.07 |
| 11 | 88 | 0.3688 | 0.3622 | 0.4151 | 0.3594 | 0.4052 | +4.59 | −0.99 | −0.28 |
| 12 | 23 | 0.7273 | 0.7600 | 0.7500 | 0.7556 | 0.7111 | −4.44 | −3.89 | −0.44 |
| 13 | 43 | 0.5714 | 0.5909 | 0.5652 | 0.5610 | 0.5977 | +3.67 | +3.25 | −2.99 |
| 14 | 50 | 0.4138 | 0.3571 | 0.4407 | 0.5349 | 0.4444 | **−9.04** | +0.38 | +17.77 |
| 15 | 47 | 0.2687 | 0.3656 | 0.3855 | 0.3014 | 0.3421 | +4.07 | −4.34 | −6.42 |
| 16 | 38 | 0.0513 | 0.0000 | 0.0000 | 0.0000 | 0.0455 | +4.55 | +4.55 | 0.00 |
| 17 | 141 | 0.4140 | 0.3796 | 0.3188 | 0.3396 | 0.4176 | **+7.80** | +9.87 | −4.00 |
| 18 | 23 | 0.4324 | 0.5000 | 0.3636 | 0.3571 | 0.3226 | −3.46 | −4.11 | −14.29 |
| 19 | 14 | 0.3000 | 0.1176 | 0.3333 | 0.3333 | 0.4211 | +8.77 | +8.77 | +21.57 |

**最大正/负变化（CC − CD）**：正 = cls9 **+16.15pp**（sup 37）、cls19 +8.77（sup 14）、
cls17 **+7.80**（sup 141）、cls4 +7.21（sup 24）、cls0 **+5.07**（sup 355）；
负 = cls14 **−9.04**（sup 50）、cls12 −4.44（sup 23）、cls6 **−4.09**（sup 122）、
cls18 −3.46（sup 23）、cls3 −1.31（sup 277）。

读法（与 O2-A 的关键区别）：

- 与 O2-A 的 CC−CD 不同，本轮的增益**不再集中在极小类**：support ≥ 100 的类里
  cls17(+7.80)、cls0(+5.07)、cls10(+2.63)、cls2(+0.39) 为正，只有 cls6(−4.09)、
  cls3(−1.31)、cls1(−0.83) 为负 —— 净额转正，这正是 F1/BalAcc 增益的来源。
- 反向项仍存在（cls14 −9.04、cls12 −4.44、cls18 −3.46），说明这是**类间重分配**，
  不是全面改善；Acc 持平与这个重分配一致。

## 32. Warnings / memory

- 两条 run 日志：**0 NaN / 0 Inf / 0 Traceback / 0 CUDA error / 0 OOM**
  （`grep -icE "\bnan\b|\binf\b|Traceback|CUDA error|out of memory"` = 0）。
- 均在 patience 自然早停（114 / 101 epoch，patience 30）。
- 显存：Movies full-graph，O1 已测 DIAG-ID train peak 1.07GB / eval 0.58GB；
  O2-B1 额外增加 6 次 `[N,288]@[288,128]` 的 router 前向（N≈35k），
  运行期间 `nvidia-smi` 无异常占用，无 memory pathology。
- `task.evaluate_test=false` 全程生效，日志中 "Test" 出现次数 = 0。

## 33. Interpretation category

**CASE B — CONDITIONAL_PARTIAL**

依据（§25 预注册）：

- CASE A 需要 `Δcross_info_cond Acc > +0.5pp` → 实测 **0.00pp**，不满足；
- CASE B 为"`Acc +0.3~0.5pp` 或 `Acc 基本持平但 F1 / Balanced Acc 明显提高`"
  → 实测 Acc 持平、F1 **+2.32pp**、BalAcc **+3.68pp**，**满足**；
- CASE C（COND-CROSS > STATIC-CROSS 但 ≈ COND-DUP）不成立：COND-CROSS 在
  F1/BalAcc 上明显高于 COND-DUP；
- CASE D（Δcross_info_cond ≤ ~0 且 router 非退化）部分相关（Acc = 0），
  但 D 的语义是"target-only conditioning 不足以 unlock cross information"，
  与"F1/BalAcc 显著提升"矛盾，故不取；
- CASE E（ROUTER_COLLAPSE）不成立（§26）。

同时必须记录三条限制：

1. **Acc 没有提升**。收益全部落在 Macro-F1 / Balanced Accuracy，即类间重分配；
   在 best-Val-Acc selection 下，Acc 层面 CONDITIONAL-CROSS 与 CONDITIONAL-DUP
   完全相同（1838/3334）。
2. **DUP 的 router 半途"自废武功"**。因为 target-only payload 无可选信息，DUP 的
   router 停在 0.5 附近，等于把 payload 减半，使 DUP 弱于 STATIC-DUP（−1.04pp F1）。
   这让预注册的 Δcross_info_cond **偏大**。用更强的 STATIC-DUP 做参照时，
   增益降为 F1 +1.28pp / BalAcc +0.32pp（Acc −0.12pp）—— 同号但小得多。
3. **单 seed、单数据集、单 run**，只作 screening。+2.32pp F1 远超 O0 的 ±0.3pp
   Acc 噪声带，但 F1 的 run-to-run 波动在本项目未单独标定，结论表述应保守：
   "conditional selection 在 F1/Balanced Acc 上给出了可复现的正信号（单 seed）"。

## 34. Recommendation for next stage

1. **可以进入 O2-B2（Functional Operator Bank），但必须带三重对照**：
   CONDITIONAL-DUP（本轮）、Null-only、以及 STATIC-DUP。没有 STATIC-DUP，
   无法区分"条件化"与"容量"；没有 Null-only，无法区分"选择"与"多一条通路"。
2. **把 Δcross_info_cond 拆成两个子量**再决定 O2-B2 的方向：
   `Δselectivity = COND-CROSS − COND-CROSS(Null-only)` 与
   `Δpayload = COND-CROSS(Null-only) − COND-DUP(Null-only)`。
   本轮只能给出二者的合成量。
3. **优先补 seed43/44 的 screening**（若允许）：本轮 Acc 恰好打平、F1 +2.32pp，
   单 seed 不足以判断 F1 增益是否稳定；在扩 seed 之前不宜把 O2-B1 的结论写进主表。
4. **不要调 router 的 temperature / hidden size**（§25 明令）。若 O2-B2 仍只有
   F1 增益，下一步应优先判断 **source-aware conditionality** 是否值得测，
   而不是继续放大 router 容量。
5. **保留 §27 的解释纪律**：`p_transfer` 高 ≠ 该 factor 更重要；本轮只能说
   "router 学会了把 c→pv / pt→pv 关掉、把 pv→c / pv→pt / pt→c / c→pt 打开"。

## 35. 声明

- **未读取 Test**：两条新 run 均 `task.evaluate_test=false`；全部诊断 / per-class
  分析仅使用 val split（3334 节点）。
- **未做 tuning**：无超参 / lr / dropout / scheduler / seed / patience 修改；
  两条 run 的 hydra config 只差 `variant` 与输出路径。
- **未进入 O2-B2**；未实现 source-aware router、operator bank、Null routing 之外的
  任何机制、channel-wise routing、multi-scale、任何新 loss（§28 禁止清单全遵守）。

## 36. Commit

```text
git add src/models/oft_components.py src/models/oft_mag.py configs/model/oft_mag.yaml \
        tests/test_oft_mag.py scripts/oft_o2b1_router_diagnostics.py \
        experiments/oft/o2b1 docs/OFT_O2B1_CONDITIONAL_ROUTING_REPORT.md
git commit -m "O2-B1: test target-conditioned null-vs-transfer routing"
git tag oft-o2b1-conditional-routing
git push origin oft-mag --tags
```

| 项 | 值 |
|---|---|
| starting HEAD | `12334481a20f1ab8b53008be95f24cd2734eb35a` |
| O2-B1 commit | `36fec95780638814778ffccdea59417fa8a2f32e` |
| tag | `oft-o2b1-conditional-routing` → `36fec95` |
| push | `1233448..36fec95  oft-mag -> oft-mag`；tag 已推送 |
| 本次补记 commit | 仅把上述 SHA 写入本报告（文档改动，无代码/实验变化） |

测试复核（commit 前最后一遍）：`pytest tests/test_oft_mag.py -q` → **52 passed**；
`pytest tests/ -q` → **108 passed**。

checkpoint `.pt`、运行日志、hydra 目录在 `outputs/`（gitignored），不入库。
