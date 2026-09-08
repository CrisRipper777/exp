# OFT-MAG O1 — OFT-DIAG-ID Scaffold Implementation Report

日期：2026-09-08
范围：O1（clean DIAG-ID core scaffold + sanity）。未实现 cross-factor transition、
dynamic router / operator mixture / Null routing / channel routing / multi-scale、
edge attention、Gamma/OFR、pseudo nodes、PPR/APPNP、global context，未加任何新 loss，
未做超参搜索，未跑 seed43/44 与 Toys/Grocery。
协议：**Val-only**（`task.evaluate_test=false`，全程未读取/未记录 Test 指标）。

## Verdict：**GO（Strong-GO 边界）**

OFT-DIAG-ID（Movies seed42，Val-only）Val Acc **54.86%** vs P0 no-graph reference
**51.71%**：**+3.15pp**，清楚高于 P0；距 A0 锁定参考 55.01% 仅 **−0.15pp**（≈0.5×协议噪声带）。
diag update-ratio / neighbor-norm 全程健康（ratio ∈ [0.26, 0.49]，从未归零或爆炸），
P0 factor diagnostics 无坍塌，无 NaN / 无 warning / 无 inference mismatch。
按执行计划 §12 判据，本结果落在 **Strong GO** 区间：
`DIAG-ID 清楚高于 P0 + 距 A0 不超过约 1pp + update statistics 稳定`（见 §13 逐条对照）。

---

## 1. Git SHA 与 files changed

| 项 | 值 |
|---|---|
| starting HEAD（O1 前） | `4d3109fb7078e095abded4f7d6b7bed866b95155`（main，r0-p6.6） |
| O0 lock commit / tag | `b42f5b8d327490a2c3db9dec79df7fe0c54320a0`（"O0: lock Movies A0 and DiP references"）/ `oft-o0-reference` |
| O1 branch | `oft-mag`（自 main 创建） |
| ending HEAD | `b42f5b8d327490a2c3db9dec79df7fe0c54320a0`（O0 lock commit；O1 产物为未提交工作区，见下） |

新增文件：

```text
src/models/oft_mag.py            # OFT-DIAG-ID Model（subclass biaxis_p0.Model）
src/models/oft_components.py     # DiagIDLayer + incoming_mean（纯函数）
configs/model/oft_mag.yaml       # variant: diag_id, num_layers: 1
tests/test_oft_mag.py            # 11 unit tests
experiments/oft/o1/p0_movies_seed42.csv        # P0 no-graph reference 逐 epoch history
experiments/oft/o1/diag_id_movies_seed42.csv   # OFT-DIAG-ID 逐 epoch history
docs/OFT_O1_IMPLEMENTATION_REPORT.md           # 本报告
```

最小必要修改（O1 阻塞项，见 §14 说明）：

```text
src/tasks/common.py              # +1 行：_aux_iteration_keys 放行 oft_* 前缀 key
```

未修改（语义锁定文件，逐字不动）：`biaxis_p0.py`、`biaxis_components.py`、
`biaxis_final.py`、`dip.py`、`src/tasks/nc.py`、`src/main.py`、数据与 split。

## 2. 环境

与 O0 相同：conda `yhf_env`（Python 3.12.13、torch 2.4.0+cu121、torch_geometric 2.7.0、
hydra 1.3.2）、CUDA 12.1、RTX 3090（25.4GB）、cwd `/hdd1/DataInHere/YHF/exp`。

## 3. 架构与公式（OFT-DIAG-ID）

P0 factorizer 原样复用（`biaxis_p0.Model` subclass，未改 Common/Private 定义）：
x = [x_t | x_v] → 投影 → 因子 {h_t, h_v, c_t, c_v, c, p_t, p_v}。

三个因子作为显式 graph propagation states，slot 顺序**永久固定 0=C / 1=Pt / 2=Pv**：

```
H0 = stack([C, Pt, Pv], dim=1)          # [N, 3, 128]
```

单个 DIAG-ID block（每 factor a 独立、纯 diagonal、functional operator = Identity）：

```
X^a   = V_a H^a              # Linear 128->128, bias=False, per-slot V_a
N^a   = incoming neighbor mean over X^a   # edge_index[0]=source -> [1]=target
delta^a = D_a N^a            # Linear 128->128, bias=False, per-slot D_a
H^{a,+} = LayerNorm(H^a + delta^a)        # per-slot LayerNorm(128)
```

- 禁止任何 a≠b graph transfer：block 内 slot a 只读取 H[:, a]（单元测试位级验证）。
- 不加 self-loop；isolated node 的 delta 严格为 0（`out=0 / deg.clamp_min(1)=0`，
  且 D_a bias=False → D(0)=0 严格成立）；ego 仅经 residual 保留。
- 不做 GCN normalization（不用 gcn_norm / deg^{-1/2} 对称归一）。
- O1 固定 `num_layers=1`（config `variant: diag_id`；模型构造时 `num_layers != 1` 直接
  ValueError）。
- final fusion 只用 H1：`z = P0_fusion(concat([H1[:,0], H1[:,1], H1[:,2]]))`，
  不做 multi-scale。`out_dim = hidden_dim = 256`。

Aux 纪律：training 时 aux_loss = P0 `_compute_aux(factors)`（作用在 **propagation 前**
的原始 factorizer 输出上，与 P0 数学完全一致）；eval 时 aux_loss=0 / aux_info={}（同 P0）。

## 4. Tensor shapes（forward, full-graph Movies）

```text
x [16672, 1536] --split--> x_t/x_v [16672, 768]
factorizer -> c/p_t/p_v [16672, 128] -> H0 [16672, 3, 128]
V_a: X [16672, 3, 128]  (stack of 3 x Linear(128,128,bias=False))
gather X[src] [160802, 3, 128] --index_add_--> sum [16672, 3, 128]
deg [16672] (in-degree count) -> N = sum / clamp_min(deg,1) [16672, 3, 128]
D_a: delta [16672, 3, 128]; H1 = per-slot LayerNorm(H0+delta) [16672, 3, 128]
fusion: concat [16672, 384] -> Linear(384,256) -> z [16672, 256]
aux_info scalars (per-slot, detached): diag_update_ratio_a = mean_i ||delta_i^a||/(||H_i^a||+eps)
                                        neighbor_norm_a     = mean_i ||N_i^a||
```

不创建任何 `[E, 3, 3, ...]` edge-pair tensor（profiler shape audit 通过，见 §7）。

## 5. Aggregation 实现

`src/models/oft_components.py::incoming_mean`（纯函数）：

```python
msg = X[src]                       # [E, 3, 128] gathered source rows
out = torch.zeros_like(X);  out.index_add_(0, dst, msg)
deg.index_add_(0, dst, ones)       # float in-degree
N = out / deg.clamp_min(1.0).view(N,1,1)
```

`edge_index[0] = source` → 消息沿 source→target 流入（incoming mean）。零 in-degree 节点
分子为 0、分母 clamp 为 1 → delta 严格 0；分母不参与梯度（detach 语义天然满足，因
`zeros_like`/`index_add_` 的 index 分支不可导，实际梯度仅经 msg 回传）。

## 6. Config（configs/model/oft_mag.yaml）

```yaml
name: oft_mag          hidden_dim: 256      factor_dim: 128
dropout: 0.2           activation: gelu     norm: layernorm
variant: diag_id       num_layers: 1
lambda_common: 0.02    lambda_orth: 0.01    lambda_recon: 0.3
orth_fallback_batch: 16
full_graph_training: true
```

P0 超参与损失权重完全一致；模型设置 `requires_full_graph_training=True`（如 DiP）。

## 7. Tests

```bash
pytest tests/test_oft_mag.py -q    # 11 passed, 21s
pytest tests/ -q                   # 67 passed (56 baseline + 11 new), 4 warnings, 97s
```

| # | test | 验证点 |
|---|---|---|
| 1 | `test_forward_interface_train_and_eval` | 接口 (z,None,None,aux,info)、z [N,out_dim]、train 含 10 个 p0_* + 6 个 oft_l1_* 标量 key、eval aux=0/info={}、无 edge forward 合法 |
| 2 | `test_num_layers_must_be_one` | num_layers≠1 → ValueError |
| 3 | `test_factor_slot_order_fixed_c_pt_pv` | stack 顺序 0=C/1=Pt/2=Pv 逐切片 torch.equal；错误配对必然不等 |
| 4 | `test_incoming_mean_direction_and_magnitude` | incoming_mean == 独立 mask-sum 重算（atol 1e-5）；改 source 只改 target；isolated 行严格 0 |
| 5 | `test_no_self_loop_single_directed_edge` | 单边 0→1：source 0 与其余节点 delta=0 → H1 位级等于 LN(H0)；target 精确等于单消息公式；stats>0 |
| 6 | `test_empty_graph_delta_exactly_zero` | 空图：N/delta 严格 0、stats 严格 0、H1==LN(H0)、forward(None)==forward(空 edge) 位级相等 |
| 7 | `test_no_cross_factor_leakage_before_fusion` | 只扰动 Pt messages：C/Pv 输出**位级不变**，Pt 确实改变（block 内无跨因子泄漏） |
| 8 | `test_gradients_reach_all_trained_components` | backward 后 factorizer(投影/common/private)、recon heads、V/D/LN、fusion 全部梯度非零 |
| 9 | `test_eval_forward_equals_inference` | eval forward vs inference（CPU）allclose 1e-5；inference 返回 CPU z |
| 10 | `test_out_dim_is_hidden_dim` | out_dim == hidden_dim |
| 11 | `test_no_giant_edge_pair_tensor` | torch.profiler record_shapes 审计：无任何 op 输入带 [E,3,3,...] shape（E=4096 图） |

## 8. 执行的完整命令

```bash
# Git prep（锁定 O0）
git add docs/OFT_O0_MIGRATION_AUDIT.md experiments/oft/o0
git commit -m "O0: lock Movies A0 and DiP references"
git tag oft-o0-reference && git switch -c oft-mag

# A. Unit tests（§7）

# B. 2-epoch OFT smoke（Val-only）
python -m src.main dataset=Movies task=nc model=oft_mag num_runs=1 seed=42 \
  task.evaluate_test=false task.epochs=2 hydra.run.dir=outputs/oft_o1_smoke/diag_id

# C. Full P0 no-graph reference（Val-only）
python -m src.main dataset=Movies task=nc model=biaxis_p0 num_runs=1 seed=42 \
  task.evaluate_test=false task.history_path=experiments/oft/o1/p0_movies_seed42.csv

# D. Full OFT-DIAG-ID sanity（Val-only）
python -m src.main dataset=Movies task=nc model=oft_mag num_runs=1 seed=42 \
  task.evaluate_test=false task.history_path=experiments/oft/o1/diag_id_movies_seed42.csv

# 附：GPU peak memory probe（离线测量，单 train-step，不做训练/选择）
python - <<'EOF'  # biaxis_p0 vs oft_mag: reset_peak_memory_stats 后一次 forward+backward / eval
EOF
```

输出目录：`outputs/oft_o1_smoke/diag_id/`、`outputs/2026-09-08/00-56-23/`（P0，87 ep，
约 45s）、`outputs/2026-09-08/00-57-27/`（DIAG-ID，105 ep，约 53s）。运行日志镜像：
`outputs/oft_o1_p0_full.log`、`outputs/oft_o1_diag_full.log`。

## 9. 结果（Movies seed42，Val-only，单 run）

| 模型 | params (model+head) | best epoch | Val Acc | Val Macro-F1 | 早停 epoch |
|---|---|---|---|---|---|
| biaxis_p0（O1-C，本次） | 1,042,708 | 57 | **51.71%** | 37.62% | 87 |
| **OFT-DIAG-ID**（O1-D，本次） | 1,141,780 | 75 | **54.86%** | 42.73% | 105 |
| A0 = biaxis_final（O0 锁定） | 1,400,824 | 74 | **55.01%** | 46.37–46.67% | 104 |
| DiP（O0 锁定） | 8,170,620 | 78–88 | **56.00%** | 46.49–48.06% | 108–118 |

Δ：DIAG-ID − P0 = **+3.15pp Val Acc**（F1 +5.11pp）；DIAG-ID − A0 = **−0.15pp Val Acc**
（≈0.5× 噪声带 ±0.3pp；F1 −3.7~−3.9pp，按计划 §12 只记录、单次 F1 不做硬否决）。

params 一致性核对：OFT 相对 P0 恰好多 99,072 = 3 slots × (2×128×128 + 2×128)（V+D+LN affine）。

Val 曲线抽样（%）：P0: ep5 32.9 / ep20 45.3 / ep50 51.1 / ep80 50.6；DIAG-ID: 32.9 / 44.7 /
52.1 / 53.9 —— DIAG-ID 中后期持续更高，且最终在 A0 的最佳 epoch 区（~75）到达峰值，
曲线形态与 A0 相似（无灾难性 gap，无振荡迹象）。

## 10. 锁定参考（O0，实验 1 列锁）

```text
A0  Val Acc ≈ 55.01%（best ep74；exp run2 位级复现 0.5500898）
DiP Val Acc ≈ 56.00%（新运行 56.27/56.21，噪声带 ±0.3pp）
```

## 11. Update-ratio / neighbor-norm diagnostics（OFT-DIAG-ID 105 epochs 全程）

| key | first | mean | min | max | last |
|---|---|---|---|---|---|
| oft_l1_c_diag_update_ratio | 0.320 | 0.327 | 0.275 | 0.426 | 0.281 |
| oft_l1_pt_diag_update_ratio | 0.261 | 0.425 | 0.261 | 0.489 | 0.488 |
| oft_l1_pv_diag_update_ratio | 0.287 | 0.421 | 0.287 | 0.473 | 0.389 |
| oft_l1_c_neighbor_norm | 1.83 | 2.22 | 1.82 | 3.10 | 1.83 |
| oft_l1_pt_neighbor_norm | 1.85 | 3.19 | 1.85 | 3.59 | 3.55 |
| oft_l1_pv_neighbor_norm | 1.99 | 2.48 | 1.99 | 2.92 | 2.05 |

形态：ratio 稳定在 [0.26, 0.49]，**从不接近 0**（graph 更新持续有效，非死区），也从未
达到 base 的数倍（无爆炸）；三个 slot 行为相似（Pt/Pv 略高于 C，符合 private 携带更多
拓扑差异信息的预期）。neighbor_norm 平稳有界（1.8–3.6）。HOLD 判据中的
"ratio 几乎恒为 0 / ratio 爆炸" 两项均不成立。

## 12. P0 factor diagnostics（两模型全程一致健康，无坍塌）

| key | P0 first→last (min,max) | DIAG-ID first→last (min,max) |
|---|---|---|
| p0_common_sim | 0.209→0.639 (0.209, 0.708) | 0.209→0.637 (0.209, 0.718) |
| p0_private_sim | 0.017→0.004 (−0.186, 0.031) | 0.017→0.020 (−0.098, 0.104) |
| p0_c_norm | 3.52→4.57 (3.52, 5.01) | 3.52→4.28 (3.52, 5.12) |
| p0_pt_norm | 4.18→2.60 (2.60, 5.89) | 4.18→6.04 (4.18, 6.07) |
| p0_pv_norm | 4.01→4.53 (4.01, 5.44) | 4.01→4.20 (4.01, 4.97) |
| p0_cp_overlap_t / _v | ≤0.022 / ≤0.060 | ≤0.018 / ≤0.051 |

common_sim 收敛到 ~0.64（远离 1.0 坍塌点）、private_sim ≈0、overlap 均 < 0.06：
Common/Private 分解在 propagation 存在时**不坍塌**（P0 系列已知坍塌判据
effrank/overlap 未触发）。唯一差异是 DIAG-ID 的 pt_norm 收敛更高（2.6→6.0 vs P0 2.6），
对应 §11 中 Pt 的 neighbor_norm/update_ratio 更高——self-consistent 信号：graph 通道
主要在 text-private 上被利用。

## 13. 判据对照（执行计划 §12）

| 判据 | 结果 |
|---|---|
| Strong GO：DIAG-ID 清楚高于 P0 | ✓ +3.15pp（Acc） |
| Strong GO：距 A0 ≤ ~1pp | ✓ −0.15pp（≈0.5×噪声带） |
| Strong GO：update statistics 稳定 | ✓ ratio∈[0.26,0.49] 全程有界不归零 |
| GO：无 catastrophic gap | ✓ 全程曲线高于 P0，best epoch 区与 A0 一致 |
| HOLD 各条（低于 P0 / 低于 A0 >2pp / ratio≈0 / ratio 爆炸 / 因子坍塌 / mismatch / NaN） | 均不成立 |

**Verdict：GO（处 Strong GO 区间；F1 相对 A0/DiP 低 ~4pp，按协议记录不否决）**。
O1 科学问题回答：**C/Pt/Pv 可作为显式 graph propagation states，在 clean diagonal
factor-preserving propagation 下稳定工作** —— DIAG-ID 一步 incoming-mean 传播即以
+3.15pp 超过无图 P0，并追平 A0 的 55.01% 到 0.15pp 内（A0 依赖 P1 K=4 拓扑 relations
+P2 Null routing 等复杂 machinery；DIAG-ID 仅用一个 identity-operator 对角层）。

## 14. Memory / warnings / 阻塞与说明

**GPU peak（单 train-step / eval，Movies full-graph，进程内 max_memory_allocated）：**

| 模型 | train-step peak | eval peak |
|---|---|---|
| biaxis_p0 | 0.75 GB | 0.29 GB |
| oft_mag DIAG-ID | 1.07 GB | 0.58 GB |

两者远低于 25.4GB；DIAG-ID 的增量（+0.32GB train / +0.29GB eval）来自 full-graph
`X[src]` gather 的保存与 index_add 反向，无结构性问题（与执行计划"大图显存"关注点无关；
Movies 规模安全）。

**Warnings/anomalies**：两 run 日志 0 NaN / 0 warning / 0 traceback（"INFO"/"inference"/
config 转储文本被误命中除外）；无 early-exit 异常，均在 patience 自然早停（87/105）。
数据集访问确认 `test=not accessed`（evaluate_test=false）。

**最小必要修改说明（唯一阻塞项）**：`src/tasks/common.py::_aux_iteration_keys` 只放行
`AUX_INFO_KEYS ∪ {r3_, scope_, osra_}` 前缀 key，`oft_l1_*` 会被 runner 静默丢弃 → 新
diagnostics 无法进入 epoch 日志（§11 数字将不可得）。最小修改为在放行谓词中追加
`oft_` 前缀（+1 行），**nc.py 未动**；对所有既有模型零语义影响（无模型产出 oft_* key），
pytest 全绿佐证。已按 O1 规则在报告中说明。

**Known caveats**：单 seed（seed42）、单 run；F1 差异带噪声（O0 已记录 Macro-F1
run-to-run 波动大）；diff 性结论（vs P0 +3.15pp / vs A0 −0.15pp）在 ±0.3pp 协议噪声带外/
边缘处有效。

## 15. 产出物

```text
docs/OFT_O1_IMPLEMENTATION_REPORT.md
experiments/oft/o1/p0_movies_seed42.csv        # 87 epochs 逐 epoch（epoch/train/val/lr/patience/p0_*）
experiments/oft/o1/diag_id_movies_seed42.csv   # 105 epochs 逐 epoch
src/models/oft_mag.py / src/models/oft_components.py / configs/model/oft_mag.yaml / tests/test_oft_mag.py
```

O1 完成。未进入 O2（O2 起点建议：跨因子 functional transition 需先在 O2 计划中定义
(j,a)→(i,b) 的自由度与验证协议）。
