# R0-P2：SimpleMAGProbe 训练 Sanity Check

> 日期：2026-09-07 | 数据集：Grocery (MAGB) | seed：42 | NC，1 run
> 协议完全复用框架 `run_nc`（full-graph、AdamW、CE、val-acc early stopping、best-val ckpt、test 仅做 sanity 且不用于 R0 分析）

## 1. 配置与数据

| 项 | 值 |
|---|---|
| dataset | Grocery (MAGB)：17,074 节点 / 71,131 无向对（142,262 有向边）/ 20 类 |
| split (seed42) | train 10,244 / val 3,415 / test 3,415（缓存文件 `Grocery_nc_seed42_train0.6_val0.2.pt`） |
| 特征 | text RoBERTa-base mean 768d (f32)；visual CLIP ViT-L/14 768d (f32)；两模态磁盘 norm ≈ 12.2 / 1.0 |
| Probe | hidden 128；模态投影 2 层 MLP（Linear→ReLU→Dropout(0.2)）；**投影前逐模态 L2 normalize**（文本 norm 12× 图像 → 防止 text 通道压死 visual）；`h_self = W_s[hT‖hV]`；单层 additive MP，`α=1/√(d_i d_j)`（loader 图双向无自环，self 单独建模） |
| 优化 | AdamW lr=1e-3 wd=1e-4；epochs 300（patience 30，val-acc 早停）；无 grad clip |
| 参数 | model 298,388（含 head 20×128）；投影 231K / W_self 33K / W_msg×2 33K |

## 2. 训练稳定性

- train CE 单调下降至 0.419（epoch 274），无 NaN / 无梯度爆炸（无 grad clip 下全程有限）
- best val acc 79.21 @ epoch ~244（早停于 epoch 274，patience 用满前 3 轮内）
- 过拟合程度：train acc 87.7 vs val 79.2（+8.5pp，单层模型正常水平）
- 产物：`experiments/r0/checkpoints/Grocery_seed42_probe.pt`（键：task/seed/model_state/head_state/data_info）
- 训练历史：`experiments/r0/results/Grocery/seed_42/probe_train_log.csv`

## 3. Sanity 对比（同 Grocery × seed42 同 split，各 1 run）

| model | 结构 | val acc | test acc | test macro-F1 |
|---|---|---|---|---|
| mlp | 2×Linear 128 | 77.07 | 77.98 | 66.59 |
| **simple_mag_probe** | 1 层 MP + 2 层投影 128 | **79.21** | **79.24** | **68.26** |
| gcn | 3 层 GCN 256 | 80.38 | 80.44 | 70.89 |

- Probe 比最接近的简单 baseline（gcn）低 **1.17pp**（test）→ 远低于 5pp 灾难阈值 ✓
- Probe 高于无图 mlp +1.26pp（test）→ 图消息确实带来增益，非欠拟合/过弱 ✓
- 说明：probe 是单层 additive 模型（无 gcn 的 3 层深度），差距在合理工程范围内；为保 relation 分解可解释性，不做架构增强（计划 §3.3 约束）。

## 4. 结论

Probe 性能不异常、训练稳定、数值不变量（unit tests 10 项 + estimator 4 项）通过 → **允许进入 counterfactual 阶段（R0-P3）**。

Grocery 选型复核：基线 gcn 80.4 稳定、类不均衡程度适中、图连通健康（audit：LCC 94.2%、0 孤立点）→ 作为首轮诊断数据集合适。
