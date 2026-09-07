# FIRM-MAG：R0 实验阶段性推进计划

> **阶段目标**：在正式实现 FIRM-MAG 之前，用最小、可控、可证伪的实验验证核心科学假设。  
> **原则**：R0 不追求 SOTA，不堆模块，不调复杂路由；先确认“问题真实存在 + 二阶 functional interaction 信号可被有效近似”。  
> **完成 R0 后再决定**：Go / Modify / No-Go。

---

## 0. R0 要回答的四个问题

FIRM-MAG 当前的核心假设不是“图邻居有的有用、有的没用”，而是：

> **同一结构关系上，Text 与 Visual 消息的联合任务效应可能具有非加性；单模态相似性/可靠性不足以决定这种联合功能性交互；任务曲率信息能够高效近似这种 interaction。**

R0 必须依次回答：

### H1：非加性交互是否真实存在？

对关系 \(j\rightarrow i\)，比较：

\[
Q_{ij}^{TV}
\quad \text{与} \quad
Q_{ij}^{T}+Q_{ij}^{V}
\]

定义：

\[
S_{ij}
=
Q_{ij}^{TV}-Q_{ij}^{T}-Q_{ij}^{V}
\]

如果绝大多数关系 \(S_{ij}\approx 0\)，则“Functional Interaction”问题本身很弱。

---

### H2：不同关系的最优传播 action 是否真的不同？

定义四种 action：

\[
\mathcal A=\{\varnothing,T,V,TV\}
\]

其中：

- \(\varnothing\)：不使用该关系；
- \(T\)：仅传播 Text；
- \(V\)：仅传播 Visual；
- \(TV\)：Text + Visual 联合传播。

若几乎所有关系都选择 \(TV\)，则动态 action routing 缺乏必要性。

---

### H3：Semantic similarity 是否不足以解释 functional utility / interaction？

至少比较：

\[
sim_T(i,j)=\cos(h_i^T,h_j^T)
\]

\[
sim_V(i,j)=\cos(h_i^V,h_j^V)
\]

与：

\[
Q_T,Q_V,S
\]

之间的相关性。

目标不是证明 similarity 完全没用，而是证明：

> **semantic relevance 不是 task-level functional effect 的充分代理。**

---

### H4：二阶 task-curvature estimator 能否逼近 exact counterfactual？

比较：

- Exact counterfactual；
- First-order gradient approximation；
- Second-order curvature approximation。

重点评价：

\[
Spearman,\ Kendall,\ ActionAcc,\ SignAcc
\]

而不是只看 MSE。

---

# 1. R0 总体阶段划分

建议严格按以下顺序推进：

| 阶段 | 内容 | 是否允许进入下一阶段 |
|---|---|---|
| R0-P0 | 仓库审查与数据集选择 | 完成代码路径/数据接口审查 |
| R0-P1 | 实现 Simple MAG Probe | 模型前向可运行 |
| R0-P2 | Probe 正常训练与 sanity check | Probe 性能不异常、训练稳定 |
| R0-P3 | Exact Counterfactual Evaluator | 能稳定输出每条关系的四种 loss |
| R0-P4 | Functional Interaction 统计 | H1/H2 初步成立 |
| R0-P5 | First/Second-order Estimator | 能输出近似 \(Q,S\) |
| R0-P6 | 近似质量与 proxy 分析 | H3/H4 初步成立 |
| R0-P7 | 3 datasets × 3 seeds 复核 | 决定 Go / Modify / No-Go |

**不要跳阶段。**

---

# 2. R0-P0：先让 AI 审查仓库，不改代码

## 2.1 目标

先搞清楚：

- 数据加载在哪里；
- Text / Visual feature 的字段和维度；
- edge_index 的方向约定；
- train / val / test split 如何读取；
- 当前 NC trainer 如何工作；
- 模型注册机制；
- 配置系统；
- 日志保存方式；
- 当前已有的最简单 GCN / multimodal baseline 如何实现。

### 第一轮只做 NC

R0 第一轮不要做 LP。

优先选择：

- NC 数据集；
- Text / Image 两个模态都完整；
- 数据规模中等；
- 当前 baseline 训练稳定；
- 不要选最难、最容易 OOM 或方差最大的那个。

第一轮：

\[
1\ dataset \times 1\ seed
\]

跑通后再扩展。

---

## 2.2 给 AI 的 Prompt：P0 仓库审查

将下面 Prompt 原样交给你的 Coding AI，并补充仓库路径/分支名。

```text
你现在只做代码审查和实验设计，不要修改任何代码。

我的目标是在当前多模态属性图（MAG）实验框架中新增一个独立的 R0 Functional Interaction Diagnostic，用于验证一个研究假设。当前阶段只做节点分类（NC）。

请仔细审查当前仓库，重点回答：

1. 数据加载入口在哪里？
2. Text feature 和 Visual feature 分别存在哪个字段，shape 是什么？
3. edge_index 的 source/target 语义是什么？无向边是否存为双向边？
4. train/val/test split 在哪里生成或读取？如何保证复用现有 split？
5. 当前 NC trainer、loss、early stopping、seed 设置分别在哪里？
6. 当前最简单的 GCN / multimodal 模型实现在哪里？
7. 模型如何注册到 config/runner？
8. 当前结果、日志、checkpoint 保存路径是什么？
9. 如果新增以下 R0 组件，最少需要新增/修改哪些文件：
   - SimpleMAGProbe
   - train_r0_probe.py
   - run_r0_counterfactual.py
   - analyze_r0.py
10. 给出一个“最小侵入”的实现方案：优先新增独立文件，不修改现有 baseline 行为。

特别要求：
- 不要重构现有项目。
- 不要修改数据划分。
- 不要改变现有 baseline 默认配置。
- 不要实现 FIRM-MAG 正式模型。
- 当前只规划 R0。
- 输出建议的文件结构、关键类/函数接口、数据流，以及可能的风险点。
```

---

# 3. R0-P1：实现 Simple MAG Probe

## 3.1 目的

Probe 不是新模型，也不是为了刷性能。

Probe 的任务是创造一个**可精确分解到单条 relation / 单个 modality message 的诊断环境**。

必须满足：

\[
z_i
=
h_i^{self}
+
\sum_{j\in\mathcal N(i)}
\left(
\delta_{ij}^{T}
+
\delta_{ij}^{V}
\right)
\]

这样才能精确构造：

\[
\varnothing,\ T,\ V,\ TV
\]

四种局部 intervention。

---

## 3.2 推荐 Probe 架构

### 模态投影

\[
h_i^T=f_T(x_i^T)
\]

\[
h_i^V=f_V(x_i^V)
\]

其中 \(f_T,f_V\) 可以是：

- Linear → ReLU → Dropout；
- 或轻量 2-layer MLP。

隐藏维度建议先沿用现有 baseline 常用维度，不要专门大调参。

---

### Self representation

\[
h_i^{self}
=
W_s[h_i^T\Vert h_i^V]
\]

---

### Relation-specific messages

对于 \(j\rightarrow i\)：

\[
\delta_{ij}^{T}
=
\alpha_{ij}W_T h_j^T
\]

\[
\delta_{ij}^{V}
=
\alpha_{ij}W_V h_j^V
\]

\(\alpha_{ij}\) 使用固定图归一化权重，例如：

\[
\alpha_{ij}
=
\frac{1}{\sqrt{d_i d_j}}
\]

或直接复用当前 GCN 的规范化逻辑。

---

### Receiver representation

\[
z_i
=
h_i^{self}
+
\sum_{j\in\mathcal N(i)}
\left(
\delta_{ij}^{T}
+
\delta_{ij}^{V}
\right)
\]

---

### Classifier

\[
o_i=W_cz_i+b
\]

\[
p_i=softmax(o_i)
\]

\[
L_i=CE(p_i,y_i)
\]

---

## 3.3 R0 Probe 的硬约束

为了保证后面的 counterfactual 有清晰解释：

**不要加入：**

- attention；
- router；
- MoE；
- OT；
- Q-Former；
- cross-modal transformer；
- edge-conditioned nonlinear gating；
- 聚合后复杂 MLP；
- 多层 message passing。

第一版建议 **1 层 relational aggregation**。

允许在进入 message passing 前使用轻量投影 MLP。

---

## 3.4 Forward 必须额外返回

训练时正常 forward 返回 logits。

诊断模式下还必须可以返回：

```text
h_text
h_visual
h_self
z_full
edge_delta_text
edge_delta_visual
edge_norm
```

其中对 edge_index 中第 \(r\) 条有向边：

```text
source = edge_index[0, r] = j
target = edge_index[1, r] = i
edge_delta_text[r]   = δ^T_{ij}
edge_delta_visual[r] = δ^V_{ij}
```

这一点必须写单元测试验证。

---

## 3.5 给 AI 的 Prompt：P1 实现 Probe

```text
请根据你上一阶段对仓库的审查，实现一个独立的 R0 SimpleMAGProbe。

研究目的：
这个 Probe 不是正式方法，而是为了把 receiver node i 的表示精确分解为：

z_i = h_self_i + sum_j (delta_text_ij + delta_visual_ij)

从而后续可以对单条有向关系 j->i 做 Null/Text/Visual/Text+Visual 四种局部 counterfactual intervention。

实现要求：

1. Text 和 Visual 分别经过轻量投影得到 h_text、h_visual。
2. self representation:
   h_self_i = W_self [h_text_i || h_visual_i]
3. 对每条有向边 j->i：
   delta_text_ij  = alpha_ij * W_text_msg h_text_j
   delta_visual_ij = alpha_ij * W_visual_msg h_visual_j
4. receiver:
   z_i = h_self_i + scatter_sum(delta_text + delta_visual, target=i)
5. classifier 直接作用于 z_i。
6. 第一版只允许 1 层 message passing。
7. 不加入 attention/router/MoE/OT/QFormer/cross-modal transformer。
8. 诊断 forward 必须能返回：
   h_text, h_visual, h_self, z_full,
   edge_delta_text, edge_delta_visual, edge_norm。
9. 保证 edge_delta_text[r]、edge_delta_visual[r] 对应 edge_index[:, r]。
10. 不修改现有 baseline 的默认行为。

同时新增最小单元测试：
- 检查 shape；
- 检查一条 edge 的 source/target 对应关系；
- 随机选一个 target node i，验证：
  z_full[i] ≈ h_self[i] + sum_{r: target(r)=i}(delta_text[r]+delta_visual[r])
  数值误差应小于 1e-5。

先完成代码和测试，不要开始大规模训练。
```

---

# 4. R0-P2：训练 Probe，先做 Sanity Check

## 4.1 训练原则

完全复用现有：

- split；
- optimizer 规范；
- early stopping；
- seed 管理；
- evaluation protocol。

第一轮：

```text
dataset = 1
seed = 1
```

建议先用正常训练配置，不要过度 grid search。

---

## 4.2 Probe 需要达到什么水平？

Probe 不要求超过 MAG SOTA。

但是不能明显失真。

至少满足：

1. train loss 正常下降；
2. val metric 正常收敛；
3. 无 NaN / exploding gradient；
4. 不出现极端欠拟合；
5. 与仓库中最简单的合理 baseline 相比，不应出现明显灾难性性能下降。

工程判断：

> 如果 Probe 比最接近的简单 baseline 低约 5 个百分点以上，需要先检查模型过弱或实现有误。

不要为了追性能加入复杂传播机制。必要时只增强**消息产生前的 modality projector**，仍保持最终 relation decomposition 是 additive 的。

---

## 4.3 训练后必须保存

```text
checkpoint.pt
train_log.txt / csv
config.yaml/json
best_val_metric
test_metric（仅作为 sanity check；不要用于 R0 functional analysis）
```

R0 interaction analysis 只使用：

\[
Validation\ nodes
\]

及其 labels。

---

## 4.4 给 AI 的 Prompt：P2 训练

```text
现在训练刚实现的 R0 SimpleMAGProbe，只做一个候选 NC 数据集和一个 seed。

要求：
1. 完全复用当前仓库的数据 split 和 evaluation protocol。
2. 记录 train loss、val metric、best epoch、最终 test metric。
3. 保存 best-validation checkpoint。
4. 不做大规模 grid search。
5. 不修改模型架构。
6. 检查是否有 NaN、梯度爆炸、明显过拟合/欠拟合。
7. 将 Probe 与仓库中最接近的简单 baseline 做 sanity comparison。
8. 输出一份 r0_probe_sanity.md，说明：
   - dataset
   - seed
   - feature dimensions
   - hidden dim
   - parameter count
   - best val
   - test
   - closest baseline result
   - 是否建议进入 counterfactual 阶段

当前阶段不要实现正式 FIRM-MAG。
```

---

# 5. R0-P3：Exact Counterfactual Evaluator

这是整个 R0 最核心的实验。

## 5.1 只分析 Validation Receiver Nodes

不得用 test label 做 R0 现象发现。

采样对象：

\[
i\in V_{val}
\]

然后从其入边：

\[
j\rightarrow i
\]

中抽样。

---

## 5.2 推荐采样策略

第一轮目标：

\[
3000\sim5000
\]

条有向 relation。

不要直接在所有 edges 上均匀随机，因为高度节点会被过度代表。

推荐：

1. 从 validation receiver nodes 中采样；
2. 尽量覆盖不同类别；
3. 按 receiver degree quantile 分层；
4. 每个 receiver 最多采 1–3 条入边；
5. 排除 self-loop；
6. 固定一个 `analysis_seed`。

如果数据集较小，允许少于 3000 条，并记录实际数量。

---

# 6. Exact 四种 Intervention

已冻结 Probe。

对于 sampled edge \(r=(j\rightarrow i)\)：

\[
\delta_T=\delta_{ij}^T
\]

\[
\delta_V=\delta_{ij}^V
\]

因为：

\[
z_i^{full}
=
b_i+\delta_T+\delta_V
\]

所以无需重新跑四次全图 forward。

直接：

### Null

\[
z_i^\varnothing
=
z_i^{full}-\delta_T-\delta_V
\]

### Text only

\[
z_i^T
=
z_i^\varnothing+\delta_T
\]

### Visual only

\[
z_i^V
=
z_i^\varnothing+\delta_V
\]

### Text + Visual

\[
z_i^{TV}
=
z_i^\varnothing+\delta_T+\delta_V
=
z_i^{full}
\]

然后只将四种 \(z\) 送进**冻结 classifier**。

---

## 6.1 Exact losses

\[
L^\varnothing_i
\]

\[
L^T_i
\]

\[
L^V_i
\]

\[
L^{TV}_i
\]

---

## 6.2 Exact gains

\[
Q_T=L^\varnothing-L^T
\]

\[
Q_V=L^\varnothing-L^V
\]

\[
Q_{TV}=L^\varnothing-L^{TV}
\]

---

## 6.3 Functional interaction

\[
S
=
Q_{TV}-Q_T-Q_V
\]

等价于：

\[
S
=
L^T+L^V-L^{TV}-L^\varnothing
\]

---

## 6.4 Normalized interaction

建议同时保存：

\[
S_{norm}
=
\frac{S}
{|Q_T|+|Q_V|+|Q_{TV}|+\epsilon}
\]

其中：

\[
\epsilon=10^{-8}
\]

---

## 6.5 Optimal action

设：

\[
Q_\varnothing=0
\]

则：

\[
a^*
=
\arg\max
\{0,Q_T,Q_V,Q_{TV}\}
\]

输出：

```text
NULL
TEXT
VISUAL
JOINT
```

---

## 6.6 给 AI 的 Prompt：P3 Exact Evaluator

```text
请实现 R0 Exact Counterfactual Evaluator。

前提：
- 加载已经训练完成并冻结的 SimpleMAGProbe best checkpoint。
- model.eval()。
- 只分析 validation receiver nodes，不使用 test labels。
- 不重新训练模型。

采样：
1. 从 validation receiver nodes 分层采样。
2. 尽量覆盖不同类别和 degree quantile。
3. 每个 receiver node 最多采 1-3 条 incoming non-self-loop relations。
4. 首轮目标 3000-5000 条 relation。
5. 固定 analysis_seed。

对每条 edge r=(j->i)，利用 Probe 已返回的：
z_full[i]
delta_text[r]
delta_visual[r]

构造：
z_null = z_full[i] - delta_text[r] - delta_visual[r]
z_text = z_null + delta_text[r]
z_visual = z_null + delta_visual[r]
z_joint = z_null + delta_text[r] + delta_visual[r]

仅通过冻结 classifier 计算四种 per-node cross-entropy：
L_null, L_text, L_visual, L_joint

然后计算：
Q_text = L_null - L_text
Q_visual = L_null - L_visual
Q_joint = L_null - L_joint
S = Q_joint - Q_text - Q_visual
S_norm = S / (abs(Q_text)+abs(Q_visual)+abs(Q_joint)+1e-8)

令 Q_null=0：
best_exact = argmax([0,Q_text,Q_visual,Q_joint])

特别要求：
- 这是“单条 message intervention”，不是删除 graph edge 后重新跑整张图。
- 第一版 Probe 只有一层传播，因此这种局部 intervention 是定义明确的。
- 检查 z_joint 与原始 z_full[i] 数值一致。
- 检查 L_joint 与原始该节点 classifier loss 数值一致。
- 保存逐 relation 结果 CSV。
- 不开始训练 Router。
```

---

# 7. R0-P4：Semantic Proxy 统计

对每条 sampled relation 保存：

\[
sim_T
=
\cos(h_i^T,h_j^T)
\]

\[
sim_V
=
\cos(h_i^V,h_j^V)
\]

建议额外保存：

```text
degree_i
degree_j
edge_norm
label_i
label_j（如果 j 恰好有可用 label，可保存用于后分析，但不要作为 router 输入）
```

核心相关性：

\[
Spearman(sim_T,Q_T)
\]

\[
Spearman(sim_V,Q_V)
\]

以及探索：

\[
(sim_T,sim_V)\rightarrow S
\]

是否存在简单映射。

---

# 8. R0-P5：First-order / Second-order Estimator

以：

\[
z^\varnothing
\]

为 Taylor expansion point。

classifier：

\[
o=W_cz+b
\]

\[
p=softmax(o)
\]

标签 one-hot 为 \(y\)。

---

## 8.1 Logit-space gradient

\[
g=p-y
\]

---

## 8.2 Cross-entropy logit Hessian

\[
C
=
Diag(p)-pp^T
\]

无需构造 hidden-dimension Hessian。

---

## 8.3 将 message 映射到 logits 空间

\[
a_T
=
W_c\delta_T
\]

\[
a_V
=
W_c\delta_V
\]

---

## 8.4 First-order

\[
Q_T^{(1)}
=
-g^Ta_T
\]

\[
Q_V^{(1)}
=
-g^Ta_V
\]

\[
Q_{TV}^{(1)}
=
-g^T(a_T+a_V)
\]

因此：

\[
S^{(1)}=0
\]

这是理论结果，不是 bug。

---

## 8.5 Second-order

\[
Q_T^{(2)}
=
-g^Ta_T
-\frac12 a_T^TCa_T
\]

\[
Q_V^{(2)}
=
-g^Ta_V
-\frac12 a_V^TCa_V
\]

\[
Q_{TV}^{(2)}
=
-g^T(a_T+a_V)
-\frac12(a_T+a_V)^TC(a_T+a_V)
\]

Interaction：

\[
S^{(2)}
=
Q_{TV}^{(2)}
-
Q_T^{(2)}
-
Q_V^{(2)}
\]

理论上：

\[
\boxed{
S^{(2)}
=
-a_T^TCa_V
}
\]

代码中应同时用“两种计算方式”得到 \(S^{(2)}\)，验证二者误差小于 `1e-6`。

---

## 8.6 给 AI 的 Prompt：P5 Curvature Estimator

```text
在现有 exact counterfactual evaluator 中新增 first-order 和 second-order functional estimator。

对每条 sampled relation，以 z_null 为 Taylor expansion point。

设 classifier logits = W_c z + b。
令：
p = softmax(W_c z_null + b)
y = one_hot(label_i)
g = p - y
C = diag(p) - p p^T

令：
a_t = W_c @ delta_text
a_v = W_c @ delta_visual

计算：

First-order:
Q_t_1  = - g^T a_t
Q_v_1  = - g^T a_v
Q_tv_1 = - g^T(a_t+a_v)

Second-order:
Q_t_2  = -g^T a_t - 0.5*a_t^T C a_t
Q_v_2  = -g^T a_v - 0.5*a_v^T C a_v
Q_tv_2 = -g^T(a_t+a_v) - 0.5*(a_t+a_v)^T C (a_t+a_v)

S_2 = Q_tv_2 - Q_t_2 - Q_v_2

同时计算：
S_2_cross = - a_t^T C a_v

要求：
abs(S_2 - S_2_cross) < 1e-6

再令：
best_first  = argmax([0,Q_t_1,Q_v_1,Q_tv_1])
best_second = argmax([0,Q_t_2,Q_v_2,Q_tv_2])

把所有值写入同一份 relation diagnostics CSV。

注意：
- 不要显式构造 hidden_dim × hidden_dim Hessian。
- 仅在 class/logit space 计算。
- model 全程 frozen。
- 当前只做 estimator，不训练任何 router。
```

---

# 9. R0 统一输出文件

建议最终形成：

```text
experiments/r0/
├── configs/
│   └── <dataset>_<seed>.yaml
├── checkpoints/
│   └── <dataset>_<seed>_probe.pt
├── results/
│   └── <dataset>/
│       └── seed_<seed>/
│           ├── r0_relation_diagnostics.csv
│           ├── r0_metrics.json
│           ├── r0_summary.md
│           └── figures/
│               ├── fig1_additive_vs_joint.png
│               ├── fig2_snorm_distribution.png
│               ├── fig3_action_distribution.png
│               ├── fig4_exact_vs_second_interaction.png
│               ├── fig5_action_confusion.png
│               └── fig6_proxy_correlations.png
```

如果当前仓库已有统一 results 目录，则适配现有规范，不必强行使用上述路径。

---

# 10. `r0_relation_diagnostics.csv` 必须包含的字段

至少：

```text
dataset
train_seed
analysis_seed

receiver_i
neighbor_j
edge_id

label_i
degree_i
degree_j
edge_norm

sim_text
sim_visual

L_null
L_text
L_visual
L_joint

Q_text
Q_visual
Q_joint
S_exact
S_norm

Q_text_1
Q_visual_1
Q_joint_1

Q_text_2
Q_visual_2
Q_joint_2
S_second

best_exact
best_first
best_second
```

可以额外增加：

```text
receiver_correct_full
full_confidence
receiver_degree_bin
```

---

# 11. R0-P6：统计指标

## 11.1 Interaction 存在性

输出：

```text
mean(S_exact)
median(S_exact)
mean(abs(S_exact))
median(abs(S_exact))

mean(S_norm)
median(S_norm)
mean(abs(S_norm))
median(abs(S_norm))

P(|S_norm| > 0.05)
P(|S_norm| > 0.10)
P(|S_norm| > 0.20)

P(S_exact > 0)
P(S_exact < 0)
```

**注意：0.05 / 0.10 / 0.20 只是 R0 工程观察阈值，不是理论定义。**

---

## 11.2 Action Diversity

统计：

```text
P(NULL)
P(TEXT)
P(VISUAL)
P(JOINT)
```

同时按：

- degree bin；
- node class；
- full-model prediction correct / incorrect；

分组观察。

---

## 11.3 Semantic Proxy

至少输出：

```text
Spearman(sim_text, Q_text)
Spearman(sim_visual, Q_visual)

Pearson(sim_text, Q_text)
Pearson(sim_visual, Q_visual)
```

可选：

```text
Spearman(sim_text * sim_visual, S_exact)
```

但不要把某个简单乘积当成正式 baseline。

---

## 11.4 Approximation Quality

对 Text / Visual / Joint 分别输出：

```text
Spearman(Q_exact, Q_first)
Spearman(Q_exact, Q_second)

Kendall(Q_exact, Q_first)
Kendall(Q_exact, Q_second)
```

还要输出 pooled score correlation。

---

## 11.5 Action Prediction

\[
ActionAcc_{first}
\]

\[
ActionAcc_{second}
\]

再给 second-order action confusion matrix。

---

## 11.6 Interaction Sign

只对：

\[
|S_{exact}|>\epsilon_s
\]

的有效 interaction 样本计算：

\[
SignAcc
=
P(sign(S_{exact})=sign(S_{second}))
\]

\(\epsilon_s\) 可以先使用很小的数值，或按 \(|S|\) 的低分位数过滤数值噪声；必须在报告里说明。

---

# 12. 必画的四张核心图

## Figure 1：Additivity Test

横轴：

\[
Q_T+Q_V
\]

纵轴：

\[
Q_{TV}
\]

加：

\[
y=x
\]

解释：

- 对角线：接近 additive；
- 上方：positive interaction；
- 下方：negative interaction。

大量点时用低透明度 scatter 或 hexbin。

---

## Figure 2：Normalized Interaction Distribution

画：

\[
S_{norm}
\]

分布。

重点看：

- 是否只集中于 0；
- 是否具有明显正/负尾部；
- \(|S_{norm}|>0.1\) 的比例。

---

## Figure 3：Optimal Action Distribution

柱状图：

```text
NULL / TEXT / VISUAL / JOINT
```

目的：

> 验证关系的最优传播方式是否具有异质性。

---

## Figure 4：Exact vs Second-order Interaction

横轴：

\[
S_{exact}
\]

纵轴：

\[
S_{second}
\]

加：

\[
y=x
\]

并标注：

```text
Spearman
Pearson
SignAcc
```

---

# 13. 建议额外两张图

## Figure 5：Exact Action vs Second-order Action Confusion Matrix

检验二阶 teacher 是否能正确判断：

```text
NULL
TEXT
VISUAL
JOINT
```

---

## Figure 6：Semantic Proxy vs Functional Effect

例如：

```text
sim_text vs Q_text
sim_visual vs Q_visual
```

或者汇总成 correlation matrix。

---

# 14. 给 AI 的 Prompt：P6 自动分析和画图

```text
请实现 analyze_r0.py，读取 r0_relation_diagnostics.csv，不重新运行模型。

必须输出：

A. Interaction statistics
- mean/median S_exact
- mean/median |S_exact|
- mean/median S_norm
- mean/median |S_norm|
- P(|S_norm|>0.05)
- P(|S_norm|>0.10)
- P(|S_norm|>0.20)
- P(S>0), P(S<0)

B. Optimal action distribution
- NULL/TEXT/VISUAL/JOINT count and percentage

C. Semantic proxy
- Spearman/Pearson(sim_text,Q_text)
- Spearman/Pearson(sim_visual,Q_visual)

D. Approximation quality
分别对 Text/Visual/Joint 计算：
- Spearman exact vs first
- Spearman exact vs second
- Kendall exact vs first
- Kendall exact vs second
并额外计算 pooled correlation。

E. Action
- best_first vs best_exact accuracy
- best_second vs best_exact accuracy
- second-order confusion matrix

F. Interaction
- Spearman(S_exact,S_second)
- Pearson(S_exact,S_second)
- 对排除数值接近 0 的样本计算 sign accuracy

必须生成：
fig1_additive_vs_joint.png
fig2_snorm_distribution.png
fig3_action_distribution.png
fig4_exact_vs_second_interaction.png
fig5_action_confusion.png
fig6_proxy_correlations.png

同时生成：
r0_metrics.json
r0_summary.md

r0_summary.md 必须只陈述观察到的数据，不要自动宣布论文 idea 成立。
```

---

# 15. 第一轮 Go / Modify / No-Go 判定

不要在看到结果以后临时改标准。

下面是**工程筛选标准**，不是论文定理。

---

## 15.1 H1：Interaction

### 正向信号

如果有明显比例：

\[
|S_{norm}|>0.1
\]

且分布具有正/负尾部，则支持继续。

### 风险信号

如果绝大多数：

\[
|S_{norm}|\approx0
\]

则 Functional Interaction 不够强。

---

## 15.2 H2：Action Diversity

如果：

```text
JOINT > 90%~95%
```

则动态 action routing 价值明显下降。

理想情况不是四类均匀，而是至少：

- 3 类 action 有非微小占比；
- 或不同数据区域出现明显 action heterogeneity。

---

## 15.3 H3：Semantic Proxy

如果：

\[
Spearman(sim_T,Q_T)
\]

和：

\[
Spearman(sim_V,Q_V)
\]

高到接近：

\[
0.8+
\]

需要警惕：

> semantic similarity 已经足够解释 utility。

如果只是中低相关，则支持“proxy ≠ functional effect”。

---

## 15.4 H4：Second-order Approximation

第一轮希望至少：

\[
Spearman(Q^{exact},Q^{(2)})\gtrsim 0.5
\]

并且：

\[
ActionAcc_{second}\gtrsim60\%
\]

更强的信号是：

\[
Spearman>0.65
\]

\[
ActionAcc>70\%
\]

并且 second-order 明显优于 first-order。

这些都是工程决策阈值，不应直接写成论文理论标准。

---

# 16. R0-P7：第一轮通过后，再做 3 datasets × 3 seeds

只有第一轮：

\[
1\ dataset\times1\ seed
\]

通过后才扩展。

---

## 16.1 数据集选择原则

选 3 个差异明显的 NC MAG 数据集：

### Dataset A

- 图结构相对稳定；
- 传统 GNN 表现正常。

### Dataset B

- structural noise / heterophily 更明显。

### Dataset C

- modality quality / modality dominance 更不均衡。

不要只挑对 Idea 最有利的三个。

---

## 16.2 Seeds

建议：

```text
42 / 43 / 44
```

如果你现有 baseline 使用另一套标准 seed，就完全沿用现有协议。

---

## 16.3 每个 dataset × seed 都保存独立 CSV

不要只保存 aggregate numbers。

后续需要分析：

- dataset effect；
- seed variance；
- degree-specific interaction；
- modality-specific pattern。

---

# 17. R0 最终阶段报告应包含

建议 AI 自动生成：

```text
R0_FINAL_REPORT.md
```

结构：

```markdown
# R0 Functional Interaction Validation

## 1. Experimental Setup
- datasets
- seeds
- splits
- probe architecture
- sampled relations

## 2. Probe Sanity
- val/test metrics
- training stability

## 3. H1: Non-additive Interaction
- statistics
- figures
- dataset/seed consistency

## 4. H2: Action Diversity
- NULL/T/V/TV distribution

## 5. H3: Proxy-Utility Mismatch
- similarity correlations

## 6. H4: Curvature Approximation
- first vs second
- action accuracy
- interaction sign accuracy

## 7. Failure Cases
- 哪些 dataset/degree/action 失败
- second-order 误差最大的样本特征

## 8. Evidence Summary
只列事实，不自动做论文式夸张结论。
```

---

# 18. R0 阶段明确禁止做的事情

R0 未完成前不要：

- 实现正式 FIRM-MAG Router；
- 上 Top-K hard routing；
- 加 MoE；
- 加 prototype；
- 加 OT；
- 加新的 contrastive loss；
- 加 modality reliability gate；
- 加 cross-modal transformer；
- 因为某个结果不理想就不断增加模块；
- 在 test set 上做 interaction 分析；
- 大量调参把 R0 现象“调出来”。

R0 是**诊断实验**，不是性能实验。

---

# 19. 建议的 Git 推进方式

每个阶段完成后独立 commit。

例如：

```text
r0-p0: audit experiment pipeline
r0-p1: add simple additive MAG probe
r0-p2: validate probe training
r0-p3: add exact relation counterfactual evaluator
r0-p5: add first and second order estimators
r0-p6: add R0 analysis and figures
r0-p7: run multi-dataset multi-seed validation
```

这样如果某一步实现有问题，很容易定位和回滚。

---

# 20. 建议你与 Coding AI 的工作方式

每一阶段都遵守：

```text
先审查
→ 给实现计划
→ 再改代码
→ 跑最小测试
→ 检查数值不变量
→ 才跑实验
```

不要直接给 AI：

> “帮我把 R0 全部实现并跑完。”

否则最容易出现：

- 它误解 edge direction；
- counterfactual 不是我们定义的 intervention；
- 偷偷重新跑整图；
- action loss 计算错；
- Hessian expansion point 用错；
- CSV 缺字段；
- 分析使用了 test labels。

---

# 21. 每个阶段都让 AI 做数值不变量检查

尤其是以下四个必须自动 assert：

### Check 1

\[
z_{full}[i]
=
h_{self}[i]
+
\sum_{j\rightarrow i}
(\delta_T+\delta_V)
\]

---

### Check 2

\[
z_{joint}
=
z_{full}[i]
\]

---

### Check 3

\[
L_{joint}
=
L_{original,node-i}
\]

---

### Check 4

\[
S_{second}
=
-a_T^TCa_V
\]

这四个如果不过，不要跑实验。

---

# 22. 完成 R0 后你需要返给我的材料

当 R0 第一轮或完整 R0 完成后，请把以下内容发给我。

最重要的是：

```text
r0_summary.md
r0_metrics.json
r0_relation_diagnostics.csv
```

以及至少四张图：

```text
fig1_additive_vs_joint.png
fig2_snorm_distribution.png
fig3_action_distribution.png
fig4_exact_vs_second_interaction.png
```

最好再带：

```text
fig5_action_confusion.png
fig6_proxy_correlations.png
r0_probe_sanity.md
实验配置文件
Probe 训练日志
```

我收到后会按照：

\[
H1\rightarrow H2\rightarrow H3\rightarrow H4
\]

逐项审查，然后决定：

\[
\boxed{
Go / Modify / No-Go
}
\]

如果是 Go，再正式进入 **R1：Functional Interaction Router 是否真的能被 distill，并带来下游性能提升**。

---

# 23. 推荐的实际执行顺序

你现在只需要按下面顺序行动：

```text
[ ] P0：把“仓库审查 Prompt”给 Coding AI
[ ] 检查它给出的实现计划，确认没有大改 baseline

[ ] P1：实现 SimpleMAGProbe
[ ] 跑 shape + decomposition unit test
[ ] 四个数值不变量先通过

[ ] P2：单数据集、单 seed 训练 Probe
[ ] 检查训练稳定和 sanity performance
[ ] 保存 best checkpoint

[ ] P3：实现 exact counterfactual evaluator
[ ] 先只跑 20 条 edge debug
[ ] 人工打印检查四种 z / loss / Q / S

[ ] P3-small：跑 100 条 relation
[ ] 检查 CSV、NaN、异常范围

[ ] P3-full：跑 3000-5000 条 relation

[ ] P4：计算 sim_text / sim_visual

[ ] P5：实现 first/second-order estimator
[ ] 检查 S_second 两种公式数值一致

[ ] P6：生成 metrics + figures

[ ] 人工审查第一轮结果
[ ] 如果第一轮支持 Idea，再扩展 3 datasets × 3 seeds

[ ] P7：汇总 R0_FINAL_REPORT.md

[ ] 将 R0 输出返给 ChatGPT，进入 Go / Modify / No-Go 审查
```

---

# 24. 当前阶段唯一目标

在 R0 完成前，不问：

> “FIRM-MAG 最终能不能超过 SOTA？”

现在只问四件事：

\[
\boxed{
\begin{aligned}
&1.\ \text{Relation-level multimodal functional interaction 是否存在？}\\
&2.\ \text{最优传播 action 是否具有异质性？}\\
&3.\ \text{Semantic proxy 是否不足？}\\
&4.\ \text{Second-order curvature 是否能高效逼近它？}
\end{aligned}
}
\]

四个问题得到稳定正向证据之后，正式模型才值得实现。
