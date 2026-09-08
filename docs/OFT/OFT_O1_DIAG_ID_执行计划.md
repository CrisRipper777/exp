# OFT-MAG O1 — DIAG-ID Core Scaffold & Sanity Plan

## 1. 目标

O1 只验证一个问题：

> `C / Pt / Pv` 能否作为显式 graph propagation states，在不引入旧 A0 machinery、不引入 cross-factor transfer 的前提下，构成一个正确、稳定、有效的 diagonal graph propagation scaffold。

第一版模型名：

\[
oxed{	ext{OFT-DIAG-ID}}
\]

其中 `ID` 表示 functional operator 固定为 Identity。

O1 **不实现**：
- cross-factor transition；
- dynamic operator router；
- operator mixture；
- channel routing；
- multi-scale；
- global/long-range。

---

## 2. 为什么先做 DIAG-ID

最终 OFT 会学习：

\[
(j,a)ightarrow(i,b)
\]

的 functional transition。

但在开放这种自由度之前，必须先证明：

1. P0 factorizer 可以原样复用；
2. `C/Pt/Pv` 能稳定地作为三个 graph states；
3. clean diagonal propagation 本身能正确利用图；
4. 新模型接口、训练、推理、日志没有基础问题。

---

## 3. Git 准备

建议先锁定 O0：

```bash
git status
git add docs/OFT_O0_MIGRATION_AUDIT.md experiments/oft/o0
git commit -m "O0: lock Movies A0 and DiP references"
git tag oft-o0-reference
git switch -c oft-mag
```

如果 O0 已经提交，不要重复 commit。

---

## 4. 新增文件

```text
src/models/oft_mag.py
src/models/oft_components.py
configs/model/oft_mag.yaml
tests/test_oft_mag.py
docs/OFT_O1_IMPLEMENTATION_REPORT.md
```

O1 原则上不要修改：
- `biaxis_p0.py`
- `biaxis_final.py`
- `dip.py`
- `src/tasks/nc.py`

---

## 5. 复用 P0

推荐：

```python
from .biaxis_p0 import Model as P0Model

class Model(P0Model):
    ...
```

复用：
- `SemanticFactorizer`
- reconstruction heads
- `_compute_aux`
- P0 final fusion

不要复制 Common/Private 逻辑。

构造：

```python
x_t, x_v = self._split_modalities(x)
factors = self.factorizer(x_t, x_v)

H = torch.stack(
    [factors["c"], factors["p_t"], factors["p_v"]],
    dim=1
)  # [N,3,128]
```

slot 顺序永久固定：

```text
0=C
1=Pt
2=Pv
```

---

## 6. OFT-DIAG-ID Block

对每个 factor `a` 独立：

\[
X_i^a=V_aH_i^a
\]

其中：

```text
V_a: 128 -> 128
bias=False
```

然后做 incoming neighbor mean：

\[
N_i^a=
rac{1}{|\mathcal N(i)|}
\sum_{j\in\mathcal N(i)}
X_j^a
\]

再：

\[
\Delta_i^a=D_aN_i^a
\]

其中：

```text
D_a: 128 -> 128
bias=False
```

最后：

\[
H_i^{a,+}
=
LayerNorm(H_i^a+\Delta_i^a)
\]

重要约束：
- 不加 self-loop；
- isolated node graph delta 必须严格为 0；
- ego 仅通过 residual 保留；
- 不允许任何 `a != b` graph transfer。

---

## 7. Aggregation 实现

`edge_index[0] = source`，`edge_index[1] = target`。

建议：

```python
src, dst = edge_index
msg = X[src]                      # [E,3,128]

out = torch.zeros_like(X)
out.index_add_(0, dst, msg)

deg = torch.zeros(N, device=X.device, dtype=X.dtype)
deg.index_add_(0, dst, torch.ones_like(dst, dtype=X.dtype))

out = out / deg.clamp_min(1).view(N,1,1)
```

不使用 GCN normalization。

---

## 8. O1 配置

```yaml
name: oft_mag

hidden_dim: 256
factor_dim: 128
dropout: 0.2
activation: gelu
norm: layernorm

variant: diag_id
num_layers: 1

lambda_common: 0.02
lambda_orth: 0.01
lambda_recon: 0.3
orth_fallback_batch: 16

full_graph_training: true
```

Forward：

```text
x
→ P0 factorizer
→ H0=[C,Pt,Pv]
→ 1 × DIAG-ID block
→ H1
→ concat [C1,Pt1,Pv1]
→ reuse P0 fusion
→ z
```

不做 multi-scale。

---

## 9. O1 diagnostics

加入 `aux_info`：

```text
oft_l1_c_diag_update_ratio
oft_l1_pt_diag_update_ratio
oft_l1_pv_diag_update_ratio

oft_l1_c_neighbor_norm
oft_l1_pt_neighbor_norm
oft_l1_pv_neighbor_norm
```

其中：

\[
update\_ratio_a
=
rac{\|\Delta^a\|}{\|H^a\|+\epsilon}
\]

保留全部 P0 diagnostics。

O1 不改 history CSV 也可以；现有 runner 会把 `aux_info` 写进 log。

---

## 10. Unit Tests

至少覆盖：

1. forward interface / output shape；
2. factor slot order；
3. incoming mean source→target 方向；
4. no self-loop；
5. empty graph 时 graph delta=0；
6. 只改 C source 时，graph block 内 Pt/Pv 不应被影响；
7. backward 后 factorizer、V/D、fusion 都有 gradient；
8. eval forward 与 inference 在小图上近似一致；
9. 不创建 `[E,3,3,...]` 巨型 tensor。

---

## 11. 实验顺序

### O1-A Tests

```bash
pytest tests/test_oft_mag.py -q
pytest tests/ -q
```

### O1-B 2 epoch smoke

```bash
python -m src.main   dataset=Movies task=nc model=oft_mag   num_runs=1 seed=42   task.evaluate_test=false   task.epochs=2   hydra.run.dir=outputs/oft_o1_smoke/diag_id
```

### O1-C 锁定 P0 no-graph reference

```bash
python -m src.main   dataset=Movies task=nc model=biaxis_p0   num_runs=1 seed=42   task.evaluate_test=false   task.history_path=experiments/oft/o1/p0_movies_seed42.csv
```

### O1-D OFT-DIAG-ID full sanity run

```bash
python -m src.main   dataset=Movies task=nc model=oft_mag   num_runs=1 seed=42   task.evaluate_test=false   task.history_path=experiments/oft/o1/diag_id_movies_seed42.csv
```

只跑 Movies seed42。

---

## 12. O1 判据

O0 已锁定：

```text
A0 Val Acc  ≈ 55.01%
DiP Val Acc ≈ 56.00%
```

O1 主比较：

```text
P0
vs
OFT-DIAG-ID
vs
A0
```

### Strong GO
- DIAG-ID 清楚高于 P0；
- 距 A0 大约不超过 1pp；
- update statistics 稳定。

### GO
- DIAG-ID 高于 P0；
- 无明显 catastrophic gap；
- graph updates 行为正确。

### HOLD
出现任一情况：
- DIAG-ID 实质性低于 P0；
- DIAG-ID 比 A0 低超过约 2pp；
- graph update ratio 几乎一直为 0；
- update ratio 爆炸到与 base 同量级数倍；
- Common/Private diagnostics 异常坍塌；
- inference mismatch；
- memory/NaN 等基础问题。

注意：O0 显示单 seed Macro-F1 波动可以显著大于 Accuracy，因此 O1 以 Accuracy 为主要 sanity metric，F1 只记录、不用单次结果做硬否决。

---

## 13. O1 产出

```text
docs/OFT_O1_IMPLEMENTATION_REPORT.md
experiments/oft/o1/p0_movies_seed42.csv
experiments/oft/o1/diag_id_movies_seed42.csv
```

报告包含：

1. starting / ending Git SHA；
2. files changed；
3. architecture；
4. tensor shapes；
5. aggregation implementation；
6. tests；
7. exact commands；
8. P0 result；
9. DIAG-ID result；
10. A0 / DiP locked refs；
11. update-ratio diagnostics；
12. P0 ownership diagnostics；
13. GPU peak memory；
14. warnings/anomalies；
15. O1 verdict。

完成后 STOP，不进入 O2。

---

# 14. AI Coding Agent Prompt

```text
你现在在 CrisRipper777/exp 仓库中工作。

OFT-MAG 的 O0 migration audit 已完成，verdict=GO。

你当前唯一任务是 O1：
实现并验证 clean OFT-DIAG-ID scaffold。

开始前必须阅读：
- CLAUDE.md
- README.md
- docs/OFT_O0_MIGRATION_AUDIT.md
- src/models/biaxis_p0.py
- src/models/biaxis_components.py
- src/models/factory.py
- src/tasks/nc.py
- configs/model/biaxis_p0.yaml
- 至少一个现有 graph model 的 forward/inference

科学目的：
O1 只验证 C/Pt/Pv 能否作为显式 graph propagation states，在 clean diagonal factor-preserving propagation 下稳定工作。
不要实现 cross-factor functional transition。

实现要求：

1. 新增：
   src/models/oft_mag.py
   src/models/oft_components.py
   configs/model/oft_mag.yaml
   tests/test_oft_mag.py

2. 精确复用 P0 factorizer / aux-loss 逻辑。
   推荐 subclass biaxis_p0.Model。
   不修改 P0 Common/Private 定义。
   当前 factor_dim=128，hidden_dim=256。

3. H = stack([C,Pt,Pv], dim=1)，shape [N,3,128]。
   slot 顺序固定 C/Pt/Pv。

4. O1 只有 DIAG-ID：
   X^a = V_a H^a，128->128，bias=False
   N^a = incoming neighbor mean(X^a)
   delta^a = D_a N^a，128->128，bias=False
   H_next^a = LayerNorm(H^a + delta^a)

5. 禁止任何 off-diagonal graph transition。
   不允许 C->Pt、C->Pv、Pt->C 等。

6. 不添加 self-loop。
   isolated node 的 graph delta 必须严格为 0。
   ego 通过 residual 保留。

7. 使用 edge_index source->target incoming mean。
   不使用 GCN normalization。

8. O1 num_layers=1。
   final fusion 只使用 H1。
   不做 multi-scale。

9. 保持框架接口：
   forward -> (z,None,None,aux_loss,aux_info)
   inference -> CPU z
   out_dim=256

10. aux_info 只新增：
   oft_l1_{c,pt,pv}_diag_update_ratio
   oft_l1_{c,pt,pv}_neighbor_norm
   保留 P0 diagnostics。

11. O1 原则上不要改 src/tasks/nc.py。
   当前 aux_info logging 已够用。

12. unit tests 至少验证：
   - interface/shape
   - factor slot order
   - source->target mean direction
   - no self-loop
   - empty graph delta=0
   - no cross-factor leakage before final fusion
   - gradients
   - eval forward vs inference equivalence
   - no giant edge-pair tensor

13. 运行：
   pytest tests/test_oft_mag.py -q
   pytest tests/ -q

14. Movies seed42 Val-only：
   A) 2 epoch OFT smoke
   B) full biaxis_p0 reference
   C) full OFT-DIAG-ID sanity run

15. 所有新实验：
   task.evaluate_test=false
   禁止读取 Test 做判断。

16. 生成 docs/OFT_O1_IMPLEMENTATION_REPORT.md，包含：
   - git SHA
   - files changed
   - exact architecture/formulas
   - tensor shapes
   - tests
   - exact commands
   - P0 Val Acc/F1
   - OFT-DIAG-ID Val Acc/F1
   - locked A0/DiP refs
   - update-ratio diagnostics
   - P0 factor diagnostics
   - memory/warnings
   - GO/HOLD verdict

O1 禁止：
- dynamic router
- operator mixture
- Null routing
- cross-factor transitions
- channel routing
- multi-scale
- edge attention
- old K=4 topology relations
- Gamma/OFR
- pseudo nodes
- PPR/APPNP
- global context
- new losses
- hyperparameter tuning
- seed43/44
- Toys/Grocery
- 进入 O2

如果遇到阻塞，只允许最小必要修改，并在报告中说明。
完成 O1 report 后立即 STOP。
```
