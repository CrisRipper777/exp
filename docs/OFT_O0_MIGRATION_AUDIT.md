# OFT-MAG O0 — 新工作目录迁移审计 + Reference Lock

日期：2026-09-08
范围：O0（迁移审计 + Reference Lock）。未实现 OFT，未进入 O1，未做超参搜索。
协议：**Val-only**（`task.evaluate_test=false`，所有新运行均不读取/不记录 Test 指标）。

## 最终 Verdict：**GO**

新工作目录 `exp` 的代码/数据/协议与历史 Reference（biaxis_final=“A0”、dip=“DiP”）
一致：biaxis_final 复现 run2 与历史值 **位级一致**（val_acc 0.550090，best epoch 74），
dip 两条新运行 {56.27, 56.21} 与历史单次运行 56.00 相差 +0.21/+0.27pp，落在协议噪声带
（±0.3pp）内。参考锁成立，O1 可启动。

---

## 1. Git SHA 与环境

| 项 | 值 |
|---|---|
| 新工作目录 | `/hdd1/DataInHere/YHF/exp`（repo `exp`，remote `git@github.com:CrisRipper777/exp.git`） |
| branch | `main`（领先 origin/main 1 个提交） |
| HEAD | `4d3109fb7078e095abded4f7d6b7bed866b95155`（r0-p6.6） |
| 历史 biaxis repo | `/hdd1/DataInHere/YHF/0901/0901`（HEAD `024292a`，benchmark 时代 commit `3112762` P3++ / `d9ee48e` initial） |
| Python | 3.12.13（conda-forge） |
| torch | 2.4.0+cu121 |
| torch_geometric | 2.7.0 |
| dgl | 2.4.0+cu121 |
| hydra-core | 1.3.2 |
| numpy | 2.4.3 |
| scikit-learn | 1.8.0 |
| CUDA | 12.1 available |
| GPU | NVIDIA GeForce RTX 3090（25.4GB，cuda:0） |
| data_root / split_root | `/hdd1/DataInHere/YHF/data` / `/hdd1/DataInHere/YHF/data/MAGB_split`（与历史 repo 相同路径） |

## 2. Tests

`pytest tests/` → **56 passed, 4 warnings, 98.77s**（exit 0）。

注：exp 未携带 0901 的 biaxis 系列单元测试（test_biaxis_*.py、test_osra_*.py、
test_perf_r2*.py 等 25 个文件），保留的是 protocol/metrics/splits/memory-checkpoint/dgf-dmgc/lgmrec
及 R0 相关测试。非阻塞（锁定的两个模型代码文件与 0901 一致，见 §6），已在问题清单记录。

## 3. 执行的完整命令

环境：`conda activate yhf_env`，cwd `/hdd1/DataInHere/YHF/exp`。

```bash
# A. 审计
git status && git branch --show-current && git rev-parse HEAD
python -c "<torch/PyG/dgl/hydra/numpy/sklearn/CUDA/GPU 版本信息>"
ls /hdd1/DataInHere/YHF/data /hdd1/DataInHere/YHF/data/MAGB_split
python -m pytest tests/ -q                                       # 56 passed

# B. Smoke（各 2 epochs，val-only）
python -m src.main dataset=Movies task=nc model=biaxis_final num_runs=1 seed=42 \
  task.evaluate_test=false task.epochs=2 hydra.run.dir=outputs/oft_o0_smoke/biaxis_final
python -m src.main dataset=Movies task=nc model=dip num_runs=1 seed=42 \
  task.evaluate_test=false task.epochs=2 hydra.run.dir=outputs/oft_o0_smoke/dip

# C. Reference reproduction（正式，val-only，历史文件为锁定产物）
python -m src.main dataset=Movies task=nc model=biaxis_final num_runs=1 seed=42 \
  task.evaluate_test=false task.history_path=experiments/oft/o0/biaxis_final_movies_seed42.csv
python -m src.main dataset=Movies task=nc model=dip num_runs=1 seed=42 \
  task.evaluate_test=false task.history_path=experiments/oft/o0/dip_movies_seed42.csv

# C 附加：run-to-run 噪声测量（同 seed 复跑，输出单独目录，不覆盖锁定文件）
python -m src.main dataset=Movies task=nc model=biaxis_final num_runs=1 seed=42 \
  task.evaluate_test=false task.history_path=experiments/oft/o0/repro_check/biaxis_final_r2.csv
python -m src.main dataset=Movies task=nc model=dip num_runs=1 seed=42 \
  task.evaluate_test=false task.history_path=experiments/oft/o0/repro_check/dip_r2.csv
```

输出目录：`outputs/oft_o0_smoke/{biaxis_final,dip}/`、`outputs/2026-09-08/00-17-31/`（biaxis run1）、
`outputs/2026-09-08/00-19-16/`（dip run1）、`outputs/2026-09-08/00-23-*/`（R2）。

## 4. 新结果（exp，2026-09-08，Movies seed42，val-only）

| 模型 | run | best epoch | Val Acc | Val Macro-F1 | early-stop epoch |
|---|---|---|---|---|---|
| biaxis_final (A0) | run1（锁定） | 74 | 55.0990% | 46.67% | 104 |
| biaxis_final (A0) | run2（噪声测量） | 74 | **55.0090%** | 46.37% | 104 |
| dip (DiP) | run1（锁定） | 78 | 56.2687% | 46.62% | 108 |
| dip (DiP) | run2（噪声测量） | 88 | 56.2088% | 48.06% | 118 |

params（model+head）：biaxis_final 1,400,824 / dip 8,170,620 —— 与历史完全一致。

## 5. 旧 Reference（0901/0901，Movies seed42）

| 模型 | 来源 | best epoch | Val Acc | Val Macro-F1 | early-stop epoch | params |
|---|---|---|---|---|---|---|
| biaxis_final | `outputs/final_nc_benchmark/main/Movies/biaxis_final/seed_42/`（2026-09-03 13:25，commit 3112762 P3++） | 74 | 55.0090% | 46.38% | 104 | 1,400,824 |
| dip | `outputs/baseline_nc/Movies/dip/seed_42/`（2026-09-02 02:26） | 82 | 55.9988% | 46.49% | 112 | 8,170,620 |

（旧 run 均为 evaluate_test=true 的正式 benchmark 运行；此处仅取 Val 指标，不使用其 Test 指标做任何判断。）

## 6. 差值

| 模型 | ΔVal Acc（新 run1 / run2 − 旧） | ΔVal F1 | Δbest epoch |
|---|---|---|---|
| biaxis_final | +0.090pp / **0.000pp** | +0.29pp / −0.01pp | 0 / 0 |
| dip | +0.270pp / +0.210pp | +0.13pp / +1.57pp | −4 / +6 |

**分析**：

- **biaxis_final**：run2 与历史 **位级一致**（0.5500898… = 0.5500898…，epoch 74，
  early-stop 104 全部相同）。run1 的 +0.09pp 属 exp 内部 run-to-run 噪声。
- **dip**：两条新运行一致偏高历史约 +0.21/+0.27pp（新运行之间仅差 0.06pp）。
  dip.py 自 2026-06-22 起 **字节级不变**（git 验证），训练循环仅增加 opt-in 开关（见下）；
  历史 dip 运行于 0901 仓库 initial commit（09-02 12:48）**之前**（09-02 02:26），
  其 nc.py/data 侧代码无法从 git 精确还原。best epoch（78/82/88）差异来自 dip 的
  val 曲线在 75–90 epoch 处呈平坦高原（55.5–56.3 区间），patience-30 早停对
  微小曲线噪声敏感。0.27pp < 0.3pp 容差，判定为协议噪声带内。
- 全 epoch 曲线对照：biaxis mean|ΔValAcc|=0.113pp / max 2.2pp（早期 epoch）；
  dip mean|ΔValAcc|=0.319pp / max 1.8pp。

## 7. 代码一致性审计（Reference Lock 核心）

`exp` 与历史 biaxis repo `0901/0901`（HEAD 024292a）逐文件 diff：

| 文件 | 状态 | 说明 |
|---|---|---|
| src/models/dip.py | **字节级一致**（自 06-22 起未变） | DiP 锁定成立 |
| src/models/biaxis_final.py / biaxis_p3_components.py / biaxis_components.py | 与 0901 HEAD 一致 | 锁定成立 |
| src/models/biaxis_p3.py | 与 benchmark 时代（3112762）差 55 行 | 全部为 2026-09-04 显存修复：context+scorer 段 activation checkpointing（`p3.memory_checkpoint: true`）。声称位级等价（该段无 dropout/随机性）；biaxis 复现位级一致佐证 |
| src/models/biaxis_p1_components.py / biaxis_p2_components.py | 差 10/3 行 | 显存修复：`features[src]` gather 提出关系循环（值位级相同） |
| src/tasks/nc.py | 与 0901 HEAD 差 OSRA hook；与 benchmark 时代差 R1.5 opt-in 功能 | 详情见下 |
| src/models/osra*.py、configs/model/osra.yaml | exp 为旧版 | 不影响锁定模型（osra 不在本次范围） |
| configs/config.yaml、dataset/Movies.yaml、task/nc.yaml、model/biaxis_final.yaml、model/dip.yaml、biaxis_p0.yaml | 与 0901 **一致** | 配置锁定成立 |
| src/main.py、src/data/*、src/utils/* | 与 0901 **一致** | — |

nc.py 相对 benchmark 时代（3112762/d9ee48e）的差异全部为 **opt-in、默认关闭** 的
R1.5 功能，逐一核验对锁定模型无训练语义影响：

1. `_scheduler_step`：`task.scheduler=null`（默认，两代 config 一致）→ 每次调用即返回，no-op。
2. data_info 新增 `"y"`/`"train_idx"` key：biaxis_p3/dip 的 `Model.__init__` 均不消费（grep 验证）。
3. `evaluate_test=false`：仅跳过 test 访问/指标；true 时行为与旧版逐行相同。
4. `history_path` 记录：只读性插桩（额外 `.item()`/`argmax` 计算，不参与梯度）。
5. 训练损失数学一致：`loss = ce + aux_weight * aux`，旧版 `criterion(logits, labels) + aux_weight * aux_loss` 逐项等价。

（0901 现行 nc.py 另有 OSRA 分组 LR/warm-head hook，exp 未携带——两个锁定模型均无
`osra_param_groups`/`warm_head_state` 属性，走统一 LR 路径，行为一致。）

## 8. 数据 / split / config 一致性

| 项 | 历史（旧 log） | 新运行（exp log） | 一致 |
|---|---|---|---|
| graph | 16672 nodes / 160802 edges | 16672 / 160802 | ✓ |
| 特征 | X (16672,1536) fp32；X_i 768 / X_t 768 | 相同 | ✓ |
| NC split | train=10003 / val=3334 / test=3335 | train=10003 / val=3334（test 未访问） | ✓ |
| split 文件 | `/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt`（5月3日，未变） | 同一文件 | ✓ |
| dataset 路径 | MoviesGraph.pt / roberta_base_512_mean / clip-vit-large-patch14 | 同一路径（文件 4月2日，未变） | ✓ |
| task config | 300ep / patience30 / AdamW lr1e-3 wd1e-4 / full-graph / eval_every1 | 相同（nc.yaml 一致） | ✓ |
| dip config | d_model256 q256 n_q8 mp_hops3 pnode512/64 dropout0.1 path_integral / lr1e-3 wd1e-5 / full_graph | 相同（dip.yaml 一致） | ✓ |
| biaxis_final config | P0(0.02/0.01/0.3) + P1(K=4) + P2(null_softmax ε0.2) + P3(full_interaction) | 相同（biaxis_final.yaml 一致） | ✓ |
| seed/device | seed42 / cuda:0 / num_runs1 | 相同 | ✓ |

未重建 split、未改动数据、未改动评价协议。

## 9. 问题清单（记录，不阻塞）

1. **文档路径仍写 `MAG_baseline`**：exp 内 CLAUDE.md/README.md 的目录名仍是
   `MAG_baseline`（cd 路径、Run 章节等），实际工作目录已为 `exp`。按 O0 指示仅记录，
   未重写文档（不阻塞运行；CLAUDE.md 中 data_root 等路径均正确）。
2. **biaxis 单元测试未随迁移带入 exp**（25 个 test 文件缺省）。模型代码本身一致且
   56 个现存测试全绿，锁定成立；后续如需动 biaxis 代码建议补回相应测试。
3. **历史 dip 运行早于 0901 仓库 initial commit**（09-02 02:26 vs 12:48），其
   runner 侧代码状态无法从 git 精确还原；dip.py 字节级一致 + 差值在噪声带内，已排除阻塞。
4. **run-to-run 非位级确定**：同 seed 同环境复跑 biaxis 出现 ±0.09pp、dip ±0.06pp
   波动（`set_seed` 未设 `cudnn.deterministic`）。O0 参考锁以「噪声带」而非「位级复现」
   定义（dip 三条运行 [55.999, 56.269] = 0.27pp 带宽）。
5. exp 为 `git status` 干净 + `experiments/oft/`（本轮产物）未跟踪；`outputs/` 已 ignore。

## 10. O0 Verdict

**GO**。

- Reference Lock 成立：Movies seed42 下 `model=biaxis_final`（A0）与 `model=dip`（DiP）
  的 Val 参考值锁定为 **55.01% / 56.00%**（历史值，exp run2 位级复现 A0），
  噪声带 ±0.3pp。
- 锁定文件：`experiments/oft/o0/biaxis_final_movies_seed42.csv`、
  `experiments/oft/o0/dip_movies_seed42.csv`（run1，逐 epoch 历史）。
- 下一阶段（O1）以 `model=biaxis_final` 为主参考（A0）、`model=dip` 为最强 baseline
  参考，统一 Val-only 协议（`task.evaluate_test=false`）。
