# R0-P0 仓库审查报告（FIRM-MAG 前置审计）

> 阶段：R0-P0 | 日期：2026-09-07 | 性质：只读审查，未修改任何代码
> 审查对象：`/hdd1/DataInHere/YHF/exp`（MAG baseline 框架，PyTorch/PyG/Hydra）

## 0. 结论速览

- **首轮数据集：Grocery × seed 42**（NC，双模态完整，结构健康，无标签缺口/协议坑）
- **Probe 走 factory 注册**（`src/models/simple_mag_probe.py`），直接复用现有 `run_nc` 训练协议，**零修改现有代码**
- 新增 5 个独立文件即可：模型、config、训练脚本（其实可省，用 `src.main`）、counterfactual 脚本、analyze 脚本、单元测试
- 结果目录：新建 `experiments/r0/`（固定路径，供分析脚本确定性读取）
- 已知风险：edge direction、self-loop 双计、MAGB image 特征磁盘 f64、文本特征未归一化（norm≈12 vs 图像≈1）

---

## 1. 数据加载与数据集规格

**入口**：`load_mag_data(cfg, task_name, seed)` — `src/data/loaders.py:264-273`，按 `cfg.dataset.source` 分发：
- `_load_magb`（loaders.py:110-190）：DGL graph（`dgl.load_graphs`，loaders.py:44-56）+ 独立 `.npy` 特征；`x = cat([text, image])`（:121）
- `_load_mmgraph`（loaders.py:193-261）：联合 CLIP `.pt`，`_split_modalities_from_joint`（:74-87）切 `x_t = x[:, :text_dim]`、`x_i = x[:, text_dim:]`，注意该函数返回顺序为 `(x_i, x_t)`（:87）

统一返回 `MAGData`（`src/data/types.py:17-41`）：`x / x_i(visual) / x_t(text) / edge_index / y / train_idx / val_idx / test_idx / num_classes / num_nodes / input_dim`。

**NC 候选数据集规格**（特征 shape 为 mmap 实测）：

| 数据集 | 节点 | 有向边 | text | visual | 类 | 备注 |
|---|---|---|---|---|---|---|
| Movies | 16,672 | 109,195 | 768 f32 | 768 f32 | 20 | 极不均衡（min 类 67 样本） |
| **Grocery** | **17,074** | **85,670** | **768 f32** | **768 f32** | **20** | **LCC 94.18%、0 孤立点、不均衡比 34.03** |
| Toys | 20,695 | 63,443 | 768 f32 | 768 f32 | 18 | — |
| Reddit-S | 15,894 | 283,080 | 768 f32 | 768 f32 | 20 | LCC 仅 1.21%，高度碎裂，排除 |
| ele-fashion | 97,766 | 199,602 | 512 | 512 | 12(yaml) | 实际 11 类（label 5 缺失），排除首轮 |
| books-nc | 685,294 | 7,235,048 | 512 | 512 | 11 | 2.6GB 特征、label-10 协议外节点，排除 |

## 2. 特征协议

- 字段：`x_t` = text、`x_i` = visual；`x = [x_t | x_i]`（两 loader 一致）
- dtype：一律 float32。**MAGB image 磁盘 f64**，必须走 loader（`_load_numpy_feature` `astype(float32)`，loaders.py:29-33）
- 加载时**不**做 L2 normalize（无 dataset yaml 设 `feature_norm`）。实测：图像 CLIP 已 L2（norm≈1），文本 RoBERTa mean-pool **未归一化**（norm≈12.25）→ Probe 前处理须显式决定并写入 config/报告
- 注意：CLAUDE.md 中 unigraph2 相关内容已过时（模型已移除，`src/models/` 无 unigraph2.py）

## 3. edge_index 约定

- **PyG 标准 [2, E]：row0=source，row1=target**（loaders.py:51-52）
- 预处理 `preprocess_edge_index`（graph_utils.py:18-30）：`ensure → remove_self_loops → to_undirected → (可选) add_self_loops`。NC 数据实际为：**无自环、无重复、双向**（`make_undirected: true, add_self_loops: false`）
- gcn 内部 `gcn_norm(..., add_self_loops=True)` 加自环做归一化（gcn.py:38-46），不写回 edge_index；lgmrec/dgf/dmgc 部分自加；mlp/mmgcn/sage 不加
- **R0 Probe 结论：loader edge_index 无自环 → self 项单独建模（h_self）即安全，切勿自行 add_self_loops**（会与 gcn 系内部自环语义重复）。边权重用 `1/sqrt(d_i d_j)`（Grocery 双向图上 d=入度=度）

## 4. Split

- MAGB NC：`load_or_create_nc_split`（splits.py:149-165）→ 缓存文件 `{split_root}/{Dataset}_nc_seed{seed}_train0.6_val0.2.pt`（stratify，train 0.6/val 0.2/test 0.2），seed 42-46 均已存在 → **R0 用同一 cfg+seed 即命中同一 split 文件，天然复用**
- MM-Graph NC：官方 `split.pt`（不受 seed 影响）
- R0 保证：R0 脚本走 `load_mag_data(cfg, "nc", seed)`，cfg 取自同一 dataset yaml

## 5. NC 训练协议（src/tasks/nc.py）

- 入口 `run_nc`（:439-468）：`num_runs` 次独立 run，seed = `cfg.seed + run_id`（`set_seed`，:106-107），输出 mean±std（无跨 run 选 best）
- 单 run：`data_info = {input_dim, num_nodes, num_classes, text_dim, visual_dim, y, train_idx}`（:110-120）→ `build_model` → **classifier 由 task 外加** `nn.Linear(model.out_dim, num_classes)`（:121-122）→ AdamW（默认 lr 1e-3/wd 1e-4，模型 yaml 可覆盖 `cfg.model.lr`，:46-55）→ CrossEntropy
- 训练模式默认 `full_graph`（整图 forward，train_idx 上 CE）；grad_clip 默认 null；每 epoch 训练 + eval（eval_every=1）
- **early stopping 用 val accuracy**（patience 30），best-val 时 clone_state_dict 暂存 model+head（CPU）
- 恢复 best state 后按 `inference_mode`（full/layerwise）评估 test
- **checkpoint 保存机制已内建**（:406-422）：`task.save_ckpt_path` 非空则 `torch.save({task, seed, model_state, head_state, data_info})`，默认 null → **R0-P2 只需 override `task.save_ckpt_path`，无需改 nc.py**
- ckpt 键与现有分析层完全对齐（`src/analysis/perf_r0_utils.py:63-71` 即按此键加载）

## 6. 最简单 baseline

- **mlp**（mlp.py:9-51）：Linear 堆叠 + 层间 norm/act/dropout；`out_dim = hidden_dim`；aux_loss 用 `z.new_tensor(0.0)` 占位
- **gcn**（gcn.py:13-96）：`GCNConv(normalize=False, add_self_loops=False)` + 显式 `gcn_norm`；`inference()` 为 layerwise
- `common.py`：`get_activation` / `make_norm`（none/batchnorm/layernorm）
- 官方 preset：mlp 128/2 层/d0.5；gcn 256/3 层/d0.2/lr 5e-3

## 7. 注册机制

`factory.py:6-8`：`importlib.import_module(f"src.models.{cfg.model.name}")` → `module.Model(cfg, data_info)`。**规则：模块文件名 == cfg.model.name，导出 `class Model`**，无 registry 表。→ Probe 满足该契约即可被 `python -m src.main model=simple_mag_probe` 直接训练。

## 8. 输出/日志

- Hydra 输出 `outputs/YYYY-MM-DD/HH-MM-SS/`：`results.json = {metric: {mean, std}}`、`main.log`、`.hydra/`
- 框架默认无 checkpoint 落盘（除 nc.py/lp.py 的 `save_ckpt_path` 机制、split 缓存、analysis 工具）

## 9. 工具与测试约定

- `src/utils/`：`device.py`（auto→cuda:0）、`logging.py`、`metrics.py`、`seeds.py`（set_seed 全套）、`summary.py`（mean_std ddof=0、count_parameters）
- `src/analysis/perf_r0_utils.py`：`R0Setup` / `resolve_cfg`（hydra compose 拼 overrides）/ `load_setup`（ckpt→model+head）/ `assert_no_test_access` / `write_csv` — 可直接复用
- `tests/`：纯 pytest 函数式，**无 conftest、无 data fixture**，合成小张量 + `OmegaConf.create`/`_CfgNode` 伪造 cfg；GPU 用例 `pytest.skip` 保护
- 默认配置：`configs/config.yaml`（seed 42、num_runs 3、device cuda:0、paths.data_root=/hdd1/DataInHere/YHF/data、split_root=…/MAGB_split）；`configs/task/nc.yaml`（epochs 300、lr 1e-3、wd 1e-4、patience 30、batch_size 1024、inference_mode full）

## 10. R0 最小侵入落地方案

新增文件（不修改任何现有文件）：

| 文件 | 内容 |
|---|---|
| `src/models/simple_mag_probe.py` | `class Model(cfg, data_info)`：f_T/f_V 投影（Linear→ReLU→Dropout）→ `h_self=W_s[h_T‖h_V]` → 单层 additive MP（`δ_T[r]=alpha_r·W_msg_T·hT[j]`，alpha=1/sqrt(d_i·d_j)，按 `edge_index[1]` scatter）→ `z=h_self+Σδ`；`out_dim`；forward 5 元组；`forward_diag` 返回 h_text/h_visual/h_self/z_full/edge_delta_text/edge_delta_visual/edge_norm（delta[r] 严格对 edge_index[:,r]） |
| `configs/model/simple_mag_probe.yaml` | name/hidden_dim/dropout/lr/wd/projector 细节 |
| `scripts/train_r0_probe.py` | 轻量驱动：hydra compose + `num_runs=1 seed=42` + `task.save_ckpt_path=<r0>/checkpoints/…`（实际可完全用 `python -m src.main` 等价完成） |
| `scripts/run_r0_counterfactual.py` | 冻结 ckpt → val 分层采样 → 四干预 exact loss/Q/S + 一/二阶估计器 → `r0_relation_diagnostics.csv` |
| `scripts/analyze_r0.py` | 统计 + fig1-6 + metrics/summary |
| `tests/test_simple_mag_probe.py` | 合成图：shape / edge 对应 / `z_full[i]≈h_self[i]+Σδ` <1e-5 |

**红线**：不动 loaders/splits/dataset yaml/nc.py 默认行为；R0 分析只用 val，不用 test label。

**风险点**：① edge direction（source→target，按 target scatter）；② self-loop 双计（probe 不自加自环）；③ MAGB image 磁盘 f64 必须走 loader；④ 文本特征 norm≈12 未归一化 → 决定是否 normalize 并记录；⑤ ele-fashion/books-nc 有标签缺口或超大图 → 首轮 Grocery 避开；⑥ 采样必须分层（plan §5.2），不整体均匀随机。

## 11. 首轮方案（待用户确认后执行）

- **dataset=Grocery, seed=42**（split 文件已存在：`Grocery_nc_seed42_train0.6_val0.2.pt`）
- Probe：hidden=128、单层、1/sqrt(d_i d_j)、文本/图像投影前均 L2-normalize（可选，实验决定后写入报告）
- sanity 参照：同 split 同 seed 跑 `gcn`、`mlp`（各 1 run）对比
- 输出：`experiments/r0/{configs,checkpoints,results/Grocery/seed_42/…}`

## 12. 环境与仓库

- 环境：conda `yhf_env`（torch 2.4.0+cu121 / PyG 2.7.0 / dgl 2.4.0）；GPU：cuda:0/cuda:1（RTX 3090 24GB ×2）
- exp/ 本无 git；已 `git init -b main` 于 exp/ 并完成框架初始导入 commit `272854c`；后续每阶段独立 commit（plan §19）
