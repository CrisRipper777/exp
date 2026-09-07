# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAG_baseline is a PyTorch/PyG/Hydra framework for benchmarking graph neural networks on **multimodal attributed graphs** (nodes with both image and text features). Two tasks: **Node Classification (NC)** and **Link Prediction (LP)**. Features are frozen — no large encoder training.

## Commands

**Run from project root** (`MAG_baseline/`). Environment: `conda activate yhf_env`.

```bash
# Single experiment
python -m src.main dataset=Movies task=nc model=mlp num_runs=3
python -m src.main dataset=sports-copurchase task=lp model=sage num_runs=3

# Smoke test (fast sanity check)
python -m src.main dataset=Movies task=nc model=mlp num_runs=1 task.epochs=1 task.max_train_batches=2 device=cpu

# Tests
pytest tests/

# Batch runs
python scripts/run_all_nc.py    # all NC datasets x all models
python scripts/run_all_lp.py    # all LP datasets x all models

# Generate MAGB splits
python scripts/make_magb_splits.py
```

## Architecture

**Entry point:** `src/main.py` (Hydra `@hydra.main`) → loads data → dispatches to `run_nc()` or `run_lp()` → saves `results.json`.

**Data layer** (`src/data/`):
- `loaders.py`: Two loaders — `_load_magb()` (DGL graph + separate `.npy` features) and `_load_mmgraph()` (joint CLIP `.pt` features). Both return `MAGData` dataclass.
- `splits.py`: Auto-generates splits for MAGB datasets; MM-Graph uses shipped split files.
- `graph_utils.py`: Edge index manipulation (canonicalize, preprocess).

**Models** (`src/models/`):
- `factory.py`: Dynamic import by name — each model module must export `class Model(cfg, data_info)`.
- 6 models: `mlp`, `gcn`, `sage`, `mmgcn`, `mgat`, `unigraph2` (each has a config in `configs/model/`).
- All encoders implement `forward()` → `(z, None, None, aux_loss, aux_info)` and `inference(x, edge_index, device, batch_size)` for full-graph eval.
- `predictor.py`: `LinkPredictor` MLP — scores (src, dst) via element-wise product.
- `common.py`: Shared `make_norm()` (defaults to BatchNorm1d), `get_activation()`.
- GCN adds self-loops internally via `gcn_norm`; others don't. Dataset configs set `add_self_loops: false`.

**Model interface contract** (`Model(cfg, data_info)`):
- `data_info` dict keys: `input_dim`, `num_nodes`, `num_classes`, `text_dim`, `visual_dim`.
- `forward(x, edge_index)` → `(z, _, _, aux_loss, aux_info)`. `z` shape `(num_nodes, hidden_dim)`.
- `inference(x, edge_index, device, batch_size)` → `z` on CPU. Must replicate forward logic without training-time augmentations (masking, dropout).
- `out_dim` attribute used by task runners for classifier/predictor input size.

**UniGraph2 specifics** (`src/models/unigraph2.py`):
- Architecture: modality projectors → L2-normalized mean fusion → MoE (8 experts, top-2) → GAT layers → domain-specific reconstruction + SPD loss.
- Has BatchNorm between GAT layers (not on last layer), matching other models' convention.
- `_fuse_features` applies L2 normalization per modality before averaging — critical for MAGB datasets where text features have ~12x larger norm than image features.
- Training losses: `aux_loss = reconstruction_loss + lambda_spd * spd_loss`.
- SPD loss uses BFS from random source nodes (CPU), bounded by `spd_k`, `spd_num_sources`, `spd_max_pairs`.
- `mask_token` is a learnable parameter replacing dropped features during training.

**Task runners** (`src/tasks/`):
- `nc.py`: Model + linear classifier. Metrics: Accuracy, Macro-F1. Early stopping on val accuracy. Gradient clipping `max_norm=1.0`.
- `lp.py`: Model + `LinkPredictor`. Metrics: MRR, Hits@1/3/10. Early stopping on val MRR. Filtered negative sampling (negatives excluded from all positive edges). Positive label edges excluded from message-passing graph. Gradient clipping `max_norm=1.0`.
- Both run `num_runs` independent runs with seeds `seed + run_id`, select best by val metric, evaluate on test.
- `inference.py`: Two modes — `full` (entire graph on GPU) and `layerwise` (batched via NeighborLoader, memory-efficient).

## Configuration (Hydra)

Root: `configs/config.yaml` with config groups `dataset/`, `task/`, `model/`. Key overrides:
- `num_runs`, `seed`, `device`
- `task.epochs`, `task.lr`, `task.batch_size`, `task.patience`, `task.num_neighbors`
- `model.hidden_dim`, `model.num_layers`, `model.dropout`, `model.num_experts`
- `task.inference_mode`: `full` (default, fast but GPU-heavy) or `layerwise` (CPU batched, memory-safe)
- Paths: `paths.data_root`, `paths.split_root`

Output: `outputs/YYYY-MM-DD/HH-MM-SS/` containing `results.json`, `main.log`, `.hydra/` config snapshot.

## Datasets

- **MAGB** (4): Movies, Toys, Grocery, Reddit-S — both NC and LP. DGL `.pt` graph + `.npy` features. Splits auto-generated under `../data/MAGB_split/`. Small graphs (~10K-130K nodes).
- **MM-Graph** (5): sports-copurchase, cloth-copurchase, books-lp (LP); ele-fashion, books-nc (NC). Joint CLIP features `.pt`, official splits. **books-nc (685K nodes) and books-lp (636K nodes) are large — see OOM section below.**

Data root: `/hdd1/DataInHere/YHF/data` (set via `paths.data_root` in config).

## Large Graph OOM Handling

UniGraph2's MoE computes all 8 expert outputs before top-k selection. On large graphs (books-nc, books-lp), default settings cause OOM. Adjust:

```bash
# Example for books-nc
python -m src.main dataset=books-nc task=nc model=unigraph2 \
  task.batch_size=256 task.num_neighbors=5 task.inference_mode=layerwise
```

Key levers (in priority order):
1. `task.batch_size`: 1024 → 256 or 128
2. `task.num_neighbors`: 15 → 5 or 3 (subgraph size is O(n^layers))
3. `task.inference_mode`: `layerwise` for memory-safe eval
4. `model.num_experts`: 8 → 4 (MoE memory scales linearly)
5. `use_spd_loss=false`: disables BFS adjacency construction

## Key Dependencies

`torch==2.4.0+cu121`, `torch-geometric==2.7.0`, `dgl==2.4.0+cu121`, `hydra-core==1.3.2`, `numpy==2.4.3`, `scikit-learn`
