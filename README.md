# MAG_baseline

`MAG_baseline` is a PyTorch / PyG / Hydra baseline framework for multimodal attributed graph experiments.

It only supports:

- Node Classification (NC)
- Link Prediction (LP)

It uses frozen features from `../data` and does not train BERT, ViT, CLIP, Qwen-VL, or other large encoders.

Implemented models:

- `mlp`
- `gcn`
- `sage`
- `mmgcn`
- `mgat`
- `unigraph2`
- `dip`

## Datasets

MAGB:

- `Movies`: NC + LP
- `Toys`: NC + LP
- `Grocery`: NC + LP
- `Reddit-S`: NC + LP

MM-Graph:

- `sports-copurchase`: LP
- `cloth-copurchase`: LP
- `books-lp`: LP
- `ele-fashion`: NC
- `books-nc`: NC

## Run

Use the requested environment:

```bash
conda activate yhf_env
cd /hdd1/DataInHere/YHF/MAG_baseline
```

Run one NC experiment:

```bash
python -m src.main dataset=Movies task=nc model=mlp num_runs=3
```

Run one LP experiment:

```bash
python -m src.main dataset=sports-copurchase task=lp model=sage num_runs=3
```

Run DiP:

```bash
python -m src.main dataset=ele-fashion task=nc model=dip num_runs=3
python -m src.main dataset=sports-copurchase task=lp model=dip num_runs=3
```

Smoke test with a short run:

```bash
python -m src.main dataset=Movies task=nc model=mlp num_runs=1 task.epochs=1 task.max_train_batches=2 device=cpu
```

## Notes

- MAGB `*Graph.pt` files are DGL graphs, so the loader converts DGL graphs to PyG `edge_index`.
- MAGB NC/LP splits are generated once under `../data/MAGB_split` when missing.
- MM-Graph NC/LP tasks use the official split files shipped in each dataset directory.
- LP evaluation ranks one positive target against fixed negative targets and reports MRR / Hits@1 / Hits@3 / Hits@10.
- Test metrics are computed once after training, by reloading the best validation checkpoint.
- Dataset graphs do not add self-loops by default; models own their self-loop policy, e.g. GCN adds them internally.
- `model=mlp` does not use graph sampling: NC uses node mini-batches and LP uses edge mini-batches.
- GNN models such as `gcn` and `sage` use PyG `NeighborLoader` / `LinkNeighborLoader`.
- `model=dip` trains with full-graph message passing by default (`model.full_graph_training=true`) so pseudo nodes aggregate graph-level modality context rather than sampled subgraphs. LP still uses edge-label mini-batches and removes the current positive labels from the full message graph.
- `model=dip` implements the exact DiP global pseudo-node recurrence; its `layerwise` inference API currently falls back to the exact full-graph DiP pass because pseudo-node states depend on all nodes at each recurrent step.
