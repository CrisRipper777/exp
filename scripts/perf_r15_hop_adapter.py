"""R15-D1: frozen-A0 2-hop adapter sanity (plan §R15-D1).

Loads the fresh A0 anchor checkpoint, freezes P0/P1/P2/P3 completely, and
precomputes F0/F1/F2 under no_grad. Two matched comparisons on M/T/G seed42:

    DA0: fresh Linear head trained on F1
    DA2: zero-init small 2-hop adapter + fresh head:
         F_out = F1 + lam * W(F2 - F1)      (W d->d bias-free, lam scalar, both
         zero-init -> starts exactly at F1; backbone gets NO gradient)

Criterion: if Movies/Toys DA2-DA0 >= +0.4pp -> second hop carries real
value (C1SG failure was optimization/backbone-drift). If ~0/negative ->
close the multi-hop hypothesis. Val only; never touches test.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize  # noqa: E402

from src.data import load_mag_data  # noqa: E402
from src.models.biaxis_final import Model  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "perf_r15" / "audit"
ANCHOR_ROOT = PROJECT_ROOT / "outputs" / "perf_r15" / "anchor"
WEAK = ["Movies", "Toys", "Grocery"]


def _resolve_cfg(dataset: str) -> object:
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=biaxis_final", "seed=42",
        ])


def _load_anchor(dataset: str, device: torch.device):
    cfg = _resolve_cfg(dataset)
    data = load_mag_data(cfg, "nc", 42)
    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
    }
    ckpt = torch.load(ANCHOR_ROOT / dataset / "A0" / "seed_42" / "model.pt",
                      map_location="cpu", weights_only=False)
    model = Model(cfg, info)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return data, model, cfg


def _precompute(model, x, edge_index, num_nodes, device):
    with torch.no_grad():
        factors, _ = model._encode(x)
        f0 = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        g1 = model._graph_update(f0, edge_index, num_nodes)
        f1 = g1["f_tilde"]
        g2 = model._graph_update(f1, edge_index, num_nodes)
        f2 = g2["f_tilde"]
    return f0, f1, f2


def _train_probe(feat_fn, feat_dim: int, y: torch.Tensor, tr: torch.Tensor, va: torch.Tensor,
                 num_classes: int, params_extra, seed: int) -> float:
    """Train head (+ extra params when given) on precomputed features;
    same protocol style: AdamW 1e-3/1e-4, patience 30, best-val-acc.
    feat_fn() -> [N, feat_dim] on the model device."""
    torch.manual_seed(seed)
    device = y.device
    head = nn.Linear(feat_dim, num_classes).to(device)
    params = list(head.parameters()) + list(params_extra)
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state = None
    patience = 30
    for epoch in range(1, 301):
        head.train()
        opt.zero_grad()
        loss = crit(head(feat_fn()[tr]), y[tr])
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(feat_fn()[va]).argmax(-1)
            val = float((pred == y[va]).float().mean().item())
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            patience = 30
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return best_val


class TwoHopAdapter(nn.Module):
    """F_out = F1 + lam * W(F2 - F1); zero-init -> starts exactly at F1."""

    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.w.weight)
        self.lam = nn.Parameter(torch.zeros(1))

    def forward(self, packed):  # packed: (f1, f2) tuple
        f1, f2 = packed
        return f1 + self.lam * self.w(f2 - f1)


def main() -> None:
    parser = argparse.ArgumentParser(description="R15-D1 frozen-A0 2-hop adapter sanity")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for di, dataset in enumerate(WEAK):
        device = torch.device(f"cuda:{gpus[di % len(gpus)]}")
        data, model, _cfg = _load_anchor(dataset, device)
        x = data.x.to(device)
        ei = data.edge_index.to(device)
        n = int(x.size(0))
        f0, f1, f2 = _precompute(model, x, ei, n, device)
        y = data.y.to(device)
        tr = data.train_idx.to(device)
        va = data.val_idx.to(device)
        nc = int(data.num_classes)
        feat_dim = 3 * model.factor_dim
        f1_flat = f1.reshape(n, -1)
        for seed in (0, 1, 2):
            acc_da0 = _train_probe(lambda: f1_flat, feat_dim, y, tr, va, nc, [], seed)
            adapter = TwoHopAdapter(model.factor_dim).to(device)  # fresh per seed
            acc_da2 = _train_probe(
                lambda a=adapter: (a((f1, f2))).reshape(n, -1),
                feat_dim, y, tr, va, nc, list(adapter.parameters()), seed)
            rows.append({
                "dataset": dataset, "seed": seed,
                "DA0_val_acc": acc_da0, "DA2_val_acc": acc_da2,
                "delta_pp": 100.0 * (acc_da2 - acc_da0),
            })
            print(f"[D1] {dataset:12s} s{seed} DA0={acc_da0:.4f} DA2={acc_da2:.4f} "
                  f"delta={100*(acc_da2-acc_da0):+.2f}pp", flush=True)
            del adapter
        del f0, f1, f2, f1_flat
        torch.cuda.empty_cache()
    path = OUT_DIR / "hop_adapter_sanity.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    means = {}
    for dataset in WEAK:
        d = [r["delta_pp"] for r in rows if r["dataset"] == dataset]
        means[dataset] = sum(d) / len(d)
    lines = ["# R15-D1 FROZEN-A0 2-HOP ADAPTER SANITY", ""]
    lines.append("> 骨干全冻结、no_grad 预计算 F0/F1/F2；DA0=仅训练新 head on F1；DA2=zero-init adapter+head。Val only。")
    lines.append("")
    lines.append("| dataset | DA0 | DA2 | Δ pp |")
    lines.append("|---|---:|---:|---:|")
    for dataset in WEAK:
        a0 = [r["DA0_val_acc"] for r in rows if r["dataset"] == dataset]
        a2 = [r["DA2_val_acc"] for r in rows if r["dataset"] == dataset]
        lines.append(f"| {dataset} | {sum(a0)/len(a0):.4f} | {sum(a2)/len(a2):.4f} | {means[dataset]:+.2f} |")
    lines.append("")
    verdict = "second-hop 本身有价值（C1SG 失败更可能与 joint optimization 有关）" \
        if means.get("Movies", 0) >= 0.4 or means.get("Toys", 0) >= 0.4 else \
        "multi-hop hypothesis 关闭"
    lines.append(f"## 结论：Movies Δ={means.get('Movies', float('nan')):+.2f}pp、"
                 f"Toys Δ={means.get('Toys', float('nan')):+.2f}pp → **{verdict}**")
    lines.append("")
    (OUT_DIR / "HOP_ADAPTER_SANITY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[D1] saved -> {OUT_DIR / 'HOP_ADAPTER_SANITY.md'}")


if __name__ == "__main__":
    main()
