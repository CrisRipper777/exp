from __future__ import annotations

import torch
from sklearn.metrics import f1_score


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    target = labels.detach().cpu().numpy()
    return float(f1_score(target, pred, average="macro", zero_division=0))


def format_pct(value: float) -> float:
    return value * 100.0


def mrr_hits_from_ranks(ranks: torch.Tensor) -> dict[str, float]:
    ranks = ranks.float()
    return {
        "mrr": float((1.0 / ranks).mean().item()),
        "hits@1": float((ranks <= 1).float().mean().item()),
        "hits@3": float((ranks <= 3).float().mean().item()),
        "hits@10": float((ranks <= 10).float().mean().item()),
    }
