from __future__ import annotations

import torch

from src.data import MAGData


VALID_INFERENCE_MODES = {"full", "layerwise"}


def resolve_inference_mode(cfg) -> str:
    mode = str(cfg.task.get("inference_mode", "full")).strip().lower()
    if mode not in VALID_INFERENCE_MODES:
        valid = ", ".join(sorted(VALID_INFERENCE_MODES))
        raise ValueError(f"task.inference_mode must be one of [{valid}], got {mode!r}")
    return mode


@torch.no_grad()
def infer_all_embeddings(
    model,
    data: MAGData,
    device: torch.device,
    uses_graph: bool,
    batch_size: int,
    inference_mode: str,
) -> torch.Tensor:
    mode = str(inference_mode).strip().lower()
    if mode not in VALID_INFERENCE_MODES:
        valid = ", ".join(sorted(VALID_INFERENCE_MODES))
        raise ValueError(f"task.inference_mode must be one of [{valid}], got {inference_mode!r}")

    model.eval()
    edge_index = data.edge_index if uses_graph else None
    if mode == "layerwise":
        return model.inference(data.x, edge_index, device=device, batch_size=batch_size)

    if hasattr(model, "_batch_n_id"):
        model._batch_n_id = None
    x = data.x.to(device)
    edge_index = edge_index.to(device) if edge_index is not None else None
    z, _, _, _, _ = model(x, edge_index)
    return z.detach().cpu()
