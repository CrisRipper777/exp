from __future__ import annotations

import copy

import torch
from omegaconf import ListConfig


AUX_INFO_KEYS = (
    "mean_r_text",
    "mean_r_visual",
    "mean_p_self",
    "mean_p_struct",
    "mean_p_proto",
    "mean_gamma",
    "proto_loss",
    "gate_loss",
    # P0 biaxis factorizer diagnostics (only emitted by model=biaxis_p0)
    "p0_common_loss",
    "p0_orth_loss",
    "p0_recon_loss",
    "p0_common_sim",
    "p0_private_sim",
    "p0_c_norm",
    "p0_pt_norm",
    "p0_pv_norm",
    "p0_cp_overlap_t",
    "p0_cp_overlap_v",
)


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def load_state_dict_cpu(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(copy.deepcopy(state_dict))


def resolve_num_neighbors(cfg) -> list[int]:
    num_layers = int(cfg.model.get("num_layers", 1))
    if num_layers < 1:
        raise ValueError(f"model.num_layers must be >= 1, got {num_layers}")

    raw_neighbors = cfg.task.get("num_neighbors", -1)
    if isinstance(raw_neighbors, str):
        raw_neighbors = raw_neighbors.strip()
        if raw_neighbors.startswith("[") and raw_neighbors.endswith("]"):
            values = [int(item.strip()) for item in raw_neighbors[1:-1].split(",") if item.strip()]
        else:
            values = [int(raw_neighbors)]
    elif isinstance(raw_neighbors, (list, tuple, ListConfig)):
        values = [int(value) for value in raw_neighbors]
    else:
        values = [int(raw_neighbors)]

    if not values:
        raise ValueError("task.num_neighbors must contain at least one value")
    if len(values) >= num_layers:
        return values[:num_layers]
    return values + [values[-1]] * (num_layers - len(values))


def _aux_iteration_keys(mapping: dict | None) -> tuple[str, ...]:
    """AUX_INFO_KEYS plus any r3_* keys actually present (R3 transition
    stats, plan §18). Backward compatible: historical models emit none of
    the extra keys, so their behavior is unchanged."""
    if not isinstance(mapping, dict):
        return AUX_INFO_KEYS
    extra = sorted(
        k for k in mapping
        if isinstance(k, str)
        and (k.startswith("r3_") or k.startswith("scope_") or k.startswith("osra_")
             or k.startswith("oft_"))
    )
    return AUX_INFO_KEYS + tuple(extra)


def update_aux_info_stats(
    sums: dict[str, float],
    counts: dict[str, float],
    aux_info: dict | None,
    weight: float = 1.0,
) -> None:
    if not isinstance(aux_info, dict):
        return
    for key in _aux_iteration_keys(aux_info):
        if key not in aux_info:
            continue
        value = aux_info[key]
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            scalar = float(value.detach().cpu().item())
        else:
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
        sums[key] = sums.get(key, 0.0) + scalar * weight
        counts[key] = counts.get(key, 0.0) + weight


def summarize_aux_info_stats(sums: dict[str, float], counts: dict[str, float]) -> dict[str, float]:
    return {
        key: sums[key] / max(counts.get(key, 0.0), 1e-12)
        for key in _aux_iteration_keys(sums)
        if key in sums
    }


def format_aux_info_stats(stats: dict[str, float]) -> str:
    return " | ".join(f"{key} {stats[key]:.4f}" for key in _aux_iteration_keys(stats) if key in stats)
