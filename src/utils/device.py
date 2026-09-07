from __future__ import annotations

import torch


def get_device(device_cfg: str) -> torch.device:
    value = str(device_cfg).lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_cfg}, but CUDA is not available in this environment")
    return torch.device(device_cfg)
