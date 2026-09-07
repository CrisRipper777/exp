from __future__ import annotations

import importlib


def build_model(cfg, data_info):
    module = importlib.import_module(f"src.models.{cfg.model.name}")
    return module.Model(cfg, data_info)
