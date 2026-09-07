"""RPTA-MAG final model (rpta_final_nc) migrated from ../RPTA (2026-09-03).

Thin adapter over the frozen RPTA source (RPTA/src/rpta/models/
rpta_mag_final.py, profile RPTA_final = Core-Current-s4): imports the RPTA
package from the path in ``model.source_root``, builds
``RPTAMAGFinal.for_node_classification_final``, and adapts it to this repo's
encoder interface:

    forward(x, edge_index) -> (z, None, None, aux_loss, aux_info)
    z = values["representation"] (hidden_dim); the NC runner adds its own
    linear head (as for every other model here).

Adaptation notes:
    - epoch schedule: RPTA gates ramp over warmup/ramp epochs via
      ``set_epoch``; the full-graph runner performs EXACTLY one training
      forward per epoch, so a training-forward counter drives the schedule.
    - aux losses: frozen objective = decomposition, common_node,
      prototype_balance, private_sharpness, factor_route_task (inner),
      decoupled_route_task (outer) with per-dataset frozen lambdas from
      RPTA/configs/rpta_final_nc_v1.yaml (passed as model overrides by the
      probe driver). labels/train_idx come from the additive data_info keys.
    - this is a performance probe; no framework behavior changes beyond the
      additive data_info keys in nc.py.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, cfg, data_info):
        super().__init__()
        source_root = str(cfg.model.get("source_root", "/hdd1/DataInHere/YHF/RPTA/src"))
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from rpta.models.rpta_mag_final import RPTAMAGFinal  # noqa: PLC0415

        self.hidden_dim = int(cfg.model.hidden_dim)
        self._lambdas = {
            "decomposition": float(cfg.model.lambda_decomposition),
            "common_node": float(cfg.model.lambda_common_node),
            "prototype_balance": float(cfg.model.lambda_prototype_balance),
            "private_sharpness": float(cfg.model.lambda_private_sharpness),
            "factor_route_task": float(cfg.model.lambda_factor_route_task),
            "decoupled_route_task": float(cfg.model.lambda_decoupled_route_task),
        }
        self._y = data_info.get("y")
        self._train_idx = data_info.get("train_idx")
        self._epoch = 0
        self.out_dim = self.hidden_dim

        self.backbone = RPTAMAGFinal.for_node_classification_final(
            text_dim=int(data_info["text_dim"]),
            image_dim=int(data_info["visual_dim"]),
            hidden_dim=self.hidden_dim,
            factor_dim=int(cfg.model.factor_dim),
            num_classes=int(data_info["num_classes"]),
            dropout=float(cfg.model.dropout),
            # gate priors are FIXED by the frozen core profile
            # (common_late_max_gate=0.0, outer_relation_prior=0.50).
        )

    def forward(self, x: torch.Tensor, edge_index=None):
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        if self.training:
            # Exactly one training forward per epoch under full-graph NC.
            self._epoch += 1
            self.backbone.set_epoch(self._epoch)
            labels = self._y.to(x.device) if self._y is not None else None
            idx = self._train_idx.to(x.device) if self._train_idx is not None else None
            _logits, aux_dict, _diag, values = self.backbone.forward_with_aux(
                x, edge_index, labels, idx, return_components=True
            )
            aux_loss = x.new_zeros(())
            for name, weight in self._lambdas.items():
                if name in aux_dict:
                    aux_loss = aux_loss + weight * aux_dict[name]
            z = values["representation"]
            aux_info = {f"rpta_{name}": float(aux_dict[name].item()) for name in aux_dict}
            return z, None, None, aux_loss, aux_info
        z = self.backbone.encode(x, edge_index)
        return z, None, None, x.new_zeros(()), {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """One exact full-graph forward (needs complete neighborhoods)."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        else:
            edge_index = edge_index.to(device)
        z, _, _, _, _ = self.forward(x, edge_index)
        return z.detach().cpu()
