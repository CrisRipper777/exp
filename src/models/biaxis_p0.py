from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .biaxis_components import ReconstructionHead, SemanticFactorizer
from .common import get_activation, make_norm


class Model(nn.Module):
    """P0-A Bi-Axis semantic factorizer (topology-free by design).

    Splits data.x = [x_t | x_v] inside the model (no runner changes), decouples
    each node's modalities into C / P_t / P_v and fuses them into z_local.

    Interface contract (MAG_baseline):
        forward(x, edge_index) -> (z, None, None, aux_loss, aux_info)
        inference(x, edge_index, device, batch_size) -> z (CPU, chunked)

    P0 discipline: ``edge_index`` is accepted for API compatibility but NEVER
    used — semantic factorization must not be contaminated by topology.
    """

    def __init__(self, cfg, data_info):
        super().__init__()
        text_dim = int(data_info["text_dim"])
        visual_dim = int(data_info["visual_dim"])
        input_dim = int(data_info["input_dim"])
        if text_dim <= 0 or visual_dim <= 0:
            raise ValueError(
                f"biaxis_p0 requires both modalities, got text_dim={text_dim}, visual_dim={visual_dim}"
            )
        if input_dim < text_dim + visual_dim:
            raise ValueError(
                f"input_dim={input_dim} < text_dim+visual_dim={text_dim + visual_dim}; "
                "data.x must be [x_t | x_v] concatenated"
            )

        hidden_dim = int(cfg.model.hidden_dim)
        factor_dim = int(cfg.model.factor_dim)
        dropout = float(cfg.model.dropout)
        activation = str(cfg.model.get("activation", "gelu"))
        norm = str(cfg.model.get("norm", "layernorm"))

        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.factor_dim = factor_dim

        self.factorizer = SemanticFactorizer(
            text_dim, visual_dim, hidden_dim, factor_dim, dropout, activation, norm
        )
        self.recon_text_head = ReconstructionHead(factor_dim, hidden_dim, activation)
        self.recon_visual_head = ReconstructionHead(factor_dim, hidden_dim, activation)
        self.fusion = nn.Sequential(
            nn.Linear(3 * factor_dim, hidden_dim),
            make_norm(norm, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
        )

        # In-model loss weights (plan §4, option A): runner aux_weight stays 1.0.
        self.lambda_common = float(cfg.model.get("lambda_common", 0.1))
        self.lambda_orth = float(cfg.model.get("lambda_orth", 0.01))
        self.lambda_recon = float(cfg.model.get("lambda_recon", 0.1))
        self.orth_fallback_batch = int(cfg.model.get("orth_fallback_batch", 16))

        self.out_dim = hidden_dim

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _split_modalities(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.size(-1) >= self.text_dim + self.visual_dim, (
            f"expected x.size(-1) >= {self.text_dim + self.visual_dim} ([x_t | x_v]), "
            f"got {x.size(-1)}"
        )
        x_t = x[:, : self.text_dim]
        x_v = x[:, self.text_dim : self.text_dim + self.visual_dim]
        return x_t, x_v

    def _encode(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        x_t, x_v = self._split_modalities(x)
        factors = self.factorizer(x_t, x_v)
        z = self.fusion(torch.cat([factors["c"], factors["p_t"], factors["p_v"]], dim=-1))
        return factors, z

    # ------------------------------------------------------------------
    # Aux losses (plan §4)
    # ------------------------------------------------------------------

    def _orth_loss(self, c: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch cross-covariance orthogonality: ||Cov(C,P)||_F^2 / d^2.

        Falls back to squared-cosine overlap for tiny batches.
        Returns (loss_term, overlap_scalar) where overlap = ||Cov||_F / d.
        """
        batch, d = c.size(0), c.size(1)
        if batch < self.orth_fallback_batch:
            c_n = F.normalize(c, dim=-1)
            p_n = F.normalize(p, dim=-1)
            cos2 = (c_n * p_n).sum(dim=-1).square().mean()
            return cos2, cos2.sqrt()
        c_c = c - c.mean(dim=0, keepdim=True)
        p_c = p - p.mean(dim=0, keepdim=True)
        cov = c_c.t() @ p_c / max(batch - 1, 1)
        overlap = cov.norm() / d
        return overlap.square(), overlap

    def _compute_aux(
        self, factors: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h_t, h_v = factors["h_t"], factors["h_v"]
        c_t, c_v = factors["c_t"], factors["c_v"]
        p_t, p_v = factors["p_t"], factors["p_v"]

        # L_common: pull c_t / c_v together (not to exactly 1 — see plan §8.1).
        c_t_n = F.normalize(c_t, dim=-1)
        c_v_n = F.normalize(c_v, dim=-1)
        common_sim = (c_t_n * c_v_n).sum(dim=-1).mean()
        common_loss = 1.0 - common_sim

        # L_orth: separate common from private within each modality.
        orth_t, overlap_t = self._orth_loss(c_t, p_t)
        orth_v, overlap_v = self._orth_loss(c_v, p_v)
        orth_loss = orth_t + orth_v

        # L_rec: keep c+p covering the modality information.
        rec_t = F.mse_loss(self.recon_text_head(c_t, p_t), h_t)
        rec_v = F.mse_loss(self.recon_visual_head(c_v, p_v), h_v)
        rec_loss = rec_t + rec_v

        aux_loss = (
            self.lambda_common * common_loss
            + self.lambda_orth * orth_loss
            + self.lambda_recon * rec_loss
        )

        p_t_n = F.normalize(p_t, dim=-1)
        p_v_n = F.normalize(p_v, dim=-1)
        private_sim = (p_t_n * p_v_n).sum(dim=-1).mean()

        aux_info = {
            "p0_common_loss": common_loss.detach(),
            "p0_orth_loss": orth_loss.detach(),
            "p0_recon_loss": rec_loss.detach(),
            "p0_common_sim": common_sim.detach(),
            "p0_private_sim": private_sim.detach(),
            "p0_c_norm": factors["c"].norm(dim=-1).mean().detach(),
            "p0_pt_norm": p_t.norm(dim=-1).mean().detach(),
            "p0_pv_norm": p_v.norm(dim=-1).mean().detach(),
            "p0_cp_overlap_t": overlap_t.detach(),
            "p0_cp_overlap_v": overlap_v.detach(),
        }
        return aux_loss, aux_info

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None):
        factors, z = self._encode(x)
        if self.training:
            aux_loss, aux_info = self._compute_aux(factors)
        else:
            aux_loss = z.new_tensor(0.0)
            aux_info = {}
        return z, None, None, aux_loss, aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """Chunked per-node inference; exact (topology-free) equivalence with
        a full-graph forward, without holding the whole graph on GPU."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        outputs = torch.empty((x.size(0), self.out_dim), dtype=x.dtype, device="cpu")
        for start in range(0, x.size(0), batch_size):
            end = min(start + batch_size, x.size(0))
            z, _, _, _, _ = self.forward(x[start:end].to(device), None)
            outputs[start:end] = z.detach().cpu()
        return outputs

    @torch.no_grad()
    def encode_factors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Offline factor extraction for P0 diagnostics.

        Returns CPU tensors: {c, c_t, c_v, p_t, p_v, z_local}. Outputs depend
        ONLY on x — ``edge_index`` is ignored by construction (tested).
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        num_nodes = int(x.size(0))
        if batch_size is None or num_nodes <= batch_size:
            factors, z = self._encode(x.to(device))
            return {
                "c": factors["c"].cpu(),
                "c_t": factors["c_t"].cpu(),
                "c_v": factors["c_v"].cpu(),
                "p_t": factors["p_t"].cpu(),
                "p_v": factors["p_v"].cpu(),
                "z_local": z.cpu(),
            }
        keys = ("c", "c_t", "c_v", "p_t", "p_v", "z_local")
        chunks: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)
            factors, z = self._encode(x[start:end].to(device))
            chunks["c"].append(factors["c"].cpu())
            chunks["c_t"].append(factors["c_t"].cpu())
            chunks["c_v"].append(factors["c_v"].cpu())
            chunks["p_t"].append(factors["p_t"].cpu())
            chunks["p_v"].append(factors["p_v"].cpu())
            chunks["z_local"].append(z.cpu())
        return {key: torch.cat(chunks[key], dim=0) for key in keys}
