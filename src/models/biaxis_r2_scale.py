"""R2-Design-2.0 model: factor-specific propagation horizon (M0/M1/M2).

Plan: docs/BiAxis_R2_Design_2_0_Factor_Specific_Propagation_Horizon_Plan.md.

Inherits the R2-B0 clean scaffold (P0 factorizer, source transforms, message
norms, rho_base residual, fusion) and changes ONLY the graph context:

    H0^f = F^f,  H1^f = P H0^f,  H2^f = P H1^f   (P = neighbor mean)

    M0: Hmix = H1                        (exact B0 1-hop, §5)
    M1: Hmix = H1 + alpha_f (H2 - H1)    (3 direct scalars, init 0, §6)
    M2: Hmix = sum_k gamma_fk Hk         (softmax(theta), theta=[-4,4,-4], §8)

then M^f = V_f(Hmix^f) -> LN -> rho_base residual -> fusion (B0 unchanged).

Discipline:
    - M1 alphas are DIRECT parameters (no sigmoid/softmax/clamp, plan §6.2);
      |alpha| > 2 is recorded as an instability warning, never auto-clamped.
    - M2 is only trained after M1 Mechanism GO (plan §20); the code exists
      now and its init degenerates to ~H1 (gamma1 ≈ 0.9993, max diff
      reported in tests, plan §36).
    - no high-pass, no K-relation/Gamma/OFR, no node-wise routing (§0).
    - B0 checkpoints load with strict=False; the ONLY admissible missing
      keys are the scale parameters (verified in the trainer).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_p1_components import neighbor_mean
from .biaxis_r2 import Model as B0Model
from .biaxis_r2_scale_components import FactorHopMixer


class Model(B0Model):
    """Bi-Axis R2-Design-2.0: factor-specific propagation horizon on the
    B0 clean scaffold."""

    def __init__(self, cfg, data_info):
        mode = str(cfg.model.get("scale_mode", "m1"))
        assert mode in FactorHopMixer.MODES, mode
        self.scale_mode = mode
        super().__init__(cfg, data_info)
        # M0 instantiates NO scale parameter (strict B0 state_dict load).
        self.mixer = FactorHopMixer(mode)

    # ------------------------------------------------------------------
    # Graph update: replace the 1-hop context with the hop mixture
    # ------------------------------------------------------------------

    def _graph_update(
        self,
        f_star: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        d = self.factor_dim
        h0 = f_star  # [N, 3, d]
        h1 = neighbor_mean(
            edge_index, h0.reshape(num_nodes, 3 * d), num_nodes,
            edge_chunk_size=self.edge_chunk_size,
        ).reshape(num_nodes, 3, d)
        h2 = neighbor_mean(
            edge_index, h1.reshape(num_nodes, 3 * d), num_nodes,
            edge_chunk_size=self.edge_chunk_size,
        ).reshape(num_nodes, 3, d)
        hmix = self.mixer(h0, h1, h2)  # [N, 3, d]

        v_block = torch.stack(
            [self.source_transforms[a](hmix[:, a]) for a in range(3)], dim=1
        )
        base_msg = torch.stack(
            [self.msg_norm_base[b](v_block[:, b]) for b in range(3)], dim=1
        )
        rho_base = torch.sigmoid(self.raw_rho_base)
        f_out = f_star + rho_base.view(1, 3, 1) * base_msg
        # n_block reports the 1-hop context (diagnostic reference); no
        # functional path exists in this model family.
        return f_out, h1, base_msg, None

    # ------------------------------------------------------------------
    # Scale diagnostics (plan §29/§30)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_scale_diagnostics(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        """JSON-safe: learned coefficients + per-factor smoothing stats
        sim(H0,H1) / sim(H0,H2) / ||H2-H1||/||H1|| (plan §30)."""
        self.eval()
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=x.device)
        x_t, x_v = self._split_modalities(x)
        factors = self.factorizer(x_t, x_v)
        num_nodes = int(x.size(0))
        f0 = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        d = self.factor_dim
        h1 = neighbor_mean(
            edge_index, f0.reshape(num_nodes, 3 * d), num_nodes,
            edge_chunk_size=self.edge_chunk_size,
        ).reshape(num_nodes, 3, d)
        h2 = neighbor_mean(
            edge_index, h1.reshape(num_nodes, 3 * d), num_nodes,
            edge_chunk_size=self.edge_chunk_size,
        ).reshape(num_nodes, 3, d)
        names = ("C", "Pt", "Pv")
        smoothing = {}
        for idx, name in enumerate(names):
            def _cos(u, v):
                un, vn = torch.nn.functional.normalize(u, dim=-1), torch.nn.functional.normalize(v, dim=-1)
                return float((un * vn).sum(dim=-1).mean().item())

            smoothing[name] = {
                "sim_h0_h1": _cos(f0[:, idx], h1[:, idx]),
                "sim_h0_h2": _cos(f0[:, idx], h2[:, idx]),
                "rel_h2_h1_gap": float(
                    ((h2[:, idx] - h1[:, idx]).norm(dim=-1) / (h1[:, idx].norm(dim=-1) + 1e-8)).mean().item()
                ),
            }
        return {"scale": self.mixer.scale_diagnostics(), "smoothing": smoothing}


def load_b0_checkpoint_into(model: nn.Module, ckpt_path: str) -> dict:
    """Load a B0 (biaxis_r2) checkpoint into a scale model with strict key
    verification: the ONLY admissible missing keys are the scale params.
    Returns {missing_extra_keys} for logging."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    # For M1/M2 the only new parameters are mixer.alpha / mixer.hop_logits.
    admissible = {"mixer.alpha", "mixer.hop_logits"}
    bad_missing = [k for k in missing if k not in admissible]
    if bad_missing or unexpected:
        raise ValueError(
            f"B0 checkpoint mismatch: bad_missing={bad_missing}, unexpected={unexpected}"
        )
    model.load_state_dict(state, strict=False)
    return {"missing_scale_keys": missing}
