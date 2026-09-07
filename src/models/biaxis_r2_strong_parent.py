"""R2-Design-2.6 strong-parent integration model
(docs/BiAxis_R2_Design_2_6_Strong_Parent_Readout_Integration.md).

    z_base = A0(x, G)                (frozen strong parent, plan §3/§55)
    F^0 = [C, Pt, Pv]                (pre-graph ownership factors)
    H0 = F^0, H1 = P F^0, H2 = P^2 F^0   (side evidence, plan §5)
    e_k^f = E_{f,k}(H_k^f)           (9 independent factor-hop experts, §6)
    z = Readout(z_base, side)        (one of five readouts, §9-§22)

Readouts:
    no_compression_concat      z = [z_base | 9 tokens]                (h + 9d)
    factor_hop_concat          z = [z_base | s_C | s_Pt | s_Pv]       (h + 3d)
    residual_side_fusion       z = z_base + R_side(ResidualFusion([s]))
    base_anchored_hier_attention  z = z_base + W_o(T_final[0] - z_base)
    readout_only_control       z = z_base + M(z_base)  (param-matched)

Token sources: "hop" = H0/H1/H2 (candidate); "h1" = three independent H1
transforms (mandatory architecture-identical control, plan §7).

Discipline:
    - parent always eval; parent_frozen controls no_grad only (plan §25/§34);
      parent dropout NEVER active (eval-mode fine-tuning, documented).
    - side_off reproduces z_base EXACTLY (bitwise, tested).
    - aux expert heads exist only when deep_supervision.enabled; they are
      never used in eval forward (removed at inference, plan §23).
    - causal overrides never modify trained weights (plan §30).
    - No Test: labels never enter the model; the training loop owns them.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .biaxis_p1_components import neighbor_mean
from .biaxis_r2_strong_parent_components import (
    CrossFactorAttention,
    ReadoutOnlyMLP,
    ResidualFusion,
    StrongParentExpert,
)

FACTOR_NAMES = ("C", "Pt", "Pv")

READOUT_TYPES = (
    "no_compression_concat",
    "factor_hop_concat",
    "residual_side_fusion",
    "base_anchored_hier_attention",
    "readout_only_control",
)

TOKEN_SOURCES = ("hop", "h1")

# Token keys per source; the first key is the ego/H0 (or first-H1) slot,
# the last key is the H2 (or third-H1) slot.
TOKEN_KEYS = {"hop": ("e0", "e1", "e2"), "h1": ("e1a", "e1b", "e1c")}

CAUSAL_OVERRIDES = (
    "full", "side_off", "h2_zero", "h2_to_h1", "h2_shuffle",
    "c_h2_off", "pt_h2_off", "pv_h2_off",
    "s_c_off", "s_pt_off", "s_pv_off", "h0_off", "h1_off", "h2_off",
)

MISMATCH_PERM_SEED = 20260904


class Model(nn.Module):
    """Strong-parent integration model wrapping a FROZEN A0 parent."""

    def __init__(self, cfg, data_info, parent: nn.Module):
        super().__init__()
        # Plain attribute (NOT a registered submodule): the parent's params
        # must never appear in self.parameters()/state_dict().
        object.__setattr__(self, "parent", parent)
        self.parent.eval()
        for p in self.parent.parameters():
            p.requires_grad_(False)
        self.parent_frozen = True

        self.factor_dim = int(parent.factor_dim)
        self.hidden_dim = int(parent.hidden_dim)
        self.edge_chunk_size = getattr(parent, "edge_chunk_size", None)

        self.readout_type = str(cfg.model.readout_type)
        assert self.readout_type in READOUT_TYPES, self.readout_type
        self.token_source = str(cfg.model.get("token_source", "hop"))
        assert self.token_source in TOKEN_SOURCES, self.token_source
        self.token_keys = TOKEN_KEYS[self.token_source]

        deep_sup = cfg.model.get("deep_supervision", {})
        self.deep_sup_enabled = bool(deep_sup.get("enabled", True))
        self.deep_sup_lambda = float(deep_sup.get("lambda", 0.1))
        self.num_classes = int(data_info["num_classes"])

        d, h = self.factor_dim, self.hidden_dim
        dropout = float(cfg.model.get("dropout", 0.2))
        activation = str(cfg.model.get("activation", "gelu"))
        norm = str(cfg.model.get("norm", "layernorm"))

        has_tokens = self.readout_type != "readout_only_control"

        # --- 9 independent factor-hop experts (plan §6) ----------------------
        self.hop_experts: nn.ModuleDict = nn.ModuleDict()
        self.aux_heads: nn.ModuleDict = nn.ModuleDict()
        if has_tokens:
            for key in self.token_keys:
                self.hop_experts[key] = nn.ModuleList(
                    [StrongParentExpert(d, dropout=0.1, activation=activation, norm=norm)
                     for _ in range(3)]
                )
            if self.deep_sup_enabled:
                for key in self.token_keys:
                    self.aux_heads[key] = nn.ModuleList(
                        [nn.Linear(d, self.num_classes) for _ in range(3)]
                    )

        # --- factor-local hop attention (FHC / RSF / HIER) -------------------
        self.factor_hop_attns: nn.ModuleList | None = None
        if self.readout_type in ("factor_hop_concat", "residual_side_fusion",
                                 "base_anchored_hier_attention"):
            from .biaxis_r2_capacity_components import HopTokenAttention

            self.factor_hop_attns = nn.ModuleList(
                [HopTokenAttention(d, heads=4, ff_mult=4, dropout=0.1,
                                   activation=activation, norm=norm)
                 for _ in range(3)]
            )

        # --- readout-specific modules -----------------------------------------
        if self.readout_type == "residual_side_fusion":
            self.side_fusion = ResidualFusion(3 * d, h, dropout=dropout,
                                              activation=activation, norm=norm)
            self.r_side = nn.Sequential(
                nn.Linear(h, h), make_norm_layer(norm, h),
                get_activation_fn(activation),
                nn.Linear(h, h),
            )
            nn.init.normal_(self.r_side[-1].weight, std=1e-3)
            nn.init.zeros_(self.r_side[-1].bias)
        elif self.readout_type == "base_anchored_hier_attention":
            self.factor_projs = nn.ModuleList([nn.Linear(d, h) for _ in range(3)])
            self.cross_attn = CrossFactorAttention(h, heads=4, ff_mult=4,
                                                   dropout=0.1,
                                                   activation=activation, norm=norm)
            self.w_out = nn.Linear(h, h)
            nn.init.normal_(self.w_out.weight, std=1e-3)
            nn.init.zeros_(self.w_out.bias)
        elif self.readout_type == "readout_only_control":
            self.readout_mlp = ReadoutOnlyMLP(
                h, width=self._solve_readout_only_width(h),
                dropout=0.1, activation=activation, norm=norm,
            )

        # --- output dim contract ----------------------------------------------
        if self.readout_type == "no_compression_concat":
            self.out_dim = h + 9 * d
        elif self.readout_type == "factor_hop_concat":
            self.out_dim = h + 3 * d
        else:
            self.out_dim = h

        self.side_parameter_count = int(
            sum(p.numel() for p in self.parameters()))
        self.parameter_count = self.side_parameter_count

    def _solve_readout_only_width(self, h: int) -> int:
        """Solve the READOUT_ONLY width so its parameter count matches the
        HIER side branch within +/-5% (dataset-independent)."""
        d = self.factor_dim
        # HIER side params: 3 factor projs (d*h+h) + 2 Pre-LN blocks (MHA
        # h,4 heads + FFN 4h) + w_out (h*h+h). Compute numerically.
        projs = 3 * (d * h + h)
        mha = h * 3 * h + 3 * h + h * h + h  # in_proj + out_proj
        ffn = h * 4 * h + 4 * h + 4 * h * h + h
        ln = 2 * h
        per_block = mha + ffn + 2 * ln
        target = projs + 2 * per_block + (h * h + h)
        # ReadoutOnlyMLP params = Linear(h,w)+LN(w) + 2*(w*w+w+2w) + Linear(w,h)
        # = (h*w + w) + 2w + 2*(w*w + 3w) + (w*h + h)
        # = 2w^2 + w*(2h + 8) + h
        import math

        a, b, c = 2.0, 2.0 * h + 8.0, float(h) - float(target)
        w = int(round((-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)))
        self._readout_match = {"target_hier_side_params": int(target),
                               "solved_width": w}
        return max(w, h)

    # ------------------------------------------------------------------
    # Parent pieces (frozen by default; D2.6-D toggles parent_frozen)
    # ------------------------------------------------------------------

    def _parent_pieces(self, x, edge_index, num_nodes):
        factors, _z_local = self.parent._encode(x)
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        graph_out = self.parent._graph_update(f_block, edge_index, num_nodes)
        f_tilde = graph_out["f_tilde"]
        z_base = self.parent.fusion(
            torch.cat([f_tilde[:, 0], f_tilde[:, 1], f_tilde[:, 2]], dim=-1))
        return f_block, z_base

    def _parent_forward(self, x, edge_index, num_nodes):
        if self.parent_frozen:
            with torch.no_grad():
                return self._parent_pieces(x, edge_index, num_nodes)
        return self._parent_pieces(x, edge_index, num_nodes)

    # ------------------------------------------------------------------
    # Side evidence (plan §5/§6)
    # ------------------------------------------------------------------

    def _hop_contexts(self, f_block, edge_index, num_nodes):
        """H0 = F, H1 = P F, H2 = P^2 F. H1-only controls never compute H2
        (neighbor_mean called exactly once — tested)."""
        d = self.factor_dim
        h1 = neighbor_mean(
            edge_index, f_block.reshape(num_nodes, 3 * d), num_nodes,
            edge_chunk_size=self.edge_chunk_size,
        ).reshape(num_nodes, 3, d)
        if self.token_source == "h1":
            return f_block, h1, None
        h2 = neighbor_mean(
            edge_index, h1.reshape(num_nodes, 3 * d), num_nodes,
            edge_chunk_size=self.edge_chunk_size,
        ).reshape(num_nodes, 3, d)
        return f_block, h1, h2

    def _token_source_tensor(self, key, h0, h1, h2_eff, causal):
        """The [N, 3, d] input the expert of slot ``key`` consumes."""
        if self.token_source == "h1":
            return h1
        if key == "e0":
            return h0
        if key == "e1":
            return h1
        # e2 (H2 slot)
        if causal == "h2_to_h1":
            return h1
        for f, name in enumerate(("c", "pt", "pv")):
            if causal == f"{name}_h2_off":
                out = h2_eff.clone()
                out[:, f] = h1[:, f]
                return out
        return h2_eff

    def _ckpt(self, module, *args):
        """Activation checkpointing for the side branches (training and any
        grad-enabled pass). Numerically identical (use_reentrant=False —
        the house-tested pattern from the OFR memory checkpoint)."""
        if self.training or torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _expert_tokens(self, h0, h1, h2_eff, causal):
        """{key: [N, 3, d]} expert outputs (with gradients in training)."""
        tokens: dict[str, torch.Tensor] = {}
        for key in self.token_keys:
            src = self._token_source_tensor(key, h0, h1, h2_eff, causal)
            e = torch.stack(
                [self._ckpt(self.hop_experts[key][f], src[:, f]) for f in range(3)],
                dim=1)
            if causal == "h2_zero" and key == self.token_keys[-1]:
                e = torch.zeros_like(e)
            if causal == "h2_off" and key == self.token_keys[-1]:
                e = torch.zeros_like(e)
            if causal == "h0_off" and key == self.token_keys[0]:
                e = torch.zeros_like(e)
            if causal == "h1_off" and (self.token_source == "h1" or key == "e1"):
                e = torch.zeros_like(e)
            tokens[key] = e
        return tokens

    # ------------------------------------------------------------------
    # Readouts (plan §9-§22)
    # ------------------------------------------------------------------

    def _factor_summaries(self, tokens):
        """s_f = HopAttn_f(token slots) -> ([N, 3, d], attn [3, 2, 3, 3])."""
        keys = list(self.token_keys)
        summaries = []
        attns = []
        for f in range(3):
            toks = torch.stack([tokens[k][:, f] for k in keys], dim=1)  # [N, 3, d]
            summary, attn = self._ckpt(self.factor_hop_attns[f], toks)
            summaries.append(summary)
            attns.append(attn)
        return torch.stack(summaries, dim=1), torch.stack(attns, dim=0)

    def _readout(self, z_base, tokens, causal, s=None):
        rt = self.readout_type
        if causal == "side_off":
            return z_base
        if rt == "readout_only_control":
            return z_base + self.readout_mlp(z_base)
        if rt == "no_compression_concat":
            blocks = [z_base]
            for f in range(3):
                for key in self.token_keys:
                    blocks.append(tokens[key][:, f])
            return torch.cat(blocks, dim=-1)
        if s is None:
            s, _attn = self._factor_summaries(tokens)
        s_c, s_pt, s_pv = s[:, 0], s[:, 1], s[:, 2]
        if causal == "s_c_off":
            s_c = torch.zeros_like(s_c)
        if causal == "s_pt_off":
            s_pt = torch.zeros_like(s_pt)
        if causal == "s_pv_off":
            s_pv = torch.zeros_like(s_pv)
        if rt == "factor_hop_concat":
            return torch.cat([z_base, s_c, s_pt, s_pv], dim=-1)
        if rt == "residual_side_fusion":
            u = self.side_fusion(torch.cat([s_c, s_pt, s_pv], dim=-1))
            return z_base + self.r_side(u)
        if rt == "base_anchored_hier_attention":
            # the (possibly ablated) factor summaries, not the original s
            s_block = torch.stack([s_c, s_pt, s_pv], dim=1)  # [N, 3, d]
            q = torch.stack([self.factor_projs[f](s_block[:, f]) for f in range(3)], dim=1)
            tokens4 = torch.stack([z_base, q[:, 0], q[:, 1], q[:, 2]], dim=1)  # [N,4,h]
            final, _attn = self._ckpt(self.cross_attn, tokens4)
            return z_base + self.w_out(final[:, 0] - z_base)
        # readout_only_control
        return z_base + self.readout_mlp(z_base)

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(self, x, edge_index=None, causal: str = "full"):
        assert causal in CAUSAL_OVERRIDES, causal
        assert self.token_source == "hop" or not causal.startswith(("h2", "c_", "pt_", "pv_")), \
            f"causal={causal} not defined for token_source={self.token_source}"
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        num_nodes = int(x.size(0))
        f_block, z_base = self._parent_forward(x, edge_index, num_nodes)
        if causal == "side_off" or self.readout_type == "readout_only_control":
            if causal == "side_off":
                return z_base, None, None, x.new_tensor(0.0), {}
            return self._readout(z_base, {}, causal), None, None, x.new_tensor(0.0), {}
        h0, h1, h2 = self._hop_contexts(f_block, edge_index, num_nodes)
        if causal == "h2_shuffle":
            from src.analysis.perf_r2d15_utils import fixed_node_permutation

            perm = fixed_node_permutation(num_nodes, MISMATCH_PERM_SEED)
            h2 = h2[perm]
        tokens = self._expert_tokens(h0, h1, h2, causal)
        z = self._readout(z_base, tokens, causal)
        return z, None, None, x.new_tensor(0.0), {}

    def forward_with_experts(self, x, edge_index=None, causal: str = "full"):
        """(z, tokens, factor_summaries, attention) with gradients — the
        deep-supervision / sensitivity entry point."""
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        num_nodes = int(x.size(0))
        f_block, z_base = self._parent_forward(x, edge_index, num_nodes)
        if causal == "side_off" or self.readout_type == "readout_only_control":
            z = self._readout(z_base, {}, causal)
            return z, {}, None, None
        h0, h1, h2 = self._hop_contexts(f_block, edge_index, num_nodes)
        if causal == "h2_shuffle":
            from src.analysis.perf_r2d15_utils import fixed_node_permutation

            perm = fixed_node_permutation(num_nodes, MISMATCH_PERM_SEED)
            h2 = h2[perm]
        tokens = self._expert_tokens(h0, h1, h2, causal)
        s = attn = None
        if self.factor_hop_attns is not None:
            s, attn = self._factor_summaries(tokens)
        z = self._readout(z_base, tokens, causal, s=s)
        return z, tokens, s, attn

    def deep_supervision_loss(self, tokens, train_idx, y_train):
        """Aux CE on every factor-hop expert output (plan §23); heads are
        removed at inference (never used in eval forward)."""
        assert self.deep_sup_enabled
        losses = []
        for key in self.token_keys:
            for f in range(3):
                losses.append(
                    torch.nn.functional.cross_entropy(
                        self.aux_heads[key][f](tokens[key][train_idx, f]), y_train))
        return sum(losses) / max(len(losses), 1)

    @torch.no_grad()
    def inference(self, x, edge_index=None, device=None, batch_size=65536):
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

    # ------------------------------------------------------------------
    # Diagnostics (plan §32/§33)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def base_preservation(self, x, edge_index) -> dict:
        """z_base vs z(side_off): must be bitwise identical; plus
        z_final vs z_base geometry (CKA / mean cosine / relative L2 /
        side-base norm ratio) at the CURRENT weights."""
        self.eval()
        num_nodes = int(x.size(0))
        _f_block, z_base = self._parent_forward(x, edge_index, num_nodes)
        z_off, _, _, _, _ = self.forward(x, edge_index, causal="side_off")
        z_full, _, _, _, _ = self.forward(x, edge_index, causal="full")
        from src.analysis.perf_r2d15_utils import linear_cka, mean_cosine, mean_relative_l2

        # concat readouts keep z_base as the leading h columns; the
        # geometry comparison is against those columns.
        z_core = z_full[:, : z_base.size(-1)]
        diff = z_core - z_base
        # GPU index_add atomics give a ~2e-6 noise floor between two parent
        # forwards (R15-0); report both the exact bitwise flag (CPU-true)
        # and the fp-tolerance check the plan accepts.
        return {
            "side_off_bitwise_equal_base": bool(torch.equal(z_off, z_base)),
            "side_off_max_abs_diff": float((z_off - z_base).abs().max().item()),
            "side_off_reproduces_base": bool(
                torch.allclose(z_off, z_base, atol=1e-6, rtol=1e-6)),
            "cka_final_base": float(linear_cka(z_core, z_base)),
            "mean_cosine": mean_cosine(z_core, z_base),
            "relative_l2": mean_relative_l2(z_core, z_base),
            "side_base_norm_ratio": float(
                diff.norm(dim=-1).mean().item()
                / (z_base.norm(dim=-1).mean().item() + 1e-8)),
        }

    def gradient_sensitivity(self, x, edge_index) -> dict:
        """||d||z||^2 / ds_f|| per factor summary (plan §33); deterministic
        eval-mode forward, no state change. Only defined for readouts with
        factor summaries (FHC/RSF/HIER)."""
        self.eval()
        if self.readout_type not in ("factor_hop_concat", "residual_side_fusion",
                                     "base_anchored_hier_attention"):
            return {}
        num_nodes = int(x.size(0))
        f_block, z_base = self._parent_forward(x, edge_index, num_nodes)
        h0, h1, h2 = self._hop_contexts(f_block, edge_index, num_nodes)
        with torch.enable_grad():
            tokens = self._expert_tokens(h0, h1, h2, "full")
            s, _ = self._factor_summaries(tokens)
            z = self._readout(z_base, tokens, "full", s=s)
            # ONE grad call for the whole [N, 3, d] summaries tensor — no
            # retain_graph (keeps the peak small on large graphs).
            g = torch.autograd.grad(z.pow(2).sum(), s)[0]
            out = {}
            for f, name in enumerate(FACTOR_NAMES):
                out[f"s_{name}"] = float(g[:, f].norm().item())
        return out

    @torch.no_grad()
    def compute_diagnostics(self, x, edge_index) -> dict:
        """JSON-safe aggregates at the CURRENT weights: base preservation,
        attention matrices, token/rank cosine stats, side param counts."""
        self.eval()
        diag = {"readout_type": self.readout_type, "token_source": self.token_source,
                "side_parameter_count": self.side_parameter_count}
        diag["base_preservation"] = self.base_preservation(x, edge_index)
        z_full, tokens, s, attn = self.forward_with_experts(x, edge_index)
        diag["final_norm"] = float(z_full.norm(dim=-1).mean().item())
        if tokens:
            diag["token_stats"] = {}
            for key, e in tokens.items():
                diag["token_stats"][key] = {
                    "norm": [float(e[:, f].norm(dim=-1).mean().item()) for f in range(3)],
                    "eff_rank": [
                        float(_eff_rank(e[:, f])) for f in range(3)]}
            keys = list(tokens.keys())
            diag["token_pairwise_cosine"] = {}
            for i, ka in enumerate(keys):
                for kb in keys[i + 1:]:
                    diag["token_pairwise_cosine"][f"{ka}_{kb}"] = [
                        float(_mean_cos(tokens[ka][:, f], tokens[kb][:, f]))
                        for f in range(3)]
        if attn is not None:
            diag["factor_hop_attention"] = [
                [[float(v) for v in row] for row in layer.cpu().tolist()]
                for f in range(3) for layer in attn[f]]
        if self.readout_type == "base_anchored_hier_attention":
            cross = []
            for block in self.cross_attn.blocks:
                cross.append(
                    [[float(v) for v in row] for row in block.mean_attn.cpu().tolist()])
            diag["cross_factor_attention"] = cross
        if self.readout_type == "readout_only_control":
            diag["readout_match"] = dict(self._readout_match)
        return diag


def make_norm_layer(norm: str, dim: int):
    from .common import make_norm

    return make_norm(norm, dim)


def get_activation_fn(activation: str):
    from .common import get_activation

    return get_activation(activation)


def _eff_rank(x: torch.Tensor) -> float:
    x64 = (x.detach().cpu() - x.detach().cpu().mean(dim=0, keepdim=True)).double()
    sv = torch.linalg.svdvals(x64)
    sv = sv / (sv.sum() + 1e-12)
    return float(torch.exp(-(sv * torch.log(sv + 1e-12)).sum()).item())


def _mean_cos(x: torch.Tensor, y: torch.Tensor) -> float:
    x_n = torch.nn.functional.normalize(x, dim=-1)
    y_n = torch.nn.functional.normalize(y, dim=-1)
    return float((x_n * y_n).sum(dim=-1).mean().item())
