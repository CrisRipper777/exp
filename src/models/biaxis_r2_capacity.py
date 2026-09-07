"""R2-Design-2.5 structured-capacity model
(docs/BiAxis_R2_Design_2_5_Structured_Capacity_Utilization_Audit.md, D2.5-C).

One implementation, seven modes (plan D2.5-C):

    EARLY_MIX     Hmix = H1 + alpha_f (H2 - H1)   (M1 control; B0 path + scalars)
    SEP_SUM       e1_f=T_f1(H1_f), e2_f=T_f2(H2_f), Fout = F + e1 + beta_f*e2
    SEP_CONCAT    Fout_f = F_f + R_f([F_f | e1_f | e2_f])        (main candidate)
    INCEPTION_012 Fout_f = F_f + R_f([F_f | e0_f | e1_f | e2_f]), H0 = F
    CAP_H1_DUP    C2-identical structure on two H1 branches (capacity control)
    WIDE_B0       H1 only; wide source transforms + wide fusion, param-matched
                  to SEP_CONCAT within +/-5% (generic-capacity control)
    DEEP_FUSION   B0 graph path unchanged; fusion -> residual 2-block MLP

Shared with the R2-B0 scaffold (unchanged): P0 factorizer / recon heads /
aux losses (semantic refiner OFF, functional transfer OFF — the fixed 50/50
common consensus), out_dim = hidden_dim contract.

Discipline:
    - EARLY_MIX at alpha=0 reproduces a loaded B0 checkpoint BITWISE
      (tested; alpha is the only admissible missing key).
    - CAP_H1_DUP never computes H2 (neighbor_mean called exactly once;
      tested by counting/poisoned-second-call).
    - SEP_CONCAT / INCEPTION_012 never average hops BEFORE their expert
      transforms (experts consume H1/H2 directly; tested by input
      recording).
    - Per-expert ablation (off_hops) zeroes ONLY the named expert branch;
      EARLY_MIX's {"e2"} forces alpha=0 (= exact B0 path).
    - Deep-supervision aux heads / path dropout exist but default OFF;
      they are D2.5-D training interventions, never inference.
    - No Test: this model never touches labels (the training loop does).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .biaxis_p0 import Model as P0Model
from .biaxis_p1_components import neighbor_mean
from .biaxis_r2_capacity_components import (
    DeepFusion,
    FactorReadout,
    HopExpert,
    HopTokenAttention,
    ResidualFusion,
    WideSourceTransform,
)
from .biaxis_r2_scale_components import FactorHopMixer

FACTOR_NAMES = ("C", "Pt", "Pv")

MODES = (
    "early_mix",
    "sep_sum",
    "sep_concat",
    "inception_012",
    "cap_h1_dup",
    "wide_b0",
    "deep_fusion",
    "hop_attention",
    "h1_attention",
)

# Modes whose graph path exposes a trainable H2 branch / H2-off ablation.
H2_MODES = ("early_mix", "sep_sum", "sep_concat", "inception_012", "hop_attention")

# Expert branch names per mode (hop_experts keys); for the attention modes
# these are the three TOKEN positions (off_hops zeroes the token).
EXPERT_KEYS: dict[str, tuple[str, ...]] = {
    "early_mix": (),
    "sep_sum": ("e1", "e2"),
    "sep_concat": ("e1", "e2"),
    "inception_012": ("e0", "e1", "e2"),
    "cap_h1_dup": ("e1a", "e1b"),
    "wide_b0": (),
    "deep_fusion": (),
    "hop_attention": ("e0", "e1", "e2"),
    "h1_attention": ("e1a", "e1b", "e1c"),
}


class Model(P0Model):
    """Bi-Axis R2-Design-2.5 capacity model (one class, seven modes)."""

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)  # P0 base: factorizer / recon / aux / 1-layer fusion

        self.capacity_mode = str(cfg.model.capacity_mode)
        assert self.capacity_mode in MODES, self.capacity_mode
        mode = self.capacity_mode
        d = self.factor_dim
        h = self.hidden_dim
        dropout = float(cfg.model.dropout)
        activation = str(cfg.model.get("activation", "gelu"))
        norm = str(cfg.model.get("norm", "layernorm"))
        self.edge_chunk_size = cfg.model.get("edge_chunk_size")

        # --- D2.5-D training interventions (default OFF) --------------------
        deep_sup = cfg.model.get("deep_supervision", {})
        self.deep_sup_enabled = bool(deep_sup.get("enabled", False))
        self.deep_sup_lambda = float(deep_sup.get("lambda", 0.1))
        self.path_dropout_p = float(cfg.model.get("path_dropout_p", 0.0))
        self.num_classes = int(data_info["num_classes"])

        self.hop_experts: nn.ModuleDict = nn.ModuleDict()
        self.readouts: nn.ModuleList = nn.ModuleList()
        self.aux_expert_heads: nn.ModuleDict = nn.ModuleDict()

        # --- B0 diagonal-path modules (EARLY_MIX / WIDE_B0 / DEEP_FUSION) ---
        if mode in ("early_mix", "wide_b0", "deep_fusion"):
            if mode == "wide_b0":
                self.wide_width = self._solve_wide_width(cfg, data_info)
                self.source_transforms = nn.ModuleList(
                    [WideSourceTransform(d, self.wide_width, activation, norm) for _ in range(3)]
                )
            else:
                self.source_transforms = nn.ModuleList(
                    [nn.Linear(d, d, bias=False) for _ in range(3)]
                )
            # bias=False: LN(0)=0 so isolated nodes keep F' = F EXACTLY.
            self.msg_norm_base = nn.ModuleList(
                [nn.LayerNorm(d, bias=False) for _ in range(3)]
            )
            self.raw_rho_base = nn.Parameter(torch.zeros(3))  # rho = 0.5 at init

        # --- EARLY_MIX scalars (reuse the M1 mixer, alpha init 0) -----------
        if mode == "early_mix":
            self.mixer = FactorHopMixer("m1")

        # --- SEP_SUM / SEP_CONCAT / INCEPTION_012 / CAP_H1_DUP experts ------
        if mode == "sep_sum":
            for key in ("e1", "e2"):
                self.hop_experts[key] = nn.ModuleList(
                    [HopExpert(d, activation, norm) for _ in range(3)]
                )
            self.beta = nn.Parameter(torch.full((3,), 0.1))  # plan C1
        elif mode in ("sep_concat", "inception_012"):
            keys = ("e0", "e1", "e2") if mode == "inception_012" else ("e1", "e2")
            for key in keys:
                self.hop_experts[key] = nn.ModuleList(
                    [HopExpert(d, activation, norm) for _ in range(3)]
                )
        elif mode == "cap_h1_dup":
            for key in ("e1a", "e1b"):
                self.hop_experts[key] = nn.ModuleList(
                    [HopExpert(d, activation, norm) for _ in range(3)]
                )

        # --- per-factor readouts (SEP_CONCAT / INCEPTION_012 / CAP_H1_DUP) --
        if mode in ("sep_concat", "inception_012", "cap_h1_dup"):
            n_experts = 3 if mode == "inception_012" else 2
            in_dim = (1 + n_experts) * d
            self.readouts = nn.ModuleList(
                [FactorReadout(in_dim, d, dropout, activation, norm) for _ in range(3)]
            )

        # --- D2.5-E hop-token attention (plan D2.5-E) ------------------------
        if mode in ("hop_attention", "h1_attention"):
            # per factor: 3 independent token projections + 2 Pre-LN blocks
            # (embed_dim=d, 4 heads, FFN 4d, dropout 0.1).
            self.token_projs: nn.ModuleDict = nn.ModuleDict()
            self.token_attns = nn.ModuleList(
                [HopTokenAttention(d, heads=4, ff_mult=4, dropout=0.1,
                                   activation=activation, norm=norm)
                 for _ in range(3)]
            )
            for key in EXPERT_KEYS[mode]:
                self.token_projs[key] = nn.ModuleList(
                    [nn.Linear(d, d) for _ in range(3)]
                )
            self.attn_last_weights: list[torch.Tensor] = []

        # --- fusion replacements --------------------------------------------
        if mode in ("sep_concat", "inception_012", "cap_h1_dup",
                    "hop_attention", "h1_attention"):
            self.fusion = DeepFusion(3 * d, h, dropout=dropout, activation=activation, norm=norm)
        elif mode == "wide_b0":
            self.fusion = DeepFusion(
                3 * d, h, mid_dim=self.wide_width,
                dropout=dropout, activation=activation, norm=norm,
            )
        elif mode == "deep_fusion":
            self.fusion = ResidualFusion(3 * d, h, dropout=dropout, activation=activation, norm=norm)

        # --- D2.5-D deep supervision: aux heads on expert outputs -----------
        if self.deep_sup_enabled:
            for key in EXPERT_KEYS[mode]:
                self.aux_expert_heads[key] = nn.ModuleList(
                    [nn.Linear(d, self.num_classes) for _ in range(3)]
                )

        self.requires_full_graph_training = bool(cfg.model.get("full_graph_training", True))
        self.parameter_count = sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Semantic ownership states (fixed 50/50 common; refiner never built)
    # ------------------------------------------------------------------

    def _ownership_states(
        self, factors: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """F^0 = [c, p_t, p_v] with c = (c_t + c_v)/2 (B0 scaffold). The
        semantic refiner is OFF in this model family by construction."""
        f0 = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        return f0, f0, None

    # ------------------------------------------------------------------
    # WIDE_B0 parameter matching (plan D2.5-C C5)
    # ------------------------------------------------------------------

    def _solve_wide_width(self, cfg, data_info) -> int:
        """Solve the width W so WIDE_B0's total parameter count matches the
        SEP_CONCAT reference within +/-5% (per-dataset base modules differ,
        so W is solved per dataset)."""
        d = self.factor_dim
        h = self.hidden_dim
        ref_cfg = copy.deepcopy(cfg)
        ref_cfg.model.capacity_mode = "sep_concat"
        ref = Model(ref_cfg, data_info)
        target = ref.parameter_count
        # P0 base excluding the 1-layer fusion (replaced by DeepFusion).
        base = sum(
            p.numel() for n, p in self.named_parameters() if not n.startswith("fusion.")
        )
        # params(W) = base + 3*(2dW + 3W + d) + (3dW + 3W + Wh + h) + 2h
        denom = 9 * d + 12 + h
        intercept = base + 3 * d + h + 2 * h
        w = int(round((target - intercept) / denom))
        w = max(w, 2 * d)  # sanity floor: never thinner than the default mid
        self._wide_match = {
            "target_sep_concat_params": int(target),
            "solved_width": w,
            "base_params": int(base),
        }
        return w

    # ------------------------------------------------------------------
    # Hop contexts (H0 = F, H1 = P F, H2 = P H1)
    # ------------------------------------------------------------------

    def _hop_contexts(
        self, f_star: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, with_h2: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns (h0, h1, h2|None), each [N, 3, d]. CAP_H1_DUP / WIDE_B0 /
        DEEP_FUSION never compute H2 (with_h2=False)."""
        d = self.factor_dim
        h1 = neighbor_mean(
            edge_index, f_star.reshape(num_nodes, 3 * d), num_nodes,
            edge_chunk_size=self.edge_chunk_size,
        ).reshape(num_nodes, 3, d)
        h2 = None
        if with_h2:
            h2 = neighbor_mean(
                edge_index, h1.reshape(num_nodes, 3 * d), num_nodes,
                edge_chunk_size=self.edge_chunk_size,
            ).reshape(num_nodes, 3, d)
        return f_star, h1, h2

    # ------------------------------------------------------------------
    # Graph update (mode-dispatched)
    # ------------------------------------------------------------------

    def _graph_update(
        self,
        f_star: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        off_hops: set[str] | None = None,
        path_dropout_h1: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        """Returns (f_out [N,3,d], internals) with internals =
        {h0, h1, h2, expert_out, msg_pre_ln, msg_post_ln, readout_input}. """
        mode = self.capacity_mode
        off_hops = set(off_hops or ())
        internals: dict = {}
        d = self.factor_dim

        # Validate the ablation set against the mode's exposed branches.
        if mode == "early_mix":
            valid_off = {"e2"}
        elif mode in ("wide_b0", "deep_fusion"):
            valid_off = set()
        else:
            valid_off = set(EXPERT_KEYS[mode])
        unknown = off_hops - valid_off
        if unknown:
            raise ValueError(f"mode={mode} does not expose off_hops={sorted(unknown)}")

        if mode == "early_mix":
            h0, h1, h2 = self._hop_contexts(f_star, edge_index, num_nodes, with_h2=True)
            hmix = self.mixer(h0, h1, h2)
            if "e2" in off_hops:  # H2-off := alpha forced to 0 (exact B0 path)
                hmix = h1
            v_block = torch.stack([self.source_transforms[a](hmix[:, a]) for a in range(3)], dim=1)
            base_msg = torch.stack([self.msg_norm_base[b](v_block[:, b]) for b in range(3)], dim=1)
            rho = torch.sigmoid(self.raw_rho_base)
            f_out = f_star + rho.view(1, 3, 1) * base_msg
            internals.update(h0=h0, h1=h1, h2=h2, msg_pre_ln=v_block, msg_post_ln=base_msg)
            return f_out, internals

        if mode in ("hop_attention", "h1_attention"):
            with_h2 = mode == "hop_attention"
            h0, h1, h2 = self._hop_contexts(f_star, edge_index, num_nodes, with_h2=with_h2)
            # per-factor tokens: [E0, E1, E2] = [proj(h0), proj(h1), proj(h2)]
            # (h1_attention: three INDEPENDENT H1 projections, plan D2.5-E
            # capacity control — never touches H2).
            token_sources = {"e0": h0, "e1": h1, "e2": h2, "e1a": h1, "e1b": h1, "e1c": h1}
            summaries: list[torch.Tensor] = []
            attn_weights: list[torch.Tensor] = []
            token_blocks: dict[str, torch.Tensor] = {}
            for f in range(3):
                tokens = []
                for key in EXPERT_KEYS[mode]:
                    t = self.token_projs[key][f](token_sources[key][:, f])  # [N, d]
                    if key in off_hops:
                        t = torch.zeros_like(t)
                    tokens.append(t)
                    token_blocks.setdefault(key, []).append(t)
                tokens_stack = torch.stack(tokens, dim=1)  # [N, 3, d]
                summary, attn = self.token_attns[f](tokens_stack)  # [N, d], [2, 3, 3]
                summaries.append(summary)
                attn_weights.append(attn)
            self.attn_last_weights = attn_weights
            corrections = torch.stack(summaries, dim=1)  # [N, 3, d]
            f_out = f_star + corrections
            internals.update(
                h0=h0, h1=h1, h2=h2,
                tokens={k: torch.stack(v, dim=1) for k, v in token_blocks.items()},
                attention=torch.stack(attn_weights, dim=0),  # [3, 2, 3, 3]
                expert_out={k: torch.stack(v, dim=1) for k, v in token_blocks.items()},
            )
            return f_out, internals

        if mode in ("wide_b0", "deep_fusion"):
            h0, h1, _ = self._hop_contexts(f_star, edge_index, num_nodes, with_h2=False)
            v_block = torch.stack([self.source_transforms[a](h1[:, a]) for a in range(3)], dim=1)
            base_msg = torch.stack([self.msg_norm_base[b](v_block[:, b]) for b in range(3)], dim=1)
            rho = torch.sigmoid(self.raw_rho_base)
            f_out = f_star + rho.view(1, 3, 1) * base_msg
            internals.update(h0=h0, h1=h1, h2=None, msg_pre_ln=v_block, msg_post_ln=base_msg)
            return f_out, internals

        # --- expert modes: sep_sum / sep_concat / inception_012 / cap_h1_dup
        keys = EXPERT_KEYS[mode]
        with_h2 = mode in H2_MODES
        h0, h1, h2 = self._hop_contexts(f_star, edge_index, num_nodes, with_h2=with_h2)
        hop_inputs = {"e0": h0, "e1": h1, "e2": h2, "e1a": h1, "e1b": h1}
        expert_out: dict[str, torch.Tensor] = {}
        for key in keys:
            e = torch.stack(
                [self.hop_experts[key][f](hop_inputs[key][:, f]) for f in range(3)], dim=1
            )  # [N, 3, d]
            if key == "e1" and self.training and path_dropout_h1 > 0.0:
                keep = (torch.rand(num_nodes, device=e.device) > path_dropout_h1).to(e.dtype)
                e = e * keep.view(num_nodes, 1, 1)
            if key in off_hops:
                e = torch.zeros_like(e)
            expert_out[key] = e

        if mode == "sep_sum":
            beta = self.beta.view(1, 3, 1)
            f_out = f_star + expert_out["e1"] + beta * expert_out["e2"]
            internals.update(h0=h0, h1=h1, h2=h2, expert_out=expert_out)
            return f_out, internals

        # concat modes: per-factor readout on [F | experts...]
        readout_in: list[torch.Tensor] = []
        corrections: list[torch.Tensor] = []
        for f in range(3):
            q = torch.cat([f_star[:, f]] + [expert_out[k][:, f] for k in keys], dim=-1)
            readout_in.append(q)
            corrections.append(self.readouts[f](q))
        corrections_block = torch.stack(corrections, dim=1)  # [N, 3, d]
        f_out = f_star + corrections_block
        internals.update(
            h0=h0, h1=h1, h2=h2, expert_out=expert_out,
            readout_input=torch.stack(readout_in, dim=1),  # [N, 3, in_d]
        )
        return f_out, internals

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index=None,
        off_hops: set[str] | None = None,
        path_dropout_h1: float = 0.0,
    ):
        factors, _z_local = self._encode(x)
        if self.training:
            aux_loss, aux_info = self._compute_aux(factors)
        else:
            aux_loss = x.new_tensor(0.0)
            aux_info = {}

        f0, f_star, _w = self._ownership_states(factors)
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        num_nodes = int(x.size(0))
        f_out, _internals = self._graph_update(
            f_star, edge_index, num_nodes, off_hops=off_hops,
            path_dropout_h1=path_dropout_h1,
        )
        z = self.fusion(torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1))
        return z, None, None, aux_loss, aux_info

    def forward_with_experts(
        self,
        x: torch.Tensor,
        edge_index=None,
        off_hops: set[str] | None = None,
        path_dropout_h1: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """(z, expert_out) with gradients flowing to the expert outputs —
        the D2.5-D deep-supervision / classifier-sensitivity entry point.
        In eval mode identical to forward() with expert extraction."""
        factors, _z_local = self._encode(x)
        if self.training:
            aux_loss, aux_info = self._compute_aux(factors)
        else:
            aux_loss = x.new_tensor(0.0)
            aux_info = {}
        f0, f_star, _w = self._ownership_states(factors)
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        num_nodes = int(x.size(0))
        f_out, internals = self._graph_update(
            f_star, edge_index, num_nodes, off_hops=off_hops,
            path_dropout_h1=path_dropout_h1,
        )
        z = self.fusion(torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1))
        return z, internals.get("expert_out", {})

    def deep_supervision_loss(self, expert_out: dict[str, torch.Tensor], train_idx, y_train) -> torch.Tensor:
        """Auxiliary task CE on the expert outputs (D2.5-D D2). The aux heads
        are REMOVED at inference (never used in eval forward)."""
        assert self.deep_sup_enabled
        losses = []
        for key, heads in self.aux_expert_heads.items():
            e = expert_out[key]  # [N, 3, d]
            for f in range(3):
                losses.append(
                    torch.nn.functional.cross_entropy(heads[f](e[train_idx, f]), y_train)
                )
        return sum(losses) / max(len(losses), 1)

    @torch.no_grad()
    def inference(self, x, edge_index=None, device=None, batch_size=65536):
        """One exact full-graph forward (graph path needs full neighborhoods)."""
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
    # State extraction (plan Prompt 1: H0/H1/H2, expert outputs, before/
    # after LN, before/after residual, pre/post fusion, per-expert ablation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_capacity_states(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        """Full internal trace in eval mode. Never touches labels; does not
        modify model state. Expects x / edge_index on the model device."""
        self.eval()
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=x.device)
        factors, _z_local = self._encode(x)
        f0, f_star, _w = self._ownership_states(factors)
        num_nodes = int(x.size(0))
        f_out, internals = self._graph_update(f_star, edge_index, num_nodes)
        z = self.fusion(torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1))
        states: dict = {"f_star": f_star, "f_out": f_out, "z": z}
        for key in ("h0", "h1", "h2", "expert_out", "msg_pre_ln", "msg_post_ln",
                    "readout_input", "attention"):
            states[key] = internals.get(key)
        states["pre_residual"] = f_star
        states["post_residual"] = f_out
        states["pre_fusion"] = torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1)
        states["post_fusion"] = z
        return states

    # ------------------------------------------------------------------
    # Mechanism diagnostics (plan D2.5-C causal usage / expert health)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_capacity_diagnostics(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        """JSON-safe aggregates at the CURRENT weights: expert effective
        rank / pairwise cosine / CKA, readout weight norms, branch scales,
        residual ratios. No labels, no state change."""
        self.eval()
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=x.device)
        factors, _z_local = self._encode(x)
        f0, f_star, _w = self._ownership_states(factors)
        num_nodes = int(x.size(0))
        f_out, internals = self._graph_update(f_star, edge_index, num_nodes)
        diag: dict = {
            "mode": self.capacity_mode,
            "residual_ratio": self._residual_ratio_stats(f_out - f_star, f_star),
        }
        if self.capacity_mode == "early_mix":
            diag["alpha"] = [float(v) for v in self.mixer.alpha.detach().cpu().tolist()]
            diag["rho_base"] = [
                float(v) for v in torch.sigmoid(self.raw_rho_base).cpu().tolist()
            ]
        if self.capacity_mode in ("wide_b0", "deep_fusion"):
            diag["rho_base"] = [
                float(v) for v in torch.sigmoid(self.raw_rho_base).cpu().tolist()
            ]
        if self.capacity_mode == "sep_sum":
            diag["beta"] = [float(v) for v in self.beta.detach().cpu().tolist()]

        experts = internals.get("expert_out")
        if experts:
            diag["experts"] = {}
            names = list(experts.keys())
            eff_ranks = {}
            for key in names:
                eff_ranks[key] = [
                    _effective_rank(experts[key][:, f]) for f in range(3)
                ]
            diag["experts"]["effective_rank"] = {
                key: [float(v) for v in vals] for key, vals in eff_ranks.items()
            }
            pair_cos: dict = {}
            for i, ka in enumerate(names):
                for kb in names[i + 1:]:
                    pair_cos[f"{ka}_{kb}"] = [
                        float(_mean_cosine(experts[ka][:, f], experts[kb][:, f])) for f in range(3)
                    ]
            diag["experts"]["pairwise_cosine"] = pair_cos
            cka: dict = {}
            for i, ka in enumerate(names):
                for kb in names[i + 1:]:
                    cka[f"{ka}_{kb}"] = [
                        float(_linear_cka(experts[ka][:, f], experts[kb][:, f])) for f in range(3)
                    ]
            diag["experts"]["cka"] = cka

        if len(self.readouts):
            diag["readout_weight_norms"] = {
                f"factor_{FACTOR_NAMES[f]}": {
                    "first": float(self.readouts[f].net[0].weight.norm().item()),
                    "last": float(self.readouts[f].net[-1].weight.norm().item()),
                }
                for f in range(3)
            }
        attn = internals.get("attention")
        if attn is not None:
            # [F, 2 layers, 3 query, 3 key] mean weights per factor
            diag["attention_weights"] = [
                [[float(v) for v in row] for row in layer.cpu().tolist()]
                for f in range(3) for layer in attn[f]
            ]
        if self.capacity_mode == "wide_b0":
            diag["wide_match"] = dict(self._wide_match)
            diag["wide_width"] = int(self.wide_width)
        diag["parameter_count"] = int(self.parameter_count)
        diag["expert_param_count"] = int(
            sum(p.numel() for p in self.hop_experts.parameters())
        )
        diag["readout_param_count"] = int(sum(p.numel() for p in self.readouts.parameters()))
        diag["fusion_param_count"] = int(sum(p.numel() for p in self.fusion.parameters()))
        diag["graph_param_count"] = int(
            sum(
                p.numel()
                for name, p in self.named_parameters()
                if not name.startswith(("factorizer.", "recon_text_head.", "recon_visual_head.", "fusion."))
            )
        )
        return diag

    def _residual_ratio_stats(self, residual: torch.Tensor, reference: torch.Tensor) -> dict:
        eps = 1e-8
        ratio = residual.norm(dim=-1) / (reference.norm(dim=-1) + eps)  # [N, 3]
        out = {}
        for idx, name in enumerate(FACTOR_NAMES):
            r = ratio[:, idx]
            out[name] = {
                "mean": float(r.mean().item()),
                "std": float(r.std(unbiased=False).item()),
            }
        return out


# ---------------------------------------------------------------------------
# Numeric helpers (module-level, reused by tests)
# ---------------------------------------------------------------------------


def _mean_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_n = torch.nn.functional.normalize(x, dim=-1)
    y_n = torch.nn.functional.normalize(y, dim=-1)
    return (x_n * y_n).sum(dim=-1).mean()


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xx = x.t() @ x
    yy = y.t() @ y
    xy = x.t() @ y
    num = xy.norm().item() ** 2
    den = float(xx.norm().item()) * float(yy.norm().item())
    return torch.tensor(num / den if den > 0 else 0.0)


def _effective_rank(x: torch.Tensor) -> torch.Tensor:
    """Shannon-entropy effective rank of the centered feature matrix."""
    x64 = (x - x.mean(dim=0, keepdim=True)).double()
    s = torch.linalg.svdvals(x64)
    s = s / (s.sum() + 1e-12)
    ent = -(s * torch.log(s + 1e-12)).sum()
    return torch.exp(ent)
