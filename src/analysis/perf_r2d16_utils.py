"""R2-Design-1.6 dual-parent shared layer (plan §7/§21/§22/§23).

Parents:
    Parent-P "A0" : biaxis_final-equivalent performance parent. The formal
                    final_nc_benchmark runs saved NO checkpoints, so A0 parent
                    checkpoints come from the R1 same-code-path baseline runs
                    (outputs/perf_r1/baseline/<ds>/A0/seed_<s>/model.pt;
                    structure bitwise == biaxis_final per the R1 audit). They
                    reproduce the formal A0 Val Acc: exact match on Toys and
                    Reddit-S, max |delta| 0.176pp (Grocery s42). DISCLOSED in
                    every report that uses them.
    Parent-C "B0" : clean diagnostic parent (outputs/perf_r2d15/b0_confirm).

Discipline: parents are NEVER retrained here; extraction is bitwise-tested
against each parent's own forward. Val only, no Test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.biaxis_p1_components import neighbor_mean  # noqa: E402

R2D16_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d16"
B0_CONFIRM_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d15" / "b0_confirm"
A0_R1_BASELINE_ROOT = PROJECT_ROOT / "outputs" / "perf_r1" / "baseline"
MISMATCH_PERM_SEED = 20260904
PARENTS = ("A0", "B0")
FACTOR_NAMES = ("C", "Pt", "Pv")
TARGET_DATASETS = ["Movies", "Toys", "Grocery"]
GUARD_DATASETS = ["ele-fashion", "Reddit-S"]
DATASETS = TARGET_DATASETS + GUARD_DATASETS
SEEDS = [42, 43, 44]


@dataclass
class ParentSetup:
    parent: str  # "A0" | "B0"
    dataset: str
    seed: int
    cfg: object
    data: object
    model: object
    head: nn.Module
    device: torch.device


def load_parent_setup(parent: str, dataset: str, seed: int, device: torch.device) -> ParentSetup:
    """Load a parent best checkpoint (model + head + data). NEVER test."""
    assert parent in PARENTS, parent
    if parent == "A0":
        from src.analysis.perf_r1_utils import load_r1_setup

        setup = load_r1_setup(dataset, seed, "A0", device)
        return ParentSetup(parent, dataset, seed, setup.cfg, setup.data,
                           setup.model, setup.head, device)
    from src.analysis.perf_r2d15_utils import load_frozen_r2_checkpoint

    setup = load_frozen_r2_checkpoint(dataset, seed, "B0", device, root=B0_CONFIRM_ROOT)
    return ParentSetup(parent, dataset, seed, setup.cfg, setup.data,
                       setup.model, setup.head, device)


def assert_no_test_access(data: object) -> None:
    assert data.train_idx is not None and data.val_idx is not None


# ---------------------------------------------------------------------------
# Parent state extraction (plan §13/§22) — bitwise == each parent's forward
# ---------------------------------------------------------------------------


def extract_parent_states(
    setup: ParentSetup, x: torch.Tensor, edge_index: torch.Tensor
) -> dict[str, torch.Tensor]:
    """{f_pre [N,3,d], n [N,3,d], f_out [N,3,d], z [N,h], base_update [N,3,d]}.

    f_pre = pre-graph semantic ownership factors [C, Pt, Pv] of the parent's
    OWN trained factorizer (plan §13: never compare coordinates across
    parents). n = simple 1-hop contexts P·F. f_out = parent graph-updated
    factor outputs BEFORE final fusion. base_update = f_out - f_pre (the
    parent factor update, used as the message-novelty reference, plan §30).
    """
    model = setup.model
    if setup.parent == "B0":
        from src.analysis.perf_r2d15_utils import extract_b0_states

        states = extract_b0_states(model, x, edge_index)
        return {
            "f_pre": states["f_pre"],
            "n": states["n"],
            "f_out": states["f_out"],
            "z": states["z"],
            "base_update": states["f_out"] - states["f_pre"],
        }

    # A0: the frozen P3 stack (R1 baseline mode). r1_pipeline re-hosts
    # x/edge_index internally and is the R1-sanctioned extraction.
    from src.analysis.perf_r1_utils import r1_pipeline

    internals = r1_pipeline(setup)
    f_pre = internals["f_block"]  # [N, 3, d] = [C, Pt, Pv]
    f_out = internals["graph_out"]["f_tilde"]
    z = internals["z_final"]
    num_nodes = int(f_pre.size(0))
    d = model.factor_dim
    n_cat = neighbor_mean(
        edge_index, f_pre.reshape(num_nodes, 3 * d), num_nodes,
        edge_chunk_size=model.edge_chunk_size,
    )
    return {
        "f_pre": f_pre,
        "n": n_cat.reshape(num_nodes, 3, d),
        "f_out": f_out,
        "z": z,
        "base_update": f_out - f_pre,
    }


def parent_forward_z(setup: ParentSetup, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """The parent's own eval-mode z (for bitwise-consistency tests)."""
    if setup.parent == "A0":
        return setup.model(x, edge_index)[0]
    return setup.model(x, edge_index)[0]


# ---------------------------------------------------------------------------
# Frozen adapter plumbing (plan §21-§28)
# ---------------------------------------------------------------------------


def adapter_z(setup: ParentSetup, f_out: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """zhat = Fusion_parent([F_parent_out + Delta]) — fusion frozen."""
    fhat = f_out + delta
    model = setup.model
    return model.fusion(torch.cat([fhat[:, 0], fhat[:, 1], fhat[:, 2]], dim=-1))


def refined_graph_forward(
    setup: ParentSetup,
    f_refined: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Feed overridden pre-graph factors through the FROZEN parent graph
    path (plan §37): returns (f_out, z). Used by the semantic residual
    screen (refinement BEFORE parent propagation)."""
    model = setup.model
    if setup.parent == "A0":
        graph_out = model._graph_update(f_refined, edge_index, num_nodes)
        f_out = graph_out["f_tilde"]
    else:
        f_out, _n, _b, _f = model._graph_update(f_refined, edge_index, num_nodes)
    z = model.fusion(torch.cat([f_out[:, 0], f_out[:, 1], f_out[:, 2]], dim=-1))
    return f_out, z


# ---------------------------------------------------------------------------
# Matched initialization (plan §43): exact save/load, no re-instantiation
# ---------------------------------------------------------------------------


def save_state(path: Path, module: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)


def load_state_into(path: Path, module: nn.Module) -> None:
    module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def make_classifier_init(seed: int, out_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    """Deterministic fresh classifier init (same RNG for every variant)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        head = nn.Linear(out_dim, num_classes).to(device)
        head.weight.normal_(0.0, head.weight.std().item(), generator=generator)
    head.bias.data.zero_()
    return head
