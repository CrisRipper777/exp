"""R2-0 architecture-falsification shared layer (plan §9 Prompt 1).

Discipline (plan §1):
    - frozen A0/OFR checkpoints are read-only; nothing here trains.
    - TEST is forbidden: the setup wrapper CUTS data.test_idx immediately
      after load (the only place the word test_idx may appear in this file).
    - fixed probe = R0 ridge_probe reused UNCHANGED (StandardScaler +
      RidgeClassifier(alpha=1.0), fit TRAIN / eval VAL).
    - weighted aggregation is src->dst, chunked over E, never [N,N];
      isolated nodes get finite zero contexts.
    - Splus is strictly topology-only: reads ONLY edge_index / num_nodes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r0_utils import (  # noqa: E402
    FACTOR_NAMES,
    SEEDS,
    extract_forward,
    load_setup as _r0_load_setup,
    ridge_probe,
    write_csv,
)
from src.models.biaxis_p1_components import neighbor_mean, zscore_columns  # noqa: E402

DATASETS = ["Movies", "Toys", "Grocery"]  # R2-0 first round (plan §1.4)
_EPS = 1e-8
_FACTOR_KEYS = {"C": "c", "Pt": "p_t", "Pv": "p_v"}


# ---------------------------------------------------------------------------
# 1. Frozen setup wrapper (plan §9): R0 load_setup + explicit no-test guard
# ---------------------------------------------------------------------------


def guard_no_test(data: object) -> object:
    """Explicit no-Test enforcement: mask test labels, then cut test_idx.

    train_idx / val_idx must be present (R2-0 protocols need them); test_idx
    must have been loaded by the data layer (so the cut is a deliberate
    discipline step, not a silent empty-split artifact). Then:
      1. clone the original test indices;
      2. mask data.y at those positions to -1 (so even a buggy downstream
         read of full-graph labels can never leak test supervision);
      3. set data.test_idx = None so nothing downstream can index Test.
    Train/val labels are untouched.
    """
    assert data.train_idx is not None, "R2-0 requires train_idx"
    assert data.val_idx is not None, "R2-0 requires val_idx"
    assert data.test_idx is not None, (
        "data layer must provide test_idx before the no-test guard cuts it"
    )
    assert data.y is not None, "R2-0 requires labels"
    test_positions = data.test_idx.clone()
    data.y = data.y.clone()
    data.y[test_positions] = -1
    data.test_idx = None
    return data


def load_setup(dataset: str, seed: int, device: torch.device):
    """R0 frozen OFR setup + explicit no-test cut (plan §1.3/§3).

    Checkpoint: outputs/p3/operator/<dataset>/OFR/seed_<seed>/model.pt
    (the R0-audited frozen A0 lifecycle). Reuses perf_r0_utils.load_setup
    unchanged (same hydra compose overrides, same Model class).
    """
    setup = _r0_load_setup(dataset, seed, device)
    guard_no_test(setup.data)
    return setup


# ---------------------------------------------------------------------------
# 2. Factor aliases (plan §9)
# ---------------------------------------------------------------------------


def factor_tensor(fex: dict, name: str) -> torch.Tensor:
    """Frozen factor [N, d_f] by name: C -> c, Pt -> p_t, Pv -> p_v."""
    if name not in _FACTOR_KEYS:
        raise KeyError(name)
    return fex["factors"][_FACTOR_KEYS[name]]


def factor_block(fex: dict) -> torch.Tensor:
    """f_block [N, 3, d_f], factor order [C, Pt, Pv] (model stack order)."""
    return fex["f_block"]


# ---------------------------------------------------------------------------
# 3. weighted_neighbor_mean (plan §9 item 3, used by R2-0B sim/diff channels)
# ---------------------------------------------------------------------------


def weighted_neighbor_mean(
    edge_index: torch.Tensor,
    weights: torch.Tensor,
    features: torch.Tensor,
    num_nodes: int,
    edge_chunk_size: int | None = None,
    eps: float = _EPS,
) -> torch.Tensor:
    """Incoming weighted neighbor mean:

        g_i = sum_{j in N(i)} w_ji * F_j / (sum_{j in N(i)} w_ji + eps)

    - src->dst message direction: ``weights[e]`` / ``features[src[e]]`` are
      aggregated onto ``dst[e]`` (identical direction to the model's
      relation aggregation and to neighbor_mean).
    - edge-chunk safe (peak transient [chunk, d], never [N,N]).
    - isolated / zero-weight-sum nodes -> finite zero (0 / eps).
    - weights [E] broadcast against features [N, d]; arithmetic runs in the
      feature dtype.
    """
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    src, dst = edge_index[0], edge_index[1]
    num_edges = int(edge_index.size(1))
    if weights.size(0) != num_edges:
        raise ValueError(f"weights {tuple(weights.shape)} != E={num_edges}")
    if num_edges == 0:
        return torch.zeros(num_nodes, features.size(-1), dtype=features.dtype, device=features.device)
    dtype = features.dtype
    device = features.device
    w = weights.to(dtype=dtype)

    acc = torch.zeros(num_nodes, features.size(-1), dtype=dtype, device=device)
    wsum = torch.zeros(num_nodes, 1, dtype=dtype, device=device)
    chunk = int(edge_chunk_size) if edge_chunk_size is not None else num_edges
    for start in range(0, num_edges, chunk):
        end = min(start + chunk, num_edges)
        dst_c, src_c = dst[start:end], src[start:end]
        w_c = w[start:end].unsqueeze(-1)
        acc.index_add_(0, dst_c, w_c * features[src_c])
        wsum.index_add_(0, dst_c, w_c.expand(-1, 1))
    return acc / (wsum + eps)


# ---------------------------------------------------------------------------
# 4. Topology-only Splus (plan §9 item 4 / §5.2)
# ---------------------------------------------------------------------------


def raw_splus(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_chunk_size: int | None = None,
    eps: float = _EPS,
) -> torch.Tensor:
    """Raw 8-column structural signature [N, 8] (BEFORE z-score / L2):

        u0 = log(1 + d),  u1 = P u0,  u2 = P u1,  u3 = P u2
        mu_d = P d,       std_d = sqrt(P(d^2) - mu_d^2)
        gap_ji = |u0_j - u0_i| (incoming edge j->i)
        mu_gap = incoming mean(gap),  std_gap = incoming pop-std(gap)

    P = D^{-1}A in message direction (D = in-degree from dst bincount).
    Strictly topology-only: reads ONLY edge_index / num_nodes (no x, no
    h_t/h_v, no C/Pt/Pv, no labels, no logits). Isolated nodes: finite
    all-zero row (every statistic divides by d + eps).
    """
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, 8, dtype=torch.float32, device=edge_index.device)
    src, dst = edge_index[0], edge_index[1]
    device = edge_index.device
    chunk = int(edge_chunk_size) if edge_chunk_size is not None else int(src.size(0))

    deg = torch.bincount(dst, minlength=num_nodes).to(torch.float32)
    u0 = torch.log1p(deg)

    def _p(v: torch.Tensor) -> torch.Tensor:
        # P v = D^{-1} A v, chunked scatter over dst.
        aggr = torch.zeros_like(v)
        for start in range(0, src.size(0), chunk):
            end = min(start + chunk, src.size(0))
            aggr.index_add_(0, dst[start:end], v[src[start:end]])
        return aggr / (deg + eps)

    u1 = _p(u0)
    u2 = _p(u1)
    u3 = _p(u2)
    mu_d = _p(deg)
    mu_d2 = _p(deg * deg)
    std_d = (mu_d2 - mu_d * mu_d).clamp_min(0.0).sqrt()

    gap = (u0[src] - u0[dst]).abs()  # [E]
    gsum = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    gsum2 = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    for start in range(0, src.size(0), chunk):
        end = min(start + chunk, src.size(0))
        g_c = gap[start:end]
        gsum.index_add_(0, dst[start:end], g_c)
        gsum2.index_add_(0, dst[start:end], g_c * g_c)
    mu_gap = gsum / (deg + eps)
    var_gap = (gsum2 / (deg + eps) - mu_gap * mu_gap).clamp_min(0.0)
    std_gap = var_gap.sqrt()

    return torch.stack([u0, u1, u2, u3, mu_d, std_d, mu_gap, std_gap], dim=1)


def compute_splus(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_chunk_size: int | None = None,
    eps: float = _EPS,
) -> torch.Tensor:
    """Splus = whole-graph column z-score then row L2-normalize of raw_splus.

    Strictly topology-only (see raw_splus). Isolated nodes keep finite rows
    (constant z-scored row, L2 unit norm via F.normalize eps guard).
    """
    raw = raw_splus(edge_index, num_nodes, edge_chunk_size=edge_chunk_size, eps=eps)
    z = zscore_columns(raw)
    return F.normalize(z, dim=-1, eps=eps)


# ---------------------------------------------------------------------------
# 5. Context concat helper (plan §9)
# ---------------------------------------------------------------------------


def context_concat(parts: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate feature blocks along the last dim:
    [F^b | N^a] / [F^b | G^a] / [F | G1 | G2 | ...] probes."""
    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# R2-0B: explicit structural-function channels (user Prompt B, §四)
# ---------------------------------------------------------------------------


def structural_edge_weights(
    edge_index: torch.Tensor,
    splus: torch.Tensor,
    eps: float = _EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Structural similarity / contrast edge weights for observed edge j->i:

        c_ji    = cos(Splus_j, Splus_i)
        w_sim   = (1 + c_ji) / 2 + eps
        w_diff  = (1 - c_ji) / 2 + eps

    Returns (w_sim, w_diff) [E] float32. Deterministic, topology-only
    (Splus is topology-only). No threshold / temperature / learned weight.

    Numerical hardening (2026-09-04): the cosine is computed in float64 and
    clamped to [-1, 1] — float32 rounding can push c slightly past 1, which
    makes w_diff negative (~1e-7) and near-zero denominators change sign,
    turning the contrastive mean into a 0/0-type ratio that swings by O(1)
    under 1e-6 input noise (measured 0.28 on Movies). The mathematical range
    of a cosine is [-1, 1]; the clamp restores the intended formula, and
    float64 removes the rounding overshoot. Formula, eps and semantics are
    unchanged.
    """
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    src, dst = edge_index[0], edge_index[1]
    a = splus[src].double()
    b = splus[dst].double()
    c = F.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0).float()
    return (1.0 + c) * 0.5 + eps, (1.0 - c) * 0.5 + eps


def explicit_channels(
    edge_index: torch.Tensor,
    f: torch.Tensor,
    splus: torch.Tensor,
    num_nodes: int,
    edge_chunk_size: int | None = None,
    eps: float = _EPS,
) -> dict[str, torch.Tensor]:
    """Explicit 4-channel structural basis for one factor F [N, d] (user §四):

        G1    = P F            (1-hop ordinary, == neighbor_mean)
        G2    = P G1           (2-hop diffusion; no explicit 2-hop edge list)
        Gsim  = weighted_neighbor_mean(w_sim, F)
        Gdiff = weighted_neighbor_mean(w_diff, F)

    All channels [N, d]; deterministic topology-only weights. Isolated
    nodes keep finite zero rows.
    """
    g1 = neighbor_mean(edge_index, f, num_nodes, edge_chunk_size=edge_chunk_size)
    g2 = neighbor_mean(edge_index, g1, num_nodes, edge_chunk_size=edge_chunk_size)
    w_sim, w_diff = structural_edge_weights(edge_index, splus, eps=eps)
    gsim = weighted_neighbor_mean(
        edge_index, w_sim, f, num_nodes, edge_chunk_size=edge_chunk_size, eps=eps
    )
    gdiff = weighted_neighbor_mean(
        edge_index, w_diff, f, num_nodes, edge_chunk_size=edge_chunk_size, eps=eps
    )
    return {"G1": g1, "G2": g2, "Gsim": gsim, "Gdiff": gdiff}


def assert_feature_dim(tensor: torch.Tensor, n_rows: int, n_cols: int, tag: str) -> None:
    """Strict probe-block shape guard (plan §十五 F): [n_rows, n_cols], finite.

    Every R2-0B/C probe block must pass through this so dimension-matched
    comparisons are enforced in production, not only in unit tests.
    """
    assert tuple(tensor.shape) == (n_rows, n_cols), (
        f"[R2-0] {tag}: expected ({n_rows}, {n_cols}), got {tuple(tensor.shape)}"
    )
    assert bool(torch.isfinite(tensor).all()), f"[R2-0] {tag}: non-finite values"


# ---------------------------------------------------------------------------
# R2-0C: node-local / target-source interaction blocks (user Prompt C)
# ---------------------------------------------------------------------------


def factor_interaction_block(factors: dict[str, torch.Tensor]) -> torch.Tensor:
    """I_factor = [C*Pt | C*Pv | Pt*Pv | |C-Pt| | |C-Pv| | |Pt-Pv|]  [N, 6d].

    factors: {"C", "Pt", "Pv"} each [N, d_f]. Pure elementwise ops (finite
    whenever the factors are finite).
    """
    C, Pt, Pv = factors["C"], factors["Pt"], factors["Pv"]
    return torch.cat(
        [C * Pt, C * Pv, Pt * Pv, (C - Pt).abs(), (C - Pv).abs(), (Pt - Pv).abs()],
        dim=-1,
    )


def modal_interaction_block(h_t: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
    """I_modal = [h_t*h_v | |h_t-h_v|]  [N, 2*hidden]."""
    return torch.cat([h_t * h_v, (h_t - h_v).abs()], dim=-1)


def cond_interaction_block(F: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    """I_ab = [F*N | |F-N|]  [N, 2*d] — target factor x neighbor source context."""
    return torch.cat([F * N, (F - N).abs()], dim=-1)


# ---------------------------------------------------------------------------
# 6. CSV helper: re-exported from perf_r0_utils (write_csv) — see imports.
# 7. no-test guard: guard_no_test above.
# ---------------------------------------------------------------------------
