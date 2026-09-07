"""R0 Exact Counterfactual Evaluator + first/second-order estimators
(FIRM-MAG plan §5-§10, §21).

For a frozen SimpleMAGProbe checkpoint it samples directed relations
j -> i whose receiver i is a VALIDATION node (never test labels), then for
each sampled relation builds the four local interventions
    z_null = z_full[i] - delta_T - delta_V
    z_text = z_null + delta_T ; z_visual = z_null + delta_V
    z_joint = z_null + delta_T + delta_V          (= z_full[i])
and evaluates them through the FROZEN classifier only (no graph re-run, no
re-training). It also computes per-edge semantic similarities (P4) and the
logit-space first/second-order functional estimators (P5).

All four numeric invariants from plan §21 are asserted before writing output:
    Check 2: z_joint == z_full[i]                 (max dev < 1e-5)
    Check 3: L_joint == L(original, node i)       (max dev < 1e-5)
    Check 4: S_second == -a_T^T C a_V             (max dev < 1e-6)
(Check 1, the additive decomposition, is enforced by unit tests
tests/test_simple_mag_probe.py and by construction: forward and
forward_with_deltas share the same _decompose code path.)

Output: per-relation CSV with all fields listed in plan §10.

Usage:
    python scripts/run_r0_counterfactual.py \
        --dataset Grocery --train-seed 42 --analysis-seed 12345 \
        --ckpt experiments/r0/checkpoints/Grocery_seed42_probe.pt \
        --out  experiments/r0/results/Grocery/seed_42/r0_relation_diagnostics.csv \
        --device cuda:0 --target-relations 4000 --max-per-receiver 3
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra import compose, initialize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="Grocery")
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--analysis-seed", type=int, default=12345)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target-relations", type=int, default=4000)
    p.add_argument("--max-per-receiver", type=int, default=3)
    return p.parse_args()


def resolve_cfg(dataset: str, seed: int):
    # hydra config_path is relative to this file (scripts/ -> ../configs).
    with initialize(config_path="../configs", version_base=None):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}",
                "task=nc",
                "model=simple_mag_probe",
                f"seed={int(seed)}",
            ],
        )


@torch.no_grad()
def load_setup(args):
    """cfg + data + frozen model + frozen classifier (best-val checkpoint)."""
    from src.data import load_mag_data
    from src.models.factory import build_model

    cfg = resolve_cfg(args.dataset, args.train_seed)
    data = load_mag_data(cfg, "nc", int(args.train_seed))
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    assert ckpt["task"] == "nc", f"unexpected ckpt task {ckpt['task']!r}"
    assert int(ckpt["seed"]) == args.train_seed, "ckpt seed != --train-seed"

    device = torch.device(args.device)
    model = build_model(cfg, ckpt["data_info"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    w = ckpt["head_state"]["weight"]
    head = torch.nn.Linear(int(w.shape[1]), int(w.shape[0])).to(device)
    head.load_state_dict(ckpt["head_state"])
    head.eval()

    # R0 discipline: only validation receivers are analyzed (plan §5.1).
    assert data.val_idx is not None
    return cfg, data, model, head, device


# ------------------------------------------------------------------------- #
# sampling (plan §5.2): stratified over val receiver degree quantile x class #
# ------------------------------------------------------------------------- #

def _degree_bins(deg: torch.Tensor) -> torch.Tensor:
    qs = torch.quantile(deg.to(torch.float32), torch.tensor([0.25, 0.5, 0.75]))
    bin_idx = torch.zeros_like(deg, dtype=torch.long)
    for q in qs:
        bin_idx = bin_idx + (deg > q).to(torch.long)
    return bin_idx


def sample_relations(
    edge_index: torch.Tensor,
    val_idx: torch.Tensor,
    labels: torch.Tensor,
    analysis_seed: int,
    target_relations: int,
    max_per_receiver: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return (receiver_ids[S], edge_ids[S], info) with one row per sampled
    directed relation; per receiver <= max_per_receiver incoming edges;
    deterministic under (analysis_seed, dataset graph, val split)."""
    num_nodes = int(edge_index.max().item()) + 1
    tgt = edge_index[1]
    order = torch.argsort(tgt, stable=True)
    counts = torch.bincount(tgt, minlength=num_nodes)
    starts = torch.cumsum(counts, dim=0) - counts

    # eligible receivers: val nodes with at least one incoming edge
    deg = counts[val_idx]
    eligible = (deg > 0).nonzero(as_tuple=True)[0]
    receivers = val_idx[eligible]
    r_deg = counts[receivers]
    bins = _degree_bins(r_deg.to(torch.float32))
    r_labels = labels[receivers]

    cells: dict[tuple[int, int], list[tuple[int, int]]] = {}  # (bin,class) -> [(node, edge_id)]
    for pos, node in enumerate(receivers.tolist()):
        lo = int(starts[node])
        hi = lo + int(counts[node])
        edge_ids = order[lo:hi].tolist()
        rng = random.Random(f"r0:{analysis_seed}:node:{node}")
        rng.shuffle(edge_ids)
        for edge_id in edge_ids[:max_per_receiver]:
            key = (int(bins[pos]), int(r_labels[pos]))
            cells.setdefault(key, []).append((node, edge_id))

    total_pool = sum(len(v) for v in cells.values())
    sampled: list[tuple[int, int]] = []
    cell_plan: dict = {}
    for key, pool in sorted(cells.items()):
        cell_size = max(1, round(target_relations * len(pool) / max(total_pool, 1)))
        cell_size = min(cell_size, len(pool))
        rng = random.Random(f"r0:{analysis_seed}:cell:{key[0]}:{key[1]}")
        chosen = rng.sample(pool, cell_size)
        sampled.extend(chosen)
        cell_plan[key] = {"pool": len(pool), "chosen": cell_size}

    rng = random.Random(f"r0:{analysis_seed}:shuffle")
    rng.shuffle(sampled)
    receivers_s = torch.tensor([n for n, _ in sampled], dtype=torch.long)
    edge_ids_s = torch.tensor([e for _, e in sampled], dtype=torch.long)
    info = {
        "num_eligible_receivers": int(receivers.numel()),
        "num_val_nodes": int(val_idx.numel()),
        "num_cells_occupied": len(cells),
        "num_cells_chosen": len([k for k, v in cell_plan.items() if v["chosen"] > 0]),
        "cell_plan": cell_plan,
        "total_pool": total_pool,
    }
    return receivers_s, edge_ids_s, info


# ------------------------------------------------------------------------- #
# main                                                                      #
# ------------------------------------------------------------------------- #

@torch.no_grad()
def run(args) -> None:
    cfg, data, model, head, device = load_setup(args)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    labels = data.y.to(device)
    num_nodes = int(x.size(0))
    num_classes = int(data.num_classes)
    logger = print

    logger(f"[data] {data.name} | nodes={num_nodes} | edges={edge_index.size(1)} | classes={num_classes}")
    logger(f"[data] val={data.val_idx.numel()} | test={data.test_idx.numel()} (NOT accessed in analysis)")

    diag = model.forward_with_deltas(x, edge_index)
    z_full, h_self = diag["z_full"], diag["h_self"]
    h_t, h_v = diag["h_text"], diag["h_visual"]
    dt, dv = diag["edge_delta_text"], diag["edge_delta_visual"]
    alpha = diag["edge_norm"]
    logger(f"[fwd] h/z: {tuple(z_full.shape)} | per-edge deltas: {tuple(dt.shape)}")

    # ---- sampling (plan §5.2) ----
    receivers_s, edge_ids_s, sinfo = sample_relations(
        edge_index.cpu(), data.val_idx, labels.cpu(),
        args.analysis_seed, args.target_relations, args.max_per_receiver,
    )
    receivers_s = receivers_s.to(device)
    edge_ids_s = edge_ids_s.to(device)
    s = receivers_s.numel()
    logger(
        f"[sample] relations={s} | eligible receivers={sinfo['num_eligible_receivers']} "
        f"of val {sinfo['num_val_nodes']} | cells chosen={sinfo['num_cells_chosen']}/{sinfo['num_cells_occupied']}"
    )
    if s < 3000:
        logger("[sample] WARNING: fewer than 3000 relations (dataset smaller than target)")

    # ---- four interventions (plan §6): all vectorized over sampled edges ----
    y_i = labels[receivers_s]
    z_i = z_full[receivers_s]                      # [S, H]
    d_t = dt[edge_ids_s]                           # [S, H]
    d_v = dv[edge_ids_s]                           # [S, H]
    edge_ids_cpu = edge_ids_s.cpu()

    # source/target bookkeeping (plan §3.4 convention check)
    src_j = edge_index[0][edge_ids_s]
    tgt_i = edge_index[1][edge_ids_s]
    assert torch.equal(tgt_i, receivers_s), "target of sampled edge != receiver"
    no_self = (src_j != tgt_i).all()
    if not no_self:
        bad = int((src_j == tgt_i).sum().item())
        logger(f"[sample] WARNING: {bad} self-loop relations present in edge_index")
    else:
        logger("[sample] no self-loops in sampled relations")

    z_null = z_i - d_t - d_v
    z_text = z_null + d_t
    z_visual = z_null + d_v
    z_joint = z_null + d_t + d_v

    def _ce(z_row: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(head(z_row), y, reduction="none")

    L_null = _ce(z_null, y_i)
    L_text = _ce(z_text, y_i)
    L_visual = _ce(z_visual, y_i)
    L_joint = _ce(z_joint, y_i)
    L_full = _ce(z_i, y_i)

    # ---- invariants (plan §21 Checks 2 & 3) ----
    dev_z = (z_joint - z_i).abs().max().item()
    dev_L = (L_joint - L_full).abs().max().item()
    logger(f"[check2] max|z_joint - z_full[i]| = {dev_z:.3e} (require <1e-5)")
    logger(f"[check3] max|L_joint - L_full(node i)| = {dev_L:.3e} (require <1e-5)")
    assert dev_z < 1e-5, "Check 2 failed"
    assert dev_L < 1e-5, "Check 3 failed"

    # ---- exact gains / interaction (plan §6.2-6.5) ----
    Q_t = L_null - L_text
    Q_v = L_null - L_visual
    Q_tv = L_null - L_joint
    S_exact = Q_tv - Q_t - Q_v
    S_norm = S_exact / (Q_t.abs() + Q_v.abs() + Q_tv.abs() + 1e-8)
    zeros = torch.zeros_like(Q_t)
    best_exact = torch.stack([zeros, Q_t, Q_v, Q_tv], dim=-1).argmax(dim=-1)

    # ---- semantic proxies (plan §7): cos(h_i, h_j) per modality ----
    sim_t = F.cosine_similarity(h_t[receivers_s], h_t[src_j], dim=-1)
    sim_v = F.cosine_similarity(h_v[receivers_s], h_v[src_j], dim=-1)

    # ---- first/second-order estimators (plan §8): expansion point z_null ----
    # logits/classes computed in float64 so Check 4 holds to <1e-6.
    o_null = head(z_null).double()                 # [S, C]
    p = F.softmax(o_null, dim=-1)                  # [S, C]
    y_onehot = F.one_hot(y_i, num_classes).double()
    g = p - y_onehot
    C_mat = torch.diag_embed(p) - torch.einsum("sc,sd->scd", p, p)  # [S, C, C]
    w_c = head.weight.double()                     # [C, H]
    a_t = torch.einsum("ch,sh->sc", w_c, d_t.double())
    a_v = torch.einsum("ch,sh->sc", w_c, d_v.double())
    a_tv = a_t + a_v

    def _q1(a: torch.Tensor) -> torch.Tensor:
        return -(g * a).sum(dim=-1)

    def _q2(a: torch.Tensor) -> torch.Tensor:
        Ca = torch.einsum("scd,sd->sc", C_mat, a)
        return _q1(a) - 0.5 * (a * Ca).sum(dim=-1)

    Q_t_1, Q_v_1, Q_tv_1 = _q1(a_t), _q1(a_v), _q1(a_tv)
    Q_t_2, Q_v_2, Q_tv_2 = _q2(a_t), _q2(a_v), _q2(a_tv)
    S_second = Q_tv_2 - Q_t_2 - Q_v_2
    S_second_cross = -torch.einsum("sc,scd,sd->s", a_t, C_mat, a_v)

    # ---- invariant (plan §21 Check 4): S_second == -a_T^T C a_V ----
    dev_s2 = (S_second - S_second_cross).abs().max().item()
    logger(f"[check4] max|S_second - (-a_T^T C a_V)| = {dev_s2:.3e} (require <1e-6)")
    assert dev_s2 < 1e-6, "Check 4 failed"

    best_first = torch.stack([zeros, Q_t_1, Q_v_1, Q_tv_1], dim=-1).argmax(dim=-1)
    best_second = torch.stack([zeros, Q_t_2, Q_v_2, Q_tv_2], dim=-1).argmax(dim=-1)

    # ---- extra per-relation metadata (plan §10) ----
    logits_full = head(z_i)
    full_pred = logits_full.argmax(dim=-1)
    receiver_correct = (full_pred == y_i)
    full_conf = F.softmax(logits_full, dim=-1).max(dim=-1).values
    deg_i = torch.bincount(edge_index[1], minlength=num_nodes)[receivers_s]
    deg_j = torch.bincount(edge_index[1], minlength=num_nodes)[src_j]
    deg = torch.bincount(edge_index[1], minlength=num_nodes)
    # degree bin of the receiver over ALL val receivers (quartiles)
    val_degs = deg[data.val_idx]
    val_qs = torch.quantile(
        val_degs.to(torch.float32),
        torch.tensor([0.25, 0.5, 0.75], device=val_degs.device),
    )
    def _bin_of(d: torch.Tensor) -> torch.Tensor:
        b = torch.zeros_like(d, dtype=torch.long)
        for q in val_qs:
            b = b + (d > q).to(torch.long)
        return b
    deg_bin_i = _bin_of(deg_i)

    rows = []
    to_cpu = lambda t: t.detach().cpu()
    for k in range(s):
        rows.append({
            "dataset": data.name,
            "train_seed": int(args.train_seed),
            "analysis_seed": int(args.analysis_seed),
            "receiver_i": int(receivers_s[k].item()),
            "neighbor_j": int(src_j[k].item()),
            "edge_id": int(edge_ids_cpu[k].item()),
            "label_i": int(y_i[k].item()),
            "label_j": int(labels[src_j[k]].item()),
            "degree_i": int(deg_i[k].item()),
            "degree_j": int(deg_j[k].item()),
            "edge_norm": float(alpha[edge_ids_cpu[k]].item()),
            "receiver_degree_bin": int(deg_bin_i[k].item()),
            "receiver_correct_full": bool(receiver_correct[k].item()),
            "full_confidence": float(full_conf[k].item()),
            "sim_text": float(sim_t[k].item()),
            "sim_visual": float(sim_v[k].item()),
            "L_null": float(L_null[k].item()),
            "L_text": float(L_text[k].item()),
            "L_visual": float(L_visual[k].item()),
            "L_joint": float(L_joint[k].item()),
            "Q_text": float(Q_t[k].item()),
            "Q_visual": float(Q_v[k].item()),
            "Q_joint": float(Q_tv[k].item()),
            "S_exact": float(S_exact[k].item()),
            "S_norm": float(S_norm[k].item()),
            "Q_text_1": float(Q_t_1[k].item()),
            "Q_visual_1": float(Q_v_1[k].item()),
            "Q_joint_1": float(Q_tv_1[k].item()),
            "Q_text_2": float(Q_t_2[k].item()),
            "Q_visual_2": float(Q_v_2[k].item()),
            "Q_joint_2": float(Q_tv_2[k].item()),
            "S_second": float(S_second[k].item()),
            "best_exact": int(best_exact[k].item()),
            "best_first": int(best_first[k].item()),
            "best_second": int(best_second[k].item()),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger(f"[out] wrote {len(rows)} rows -> {args.out}")

    # quick gate preview (formal statistics are analyze_r0's job)
    logger(
        f"[preview] mean(S_exact)={S_exact.mean().item():+.4f} | "
        f"mean(|S_norm|)={S_norm.abs().mean().item():.4f} | "
        f"P(|S_norm|>0.1)={(S_norm.abs() > 0.1).float().mean().item():.3f}"
    )
    dist = best_exact.bincount(minlength=4)
    logger(f"[preview] best_exact NULL/T/V/TV = {dist.tolist()}")


if __name__ == "__main__":
    run(_parse_args())
