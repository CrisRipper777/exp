"""R0-P6.5-B: Real Pair vs Shuffled Pair Control (inference only).

For every sampled relation row (receiver i, neighbor j) in the existing
r0_relation_diagnostics.csv whose receiver ALSO has another sampled incoming
edge (i,k), k != j, constructs the shuffled intervention:

    real:     (delta_T[j->i], delta_V[j->i])
    shuffled: (delta_T[j->i], delta_V[k->i])          k != j

with k chosen among the receiver's other sampled neighbors to best match
message norm / edge norm (smallest |log alpha_j/alpha_k| + |log ||dV_j||/||dV_k|||).

No model training, no modification of the existing CSV or of the probe:
the frozen best-val checkpoint is loaded once and run in eval mode
(forward_with_deltas) to recover per-edge deltas; all real-side values are
copied verbatim from the existing CSV rows.

Per matched row the script emits (one CSV row):

  real:      S_exact, S_norm, S_second, Q_best        (from r0_relation_diagnostics.csv)
  shuffled:  Q_text/Q_visual/Q_joint (exact CE gains),
             S_exact_sh, S_norm_sh, S_second_sh, Q_best_sh

Second-order shuffled values are computed at expansion point
z_null = z_i - delta_T[j] - delta_V[k] (both acted-on messages removed),
matching the P5 protocol; the algebraic identity S2 == -a_T^T C a_V is
asserted on the batch (Check 4 analog, <1e-10 in float64). Only validation
receivers are touched (asserted against the CSV provenance).

Usage:
    conda run -n yhf_env python scripts/run_r0_p65b_shuffle_control.py \
        --dataset Grocery --train-seed 42 --analysis-seed 12345 \
        --ckpt experiments/r0/checkpoints/Grocery_seed42_probe.pt \
        --csv experiments/r0/results/Grocery/seed_42/r0_relation_diagnostics.csv \
        --out experiments/r0/results/Grocery/seed_42/r0_p65b_shuffle_pairs.csv \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_r0_counterfactual import load_setup  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="Grocery")
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--analysis-seed", type=int, default=12345)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True,
                   help="existing r0_relation_diagnostics.csv (read only)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


@torch.no_grad()
def run(args) -> None:
    import numpy as np

    NS = argparse.Namespace(
        dataset=args.dataset, train_seed=args.train_seed,
        ckpt=args.ckpt, device=args.device,
    )
    cfg, data, model, head, device = load_setup(NS)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    labels = data.y.to(device)
    val_set = set(data.val_idx.tolist())
    logger = print

    # ---- recover per-edge deltas (same code path as the original run) ----
    diag = model.forward_with_deltas(x, edge_index)
    dt = diag["edge_delta_text"]          # [E, H]
    dv = diag["edge_delta_visual"]        # [E, H]
    alpha = diag["edge_norm"]             # [E]
    z_full = diag["z_full"]
    w_c = head.weight.double()            # [C, H]

    # ---- read existing CSV rows (receivers only; all val) ----
    with args.csv.open() as f:
        rows = list(csv.DictReader(f))
    recv = np.array([int(r["receiver_i"]) for r in rows])
    nb_j = np.array([int(r["neighbor_j"]) for r in rows])
    eid_j = np.array([int(r["edge_id"]) for r in rows])
    assert set(recv.tolist()) <= val_set, "CSV contains non-validation receivers"
    S_exact = np.array([float(r["S_exact"]) for r in rows])
    S_norm = np.array([float(r["S_norm"]) for r in rows])
    S_second = np.array([float(r["S_second"]) for r in rows])
    Q_t = np.array([float(r["Q_text"]) for r in rows])
    Q_v = np.array([float(r["Q_visual"]) for r in rows])
    Q_j = np.array([float(r["Q_joint"]) for r in rows])
    Q_best_real = np.maximum.reduce([np.zeros(len(rows)), Q_t, Q_v, Q_j])
    logger(f"[rows] {len(rows)} relations from existing CSV (val receivers only)")

    # ---- per-receiver grouping; pair each row with its best-matching k ----
    recv_l = recv.tolist()
    eid_l = eid_j.tolist()
    nb_l = nb_j.tolist()
    nv = dv.norm(dim=-1).cpu().numpy()     # visual message norm per edge
    al = alpha.cpu().numpy()

    groups: dict[int, list[int]] = {}
    for r_i, r in enumerate(recv_l):
        groups.setdefault(r, []).append(r_i)

    partner: list[int] = []                # partner row index per paired row
    for r_i, members in groups.items():
        if len(members) < 2:
            continue
        for a in members:
            best_b, best_s = -1, float("inf")
            for b in members:
                if b == a:
                    continue
                # shuffling replaces visual j->k and the edge (i,k) vs (i,j)
                s = abs(np.log(al[eid_l[b]]) - np.log(al[eid_l[a]])) \
                    + abs(np.log(nv[eid_l[b]]) - np.log(nv[eid_l[a]]))
                if s < best_s:
                    best_b, best_s = b, s
            partner.append((a, best_b))

    paired = [a for a, _ in partner]
    logger(f"[pairs] {len(paired)} rows have a same-receiver shuffled partner "
           f"({len(rows) - len(paired)} rows excluded: receiver sampled once)")

    if not paired:
        raise SystemExit("no receivers with >=2 sampled relations; nothing to do")

    # ---- tensors for all shuffled interventions ----
    A = torch.tensor([a for a, _ in partner], dtype=torch.long)
    B = torch.tensor([b for _, b in partner], dtype=torch.long)
    i_v = torch.from_numpy(recv[A.numpy()]).to(device)
    ej = torch.from_numpy(eid_j[A.numpy()]).to(device)
    ek = torch.from_numpy(eid_j[B.numpy()]).to(device)
    sA, sB = edge_index[0][ej], edge_index[0][ek]
    tA, tB = edge_index[1][ej], edge_index[1][ek]
    assert torch.equal(tA, i_v) and torch.equal(tB, i_v), "sampled edges must point at receiver"
    assert torch.equal(sA, torch.from_numpy(nb_j[A.numpy()]).to(device)), \
        "CSV neighbor_j != edge source"
    assert not torch.equal(sA, sB), "duplicate source for same receiver (multigraph?)"
    logger("[pairs] all sources distinct within a receiver (no duplicate edges)")

    z_i = z_full[i_v]
    dT_j = dt[ej]
    dV_j = dv[ej]
    dV_k = dv[ek]

    # ---- shuffled exact counterfactuals ----
    z_null_sh = z_i - dT_j - dV_k
    z_text_sh = z_i - dV_k                       # null + delta_T[j]
    z_vis_sh = z_i - dT_j                        # null + delta_V[k]
    z_joint_sh = z_null_sh + dT_j + dV_k         # == z_i by construction

    y = labels[i_v]

    def _ce(z_row: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(head(z_row), yy, reduction="none")

    L0_sh = _ce(z_null_sh, y)
    Lt_sh = _ce(z_text_sh, y)
    Lv_sh = _ce(z_vis_sh, y)
    Lj_sh = _ce(z_joint_sh, y)
    L_full_i = _ce(z_i, y)
    # Check 2/3 analogs for the shuffled intervention
    dev_z = (z_joint_sh - z_i).abs().max().item()
    dev_l = (Lj_sh - L_full_i).abs().max().item()
    logger(f"[check-sh] max|z_joint_sh - z_full[i]| = {dev_z:.3e} | "
           f"max|L_joint_sh - L(node i)| = {dev_l:.3e} (require <1e-5)")
    assert dev_z < 1e-5 and dev_l < 1e-5

    Q_t_sh = L0_sh - Lt_sh
    Q_v_sh = L0_sh - Lv_sh
    Q_j_sh = L0_sh - Lj_sh
    S_exact_sh = Q_j_sh - Q_t_sh - Q_v_sh
    S_norm_sh = S_exact_sh / (Q_t_sh.abs() + Q_v_sh.abs() + Q_j_sh.abs() + 1e-8)
    Q_best_sh = torch.stack([torch.zeros_like(Q_t_sh), Q_t_sh, Q_v_sh, Q_j_sh]).max(dim=0).values

    # ---- shuffled second-order at expansion point z_null_sh ----
    o0 = head(z_null_sh).double()
    p = F.softmax(o0, dim=-1)
    y_oh = F.one_hot(y, int(data.num_classes)).double()
    g = p - y_oh
    Cm = torch.diag_embed(p) - torch.einsum("sc,sd->scd", p, p)
    aT = torch.einsum("ch,sh->sc", w_c, dT_j.double())
    aV = torch.einsum("ch,sh->sc", w_c, dV_k.double())
    aTV = aT + aV

    def _q1(a): return -(g * a).sum(dim=-1)
    def _q2(a):
        Ca = torch.einsum("scd,sd->sc", Cm, a)
        return _q1(a) - 0.5 * (a * Ca).sum(dim=-1)

    S2_sh_cross = -torch.einsum("sc,scd,sd->s", aT, Cm, aV)
    S2_sh_diff = _q2(aTV) - _q2(aT) - _q2(aV)
    dev_s2 = (S2_sh_cross - S2_sh_diff).abs().max().item()
    logger(f"[check4-sh] max|S2_diff - S2_cross| = {dev_s2:.3e} (require <1e-10)")
    assert dev_s2 < 1e-10

    # ---- write pair rows ----
    rows_out = []
    src_j = edge_index[0][ej]
    src_k = edge_index[0][ek]
    for t in range(A.numel()):
        a_i, b_i = int(A[t]), int(B[t])
        rows_out.append({
            "dataset": rows[a_i]["dataset"],
            "train_seed": int(args.train_seed),
            "analysis_seed": int(args.analysis_seed),
            "receiver_i": int(i_v[t]),
            "neighbor_j": int(src_j[t]),
            "neighbor_k": int(src_k[t]),
            "edge_id_j": int(ej[t]),
            "edge_id_k": int(ek[t]),
            "alpha_j": float(alpha[ej[t]]),
            "alpha_k": float(alpha[ek[t]]),
            "norm_deltaV_j": float(dV_j[t].norm()),
            "norm_deltaV_k": float(dV_k[t].norm()),
            "label_i": int(y[t]),
            # real side (copied verbatim from the existing CSV)
            "S_exact_real": float(S_exact[a_i]),
            "S_norm_real": float(S_norm[a_i]),
            "S_second_real": float(S_second[a_i]),
            "Q_best_real": float(Q_best_real[a_i]),
            # shuffled side
            "Q_text_sh": float(Q_t_sh[t]),
            "Q_visual_sh": float(Q_v_sh[t]),
            "Q_joint_sh": float(Q_j_sh[t]),
            "S_exact_sh": float(S_exact_sh[t]),
            "S_norm_sh": float(S_norm_sh[t]),
            "S_second_sh": float(S2_sh_cross[t]),
            "Q_best_sh": float(Q_best_sh[t]),
            "receiver_correct_full": rows[a_i]["receiver_correct_full"],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    logger(f"[out] wrote {len(rows_out)} shuffled-pair rows -> {args.out}")

    sa_r = np.abs(S_exact[np.array(paired)])
    sa_s = np.abs(S_exact_sh.cpu().numpy())
    logger(f"[preview] n={len(paired)} | mean|S_exact| real={sa_r.mean():.5f} "
           f"shuffle={sa_s.mean():.5f} | median real={np.median(sa_r):.5f} "
           f"shuffle={np.median(sa_s):.5f}")
    s2_r = np.abs(S_second[np.array(paired)])
    s2_s = np.abs(S2_sh_cross.cpu().numpy())
    logger(f"[preview] mean|S_second| real={s2_r.mean():.5f} "
           f"shuffle={s2_s.mean():.5f} | median real={np.median(s2_r):.5f} "
           f"shuffle={np.median(s2_s):.5f}")
    logger(f"[preview] P(|S_norm|>0.1) real="
           f"{(np.abs(S_norm[np.array(paired)]) > 0.1).mean():.3f} "
           f"shuffle={(S_norm_sh.abs() > 0.1).float().mean().item():.3f}")


if __name__ == "__main__":
    run(_parse_args())
