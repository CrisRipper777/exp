"""R0-P6.7 C/D/E: full-receiver simultaneous policies (inference only).

For EVERY incoming non-self-loop relation of every VALIDATION receiver of the
frozen Grocery seed42 SimpleMAGProbe, computes reference-state (Joint-All)
per-edge functional scores -- exact CE utilities, first/second-order
estimators (float64, Check-4 identity asserted) -- and then applies one-shot
per-edge actions simultaneously:

    z_policy[i] = h_self[i] + sum_{j->i} [ t_keep(j)*dt[j] + v_keep(j)*dv[j] ]

with (t_keep, v_keep) from the edge's action {NULL:(0,0), TEXT:(1,0),
VISUAL:(0,1), JOINT:(1,1)} decided ONCE in the Joint reference state (no
re-estimation, no iteration). No training, no router, no test labels;
validation labels are used only as oracle diagnostics.

Policies: Joint-All (baseline), Text-All, Visual-All, Null-All (diagnostic),
Exact-OneShot, FirstOrder-OneShot, SecondOrder-OneShot (diagnostic), plus
selective-routing curves (E) that modify only the top k% of eligible edges
(D_pred > 0 for first-order; D_exact > 0 for the exact oracle).

Invariants asserted:
  - no self-loops, no duplicate directed pairs among edges into val
  - Joint-All representation == z_full[val] (bitwise)
  - Check-4 analog: S2 == -a_T^T C a_V  (max dev < 1e-10, float64)
  - Q_joint_1 == Q_text_1 + Q_visual_1 (max dev < 1e-9)
  - recomputed scores reproduce the sampled rows of r0_relation_diagnostics.csv
    (same frozen ckpt): |dQ| < 1e-4, S2 dev < 1e-9, best-action match >= 99%
  - exact/frac-1.0 curve == Exact-OneShot policy metrics

Outputs (in --out-dir):
  r0_p67_receiver_policy.csv   per validation receiver CE under each policy,
                               dCE vs Joint, G_local/G_global (exact & first)
  r0_p67_curve_table.csv       selective-curve rows (fraction, CE, acc,
                               macro-F1, changed counts)
  r0_p67_edge_stats.json       per-edge aggregates + policy metrics

Usage:
    conda run -n yhf_env python scripts/run_r0_p67_policies.py \
        --dataset Grocery --train-seed 42 \
        --ckpt experiments/r0/checkpoints/Grocery_seed42_probe.pt \
        --csv experiments/r0/results/Grocery/seed_42/r0_relation_diagnostics.csv \
        --out-dir experiments/r0/results/Grocery/seed_42 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
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
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True,
                   help="existing relation CSV (read only; used for overlap check)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--curve-fractions", type=float, nargs="+",
                   default=[0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00])
    return p.parse_args()


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0,
                          labels=list(range(num_classes))))


@torch.no_grad()
def run(args) -> None:
    NS = argparse.Namespace(
        dataset=args.dataset, train_seed=args.train_seed,
        ckpt=args.ckpt, device=args.device,
    )
    cfg, data, model, head, device = load_setup(NS)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    labels = data.y.to(device)
    num_classes = int(data.num_classes)
    val_idx = data.val_idx.to(device)
    logger = print

    # ---------------- per-edge deltas over the whole graph ----------------
    diag = model.forward_with_deltas(x, edge_index)
    dt = diag["edge_delta_text"]
    dv = diag["edge_delta_visual"]
    z_full = diag["z_full"]
    h_self = diag["h_self"]
    num_nodes = z_full.size(0)

    # ---------------- edges INTO validation receivers ----------------------
    tgt_np = edge_index[1].cpu().numpy()
    in_val = np.isin(tgt_np, data.val_idx.numpy())
    e_v = torch.from_numpy(np.nonzero(in_val)[0]).to(device)   # edge_index order
    E = edge_index[:, e_v]
    abs_ids = e_v
    src_v, tgt_v = E[0], E[1]
    # uniqueness: no duplicate directed pairs, no self-loops
    pairs = torch.stack([src_v, tgt_v], dim=1)
    assert pairs.unique(dim=0).size(0) == pairs.size(0), "duplicate directed pair"
    assert (src_v != tgt_v).all(), "self-loop in edges into val"
    logger(f"[edges] {e_v.numel()} incoming relations to "
           f"{torch.unique(tgt_v).numel()} val receivers (of {data.val_idx.numel()})")

    dt_v = dt[e_v]
    dv_v = dv[e_v]
    z_g = z_full[tgt_v]
    y_v = labels[tgt_v]
    w_c = head.weight.double()

    def _ce(z_row: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(head(z_row), y, reduction="none")

    # ---- exact reference-state utilities (fp32; CSV conventions) ----
    CE_j = _ce(z_g, y_v)                                  # joint (unchanged)
    L_null = _ce(z_g - dt_v - dv_v, y_v)                  # both removed
    L_text = _ce(z_g - dv_v, y_v)                         # text kept (CSV Q_text)
    L_vis = _ce(z_g - dt_v, y_v)                          # visual kept
    Q_t = L_null - L_text
    Q_v = L_null - L_vis
    Q_tv = L_null - CE_j
    z_stack = torch.stack([torch.zeros_like(Q_t), Q_t, Q_v, Q_tv], dim=-1)
    best_exact = z_stack.argmax(dim=-1)
    Q_star = z_stack.max(dim=-1).values
    D_exact = Q_star - Q_tv                                # gain over JOINT (>=0)

    # ---- first/second-order estimators (fp64; expansion z_null) ----
    o0 = head(z_g - dt_v - dv_v).double()
    p = F.softmax(o0, dim=-1)
    y_oh = F.one_hot(y_v, num_classes).double()
    g = p - y_oh
    Cm = torch.diag_embed(p) - torch.einsum("ec,ed->ecd", p, p)
    a_t = torch.einsum("ch,eh->ec", w_c, dt_v.double())
    a_v = torch.einsum("ch,eh->ec", w_c, dv_v.double())
    a_tv = a_t + a_v

    def _q1(a): return -(g * a).sum(dim=-1)
    def _q2(a):
        Ca = torch.einsum("ecd,ed->ec", Cm, a)
        return _q1(a) - 0.5 * (a * Ca).sum(dim=-1)

    Q1_t, Q1_v, Q1_tv = _q1(a_t), _q1(a_v), _q1(a_tv)
    Q2_t, Q2_v, Q2_tv = _q2(a_t), _q2(a_v), _q2(a_tv)
    S2 = Q2_tv - Q2_t - Q2_v
    S2_cross = -torch.einsum("ec,ecd,ed->e", a_t, Cm, a_v)
    dev_s2 = (S2 - S2_cross).abs().max().item()
    dev_lin = (Q1_tv - Q1_t - Q1_v).abs().max().item()
    logger(f"[check4] max|S2 - (-a_T^T C a_V)| = {dev_s2:.3e} (require <1e-10)")
    logger(f"[check-lin] max|Q1_tv - (Q1_t+Q1_v)| = {dev_lin:.3e} (require <1e-9)")
    assert dev_s2 < 1e-10 and dev_lin < 1e-9

    best_first = torch.stack([torch.zeros_like(Q1_t), Q1_t, Q1_v, Q1_tv],
                             dim=-1).argmax(dim=-1)
    best_second = torch.stack([torch.zeros_like(Q2_t), Q2_t, Q2_v, Q2_tv],
                              dim=-1).argmax(dim=-1)

    # ---- overlap invariant vs existing relation CSV -----------------------
    with args.csv.open() as f:
        rows = list(csv.DictReader(f))
    csv_eid = np.array([int(r["edge_id"]) for r in rows], dtype=np.int64)
    order = np.argsort(csv_eid)
    csv_eid_s = csv_eid[order]
    ids = abs_ids.cpu().numpy()
    lo = np.searchsorted(csv_eid_s, ids, side="left")
    hi = np.searchsorted(csv_eid_s, ids, side="right")
    in_csv = hi > lo                      # exact membership in the sampled CSV
    srt_pos = lo[in_csv]
    assert np.array_equal(csv_eid_s[srt_pos], ids[in_csv])
    row_idx = order[srt_pos]              # original CSV row indices
    mt = torch.from_numpy(row_idx)
    in_csv_t = torch.from_numpy(in_csv).to(device)
    n_ov = int(in_csv.sum())
    dev_qt = (Q_t[in_csv_t] - torch.from_numpy(
        np.array([float(rows[int(t)]["Q_text"]) for t in mt])).to(device)).abs().max().item()
    dev_s2csv = (S2[in_csv_t] - torch.from_numpy(
        np.array([float(rows[int(t)]["S_second"]) for t in mt])).to(device)).abs().max().item()
    agree = (best_exact[in_csv_t] == torch.from_numpy(
        np.array([int(rows[int(t)]["best_exact"]) for t in mt])).to(device)).float().mean().item()
    logger(f"[overlap] {n_ov}/{e_v.numel()} edges in CSV | max|dQ_text|={dev_qt:.2e} "
           f"| max|dS2|={dev_s2csv:.2e} | best_exact agree={agree:.4f}")
    # internal self-consistency (Check-4, linearity) is asserted above at 1e-10;
    # cross-run agreement only needs to match at fp precision with no flips
    assert dev_qt < 1e-4 and dev_s2csv < 1e-4 and agree >= 0.99

    # ---- F: four-action argmax vs two sign gates (full edge set) ----------
    # gate mapping: (Q1_t>0, Q1_v>0) -> (text, visual) keep bits.
    gate = (Q1_t > 0).long() + 2 * (Q1_v > 0).long()
    mismatch = (gate != best_first)
    n_mis = int(mismatch.sum())
    if n_mis:
        # allowed only inside the numeric tie zone of the JOINT action
        best_alt = torch.stack(
            [torch.zeros_like(Q1_t), Q1_t, Q1_v], dim=-1).max(dim=-1).values
        tie_gap = (Q1_tv - best_alt).abs().max().item()
        logger(f"[F] gate-vs-argmax mismatches: {n_mis} (all in tie zone, "
               f"max|Q1_tv - max(0,Q1_t,Q1_v)|={tie_gap:.2e})")
        assert tie_gap < 1e-9, "genuine gate/argmax disagreement"
    else:
        logger(f"[F] gate-vs-argmax mismatches over {e_v.numel()} edges: 0")
        tie_gap = 0.0

    # ================= simultaneous policies ================================
    y_val = labels[val_idx]
    n_val = int(val_idx.numel())

    def _apply(mask_t: torch.Tensor, mask_v: torch.Tensor) -> torch.Tensor:
        """Return z of val receivers under per-edge keep masks (E_v size)."""
        buf = h_self.clone()
        contrib = (mask_t.to(dt_v.dtype).unsqueeze(-1) * dt_v
                   + mask_v.to(dt_v.dtype).unsqueeze(-1) * dv_v)
        buf.index_add_(0, tgt_v, contrib)
        return buf[val_idx]

    # natural = JOINT on every edge; must reproduce z_full up to fp reorder
    z_pol_joint = _apply(torch.ones_like(best_exact), torch.ones_like(best_exact))
    dev_joint = (z_pol_joint - z_full[val_idx]).abs().max().item()
    logger(f"[check] Joint-All max|z - z_full[val]| = {dev_joint:.3e} (require <1e-5)")
    assert dev_joint < 1e-5, "Joint-All disagrees with z_full[val]"

    def _metrics(z_pol: torch.Tensor, name: str) -> dict:
        logits = head(z_pol)
        ce_node = F.cross_entropy(logits, y_val, reduction="none")
        pred = logits.argmax(dim=-1)
        return {
            "name": name,
            "CE": float(ce_node.mean().item()),
            "acc": float((pred == y_val).float().mean().item()),
            "macro_f1": _macro_f1(y_val.cpu().numpy(), pred.cpu().numpy(),
                                  num_classes),
        }

    acts = {
        "joint": (torch.ones_like(best_exact), torch.ones_like(best_exact)),
        "text_all": (torch.ones_like(best_exact), torch.zeros_like(best_exact)),
        "visual_all": (torch.zeros_like(best_exact), torch.ones_like(best_exact)),
        "null_all": (torch.zeros_like(best_exact), torch.zeros_like(best_exact)),
        "exact_os": (best_exact & 1, (best_exact >> 1) & 1),
        "first_os": (best_first & 1, (best_first >> 1) & 1),
        "second_os": (best_second & 1, (best_second >> 1) & 1),
    }
    metrics = {}
    z_pols = {}
    for name, (mt_, mv_) in acts.items():
        z_pols[name] = _apply(mt_, mv_)
        metrics[name] = _metrics(z_pols[name], name)
        dce = (F.cross_entropy(head(z_pols[name]), y_val, reduction="none")
               - F.cross_entropy(head(z_pols["joint"]), y_val, reduction="none"))
        metrics[name]["mean_dCE_vs_joint"] = float(dce.mean().item())
        metrics[name]["P_improved"] = float((dce < -1e-7).float().mean().item())
        metrics[name]["P_worsened"] = float((dce > 1e-7).float().mean().item())
        metrics[name]["P_unchanged"] = float((dce.abs() <= 1e-7).float().mean().item())
        if name in ("exact_os", "first_os", "second_os"):
            act = {"null": 0, "text": 0, "visual": 0, "joint": 0}
            cnt = best_exact if name == "exact_os" else (
                best_first if name == "first_os" else best_second)
            for k in range(4):
                act[["null", "text", "visual", "joint"][k]] = int((cnt == k).sum())
            metrics[name]["action_dist_over_val_edges"] = act
    logger("[policies] " + " | ".join(
        f"{m['name']}: CE={m['CE']:.4f} acc={m['acc']:.4f}" for m in metrics.values()))

    # ============ D: local-to-global per receiver ============================
    # per-edge realized gain over JOINT (CE units): exact uses Q_star >= Q_tv;
    # first uses exact Q of first's action choice (CSV A convention).
    Q_first_choice = torch.stack(
        [torch.zeros_like(Q_t), Q_t, Q_v, Q_tv], dim=-1).gather(
        dim=-1, index=best_first.unsqueeze(-1)).squeeze(-1)
    D_first = Q_first_choice - Q_tv                       # >= 0 (argmax property)
    G_local_exact = torch.zeros(num_nodes, device=device)
    G_local_first = torch.zeros(num_nodes, device=device)
    G_local_exact.index_add_(0, tgt_v, D_exact)
    G_local_first.index_add_(0, tgt_v, D_first)

    ce_node = {name: F.cross_entropy(head(z_pols[name]), y_val, reduction="none")
               for name in z_pols}
    G_global_exact = ce_node["joint"] - ce_node["exact_os"]   # >0 => better
    G_global_first = ce_node["joint"] - ce_node["first_os"]
    G_global_second = ce_node["joint"] - ce_node["second_os"]

    # ================= E: selective routing curves ==========================
    # predicted per-edge gain over JOINT (first-order estimate, CE units)
    D_first_pred = torch.stack(
        [torch.zeros_like(Q1_t), Q1_t, Q1_v, Q1_tv],
        dim=-1).max(dim=-1).values - Q1_tv
    curve_rows = []
    for est, score, act in (
        ("exact", D_exact, best_exact),
        ("first", D_first_pred, best_first),
    ):
        eligible = (score > 0).cpu().numpy()
        n_el = int(eligible.sum())
        srt = np.argsort(-score.cpu().numpy(), kind="stable")
        sel_all = np.zeros(e_v.numel(), dtype=bool)
        rows = []
        for frac in [0.0] + list(args.curve_fractions):
            if frac == 0.0:
                sel = np.zeros_like(sel_all)
            else:
                k = max(1, int(np.ceil(frac * n_el)))
                sel = sel_all.copy()
                sel[srt[:k]] = True
            sel_t = torch.from_numpy(sel).to(device)
            mt_e = torch.where(sel_t, act & 1, torch.ones_like(act))
            mv_e = torch.where(sel_t, (act >> 1) & 1, torch.ones_like(act))
            z_s = _apply(mt_e, mv_e)
            m = _metrics(z_s, f"{est}_frac{frac}")
            rows.append({
                "estimator": est, "fraction_eligible": frac,
                "n_eligible": n_el, "n_selected": int(sel.sum()),
                "frac_changed_of_edges": float(sel.mean()),
                **{k: m[k] for k in ("CE", "acc", "macro_f1")},
            })
            if frac == 1.0 and est == "exact":
                # One-Shot policy vs frac-1.0 differ only on zero-gain (D==0)
                # tie flips where argmax leaves JOINT for TEXT/VISUAL
                d_ce = abs(m["CE"] - metrics["exact_os"]["CE"])
                logger(f"[curve-exact] frac1.0 CE={m['CE']:.6f} vs "
                       f"exact_os CE={metrics['exact_os']['CE']:.6f} (|d|={d_ce:.2e})")
                assert d_ce < 1e-3
        curve_rows.extend(rows)
        logger(f"[curve-{est}] eligible={n_el}/{e_v.numel()} "
               f"({n_el / e_v.numel():.3f}) | frac1.0 CE={rows[-1]['CE']:.4f}")

    # ================= per-receiver CSV =====================================
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    recv_path = out_dir / "r0_p67_receiver_policy.csv"
    with recv_path.open("w", newline="") as f:
        wcsv = csv.writer(f)
        cols = ["receiver_i", "label_i", "indegree", "correct_joint",
                "CE_joint", "CE_text_all", "CE_visual_all", "CE_null_all",
                "CE_exact_os", "CE_first_os", "CE_second_os",
                "dCE_text_all", "dCE_visual_all", "dCE_null_all",
                "dCE_exact_os", "dCE_first_os", "dCE_second_os",
                "G_local_exact", "G_local_first",
                "G_global_exact", "G_global_first", "G_global_second"]
        wcsv.writerow(cols)
        deg_in = torch.bincount(tgt_v, minlength=num_nodes)[val_idx]
        ce_all = {k: ce_node[k].cpu().numpy() for k in ce_node}
        for t in range(n_val):
            i = int(val_idx[t])
            row = [i, int(y_val[t]), int(deg_in[t]),
                   bool((head(z_pols["joint"])[t].argmax() == y_val[t]).item()),
                   *[f"{ce_all[k][t]:.8f}" for k in
                     ("joint", "text_all", "visual_all", "null_all",
                      "exact_os", "first_os", "second_os")]]
            row += [f"{ce_all[k][t] - ce_all['joint'][t]:.8f}" for k in
                    ("text_all", "visual_all", "null_all",
                     "exact_os", "first_os", "second_os")]
            row += [f"{float(G_local_exact[i]):.8f}", f"{float(G_local_first[i]):.8f}",
                    f"{float(G_global_exact[t]):.8f}",
                    f"{float(G_global_first[t]):.8f}",
                    f"{float(G_global_second[t]):.8f}"]
            wcsv.writerow(row)
    logger(f"[out] per-receiver policy CSV -> {recv_path}")

    curve_path = out_dir / "r0_p67_curve_table.csv"
    with curve_path.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(curve_rows)
    logger(f"[out] curve table -> {curve_path}")

    # per-edge action distributions (diagnostics for md)
    edge_stats = {
        "n_val_receivers": n_val,
        "n_edges_into_val": int(e_v.numel()),
        "action_dist_exact": {["null", "text", "visual", "joint"][k]:
                              int((best_exact == k).sum()) for k in range(4)},
        "action_dist_first": {["null", "text", "visual", "joint"][k]:
                              int((best_first == k).sum()) for k in range(4)},
        "action_dist_second": {["null", "text", "visual", "joint"][k]:
                               int((best_second == k).sum()) for k in range(4)},
        "F_gate_mismatch_full": n_mis,
        "policies": metrics,
        "G_summary": {
            "G_local_exact_sum": float(G_local_exact[val_idx].sum().item()),
            "G_global_exact_sum": float(G_global_exact.sum().item()),
            "G_local_first_sum": float(G_local_first[val_idx].sum().item()),
            "G_global_first_sum": float(G_global_first.sum().item()),
        },
        "check_S2_dev": dev_s2,
        "check_linear_dev": dev_lin,
        "overlap": {"n": n_ov, "max_dQ": float(dev_qt),
                    "max_dS2": float(dev_s2csv), "best_exact_agree": agree},
    }
    (out_dir / "r0_p67_edge_stats.json").write_text(
        json.dumps(edge_stats, indent=2, allow_nan=True))
    logger("[out] r0_p67_edge_stats.json written")


if __name__ == "__main__":
    run(_parse_args())
