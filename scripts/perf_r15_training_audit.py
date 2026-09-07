"""R15-1 training / gradient / optimization audit (plan §9-§12).

- training history synthesis from the fresh anchor summaries
  (best/stop epoch, train-val gap, last-20 val slope, plateau/overfit).
- gradient decomposition on Movies/Toys/Grocery at TWO states:
  seed42 initialization + fresh A0 best checkpoint. Per-group (G0-G9)
  norms of g_CE / g_common / g_orth / g_recon / g_aux / g_total, the
  aux/CE ratio, global cos(g_CE, g_aux-*) and per-group update ratios
  after one AdamW step.

Outputs under outputs/perf_r15/audit/. Fact judgments only — no tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize  # noqa: E402

from src.data import load_mag_data  # noqa: E402
from src.models.biaxis_final import Model  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "perf_r15" / "audit"
ANCHOR_ROOT = PROJECT_ROOT / "outputs" / "perf_r15" / "anchor"
WEAK = ["Movies", "Toys", "Grocery"]
ALL_DS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]


def _resolve_cfg(dataset: str) -> object:
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=biaxis_final", "seed=42",
        ])


def _load_anchor(dataset: str, state: str, device: torch.device):
    cfg = _resolve_cfg(dataset)
    data = load_mag_data(cfg, "nc", 42)
    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
    }
    torch.manual_seed(42)
    model = Model(cfg, info).to(device)
    head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    if state == "best":
        ckpt = torch.load(ANCHOR_ROOT / dataset / "A0" / "seed_42" / "model.pt",
                          map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        head.load_state_dict(ckpt["head_state"])
    return data, model, head


def _groups(model, head) -> dict[str, list[torch.Tensor]]:
    def params(*mods):
        out = []
        for m in mods:
            out += list(m.parameters())
        return out

    return {
        "G0_text_projector": params(model.factorizer.text_projector),
        "G1_visual_projector": params(model.factorizer.visual_projector),
        "G2_common_encoder": params(model.factorizer.common_encoder),
        "G3_private_text_encoder": params(model.factorizer.private_text_encoder),
        "G4_private_visual_encoder": params(model.factorizer.private_visual_encoder),
        "G5_local_fusion": params(model.fusion),
        "G6_structural": params(model.struct_signature_mlp, model.edge_token_mlp, model.relation_prototypes),
        "G7_scorer": params(model.transport_scorer) + [model.null_score],
        "G8_operator": params(model.operator),
        "G9_classifier": params(head),
        "G10_recon_heads": params(model.recon_text_head, model.recon_visual_head),
    }


def _aux_components(model, factors) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unweighted P0 components, mirroring biaxis_p0._compute_aux."""
    h_t, h_v = factors["h_t"], factors["h_v"]
    c_t, c_v = factors["c_t"], factors["c_v"]
    p_t, p_v = factors["p_t"], factors["p_v"]
    c_t_n = F.normalize(c_t, dim=-1)
    c_v_n = F.normalize(c_v, dim=-1)
    common = 1.0 - (c_t_n * c_v_n).sum(dim=-1).mean()
    orth_t, _ = model._orth_loss(c_t, p_t)
    orth_v, _ = model._orth_loss(c_v, p_v)
    orth = orth_t + orth_v
    rec_t = F.mse_loss(model.recon_text_head(c_t, p_t), h_t)
    rec_v = F.mse_loss(model.recon_visual_head(c_v, p_v), h_v)
    return common, orth, rec_t + rec_v


def _gradient_audit(dataset: str, state: str, device: torch.device, rows_norm, rows_cos, rows_upd):
    data, model, head = _load_anchor(dataset, state, device)
    model.train()
    head.train()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    tr = data.train_idx.to(device)
    y = data.y.to(device)
    crit = torch.nn.CrossEntropyLoss()
    groups = _groups(model, head)
    group_names = list(groups)
    lam = (model.lambda_common, model.lambda_orth, model.lambda_recon)

    def _norm(gs):
        parts = [p.grad.square().sum() for p in gs if p.grad is not None]
        if not parts:
            return 0.0
        return float(torch.sqrt(torch.stack(parts).sum()).item())

    def _flat(gs):
        parts = [p.grad.flatten() for p in gs if p.grad is not None]
        if not parts:
            return torch.zeros(0, device=x.device)
        return torch.cat(parts)

    def _zero():
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)

    # CE gradient
    _zero()
    z, _, _, _, _ = model(x, ei)
    ce = crit(head(z[tr]), y[tr])
    ce.backward()
    ce_norms = {g: _norm(groups[g]) for g in group_names}
    g_ce = {g: _flat(groups[g]) for g in group_names}
    del z, ce
    torch.cuda.empty_cache()

    # aux component gradients: each component gets its OWN fresh graph
    # (chaining retain_graph across zero_grad proved fragile — some
    # component gradients came out zero; fresh-graph passes are robust).
    def _aux_pass(component: str):
        _zero()
        factors, _ = model._encode(x)
        common, orth, recon = _aux_components(model, factors)
        if component == "common":
            loss = lam[0] * common
        elif component == "orth":
            loss = lam[1] * orth
        elif component == "recon":
            loss = lam[2] * recon
        else:
            loss = lam[0] * common + lam[1] * orth + lam[2] * recon
        loss.backward()
        norms = {g: _norm(groups[g]) for g in group_names}
        flats = {g: _flat(groups[g]) for g in group_names}
        del factors, common, orth, recon, loss
        torch.cuda.empty_cache()
        return norms, flats

    c_norms, g_c = _aux_pass("common")
    o_norms, g_o = _aux_pass("orth")
    r_norms, g_r = _aux_pass("recon")
    a_norms, g_a = _aux_pass("aux")
    t_norms = {g: (ce_norms[g] ** 2 + a_norms[g] ** 2) ** 0.5 for g in group_names}

    total_ce = sum(v ** 2 for v in ce_norms.values()) ** 0.5
    total_a = sum(v ** 2 for v in a_norms.values()) ** 0.5
    for g in group_names:
        rows_norm.append({
            "dataset": dataset, "state": state, "group": g,
            "g_CE": ce_norms[g], "g_common": c_norms[g], "g_orth": o_norms[g],
            "g_recon": r_norms[g], "g_aux": a_norms[g], "g_total": t_norms[g],
        })

    def _cos(a: dict, b: dict) -> float:
        # cosine over the parameters where BOTH losses have gradients
        # (per-loss support differs: CE skips recon heads, aux skips the
        # classifier)
        parts = [(a[g], b[g]) for g in group_names
                 if a[g].numel() > 0 and b[g].numel() > 0]
        if not parts:
            return float("nan")
        va = torch.cat([p[0] for p in parts])
        vb = torch.cat([p[1] for p in parts])
        return float((va * vb).sum().item() / (va.norm().item() * vb.norm().item() + 1e-12))

    rows_cos.append({
        "dataset": dataset, "state": state,
        "cos_CE_common": _cos(g_ce, g_c), "cos_CE_orth": _cos(g_ce, g_o),
        "cos_CE_recon": _cos(g_ce, g_r), "cos_CE_aux": _cos(g_ce, g_a),
        "aux_CE_norm_ratio": total_a / (total_ce + 1e-12),
    })

    # one optimizer step update ratios (AdamW, training hyper-params)
    _zero()
    z, _, _, aux, _ = model(x, ei)
    loss = crit(head(z[tr]), y[tr]) + aux
    loss.backward()
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4
    )
    before = {g: [p.detach().clone() for p in groups[g]] for g in group_names}
    opt.step()
    for g in group_names:
        num = sum((p - b).square().sum().item() for p, b in zip(groups[g], before[g])) ** 0.5
        den = sum(b.square().sum().item() for b in before[g]) ** 0.5
        # zero-initialized groups (operator residuals, null score at init)
        # have ||theta|| ~ 0 -> the ratio is undefined; report None so the
        # imbalance spread only covers groups with a meaningful norm.
        rows_upd.append({"dataset": dataset, "state": state, "group": g,
                         "update_ratio": (num / den) if den > 1e-8 else None})
    del z, loss, before
    torch.cuda.empty_cache()


def _history_synthesis(rows_hist: list[dict]) -> list[str]:
    for dataset in ALL_DS:
        summary_path = ANCHOR_ROOT / dataset / "A0" / "seed_42" / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open(encoding="utf-8") as f:
            s = json.load(f)
        h = s.get("history") or {}
        rows_hist.append({
            "dataset": dataset,
            "best_epoch": h.get("best_epoch"),
            "stop_epoch": h.get("stop_epoch"),
            "best_val_acc": h.get("best_val_acc"),
            "train_val_gap_at_best": h.get("train_val_gap_at_best"),
            "last20_val_slope_pp_per_epoch": h.get("last20_val_slope_pp_per_epoch"),
            "hit_max_epoch": h.get("hit_max_epoch"),
            "early_plateau": h.get("early_plateau"),
            "overfit_evidence": h.get("overfit_evidence"),
            "epoch_time_sec": s.get("epoch_time_sec"),
            "train_peak_gpu_mb": s.get("train_peak_gpu_mb"),
        })
    lines = ["# R15-1 TRAINING AUDIT — 事实判定", ""]
    lines.append("## 训练历史合成（fresh anchor seed42）")
    lines.append("")
    lines.append("| dataset | best ep | stop ep | best Val | gap | last20 slope pp/ep | hit300 | plateau | overfit | epoch s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows_hist:
        lines.append(
            f"| {r['dataset']} | {r['best_epoch']} | {r['stop_epoch']} | {r['best_val_acc']} | "
            f"{r['train_val_gap_at_best']:.4f} | {r['last20_val_slope_pp_per_epoch']:+.4f} | "
            f"{r['hit_max_epoch']} | {r['early_plateau']} | {r['overfit_evidence']} | {r['epoch_time_sec']} |"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="R15-1 training/gradient audit")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_norm: list[dict] = []
    rows_cos: list[dict] = []
    rows_upd: list[dict] = []
    for di, dataset in enumerate(WEAK):
        device = torch.device(f"cuda:{gpus[di % len(gpus)]}")
        for state in ("init", "best"):
            print(f"[audit] gradients: {dataset} {state}", flush=True)
            _gradient_audit(dataset, state, device, rows_norm, rows_cos, rows_upd)

    def _write_csv(name, rows):
        path = OUT_DIR / name
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    _write_csv("gradient_norms.csv", rows_norm)
    _write_csv("gradient_cosines.csv", rows_cos)
    _write_csv("update_ratios.csv", rows_upd)
    rows_hist: list[dict] = []
    hist_lines = _history_synthesis(rows_hist)
    _write_csv("training_history_summary.csv", rows_hist)

    lines = hist_lines
    lines += [
        "## 梯度分解（init vs best，M/T/G）",
        "",
        "aux/CE norm ratio 判定：<0.2 弱 / 0.2-1 同量级 / >1 可能主导 / >3 强失衡。",
        "cos(g_CE, g_aux) 判定：<−0.2 实质冲突 / <−0.4 强冲突。",
        "",
    ]
    lines.append("| dataset | state | R_aux/CE | cos(CE,common) | cos(CE,orth) | cos(CE,recon) | cos(CE,aux) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows_cos:
        lines.append(
            f"| {r['dataset']} | {r['state']} | {r['aux_CE_norm_ratio']:.3f} | "
            f"{r['cos_CE_common']:+.3f} | {r['cos_CE_orth']:+.3f} | "
            f"{r['cos_CE_recon']:+.3f} | {r['cos_CE_aux']:+.3f} |"
        )
    lines.append("")
    lines.append("| dataset | state | group | ‖g_CE‖ | ‖g_aux‖ | ‖g_total‖ |")
    lines.append("|---|---|---|---:|---:|---:|")
    for r in rows_norm:
        lines.append(
            f"| {r['dataset']} | {r['state']} | {r['group']} | "
            f"{r['g_CE']:.4f} | {r['g_aux']:.4f} | {r['g_total']:.4f} |"
        )
    lines.append("")
    lines.append("| dataset | state | group | U = ‖Δθ‖/‖θ‖ |")
    lines.append("|---|---|---|---:|")
    for r in rows_upd:
        u = r["update_ratio"]
        lines.append(
            f"| {r['dataset']} | {r['state']} | {r['group']} | "
            f"{u:.3e} |" if u is not None else
            f"| {r['dataset']} | {r['state']} | {r['group']} | — |"
        )
    lines.append("")

    # Fact judgments
    ratios = [r["aux_CE_norm_ratio"] for r in rows_cos]
    c_conf = [r["cos_CE_aux"] for r in rows_cos]
    upd = [r["update_ratio"] for r in rows_upd
           if r["group"] not in ("G10_recon_heads",) and r["update_ratio"] is not None]
    upd_by_group = {}
    for r in rows_upd:
        if r["update_ratio"] is None:
            continue
        upd_by_group.setdefault(r["group"], []).append(r["update_ratio"])
    group_means = {g: statistics.mean(v) for g, v in upd_by_group.items()}
    spread = max(group_means.values()) / (min(group_means.values()) + 1e-12) if group_means else 1.0
    mean_ratio = statistics.mean(ratios) if ratios else float("nan")
    min_cos = min(c_conf) if c_conf else float("nan")
    overfit = [r["overfit_evidence"] for r in rows_hist]
    plateau = [r["early_plateau"] for r in rows_hist]

    lines.append("## 判定")
    lines.append("")
    lines.append(f"- aux objective imbalance：{'evidence' if mean_ratio > 1.0 or mean_ratio < 0.2 else 'no evidence'} "
                 f"（mean R_aux/CE = {mean_ratio:.3f}）")
    lines.append(f"- CE-vs-aux gradient conflict：{'evidence' if min_cos < -0.2 else 'no evidence'} "
                 f"（min cos = {min_cos:+.3f}）")
    lines.append(f"- unified LR group imbalance：{'evidence' if spread > 10 else 'no evidence'} "
                 f"（max/min update ratio = {spread:.1f}×）")
    lines.append(f"- under-training / overfitting："
                 f"{'overfit evidence' if any(overfit) else 'no overfit evidence'}"
                 f"；early plateau {'present' if any(plateau) else 'absent'}；"
                 f"hit300 {sum(r['hit_max_epoch'] for r in rows_hist)}/{len(rows_hist)}")
    lines.append("")
    (OUT_DIR / "R15_TRAINING_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] saved -> {OUT_DIR / 'R15_TRAINING_AUDIT.md'}")


if __name__ == "__main__":
    main()
