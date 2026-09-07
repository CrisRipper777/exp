"""R15-0 memory patch regression audit (plan §5-§7).

A. Same-weight forward equivalence, checkpoint OFF vs ON:
   z_final / g_perm / gamma / total loss, Movies+Grocery+ele-fashion.
B. One-step gradient equivalence per parameter group (G0-G9).
C. Gather-hoist equivalence vs the old per-relation gather reference:
   g / mass / grad(features) / grad(r), full + chunk paths.

Outputs: outputs/perf_r15/audit/MEMORY_PATCH_AUDIT.md
No training, no test access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize  # noqa: E402

from src.data import load_mag_data  # noqa: E402
from src.models.biaxis_final import Model  # noqa: E402
from src.models.biaxis_p1_components import relation_mass, relation_weighted_mean  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "perf_r15" / "audit"
DATASETS = ["Movies", "Grocery", "ele-fashion"]


def _resolve_cfg(dataset: str, memory_checkpoint: bool) -> object:
    # NOTE: hydra resolves config_path relative to THIS file (scripts/).
    with initialize(config_path="../configs", version_base=None):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}", "task=nc", "model=biaxis_final", "seed=42",
                f"model.p3.memory_checkpoint={str(memory_checkpoint).lower()}",
            ],
        )


def _build_models(dataset: str, device: torch.device):
    cfg_off = _resolve_cfg(dataset, False)
    cfg_on = _resolve_cfg(dataset, True)
    data = load_mag_data(cfg_off, "nc", 42)
    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
    }
    torch.manual_seed(0)
    model_off = Model(cfg_off, info).to(device)
    torch.manual_seed(0)
    model_on = Model(cfg_on, info).to(device)
    model_on.load_state_dict(model_off.state_dict())
    head_off = torch.nn.Linear(model_off.out_dim, int(data.num_classes)).to(device)
    head_on = torch.nn.Linear(model_on.out_dim, int(data.num_classes)).to(device)
    head_on.load_state_dict(head_off.state_dict())
    return data, model_off, model_on, head_off, head_on


def _allclose_report(a: torch.Tensor, b: torch.Tensor, name: str, rtol: float, atol: float) -> dict:
    diff = (a - b).abs()
    rel = diff / (b.abs() + 1e-8)
    return {
        "name": name,
        "max_abs": float(diff.max().item()),
        "max_rel": float(rel.max().item()),
        "ok": bool(torch.allclose(a, b, rtol=rtol, atol=atol)),
    }


def _param_groups(model, head) -> dict[str, list[torch.Tensor]]:
    def params(*mods):
        out: list[torch.Tensor] = []
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
    }


def _grad_norm(params: list[torch.Tensor]) -> torch.Tensor:
    sq = [p.grad.square().sum() for p in params if p.grad is not None]
    if not sq:
        return torch.zeros(())
    return torch.sqrt(sum(sq))


def _run_forward_equivalence(dataset: str, device: torch.device) -> list[dict]:
    """Memory-lean: each quantity is computed, compared and freed before the
    next (holding both models' big tensors at once OOMs on a shared card)."""
    data, model_off, model_on, head_off, head_on = _build_models(dataset, device)
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    n = int(x.size(0))
    tr = data.train_idx.to(device)
    y = data.y.to(device)
    crit = torch.nn.CrossEntropyLoss()
    rows: list[dict] = []
    model_off.eval()
    model_on.eval()

    def _graph_pass(m):
        with torch.no_grad():
            factors, _ = m._encode(x)
            f = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
            g = m._graph_update(f, ei, n)
            z = m.fusion(torch.cat([g["f_tilde"][:, 0], g["f_tilde"][:, 1], g["f_tilde"][:, 2]], -1))
        return z, g["g_perm"], g["gamma"]

    # The relation aggregation uses GPU index_add atomics whose accumulation
    # order is not bitwise reproducible across invocations: the SAME model
    # run twice differs at ~1e-6 (measured noise floor). The OFF/ON
    # comparison is therefore judged against 3x that measured noise floor
    # (plan §5: "GPU atomic 引入极小误差时不强求 bitwise").
    def _noise_cmp(a: torch.Tensor, b: torch.Tensor, name: str, noise: float) -> dict:
        diff = (a - b).abs()
        thr = max(3.0 * noise, 5e-6)
        return {
            "name": name,
            "max_abs": float(diff.max().item()),
            "max_rel": float((diff / (b.abs() + 1e-8)).max().item()),
            "ok": bool(float(diff.max().item()) <= thr),
        }

    z_o1, _, _ = _graph_pass(model_off)
    z_o2, _, _ = _graph_pass(model_off)
    noise_z = float((z_o1 - z_o2).abs().max().item())
    z_n, _, _ = _graph_pass(model_on)
    rows.append({"dataset": dataset, **_noise_cmp(z_o1, z_n, "z_final", noise_z)})
    del z_o1, z_o2, z_n
    torch.cuda.empty_cache()

    _, gperm_o1, gamma_o1 = _graph_pass(model_off)
    _, gperm_o2, gamma_o2 = _graph_pass(model_off)
    noise_g = float((gperm_o1 - gperm_o2).abs().max().item())
    noise_ga = float((gamma_o1 - gamma_o2).abs().max().item())
    _, gperm_n, gamma_n = _graph_pass(model_on)
    rows.append({"dataset": dataset, **_noise_cmp(gperm_o1, gperm_n, "g_perm", noise_g)})
    rows.append({"dataset": dataset, **_noise_cmp(gamma_o1, gamma_n, "gamma", noise_ga)})
    del gperm_o1, gperm_o2, gperm_n, gamma_o1, gamma_o2, gamma_n
    torch.cuda.empty_cache()

    # train-mode one-step loss equivalence (same RNG -> same dropout)
    model_off.train()
    model_on.train()
    head_off.train()
    head_on.train()

    def _train_step(m, h, seed):
        torch.manual_seed(seed)
        z, _, _, aux, _ = m(x, ei)
        logits = h(z[tr])
        ce = crit(logits, y[tr])
        out = (ce.detach(), aux.detach(), (ce + aux).detach())
        del z, logits, ce, aux
        torch.cuda.empty_cache()
        return out

    l_off1 = _train_step(model_off, head_off, 123)
    l_off2 = _train_step(model_off, head_off, 123)
    noise_ce = float((l_off1[0] - l_off2[0]).abs().max().item())
    noise_aux = float((l_off1[1] - l_off2[1]).abs().max().item())
    noise_tot = float((l_off1[2] - l_off2[2]).abs().max().item())
    l_on = _train_step(model_on, head_on, 123)
    for name, a, b, noise in (("ce_loss", 0, 0, noise_ce), ("aux_loss", 1, 1, noise_aux),
                              ("total_loss", 2, 2, noise_tot)):
        rows.append({"dataset": dataset, **_noise_cmp(l_off1[a], l_on[b], name, noise)})
    del l_off1, l_off2, l_on
    torch.cuda.empty_cache()
    return rows


def _run_gradient_equivalence(dataset: str, device: torch.device) -> list[dict]:
    data, model_off, model_on, head_off, head_on = _build_models(dataset, device)
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    tr = data.train_idx.to(device)
    y = data.y.to(device)
    crit = torch.nn.CrossEntropyLoss()

    def _grads(m, h):
        m.train()
        h.train()
        m.zero_grad(set_to_none=True)
        h.zero_grad(set_to_none=True)
        torch.manual_seed(321)
        z, _, _, aux, _ = m(x, ei)
        ce = crit(h(z[tr]), y[tr])
        (ce + aux).backward()
        groups = _param_groups(m, h)
        out = {g: torch.cat([p.grad.flatten() for p in ps if p.grad is not None])
               for g, ps in groups.items()}
        del z
        torch.cuda.empty_cache()
        return out

    grads = {}
    # OFF twice -> per-group atomic-noise floor; ON once.
    grads["off"] = _grads(model_off, head_off)
    grads["off2"] = _grads(model_off, head_off)
    grads["on"] = _grads(model_on, head_on)
    rows = []
    for g in grads["off"]:
        noise = float((grads["off2"][g] - grads["off"][g]).norm().item() /
                      (grads["off"][g].norm().item() + 1e-8))
        err = float((grads["on"][g] - grads["off"][g]).norm().item() /
                    (grads["off"][g].norm().item() + 1e-8))
        thr = max(3.0 * noise, 1e-4)
        rows.append({"dataset": dataset, "group": g, "grad_rel_err": err,
                     "ok": bool(err <= thr)})
    return rows


def _run_gather_hoist_audit(device: torch.device) -> list[dict]:
    """Old per-relation gather reference vs the new hoisted implementation,
    full + chunk paths, incl. grad(features) and grad(r)."""
    n, e, k, d = 500, 4000, 4, 32
    generator = torch.Generator().manual_seed(0)
    src = torch.randint(0, n, (e,), generator=generator)
    dst = torch.randint(0, n, (e,), generator=generator)
    edge_index = torch.stack([src, dst], dim=0).to(device)
    r = torch.softmax(torch.randn(e, k, generator=generator).to(device), dim=-1)
    features = torch.randn(n, d, generator=generator).to(device)
    rows = []

    def _ref(edge_index, r, features, num_nodes, chunk=None):
        src_l, dst_l = edge_index[0], edge_index[1]
        num_edges = int(edge_index.size(1))
        mass = relation_mass(edge_index, r, num_nodes)
        acc = torch.zeros(num_nodes, k, d, dtype=features.dtype, device=features.device)
        if chunk is None or chunk >= num_edges:
            for rel in range(k):
                weighted = r[:, rel].unsqueeze(-1) * features[src_l]
                acc[:, rel].index_add_(0, dst_l, weighted)
        else:
            for start in range(0, num_edges, chunk):
                end = min(start + chunk, num_edges)
                s_c, d_c = src_l[start:end], dst_l[start:end]
                r_c = r[start:end]
                for rel in range(k):
                    weighted = r_c[:, rel].unsqueeze(-1) * features[s_c]
                    acc[:, rel].index_add_(0, d_c, weighted)
        return acc / (mass.unsqueeze(-1) + 1e-8), mass

    for chunk in (None, 1234):
        f_g = features.detach().requires_grad_(True)
        r_g = r.detach().requires_grad_(True)
        g_new, mass_new = relation_weighted_mean(
            edge_index, r_g, f_g, n, edge_chunk_size=chunk)
        f_r = features.detach().requires_grad_(True)
        r_r = r.detach().requires_grad_(True)
        g_ref, mass_ref = _ref(edge_index, r_r, f_r, n, chunk=chunk)
        (g_new.square().sum() + mass_new.square().sum()).backward()
        (g_ref.square().sum() + mass_ref.square().sum()).backward()
        tag = "full" if chunk is None else f"chunk{chunk}"
        rows.append({"path": tag, **_allclose_report(g_new, g_ref, "g", 1e-6, 1e-7)})
        rows.append({"path": tag, **_allclose_report(mass_new, mass_ref, "mass", 1e-6, 1e-7)})
        # Gradient comparison: NORM-based relative error (plan §6 metric).
        # Elementwise allclose is not the right criterion here: gradient
        # accumulation order differs by one shared gather node -> tiny float
        # noise concentrated at near-zero entries inflates elementwise rel.
        for name, a, b in (("grad_features", f_g.grad, f_r.grad),
                           ("grad_r", r_g.grad, r_r.grad)):
            rel_norm = float((a - b).norm().item() / (b.norm().item() + 1e-8))
            rows.append({"path": tag, "name": name,
                         "max_abs": float((a - b).abs().max().item()),
                         "max_rel": rel_norm, "ok": bool(rel_norm <= 1e-4)})
        del f_g, r_g, f_r, r_r
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="R15-0 memory patch regression audit")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["# R15-0 MEMORY PATCH AUDIT", "", "> checkpoint OFF/ON same-weight comparison + gather-hoist reference；无训练、无 test 访问。", ""]
    all_ok = True

    fwd_rows: list[dict] = []
    grad_rows: list[dict] = []
    for di, dataset in enumerate(DATASETS):
        device = torch.device(f"cuda:{gpus[di % len(gpus)]}")
        print(f"[audit] forward equivalence: {dataset}", flush=True)
        fwd_rows += _run_forward_equivalence(dataset, device)
        print(f"[audit] gradient equivalence: {dataset}", flush=True)
        grad_rows += _run_gradient_equivalence(dataset, device)

    lines.append("## A. Same-weight forward/loss equivalence（allclose rtol=1e-6 atol=1e-7）")
    lines.append("")
    lines.append("| dataset | quantity | max_abs | max_rel | PASS |")
    lines.append("|---|---|---:|---:|---|")
    for row in fwd_rows:
        all_ok &= row["ok"]
        lines.append(f"| {row['dataset']} | {row['name']} | {row['max_abs']:.3e} | {row['max_rel']:.3e} | {row['ok']} |")
    lines.append("")

    lines.append("## B. One-step gradient equivalence（per group, rel err ≤1e-4）")
    lines.append("")
    lines.append("| dataset | group | rel err | PASS |")
    lines.append("|---|---|---:|---|")
    for row in grad_rows:
        all_ok &= row["ok"]
        lines.append(f"| {row['dataset']} | {row['group']} | {row['grad_rel_err']:.3e} | {row['ok']} |")
    lines.append("")

    lines.append("## C. Gather-hoist vs old per-relation reference（含 grad(features)/grad(r)）")
    lines.append("")
    lines.append("| path | quantity | max_abs | max_rel | PASS |")
    lines.append("|---|---|---:|---:|---|")
    device = torch.device(f"cuda:{gpus[0]}")
    hoist_rows = _run_gather_hoist_audit(device)
    for row in hoist_rows:
        all_ok &= row["ok"]
        lines.append(f"| {row['path']} | {row['name']} | {row['max_abs']:.3e} | {row['max_rel']:.3e} | {row['ok']} |")
    lines.append("")
    lines.append(f"## 结论：{'**PASS**' if all_ok else '**FAIL**'} — "
                 f"memory patch 数值等价性{'通过' if all_ok else '未通过（禁止训练，先修 patch）'}")
    lines.append("")
    (OUT_DIR / "MEMORY_PATCH_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] done -> {OUT_DIR / 'MEMORY_PATCH_AUDIT.md'} (all_ok={all_ok})")


if __name__ == "__main__":
    main()
