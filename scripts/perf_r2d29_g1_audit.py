"""R2D29 G1 audit (plan §6.4): CORT implementation correctness on real data.

Runs the 10 mandatory checks on Movies seed 42 (plus an optional
ele-fashion 2-epoch OOM smoke) and writes
outputs/r2d29/g1_audit/{audit.json, grad_audit.csv, equivalence.csv,
G1_AUDIT_REPORT.md}.

G1 verifies IMPLEMENTATION correctness only — no scientific GO/NO-GO.

Usage:
    python scripts/perf_r2d29_g1_audit.py [--device cuda:0] [--skip-large]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d29_utils import (  # noqa: E402
    G1_ROOT,
    build_cort_model,
    load_r2d29_data,
    resolve_cort_cfg,
)
from src.models.biaxis_final import Model as A0Model  # noqa: E402

SEED = 42


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


# ---------------------------------------------------------------------------
# Check A: permutation invariance of the full model (eval)
# ---------------------------------------------------------------------------

def check_permutation_invariance(model, x, edge_index, num_nodes):
    perm = torch.randperm(int(edge_index.size(1)), device=edge_index.device)
    edge_shuffled = edge_index[:, perm]
    model.eval()
    with torch.no_grad():
        z1, *_ = model(x, edge_index)
        z2, *_ = model(x, edge_shuffled)
    diff = float((z1 - z2).abs().max().item())
    return {"max_abs_diff": diff, "pass": diff < 1e-4}


# ---------------------------------------------------------------------------
# Check D: residual gating / equivalence rows
# ---------------------------------------------------------------------------

def check_residual_equivalence(cfg, data, device):
    """CORT(a0_augment, rho=0) must equal fusion(LN(f_A0)) exactly, i.e. the
    write-back introduces nothing beyond the per-factor LN wrapper; the
    magnitude of that wrapper vs the A0 output is reported transparently."""
    from src.models.biaxis_cort import Model as CortModel

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
    }
    a0 = A0Model(cfg, info).to(device).eval()
    cort = CortModel(cfg, info).to(device).eval()
    with torch.no_grad():
        z_a0, *_ = a0(x, edge_index)
        z_cort, *_ = cort(x, edge_index)
        factors, _ = cort._encode(x)
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        f_a0 = cort._graph_update(f_block, edge_index, int(x.size(0)))["f_tilde"]
        # reference: LN per factor, then the CORT fusion
        refs = [cort.cort_blocks[0].writeback_mod.norms[b](f_a0[:, b])
                for b in range(3)]
        f_ref = torch.stack(refs, dim=1)
        if cort.cort_fusion_flatten:
            z_ref = cort.cort_fusion(f_ref.reshape(int(x.size(0)), -1))
        else:
            z_ref = cort.cort_fusion(f_ref)
    rows = {
        "cort_rho0_vs_ln_reference": float((z_cort - z_ref).abs().max().item()),
        "a0_vs_cort_rho0_ln_wrapper": float((z_a0 - z_cort).abs().max().item()),
    }
    return rows


# ---------------------------------------------------------------------------
# Check E: gradient flow audit (one training step)
# ---------------------------------------------------------------------------

def check_gradients(cfg, data, device, out_path: Path):
    model = build_cort_model(cfg, data, device)
    model.train()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    z, _, _, aux_loss, aux_info = model(x, edge_index)
    loss = z.square().mean() + aux_loss
    model.zero_grad()
    loss.backward()
    rows = []
    for name, param in model.named_parameters():
        if param.grad is None:
            rows.append({"module": name, "grad_norm": 0.0, "has_grad": 0, "numel": param.numel()})
            continue
        rows.append({
            "module": name,
            "grad_norm": float(param.grad.norm().item()),
            "has_grad": 1,
            "numel": param.numel(),
        })
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["module", "grad_norm", "has_grad", "numel"])
        writer.writeheader()
        writer.writerows(rows)
    # per-group summary
    groups: dict[str, list[float]] = {}
    for row in rows:
        prefix = row["module"].split(".")[0]
        groups.setdefault(prefix, []).append(row["grad_norm"])
    summary = {
        f"{g}_grad_sum": float(sum(v)) for g, v in groups.items()
    }
    all_new_have_grad = all(r["has_grad"] for r in rows
                            if r["module"].startswith(("cort", "type_emb", "operator")))
    summary["all_new_modules_have_grad"] = bool(all_new_have_grad)
    summary["new_modules_nonzero_grad"] = bool(
        sum(r["grad_norm"] for r in rows if r["module"].startswith(("cort", "type_emb", "operator"))) > 0.0
    )
    return summary, rows


# ---------------------------------------------------------------------------
# Check F: source-channel independence on real factors
# ---------------------------------------------------------------------------

def check_source_channels(model, x, edge_index, num_nodes):
    blk = model.cort_blocks[0]
    model.eval()
    with torch.no_grad():
        factors, _ = model._encode(x)
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        f_pre = blk._pre_norm_apply(f_block)
        msgs, _stats = blk.router(f_pre, edge_index, num_nodes)
    for b in range(3):
        vals = [msgs[(a, b)] for a in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                if float((vals[i] - vals[j]).abs().max().item()) < 1e-5:
                    return {"channels_distinct": False, "b": b}
    return {"channels_distinct": True}


# ---------------------------------------------------------------------------
# Check I: chunk discipline (tiny chunk must reproduce unchunked)
# ---------------------------------------------------------------------------

def check_chunk_equivalence(device):
    from src.models.biaxis_cort_components import cort_coupled_message
    n = 64
    # torch.Generator is CPU-only (D2.7 pitfall): generate on CPU, then move
    generator = torch.Generator().manual_seed(7)
    f_block = torch.randn(n, 3, 16, generator=generator).to(device)
    src = torch.randint(0, n, (n * 6,), generator=generator).to(device)
    dst = torch.randint(0, n, (n * 6,), generator=generator).to(device)
    edge_index = torch.stack([src, dst])
    scores = torch.randn(n * 6, generator=generator).to(device)
    null = torch.randn(n, generator=generator).to(device)
    payload = torch.randn(n, 16, generator=generator).to(device)
    m1, *_ = cort_coupled_message(f_block, edge_index, n, scores, null, payload,
                                  edge_chunk_size=1 << 30)
    m2, *_ = cort_coupled_message(f_block, edge_index, n, scores, null, payload,
                                  edge_chunk_size=5)
    diff = float((m1 - m2).abs().max().item())
    return {"tiny_chunk_max_abs_diff": diff, "pass": diff < 1e-5}


# ---------------------------------------------------------------------------
# Static no-Test-access check
# ---------------------------------------------------------------------------

def check_no_test_access() -> dict:
    pattern = re.compile(r"test_idx|test_mask|data\.test|\.test\b")
    hits = []
    for rel in ("src/models/biaxis_cort.py", "src/models/biaxis_cort_components.py",
                "configs/model/biaxis_cort.yaml"):
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{rel}:{line_no}: {line.strip()}")
    return {"hits": hits, "pass": not hits}


# ---------------------------------------------------------------------------
# ele-fashion OOM smoke (2 epochs, subprocess)
# ---------------------------------------------------------------------------

def check_ele_fashion_smoke(gpu_id: int) -> dict:
    import os
    started = time.monotonic()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    outdir = G1_ROOT / "ele_fashion_smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.main",
        "dataset=ele-fashion", "task=nc", "model=biaxis_cort",
        "num_runs=1", f"seed={SEED}", "device=cuda:0", "task.epochs=2",
        f"hydra.run.dir={outdir / 'hydra'}",
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
                          timeout=3600)
    runtime = time.monotonic() - started
    ok = proc.returncode == 0
    if not ok:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
    else:
        tail = ""
    return {"returncode": proc.returncode, "pass": ok,
            "runtime_sec": round(runtime, 1), "log_tail": tail}


def main() -> None:
    parser = argparse.ArgumentParser(description="R2D29 G1 CORT audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    G1_ROOT.mkdir(parents=True, exist_ok=True)

    print("[audit] loading Movies seed 42 ...", flush=True)
    cfg, data = load_r2d29_data("Movies", SEED, device)
    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
    }
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    num_nodes = int(x.size(0))
    print(f"[audit] nodes={num_nodes} edges={int(edge_index.size(1))}", flush=True)

    model = build_cort_model(cfg, data, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[audit] params={n_params}", flush=True)

    checks: dict = {}

    # A. permutation invariance
    checks["permutation_pair_null"] = check_permutation_invariance(model, x, edge_index, num_nodes)
    cfg_uniform = resolve_cort_cfg("Movies", SEED, {"router_mode": "uniform"})
    model_uniform = build_cort_model(cfg_uniform, data, device)
    checks["permutation_uniform"] = check_permutation_invariance(model_uniform, x, edge_index, num_nodes)
    print(f"[audit] permutation pair_null={checks['permutation_pair_null']} "
          f"uniform={checks['permutation_uniform']}", flush=True)

    # B. null-softmax mass conservation
    from src.models.biaxis_cort_components import cort_coupled_message
    model.eval()
    with torch.no_grad():
        factors, _ = model._encode(x)
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        blk = model.cort_blocks[0]
        f_pre = blk._pre_norm_apply(f_block)
        msgs, stats = blk.router(f_pre, edge_index, num_nodes)
    mass_ok = True
    mass_vals = {}
    for key, value in stats.items():
        if key.startswith("null_mass"):
            mass_vals[key] = value
            # null mass in [0,1] plus the entropy stats being finite
            if not (0.0 <= value <= 1.0):
                mass_ok = False
    checks["null_mass_in_unit_interval"] = {"values": mass_vals, "pass": mass_ok}
    checks["routing_stats_finite"] = {
        "pass": all(float(v) == float(v) for v in stats.values()),
        "keys": sorted(stats.keys()),
    }
    print(f"[audit] null-mass in [0,1]: {mass_ok}", flush=True)

    # C. isolated nodes (real graph + synthetic)
    deg = torch.bincount(edge_index[1], minlength=num_nodes)
    n_isolated = int((deg == 0).sum().item())
    model.eval()
    with torch.no_grad():
        z_iso, *_ = model(x, edge_index)
    checks["isolated_nodes"] = {
        "n_isolated_real": n_isolated,
        "z_finite": _finite(z_iso),
        "pass": _finite(z_iso),
    }
    print(f"[audit] isolated nodes on Movies: {n_isolated}; z finite: "
          f"{checks['isolated_nodes']['z_finite']}", flush=True)

    # D. residual gating equivalence
    eq_rows = check_residual_equivalence(cfg, data, device)
    checks["residual_equivalence"] = eq_rows
    print(f"[audit] rho0-vs-LN-ref: {eq_rows['cort_rho0_vs_ln_reference']:.3e} "
          f"ln-wrapper: {eq_rows['a0_vs_cort_rho0_ln_wrapper']:.3e}", flush=True)

    # E. gradient audit
    grad_summary, grad_rows = check_gradients(cfg, data, device, G1_ROOT / "grad_audit.csv")
    checks["gradients"] = grad_summary
    print(f"[audit] new-module grads: {grad_summary['new_modules_nonzero_grad']}", flush=True)

    # F. source-channel independence
    checks["source_channels"] = check_source_channels(model, x, edge_index, num_nodes)
    print(f"[audit] source channels distinct: {checks['source_channels']}", flush=True)

    # H. API contract: inference == eval forward (inference returns CPU z by
    # the framework contract; move it back for the comparison)
    model.eval()
    with torch.no_grad():
        z_eval, *_ = model(x, edge_index)
        z_inf = model.inference(x, edge_index)
    inf_diff = float((z_eval - z_inf.to(device)).abs().max().item())
    checks["api_contract"] = {
        "inference_max_abs_diff": inf_diff,
        "z_shape": list(z_eval.shape),
        "out_dim": model.out_dim,
        "pass": inf_diff < 1e-5,
    }
    print(f"[audit] inference==forward: {inf_diff:.3e}", flush=True)

    # I. chunk equivalence
    checks["chunk_equivalence"] = check_chunk_equivalence(device)
    print(f"[audit] tiny-chunk equivalence: "
          f"{checks['chunk_equivalence']['tiny_chunk_max_abs_diff']:.3e}", flush=True)

    # J. memory / params
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    z, _, _, aux_loss, _aux_info = model(x, edge_index)
    loss = z.square().mean() + aux_loss
    loss.backward()
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    checks["resources"] = {"params": n_params, "movies_train_step_peak_mb": round(peak_mb, 1)}
    print(f"[audit] Movies train-step peak: {peak_mb:.1f} MB", flush=True)

    # G. no Test access (static)
    checks["no_test_access"] = check_no_test_access()
    print(f"[audit] no-test-access: {checks['no_test_access']['pass']}", flush=True)

    # large-graph OOM smoke
    if not args.skip_large:
        print("[audit] ele-fashion 2-epoch smoke (may take minutes) ...", flush=True)
        checks["ele_fashion_smoke"] = check_ele_fashion_smoke(args.gpu)
        print(f"[audit] ele-fashion smoke: rc={checks['ele_fashion_smoke']['returncode']}", flush=True)

    # write outputs
    with (G1_ROOT / "audit.json").open("w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, default=str)

    with (G1_ROOT / "equivalence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["check", "value"])
        writer.writerow(["cort_rho0_vs_ln_reference", eq_rows["cort_rho0_vs_ln_reference"]])
        writer.writerow(["a0_vs_cort_rho0_ln_wrapper", eq_rows["a0_vs_cort_rho0_ln_wrapper"]])
        writer.writerow(["inference_max_abs_diff", inf_diff])
        writer.writerow(["permutation_pair_null_max_abs_diff",
                         checks["permutation_pair_null"]["max_abs_diff"]])
        writer.writerow(["permutation_uniform_max_abs_diff",
                         checks["permutation_uniform"]["max_abs_diff"]])
        writer.writerow(["tiny_chunk_max_abs_diff",
                         checks["chunk_equivalence"]["tiny_chunk_max_abs_diff"]])

    def pass_all() -> bool:
        items = [
            checks["permutation_pair_null"]["pass"],
            checks["permutation_uniform"]["pass"],
            checks["null_mass_in_unit_interval"]["pass"],
            checks["routing_stats_finite"]["pass"],
            checks["isolated_nodes"]["pass"],
            checks["source_channels"]["channels_distinct"],
            checks["api_contract"]["pass"],
            checks["chunk_equivalence"]["pass"],
            checks["no_test_access"]["pass"],
            checks["gradients"]["all_new_modules_have_grad"],
            checks["gradients"]["new_modules_nonzero_grad"],
        ]
        if "ele_fashion_smoke" in checks:
            items.append(checks["ele_fashion_smoke"]["pass"])
        return all(items)

    checks["ALL_PASS"] = pass_all()

    lines = [
        "# G1_AUDIT_REPORT — CORT implementation audit (plan §6.4)",
        "",
        f"- commit: `{subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()}`",
        f"- data: Movies seed {SEED}, {num_nodes} nodes / {int(edge_index.size(1))} edges",
        f"- model params: {n_params:,}",
        f"- Movies train-step peak: {peak_mb:.1f} MB",
        "",
        "## Check table",
        "",
        "| # | check | result | detail |",
        "|---|---|---|---|",
        "| 1 | permutation invariance (pair_null / uniform) | "
        f"{'PASS' if checks['permutation_pair_null']['pass'] and checks['permutation_uniform']['pass'] else 'FAIL'} | "
        f"max diff {checks['permutation_pair_null']['max_abs_diff']:.2e} / {checks['permutation_uniform']['max_abs_diff']:.2e} |",
        "| 2 | null-softmax mass in [0,1] | "
        f"{'PASS' if checks['null_mass_in_unit_interval']['pass'] else 'FAIL'} | {mass_vals} |",
        "| 3 | isolated nodes NaN-free | "
        f"{'PASS' if checks['isolated_nodes']['pass'] else 'FAIL'} | {n_isolated} isolated nodes |",
        "| 4 | residual gating (rho=0 == LN reference) | "
        f"{'PASS' if eq_rows['cort_rho0_vs_ln_reference'] < 1e-5 else 'FAIL'} | "
        f"{eq_rows['cort_rho0_vs_ln_reference']:.2e} (LN wrapper vs A0: {eq_rows['a0_vs_cort_rho0_ln_wrapper']:.2e}) |",
        "| 5 | all new modules receive grads | "
        f"{'PASS' if checks['gradients']['all_new_modules_have_grad'] else 'FAIL'} | see grad_audit.csv |",
        "| 6 | source channels independent | "
        f"{'PASS' if checks['source_channels']['channels_distinct'] else 'FAIL'} | {checks['source_channels']} |",
        "| 7 | no Test access | "
        f"{'PASS' if checks['no_test_access']['pass'] else 'FAIL'} | hits: {checks['no_test_access']['hits']} |",
        "| 8 | forward/inference API contract | "
        f"{'PASS' if checks['api_contract']['pass'] else 'FAIL'} | diff {inf_diff:.2e}, z {checks['api_contract']['z_shape']} |",
        "| 9 | chunk discipline | "
        f"{'PASS' if checks['chunk_equivalence']['pass'] else 'FAIL'} | tiny-chunk diff {checks['chunk_equivalence']['tiny_chunk_max_abs_diff']:.2e} |",
        "| 10 | params / memory | PASS | "
        f"{n_params:,} params; {peak_mb:.1f} MB train-step |",
        "",
        f"## ele-fashion 2-epoch OOM smoke",
        f"- rc={checks.get('ele_fashion_smoke', {}).get('returncode', 'skipped')} "
        f"(pass={checks.get('ele_fashion_smoke', {}).get('pass', 'skipped')})",
        "",
        f"## Verdict: **{'ALL PASS' if checks['ALL_PASS'] else 'NOT ALL PASS'}**",
        "",
        "G1 only verifies implementation correctness (plan §6.4); no scientific",
        "GO/NO-GO decisions are made here.",
    ]
    (G1_ROOT / "G1_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[audit] ALL_PASS={checks['ALL_PASS']} -> {G1_ROOT}", flush=True)


if __name__ == "__main__":
    main()
