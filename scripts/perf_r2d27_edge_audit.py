"""R2-D2.7-B: edge-utility structure & causal ranking audit
(docs/BiAxis_R2_Design_2_7_PreAggregation_Neighbor_Utility_Audit.md §22-§28).

Uses the PAIR_EDGE best checkpoints. No retraining. No Test.

Per target / factor-pair: null mass, entropy, normalized entropy, Gini,
top-10/25% mass, effective neighbor count.
Factor-pair diversity: 9x9 JSD / Spearman / top-k overlap.
Heuristic correlations: cos(F_i^b,F_j^a), source degree, target degree.
Train-label-only homophily: same/different-label utility (TRAIN only).
Causal ranking: REMOVE_TOP/RANDOM/BOTTOM 10/25/50%, KEEP_TOP 25/50%.
Permutation controls: within-target shuffle, source-node shuffle,
factor-id shuffle.

Outputs: outputs/perf_r2d27/edge_audit/<ds>/PAIR_EDGE/seed_<s>/summary.json
    + the CSVs / report via scripts/summarize_perf_r2d27.py --stage edge_audit

Usage:
    python scripts/perf_r2d27_edge_audit.py --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d27_utils import (  # noqa: E402
    DATASETS,
    R2D27_ROOT,
    SEEDS,
    causal_metrics,
    load_a0_parent,
)

EDGE_AUDIT_ROOT = R2D27_ROOT / "edge_audit"
MATRIX_ROOT = R2D27_ROOT / "matrix"

CAUSAL_KEYS = (
    "full", "remove_top_10", "remove_top_25", "remove_top_50",
    "remove_random_10", "remove_random_25", "remove_random_50",
    "remove_bottom_10", "remove_bottom_25", "remove_bottom_50",
    "keep_top_25", "keep_top_50",
    "within_target_shuffle", "source_shuffle", "factor_id_shuffle",
)


class _Semaphore:
    def __init__(self, value: int) -> None:
        self._cond = threading.Condition()
        self._value = int(value)

    def acquire(self) -> None:
        with self._cond:
            while self._value < 1:
                self._cond.wait()
            self._value -= 1

    def release(self) -> None:
        with self._cond:
            self._value += 1
            self._cond.notify_all()


def _gini(values: torch.Tensor) -> float:
    """Gini coefficient of a 1-D tensor (exact, sorted)."""
    if values.numel() < 2:
        return 0.0
    v = torch.sort(values.float())[0]
    n = float(v.numel())
    idx = torch.arange(1, v.numel() + 1, dtype=torch.float64)
    s = v.double().sum()
    if s <= 0:
        return 0.0
    return float((2.0 * (idx * v.double()).sum() / (n * s)) - (n + 1.0) / n)


def run_worker(dataset: str, seed: int, outdir: Path, force: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists() and not force:
        print(f"[{dataset} s{seed}] SKIP", flush=True)
        return
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    setup = load_a0_parent(dataset, seed, device)
    from scripts.perf_r2d27_matrix import resolve_cfg

    cfg = resolve_cfg(dataset, seed, "PAIR_EDGE")
    info = {
        "input_dim": setup.data.input_dim, "num_nodes": setup.data.num_nodes,
        "num_classes": setup.data.num_classes,
        "text_dim": int(setup.data.x_t.shape[1]), "visual_dim": int(setup.data.x_i.shape[1]),
    }
    ckpt_path = MATRIX_ROOT / dataset / "PAIR_EDGE" / f"seed_{seed}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from src.models.biaxis_r2_neighbor_utility import Model

    model = Model(cfg, info, setup.parent).to(device)
    model.load_state_dict(ckpt["model_state"])
    head = torch.nn.Linear(model.out_dim, int(setup.data.num_classes)).to(device)
    head.load_state_dict(ckpt["head_state"])
    model.eval()
    head.eval()

    x = setup.data.x.to(device)
    ei = setup.data.edge_index.to(device)
    num_nodes = int(x.size(0))
    num_edges = int(ei.size(1))
    src, dst = ei[0], ei[1]

    with torch.no_grad():
        f_block, _ = model._parent_ctx(x, ei, num_nodes)
        exported = model.export_pair_scores(x, ei)

    # ---- per-pair per-target statistics (CPU; exported tensors are CPU) -----
    dst_c = dst.cpu()
    pair_rows = []
    for a in range(3):
        for b in range(3):
            key = f"pair_{a}{b}"
            stats = exported[key]
            s, alpha, null = stats["scores"], stats["alpha"], stats["null_mass"]
            # per-target aggregates via scatter
            deg = torch.bincount(dst_c, minlength=num_nodes).float()
            # normalized entropy per target over {null} + neighbors
            total = alpha.new_zeros(num_nodes).scatter_add(0, dst_c, alpha) + null
            p_nei = alpha / (total[dst_c] + 1e-12)
            p_null = null / (total + 1e-12)
            ent = -(p_nei * torch.log(p_nei + 1e-12)).new_zeros(num_nodes)
            ent = ent.scatter_add(0, dst_c, -(p_nei * torch.log(p_nei + 1e-12)))
            ent = ent - p_null * torch.log(p_null + 1e-12)
            norm_ent = ent / (torch.log(deg + 1.0) + 1e-12)
            eff_count = torch.exp(ent)
            # top-k mass (per target: mass of top 25%/50% of that target's
            # neighbors — computed via per-target sorted alphas; here use
            # global-per-pair top-k mass for the summary plus per-target
            # Gini via scatter of sorted contributions is too heavy; export
            # per-edge alpha and compute Gini per target with a scatter loop
            # on a node sample (all nodes for M/T/G, subsample for guards).
            top10_q = torch.quantile(s, 0.90)
            top25_q = torch.quantile(s, 0.75)
            mass_top10 = float((alpha[s >= top10_q].sum() / (alpha.sum() + 1e-12)).item())
            mass_top25 = float((alpha[s >= top25_q].sum() / (alpha.sum() + 1e-12)).item())
            pair_rows.append({
                "dataset": dataset, "seed": seed, "pair": key,
                "null_mass_mean": float(null.mean().item()),
                "null_mass_frac_lt_05": float((null < 0.05).float().mean().item()),
                "null_mass_frac_gt_95": float((null > 0.95).float().mean().item()),
                "real_mass_mean": float((1.0 - null).mean().item()),
                "entropy_mean": float(ent.mean().item()),
                "norm_entropy_mean": float(norm_ent.mean().item()),
                "eff_neighbor_count_mean": float(eff_count.mean().item()),
                "gini_global": _gini(alpha),
                "mass_top10": mass_top10,
                "mass_top25": mass_top25,
            })

    # ---- pair diversity (9x9 JSD / Spearman / top-k overlap) ---------------
    # use per-edge alpha of each pair (same edge order) — Spearman on scores
    pair_scores = {}
    for a in range(3):
        for b in range(3):
            pair_scores[f"pair_{a}{b}"] = exported[f"pair_{a}{b}"]["scores"].cpu().float()
    keys = list(pair_scores.keys())
    jsd = [[0.0] * 9 for _ in range(9)]
    spearman = [[0.0] * 9 for _ in range(9)]
    topk_overlap = [[0.0] * 9 for _ in range(9)]
    k_top = max(1, int(num_edges * 0.1))
    for i, ki in enumerate(keys):
        top_i = set(torch.topk(pair_scores[ki], k_top).indices.tolist())
        for j, kj in enumerate(keys):
            if i == j:
                jsd[i][j] = 0.0
                spearman[i][j] = 1.0
                topk_overlap[i][j] = 1.0
                continue
            # JSD between the two alpha distributions (alpha sums ~ N)
            a1 = exported[ki]["alpha"].cpu().float()
            a2 = exported[kj]["alpha"].cpu().float()
            p1 = a1 / (a1.sum() + 1e-12)
            p2 = a2 / (a2.sum() + 1e-12)
            m_ = 0.5 * (p1 + p2)
            kl = lambda p, q: (p * torch.log((p + 1e-12) / (q + 1e-12))).sum()
            jsd[i][j] = float((0.5 * kl(p1, m_) + 0.5 * kl(p2, m_)).item())
            # Spearman rank correlation of the scores
            r1 = torch.argsort(torch.argsort(pair_scores[ki]))
            r2 = torch.argsort(torch.argsort(pair_scores[kj]))
            spearman[i][j] = float(
                torch.corrcoef(torch.stack([r1.float(), r2.float()]))[0, 1].item())
            top_j = set(torch.topk(pair_scores[kj], k_top).indices.tolist())
            topk_overlap[i][j] = len(top_i & top_j) / k_top

    # ---- heuristic correlations (CPU; exported scores are CPU) ---------------
    src_c, dst_c = src.cpu(), dst.cpu()
    f_block_c = f_block.cpu()
    corr_rows = []
    for a in range(3):
        for b in range(3):
            s = exported[f"pair_{a}{b}"]["scores"].float()
            cos_ab = torch.nn.functional.cosine_similarity(
                f_block_c[dst_c, b], f_block_c[src_c, a], dim=-1)
            src_deg = torch.bincount(src_c, minlength=num_nodes).float()[src_c]
            dst_deg = torch.bincount(dst_c, minlength=num_nodes).float()[dst_c]
            corr_rows.append({
                "dataset": dataset, "seed": seed, "pair": f"pair_{a}{b}",
                "corr_cos": float(torch.corrcoef(torch.stack([s, cos_ab]))[0, 1].item()),
                "corr_src_deg": float(torch.corrcoef(torch.stack([s, src_deg]))[0, 1].item()),
                "corr_dst_deg": float(torch.corrcoef(torch.stack([s, dst_deg]))[0, 1].item()),
            })

    # ---- train-label-only homophily (train->train edges only) --------------
    train_set = set(int(i) for i in setup.data.train_idx.tolist())
    y = setup.data.y.to(device)
    # edge-correspondence-preserving train-train mask (NOT per-endpoint filters)
    mask = torch.tensor([int(v) in train_set for v in src.tolist()]) \
        & torch.tensor([int(v) in train_set for v in dst.tolist()])
    src_tt = torch.tensor([int(v) for v in src.tolist()])[mask]
    dst_tt = torch.tensor([int(v) for v in dst.tolist()])[mask]
    same_tt = (y[src_tt.to(device)] == y[dst_tt.to(device)]).cpu()
    homo_rows = []
    for a in range(3):
        for b in range(3):
            s_all = exported[f"pair_{a}{b}"]["scores"].cpu().float()
            alpha_all = exported[f"pair_{a}{b}"]["alpha"].cpu().float()
            s_tt = s_all[mask]
            a_tt = alpha_all[mask]
            homo_rows.append({
                "dataset": dataset, "seed": seed, "pair": f"pair_{a}{b}",
                "same_label_mean_score": float(s_tt[same_tt].mean().item())
                if same_tt.any() else None,
                "diff_label_mean_score": float(s_tt[~same_tt].mean().item())
                if (~same_tt).any() else None,
                "same_label_mean_alpha": float(a_tt[same_tt].mean().item())
                if same_tt.any() else None,
                "diff_label_mean_alpha": float(a_tt[~same_tt].mean().item())
                if (~same_tt).any() else None,
            })

    # ---- causal ranking + permutation controls ------------------------------
    causal = causal_metrics(model, head, x, ei, setup.data, device,
                            causal_keys=CAUSAL_KEYS)

    summary = {
        "dataset": dataset, "seed": seed,
        "pair_stats": pair_rows,
        "pair_diversity": {"keys": keys, "jsd": jsd, "spearman": spearman,
                           "topk_overlap": topk_overlap, "k_top": k_top},
        "heuristic_corr": corr_rows,
        "homophily": homo_rows,
        "causal": causal,
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] {dataset} s{seed} edge audit done", flush=True)


def _run_one(dataset, seed, gpu, force):
    outdir = EDGE_AUDIT_ROOT / dataset / "PAIR_EDGE" / f"seed_{seed}"
    tag = f"[{gpu}] {dataset} seed={seed}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--dataset", dataset, "--seed", str(seed),
        "--outdir", str(outdir),
    ]
    if force:
        cmd += ["--force"]
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "run.log"
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"{tag} FAILED rc={proc.returncode}", flush=True)
        print(log.read_text(encoding="utf-8")[-3000:], flush=True)
        return
    print(f"{tag} OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-D2.7-B edge-utility audit")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.dataset, args.seed, Path(args.outdir), args.force)
        return

    datasets = DATASETS if not args.datasets else [d for d in args.datasets.split(",")]
    seeds = SEEDS if not args.seeds else [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = [(d, s) for d in datasets for s in seeds]
    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out=outputs/perf_r2d27/edge_audit",
          flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (d, s) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(_run_one, d, s, gpu, args.force)] = (d, s)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


if __name__ == "__main__":
    main()
