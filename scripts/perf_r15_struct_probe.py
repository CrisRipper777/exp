"""R15-D2: structural observation headroom probe (plan §R15-D2).

Current structural input (P1 M2) is S3 = [log d, P log d, P^2 log d].
Construct the extended topology-only observation

    S+ = [log d, P log d, P^2 log d, P^3 log d,
          mean_N(d), std_N(d), mean_N(log d), std_N(log d)]

(N = neighbor aggregates over the message direction), then fixed
StandardScaler + Ridge(alpha=1.0) probes (train fit / val eval) on M/T/G:

    Probe(z_final), Probe([z_final|S3]), Probe([z_final|S+])
    Delta_struct_headroom = Probe([z_final|S+]) - Probe([z_final|S3])

>= +0.3~0.5pp on >=2/3 weak datasets => strong R2 evidence that structural
observation bandwidth is the binding constraint. Val only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize  # noqa: E402

from src.data import load_mag_data  # noqa: E402
from src.models.biaxis_final import Model  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "perf_r15" / "audit"
ANCHOR_ROOT = PROJECT_ROOT / "outputs" / "perf_r15" / "anchor"
WEAK = ["Movies", "Toys", "Grocery"]
_EPS = 1e-8


def _resolve_cfg(dataset: str) -> object:
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=biaxis_final", "seed=42",
        ])


def _z_final(dataset: str, device: torch.device):
    cfg = _resolve_cfg(dataset)
    data = load_mag_data(cfg, "nc", 42)
    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
    }
    ckpt = torch.load(ANCHOR_ROOT / dataset / "A0" / "seed_42" / "model.pt",
                      map_location="cpu", weights_only=False)
    model = Model(cfg, info)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    with torch.no_grad():
        z = model.forward(x, ei)[0]
    return data, z.detach().cpu().numpy()


def _struct_obs(edge_index: torch.Tensor, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    src, dst = edge_index[0], edge_index[1]
    deg = torch.bincount(dst, minlength=num_nodes).to(torch.float32)
    logd = torch.log1p(deg)

    def _p(v):
        acc = torch.zeros_like(v)
        acc.index_add_(0, dst, v[src])
        return acc / (deg + _EPS)

    p_logd = _p(logd)
    p2_logd = _p(p_logd)
    p3_logd = _p(p2_logd)

    def _nb_agg(v):
        acc = torch.zeros_like(v)
        acc.index_add_(0, dst, v[src])
        return acc / (deg + _EPS)

    mean_d = _nb_agg(deg)
    mean_d2 = _nb_agg(deg * deg)
    std_d = (mean_d2 - mean_d * mean_d).clamp_min(0.0).sqrt()
    mean_logd = _nb_agg(logd)
    mean_logd2 = _nb_agg(logd * logd)
    std_logd = (mean_logd2 - mean_logd * mean_logd).clamp_min(0.0).sqrt()

    s3 = torch.stack([logd, p_logd, p2_logd], dim=1)
    sp = torch.stack([logd, p_logd, p2_logd, p3_logd,
                      mean_d, std_d, mean_logd, std_logd], dim=1)
    return s3.numpy(), sp.numpy()


def _probe(feat: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray) -> dict:
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(feat[tr])
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(x_tr, y[tr])
    x_va = scaler.transform(feat[va])
    pred = clf.predict(x_va)
    return {"val_acc": float(accuracy_score(y[va], pred)),
            "val_macro_f1": float(f1_score(y[va], pred, average="macro"))}


def main() -> None:
    parser = argparse.ArgumentParser(description="R15-D2 structural headroom probe")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for di, dataset in enumerate(WEAK):
        device = torch.device(f"cuda:{gpus[di % len(gpus)]}")
        data, z = _z_final(dataset, device)
        s3, sp = _struct_obs(data.edge_index, int(data.num_nodes))
        y = data.y.numpy()
        tr = data.train_idx.numpy()
        va = data.val_idx.numpy()
        variants = {
            "z": z,
            "z_S3": np.concatenate([z, s3], axis=1),
            "z_Splus": np.concatenate([z, sp], axis=1),
        }
        vals = {}
        for name, feat in variants.items():
            p = _probe(feat, y, tr, va)
            vals[name] = p
            print(f"[D2] {dataset:12s} {name:8s} val={p['val_acc']:.4f}", flush=True)
        rows.append({
            "dataset": dataset,
            "probe_z": vals["z"]["val_acc"],
            "probe_z_S3": vals["z_S3"]["val_acc"],
            "probe_z_Splus": vals["z_Splus"]["val_acc"],
            "delta_struct_headroom_pp": 100.0 * (vals["z_Splus"]["val_acc"] - vals["z_S3"]["val_acc"]),
        })
    path = OUT_DIR / "struct_headroom_probe.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# R15-D2 STRUCTURAL OBSERVATION HEADROOM", ""]
    lines.append("> 固定 StandardScaler + Ridge(α=1)，train fit / val eval；S+=S3+P³logd+邻居度统计。")
    lines.append("")
    lines.append("| dataset | Probe(z) | Probe(z\\|S3) | Probe(z\\|S+) | Δ_headroom pp |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['probe_z']:.4f} | {r['probe_z_S3']:.4f} | "
                     f"{r['probe_z_Splus']:.4f} | {r['delta_struct_headroom_pp']:+.2f} |")
    lines.append("")
    n_ge = sum(1 for r in rows if r["delta_struct_headroom_pp"] >= 0.3)
    verdict = "structural observation bandwidth 不足的强证据（R2 方向）" if n_ge >= 2 else \
        "无结构性 headroom 证据"
    lines.append(f"## 结论：≥+0.3pp 的弱项数据集 {n_ge}/3 → **{verdict}**")
    lines.append("")
    (OUT_DIR / "STRUCT_HEADROOM_PROBE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[D2] saved -> {OUT_DIR / 'STRUCT_HEADROOM_PROBE.md'}")


if __name__ == "__main__":
    main()
