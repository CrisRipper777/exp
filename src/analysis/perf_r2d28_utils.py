"""R2-Design-2.8 v2 shared analysis layer
(docs/BiAxis_R2_Design_2_8_v2_Identifiable_Relational_Function_Decomposition.md).

Discipline (v2):
    - A0 is the ONLY primary parent; fully frozen in D2.8-B..E
      (controlled parent adaptation allowed only in D2.8-G).
    - Unified formula with identifiability controls: exposure / real-neighbor
      composition / simplex channel / NormMatch operator; staged freezing
      (Rule V) — when a new function is tested, previously selected
      functions are loaded from the previous stage's checkpoint and frozen.
    - r / pi / lambda / operator routing are shared-predictor outputs, never
      free node/edge tables (v2 §2).
    - No auxiliary / edge-label supervision. Val only — NEVER test.
    - Classifier init: reuse the D2.7 per-(dataset, seed) shared head-init
      files so D2.7/D2.8 variants share identical classifier initialization.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d27_utils import (  # noqa: E402
    DATASETS,
    GUARD_DATASETS,
    R2D27_ROOT,
    SEEDS,
    TARGET_DATASETS,
    UtilitySetup,
    assert_no_test_access,
    causal_metrics,
    load_a0_parent,
    load_or_make_head_init,
    scheduled_lr,
)

R2D28_ROOT = PROJECT_ROOT / "outputs" / "perf_r2d28"
HEAD_INIT_ROOT = R2D27_ROOT / "head_init"  # shared classifier init with D2.7

# stage output roots (v2 plan §16 completion package)
AUDIT_ROOT = R2D28_ROOT / "audit"
REPAIR_ROOT = R2D28_ROOT / "repair"
EXPOSURE_ROOT = R2D28_ROOT / "exposure"
COMPOSITION_ROOT = R2D28_ROOT / "composition"
CHANNEL_ROOT = R2D28_ROOT / "channel"
OPERATOR_ROOT = R2D28_ROOT / "operator"
FACTORIAL_ROOT = R2D28_ROOT / "factorial"
CONFIRM_ROOT = R2D28_ROOT / "confirm"
SUMMARY_ROOT = R2D28_ROOT / "summary"

CLASSIFIER_SEED = 20260904

# variant label -> model config overrides (stages fill exposure= with E*)
EXPOSURE_VARIANTS = {
    "E0": dict(exposure="fixed_full"),
    "E1": dict(exposure="node"),
    "E2": dict(exposure="target"),
    "E3": dict(exposure="source"),
    "E4": dict(exposure="pair"),
}

COMPOSITION_VARIANTS = {
    "C0": dict(composition="uniform"),
    "C1": dict(composition="generic"),
    "C2": dict(composition="target"),
    "C3": dict(composition="source"),
    "C4": dict(composition="pair"),
}

CHANNEL_VARIANTS = {
    "M0": dict(channel="mean"),
    "M1": dict(channel="softmax"),
    "M2": dict(channel="concat"),
    "M2_MEAN_DUP": dict(channel="concat", mean_dup=True),
    "M3": dict(channel="attn"),
    "M3_MEAN_DUP": dict(channel="attn", mean_dup=True),
}

OPERATOR_VARIANTS = {
    "O0": dict(operator="linear"),
    "O1": dict(operator="static_pair"),
    "O2": dict(operator="target_film"),
    "O3": dict(operator="edge_film"),
    "O4": dict(operator="basis"),
    "O4_UNIFORM": dict(operator="basis", uniform_router=True),
    "O4_TARGET": dict(operator="basis", target_router=True),
}

# D2.7 reference checkpoints for D2.8-A (no retraining)
D27_CKPT_ROOTS = {
    "PAIR_EDGE": R2D27_ROOT / "matrix",
    "TARGET_FACTOR_ONLY": R2D27_ROOT / "ownership",
}

REPAIR_CAUSAL_KEYS = (
    "full",
    "within_target_shuffle_fixed",
    "remove_top_per_target_10", "remove_top_per_target_25", "remove_top_per_target_50",
    "remove_random_per_target_10", "remove_random_per_target_25", "remove_random_per_target_50",
    "remove_bottom_per_target_10", "remove_bottom_per_target_25", "remove_bottom_per_target_50",
    "keep_top_per_target_25", "keep_top_per_target_50",
)


def resolve_cfg(dataset: str, seed: int, overrides: dict) -> object:
    from hydra import compose, initialize_config_dir

    ov = [f"dataset={dataset}", "task=nc", "model=biaxis_r2_relfunc",
          f"seed={int(seed)}"]
    for k, v in overrides.items():
        if isinstance(v, bool):
            ov.append(f"model.{k}={'true' if v else 'false'}")
        else:
            ov.append(f"model.{k}={v}")
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"),
                               version_base=None):
        return compose(config_name="config", overrides=ov)


def build_model(cfg: object, data, parent: nn.Module, device: torch.device) -> nn.Module:
    from src.models.biaxis_r2_relfunc import Model

    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
    }
    return Model(cfg, info, parent).to(device)


def load_frozen_components(model: nn.Module, ckpt_path: Path,
                           prefixes: list[str]) -> dict:
    """Rule V staged loading: copy only the named state-dict prefixes from the
    previous stage's checkpoint into the current model (fresh init elsewhere).
    Returns the number of copied parameters."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    model_state = model.state_dict()
    copied = 0
    for prefix in prefixes:
        matched = {k: v for k, v in state.items() if k.startswith(prefix)}
        if not matched:
            # NO-GO slot: neither the checkpoint nor the model has modules
            # for this prefix (e.g. E0 fixed_full has no exposure_net)
            assert not any(k.startswith(prefix) for k in model_state), \
                f"model has {prefix!r} keys but {ckpt_path} does not"
            continue
        for k, v in matched.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                copied += v.numel()
    model.load_state_dict(model_state)
    model._apply_freezes()
    return {"copied_params": int(copied)}


def exposure_prefixes() -> list[str]:
    return ["exposure_net.", "exposure_emb.", "payload."]


def composition_prefixes() -> list[str]:
    return ["comp_net.", "comp_local_proj.", "comp_emb."]


def channel_prefixes() -> list[str]:
    return ["channel_net.", "channel_emb."]


def operator_prefixes() -> list[str]:
    return ["operator_net.", "operator_emb."]


def train_relfunc_model(
    data, model: nn.Module, head: nn.Module, device: torch.device,
    *, total_epochs: int = 300, patience: int = 30,
    history_callback=None,
) -> dict:
    """A0 fully frozen; TRAINABLE side components + classifier lr 1e-3
    wd 1e-4, warmup10+cosine, grad clip 1.0, best Val Acc (v2 §7 protocol).
    Frozen staged components are excluded from the optimizer (Rule V)."""
    assert_no_test_access(data)
    model = model.to(device)
    head = head.to(device)
    model.parent.eval()
    model.parent_frozen = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    train_idx = data.train_idx.to(device)
    y_train = data.y[data.train_idx].to(device)
    val_idx = data.val_idx.to(device)
    y_val = data.y[data.val_idx].to(device)
    criterion = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(trainable + list(head.parameters()),
                            lr=1e-3, weight_decay=1e-4)

    def _apply_lr(epoch: int) -> None:
        for pg in opt.param_groups:
            pg["lr"] = scheduled_lr(epoch, total_epochs, 1e-3)

    history: list[dict] = []
    best_acc, best_epoch, best_state = -1.0, None, None
    patience_left = patience
    stop_epoch = total_epochs
    for epoch in range(1, total_epochs + 1):
        _apply_lr(epoch)
        opt.zero_grad(set_to_none=True)
        model.train()
        z, _, _, _, _ = model(x, ei)
        loss = criterion(head(z[train_idx]), y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable + list(head.parameters()), 1.0)
        opt.step()
        with torch.no_grad():
            model.eval()
            z_eval, _, _, _, _ = model(x, ei)
            pred_v = head(z_eval[val_idx]).argmax(-1)
            acc = float((pred_v == y_val).float().mean().item())
            del z_eval
        row = {"epoch": epoch, "lr": float(scheduled_lr(epoch, total_epochs, 1e-3)),
               "train_ce": float(loss.item()), "val_acc": acc}
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                stop_epoch = epoch
                break
        if history_callback is not None:
            history_callback(row)
        history.append(row)

    model.load_state_dict(best_state["model"])
    head.load_state_dict(best_state["head"])
    model.eval()
    head.eval()
    with torch.no_grad():
        z_best, _, _, _, _ = model(x, ei)
    from src.analysis.perf_r2d15_utils import val_metrics_with_head

    m_full = val_metrics_with_head(head, z_best, data, device)
    return {
        "best_val_acc": best_acc,
        "best_val_macro_f1": m_full["val_macro_f1"],
        "per_class_f1": m_full["per_class_f1"],
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "history": history,
        "z_best": z_best,
        "trainable_params": int(sum(p.numel() for p in trainable)),
    }


def train_relfunc_parent_adapt(
    data, model: nn.Module, head: nn.Module, device: torch.device,
    *, total_epochs: int = 300, patience: int = 30,
    unfreeze_epoch: int = 31, parent_lr: float = 1e-4,
    history_callback=None,
) -> dict:
    """v2 §13 single permitted controlled parent adaptation: epochs 1..30 the
    parent stays fully frozen; epoch 31+ only the A0 final fusion and the
    graph operator are released (P0 factorizer stays frozen). Parent lr 1e-4,
    new-branch lr 1e-3, parent wd 0 (it was trained long ago; released
    gently). Best Val Acc, warmup10+cosine for the branch."""
    assert_no_test_access(data)
    model = model.to(device)
    head = head.to(device)
    model.parent_frozen = True
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    train_idx = data.train_idx.to(device)
    y_train = data.y[data.train_idx].to(device)
    val_idx = data.val_idx.to(device)
    y_val = data.y[data.val_idx].to(device)
    criterion = nn.CrossEntropyLoss()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": trainable + list(head.parameters()),
         "lr": 1e-3, "weight_decay": 1e-4},
        {"params": [], "lr": parent_lr, "weight_decay": 0.0},  # parent group
    ])

    def _parent_params():
        parent = model.parent
        out = []
        for name in ("fusion", "operator"):
            mod = getattr(parent, name, None)
            if mod is not None:
                out += list(mod.parameters())
        return out

    parent_released = False

    def _apply_lr(epoch: int) -> None:
        opt.param_groups[0]["lr"] = scheduled_lr(epoch, total_epochs, 1e-3)
        if parent_released:
            opt.param_groups[1]["lr"] = parent_lr

    history: list[dict] = []
    best_acc, best_epoch, best_state = -1.0, None, None
    patience_left = patience
    stop_epoch = total_epochs
    for epoch in range(1, total_epochs + 1):
        if epoch == unfreeze_epoch and not parent_released:
            parent_released = True
            model.parent_frozen = False
            for p in _parent_params():
                p.requires_grad_(True)
            opt.param_groups[1]["params"] = _parent_params()
        _apply_lr(epoch)
        opt.zero_grad(set_to_none=True)
        model.train()
        z, _, _, _, _ = model(x, ei)
        loss = criterion(head(z[train_idx]), y_train)
        loss.backward()
        clip_params = trainable + list(head.parameters()) + \
            (opt.param_groups[1]["params"] if parent_released else [])
        nn.utils.clip_grad_norm_(clip_params, 1.0)
        opt.step()
        with torch.no_grad():
            model.eval()
            z_eval, _, _, _, _ = model(x, ei)
            pred_v = head(z_eval[val_idx]).argmax(-1)
            acc = float((pred_v == y_val).float().mean().item())
            del z_eval
        row = {"epoch": epoch, "lr": float(opt.param_groups[0]["lr"]),
               "train_ce": float(loss.item()), "val_acc": acc}
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                stop_epoch = epoch
                break
        if history_callback is not None:
            history_callback(row)
        history.append(row)

    model.load_state_dict(best_state["model"])
    head.load_state_dict(best_state["head"])
    model.eval()
    head.eval()
    with torch.no_grad():
        z_best, _, _, _, _ = model(x, ei)
    from src.analysis.perf_r2d15_utils import val_metrics_with_head

    m_full = val_metrics_with_head(head, z_best, data, device)
    return {
        "best_val_acc": best_acc,
        "best_val_macro_f1": m_full["val_macro_f1"],
        "per_class_f1": m_full["per_class_f1"],
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "history": history,
        "z_best": z_best,
        "trainable_params": int(sum(p.numel() for p in trainable)),
        "unfreeze_epoch": int(unfreeze_epoch),
    }


# ---------------------------------------------------------------------------
# Job orchestration (D2.7 matrix pattern: per-GPU subprocess workers)
# ---------------------------------------------------------------------------


def launch_jobs(stage_script: Path, jobs: list, out_root: Path, gpus: list[int],
                force: bool = False, epochs: int | None = None,
                extra_flags: list[str] | None = None) -> None:
    import os
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def _run_one(dataset, variant, seed, gpu):
        outdir = out_root / dataset / variant / f"seed_{seed}"
        tag = f"[{gpu}] {dataset} {variant} seed={seed}"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        cmd = [
            sys.executable, str(stage_script),
            "--worker", "--dataset", dataset, "--variant", variant,
            "--seed", str(seed), "--outdir", str(outdir), "--out-root", str(out_root),
        ]
        if epochs is not None:
            cmd += ["--epochs", str(int(epochs))]
        if force:
            cmd += ["--force"]
        if extra_flags:
            cmd += extra_flags
        outdir.mkdir(parents=True, exist_ok=True)
        log = outdir / "run.log"
        with log.open("w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                                  stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(f"{tag} FAILED rc={proc.returncode}", flush=True)
            print(log.read_text(encoding="utf-8")[-3000:], flush=True)
            return
        print(f"{tag} OK", flush=True)

    locks = {g: _Semaphore(1) for g in gpus}
    print(f"[driver] jobs={len(jobs)} gpus={gpus} out={out_root}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, (d, v, s) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures[executor.submit(_run_one, d, v, s, gpu)] = (d, v, s)
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"JOB ERROR {job}: {exc}", flush=True)
    print("[driver] done", flush=True)


# ---------------------------------------------------------------------------
# Small reporting helpers shared by stage summarizers
# ---------------------------------------------------------------------------


def load_summaries(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.rglob("summary.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            d["_dir"] = str(p.parent)
            rows.append(d)
        except Exception:  # noqa: BLE001
            continue
    return rows


def paired_delta(rows: list[dict], cand_variant: str, base_variant: str,
                 metric: str, ds_list: list[str] = TARGET_DATASETS,
                 seeds: list[int] = SEEDS) -> dict:
    """Paired (dataset, seed) candidate - base in percentage points."""
    def _by(v):
        return {(r["dataset"], r["seed"]): r for r in rows if r["variant"] == v}

    cand, base = _by(cand_variant), _by(base_variant)
    per_ds = []
    for ds in ds_list:
        deltas = []
        for s in seeds:
            c, b = cand.get((ds, s)), base.get((ds, s))
            if c and b and metric in c and metric in b:
                deltas.append(100.0 * (c[metric] - b[metric]))
        if deltas:
            per_ds.append((ds, math.fsum(deltas) / len(deltas),
                           sum(1 for d in deltas if d > 0), len(deltas)))
    mean = math.fsum(m for _, m, _, _ in per_ds) / len(per_ds) if per_ds else None
    n_pos = sum(1 for _, m, _, _ in per_ds if m > 0)
    return {"per_ds": per_ds, "mean": mean, "n_pos": n_pos}


def mean_std_pp(rows: list[dict], cand_variant: str, base_variant: str,
                metric: str, ds_list: list[str] = TARGET_DATASETS,
                seeds: list[int] = SEEDS) -> str:
    """'mean±std (n_pos/n)' pp string over paired seeds."""
    d = paired_delta(rows, cand_variant, base_variant, metric, ds_list, seeds)
    if d["mean"] is None:
        return "n/a"
    all_deltas = [100.0 * (c[metric] - b[metric])
                  for ds in ds_list for s in seeds
                  for c in [{(r["dataset"], r["seed"]): r
                             for r in rows if r["variant"] == cand_variant}.get((ds, s))]
                  for b in [{(r["dataset"], r["seed"]): r
                             for r in rows if r["variant"] == base_variant}.get((ds, s))]
                  if c and b and metric in c and metric in b]
    n = len(all_deltas)
    mean = math.fsum(all_deltas) / n if n else 0.0
    var = math.fsum((v - mean) ** 2 for v in all_deltas) / n if n else 0.0
    n_pos = sum(1 for v in all_deltas if v > 0)
    return f"{mean:+.3f}±{math.sqrt(var):.3f} ({n_pos}/{n})"
