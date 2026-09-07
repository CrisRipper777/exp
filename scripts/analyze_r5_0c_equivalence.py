"""R5-0c analytical and numerical equivalence audit for FactorGlobalRelay.

This is a CPU-only audit.  It does not train, load checkpoints, or touch Test
data.  The output root is new and intentionally separate from R5-0b.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs" / "r5_0c_equivalence"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.biaxis_scope_v2_components import FactorGlobalRelay


def _copy_corresponding_weights(source: FactorGlobalRelay, target: FactorGlobalRelay) -> None:
    for group in ("value", "slot_update", "readout"):
        for source_module, target_module in zip(getattr(source, group), getattr(target, group)):
            target_module.load_state_dict(source_module.state_dict())


def _state_initialization_comparison(seed: int = 600) -> list[dict]:
    torch.manual_seed(seed)
    s1 = FactorGlobalRelay(8, 1, "gelu", assignment_mode="learned")
    torch.manual_seed(seed)
    s4 = FactorGlobalRelay(8, 4, "gelu", assignment_mode="uniform")
    state1 = s1.state_dict()
    state4 = s4.state_dict()
    rows = []
    for name in sorted(set(state1) | set(state4)):
        left = state1.get(name)
        right = state4.get(name)
        row = {
            "parameter": name,
            "s1_shape": list(left.shape) if left is not None else "",
            "s4_shape": list(right.shape) if right is not None else "",
            "comparable": bool(left is not None and right is not None and left.shape == right.shape),
            "max_abs_diff": "",
            "elementwise_equal": "",
            "note": "",
        }
        if left is None or right is None:
            row["note"] = "parameter only present in one state dict"
        elif left.shape != right.shape:
            row["note"] = "assign layer shape differs because num_slots differs"
        else:
            diff = (left - right).abs()
            row["max_abs_diff"] = float(diff.max().item())
            row["elementwise_equal"] = bool(torch.equal(left, right))
            row["note"] = "same-shape parameter; S4 assign initialization consumed extra RNG before it"
        rows.append(row)
    return rows


def _forward_and_gradient_audit() -> dict:
    torch.manual_seed(601)
    s1 = FactorGlobalRelay(8, 1, "gelu", assignment_mode="learned")
    s4 = FactorGlobalRelay(8, 4, "gelu", assignment_mode="uniform")
    _copy_corresponding_weights(s1, s4)
    h1 = torch.randn(23, 8, requires_grad=True)
    h4 = h1.detach().clone().requires_grad_(True)
    outputs1 = [s1(h1, factor) for factor in range(3)]
    outputs4 = [s4(h4, factor) for factor in range(3)]
    max_forward = max(float((left - right).abs().max().item()) for left, right in zip(outputs1, outputs4))
    loss1 = sum(output.square().mean() for output in outputs1)
    loss4 = sum(output.square().mean() for output in outputs4)
    loss1.backward()
    loss4.backward()
    max_input_grad = float((h1.grad - h4.grad).abs().max().item())
    grad_diffs = {}
    for name, parameter in s1.named_parameters():
        if name.startswith("assign."):
            continue
        counterpart = dict(s4.named_parameters())[name]
        grad_diffs[name] = float((parameter.grad - counterpart.grad).abs().max().item())

    torch.manual_seed(602)
    one_slot = FactorGlobalRelay(8, 1, "gelu", assignment_mode="learned")
    h = torch.randn(23, 8)
    before = one_slot(h, 0)
    with torch.no_grad():
        one_slot.assign[0].weight.normal_(std=100.0)
        one_slot.assign[0].bias.normal_(std=100.0)
    after = one_slot(h, 0)

    torch.manual_seed(603)
    perm_relay = FactorGlobalRelay(8, 4, "gelu", assignment_mode="learned")
    h_perm = torch.randn(23, 8)
    permutation = torch.tensor([2, 0, 3, 1])
    original_assignment = perm_relay._assignment
    with patch.object(
        perm_relay,
        "_assignment",
        side_effect=lambda values, factor: original_assignment(values, factor)[:, permutation.to(values.device)],
    ):
        permuted = perm_relay(h_perm, 0)
    normal = perm_relay(h_perm, 0)

    return {
        "uniform_s4_vs_learned_s1": {
            "max_forward_abs_diff": max_forward,
            "bitwise_equal": max_forward == 0.0,
            "max_input_gradient_abs_diff": max_input_grad,
            "parameter_gradient_max_abs_diff": max(grad_diffs.values()),
            "parameter_gradient_diffs": grad_diffs,
        },
        "s1_learned_assignment_no_effect": {
            "max_output_abs_diff_after_assign_perturbation": float((before - after).abs().max().item()),
            "bitwise_equal": torch.equal(before, after),
        },
        "slot_permutation_invariance": {
            "permutation": permutation.tolist(),
            "max_output_abs_diff": float((normal - permuted).abs().max().item()),
            "allclose_at_1e-6": bool(torch.allclose(normal, permuted, rtol=1e-6, atol=1e-7)),
        },
    }


def main() -> None:
    parameter_rows = _state_initialization_comparison()
    audit = _forward_and_gradient_audit()
    common = [row for row in parameter_rows if row["comparable"]]
    unequal_common = [row for row in common if not row["elementwise_equal"]]
    audit["rng_initialization_effect"] = {
        "same_seed": 600,
        "same_shape_parameter_count": len(common),
        "same_shape_unequal_parameter_count": len(unequal_common),
        "same_shape_unequal_parameters": [row["parameter"] for row in unequal_common],
        "interpretation": "S4 assign[d->4] consumes three extra Linear initializations versus S1 assign[d->1], so later branch parameters differ under the same global seed.",
    }
    summary_dir = OUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "R5_0C_INIT_COMPARISON.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["parameter", "s1_shape", "s4_shape", "comparable", "max_abs_diff", "elementwise_equal", "note"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(parameter_rows)
    (summary_dir / "R5_0C_EQUIVALENCE.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
