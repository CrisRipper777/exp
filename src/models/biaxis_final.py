"""Final frozen Bi-Axis model (review §18).

Thin alias of biaxis_p3: the frozen structure is pinned by
configs/model/biaxis_final.yaml (p2.mode=null_softmax,
p3.operator_mode=full_interaction — the validation-selected Full
Cell-conditioned Operator, review §15 terminology), while the model class
inherits all P3 checks (NullSoftmax + deterministic=false hard asserts).

Use `model=biaxis_final` for every future baseline comparison, LP run and
paper result table; ablation variants stay in biaxis_p3.yaml.
"""

from .biaxis_p3 import Model

__all__ = ["Model"]
