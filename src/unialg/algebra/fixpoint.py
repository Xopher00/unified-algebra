"""Fixpoint iteration as a Hydra lambda term.

    fixpoint() -> Hydra lambda: λstate. fixpoint(step, pred, state)

The corresponding Primitive (ua.prim.fixpoint) is in assembly/primitives.py.
Output is a pair: (final_state, iteration_count).
"""

from __future__ import annotations

import hydra.core as core
from hydra.dsl.meta.phantoms import var
from unialg.utils import bind_composition


def fixpoint(
    name: str,
    step_name: str,
    predicate_name: str,
) -> tuple[core.Name, core.Term]:
    """Build a fixpoint iteration as a Hydra lambda term.

    Args:
        name:            identifier (e.g. "converge")
        step_name:       name of a 1-input equation: step(state) → new_state
        predicate_name:  name of a 1-input equation: pred(state) → float32 residual

    Returns:
        (Name("ua.fixpoint.<name>"), lambda_term)
        The lambda term is: λstate. fixpoint(step, pred, state)
        Output is a pair: (final_state, iteration_count)
    """
    step_fn = var(f"ua.equation.{step_name}")
    pred_fn = var(f"ua.equation.{predicate_name}")
    body = var("ua.prim.fixpoint") @ step_fn @ pred_fn @ var("state")
    return bind_composition("fixpoint", name, "state", body)
