"""Fold (catamorphism) and unfold (anamorphism) as Hydra lambda terms.

    fold()   -> Hydra lambda: λseq. foldl(step, init, seq)
    unfold() -> Hydra lambda: λstate. unfold_n(step, n, state)

Fold uses Hydra's built-in `hydra.lib.lists.foldl` primitive — no custom
iteration logic needed. The step function is resolved through normal Hydra
reduction (var lookup → primitive/bound_term → beta-reduce).

Unfold uses a custom higher-order primitive `ua.prim.unfold_n` since Hydra
has no built-in unfold/iterate. It follows the same pattern as Hydra's own
higher-order primitives (foldl, map, etc.) — calling reduce_term internally.

Both are completely backend-agnostic. Weight tying is automatic: the same
step function term is applied at every recursive step.
"""

from __future__ import annotations

import hydra.core as core
from hydra.dsl.meta.phantoms import var, int32, TTerm
from unialg.utils import bind_composition


# ---------------------------------------------------------------------------
# Fold (catamorphism) — uses Hydra's built-in foldl
# ---------------------------------------------------------------------------

def fold(
    name: str,
    step_name: str,
    init_term: core.Term,
) -> tuple[core.Name, core.Term]:
    """Build a fold as a Hydra lambda term.

    Args:
        name:       identifier (e.g. "rnn")
        step_name:  name of a 2-input equation/path: step(state, element) → new_state
        init_term:  pre-encoded Hydra tensor term for initial state

    Returns:
        (Name("ua.fold.<name>"), lambda_term)
        The lambda term is: λseq. foldl(step, init, seq)
    """
    step_fn = var(f"ua.equation.{step_name}")
    body = var("hydra.lib.lists.foldl") @ step_fn @ TTerm(init_term) @ var("seq")
    return bind_composition("fold", name, "seq", body)


def unfold(
    name: str,
    step_name: str,
    n_steps: int,
) -> tuple[core.Name, core.Term]:
    """Build an unfold as a Hydra lambda term.

    Args:
        name:       identifier (e.g. "stream")
        step_name:  name of a 1-input equation/path: step(state) → new_state
        n_steps:    number of unfolding iterations

    Returns:
        (Name("ua.unfold.<name>"), lambda_term)
        The lambda term is: λstate. unfold_n(step, n, state)
    """
    step_fn = var(f"ua.equation.{step_name}")
    body = var("ua.prim.unfold_n") @ step_fn @ int32(n_steps) @ var("state")
    return bind_composition("unfold", name, "state", body)
