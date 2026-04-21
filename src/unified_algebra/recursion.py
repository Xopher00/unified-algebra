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
import hydra.graph
from hydra.dsl import prims
from hydra.dsl.meta.phantoms import var, int32, TTerm
from hydra.sources.libraries import fun as _fun
from .utils import bind_composition


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


# ---------------------------------------------------------------------------
# Unfold (anamorphism) — custom higher-order Primitive
# ---------------------------------------------------------------------------

def _unfold_n_compute(step, n, init):
    outputs = []
    state = init
    for _ in range(n):
        state = step(state)
        outputs.append(state)
    return tuple(outputs)

_a_var = prims.variable("a")
_unfold_n_primitive = prims.prim3(
    core.Name("ua.prim.unfold_n"), _unfold_n_compute, [prims.v("a")],
    _fun(_a_var, _a_var),
    prims.int32(),
    _a_var,
    prims.list_(_a_var),
)
del _a_var


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




# ---------------------------------------------------------------------------
# Fixpoint iteration — custom higher-order Primitive
# ---------------------------------------------------------------------------

def _fixpoint_primitive(epsilon: float, max_iter: int) -> hydra.graph.Primitive:
    """Create the ua.prim.fixpoint higher-order Primitive.

    Signature: (state → state) → (state → float32) → state → pair(state, int32)

    Iterates a step function until the predicate returns a value <= epsilon,
    or max_iter is reached. Returns (final_state, iteration_count).
    Epsilon and max_iter are baked into the closure at resolution time.
    """
    from hydra.sources.libraries import fun

    prim_name = core.Name("ua.prim.fixpoint")
    a = prims.variable("a")
    _a = prims.v("a")

    def compute(step, pred, init):
        state = init
        for i in range(max_iter):
            residual = pred(state)
            if residual <= epsilon:
                return (state, i)
            state = step(state)
        return (state, max_iter)

    return prims.prim3(
        prim_name, compute, [_a],
        fun(a, a),                          # step: a → a
        fun(a, prims.float32()),            # pred: a → float32
        a,                                  # init state
        prims.pair(a, prims.int32()),       # output: pair(final_state, iterations)
    )


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



