"""Custom Hydra Primitives the DSL registers beyond Hydra's stdlib.

All higher-order primitives live here:
  unfold_n_primitive, fixpoint_primitive, lens_fwd_primitive, lens_bwd_primitive
"""

from __future__ import annotations

import hydra.core as core
import hydra.graph
from hydra.dsl import prims
from hydra.sources.libraries import fun as _fun


# ---------------------------------------------------------------------------
# Unfold primitive (anamorphism)
# ---------------------------------------------------------------------------

def _unfold_n_compute(step, n, init):
    outputs = []
    state = init
    for _ in range(n):
        state = step(state)
        outputs.append(state)
    return tuple(outputs)

_a_var = prims.variable("a")
unfold_n_primitive = prims.prim3(
    core.Name("ua.prim.unfold_n"), _unfold_n_compute, [prims.v("a")],
    _fun(_a_var, _a_var),
    prims.int32(),
    _a_var,
    prims.list_(_a_var),
)
del _a_var


# ---------------------------------------------------------------------------
# Fixpoint primitive (convergence iteration)
# ---------------------------------------------------------------------------

def fixpoint_primitive(epsilon: float, max_iter: int) -> hydra.graph.Primitive:
    """Create the ua.prim.fixpoint higher-order Primitive.

    Signature: (state → state) → (state → float32) → state → pair(state, int32)

    Iterates a step function until the predicate returns a value <= epsilon,
    or max_iter is reached. Returns (final_state, iteration_count).
    Epsilon and max_iter are baked into the closure at resolution time.
    """
    from hydra.sources.libraries import fun

    prim_name = core.Name(f"ua.prim.fixpoint.{epsilon}.{max_iter}")
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
        fun(a, a),
        fun(a, prims.float32()),
        a,
        prims.pair(a, prims.int32()),
    )


# ---------------------------------------------------------------------------
# Lens threading primitives (height-2 optics)
# ---------------------------------------------------------------------------

def _lens_fwd_compute(fwd_fns_list, state):
    residuals = []
    current = state
    for fn in fwd_fns_list:
        result = fn(current)
        if isinstance(result, tuple) and len(result) == 2:
            current, residual = result
        elif isinstance(result, core.TermPair):
            current, residual = result.value
        else:
            raise TypeError(
                f"lens_fwd: forward equation did not return a pair, "
                f"got {type(result).__name__}: {result!r}"
            )
        residuals.append(residual)
    return (current, tuple(residuals))


def _lens_bwd_compute(bwd_fns_list, feedback_and_residuals):
    feedback_term, residuals_term = feedback_and_residuals
    if isinstance(residuals_term, core.TermList):
        residuals = list(residuals_term.value)
    elif isinstance(residuals_term, (list, tuple)):
        residuals = list(residuals_term)
    else:
        residuals = [residuals_term]
    current_feedback = feedback_term
    for fn, residual in zip(reversed(bwd_fns_list), reversed(residuals)):
        current_feedback = fn((current_feedback, residual))
    return current_feedback


_a_var2 = prims.variable("a")
_r_var = prims.variable("r")

lens_fwd_primitive = prims.prim2(
    core.Name("ua.prim.lens_fwd"), _lens_fwd_compute, [prims.v("a"), prims.v("r")],
    prims.list_(_fun(_a_var2, prims.pair(_a_var2, _r_var))),
    _a_var2,
    prims.pair(_a_var2, prims.list_(_r_var)),
)

lens_bwd_primitive = prims.prim2(
    core.Name("ua.prim.lens_bwd"), _lens_bwd_compute, [prims.v("a"), prims.v("r")],
    prims.list_(_fun(prims.pair(_a_var2, _r_var), _a_var2)),
    prims.pair(_a_var2, prims.list_(_r_var)),
    _a_var2,
)

del _a_var2, _r_var
