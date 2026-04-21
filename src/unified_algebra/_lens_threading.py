"""Private: residual-threading primitives for lenses with residual_sort.

When a lens has residual_sort set, lens_path() delegates to _lens_path_threaded()
here. The forward primitive chains equations collecting residuals; the backward
primitive applies equations in reverse injecting residuals.

This module is not part of the public API — it is imported only by composition.py
and _assembly.py.
"""

from __future__ import annotations

import hydra.core as core
import hydra.graph
from hydra.dsl import prims
from hydra.dsl.meta.phantoms import var, list_
from .utils import bind_composition


def _lens_fwd_primitive() -> hydra.graph.Primitive:
    """Create the ua.prim.lens_fwd higher-order Primitive.

    Signature: list<(a→pair(a,r))> → a → pair(a, list<r>)

    Chains a sequence of forward functions, each producing (output, residual).
    Returns (final_output, [r1, r2, ..., rn]).
    """
    from hydra.sources.libraries import fun
    from hydra.core import TermPair

    prim_name = core.Name("ua.prim.lens_fwd")
    a = prims.variable("a")
    r = prims.variable("r")
    _a = prims.v("a")
    _r = prims.v("r")

    def compute(fwd_fns_list, state):
        residuals = []
        current = state
        for fn in fwd_fns_list:
            result = fn(current)
            if isinstance(result, tuple) and len(result) == 2:
                current, residual = result
            elif isinstance(result, TermPair):
                current, residual = result.value
            else:
                raise TypeError(
                    f"lens_fwd: forward equation did not return a pair, "
                    f"got {type(result).__name__}: {result!r}"
                )
            residuals.append(residual)
        return (current, tuple(residuals))

    return prims.prim2(
        prim_name, compute, [_a, _r],
        prims.list_(fun(a, prims.pair(a, r))),
        a,
        prims.pair(a, prims.list_(r)),
    )


def _lens_bwd_primitive() -> hydra.graph.Primitive:
    """Create the ua.prim.lens_bwd higher-order Primitive.

    Signature: list<(pair(a,r)→a)> → pair(a, list<r>) → a

    Applies backward functions in reverse order, each consuming (feedback, residual).
    """
    from hydra.sources.libraries import fun
    from hydra.core import TermList

    prim_name = core.Name("ua.prim.lens_bwd")
    a = prims.variable("a")
    r = prims.variable("r")
    _a = prims.v("a")
    _r = prims.v("r")

    def compute(bwd_fns_list, feedback_and_residuals):
        feedback_term, residuals_term = feedback_and_residuals
        if isinstance(residuals_term, TermList):
            residuals = list(residuals_term.value)
        elif isinstance(residuals_term, (list, tuple)):
            residuals = list(residuals_term)
        else:
            residuals = [residuals_term]
        current_feedback = feedback_term
        for fn, residual in zip(reversed(bwd_fns_list), reversed(residuals)):
            current_feedback = fn((current_feedback, residual))
        return current_feedback

    return prims.prim2(
        prim_name, compute, [_a, _r],
        prims.list_(fun(prims.pair(a, r), a)),
        prims.pair(a, prims.list_(r)),
        a,
    )


def _lens_path_threaded(
    name: str,
    fwd_eq_names: list[str],
    bwd_eq_names: list[str],
) -> tuple[tuple[core.Name, core.Term], tuple[core.Name, core.Term]]:
    """Build a lens path with residual threading (height-2).

    Forward collects residuals at each step; backward injects them in reverse.
    """
    fwd_list = list_([var(f"ua.equation.{n}") for n in fwd_eq_names])
    fwd_body = var("ua.prim.lens_fwd") @ fwd_list @ var("x")
    bwd_list = list_([var(f"ua.equation.{n}") for n in bwd_eq_names])
    bwd_body = var("ua.prim.lens_bwd") @ bwd_list @ var("p")

    fwd_pair = bind_composition("path", f"{name}.fwd", "x", fwd_body)
    bwd_pair = bind_composition("path", f"{name}.bwd", "p", bwd_body)
    return fwd_pair, bwd_pair
