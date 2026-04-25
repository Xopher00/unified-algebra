"""Equation resolution: compile Equation declarations to Hydra Primitives.

Each resolve function returns (Primitive, Callable) — the Primitive for the
Hydra graph, and the raw compute closure for native (non-Hydra) execution.

When `n_params + n_inputs > 3` (Hydra's prim3 cap), the resolver list-packs
params and/or tensor inputs into single coder slots so the primitive fits.
The compute closure receives packed args and unpacks them internally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from hydra.dsl.prims import prim1, prim2, prim3, float32 as float32_coder, list_ as list_coder
from hydra.graph import Primitive

from unialg.algebra.equation import Equation
from unialg.algebra.contraction import contract_and_apply, contract_merge

if TYPE_CHECKING:
    from unialg.backend import Backend


_PRIMS = {1: prim1, 2: prim2, 3: prim3}


def _make_prim(prim_name, compute, coders, out_coder) -> Primitive:
    n = len(coders)
    if n not in _PRIMS:
        raise ValueError(f"Primitive '{prim_name.value}': packed arity {n} exceeds max 3")
    return _PRIMS[n](prim_name, compute, [], *coders, out_coder)


def _build_resolved(in_coder, n_params: int, n_inputs: int, sr, compiled, backend, nl_fn):
    """Build (coders, hydra_compute, native_fn) for an equation.

    When total arity > 3, params and tensors are list_-packed into single slots
    for the Hydra primitive (to fit prim3). The native_fn stays variadic so
    callers (compositions, Program.__call__, compiled fast-path) keep the
    uniform `fn(*args)` interface across all arities.
    """
    def _core(params, tensors):
        return contract_and_apply(compiled, list(tensors), sr, backend, nl_fn, tuple(params))

    def native_fn(*args):
        return _core(args[:n_params], args[n_params:])

    if n_params + n_inputs <= 3:
        coders = [float32_coder()] * n_params + [in_coder] * n_inputs
        return coders, native_fn, native_fn

    coders = []
    if n_params > 0:
        coders.append(list_coder(float32_coder()))
    if n_inputs > 0:
        coders.append(list_coder(in_coder))

    def hydra_compute(*args):
        i = 0
        params = tuple(args[i]) if n_params > 0 else ()
        if n_params > 0:
            i += 1
        tensors = list(args[i]) if n_inputs > 0 else []
        return _core(params, tensors)

    return coders, hydra_compute, native_fn


def resolve_equation(eq: Equation, backend: "Backend", ctx=None) -> tuple[Primitive, Callable]:
    """Compile an Equation to a Hydra Primitive and a raw compute callable.

    Returns (primitive, native_fn) where native_fn is the bare closure that
    calls contract_and_apply directly — usable without Hydra plumbing.
    """
    has_einsum, has_nl, nl_fn, in_coder, out_coder, prim_name, sr, compiled, n_inputs, n_params = \
        ctx or eq.compile(backend)

    if not has_einsum and not has_nl:
        raise ValueError(f"Equation '{eq.name}' has neither einsum nor nonlinearity")
    if not has_einsum:
        n_inputs = 1

    coders, hydra_compute, native_fn = _build_resolved(
        in_coder, n_params, n_inputs, sr, compiled, backend, nl_fn)
    prim = _make_prim(prim_name, hydra_compute, coders, out_coder)
    return prim, native_fn


def resolve_equation_as_merge(eq: Equation, backend: "Backend", ctx=None) -> tuple[Primitive, Callable]:
    """Compile an Equation as a list-consuming merge Primitive and raw callable."""
    has_einsum, has_nl, nl_fn, in_coder, out_coder, prim_name, sr, compiled, n_inputs, _ = \
        ctx or eq.compile(backend)

    if has_einsum:
        if n_inputs not in (1, 2):
            raise ValueError(
                f"List-merge equation '{eq.name}': einsum must have 1 or 2 inputs, got {n_inputs}")

        def compute_merge(tensors):
            return contract_merge(compiled, tensors, sr, backend, nl_fn, n_inputs, eq.name)

        prim = _make_prim(prim_name, compute_merge, [list_coder(in_coder)], out_coder)
        return prim, compute_merge

    elif has_nl:
        def compute_nl(tensors):
            result = tensors[0]
            for t in tensors[1:]:
                result = result + t
            return nl_fn(result)

        prim = _make_prim(prim_name, compute_nl, [list_coder(in_coder)], out_coder)
        return prim, compute_nl

    else:
        raise ValueError(f"List-merge equation '{eq.name}' has neither einsum nor nonlinearity")
