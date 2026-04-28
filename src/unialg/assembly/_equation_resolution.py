"""Equation lowering: compile Equation declarations to Hydra Primitives.

This is the assembly-layer counterpart to algebra/equation.py. While
equation.py holds the pure declaration, this module turns declarations
into executable Hydra Primitives by resolving against a backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hydra.dsl.prims import prim1, prim2, prim3, float32 as float32_coder, list_ as list_coder
from hydra.graph import Primitive

from unialg.algebra.contraction import compile_einsum, contract_and_apply, contract_merge

if TYPE_CHECKING:
    from unialg.algebra.equation import Equation
    from unialg.backend import Backend


@dataclass(frozen=True, slots=True)
class EquationCompiled:
    has_einsum: bool
    has_nl: bool
    nl_fn: object
    in_coder: object
    out_coder: object
    prim_name: object
    sr: object
    compiled: object
    n_inputs: int
    n_params: int


_PRIMS = {1: prim1, 2: prim2, 3: prim3}


def _make_prim(prim_name, compute, coders, out_coder) -> Primitive:
    n = len(coders)
    if n not in _PRIMS:
        raise ValueError(f"Primitive '{prim_name.value}': packed arity {n} exceeds max 3")
    return _PRIMS[n](prim_name, compute, [], *coders, out_coder)


def _build_resolved(in_coder, n_params, n_inputs, sr, compiled, backend, nl_fn):
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


def resolve_semirings(semirings: dict, backend: "Backend") -> dict:
    """Resolve a name→Semiring dict to name→Semiring.Resolved."""
    return {name: sr.resolve(backend) for name, sr in semirings.items()}


def compile_equation(eq: "Equation", backend: "Backend") -> EquationCompiled:
    """Prepare an Equation for resolution against a backend."""
    eq.validate_axes()
    einsum_str = eq.effective_einsum()
    has_einsum = bool(einsum_str)
    has_nl = bool(eq.nonlinearity)
    if has_einsum:
        sr = eq.semiring.resolve(backend)
        compiled = compile_einsum(einsum_str)
        n_inputs = len(compiled.input_vars)
        if eq.adjoint:
            if sr.residual_elementwise is None:
                raise ValueError(
                    f"Op '{eq.name}': adjoint=true requires a residual= op on the semiring")
            from dataclasses import replace as _replace
            _res, _red = sr.residual_elementwise, sr.times_reduce
            sr = _replace(sr, contraction_fn=lambda cs, be, p: cs(_res, _red))
    else:
        sr = compiled = None
        n_inputs = 0
    in_coder, out_coder = eq.coders(backend)
    nl_fn = backend.unary(eq.nonlinearity) if has_nl else None
    return EquationCompiled(
        has_einsum=has_einsum, has_nl=has_nl, nl_fn=nl_fn,
        in_coder=in_coder, out_coder=out_coder,
        prim_name=eq.prim_name, sr=sr, compiled=compiled,
        n_inputs=n_inputs, n_params=len(eq.param_slots))


def resolve_equation(eq: "Equation", backend: "Backend"):
    """Compile an Equation to a Hydra Primitive."""
    ctx = compile_equation(eq, backend)
    if not ctx.has_einsum and not ctx.has_nl:
        raise ValueError(f"Equation '{eq.name}' has neither einsum nor nonlinearity")
    n_inputs = 1 if not ctx.has_einsum else ctx.n_inputs
    coders, hydra_compute, native_fn = _build_resolved(
        ctx.in_coder, ctx.n_params, n_inputs, ctx.sr, ctx.compiled, backend, ctx.nl_fn)
    prim = _make_prim(ctx.prim_name, hydra_compute, coders, ctx.out_coder)
    return prim, native_fn, ctx.sr, ctx.in_coder


def resolve_equation_as_merge(eq: "Equation", backend: "Backend", prim_name_override=None):
    """Compile an Equation as a list-consuming merge Primitive (for fan compositions)."""
    ctx = compile_equation(eq, backend)
    pn = prim_name_override if prim_name_override is not None else ctx.prim_name
    if ctx.has_einsum:
        def compute_merge(tensors):
            return contract_merge(ctx.compiled, tensors, ctx.sr, backend, ctx.nl_fn, ctx.n_inputs, eq.name)
        compute_merge.n_inputs = ctx.n_inputs
        prim = _make_prim(pn, compute_merge, [list_coder(ctx.in_coder)], ctx.out_coder)
        return prim, compute_merge, ctx.sr, ctx.in_coder
    elif ctx.has_nl:
        def compute_nl(tensors):
            result = tensors[0]
            for t in tensors[1:]:
                result = result + t
            return ctx.nl_fn(result)
        compute_nl.n_inputs = 1
        prim = _make_prim(pn, compute_nl, [list_coder(ctx.in_coder)], ctx.out_coder)
        return prim, compute_nl, ctx.sr, ctx.in_coder
    else:
        raise ValueError(f"List-merge equation '{eq.name}' has neither einsum nor nonlinearity")
