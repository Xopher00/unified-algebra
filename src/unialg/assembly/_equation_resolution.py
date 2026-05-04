"""Equation lowering: compile Equation declarations to Hydra Primitives.

This is the assembly-layer counterpart to algebra/equation.py. While
equation.py holds the pure declaration, this module turns declarations
into executable Hydra Primitives by resolving against a backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra.core as core
from hydra.graph import Primitive
from hydra.dsl.prims import prim1, prim2, prim3, float32 as float32_coder, list_ as list_coder
from hydra.dsl.python import FrozenDict

from unialg.algebra import compile_einsum, contract_and_apply, contract_merge, Equation
from ._validation import validate_pipeline

if TYPE_CHECKING:
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
    skip_fn: object = None


_PRIMS = {1: prim1, 2: prim2, 3: prim3}


def _make_prim(prim_name, compute, coders, out_coder) -> Primitive:
    n = len(coders)
    if n not in _PRIMS:
        raise ValueError(f"Primitive '{prim_name.value}': packed arity {n} exceeds max 3")
    return _PRIMS[n](prim_name, compute, [], *coders, out_coder)


def _build_resolved(in_coder, n_params, n_inputs, sr, compiled, backend, nl_fn, skip_fn=None):
    def _core(params, tensors):
        result = contract_and_apply(compiled, list(tensors), sr, backend, nl_fn, tuple(params))
        if skip_fn is not None:
            result = skip_fn(result, tensors[-1])
        return result

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
            sr = sr.with_adjoint(eq.name)
        skip_fn = sr.plus_elementwise if eq.skip else None
    else:
        sr = compiled = None
        n_inputs = 0
        skip_fn = None
    in_coder, out_coder = eq.coders(backend)
    nl_fn = backend.unary(eq.nonlinearity) if has_nl else None
    return EquationCompiled(
        has_einsum=has_einsum, has_nl=has_nl, nl_fn=nl_fn,
        in_coder=in_coder, out_coder=out_coder,
        prim_name=eq.prim_name, sr=sr, compiled=compiled,
        n_inputs=n_inputs, n_params=len(eq.param_slots), skip_fn=skip_fn)


def resolve_equation(eq: "Equation", backend: "Backend"):
    """Compile an Equation to a Hydra Primitive.

    Returns (prim, native_fn, sr, in_coder, n_params, n_inputs, is_list_packed).
    is_list_packed is True when n_params + n_inputs > 3.
    """
    from unialg.algebra.sort import ProductSort
    ctx = compile_equation(eq, backend)
    if not ctx.has_einsum and not ctx.has_nl:
        raise ValueError(f"Equation '{eq.name}' has neither einsum nor nonlinearity")
    n_inputs = 1 if not ctx.has_einsum else ctx.n_inputs

    if isinstance(eq.domain_sort, ProductSort) and ctx.has_einsum:
        pair_coder = eq.domain_sort.coder(backend)

        def _pair_native(pair_val, *params):
            tensors = list(pair_val) if not isinstance(pair_val, list) else pair_val
            return contract_and_apply(ctx.compiled, tensors, ctx.sr, backend, ctx.nl_fn, tuple(params))

        coders = [float32_coder()] * ctx.n_params + [pair_coder]
        prim = _make_prim(ctx.prim_name, _pair_native, coders, ctx.out_coder)
        is_list_packed = False
        return prim, _pair_native, ctx.sr, pair_coder, ctx.n_params, 1, is_list_packed

    coders, hydra_compute, native_fn = _build_resolved(
        ctx.in_coder, ctx.n_params, n_inputs, ctx.sr, ctx.compiled, backend, ctx.nl_fn,
        skip_fn=ctx.skip_fn)
    prim = _make_prim(ctx.prim_name, hydra_compute, coders, ctx.out_coder)
    is_list_packed = (ctx.n_params + n_inputs) > 3
    return prim, native_fn, ctx.sr, ctx.in_coder, ctx.n_params, n_inputs, is_list_packed


def _resolve_equations(eq_terms, backend, semirings, extra_sorts=None):
    """Returns (eq_by_name, primitives, native_fns, coder, schema_types, list_packed_info)."""
    eq_by_name: dict[str, Equation] = {}
    for i, t in enumerate(eq_terms):
        eq = Equation.from_term(t)
        if eq.name in eq_by_name:
            raise ValueError(f"Duplicate equation name '{eq.name}' (positions {list(eq_by_name).index(eq.name)} and {i})")
        eq_by_name[eq.name] = eq

    primitives: dict = {}
    native_fns: dict = {
        core.Name("ua.equation.fst"):            lambda p: p[0],
        core.Name("ua.equation.snd"):            lambda p: p[1],
        core.Name("ua.equation.pair_construct"): lambda a, b: (a, b),
    }
    list_packed_info: dict = {}
    coder = None
    resolved_sr = resolve_semirings(semirings, backend) if semirings else {}

    schema: dict = {}
    for eq in eq_by_name.values():
        eq.register_sorts(schema)
        prim, native_fn, sr, eq_coder, n_params, n_inputs, is_list_packed = resolve_equation(eq, backend)
        sr_name = eq.semiring_name
        if sr_name and sr is not None:
            resolved_sr.setdefault(sr_name, sr)
        if coder is None:
            coder = eq_coder
        primitives[prim.name] = prim
        native_fns[prim.name] = native_fn
        if is_list_packed:
            list_packed_info[prim.name] = (n_params, n_inputs)

    for st in (extra_sorts or []):
        if st is not None:
            st.register_schema(schema)
    schema_types = FrozenDict(schema)
    validate_pipeline(list(eq_by_name.values()), schema_types)

    return eq_by_name, primitives, native_fns, coder, schema_types, list_packed_info


def resolve_equation_as_merge(eq: "Equation", backend: "Backend", prim_name_override=None):
    """Compile an Equation as a list-consuming merge Primitive (for fan compositions)."""
    ctx = compile_equation(eq, backend)
    pn = prim_name_override if prim_name_override is not None else ctx.prim_name
    if ctx.has_einsum:
        def compute_fn(tensors):
            return contract_merge(ctx.compiled, tensors, ctx.sr, backend, ctx.nl_fn, ctx.n_inputs, eq.name)
        compute_fn.n_inputs = ctx.n_inputs
    elif ctx.has_nl:
        def compute_fn(tensors):
            result = tensors[0]
            for t in tensors[1:]:
                result = ctx.sr.plus_elementwise(result, t)
            return ctx.nl_fn(result)
        compute_fn.n_inputs = 1
    else:
        raise ValueError(f"List-merge equation '{eq.name}' has neither einsum nor nonlinearity")
    prim = _make_prim(pn, compute_fn, [list_coder(ctx.in_coder)], ctx.out_coder)
    return prim, compute_fn, ctx.sr, ctx.in_coder
