"""Compile typed morphism terms into runtime callables."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TypeAlias

import hydra.core as core
from hydra.core import Name, TermApplication, TermRecord, TermVariable
from hydra.dsl.prims import prim1
from hydra.dsl.python import Right
import hydra.dsl.terms as Terms

from hydra.extract.core import record as _extract_record

from unialg.morphism import TypedMorphism
from unialg.morphism import (
    _EQUATION_PREFIX,
    _BIMAP_NAME,
    _LENS_TYPE,
    _LENS_SEQ_TYPE,
)

apply = Terms.apply

# Hydra's ``Terms.compose`` builds ``Lambda(Name("arg_"), …)``. Hydra's
# ``Terms.constant`` builds ``Lambda(Name("_"), …)`` (matching
# ``hydra.constants.ignored_variable``). The dispatcher uses these to
# distinguish ``seq``-shaped terms from ``lit``-shaped terms.
_SEQ_ARG = "arg_"
_LIT_PARAM = "_"


@dataclass(frozen=True, slots=True)
class CompiledLens:
    """Compiled lens: forward and backward callables, plus optional residual sort for optic threading."""
    forward: Callable
    backward: Callable
    residual_sort: object | None = None


CompiledMorphism: TypeAlias = Callable | CompiledLens

__all__ = ["compile_morphism", "register_cells", "CompiledMorphism", "CompiledLens"]

MORPHISM_PRIM_PREFIX = "ua.morphism."

_NO_MATCH = object()


def _decode_literal(coder, init_term):
    try:
        match coder.encode(None, None, init_term):
            case Right(value=v): return v, False
            case _: return None, False
    except Exception:
        match init_term.value:
            case core.LiteralFloat(value=v): return v, True
            case core.LiteralInteger(value=v): return int(v), True
            case _: return None, False


def _compose(f, g, x):
    return g(f(x))

def _bimap(f, g, pair):
    return (f(pair[0]), g(pair[1]))

def _unwrap_typed_morphism(term):
    return term.term if isinstance(term, TypedMorphism) else term


def compile_morphism(
    term,
    graph,
    native_fns: dict,
    coder,
    backend,
) -> CompiledMorphism | None:
    """Compile a morphism term into a callable or ``CompiledLens``.

    Returns ``None`` if no structural compiler matches the term shape.
    The caller should then register the term's Hydra lambda as a ``bound_term``
    so that ``reduce_term`` evaluates it via the canonical Hydra path.
    """
    return _compile_term(term, graph, native_fns, coder, backend)


# ---------------------------------------------------------------------------
# Hydra-term-shape dispatcher
# ---------------------------------------------------------------------------

def _try_structural_lambda(term):
    """Match iden, copy, and delete lambda shapes."""
    if not isinstance(term, core.TermLambda):
        return None
    param = term.value.parameter.value
    body = term.value.body
    if isinstance(body, core.TermUnit):
        return lambda *_args: None
    if isinstance(body, TermVariable) and body.value.value == param:
        return lambda x: x
    if isinstance(body, core.TermPair):
        a, b = body.value
        if (isinstance(a, TermVariable) and a.value.value == param
                and isinstance(b, TermVariable) and b.value.value == param):
            return lambda x: (x, x)
    return None


def _try_lens(term, graph, native_fns, coder, backend):
    """Match the typed lens record shape."""
    if not isinstance(term, TermRecord):
        return None
    match _extract_record(_LENS_TYPE, graph, term):
        case Right(value=fields):
            field_map = {f.name: f.term for f in fields}
            fwd_term = field_map.get(Name("forward"))
            bwd_term = field_map.get(Name("backward"))
            if fwd_term is None or bwd_term is None:
                return None
            residual_term = field_map.get(Name("residualSort"))
            residual_sort = None
            if residual_term is not None:
                from unialg.algebra.sort import sort_wrap
                try:
                    residual_sort = sort_wrap(residual_term)
                except Exception:
                    pass
            fwd = _compile_term(fwd_term, graph, native_fns, coder, backend)
            bwd = _compile_term(bwd_term, graph, native_fns, coder, backend)
            if fwd is None or bwd is None:
                return None
            if isinstance(fwd, CompiledLens) or isinstance(bwd, CompiledLens):
                raise ValueError(
                    f"lens: forward and backward must be Para morphisms, "
                    f"got fwd={type(fwd).__name__} bwd={type(bwd).__name__}"
                )
            return CompiledLens(forward=fwd, backward=bwd, residual_sort=residual_sort)
        case _:
            return None


def _try_lens_seq(term, graph, native_fns, coder, backend):
    """Match the LensSeq record and build a threaded CompiledLens."""
    if not isinstance(term, TermRecord):
        return None
    match _extract_record(_LENS_SEQ_TYPE, graph, term):
        case Right(value=fields):
            field_map = {f.name: f.term for f in fields}
            first_term = field_map.get(Name("first"))
            second_term = field_map.get(Name("second"))
            if first_term is None or second_term is None:
                return None
            cl1 = _compile_term(first_term, graph, native_fns, coder, backend)
            cl2 = _compile_term(second_term, graph, native_fns, coder, backend)
            if cl1 is None or cl2 is None:
                return None
            if not isinstance(cl1, CompiledLens) or not isinstance(cl2, CompiledLens):
                raise ValueError(
                    f"lens_seq: both components must compile to CompiledLens, "
                    f"got cl1={type(cl1).__name__} cl2={type(cl2).__name__}"
                )
            def _fwd(s, _cl1=cl1, _cl2=cl2):
                r1, a = _cl1.forward(s)
                r2, b = _cl2.forward(a)
                return (r1, r2), b

            def _bwd(residual_b_pair, _cl1=cl1, _cl2=cl2):
                (r1, r2), b_prime = residual_b_pair
                a_prime = _cl2.backward((r2, b_prime))
                return _cl1.backward((r1, a_prime))

            if cl1.residual_sort is not None and cl2.residual_sort is not None:
                from unialg.algebra.sort import ProductSort
                residual_sort = ProductSort([cl1.residual_sort, cl2.residual_sort])
            else:
                residual_sort = None
            return CompiledLens(forward=_fwd, backward=_bwd, residual_sort=residual_sort)
        case _:
            return None


def _compile_binary(term, graph, native_fns, coder, backend):
    """Compile the Hydra term shapes emitted by ``morphism.seq`` and ``par``."""

    def app(t):
        if not isinstance(t, TermApplication):
            return None
        return t.value.function, t.value.argument

    def is_ref(t, name):
        return isinstance(t, TermVariable) and t.value.value == name

    def finish(op, f_term, g_term):
        f = _compile_term(f_term, graph, native_fns, coder, backend)
        g = _compile_term(g_term, graph, native_fns, coder, backend)
        return None if f is None or g is None else op(f, g)

    def nested_app(subject, at, marker_name, op):
        outer = app(subject)
        inner = app(outer[at]) if outer else None
        if inner and is_ref(inner[at], marker_name):
            other = 1 - at
            return finish(op, inner[other], outer[other])

    if isinstance(term, core.TermLambda) and term.value.parameter.value == _SEQ_ARG:
        op = lambda f, g: backend.compile(partial(_compose, f, g))
        subject, at, marker = term.value.body, 1, _SEQ_ARG
    else:
        op = lambda f, g: backend.compile(partial(_bimap, f, g))
        subject, at, marker = term, 0, _BIMAP_NAME

    compiled = nested_app(subject, at, marker, op)
    if compiled is not None:
        return compiled

    return _NO_MATCH


def _compile_term(term, graph, native_fns, coder, backend):
    """Dispatch a morphism term to its compiled runtime representation."""
    typed = term if isinstance(term, TypedMorphism) else None
    term = _unwrap_typed_morphism(term)

    fn = _try_structural_lambda(term)
    if fn is not None:
        return backend.compile(fn)

    lens = _try_lens(term, graph, native_fns, coder, backend)
    if lens is not None:
        return lens

    lens_seq = _try_lens_seq(term, graph, native_fns, coder, backend)
    if lens_seq is not None:
        return lens_seq

    if isinstance(term, core.TermLambda) and term.value.parameter.value == _LIT_PARAM:
        value, _ = _decode_literal(coder, term.value.body)
        return None if value is None else (lambda *_args, _v=value: _v)

    if isinstance(term, TermVariable):
        name = term.value.value
        if name.startswith(_EQUATION_PREFIX):
            fn = native_fns.get(Name(name))
            if fn is None:
                eq_name = name[len(_EQUATION_PREFIX):]
                raise ValueError(
                    f"compile_morphism: equation {eq_name!r} is not registered in "
                    f"native_fns. Ensure the equation is declared before the "
                    f"morphism that references it."
                )
            return fn
        if typed is None:
            raise TypeError(
                f"compile_morphism: unrecognized term reference name {name!r}. "
                f"Expected a name starting with {_EQUATION_PREFIX!r}."
            )

    compiled = _compile_binary(term, graph, native_fns, coder, backend)
    if compiled is not _NO_MATCH:
        return compiled

    if typed is not None:
        # No structural compiler matched. Return None so the caller
        # registers the TypedMorphism.term as a bound_term for reduce_term.
        return None

    raise TypeError(
        f"compile_morphism: unrecognized term shape {type(term).__name__}. "
        f"Expected structural lambda (iden/copy/delete), constant lambda "
        f"(lit), TermVariable (equation ref), lens record, or seq/par "
        f"application."
    )


def register_cells(named_cells, graph, bound_terms, primitives, native_fns, coder, backend):
    """Compile each morphism and register its primitive(s) into the graph dicts."""
    for named in named_cells:
        fn = compile_morphism(named.cell, graph, native_fns, coder, backend)

        if fn is None:
            cell = named.cell
            raw_term = cell.term if isinstance(cell, TypedMorphism) else cell
            bt_name = Name(f"{MORPHISM_PRIM_PREFIX}{named.name}")
            bound_terms[bt_name] = raw_term
            bound_terms.setdefault(Name(f"ua.equation.{named.name}"), raw_term)
            continue

        if isinstance(named.cell, TypedMorphism):
            cell = named.cell
            in_c = cell.domain_sort.coder(backend) if hasattr(cell.domain_sort, 'coder') else coder
            out_c = cell.codomain_sort.coder(backend) if hasattr(cell.codomain_sort, 'coder') else coder
        else:
            in_c, out_c = coder, coder

        eq_alias = Name(f"ua.equation.{named.name}")
        if hasattr(fn, "forward") and hasattr(fn, "backward"):
            fwd_name = Name(f"{MORPHISM_PRIM_PREFIX}{named.name}.forward")
            bwd_name = Name(f"{MORPHISM_PRIM_PREFIX}{named.name}.backward")
            primitives[fwd_name] = prim1(fwd_name, fn.forward, [], in_c, out_c)
            primitives[bwd_name] = prim1(bwd_name, fn.backward, [], out_c, in_c)
            native_fns[fwd_name] = fn.forward
            native_fns[bwd_name] = fn.backward
            primitives.setdefault(eq_alias, prim1(eq_alias, fn.forward, [], in_c, out_c))
            native_fns.setdefault(eq_alias, fn.forward)
        else:
            prim_name = Name(f"{MORPHISM_PRIM_PREFIX}{named.name}")
            primitives[prim_name] = prim1(prim_name, fn, [], in_c, out_c)
            native_fns[prim_name] = fn
            primitives.setdefault(eq_alias, prim1(eq_alias, fn, [], in_c, out_c))
            native_fns.setdefault(eq_alias, fn)
