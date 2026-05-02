"""Compile typed morphism terms into runtime callables."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TypeAlias

import hydra.core as core
from hydra.context import Context
from hydra.core import Name, TermApplication, TermRecord, TermVariable
from hydra.dsl.python import FrozenDict, Right, Left
import hydra.dsl.terms as Terms
from hydra.reduction import reduce_term

from hydra.extract.core import record as _extract_record

from unialg.morphism import TypedMorphism
from unialg.morphism import (
    _EQUATION_PREFIX,
    _BIMAP_NAME,
    _LENS_TYPE
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
    """Compiled lens artifact stored in ``Program._compiled_fns``."""
    forward: Callable
    backward: Callable
    residual_sort: object | None = None


CompiledMorphism: TypeAlias = Callable | CompiledLens
Matcher: TypeAlias = Callable[[object], tuple[int, list, list]]

__all__ = ["compile_morphism", "CompiledMorphism", "CompiledLens", "Matcher"]

_NO_MATCH = object()
_EMPTY_CX = Context(trace=(), messages=(), other=FrozenDict({}))


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


def _boundary_coder(boundary, backend, default_coder):
    return boundary.coder(backend) if hasattr(boundary, "coder") else default_coder


def compile_morphism(
    term,
    graph,
    native_fns: dict,
    coder,
    backend,
    *,
    matchers: dict[str, Matcher] | None = None,
) -> CompiledMorphism | None:
    """Compile a morphism term into a callable or ``CompiledLens``."""
    return _compile_term(term, graph, native_fns, coder, backend, matchers=matchers)


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


def _try_lens(term, graph, native_fns, coder, backend, *, matchers):
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
            fwd = _compile_term(fwd_term, graph, native_fns, coder, backend, matchers=matchers)
            bwd = _compile_term(bwd_term, graph, native_fns, coder, backend, matchers=matchers)
            if fwd is None or bwd is None:
                return None
            if isinstance(fwd, CompiledLens) or isinstance(bwd, CompiledLens):
                raise ValueError(
                    f"lens: forward and backward must be Para morphisms, "
                    f"got fwd={type(fwd).__name__} bwd={type(bwd).__name__}"
                )
            return CompiledLens(forward=fwd, backward=bwd, residual_sort=None)
        case _:
            return None


def _compile_via_hydra(
    morphism: TypedMorphism,
    graph,
    coder,
    backend,
):
    """Compile a typed Hydra term by reducing its application."""
    term = morphism.term
    in_coder = _boundary_coder(morphism.domain, backend, coder)
    out_coder = _boundary_coder(morphism.codomain, backend, coder)

    def runtime(value):
        match in_coder.decode(_EMPTY_CX, value):
            case Right(value=arg_term):
                pass
            case Left(value=err):
                raise ValueError(f"compile_morphism: failed to encode input: {err}")

        match reduce_term(_EMPTY_CX, graph, True, apply(term, arg_term)):
            case Right(value=reduced):
                pass
            case Left(value=err):
                raise RuntimeError(f"compile_morphism: reduce_term failed: {err}")

        match out_coder.encode(_EMPTY_CX, graph, reduced):
            case Right(value=result):
                return result
            case Left(value=err):
                raise RuntimeError(f"compile_morphism: failed to decode result: {err}")

    return runtime




def _compile_binary(term, graph, native_fns, coder, backend, *, matchers=None):
    """Compile the Hydra term shapes emitted by ``morphism.seq`` and ``par``."""

    def app(t):
        if not isinstance(t, TermApplication):
            return None
        return t.value.function, t.value.argument

    def is_ref(t, name):
        return isinstance(t, TermVariable) and t.value.value == name

    def finish(op, f_term, g_term):
        f = _compile_term(f_term, graph, native_fns, coder, backend, matchers=matchers)
        g = _compile_term(g_term, graph, native_fns, coder, backend, matchers=matchers)
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


def _compile_term(term, graph, native_fns, coder, backend, *, matchers=None):
    """Dispatch a morphism term to its compiled runtime representation."""
    typed = term if isinstance(term, TypedMorphism) else None
    term = _unwrap_typed_morphism(term)

    fn = _try_structural_lambda(term)
    if fn is not None:
        return backend.compile(fn)

    lens = _try_lens(term, graph, native_fns, coder, backend, matchers=matchers)
    if lens is not None:
        return lens

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

    compiled = _compile_binary(term, graph, native_fns, coder, backend, matchers=matchers)
    if compiled is not _NO_MATCH:
        return compiled

    if typed is not None:
        import warnings
        warnings.warn(
            f"compile_morphism: no fast-path matcher for {type(term).__name__}; "
            f"falling back to Hydra reduce_term",
            stacklevel=2,
        )
        return _compile_via_hydra(typed, graph, coder, backend)

    raise TypeError(
        f"compile_morphism: unrecognized term shape {type(term).__name__}. "
        f"Expected structural lambda (iden/copy/delete), constant lambda "
        f"(lit), TermVariable (equation ref), lens record, or seq/par "
        f"application."
    )
