"""Compiled program wrapper hiding Hydra plumbing from user code.

This is the single callable surface for unified-algebra programs. It wraps
a hydra.graph.Graph and provides encode/reduce/decode in a single __call__,
plus entry point enumeration and parameter rebinding.

Entry points that can be statically compiled (paths, single equations without
param_slots) bypass reduce_term entirely: the _compute closures are composed
once at compile_program time and called directly on native arrays, with no
wire encode/decode between equations. This lets torch.compile or jax.jit fuse
across equation boundaries.

Entry points that cannot be statically compiled (fixpoints, fans, folds,
equations with param_slots) fall back to the reduce_term interpreter.
"""

from __future__ import annotations

import hydra.core as core
import hydra.graph as gr
from hydra.checking import type_of_term
from hydra.context import Context
from hydra.dsl.python import FrozenDict, Just, Left, Node, Right
from hydra.dsl.terms import apply, var
from hydra.lexical import lookup_primitive, lookup_term
from hydra.reduction import reduce_term

import unialg.algebra as alg
from unialg.assembly.graph import assemble_graph, rebind_params
from unialg.terms import tensor_coder

EMPTY_CX = Context(trace=(), messages=(), other=FrozenDict({}))


# ---------------------------------------------------------------------------
# Hydra type checking
# ---------------------------------------------------------------------------

def type_check_term(
    graph: gr.Graph,
    term: core.Term,
    label: str = "term",
) -> core.Type:
    """Type-check a Hydra term against the graph, returning its Type.

    Uses Hydra's own type checker (hydra.checking.type_of_term).
    Raises TypeError with a descriptive message on failure.
    """
    result = type_of_term(EMPTY_CX, graph, term)
    match result:
        case Left(value=err):
            raise TypeError(f"Hydra type error in {label}: {err}")
        case Right(value=typ):
            return typ


# ---------------------------------------------------------------------------
# Recognised bound_term prefixes
# ---------------------------------------------------------------------------

_ENTRY_PREFIXES = (
    "ua.path.",
    "ua.fan.",
    "ua.fold.",
    "ua.unfold.",
    "ua.fixpoint.",
    "ua.parallel.",
    "ua.equation.",
)


def _short_name(full: str) -> str | None:
    for prefix in _ENTRY_PREFIXES:
        if full.startswith(prefix):
            short = full[len(prefix):]
            if short.endswith(".__merge__"):
                return None
            return short
    return None


def _resolve_full_name(entry_point: str, graph: gr.Graph) -> str:
    for prefix in _ENTRY_PREFIXES:
        name = core.Name(f"{prefix}{entry_point}")
        if isinstance(lookup_term(graph, name), Just) or isinstance(lookup_primitive(graph, name), Just):
            return name.value
    available = sorted(
        short
        for key in list(graph.bound_terms) + list(graph.primitives)
        if (short := _short_name(key.value)) is not None
    )
    raise ValueError(
        f"Unknown entry point {entry_point!r}. "
        f"Available entry points: {available}"
    )


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

class Program:
    """A compiled unified-algebra program. Callable by entry-point name.

    Do not construct directly — use compile_program().
    """

    def __init__(self, graph, backend, coder, cx, compiled_fns: dict | None = None,
                 *, _build_args: dict | None = None):
        self._graph = graph
        self._backend = backend
        self._coder = coder
        self._cx = cx
        self._compiled_fns = compiled_fns or {}
        self._build_args = _build_args

    @property
    def graph(self):
        """The underlying hydra.graph.Graph — for type-checking and introspection."""
        return self._graph

    def entry_points(self) -> list[str]:
        """Names of callable bound_terms (paths, fans, folds, unfolds, fixpoints, equations).

        Returns short names (ua.X. prefix stripped). Lens paths appear as
        "name.fwd" and "name.bwd".
        """
        result = []
        for key in list(self._graph.bound_terms) + list(self._graph.primitives):
            short = _short_name(key.value)
            if short is not None:
                result.append(short)
        return sorted(set(result))

    def __call__(self, entry_point: str, *args):
        """Invoke an entry point on numpy/torch arrays.

        For statically compiled entry points (paths, single equations): calls
        the composed native function directly — no Hydra encode/decode, no
        wire format between equations, JIT-friendly.

        For non-compiled entry points (fixpoints, fans with params, etc.):
        falls back to encode → reduce_term → decode.
        """
        # Fast path: statically compiled — native arrays in, native arrays out
        if entry_point in self._compiled_fns:
            return self._compiled_fns[entry_point](*args)

        # Fallback: reduce_term interpreter
        full_name = _resolve_full_name(entry_point, self._graph)

        encoded_args = []
        for i, arg in enumerate(args):
            enc_result = self._coder.decode(None, arg)
            match enc_result:
                case Right(value=enc):
                    encoded_args.append(enc)
                case Left(value=err):
                    raise ValueError(
                        f"Failed to encode argument {i} for entry point "
                        f"{entry_point!r}: {err}"
                    )

        term = var(full_name)
        for enc in encoded_args:
            term = apply(term, enc)

        red_result = reduce_term(self._cx, self._graph, True, term)
        match red_result:
            case Left(value=err):
                raise RuntimeError(
                    f"reduce_term failed for entry point {entry_point!r}: {err}"
                )
            case Right(value=reduced):
                pass

        dec_result = self._coder.encode(None, None, reduced)
        match dec_result:
            case Right(value=arr):
                return arr
            case Left(value=err):
                raise RuntimeError(
                    f"Failed to decode result for entry point {entry_point!r}: {err}"
                )

    def rebind(self, **params) -> "Program":
        """Return a new Program with named parameters substituted.

        Accepts float/int scalars (wrapped as Hydra literals) or pre-wrapped
        Hydra Terms. Returns a new Program; the original is unchanged.

        Recompiles the full graph so fused primitives and compiled_fns
        reflect the new parameter values.
        """
        wrapped = {k: _wrap_scalar(v) for k, v in params.items()}
        if self._build_args is None:
            new_graph = rebind_params(self._graph, wrapped)
            return Program(new_graph, self._backend, self._coder, self._cx,
                           self._compiled_fns)
        existing_hp = self._build_args.get('params') or {}
        merged_hp = {**existing_hp, **wrapped}
        return compile_program(
            self._build_args['equations'], backend=self._backend,
            specs=self._build_args['specs'], params=merged_hp,
            lenses=self._build_args['lenses'],
            extra_sorts=self._build_args['extra_sorts'],
            semirings=self._build_args['semirings'],
        )

    def type_check(self, entry_point: str):
        """Return the Hydra Type of the named entry point."""
        full_name = _resolve_full_name(entry_point, self._graph)
        name = core.Name(full_name)
        match lookup_term(self._graph, name):
            case Just(value=term):
                return type_check_term(self._graph, term)
        match lookup_primitive(self._graph, name):
            case Just(value=prim):
                return prim.type.type
        raise KeyError(f"Entry point not found: {full_name}")


# ---------------------------------------------------------------------------
# Scalar wrapping helper
# ---------------------------------------------------------------------------

def _wrap_scalar(v):
    """Wrap a Python scalar into a Hydra literal term, or pass through if already a Term."""
    if isinstance(v, Node):
        return v
    if isinstance(v, float):
        return core.TermLiteral(value=core.LiteralFloat(value=v))
    if isinstance(v, int):
        return core.TermLiteral(value=core.LiteralInteger(value=v))
    raise TypeError(
        f"rebind: cannot wrap value of type {type(v).__name__!r}; "
        f"expected float, int, or hydra.core.Term"
    )


# ---------------------------------------------------------------------------
# compile_program
# ---------------------------------------------------------------------------

def compile_program(
    equations: list,
    *,
    backend,
    specs: list | None = None,
    params: dict | None = None,
    lenses: list | None = None,
    extra_sorts: list | None = None,
    semirings: dict | None = None,
) -> Program:
    """Compile a unified-algebra specification to a runnable Program.

    This is the single entry point for converting DSL terms into a callable.
    Parser output (future) and hand-written Python both flow through here.
    """
    graph, native_fns, compiled_fns = assemble_graph(
        equations,
        backend,
        specs=specs,
        params=params,
        lenses=lenses,
        extra_sorts=extra_sorts,
        semirings=semirings,
    )
    coder = tensor_coder(backend)
    build_args = dict(equations=equations, specs=specs, lenses=lenses,
                      extra_sorts=extra_sorts, semirings=semirings,
                      params=params)
    return Program(graph, backend, coder, EMPTY_CX, compiled_fns=compiled_fns,
                   _build_args=build_args)
