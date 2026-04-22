"""Compiled program wrapper hiding Hydra plumbing from user code.

This is the single callable surface for unified-algebra programs. It wraps
a hydra.graph.Graph and provides encode/reduce/decode in a single __call__,
plus entry point enumeration and hyperparam rebinding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import unialg.algebra as alg
from unialg.assembly.graph import assemble_graph, rebind_hyperparams

if TYPE_CHECKING:
    import hydra.core as core
    import hydra.graph


# ---------------------------------------------------------------------------
# Hydra type checking
# ---------------------------------------------------------------------------

def type_check_term(
    graph: hydra.graph.Graph,
    term: core.Term,
    label: str = "term",
) -> core.Type:
    """Type-check a Hydra term against the graph, returning its Type.

    Uses Hydra's own type checker (hydra.checking.type_of_term).
    Raises TypeError with a descriptive message on failure.
    """
    from hydra.context import Context
    from hydra.dsl.python import FrozenDict, Right, Left
    from hydra.checking import type_of_term

    cx = Context(trace=(), messages=(), other=FrozenDict({}))
    result = type_of_term(cx, graph, term)
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
    "ua.equation.",
)


def _short_name(full: str) -> str | None:
    for prefix in _ENTRY_PREFIXES:
        if full.startswith(prefix):
            return full[len(prefix):]
    return None


def _resolve_full_name(entry_point: str, bound_terms: dict, primitives: dict | None = None) -> str:
    all_keys = list(bound_terms)
    if primitives:
        all_keys += list(primitives)
    for prefix in _ENTRY_PREFIXES:
        candidate = f"{prefix}{entry_point}"
        for key in all_keys:
            if key.value == candidate:
                return candidate
    available = sorted(
        short
        for key in all_keys
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

    def __init__(self, graph, backend, coder, cx):
        self._graph = graph
        self._backend = backend
        self._coder = coder
        self._cx = cx

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

        Encodes each arg, builds the applied Hydra term, runs reduce_term,
        and decodes the result. All positional args are array-shaped.
        Does not commit to interpreted execution — a future compile_program(...,
        mode="native") may use a different backend without changing this surface.
        """
        from hydra.dsl.python import Right, Left
        from hydra.dsl.terms import apply, var
        from hydra.reduction import reduce_term

        full_name = _resolve_full_name(entry_point, self._graph.bound_terms, self._graph.primitives)

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

    def rebind(self, **hyperparams) -> "Program":
        """Return a new Program with hyperparameters substituted.

        Accepts float/int scalars (wrapped as Hydra literals) or pre-wrapped
        Hydra Terms. Returns a new Program; the original is unchanged.
        """
        wrapped = {k: _wrap_scalar(v) for k, v in hyperparams.items()}
        new_graph = rebind_hyperparams(self._graph, wrapped)
        return Program(new_graph, self._backend, self._coder, self._cx)

    def type_check(self, entry_point: str):
        """Return the Hydra Type of the named entry point."""
        from hydra.core import Name

        full_name = _resolve_full_name(entry_point, self._graph.bound_terms, self._graph.primitives)
        term = self._graph.bound_terms[Name(full_name)]
        return type_check_term(self._graph, term)


# ---------------------------------------------------------------------------
# Scalar wrapping helper
# ---------------------------------------------------------------------------

def _wrap_scalar(v):
    """Wrap a Python scalar into a Hydra literal term, or pass through if already a Term."""
    import hydra.core as core

    from hydra.dsl.python import Node
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
    hyperparams: dict | None = None,
    lenses: list | None = None,
    extra_sorts: list | None = None,
    semirings: dict | None = None,
) -> Program:
    """Compile a unified-algebra specification to a runnable Program.

    This is the single entry point for converting DSL terms into a callable.
    Parser output (future) and hand-written Python both flow through here.
    """
    from hydra.context import Context
    from hydra.dsl.python import FrozenDict

    graph = assemble_graph(
        equations,
        backend,
        specs=specs,
        hyperparams=hyperparams,
        lenses=lenses,
        extra_sorts=extra_sorts,
        semirings=semirings,
    )
    cx = Context(trace=(), messages=(), other=FrozenDict({}))
    return Program(graph, backend, alg.tensor_coder(), cx)
