"""Compiled program wrapper hiding Hydra plumbing from user code.

This is the single callable surface for unified-algebra programs. It wraps
a hydra.graph.Graph and provides encode/reduce/decode in a single __call__,
plus entry point enumeration and parameter rebinding.

All entry points execute via the canonical Hydra path:
  decode(arg) → reduce_term(graph, term) → encode(result)

Backend callables (equations, morphisms, seq/par compositions) are registered
as Hydra primitives or bound_terms. reduce_term dispatches through them.
TermCoder bridges Hydra terms and backend arrays at the program boundary only.
"""

from __future__ import annotations

import hydra.core as core
import hydra.graph as gr
from hydra.checking import type_of_term
from hydra.dsl.python import Just, Left, Right, Node
from hydra.dsl.terms import apply, var, list_ as term_list
from hydra.lexical import empty_context, lookup_primitive, lookup_term
from hydra.reduction import reduce_term

import hydra.dsl.terms as _hterms
from unialg.terms import tensor_coder
from .graph import assemble_graph, rebind_params

EMPTY_CX = empty_context()


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
    "ua.morphism.",
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

    def __init__(self, graph, backend, coder, cx, *, _build_args: dict | None = None, _list_packed_info: dict | None = None):
        self._graph = graph
        self._backend = backend
        self._coder = coder
        self._cx = cx
        self._build_args = _build_args
        self._list_packed_info: dict = _list_packed_info or {}

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

        Canonical path: decode args → build application term → reduce_term →
        encode result. Backend callables execute through registered primitives.

        List-packed primitives (n_params + n_inputs > 3) receive their arguments
        as Hydra TermList nodes so reduce_term dispatches correctly.
        """
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

        prim_key = core.Name(full_name)
        packed = self._list_packed_info.get(prim_key)
        term = var(full_name)
        if packed is not None:
            n_params, n_inputs = packed
            if n_params > 0:
                term = apply(term, term_list(encoded_args[:n_params]))
            if n_inputs > 0:
                term = apply(term, term_list(encoded_args[n_params:n_params + n_inputs]))
        else:
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
                           _list_packed_info=self._list_packed_info)
        existing_hp = self._build_args.get('params') or {}
        merged_hp = {**existing_hp, **wrapped}
        return compile_program(
            self._build_args['equations'], backend=self._backend,
            params=merged_hp,
            extra_sorts=self._build_args.get('extra_sorts'),
            semirings=self._build_args.get('semirings'),
            cells=self._build_args.get('cells'),
            share_groups=self._build_args.get('share_groups'),
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
        return _hterms.float32(float(v))
    if isinstance(v, int):
        return _hterms.int32(int(v))
    raise TypeError(
        f"rebind: cannot wrap value of type {type(v).__name__!r}; "
        f"expected float, int, or a Hydra term"
    )


# ---------------------------------------------------------------------------
# compile_program
# ---------------------------------------------------------------------------

def compile_program(
    equations: list,
    *,
    backend,
    params: dict | None = None,
    extra_sorts: list | None = None,
    semirings: dict | None = None,
    cells: list | None = None,
    share_groups: dict | None = None,
) -> Program:
    """Compile a unified-algebra specification to a runnable Program.

    This is the single entry point for converting DSL terms into a callable.
    Parser output and hand-written Python both flow through here. ``cells``
    is the list of ``NamedCell`` entries produced by the morphism DSL.

    ``share_groups`` maps a group name to the list of op names sharing that
    parameter slot.  The first op in each group is canonical; all others alias
    to it.  This implements Para's ``∆P : P → P × P`` tying semantics.
    """
    params = dict(params or {})
    for _group_name, op_names in (share_groups or {}).items():
        canonical = op_names[0]
        for op_name in op_names[1:]:
            params[op_name] = params.get(canonical)

    graph, native_fns, list_packed_info = assemble_graph(
        equations,
        backend,
        params=params or None,
        extra_sorts=extra_sorts,
        semirings=semirings,
        cells=cells,
    )
    coder = tensor_coder(backend)
    build_args = dict(equations=equations, extra_sorts=extra_sorts,
                      semirings=semirings, params=params, cells=cells,
                      share_groups=share_groups)
    return Program(graph, backend, coder, EMPTY_CX, _build_args=build_args,
                   _list_packed_info=list_packed_info)

