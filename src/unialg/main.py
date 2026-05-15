"""Entry point: surface syntax → Hydra reduction.

  load_program(src)    -> (Program, dict[str, BackendOps])
  compile_program(src) -> CompiledProgram
  lower(node, graph)                  -> Term
  run(morphism, argument, ctx, graph) -> Term

CompiledProgram compiles all routes top-to-bottom; the final route is
the program output.  Call compiled.run(argument) to reduce it.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from hydra.dsl.python import Right
import hydra.dsl.meta.phantoms as P
import hydra.dsl.terms as Terms
import hydra.lib.maps as HMaps
import hydra.lexical as L
import hydra.reduction as R
import hydra.sources.libraries as Libs

from .syntax.parse import parse_program, Program
from .syntax.expressions import MorphismExpr
from .semantics.morphisms import Morphism
from .semantics.typeops import _EMPTY_GRAPH
from .structure.realize import realize_normalized
from .emitters.backend import BackendOps
from .emitters.codecs import _term_value, _expect_right, coder_for_type

_BACKEND_DIR = Path(__file__).parent / "emitters" / "backends"
_DEFAULT_GRAPH = None
_DEFAULT_CTX = None


def default_context():
    """Return the standard Hydra evaluation context."""
    global _DEFAULT_CTX
    if _DEFAULT_CTX is None:
        _DEFAULT_CTX = L.empty_context()
    return _DEFAULT_CTX


def default_graph():
    """Return the standard Hydra graph with all built-in library primitives."""
    global _DEFAULT_GRAPH
    if _DEFAULT_GRAPH is None:
        primitives = []
        for attr in dir(Libs):
            if attr.startswith("register_") and attr.endswith("_primitives"):
                primitives.extend(getattr(Libs, attr)().values())
        _DEFAULT_GRAPH = _augment_graph(_EMPTY_GRAPH, primitives)
    return _DEFAULT_GRAPH


# ---------------------------------------------------------------------------
# Backend loading
# ---------------------------------------------------------------------------

def _load_backend(name: str) -> tuple[BackendOps, dict[str, MorphismExpr]]:
    """Load a backend spec and expose its primitive aliases to the parser."""
    ops = BackendOps.from_spec(_BACKEND_DIR / f"{name}.json")
    return ops, {alias: ops[alias].node for alias in ops.keys()}


def load_program(src: str) -> tuple[Program, dict[str, BackendOps]]:
    """Parse *src* and resolve `load <backend>` directives via BackendOps."""
    backends: dict[str, BackendOps] = {}

    def handler(name: str) -> dict[str, MorphismExpr]:
        """Load one backend directive and return parser environment bindings."""
        ops, env = _load_backend(name)
        backends[name] = ops
        return env

    return parse_program(src, load_handler=handler), backends


# ---------------------------------------------------------------------------
# Lowering and reduction
# ---------------------------------------------------------------------------

def _augment_graph(graph, aux_primitives):
    """Return ``graph`` with auxiliary primitives merged by name."""
    if not aux_primitives:
        return graph
    prims = dict(graph.primitives)
    for prim in aux_primitives:
        prims[prim.name] = prim
    return dataclasses.replace(graph, primitives=HMaps.from_list(list(prims.items())))


def _apply_and_reduce(term, argument, ctx, graph, label):
    """Apply a raw term to one raw argument and unwrap Hydra reduction success."""
    result = R.reduce_term(ctx, graph, True, Terms.apply(term, argument))
    if isinstance(result, Right):
        return result.value
    raise RuntimeError(f"Reduction failed for {label}: {result}")


def lower(node: MorphismExpr, graph, _extra_prims=None):
    """Realize a morphism expression as a raw Hydra term without evaluating it."""
    return realize_normalized(node, graph, _extra_prims)


def run(morphism: Morphism, argument, ctx, graph):
    """Apply a morphism to a raw Hydra argument and reduce it."""
    extra_prims = []
    term = lower(morphism.node, graph, extra_prims)
    aux = morphism.aux_primitives + tuple(extra_prims)
    g = _augment_graph(graph, aux) if aux else graph
    return _apply_and_reduce(term, argument, ctx, g, morphism)


# ---------------------------------------------------------------------------
# Whole-program compilation
# ---------------------------------------------------------------------------

class CompiledProgram:
    """The final route from a parsed program, lowered to a Hydra term.

    Build with ``compile_program``; call ``run(argument)`` to reduce.
    """

    def __init__(self, term, graph, ctx, backends: dict):
        self._term = term
        self._graph = graph
        self._ctx = ctx
        # Find first backend with a binary adapter — drives store-based execution path.
        self._backend_ops: BackendOps | None = None
        for ops in backends.values():
            if ops.binary_adapter is not None:
                self._backend_ops = ops
                break
        # LEGACY/TEMPORARY: derive arg coder from the first backend primitive's declared type.
        # Compatibility bridge for homogeneous single-backend programs.
        first_bp = next(
            (bp for ops in backends.values() for bp in ops.primitives.values()),
            None,
        )
        self._arg_coder = coder_for_type(first_bp.arg_type) if first_bp is not None else None

    def run(self, *args):
        """Run the compiled program, returning a native backend value or Python scalar."""
        store = self._backend_ops.store if self._backend_ops is not None else None
        if store is not None:
            store.reset()
        try:
            if args:
                if self._arg_coder is None:
                    raise ValueError("run: no backend loaded — cannot encode arguments")
                if self._backend_ops is not None:
                    _ops = self._backend_ops
                    _coder = self._arg_coder
                    _ctx = self._ctx
                    def _encode(v):
                        key = _ops.put_input(v)
                        return _expect_right(_coder.decode(_ctx, key), "run")
                    encoded = [_encode(v) for v in args]
                else:
                    encoded = [_expect_right(self._arg_coder.decode(self._ctx, v), "run") for v in args]
                argument = Terms.pair(encoded[0], encoded[1]) if len(encoded) == 2 else encoded[0]
            else:
                argument = P.unit().value
            result = _apply_and_reduce(self._term, argument, self._ctx, self._graph, "program")
            if self._backend_ops is not None:
                result_key = _term_value(result, "program")
                output = self._backend_ops.get_output(result_key)
                return output
            return _term_value(result, "program")
        finally:
            if store is not None:
                store.reset()


def compile_program(src: str) -> CompiledProgram:
    """Parse and compile *src*; the final route is the program output."""
    prog, backends = load_program(src)
    if not prog.morphisms:
        raise ValueError("compile_program: program defines no routes")
    g = default_graph()
    for ops in backends.values():
        g = ops.to_graph(g)
    term = None
    for _, expr_node in prog.morphisms.items():
        extra = []
        term = lower(expr_node, g, extra)
        g = _augment_graph(g, extra)
    return CompiledProgram(term, g, default_context(), backends)
