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
import hydra.inference as HI
from hydra.core import TypeFunction

from .syntax import parse_program, Program, MorphismExpr
from .semantics import Morphism
from .structure import realize_normalized
from .emitters import BackendOps, coder_for_type
from .emitters.codecs import _term_value, expect_right

_BACKEND_DIR = Path(__file__).parent / "emitters" / "backends"
_DEFAULT_CTX = None


def default_context():
    """Return the standard Hydra evaluation context."""
    global _DEFAULT_CTX
    if _DEFAULT_CTX is None:
        _DEFAULT_CTX = L.empty_context()
    return _DEFAULT_CTX


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


def _pack_args(args: tuple):
    """Pack positional run args into the left-nested product shape."""
    if len(args) == 1:
        return args[0]
    out = (args[0], args[1])
    for arg in args[2:]:
        out = (out, arg)
    return out


# ---------------------------------------------------------------------------
# Whole-program compilation
# ---------------------------------------------------------------------------

class CompiledProgram:
    """The final route from a parsed program, lowered to a Hydra term.

    Build with ``compile_program``; call ``run(argument)`` to reduce.
    """

    def __init__(self, term, graph, ctx, backends: dict, input_type, output_type):
        self._term = term
        self._graph = graph
        self._ctx = ctx
        self._input_type = input_type
        self._output_type = output_type
        # Find first backend with a binary adapter — drives store-based execution path.
        self._backend_ops: BackendOps | None = None
        for ops in backends.values():
            if ops.binary_adapter is not None:
                self._backend_ops = ops
                break

    def run(self, *args):
        """Run the compiled program, returning a native backend value or Python scalar."""
        store = self._backend_ops.store if self._backend_ops is not None else None
        if store is not None:
            store.reset()
        try:
            if args:
                input_coder = coder_for_type(self._input_type)
                input_value = _pack_args(args)
                if self._backend_ops is not None:
                    input_value = self._backend_ops.encode_boundary_input(self._input_type, input_value)
                argument = expect_right(input_coder.decode(self._ctx, input_value), "run")
            else:
                argument = P.unit().value
            result = _apply_and_reduce(self._term, argument, self._ctx, self._graph, "program")
            result_value = _term_value(result, "program")
            if self._backend_ops is not None:
                return self._backend_ops.decode_boundary_output(self._output_type, result_value)
            return result_value
        finally:
            if store is not None:
                store.reset()


def _infer_boundary_types(term, ctx, graph):
    """Infer the domain and codomain of a compiled morphism term via Hydra type inference."""
    result = HI.infer_type_of_term(ctx, graph, term, "program boundary")
    if not isinstance(result, Right):
        raise RuntimeError(f"compile_program: could not infer boundary types: {result}")
    fn_type = result.value.type
    if not isinstance(fn_type, TypeFunction):
        raise RuntimeError(f"compile_program: expected function type, got {type(fn_type).__name__}")
    return fn_type.value.domain, fn_type.value.codomain


def compile_program(src: str) -> CompiledProgram:
    """Parse and compile *src*; the final route is the program output."""
    prog, backends = load_program(src)
    if not prog.morphisms:
        raise ValueError("compile_program: program defines no routes")
    g = BackendOps.build_graph(backends)
    term = None
    for _, expr_node in prog.morphisms.items():
        extra = []
        term = lower(expr_node, g, extra)
        g = _augment_graph(g, extra)
    ctx = default_context()
    input_type, output_type = _infer_boundary_types(term, ctx, g)
    return CompiledProgram(term, g, ctx, backends, input_type, output_type)
