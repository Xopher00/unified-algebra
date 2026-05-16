"""Compiler orchestration for unialg programs.

The source pipeline is:

    text -> parse_program -> construct_program -> realize_normalized -> reduce

Use ``compile_program`` for source text and ``compile_morphism`` when the
caller already has a typed ``Morphism``.  This module coordinates the layers;
syntax, semantic construction, and realization remain in their own modules.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from hydra.core import Term
from hydra.dsl.python import Right
import hydra.dsl.terms as Terms
from hydra.graph import Graph
import hydra.lexical as L
import hydra.lib.maps as HMaps
import hydra.reduction as R

from .objects import standard_graph
from .semantics.construct import construct_program
from .semantics.morphisms import Morphism
from .syntax import expressions as expr
from .syntax.parse import parse_program
from .structure.realize import realize_normalized


def default_context():
    """Return Hydra's empty evaluation context."""
    return L.empty_context()


def _augment_graph(graph, aux_primitives):
    """Return ``graph`` extended with morphism-local Hydra primitives."""
    for prim in aux_primitives:
        graph = dataclasses.replace(
            graph, primitives=HMaps.insert(prim.name, prim, graph.primitives)
        )
    return graph


def _apply_and_reduce(term, argument, ctx, graph, label):
    """Apply ``term`` to ``argument`` and unwrap a successful Hydra reduction."""
    result = R.reduce_term(ctx, graph, True, Terms.apply(term, argument))
    if isinstance(result, Right):
        return result.value
    raise RuntimeError(f"Reduction failed for {label}: {result}")


def load_backend(spec_path) -> dict[str, Morphism]:
    """Load a backend spec and produce typed Morphisms for each primitive.

    Term construction is deferred to realize.py. Only type info is resolved here.
    """
    from .runtime.backend import BackendOps
    ops = BackendOps.from_spec(spec_path)
    env: dict[str, Morphism] = {}
    for name, bp in ops.primitives.items():
        env[name] = Morphism(
            node=expr.BackendPrim(bp.primitive, bp.arity, bp.dom, bp.result_type),
            aux_primitives=(bp.primitive,),
        )
    return env


@dataclass(frozen=True)
class CompiledProgram:
    """A compiled morphism bundled with its execution context."""

    term: Term
    graph: Graph
    aux_primitives: tuple

    def run(self, argument, ctx=None):
        """Apply this program to a Hydra argument and reduce."""
        g = _augment_graph(self.graph, self.aux_primitives) if self.aux_primitives else self.graph
        return _apply_and_reduce(self.term, argument, ctx or default_context(), g, "CompiledProgram.run")


def compile_morphism(morphism: Morphism, graph=None) -> CompiledProgram:
    """Compile an already-constructed typed ``Morphism``.

    The morphism must already be semantically constructed via the combinators
    in semantics/morphisms.py. This function realizes it to a normalized Hydra
    term and bundles the result for execution.
    """
    g = graph or standard_graph()
    extra_prims: list = []
    term = realize_normalized(morphism.node, g, extra_prims)
    all_prims = morphism.aux_primitives + tuple(extra_prims)
    return CompiledProgram(term=term, graph=g, aux_primitives=all_prims)


def _program_output(routes: dict[str, Morphism], route: str | None) -> Morphism:
    """Return the explicit route, or the final route in source order."""
    if not routes:
        raise ValueError("compile_program: program defines no routes")
    if route is not None:
        if route not in routes:
            raise KeyError(f"compile_program: unknown route {route!r}")
        return routes[route]
    return next(reversed(routes.values()))


def _resolve_backend_spec(name: str) -> str:
    from pathlib import Path
    spec = Path(__file__).parent / "runtime" / "backends" / f"{name}.json"
    if not spec.exists():
        raise ValueError(f"unknown backend {name!r}: {spec} not found")
    return str(spec)


def compile_program(
    src: str,
    *,
    env: dict[str, Morphism] | None = None,
    graph=None,
    route: str | None = None,
) -> CompiledProgram:
    """Parse, semantically construct, and compile a source program."""
    parsed = parse_program(src)
    base_env = dict(env) if env else {}
    for backend_name in parsed.loads:
        base_env.update(load_backend(_resolve_backend_spec(backend_name)))
    constructed = construct_program(parsed, base_env)
    return compile_morphism(_program_output(constructed.routes, route), graph)


def lower(morphism: Morphism, graph, _extra_prims=None):
    """Realize a typed morphism as a raw Hydra term without evaluating it."""
    return realize_normalized(morphism.node, graph, _extra_prims)


def run(morphism: Morphism, argument, ctx, graph):
    """Apply a morphism to a raw Hydra argument and reduce it."""
    prog = compile_morphism(morphism, graph)
    return prog.run(argument, ctx)
