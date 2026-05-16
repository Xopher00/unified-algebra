"""Orchestration: typed Morphism → compiled Hydra program → reduced value.

This is the public entry point for the unialg compiler pipeline.
It owns the boundary between semantic construction (Morphism objects)
and backend realization (Hydra reduction).

Contract: callers must supply a fully-constructed Morphism (built via
compose/pair/case/poly_fmap from semantics/morphisms.py). This module
does not perform semantic construction or parse surface syntax.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from hydra.context import Context
from hydra.core import Term
from hydra.dsl.python import Right
import hydra.dsl.terms as Terms
from hydra.graph import Graph
import hydra.lib.maps as HMaps
import hydra.reduction as R

from .objects import EMPTY_GRAPH
from .semantics.morphisms import Morphism
from .syntax import expressions as expr
from .structure.realize import realize_normalized


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
    from .emitters.backend import BackendOps
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
        return _apply_and_reduce(self.term, argument, ctx or Context(), g, "CompiledProgram.run")


def compile_program(morphism: Morphism, graph=None) -> CompiledProgram:
    """Compile a typed Morphism into an executable program.

    The morphism must already be semantically constructed via the combinators
    in semantics/morphisms.py. This function realizes it to a normalized Hydra
    term and bundles the result for execution.
    """
    g = graph or EMPTY_GRAPH
    extra_prims: list = []
    term = realize_normalized(morphism.node, g, extra_prims)
    all_prims = morphism.aux_primitives + tuple(extra_prims)
    return CompiledProgram(term=term, graph=g, aux_primitives=all_prims)


def lower(morphism: Morphism, graph, _extra_prims=None):
    """Realize a typed morphism as a raw Hydra term without evaluating it."""
    return realize_normalized(morphism.node, graph, _extra_prims)


def run(morphism: Morphism, argument, ctx, graph):
    """Apply a morphism to a raw Hydra argument and reduce it."""
    prog = compile_program(morphism, graph)
    return prog.run(argument, ctx)
