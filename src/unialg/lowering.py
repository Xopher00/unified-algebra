"""Lowering: typed Morphism objects → Hydra Terms → reduced values.

Layer responsibility: owns the boundary between algebraic (Morphism objects)
and backend realization (Hydra reduction).
"""

from __future__ import annotations

import dataclasses

from hydra.dsl.python import Right
import hydra.dsl.terms as Terms
import hydra.lib.maps as HMaps
import hydra.reduction as R

from .semantics.morphisms import Morphism
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


def lower(morphism: Morphism, graph, _extra_prims=None):
    """Realize a typed morphism as a raw Hydra term without evaluating it."""
    return realize_normalized(morphism.node, graph, _extra_prims)


def run(morphism: Morphism, argument, ctx, graph):
    """Apply a morphism to a raw Hydra argument and reduce it.

    Any auxiliary primitives carried by the morphism are registered in a
    temporary graph before reduction.  ``argument`` must already be a Hydra
    ``Term`` matching the morphism's raw domain.
    """
    extra_prims = []
    term = lower(morphism, graph, extra_prims)
    aux = morphism.aux_primitives + tuple(extra_prims)
    g = _augment_graph(graph, aux) if aux else graph
    return _apply_and_reduce(term, argument, ctx, g, morphism)
