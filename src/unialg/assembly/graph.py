"""DAG validation and graph assembly for equation sets.

Handles topological ordering, sort/rank junction checking, and
Hydra Graph construction from resolved equations.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import hydra.core as core
import hydra.graph
from hydra.dsl.python import FrozenDict, Nothing
from hydra.lexical import empty_graph
import hydra.substitution as subst
import hydra.typing

import unialg.assembly.specs as sp
from unialg.algebra.equation import Equation
from unialg.algebra.sort import sort_wrap
from unialg.assembly.pipeline import EquationPipeline

if TYPE_CHECKING:
    from unialg.backend import Backend


def _build_compositions(
    specs: list,
    eq_by_name: dict[str, core.Term],
    primitives: dict,
    native_fns: dict,
    bound_terms: dict,
    schema_types,
    **kwargs,
) -> None:
    """Validate and build all composition specs. Mutates primitives, native_fns, and bound_terms."""
    for spec in specs:
        spec.validate(eq_by_name, schema_types)
        for name, term in spec.build(primitives, native_fns, **kwargs):
            bound_terms[name] = term


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    sort_terms: list[core.Term],
    primitives: dict | None = None,
    bound_terms: dict | None = None,
) -> hydra.graph.Graph:
    """Assemble a Hydra Graph with sorts registered as schema_types.

    Args:
        sort_terms:  list of sort record terms (from sort())
        primitives:  optional dict of Name -> Primitive (for Phase 3+)
        bound_terms: optional dict of Name -> Term
    """
    tensor_name = core.Name("ua.tensor.NDArray")
    schema = {tensor_name: core.TypeScheme((), core.TypeVariable(tensor_name), Nothing())}
    # Register each sort's structural component names so sort_type_from_term results are ground.
    for st in sort_terms:
        sort_wrap(st).register_schema(schema)
    return dataclasses.replace(
        empty_graph(),
        bound_terms=FrozenDict(bound_terms or {}),
        primitives=FrozenDict(primitives or {}),
        schema_types=FrozenDict(schema),
    )


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def assemble_graph(
    eq_terms: list[core.Term],
    backend: Backend,
    extra_sorts: list[core.Term] | None = None,
    specs: list | None = None,
    hyperparams: dict[str, core.Term] | None = None,
    lenses: list[core.Term] | None = None,
    semirings: dict[str, core.Term] | None = None,
) -> tuple[hydra.graph.Graph, dict]:
    """Resolve equation terms and assemble a Hydra Graph.

    Args:
        eq_terms:     list of equation record terms
        backend:      Backend providing op implementations
        extra_sorts:  additional sort terms to register
        specs:        list of composition specs (PathSpec, FanSpec, FoldSpec,
                      UnfoldSpec, LensPathSpec, FixpointSpec)
        hyperparams:  dict of param_name → scalar Term
        lenses:       list of lens record terms (from lens())
        semirings:    optional dict of semiring_name → semiring Term, used
                      as fallback when a residual path's semiring is not
                      referenced by any equation
    """
    all_specs = list(specs or [])
    merge_names: set[str] = {spec.merge_name for spec in all_specs if isinstance(spec, sp.FanSpec)}

    pipeline = EquationPipeline(eq_terms, backend, merge_names, semirings=semirings)

    bound_terms: dict[core.Name, core.Term] = {}
    if hyperparams:
        for param_name, param_term in hyperparams.items():
            bound_terms[core.Name(f"ua.param.{param_name}")] = param_term

    _build_compositions(all_specs, pipeline.eq_by_name, pipeline.primitives,
                        pipeline.native_fns, bound_terms, pipeline.schema_types,
                        resolved_semirings=pipeline.resolved_semirings, coder=pipeline.coder)

    seen_sorts: dict[str, core.Term] = {}
    for eq in pipeline.eq_by_name.values():
        for st in (eq.domain_sort, eq.codomain_sort):
            seen_sorts.setdefault(str(sort_wrap(st).type_), st)
    all_sorts = list(seen_sorts.values()) + list(extra_sorts or [])
    graph = build_graph(all_sorts, primitives=pipeline.primitives, bound_terms=bound_terms)
    return graph, pipeline.native_fns


# ---------------------------------------------------------------------------
# Hyperparameter rebinding
# ---------------------------------------------------------------------------

def rebind_hyperparams(
    graph: hydra.graph.Graph,
    updates: dict[str, core.Term],
) -> hydra.graph.Graph:
    """Return a new Graph with hyperparameters substituted into term bodies.

    Uses Hydra's structural substitution to replace var("ua.param.X")
    references directly inside lambda bodies, respecting variable scoping.

    Args:
        graph:   existing Hydra Graph
        updates: dict of param_name → new scalar Term
                 (keys without "ua.param." prefix — it's added automatically)
    """
    param_updates = {core.Name(f"ua.param.{k}"): v for k, v in updates.items()}
    ts = hydra.typing.TermSubst(FrozenDict(param_updates))
    new_terms = {name: subst.substitute_in_term(ts, term) for name, term in graph.bound_terms.items()}
    new_terms.update(param_updates)  # also bind the params themselves
    return dataclasses.replace(graph, bound_terms=FrozenDict(new_terms))
