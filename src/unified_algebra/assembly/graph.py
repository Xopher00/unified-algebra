"""DAG validation and graph assembly for equation sets.

Handles topological ordering, sort/rank junction checking, and
Hydra Graph construction from resolved equations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra.dsl.python import FrozenDict
from ..algebra.sort import sort_type_from_term
from ..specs import PathSpec, FanSpec, FoldSpec, UnfoldSpec, LensPathSpec, FixpointSpec
from .validation import validate_pipeline, _register_sort_components
from ._assembly import (
    _resolve_all_primitives, _register_hyperparams,
    _build_compositions, _build_lens_by_name, _collect_sorts,
    _register_residual_prims,
)

if TYPE_CHECKING:
    import hydra.core as core
    import hydra.graph
    from ..backend import Backend


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
    import hydra.core as core
    import hydra.graph
    from hydra.dsl.python import FrozenDict, Nothing

    schema = {}
    terms = dict(bound_terms or {})

    # Register the tensor type
    tensor_name = core.Name("ua.tensor.NDArray")
    schema[tensor_name] = core.TypeScheme(
        (), core.TypeVariable(tensor_name), Nothing()
    )

    # Register each sort's structural component names so sort_type_from_term
    # results are ground (matches what _build_schema does in assemble_graph).
    for st in sort_terms:
        _register_sort_components(st, schema)

    return hydra.graph.Graph(
        bound_terms=FrozenDict(terms),
        bound_types=FrozenDict({}),
        class_constraints=FrozenDict({}),
        lambda_variables=frozenset(),
        metadata=FrozenDict({}),
        primitives=FrozenDict(primitives or {}),
        schema_types=FrozenDict(schema),
        type_variables=frozenset(),
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
) -> hydra.graph.Graph:
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
    validate_pipeline(eq_terms)

    all_specs = list(specs or [])

    merge_names: set[str] = set()
    for spec in all_specs:
        if isinstance(spec, FanSpec):
            merge_names.add(spec.merge_name)

    primitives, eq_by_name = _resolve_all_primitives(eq_terms, backend, merge_names)

    import hydra.core as core
    from .validation import _build_schema
    schema_types = _build_schema(eq_terms)

    bound_terms: dict[core.Name, core.Term] = {}

    _register_hyperparams(hyperparams, bound_terms)
    lens_by_name = _build_lens_by_name(lenses, eq_by_name)
    _register_residual_prims(all_specs, eq_by_name, primitives, backend,
                             semirings=semirings)
    _build_compositions(all_specs, eq_by_name, primitives, bound_terms, lens_by_name, schema_types)

    all_sorts = _collect_sorts(eq_terms, extra_sorts)
    return build_graph(all_sorts, primitives=primitives, bound_terms=bound_terms)


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
    import hydra.core as core
    import hydra.graph as hgraph
    import hydra.substitution as subst
    import hydra.typing

    param_updates = {
        core.Name(f"ua.param.{k}"): v for k, v in updates.items()
    }
    ts = hydra.typing.TermSubst(FrozenDict(param_updates))

    new_terms = {
        name: subst.substitute_in_term(ts, term)
        for name, term in graph.bound_terms.items()
    }
    # Also update bound_terms entries for the params themselves
    new_terms.update(param_updates)

    return hgraph.Graph(
        bound_terms=FrozenDict(new_terms),
        bound_types=graph.bound_types,
        class_constraints=graph.class_constraints,
        lambda_variables=graph.lambda_variables,
        metadata=graph.metadata,
        primitives=graph.primitives,
        schema_types=graph.schema_types,
        type_variables=graph.type_variables,
    )
