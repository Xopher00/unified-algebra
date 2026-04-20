"""DAG validation and graph assembly for equation sets.

Handles topological ordering, sort/rank junction checking, and
Hydra Graph construction from resolved equations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from hydra.dsl.python import FrozenDict
from .sort import sort_type_from_term
from .validation import resolve_dag, validate_pipeline, ua_primitives
from ._assembly import (
    _collect_merge_names, _resolve_all_primitives, _register_hyperparams,
    _build_paths, _build_fans, _build_recursion, _build_lenses, _collect_sorts,
)

if TYPE_CHECKING:
    import hydra.core as core
    import hydra.graph
    from .backend import Backend


# ---------------------------------------------------------------------------
# NamedTuple spec types
# ---------------------------------------------------------------------------

class PathSpec(NamedTuple):
    """Convenience spec for a sequential path composition."""
    name: str
    eq_names: list[str]
    domain_sort: object  # core.Term
    codomain_sort: object  # core.Term


class FanSpec(NamedTuple):
    """Convenience spec for a parallel fan composition."""
    name: str
    branch_names: list[str]
    merge_name: str
    domain_sort: object  # core.Term
    codomain_sort: object  # core.Term


class FoldSpec(NamedTuple):
    """Convenience spec for a fold (catamorphism)."""
    name: str
    step_name: str
    init_term: object  # core.Term
    domain_sort: object  # core.Term
    state_sort: object  # core.Term


class UnfoldSpec(NamedTuple):
    """Convenience spec for an unfold (anamorphism)."""
    name: str
    step_name: str
    n_steps: int
    domain_sort: object  # core.Term
    state_sort: object  # core.Term


class LensPathSpec(NamedTuple):
    """Convenience spec for a bidirectional lens path."""
    name: str
    lens_names: list[str]
    domain_sort: object  # core.Term
    codomain_sort: object  # core.Term


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

    # Register each sort with semiring identity (and batched flag) in the type
    for st in sort_terms:
        sort_type = sort_type_from_term(st)
        sort_type_name = sort_type.value  # core.Name
        schema[sort_type_name] = core.TypeScheme(
            (), core.TypeVariable(sort_type_name), Nothing()
        )
        terms[sort_type_name] = st

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
    paths: list[tuple[str, list[str], core.Term, core.Term]] | None = None,
    fans: list[tuple[str, list[str], str, core.Term, core.Term]] | None = None,
    folds: list[tuple[str, str, core.Term, core.Term, core.Term]] | None = None,
    unfolds: list[tuple[str, str, int, core.Term, core.Term]] | None = None,
    hyperparams: dict[str, core.Term] | None = None,
    lenses: list[core.Term] | None = None,
    lens_paths: list[tuple[str, list[str], core.Term, core.Term]] | None = None,
) -> hydra.graph.Graph:
    """Resolve equation terms and assemble a Hydra Graph.

    Validates sort/rank junctions, resolves each equation into a Primitive,
    builds lambda terms for paths/fans/folds/unfolds/lens_paths, collects all
    sorts, and builds the Graph with Hydra's standard library included.

    Args:
        eq_terms:    list of equation record terms
        backend:     Backend providing op implementations
        extra_sorts: additional sort terms to register
        paths:       list of (name, eq_names, domain_sort, codomain_sort)
        fans:        list of (name, branch_names, merge_name, domain_sort, codomain_sort)
        folds:       list of (name, step_name, init_term, domain_sort, state_sort)
        unfolds:     list of (name, step_name, n_steps, domain_sort, state_sort)
        hyperparams: dict of param_name → scalar Term (e.g. {"temperature": Terms.float32(1.0)})
        lenses:      list of lens record terms (from lens())
        lens_paths:  list of (name, lens_names, domain_sort, codomain_sort[, params])
                     Each entry produces two bound_terms: "ua.path.<name>.fwd" and
                     "ua.path.<name>.bwd".
    """
    # Ordering invariants:
    # 1. Pipeline validation first (catches sort/rank errors early)
    # 2. merge_names collected from fans BEFORE equation resolution
    #    (determines resolve_equation vs resolve_list_merge dispatch)
    # 3. eq_by_name built during resolution, shared by all composition builders
    # 4. lens_by_name built inside _build_lenses BEFORE lens_paths processing

    validate_pipeline(eq_terms)

    merge_names = _collect_merge_names(fans)
    primitives, eq_by_name = _resolve_all_primitives(eq_terms, backend, merge_names)

    import hydra.core as core
    bound_terms: dict[core.Name, core.Term] = {}

    _register_hyperparams(hyperparams, bound_terms)
    _build_paths(paths, eq_by_name, bound_terms)
    _build_fans(fans, eq_by_name, bound_terms)
    _build_recursion(folds, unfolds, eq_by_name, primitives, bound_terms)
    _build_lenses(lenses, lens_paths, eq_by_name, bound_terms)

    all_sorts = _collect_sorts(eq_terms, extra_sorts)
    return build_graph(all_sorts, primitives=primitives, bound_terms=bound_terms)


# ---------------------------------------------------------------------------
# Hyperparameter rebinding
# ---------------------------------------------------------------------------

def rebind_hyperparams(
    graph: hydra.graph.Graph,
    updates: dict[str, core.Term],
) -> hydra.graph.Graph:
    """Return a new Graph with updated hyperparameter bound_terms.

    Does NOT re-resolve primitives — just swaps the bound_term values.
    Cheap operation (dict copy, not recompilation).

    Args:
        graph:   existing Hydra Graph
        updates: dict of param_name → new scalar Term
                 (keys without "ua.param." prefix — it's added automatically)
    """
    import hydra.core as core
    import hydra.graph as hgraph

    new_terms = dict(graph.bound_terms)
    for key, val in updates.items():
        new_terms[core.Name(f"ua.param.{key}")] = val

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
