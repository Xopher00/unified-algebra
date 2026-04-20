"""DAG validation and graph assembly for equation sets.

Handles topological ordering, sort/rank junction checking, and
Hydra Graph construction from resolved equations.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from unified_algebra.utils import record_fields, string_value
from .sort import sort_type_from_term, check_sort_junction, check_rank_junction, build_graph
from .morphism import resolve_equation
from .composition import path, fan, validate_path, validate_fan
from .recursion import fold, unfold, _unfold_n_primitive, validate_fold, validate_unfold

if TYPE_CHECKING:
    import hydra.core as core
    import hydra.graph
    from .backend import Backend


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
# Helpers
# ---------------------------------------------------------------------------

def _eq_name(eq_term: core.Term) -> str:
    return string_value(record_fields(eq_term)["name"])


def _eq_inputs(eq_term: core.Term) -> list[str]:
    inputs_term = record_fields(eq_term)["inputs"]
    return [string_value(t) for t in inputs_term.value]


def _any_has_inputs(eq_terms: list[core.Term]) -> bool:
    return any(_eq_inputs(eq) for eq in eq_terms)


# ---------------------------------------------------------------------------
# DAG resolution
# ---------------------------------------------------------------------------

def resolve_dag(eq_terms: list[core.Term]) -> list[tuple[core.Term, core.Term, int]]:
    """Return all (upstream, downstream, input_slot) edges in topological order.

    Raises ValueError on cycles.
    """
    by_name = {_eq_name(eq): eq for eq in eq_terms}

    edges = []
    in_degree = {_eq_name(eq): 0 for eq in eq_terms}
    children = {_eq_name(eq): [] for eq in eq_terms}

    for eq in eq_terms:
        name = _eq_name(eq)
        for slot, inp in enumerate(_eq_inputs(eq)):
            if inp in by_name:
                edges.append((by_name[inp], eq, slot))
                children[inp].append(name)
                in_degree[name] += 1

    # Kahn's topological sort
    queue = deque(n for n, d in in_degree.items() if d == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(eq_terms):
        raise ValueError("Cycle detected in equation DAG")

    return edges


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_dag(eq_terms: list[core.Term]) -> None:
    edges = resolve_dag(eq_terms)
    for upstream, downstream, slot in edges:
        check_sort_junction(upstream, downstream)
        check_rank_junction(upstream, downstream, slot)


def validate_pipeline(eq_terms: list[core.Term]) -> None:
    """Check sort and rank junctions across equations.

    Linear mode: when no equation declares inputs, validates consecutive pairs.
    DAG mode: when any equation declares inputs, resolves the full DAG.
    """
    if _any_has_inputs(eq_terms):
        _validate_dag(eq_terms)
    else:
        for upstream, downstream in zip(eq_terms, eq_terms[1:]):
            check_sort_junction(upstream, downstream)


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def ua_primitives(
    eq_terms: list[core.Term],
    backend: Backend,
) -> dict[core.Name, hydra.graph.Primitive]:
    """Resolve all equation terms into Hydra Primitives.

    Follows the same pattern as hydra.sources.libraries.standard_library():
    returns a dict of Name → Primitive, separate from graph assembly.
    """
    primitives = {}
    for eq_term in eq_terms:
        prim = resolve_equation(eq_term, backend)
        primitives[prim.name] = prim
    return primitives


def assemble_graph(
    eq_terms: list[core.Term],
    backend: Backend,
    extra_sorts: list[core.Term] | None = None,
    paths: list[tuple[str, list[str], core.Term, core.Term]] | None = None,
    fans: list[tuple[str, list[str], str, core.Term, core.Term]] | None = None,
    folds: list[tuple[str, str, core.Term, core.Term, core.Term]] | None = None,
    unfolds: list[tuple[str, str, int, core.Term, core.Term]] | None = None,
) -> hydra.graph.Graph:
    """Resolve equation terms and assemble a Hydra Graph.

    Validates sort/rank junctions, resolves each equation into a Primitive,
    builds lambda terms for paths/fans/folds/unfolds, collects all sorts,
    and builds the Graph with Hydra's standard library included.

    Args:
        eq_terms:    list of equation record terms
        backend:     Backend providing op implementations
        extra_sorts: additional sort terms to register
        paths:       list of (name, eq_names, domain_sort, codomain_sort)
        fans:        list of (name, branch_names, merge_name, domain_sort, codomain_sort)
        folds:       list of (name, step_name, init_term, domain_sort, state_sort)
        unfolds:     list of (name, step_name, n_steps, domain_sort, state_sort)
    """
    validate_pipeline(eq_terms)

    # Assemble primitives: Hydra standard library + UA equations
    from hydra.sources.libraries import standard_library
    primitives = dict(standard_library())
    primitives.update(ua_primitives(eq_terms, backend))

    # Build equation lookup for composition validation
    eq_by_name = {_eq_name(eq): eq for eq in eq_terms}

    # Build bound_terms for compositions
    bound_terms: dict[core.Name, core.Term] = {}

    if paths:
        for (pname, eq_names, domain_sort, codomain_sort, *rest) in paths:
            params = rest[0] if rest else None
            validate_path(eq_by_name, eq_names, domain_sort, codomain_sort)
            term_name, term = path(pname, eq_names, domain_sort, codomain_sort, params)
            bound_terms[term_name] = term

    if fans:
        for (fname, branch_names, merge_name, domain_sort, codomain_sort) in fans:
            validate_fan(eq_by_name, branch_names, merge_name, domain_sort, codomain_sort)
            term_name, term = fan(fname, branch_names, merge_name, domain_sort, codomain_sort)
            bound_terms[term_name] = term

    if unfolds:
        unfold_prim = _unfold_n_primitive()
        primitives[unfold_prim.name] = unfold_prim

    if folds:
        for (fname, step_name, init_term, domain_sort, state_sort) in folds:
            validate_fold(eq_by_name, step_name, domain_sort, state_sort)
            term_name, term = fold(fname, step_name, init_term, domain_sort, state_sort)
            bound_terms[term_name] = term

    if unfolds:
        for (uname, step_name, n_steps, domain_sort, state_sort) in unfolds:
            validate_unfold(eq_by_name, step_name, domain_sort, state_sort)
            term_name, term = unfold(uname, step_name, n_steps, domain_sort, state_sort)
            bound_terms[term_name] = term

    seen_sorts: dict[str, core.Term] = {}
    for eq_term in eq_terms:
        fields = record_fields(eq_term)
        for key in ("domainSort", "codomainSort"):
            st = fields[key]
            type_key = sort_type_from_term(st).value.value
            seen_sorts.setdefault(type_key, st)

    all_sorts = list(seen_sorts.values()) + list(extra_sorts or [])
    return build_graph(all_sorts, primitives=primitives, bound_terms=bound_terms)
