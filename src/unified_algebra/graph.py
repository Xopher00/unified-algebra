"""DAG validation and graph assembly for equation sets.

Handles topological ordering, sort/rank junction checking, and
Hydra Graph construction from resolved equations.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from unified_algebra._hydra_setup import record_fields, string_value
from .sort import sort_type_from_term, check_sort_junction, check_rank_junction, build_graph
from .morphism import resolve_equation

if TYPE_CHECKING:
    import hydra.core as core
    import hydra.graph
    from .backend import Backend


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

def assemble_graph(
    eq_terms: list[core.Term],
    backend: Backend,
    extra_sorts: list[core.Term] | None = None,
) -> hydra.graph.Graph:
    """Resolve equation terms and assemble a Hydra Graph.

    Validates sort/rank junctions, resolves each equation into a Primitive,
    collects all referenced sorts, and builds the Graph.
    """
    validate_pipeline(eq_terms)

    primitives = {}
    for eq_term in eq_terms:
        prim = resolve_equation(eq_term, backend)
        primitives[prim.name] = prim

    seen_sorts: dict[str, core.Term] = {}
    for eq_term in eq_terms:
        fields = record_fields(eq_term)
        for key in ("domainSort", "codomainSort"):
            st = fields[key]
            type_key = sort_type_from_term(st).value.value
            seen_sorts.setdefault(type_key, st)

    all_sorts = list(seen_sorts.values()) + list(extra_sorts or [])
    return build_graph(all_sorts, primitives=primitives)
