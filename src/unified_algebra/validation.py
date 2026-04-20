"""DAG validation and pipeline checking for equation sets.

Handles topological ordering, sort/rank junction checking, and
resolving equation sets into Hydra Primitives.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from unified_algebra.utils import record_fields, string_value
from .sort import check_sort_junction, check_rank_junction
from .morphism import resolve_equation, resolve_list_merge

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
# Primitive resolution
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
