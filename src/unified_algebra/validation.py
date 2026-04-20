"""DAG validation and pipeline checking for equation sets.

Handles topological ordering, sort/rank junction checking, and
resolving equation sets into Hydra Primitives.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from .utils import record_fields, string_value, eq_name
from .sort import check_sort_junction, check_rank_junction, _check_sort
from .morphism import resolve_equation, resolve_list_merge

if TYPE_CHECKING:
    import hydra.core as core
    import hydra.graph
    from .backend import Backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




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
    by_name = {eq_name(eq): eq for eq in eq_terms}

    edges = []
    in_degree = {eq_name(eq): 0 for eq in eq_terms}
    children = {eq_name(eq): [] for eq in eq_terms}

    for eq in eq_terms:
        name = eq_name(eq)
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

def validate_path(
    eq_terms_by_name: dict[str, "core.Term"],
    eq_names: list[str],
    domain_sort: "core.Term",
    codomain_sort: "core.Term",
) -> None:
    """Validate sort junctions along a path."""
    _check_sort(eq_terms_by_name, eq_names[0], "domainSort", domain_sort,
                f"Path domain != first equation '{eq_names[0]}' domain")
    for i in range(len(eq_names) - 1):
        check_sort_junction(eq_terms_by_name[eq_names[i]], eq_terms_by_name[eq_names[i + 1]])
    _check_sort(eq_terms_by_name, eq_names[-1], "codomainSort", codomain_sort,
                f"Path codomain != last equation '{eq_names[-1]}' codomain")


def validate_fan(
    eq_terms_by_name: dict[str, "core.Term"],
    branch_names: list[str],
    merge_name: str,
    domain_sort: "core.Term",
    codomain_sort: "core.Term",
) -> None:
    """Validate sort junctions in a fan."""
    for bname in branch_names:
        _check_sort(eq_terms_by_name, bname, "domainSort", domain_sort,
                    f"Fan branch '{bname}' domain != fan domain")
    merge_fields = record_fields(eq_terms_by_name[merge_name])
    for bname in branch_names:
        _check_sort(eq_terms_by_name, bname, "codomainSort", merge_fields["domainSort"],
                    f"Fan branch '{bname}' codomain != merge '{merge_name}' domain")
    _check_sort(eq_terms_by_name, merge_name, "codomainSort", codomain_sort,
                f"Fan merge '{merge_name}' codomain != fan codomain")


def validate_fold(
    eq_terms_by_name: dict[str, "core.Term"],
    step_name: str,
    domain_sort: "core.Term",
    state_sort: "core.Term",
) -> None:
    """Validate sort junctions for a fold."""
    if step_name not in eq_terms_by_name:
        raise ValueError(f"Fold step equation '{step_name}' not found")
    _check_sort(eq_terms_by_name, step_name, "codomainSort", state_sort,
                f"Fold step '{step_name}' codomain != state sort")


def validate_unfold(
    eq_terms_by_name: dict[str, "core.Term"],
    step_name: str,
    domain_sort: "core.Term",
    state_sort: "core.Term",
) -> None:
    """Validate sort junctions for an unfold."""
    if step_name not in eq_terms_by_name:
        raise ValueError(f"Unfold step equation '{step_name}' not found")
    _check_sort(eq_terms_by_name, step_name, "domainSort", domain_sort,
                f"Unfold step '{step_name}' domain != state sort")
    _check_sort(eq_terms_by_name, step_name, "codomainSort", domain_sort,
                f"Unfold step '{step_name}' codomain != state sort")


def validate_fixpoint(
    eq_terms_by_name: dict[str, "core.Term"],
    step_name: str,
    predicate_name: str,
    domain_sort: "core.Term",
) -> None:
    """Validate sort junctions for a fixpoint iteration."""
    if step_name not in eq_terms_by_name:
        raise ValueError(f"Fixpoint step equation '{step_name}' not found")
    if predicate_name not in eq_terms_by_name:
        raise ValueError(f"Fixpoint predicate equation '{predicate_name}' not found")
    _check_sort(eq_terms_by_name, step_name, "domainSort", domain_sort,
                f"Fixpoint step '{step_name}' domain != state sort")
    _check_sort(eq_terms_by_name, step_name, "codomainSort", domain_sort,
                f"Fixpoint step '{step_name}' codomain != state sort")
    _check_sort(eq_terms_by_name, predicate_name, "domainSort", domain_sort,
                f"Fixpoint predicate '{predicate_name}' domain != state sort")


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
