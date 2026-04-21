"""DAG validation and pipeline checking for equation sets.

Handles topological ordering, sort/rank junction checking, and
resolving equation sets into Hydra Primitives.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from .views import EquationView
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
    return EquationView(eq_term).inputs


def _any_has_inputs(eq_terms: list[core.Term]) -> bool:
    return any(_eq_inputs(eq) for eq in eq_terms)


# ---------------------------------------------------------------------------
# DAG resolution
# ---------------------------------------------------------------------------

def resolve_dag(eq_terms: list[core.Term]) -> list[tuple[core.Term, core.Term, int]]:
    """Return all (upstream, downstream, input_slot) edges in topological order.

    Raises ValueError on cycles.
    """
    by_name = {EquationView(eq).name: eq for eq in eq_terms}

    edges = []
    in_degree = {EquationView(eq).name: 0 for eq in eq_terms}
    children = {EquationView(eq).name: [] for eq in eq_terms}

    for eq in eq_terms:
        name = EquationView(eq).name
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
# Pipeline validation
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
# Composition validation — validate_spec owns the logic
# ---------------------------------------------------------------------------

def _require_eq(eq_terms_by_name, name, label):
    """Raise ValueError if an equation name is missing."""
    if name not in eq_terms_by_name:
        raise ValueError(f"{label} equation '{name}' not found")


def _check_endomorphism(eq_terms_by_name, name, sort, label):
    """Check that an equation's domain and codomain both match a sort."""
    _check_sort(eq_terms_by_name, name, "domainSort", sort,
                f"{label} '{name}' domain != state sort")
    _check_sort(eq_terms_by_name, name, "codomainSort", sort,
                f"{label} '{name}' codomain != state sort")


def validate_spec(eq_terms_by_name, spec):
    """Validate sort junctions for a composition spec."""
    from .graph import PathSpec, FanSpec, FoldSpec, UnfoldSpec, FixpointSpec

    match spec:
        case PathSpec(eq_names=eq_names, domain_sort=ds, codomain_sort=cs):
            _check_sort(eq_terms_by_name, eq_names[0], "domainSort", ds,
                        f"Path domain != first equation '{eq_names[0]}' domain")
            for i in range(len(eq_names) - 1):
                check_sort_junction(eq_terms_by_name[eq_names[i]], eq_terms_by_name[eq_names[i + 1]])
            _check_sort(eq_terms_by_name, eq_names[-1], "codomainSort", cs,
                        f"Path codomain != last equation '{eq_names[-1]}' codomain")

        case FanSpec(branch_names=bns, merge_name=mn, domain_sort=ds, codomain_sort=cs):
            for bname in bns:
                _check_sort(eq_terms_by_name, bname, "domainSort", ds,
                            f"Fan branch '{bname}' domain != fan domain")
            merge_eq = EquationView(eq_terms_by_name[mn])
            for bname in bns:
                _check_sort(eq_terms_by_name, bname, "codomainSort", merge_eq.domain_sort,
                            f"Fan branch '{bname}' codomain != merge '{mn}' domain")
            _check_sort(eq_terms_by_name, mn, "codomainSort", cs,
                        f"Fan merge '{mn}' codomain != fan codomain")

        case FoldSpec(step_name=sn, state_sort=ss):
            _require_eq(eq_terms_by_name, sn, "Fold step")
            _check_sort(eq_terms_by_name, sn, "codomainSort", ss,
                        f"Fold step '{sn}' codomain != state sort")

        case UnfoldSpec(step_name=sn, domain_sort=ds):
            _require_eq(eq_terms_by_name, sn, "Unfold step")
            _check_endomorphism(eq_terms_by_name, sn, ds, "Unfold step")

        case FixpointSpec(step_name=sn, predicate_name=pn, domain_sort=ds):
            _require_eq(eq_terms_by_name, sn, "Fixpoint step")
            _require_eq(eq_terms_by_name, pn, "Fixpoint predicate")
            _check_endomorphism(eq_terms_by_name, sn, ds, "Fixpoint step")
            _check_sort(eq_terms_by_name, pn, "domainSort", ds,
                        f"Fixpoint predicate '{pn}' domain != state sort")

        case _:
            raise TypeError(f"Unknown spec type: {type(spec).__name__}")


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
