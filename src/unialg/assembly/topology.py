"""DAG resolution and pipeline validation for equation sets."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import hydra.core as core
from hydra.context import Context
from hydra.dsl.python import FrozenDict, Nothing, Left
from hydra.typing import TypeConstraint
from hydra.unification import unify_type_constraints

import unialg.algebra as alg
from unialg.algebra.sort import Sort, ProductSort
from unialg.resolve.morphism import Equation

if TYPE_CHECKING:
    import hydra.graph
    from unialg.backend import Backend

_CX = Context(trace=(), messages=(), other=FrozenDict({}))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_sort_components(sort_term, schema):
    """Register structural sort type component names into schema (mutates schema)."""
    if isinstance(Sort.from_term(sort_term), ProductSort):
        for elem in ProductSort.from_term(sort_term).elements:
            _register_sort_components(elem, schema)
        return
    sv = Sort.from_term(sort_term)
    sn = core.Name(f"ua.sort.{sv.name}")
    schema[sn] = core.TypeScheme((), core.TypeVariable(sn), Nothing())
    srn = core.Name(f"ua.semiring.{sv.semiring_name}")
    schema[srn] = core.TypeScheme((), core.TypeVariable(srn), Nothing())
    bn = core.Name("ua.batched")
    schema[bn] = core.TypeScheme((), core.TypeVariable(bn), Nothing())


def _build_schema(eq_terms, extra_sorts=()):
    """Build schema_types registering sort component names as ground types.

    Registers ua.sort.<name>, ua.semiring.<name>, and ua.batched as ground
    TypeVariables so the unifier treats them as concrete (non-free) types.
    extra_sorts allows callers to include sorts referenced in spec annotations
    that may not appear in any equation.
    """
    schema = {}
    for eq in eq_terms:
        v = Equation.from_term(eq)
        for st in (v.domain_sort, v.codomain_sort):
            _register_sort_components(st, schema)
    for st in extra_sorts:
        if st is not None:
            _register_sort_components(st, schema)
    return FrozenDict(schema)


def _unify(constraints, schema):
    """Run unification. Raise TypeError on failure."""
    if constraints:
        result = unify_type_constraints(_CX, schema, tuple(constraints))
        if isinstance(result, Left):
            raise TypeError(result.value.message)


# ---------------------------------------------------------------------------
# DAG resolution
# ---------------------------------------------------------------------------

def topo_edges(eq_terms):
    """Return (upstream, downstream, slot) edges in topological order."""
    by_name = {Equation.from_term(eq).name: eq for eq in eq_terms}
    edges, in_degree, children = [], {}, {}
    for eq in eq_terms:
        n = Equation.from_term(eq).name
        in_degree[n] = 0
        children[n] = []
    for eq in eq_terms:
        eq_obj = Equation.from_term(eq)
        n = eq_obj.name
        for slot, inp in enumerate(eq_obj.inputs):
            if inp in by_name:
                edges.append((by_name[inp], eq, slot))
                children[inp].append(n)
                in_degree[n] += 1
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

def validate_pipeline(eq_terms, schema_types=None):
    """Check sort and rank junctions across equations. Requires inputs= wiring."""
    if schema_types is None:
        schema_types = _build_schema(eq_terms)
    cs = []
    for up, down, slot in topo_edges(eq_terms):
        u, d = Equation.from_term(up), Equation.from_term(down)
        cs.append(TypeConstraint(Sort.from_term(u.codomain_sort).type_,
                                 Sort.from_term(d.domain_sort).type_,
                                 f"'{u.name}' codomain != '{d.name}' domain"))
        alg.check_rank_junction(up, down, slot)
    _unify(cs, schema_types)


def validate_spec(eq_by_name: dict, spec, schema_types=None) -> None:
    """Validate sort junctions for a composition spec.

    Delegates constraint generation to spec.constraints(eq_by_name), then
    unifies against the schema. Specs with no sort constraints (LensPathSpec,
    LensFanSpec) return [] from constraints() and are silently skipped.
    """
    cs = spec.constraints(eq_by_name)
    if not cs:
        return
    if schema_types is None:
        schema_types = _build_schema(list(eq_by_name.values()), spec.sort_terms())
    _unify(cs, schema_types)
