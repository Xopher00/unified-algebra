"""DAG validation and pipeline checking for equation sets."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import hydra.core as core
from hydra.context import Context
from hydra.dsl.python import FrozenDict, Nothing, Left
from hydra.typing import TypeConstraint
from hydra.unification import unify_type_constraints

from .views import EquationView, SortView
from .sort import check_rank_junction, sort_type_from_term, is_product_sort
from .morphism import resolve_equation, resolve_list_merge

if TYPE_CHECKING:
    import hydra.graph
    from .backend import Backend

_CX = Context(trace=(), messages=(), other=FrozenDict({}))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_eq(eq_by_name, name, label):
    if name not in eq_by_name:
        raise ValueError(f"{label} equation '{name}' not found")


def _st(eq_by_name, eq_name, field):
    """Sort type from an equation's domain or codomain field."""
    v = EquationView(eq_by_name[eq_name])
    return sort_type_from_term(v.domain_sort if field == "d" else v.codomain_sort)


def _register_sort_components(sort_term, schema):
    """Register structural sort type component names into schema (mutates schema)."""
    if is_product_sort(sort_term):
        from .sort import product_sort_elements
        for elem in product_sort_elements(sort_term):
            _register_sort_components(elem, schema)
        return
    sv = SortView(sort_term)
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
        v = EquationView(eq)
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

def resolve_dag(eq_terms):
    """Return (upstream, downstream, slot) edges in topological order."""
    by_name = {EquationView(eq).name: eq for eq in eq_terms}
    edges, in_degree, children = [], {}, {}
    for eq in eq_terms:
        n = EquationView(eq).name
        in_degree[n] = 0
        children[n] = []
    for eq in eq_terms:
        n = EquationView(eq).name
        for slot, inp in enumerate(EquationView(eq).inputs):
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
    for up, down, slot in resolve_dag(eq_terms):
        u, d = EquationView(up), EquationView(down)
        cs.append(TypeConstraint(sort_type_from_term(u.codomain_sort),
                                 sort_type_from_term(d.domain_sort),
                                 f"'{u.name}' codomain != '{d.name}' domain"))
        check_rank_junction(up, down, slot)
    _unify(cs, schema_types)


# ---------------------------------------------------------------------------
# Composition validation — constraint producers
# ---------------------------------------------------------------------------

def _path_cs(eq, spec):
    cs = []
    if spec.domain_sort is not None:
        cs.append(TypeConstraint(_st(eq, spec.eq_names[0], "d"), sort_type_from_term(spec.domain_sort),
                                 f"Path domain != '{spec.eq_names[0]}' domain"))
    for a, b in zip(spec.eq_names, spec.eq_names[1:]):
        cs.append(TypeConstraint(_st(eq, a, "c"), _st(eq, b, "d"), f"'{a}' codomain != '{b}' domain"))
    if spec.codomain_sort is not None:
        cs.append(TypeConstraint(_st(eq, spec.eq_names[-1], "c"), sort_type_from_term(spec.codomain_sort),
                                 f"Path codomain != '{spec.eq_names[-1]}' codomain"))
    return cs

def _fan_cs(eq, spec):
    cs = []
    md = sort_type_from_term(EquationView(eq[spec.merge_name]).domain_sort)
    if spec.domain_sort is not None:
        for b in spec.branch_names:
            cs.append(TypeConstraint(_st(eq, b, "d"), sort_type_from_term(spec.domain_sort), f"Fan branch '{b}' domain mismatch"))
    for b in spec.branch_names:
        cs.append(TypeConstraint(_st(eq, b, "c"), md, f"Fan branch '{b}' codomain != merge domain"))
    if spec.codomain_sort is not None:
        cs.append(TypeConstraint(_st(eq, spec.merge_name, "c"), sort_type_from_term(spec.codomain_sort), f"Fan merge codomain mismatch"))
    return cs

def _fold_cs(eq, spec):
    _require_eq(eq, spec.step_name, "Fold step")
    return [TypeConstraint(_st(eq, spec.step_name, "c"), sort_type_from_term(spec.state_sort), f"Fold step codomain != state sort")]

def _unfold_cs(eq, spec):
    _require_eq(eq, spec.step_name, "Unfold step")
    ds = sort_type_from_term(spec.domain_sort)
    return [TypeConstraint(_st(eq, spec.step_name, "d"), ds, f"Unfold step domain != state sort"),
            TypeConstraint(_st(eq, spec.step_name, "c"), ds, f"Unfold step codomain != state sort")]

def _fixpoint_cs(eq, spec):
    _require_eq(eq, spec.step_name, "Fixpoint step")
    _require_eq(eq, spec.predicate_name, "Fixpoint predicate")
    ds = sort_type_from_term(spec.domain_sort)
    return [TypeConstraint(_st(eq, spec.step_name, "d"), ds, f"Fixpoint step domain != state sort"),
            TypeConstraint(_st(eq, spec.step_name, "c"), ds, f"Fixpoint step codomain != state sort"),
            TypeConstraint(_st(eq, spec.predicate_name, "d"), ds, f"Fixpoint predicate domain != state sort")]


def _spec_sorts(spec):
    """Collect sort terms declared on a spec (domain_sort, codomain_sort, state_sort)."""
    sorts = []
    for attr in ("domain_sort", "codomain_sort", "state_sort"):
        val = getattr(spec, attr, None)
        if val is not None:
            sorts.append(val)
    return sorts


def validate_spec(eq_by_name, spec, schema_types=None):
    """Validate sort junctions via Hydra constraint unification."""
    from .graph import PathSpec, FanSpec, FoldSpec, UnfoldSpec, FixpointSpec
    builders = {PathSpec: _path_cs, FanSpec: _fan_cs, FoldSpec: _fold_cs,
                UnfoldSpec: _unfold_cs, FixpointSpec: _fixpoint_cs}
    builder = builders.get(type(spec))
    if builder is None:
        raise TypeError(f"Unknown spec type: {type(spec).__name__}")
    cs = builder(eq_by_name, spec)
    if not cs:
        return
    if schema_types is None:
        schema_types = _build_schema(list(eq_by_name.values()), _spec_sorts(spec))
    _unify(cs, schema_types)


