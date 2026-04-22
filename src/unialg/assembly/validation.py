"""Composition spec validation via Hydra constraint unification."""

from __future__ import annotations

from hydra.typing import TypeConstraint

import unialg.views as vw
import unialg.algebra as alg
import unialg.specs as sp
from unialg.assembly.topology import _build_schema, _unify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_eq(eq_by_name, name, label):
    if name not in eq_by_name:
        raise ValueError(f"{label} equation '{name}' not found")


def _eq_sort_type(eq_by_name, eq_name, field):
    """Sort type from an equation's domain or codomain field."""
    v = vw.EquationView(eq_by_name[eq_name])
    return alg.sort_type_from_term(v.domain_sort if field == "d" else v.codomain_sort)


# ---------------------------------------------------------------------------
# Composition validation — constraint producers
# ---------------------------------------------------------------------------

def _path_cs(eq, spec):
    cs = []
    if spec.domain_sort is not None:
        cs.append(TypeConstraint(_eq_sort_type(eq, spec.eq_names[0], "d"), alg.sort_type_from_term(spec.domain_sort),
                                 f"Path domain != '{spec.eq_names[0]}' domain"))
    for a, b in zip(spec.eq_names, spec.eq_names[1:]):
        cs.append(TypeConstraint(_eq_sort_type(eq, a, "c"), _eq_sort_type(eq, b, "d"), f"'{a}' codomain != '{b}' domain"))
    if spec.codomain_sort is not None:
        cs.append(TypeConstraint(_eq_sort_type(eq, spec.eq_names[-1], "c"), alg.sort_type_from_term(spec.codomain_sort),
                                 f"Path codomain != '{spec.eq_names[-1]}' codomain"))
    return cs

def _fan_cs(eq, spec):
    cs = []
    md = alg.sort_type_from_term(vw.EquationView(eq[spec.merge_name]).domain_sort)
    if spec.domain_sort is not None:
        for b in spec.branch_names:
            cs.append(TypeConstraint(_eq_sort_type(eq, b, "d"), alg.sort_type_from_term(spec.domain_sort), f"Fan branch '{b}' domain mismatch"))
    for b in spec.branch_names:
        cs.append(TypeConstraint(_eq_sort_type(eq, b, "c"), md, f"Fan branch '{b}' codomain != merge domain"))
    if spec.codomain_sort is not None:
        cs.append(TypeConstraint(_eq_sort_type(eq, spec.merge_name, "c"), alg.sort_type_from_term(spec.codomain_sort), f"Fan merge codomain mismatch"))
    return cs

def _fold_cs(eq, spec):
    _require_eq(eq, spec.step_name, "Fold step")
    return [TypeConstraint(_eq_sort_type(eq, spec.step_name, "c"), alg.sort_type_from_term(spec.state_sort), f"Fold step codomain != state sort")]

def _unfold_cs(eq, spec):
    _require_eq(eq, spec.step_name, "Unfold step")
    ds = alg.sort_type_from_term(spec.domain_sort)
    return [TypeConstraint(_eq_sort_type(eq, spec.step_name, "d"), ds, f"Unfold step domain != state sort"),
            TypeConstraint(_eq_sort_type(eq, spec.step_name, "c"), ds, f"Unfold step codomain != state sort")]

def _fixpoint_cs(eq, spec):
    _require_eq(eq, spec.step_name, "Fixpoint step")
    _require_eq(eq, spec.predicate_name, "Fixpoint predicate")
    ds = alg.sort_type_from_term(spec.domain_sort)
    return [TypeConstraint(_eq_sort_type(eq, spec.step_name, "d"), ds, f"Fixpoint step domain != state sort"),
            TypeConstraint(_eq_sort_type(eq, spec.step_name, "c"), ds, f"Fixpoint step codomain != state sort"),
            TypeConstraint(_eq_sort_type(eq, spec.predicate_name, "d"), ds, f"Fixpoint predicate domain != state sort")]


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
    builders = {sp.PathSpec: _path_cs, sp.FanSpec: _fan_cs, sp.FoldSpec: _fold_cs,
                sp.UnfoldSpec: _unfold_cs, sp.FixpointSpec: _fixpoint_cs}
    builder = builders.get(type(spec))
    if builder is None:
        raise TypeError(f"Unknown spec type: {type(spec).__name__}")
    cs = builder(eq_by_name, spec)
    if not cs:
        return
    if schema_types is None:
        schema_types = _build_schema(list(eq_by_name.values()), _spec_sorts(spec))
    _unify(cs, schema_types)
