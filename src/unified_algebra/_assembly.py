"""Private sub-functions for assemble_graph (graph.py orchestrator).

This module is not part of the public API — it is imported only by graph.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .views import EquationView, LensView
from .sort import sort_type_from_term
from .morphism import resolve_equation, resolve_list_merge
from .specs import PathSpec, FanSpec, FoldSpec, UnfoldSpec, LensPathSpec, FixpointSpec
from .composition import path, fan, validate_lens, lens_path
from .recursion import fold, unfold, _unfold_n_primitive, fixpoint, _fixpoint_primitive
from ._lens_threading import _lens_fwd_primitive, _lens_bwd_primitive
from .validation import validate_spec

if TYPE_CHECKING:
    import hydra.core as core
    from .backend import Backend



# ---------------------------------------------------------------------------
# Primitive resolution and hyperparameter registration
# ---------------------------------------------------------------------------

def _resolve_all_primitives(
    eq_terms: list[core.Term],
    backend: Backend,
    merge_names: set[str],
) -> tuple[dict, dict[str, core.Term]]:
    """Resolve equations to primitives and build eq_by_name lookup.

    Fan merges get list-merge resolution; all others get standard resolution.
    Returns (primitives_dict, eq_by_name_dict).
    The primitives_dict is pre-populated with Hydra's standard_library().
    """
    from hydra.sources.libraries import standard_library
    primitives = dict(standard_library())

    eq_by_name: dict[str, core.Term] = {EquationView(eq).name: eq for eq in eq_terms}
    for eq_term in eq_terms:
        name = EquationView(eq_term).name
        if name in merge_names:
            prim = resolve_list_merge(eq_term, backend)
        else:
            prim = resolve_equation(eq_term, backend)
        primitives[prim.name] = prim

    return primitives, eq_by_name


def _register_hyperparams(
    hyperparams: dict[str, core.Term] | None,
    bound_terms: dict,
) -> None:
    """Register hyperparameters as bound_terms. Mutates bound_terms."""
    import hydra.core as core
    if hyperparams:
        for param_name, param_term in hyperparams.items():
            bound_terms[core.Name(f"ua.param.{param_name}")] = param_term


# ---------------------------------------------------------------------------
# Lens validation
# ---------------------------------------------------------------------------

def _build_lens_by_name(
    lenses: list[core.Term] | None,
    eq_by_name: dict[str, core.Term],
) -> dict[str, core.Term]:
    """Validate lenses and build lens_by_name lookup."""
    lens_by_name: dict[str, core.Term] = {}
    if lenses:
        for lens_term in lenses:
            lens_by_name[LensView(lens_term).name] = lens_term
            validate_lens(eq_by_name, lens_term)
    return lens_by_name


# ---------------------------------------------------------------------------
# Composition dispatch
# ---------------------------------------------------------------------------

def _build_compositions(
    specs: list,
    eq_by_name: dict[str, core.Term],
    primitives: dict,
    bound_terms: dict,
    lens_by_name: dict[str, core.Term],
    schema_types,
) -> None:
    """Validate and build all composition specs. Mutates primitives and bound_terms."""
    for spec in specs:
        # LensPathSpec — lens validation handled by _build_lens_by_name
        if not isinstance(spec, LensPathSpec):
            validate_spec(eq_by_name, spec, schema_types)

        if isinstance(spec, PathSpec):
            results = [path(spec.name, spec.eq_names, spec.params)]

        elif isinstance(spec, FanSpec):
            results = [fan(spec.name, spec.branch_names, spec.merge_name)]

        elif isinstance(spec, FoldSpec):
            results = [fold(spec.name, spec.step_name, spec.init_term)]

        elif isinstance(spec, UnfoldSpec):
            unfold_prim = _unfold_n_primitive
            primitives.setdefault(unfold_prim.name, unfold_prim)
            results = [unfold(spec.name, spec.step_name, spec.n_steps)]

        elif isinstance(spec, LensPathSpec):
            if any(LensView(lens_by_name[ln]).residual_sort is not None for ln in spec.lens_names):
                for prim in (_lens_fwd_primitive(), _lens_bwd_primitive()):
                    primitives.setdefault(prim.name, prim)
            (fwd, bwd) = lens_path(spec.name, spec.lens_names, lens_by_name, spec.params)
            results = [fwd, bwd]

        elif isinstance(spec, FixpointSpec):
            fp_prim = _fixpoint_primitive(spec.epsilon, spec.max_iter)
            primitives[fp_prim.name] = fp_prim
            results = [fixpoint(spec.name, spec.step_name, spec.predicate_name)]

        else:
            raise TypeError(f"Unknown composition spec type: {type(spec).__name__}")

        for name, term in results:
            bound_terms[name] = term


# ---------------------------------------------------------------------------
# Sort collection
# ---------------------------------------------------------------------------

def _collect_sorts(
    eq_terms: list[core.Term],
    extra_sorts: list[core.Term] | None,
) -> list[core.Term]:
    """Collect all unique sorts from equations + extras.

    Deduplicates by TypeVariable name; extras are appended after equation sorts.
    """
    seen_sorts: dict[str, core.Term] = {}
    for eq_term in eq_terms:
        v = EquationView(eq_term)
        for st in (v.domain_sort, v.codomain_sort):
            seen_sorts.setdefault(str(sort_type_from_term(st)), st)
    return list(seen_sorts.values()) + list(extra_sorts or [])
