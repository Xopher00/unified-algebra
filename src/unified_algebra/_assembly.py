"""Private sub-functions for assemble_graph (graph.py orchestrator).

This module is not part of the public API — it is imported only by graph.py.
Each function handles one concern in the graph assembly pipeline:

    _collect_merge_names   — which equations need list-merge resolution
    _resolve_all_primitives — resolve equations → Hydra Primitives
    _register_hyperparams  — inject scalar hyperparams as bound_terms
    _build_paths           — validate + build sequential path lambda terms
    _build_fans            — validate + build parallel fan lambda terms
    _build_recursion       — validate + build fold/unfold lambda terms
    _build_lenses          — validate + build bidirectional lens-path terms
    _collect_sorts         — gather all unique sorts from equations + extras
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unified_algebra.utils import record_fields, string_value
from .sort import sort_type_from_term
from .morphism import resolve_equation, resolve_list_merge
from .composition import path, fan, validate_path, validate_fan
from .recursion import fold, unfold, _unfold_n_primitive, validate_fold, validate_unfold
from .lens import validate_lens, lens_path
from .validation import _eq_name

if TYPE_CHECKING:
    import hydra.core as core
    from .backend import Backend


# ---------------------------------------------------------------------------
# Sub-functions
# ---------------------------------------------------------------------------

def _collect_merge_names(fans: list[tuple] | None) -> set[str]:
    """Collect equation names used as fan merges (need list-merge resolution)."""
    merge_names: set[str] = set()
    if fans:
        for (_, _, merge_name, _, _) in fans:
            merge_names.add(merge_name)
    return merge_names


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

    eq_by_name: dict[str, core.Term] = {_eq_name(eq): eq for eq in eq_terms}
    for eq_term in eq_terms:
        name = _eq_name(eq_term)
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


def _build_paths(
    paths: list[tuple] | None,
    eq_by_name: dict[str, core.Term],
    bound_terms: dict,
) -> None:
    """Validate and build path lambda terms. Mutates bound_terms."""
    if paths:
        for (pname, eq_names, domain_sort, codomain_sort, *rest) in paths:
            params = rest[0] if rest else None
            validate_path(eq_by_name, eq_names, domain_sort, codomain_sort)
            term_name, term = path(pname, eq_names, domain_sort, codomain_sort, params)
            bound_terms[term_name] = term


def _build_fans(
    fans: list[tuple] | None,
    eq_by_name: dict[str, core.Term],
    bound_terms: dict,
) -> None:
    """Validate and build fan lambda terms. Mutates bound_terms."""
    if fans:
        for (fname, branch_names, merge_name, domain_sort, codomain_sort) in fans:
            validate_fan(eq_by_name, branch_names, merge_name, domain_sort, codomain_sort)
            term_name, term = fan(fname, branch_names, merge_name, domain_sort, codomain_sort)
            bound_terms[term_name] = term


def _build_recursion(
    folds: list[tuple] | None,
    unfolds: list[tuple] | None,
    eq_by_name: dict[str, core.Term],
    primitives: dict,
    bound_terms: dict,
) -> None:
    """Validate and build fold/unfold terms. Mutates primitives and bound_terms.

    The unfold_n primitive must be registered before any unfold terms are built,
    so primitives registration happens here before the per-unfold loop.
    """
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


def _build_lenses(
    lenses: list[core.Term] | None,
    lens_paths_specs: list[tuple] | None,
    eq_by_name: dict[str, core.Term],
    bound_terms: dict,
) -> None:
    """Validate lenses and build bidirectional path terms. Mutates bound_terms.

    Builds lens_by_name lookup internally before processing lens_paths_specs.
    """
    import hydra.core as core
    lens_by_name: dict[str, core.Term] = {}
    if lenses:
        for lens_term in lenses:
            fields = record_fields(lens_term)
            lname = string_value(fields["name"])
            lens_by_name[lname] = lens_term
            validate_lens(eq_by_name, lens_term)

    if lens_paths_specs:
        for (lpname, lp_lens_names, domain_sort, codomain_sort, *rest) in lens_paths_specs:
            lp_params = rest[0] if rest else None
            (fwd_name, fwd_term), (bwd_name, bwd_term) = lens_path(
                lpname, lp_lens_names, lens_by_name, domain_sort, codomain_sort, lp_params
            )
            bound_terms[fwd_name] = fwd_term
            bound_terms[bwd_name] = bwd_term


def _collect_sorts(
    eq_terms: list[core.Term],
    extra_sorts: list[core.Term] | None,
) -> list[core.Term]:
    """Collect all unique sorts from equations + extras.

    Deduplicates by TypeVariable name; extras are appended after equation sorts.
    """
    seen_sorts: dict[str, core.Term] = {}
    for eq_term in eq_terms:
        fields = record_fields(eq_term)
        for key in ("domainSort", "codomainSort"):
            st = fields[key]
            type_key = sort_type_from_term(st).value.value
            seen_sorts.setdefault(type_key, st)
    return list(seen_sorts.values()) + list(extra_sorts or [])
