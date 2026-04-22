"""Lenses: bidirectional morphisms pairing forward and backward equations.

    lens()          -> Hydra record term (declaration)
    validate_lens() -> sort-junction validation
    lens_path()     -> bidirectional sequential composition
    lens_fan()      -> bidirectional parallel composition
"""

from __future__ import annotations

from ..views import EquationView, LensView
import hydra.core as core
from hydra.dsl.meta.phantoms import record, string, unit, var, list_, TTerm

from ..algebra.sort import sort_type_from_term
from ..utils import bind_composition
from .paths import path, fan


# ---------------------------------------------------------------------------
# Lens declaration
# ---------------------------------------------------------------------------

LENS_TYPE_NAME = core.Name("ua.lens.Lens")


def lens(
    name: str,
    forward: str,
    backward: str,
    residual_sort: core.Term | None = None,
) -> core.Term:
    """Create a lens as a Hydra record term.

    A lens pairs a forward equation with a backward equation and an optional
    residual sort that carries information between the two legs.
    """
    return record(LENS_TYPE_NAME, [
        core.Name("name") >> string(name),
        core.Name("forward") >> string(forward),
        core.Name("backward") >> string(backward),
        core.Name("residualSort") >> (TTerm(residual_sort) if residual_sort is not None else unit()),
    ]).value


def validate_lens(
    eq_terms_by_name: dict[str, core.Term],
    lens_term: core.Term,
) -> None:
    """Validate the sort junctions of a lens.

    Plain: forward.domain == backward.codomain, forward.codomain == backward.domain.
    With residual_sort: additionally checks that forward codomain and backward domain
    are product sorts containing the residual.
    """
    from ..algebra.sort import is_product_sort, product_sort_elements

    lv = LensView(lens_term)
    lname = lv.name
    fwd_name = lv.forward
    bwd_name = lv.backward
    residual_sort = lv.residual_sort

    if fwd_name not in eq_terms_by_name:
        raise TypeError(f"Lens '{lname}': forward equation '{fwd_name}' not found")
    if bwd_name not in eq_terms_by_name:
        raise TypeError(f"Lens '{lname}': backward equation '{bwd_name}' not found")

    fwd_eq = EquationView(eq_terms_by_name[fwd_name])
    bwd_eq = EquationView(eq_terms_by_name[bwd_name])

    # Base check: forward domain == backward codomain
    actual = sort_type_from_term(fwd_eq.domain_sort)
    expected = sort_type_from_term(bwd_eq.codomain_sort)
    if actual != expected:
        raise TypeError(
            f"Lens '{lname}': forward domain != backward codomain: {actual} != {expected}"
        )

    # If residual_sort is set, check product sort constraints
    has_residual = residual_sort is not None
    if has_residual:
        fwd_codomain = fwd_eq.codomain_sort
        if not is_product_sort(fwd_codomain):
            raise TypeError(
                f"Lens '{lname}': forward codomain must be a product sort (has residual)"
            )
        bwd_domain = bwd_eq.domain_sort
        if not is_product_sort(bwd_domain):
            raise TypeError(
                f"Lens '{lname}': backward domain must be a product sort (has residual)"
            )
        residual_type = sort_type_from_term(residual_sort)
        if not any(sort_type_from_term(e) == residual_type for e in product_sort_elements(fwd_codomain)):
            raise TypeError(f"Lens '{lname}': forward codomain missing residual sort")
        if not any(sort_type_from_term(e) == residual_type for e in product_sort_elements(bwd_domain)):
            raise TypeError(f"Lens '{lname}': backward domain missing residual sort")
    else:
        # Plain lens: forward codomain == backward domain
        actual_co = sort_type_from_term(fwd_eq.codomain_sort)
        expected_co = sort_type_from_term(bwd_eq.domain_sort)
        if actual_co != expected_co:
            raise TypeError(
                f"Lens '{lname}': forward codomain != backward domain: {actual_co} != {expected_co}"
            )


def lens_path(
    name: str,
    lens_names: list[str],
    lens_terms_by_name: dict[str, core.Term],
    params: dict[str, list[core.Term]] | None = None,
) -> tuple[tuple[core.Name, core.Term], tuple[core.Name, core.Term]]:
    """Compose lenses sequentially into a bidirectional path.

    Plain lenses (no residual_sort): forward is left-to-right, backward is
    right-to-left — two independent paths.

    With residual_sort: forward collects residuals at each step via
    ua.prim.lens_fwd, backward injects them in reverse via ua.prim.lens_bwd.
    The primitives must be registered in the graph before the
    terms can be reduced — _build_lenses in _assembly.py handles this.
    """
    if not lens_names:
        raise ValueError(f"lens_path '{name}' must have at least one lens")

    fwd_eq_names = []
    bwd_eq_names = []
    has_residual = False
    for ln in lens_names:
        lv = LensView(lens_terms_by_name[ln])
        fwd_eq_names.append(lv.forward)
        bwd_eq_names.append(lv.backward)
        if lv.residual_sort is not None:
            has_residual = True

    if has_residual and len(lens_names) > 1:
        # Multiple lenses with residual: thread residuals through forward/backward.
        fwd_pair = bind_composition("path", f"{name}.fwd", "x",
                                   var("ua.prim.lens_fwd") @ list_([var(f"ua.equation.{n}") for n in fwd_eq_names]) @ var("x"))
        bwd_pair = bind_composition("path", f"{name}.bwd", "p",
                                   var("ua.prim.lens_bwd") @ list_([var(f"ua.equation.{n}") for n in bwd_eq_names]) @ var("p"))
        return fwd_pair, bwd_pair

    # Single lens or plain lens: two independent paths.
    fwd_name_obj, fwd_term = path(f"{name}.fwd", fwd_eq_names, params)
    bwd_name_obj, bwd_term = path(f"{name}.bwd", list(reversed(bwd_eq_names)))

    return (fwd_name_obj, fwd_term), (bwd_name_obj, bwd_term)


def lens_fan(
    name: str,
    lens_names: list[str],
    merge_lens_name: str,
    lens_terms_by_name: dict[str, core.Term],
) -> tuple[tuple[core.Name, core.Term], tuple[core.Name, core.Term]]:
    """Compose lenses in parallel into a bidirectional fan.

    Forward fan: branch forwards + forward merge.
    Backward fan: branch backwards + backward merge.
    """
    if not lens_names:
        raise ValueError(f"lens_fan '{name}' must have at least one branch lens")

    fwd_branch_names = []
    bwd_branch_names = []
    for ln in lens_names:
        lv = LensView(lens_terms_by_name[ln])
        fwd_branch_names.append(lv.forward)
        bwd_branch_names.append(lv.backward)

    merge_lv = LensView(lens_terms_by_name[merge_lens_name])
    fwd_merge_name = merge_lv.forward
    bwd_merge_name = merge_lv.backward

    fwd_name_obj, fwd_term = fan(f"{name}.fwd", fwd_branch_names, fwd_merge_name)
    bwd_name_obj, bwd_term = fan(f"{name}.bwd", bwd_branch_names, bwd_merge_name)

    return (fwd_name_obj, fwd_term), (bwd_name_obj, bwd_term)
