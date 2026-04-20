"""Lenses: bidirectional morphisms pairing forward and backward equations.

A lens is a declaration that two equations are the forward and backward
legs of the same bidirectional morphism. What "backward" means depends
on the semiring: gradients (real), path recovery (tropical), dual closure
(max-min), likelihood propagation (probabilistic), etc.

lens_path() composes lenses sequentially: forward left-to-right,
backward right-to-left. Both directions use the existing path() machinery.

Theoretical grounding: lenses are dialenses of height 1 (Capucci, Gavranovic
et al., MFPS 2024). The forward leg runs in the base category; the backward
leg runs in the dual (or fibred dual) category. The residual sort is the
carrier of information passed from forward to backward — it is optional at
height 1 (plain lenses) and becomes mandatory at height 2 (optics).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unified_algebra.utils import record_fields, string_value
import hydra.core as core
import hydra.dsl.terms as Terms

from .sort import sort_type_from_term
from .composition import path, fan

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Lens type name
# ---------------------------------------------------------------------------

LENS_TYPE_NAME = core.Name("ua.lens.Lens")


# ---------------------------------------------------------------------------
# Lens declaration
# ---------------------------------------------------------------------------

def lens(
    name: str,
    forward: str,
    backward: str,
    residual_sort: core.Term | None = None,
) -> core.Term:
    """Create a lens as a Hydra record term.

    A lens pairs a forward equation with a backward equation and an optional
    residual sort that carries information between the two legs.

    Args:
        name:          identifier (e.g. "linear_lens")
        forward:       name of the forward equation (domain → codomain)
        backward:      name of the backward equation (codomain → domain)
        residual_sort: optional sort term for the residual type connecting
                       forward to backward. When None, stored as unit.

    Returns:
        A Hydra TermRecord representing the lens declaration.
    """
    return Terms.record(LENS_TYPE_NAME, [
        Terms.field("name", Terms.string(name)),
        Terms.field("forward", Terms.string(forward)),
        Terms.field("backward", Terms.string(backward)),
        Terms.field("residualSort", residual_sort if residual_sort is not None else Terms.unit()),
    ])


# ---------------------------------------------------------------------------
# Field extraction helper
# ---------------------------------------------------------------------------

def _lens_fields(lens_term: core.Term) -> dict[str, object]:
    """Extract the lens record's string-valued fields as plain strings.

    Returns a dict with keys 'name', 'forward', 'backward' as str values,
    plus 'residualSort' as the raw Term.
    """
    fields = record_fields(lens_term)
    return {
        "name": string_value(fields["name"]),
        "forward": string_value(fields["forward"]),
        "backward": string_value(fields["backward"]),
        "residualSort": fields.get("residualSort"),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_lens(
    eq_terms_by_name: dict[str, core.Term],
    lens_term: core.Term,
) -> None:
    """Validate the sort junctions of a lens.

    Checks that the forward and backward equations form a consistent
    bidirectional pair:

        forward:  domain → codomain
        backward: codomain → domain

    Raises TypeError if either equation is unknown or if the sorts do not
    satisfy the lens compatibility conditions.

    Args:
        eq_terms_by_name: dict of equation name → equation record term
        lens_term:        lens record term (from lens())
    """
    lf = _lens_fields(lens_term)
    lname = lf["name"]
    fwd_name = lf["forward"]
    bwd_name = lf["backward"]

    if fwd_name not in eq_terms_by_name:
        raise TypeError(
            f"Lens '{lname}': forward equation '{fwd_name}' not found"
        )
    if bwd_name not in eq_terms_by_name:
        raise TypeError(
            f"Lens '{lname}': backward equation '{bwd_name}' not found"
        )

    fwd_fields = record_fields(eq_terms_by_name[fwd_name])
    bwd_fields = record_fields(eq_terms_by_name[bwd_name])

    # forward:  domain → codomain
    # backward: codomain → domain
    # So: forward.domain must equal backward.codomain
    #   and forward.codomain must equal backward.domain

    fwd_domain = sort_type_from_term(fwd_fields["domainSort"])
    fwd_codomain = sort_type_from_term(fwd_fields["codomainSort"])
    bwd_domain = sort_type_from_term(bwd_fields["domainSort"])
    bwd_codomain = sort_type_from_term(bwd_fields["codomainSort"])

    if fwd_domain != bwd_codomain:
        raise TypeError(
            f"Lens '{lname}': forward domain {fwd_domain.value.value!r} != "
            f"backward codomain {bwd_codomain.value.value!r}"
        )

    if fwd_codomain != bwd_domain:
        raise TypeError(
            f"Lens '{lname}': forward codomain {fwd_codomain.value.value!r} != "
            f"backward domain {bwd_domain.value.value!r}"
        )


# ---------------------------------------------------------------------------
# Lens path composition
# ---------------------------------------------------------------------------

def lens_path(
    name: str,
    lens_names: list[str],
    lens_terms_by_name: dict[str, core.Term],
    domain_sort: core.Term,
    codomain_sort: core.Term,
    params: dict[str, list[core.Term]] | None = None,
) -> tuple[tuple[core.Name, core.Term], tuple[core.Name, core.Term]]:
    """Compose lenses sequentially into a bidirectional path.

    The forward path applies each lens's forward equation left-to-right
    (domain → codomain). The backward path applies each lens's backward
    equation in REVERSED order (codomain → domain), matching the categorical
    composition law for lenses.

    Args:
        name:                identifier for the composed lens path
        lens_names:          ordered list of lens names to compose
        lens_terms_by_name:  dict of lens name → lens record term
        domain_sort:         sort term for the overall domain (forward input)
        codomain_sort:       sort term for the overall codomain (forward output)
        params:              optional path params forwarded to path() for the
                             forward leg; backward leg uses None

    Returns:
        ((fwd_Name, fwd_Term), (bwd_Name, bwd_Term))
        where fwd is bound as "ua.path.<name>.fwd" and
              bwd is bound as "ua.path.<name>.bwd"
    """
    if not lens_names:
        raise ValueError(f"lens_path '{name}' must have at least one lens")

    # Extract forward and backward equation names from each lens
    fwd_eq_names = []
    bwd_eq_names = []
    for ln in lens_names:
        lf = _lens_fields(lens_terms_by_name[ln])
        fwd_eq_names.append(lf["forward"])
        bwd_eq_names.append(lf["backward"])

    # Forward: left-to-right through the lens chain
    fwd_name_obj, fwd_term = path(
        f"{name}.fwd",
        fwd_eq_names,
        domain_sort,
        codomain_sort,
        params,
    )

    # Backward: right-to-left (reversed) through the lens chain
    # Backward leg maps codomain → domain (the "put" or "gradient" direction)
    bwd_name_obj, bwd_term = path(
        f"{name}.bwd",
        list(reversed(bwd_eq_names)),
        codomain_sort,
        domain_sort,
        None,
    )

    return (fwd_name_obj, fwd_term), (bwd_name_obj, bwd_term)


# ---------------------------------------------------------------------------
# Lens fan composition
# ---------------------------------------------------------------------------

def lens_fan(
    name: str,
    lens_names: list[str],
    merge_lens_name: str,
    lens_terms_by_name: dict[str, core.Term],
    domain_sort: core.Term,
    codomain_sort: core.Term,
) -> tuple[tuple[core.Name, core.Term], tuple[core.Name, core.Term]]:
    """Compose lenses in parallel into a bidirectional fan.

    The forward fan applies each lens's forward equation as a branch and
    uses the merge lens's forward equation to combine results.
    The backward fan applies each lens's backward equation as a branch and
    uses the merge lens's backward equation to combine feedback.

    Args:
        name:                identifier for the composed lens fan
        lens_names:          names of the branch lenses (parallel branches)
        merge_lens_name:     name of the merge lens (forward merge + backward merge)
        lens_terms_by_name:  dict of lens name → lens record term
        domain_sort:         sort term for the fan's input
        codomain_sort:       sort term for the fan's output

    Returns:
        ((fwd_Name, fwd_Term), (bwd_Name, bwd_Term))
        where fwd is bound as "ua.fan.<name>.fwd" and
              bwd is bound as "ua.fan.<name>.bwd"
    """
    if not lens_names:
        raise ValueError(f"lens_fan '{name}' must have at least one branch lens")

    # Extract forward and backward branch names
    fwd_branch_names = []
    bwd_branch_names = []
    for ln in lens_names:
        lf = _lens_fields(lens_terms_by_name[ln])
        fwd_branch_names.append(lf["forward"])
        bwd_branch_names.append(lf["backward"])

    # Extract merge equation names
    merge_lf = _lens_fields(lens_terms_by_name[merge_lens_name])
    fwd_merge_name = merge_lf["forward"]
    bwd_merge_name = merge_lf["backward"]

    # Forward fan: branches + forward merge
    fwd_name_obj, fwd_term = fan(
        f"{name}.fwd",
        fwd_branch_names,
        fwd_merge_name,
        domain_sort,
        codomain_sort,
    )

    # Backward fan: backward branches + backward merge
    bwd_name_obj, bwd_term = fan(
        f"{name}.bwd",
        bwd_branch_names,
        bwd_merge_name,
        codomain_sort,
        domain_sort,
    )

    return (fwd_name_obj, fwd_term), (bwd_name_obj, bwd_term)
