"""Sequential and parallel composition of equations as Hydra lambda terms.

    path()  -> Hydra lambda term (sequential: f_n . ... . f_1)
    fan()   -> Hydra lambda term (parallel: merge(b1(x), b2(x), ...))

Both produce TermLambda values stored in the Graph's bound_terms dict.
When reduce_term encounters var("ua.path.ffn"), it resolves to the lambda
in bound_terms, then beta-reduces: the inner var("ua.equation.X") references
resolve to Primitives in the same Graph.
"""

from __future__ import annotations

from unified_algebra.utils import record_fields
import hydra.core as core
import hydra.dsl.terms as Terms

from .sort import sort_type_from_term, check_sort_junction


# ---------------------------------------------------------------------------
# Path (sequential composition)
# ---------------------------------------------------------------------------

def path(
    name: str,
    eq_names: list[str],
    domain_sort: core.Term,
    codomain_sort: core.Term,
    params: dict[str, list[core.Term]] | None = None,
) -> tuple[core.Name, core.Term]:
    """Build a sequential composition as a Hydra lambda term.

    Args:
        name:           identifier (e.g. "ffn")
        eq_names:       ordered list of equation names, applied left-to-right
                        (first equation receives input, last produces output)
        domain_sort:    sort term for the path's input
        codomain_sort:  sort term for the path's output
        params:         optional pre-bound arguments per equation, e.g.
                        {"linear": [W_term]} applies W before the pipeline input

    Returns:
        (Name("ua.path.<name>"), lambda_term)
        The lambda term is: lx. eq_n(... eq_2(eq_1(x)) ...)
    """
    if not eq_names:
        raise ValueError(f"Path '{name}' must have at least one equation")

    body: core.Term = Terms.var("x")
    for eq_name in eq_names:
        fn: core.Term = Terms.var(f"ua.equation.{eq_name}")
        if params and eq_name in params:
            for p in params[eq_name]:
                fn = Terms.apply(fn, p)
        body = Terms.apply(fn, body)

    term = Terms.lambda_("x", body)
    return (core.Name(f"ua.path.{name}"), term)


# ---------------------------------------------------------------------------
# Fan (parallel composition)
# ---------------------------------------------------------------------------

def fan(
    name: str,
    branch_names: list[str],
    merge_name: str,
    domain_sort: core.Term,
    codomain_sort: core.Term,
) -> tuple[core.Name, core.Term]:
    """Build a parallel composition as a Hydra lambda term.

    Args:
        name:           identifier (e.g. "attn")
        branch_names:   names of equations applied in parallel to the same input
        merge_name:     name of the equation that merges branch outputs
        domain_sort:    sort term for the fan's input (shared by all branches)
        codomain_sort:  sort term for the fan's output (merge's codomain)

    Returns:
        (Name("ua.fan.<name>"), lambda_term)
        The lambda term is: λx. merge([b1(x), b2(x), ..., bn(x)])
        The merge equation is resolved as a list-consuming prim1.
    """
    if not branch_names:
        raise ValueError(f"Fan '{name}' must have at least one branch")

    # Build list of branch results applied to input
    branch_results = [
        Terms.apply(Terms.var(f"ua.equation.{bname}"), Terms.var("x"))
        for bname in branch_names
    ]
    list_term = Terms.list_(branch_results)
    body = Terms.apply(Terms.var(f"ua.equation.{merge_name}"), list_term)

    term = Terms.lambda_("x", body)
    return (core.Name(f"ua.fan.{name}"), term)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_path(
    eq_terms_by_name: dict[str, core.Term],
    eq_names: list[str],
    domain_sort: core.Term,
    codomain_sort: core.Term,
) -> None:
    """Validate sort junctions along a path.

    Checks:
    1. domain_sort matches first equation's domain
    2. Each consecutive pair: codomain[i] == domain[i+1]  (via check_sort_junction)
    3. Last equation's codomain matches codomain_sort
    """
    path_domain = sort_type_from_term(domain_sort)
    first_domain = sort_type_from_term(
        record_fields(eq_terms_by_name[eq_names[0]])["domainSort"]
    )
    if path_domain != first_domain:
        raise TypeError(
            f"Path domain {path_domain.value.value!r} != "
            f"first equation '{eq_names[0]}' domain {first_domain.value.value!r}"
        )

    for i in range(len(eq_names) - 1):
        check_sort_junction(
            eq_terms_by_name[eq_names[i]],
            eq_terms_by_name[eq_names[i + 1]],
        )

    path_codomain = sort_type_from_term(codomain_sort)
    last_codomain = sort_type_from_term(
        record_fields(eq_terms_by_name[eq_names[-1]])["codomainSort"]
    )
    if path_codomain != last_codomain:
        raise TypeError(
            f"Path codomain {path_codomain.value.value!r} != "
            f"last equation '{eq_names[-1]}' codomain {last_codomain.value.value!r}"
        )


def validate_fan(
    eq_terms_by_name: dict[str, core.Term],
    branch_names: list[str],
    merge_name: str,
    domain_sort: core.Term,
    codomain_sort: core.Term,
) -> None:
    """Validate sort junctions in a fan.

    Checks:
    1. domain_sort matches each branch's domain
    2. Each branch's codomain is compatible with the merge's domain
    3. merge's codomain matches codomain_sort
    """
    fan_domain = sort_type_from_term(domain_sort)

    for bname in branch_names:
        branch_fields = record_fields(eq_terms_by_name[bname])
        branch_domain = sort_type_from_term(branch_fields["domainSort"])
        if branch_domain != fan_domain:
            raise TypeError(
                f"Fan branch '{bname}' domain {branch_domain.value.value!r} != "
                f"fan domain {fan_domain.value.value!r}"
            )

    # Check branch codomains match merge's domain
    merge_fields = record_fields(eq_terms_by_name[merge_name])
    merge_domain = sort_type_from_term(merge_fields["domainSort"])
    for bname in branch_names:
        branch_codomain = sort_type_from_term(
            record_fields(eq_terms_by_name[bname])["codomainSort"]
        )
        if branch_codomain != merge_domain:
            raise TypeError(
                f"Fan branch '{bname}' codomain {branch_codomain.value.value!r} != "
                f"merge '{merge_name}' domain {merge_domain.value.value!r}"
            )

    merge_codomain = sort_type_from_term(merge_fields["codomainSort"])
    fan_codomain = sort_type_from_term(codomain_sort)
    if merge_codomain != fan_codomain:
        raise TypeError(
            f"Fan merge '{merge_name}' codomain {merge_codomain.value.value!r} != "
            f"fan codomain {fan_codomain.value.value!r}"
        )
