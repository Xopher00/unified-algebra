"""Sequential and parallel composition of equations as Hydra lambda terms.

    path()  -> Hydra lambda term (sequential: f_n . ... . f_1)
    fan()   -> Hydra lambda term (parallel: merge(b1(x), b2(x), ...))

Both produce TermLambda values stored in the Graph's bound_terms dict.
When reduce_term encounters var("ua.path.ffn"), it resolves to the lambda
in bound_terms, then beta-reduces: the inner var("ua.equation.X") references
resolve to Primitives in the same Graph.
"""

from __future__ import annotations

from .utils import record_fields, string_value, lens_fields
import hydra.core as core
import hydra.dsl.terms as Terms

from .sort import sort_type_from_term, _check_sort


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
# Lenses: bidirectional morphisms pairing forward and backward equations
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
    return Terms.record(LENS_TYPE_NAME, [
        Terms.field("name", Terms.string(name)),
        Terms.field("forward", Terms.string(forward)),
        Terms.field("backward", Terms.string(backward)),
        Terms.field("residualSort", residual_sort if residual_sort is not None else Terms.unit()),
    ])





def validate_lens(
    eq_terms_by_name: dict[str, core.Term],
    lens_term: core.Term,
) -> None:
    """Validate the sort junctions of a lens.

    Plain: forward.domain == backward.codomain, forward.codomain == backward.domain.
    With residual_sort: additionally checks that forward codomain and backward domain
    are product sorts containing the residual.
    """
    from .sort import is_product_sort, product_sort_elements

    lf = lens_fields(lens_term)
    lname = lf["name"]
    fwd_name = lf["forward"]
    bwd_name = lf["backward"]
    residual_sort = lf["residualSort"]

    if fwd_name not in eq_terms_by_name:
        raise TypeError(f"Lens '{lname}': forward equation '{fwd_name}' not found")
    if bwd_name not in eq_terms_by_name:
        raise TypeError(f"Lens '{lname}': backward equation '{bwd_name}' not found")

    fwd_fields = record_fields(eq_terms_by_name[fwd_name])
    bwd_fields = record_fields(eq_terms_by_name[bwd_name])

    # Base check: forward domain == backward codomain
    _check_sort(
        eq_terms_by_name, fwd_name, "domainSort", bwd_fields["codomainSort"],
        f"Lens '{lname}': forward domain != backward codomain"
    )

    # If residual_sort is set, check product sort constraints
    has_residual = (
        residual_sort is not None
        and not isinstance(residual_sort, core.TermUnit)
    )
    if has_residual:
        fwd_codomain = fwd_fields["codomainSort"]
        if not is_product_sort(fwd_codomain):
            raise TypeError(
                f"Lens '{lname}': forward codomain must be a product sort (has residual)"
            )
        bwd_domain = bwd_fields["domainSort"]
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
        _check_sort(
            eq_terms_by_name, fwd_name, "codomainSort", bwd_fields["domainSort"],
            f"Lens '{lname}': forward codomain != backward domain"
        )


def lens_path(
    name: str,
    lens_names: list[str],
    lens_terms_by_name: dict[str, core.Term],
    domain_sort: core.Term,
    codomain_sort: core.Term,
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
        lf = lens_fields(lens_terms_by_name[ln])
        fwd_eq_names.append(lf["forward"])
        bwd_eq_names.append(lf["backward"])
        rs = lf.get("residualSort")
        if rs is not None and not isinstance(rs, core.TermUnit):
            has_residual = True

    if has_residual and len(lens_names) > 1:
        # Multiple lenses with residual: thread residuals through forward/backward.
        from ._lens_threading import _lens_path_threaded
        return _lens_path_threaded(name, fwd_eq_names, bwd_eq_names, domain_sort, codomain_sort)

    # Single lens or plain lens: two independent paths.
    # For a single lens with residual, the forward equation already returns the
    # product sort pair; the backward equation accepts the same pair — no additional
    # threading needed.
    fwd_name_obj, fwd_term = path(
        f"{name}.fwd", fwd_eq_names, domain_sort, codomain_sort, params,
    )
    bwd_name_obj, bwd_term = path(
        f"{name}.bwd", list(reversed(bwd_eq_names)), codomain_sort, domain_sort, None,
    )

    return (fwd_name_obj, fwd_term), (bwd_name_obj, bwd_term)


def lens_fan(
    name: str,
    lens_names: list[str],
    merge_lens_name: str,
    lens_terms_by_name: dict[str, core.Term],
    domain_sort: core.Term,
    codomain_sort: core.Term,
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
        lf = lens_fields(lens_terms_by_name[ln])
        fwd_branch_names.append(lf["forward"])
        bwd_branch_names.append(lf["backward"])

    merge_lf = lens_fields(lens_terms_by_name[merge_lens_name])
    fwd_merge_name = merge_lf["forward"]
    bwd_merge_name = merge_lf["backward"]

    fwd_name_obj, fwd_term = fan(
        f"{name}.fwd", fwd_branch_names, fwd_merge_name, domain_sort, codomain_sort,
    )
    bwd_name_obj, bwd_term = fan(
        f"{name}.bwd", bwd_branch_names, bwd_merge_name, codomain_sort, domain_sort,
    )

    return (fwd_name_obj, fwd_term), (bwd_name_obj, bwd_term)

