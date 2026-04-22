"""Sequential and parallel composition of equations as Hydra lambda terms.

    path()  -> Hydra lambda term (sequential: f_n . ... . f_1)
    fan()   -> Hydra lambda term (parallel: merge(b1(x), b2(x), ...))

Both produce TermLambda values stored in the Graph's bound_terms dict.
When reduce_term encounters var("ua.path.ffn"), it resolves to the lambda
in bound_terms, then beta-reduces: the inner var("ua.equation.X") references
resolve to Primitives in the same Graph.
"""

from __future__ import annotations

import hydra.core as core
from hydra.dsl.meta.phantoms import var, list_

from unialg.utils import bind_composition


# ---------------------------------------------------------------------------
# Path (sequential composition)
# ---------------------------------------------------------------------------

def path(
    name: str,
    eq_names: list[str],
    params: dict[str, list[core.Term]] | None = None,
    residual: bool = False,
    residual_semiring: str | None = None,
) -> tuple[core.Name, core.Term]:
    """Build a sequential composition as a Hydra lambda term.

    Args:
        name:           identifier (e.g. "ffn")
        eq_names:       ordered list of equation names, applied left-to-right
                        (first equation receives input, last produces output)
        params:         optional pre-bound arguments per equation, e.g.
                        {"linear": [W_term]} applies W before the pipeline input

    Returns:
        (Name("ua.path.<name>"), lambda_term)
        The lambda term is: lx. eq_n(... eq_2(eq_1(x)) ...)
    """
    if not eq_names:
        raise ValueError(f"Path '{name}' must have at least one equation")

    from hydra.dsl.meta.phantoms import TTerm
    body: TTerm = var("x")
    for eq_name in eq_names:
        fn: TTerm = var(f"ua.equation.{eq_name}")
        if params and eq_name in params:
            for p in params[eq_name]:
                fn = fn @ TTerm(p)
        body = fn @ body

    if residual:
        sr_name = residual_semiring or "default"
        body = var(f"ua.prim.residual_add.{sr_name}") @ body @ var("x")

    return bind_composition("path", name, "x", body)


# ---------------------------------------------------------------------------
# Fan (parallel composition)
# ---------------------------------------------------------------------------

def fan(
    name: str,
    branch_names: list[str],
    merge_name: str,
) -> tuple[core.Name, core.Term]:
    """Build a parallel composition as a Hydra lambda term.

    Args:
        name:           identifier (e.g. "attn")
        branch_names:   names of equations applied in parallel to the same input
        merge_name:     name of the equation that merges branch outputs

    Returns:
        (Name("ua.fan.<name>"), lambda_term)
        The lambda term is: λx. merge([b1(x), b2(x), ..., bn(x)])
        The merge equation is resolved as a list-consuming prim1.
    """
    if not branch_names:
        raise ValueError(f"Fan '{name}' must have at least one branch")

    # Build list of branch results applied to input
    branch_results = [
        var(f"ua.equation.{bname}") @ var("x")
        for bname in branch_names
    ]
    list_term = list_(branch_results)
    body = var(f"ua.equation.{merge_name}") @ list_term

    return bind_composition("fan", name, "x", body)
