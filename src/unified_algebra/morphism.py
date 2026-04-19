"""Tensor equations as Hydra Primitives.

An equation is the fundamental construct of the DSL — simultaneously a
morphism (typed: domain → codomain) and a tensor equation (einsum +
semiring, with optional pointwise nonlinearity).

    equation()          → Hydra record term (the declaration)
    resolve_equation()  → Hydra Primitive (the compiled callable)

This follows the same pattern as semiring.py:
    semiring()          → Hydra record term
    resolve_semiring()  → ResolvedSemiring with callables
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unified_algebra._hydra_setup import record_fields, string_value
import hydra.core as core
import hydra.dsl.terms as Terms
from hydra.dsl.prims import prim1, prim2, prim3

from .semiring import resolve_semiring
from .sort import tensor_coder
from .contraction import compile_equation, semiring_contract

if TYPE_CHECKING:
    from .backend import Backend


# ---------------------------------------------------------------------------
# Equation as Hydra record term
# ---------------------------------------------------------------------------

EQUATION_TYPE_NAME = core.Name("ua.equation.Equation")


def equation(name: str, einsum: str | None,
             domain_sort: core.Term, codomain_sort: core.Term,
             semiring_term: core.Term | None = None,
             nonlinearity: str | None = None,
             inputs: tuple[str, ...] = ()) -> core.Term:
    """Create a tensor equation as a Hydra record term.

    Args:
        name:           identifier (e.g. "linear1", "attention")
        einsum:         einsum equation string, or None for pure pointwise
        domain_sort:    sort term for input (from sort())
        codomain_sort:  sort term for output (from sort())
        semiring_term:  semiring term (from semiring()), required if einsum is set
        nonlinearity:   optional pointwise op name (e.g. "relu"), applied after contraction
        inputs:         names of upstream tensors this equation reads from
    """
    return Terms.record(EQUATION_TYPE_NAME, [
        Terms.field("name", Terms.string(name)),
        Terms.field("einsum", Terms.string(einsum or "")),
        Terms.field("domainSort", domain_sort),
        Terms.field("codomainSort", codomain_sort),
        Terms.field("semiring", semiring_term if semiring_term is not None else Terms.unit()),
        Terms.field("nonlinearity", Terms.string(nonlinearity or "")),
        Terms.field("inputs", Terms.list_([Terms.string(n) for n in inputs])),
    ])


# ---------------------------------------------------------------------------
# Resolve equation → Hydra Primitive
# ---------------------------------------------------------------------------

def resolve_equation(eq_term: core.Term, backend: Backend) -> hydra.graph.Primitive:
    """Resolve an equation term into a Hydra Primitive.

    Reads the equation's fields, resolves its semiring against the backend,
    compiles the einsum, and produces a Primitive with the correct arity.
    """
    fields = record_fields(eq_term)
    name = string_value(fields["name"])
    einsum_str = string_value(fields["einsum"])
    nl_str = string_value(fields["nonlinearity"])

    has_einsum = bool(einsum_str)
    has_nl = bool(nl_str)

    coder = tensor_coder()

    prim_name = core.Name(f"ua.equation.{name}")

    if has_einsum:
        sr = resolve_semiring(fields["semiring"], backend)
        eq = compile_equation(einsum_str)
        n_inputs = len(eq.input_vars)
        nl_fn = backend.unary(nl_str) if has_nl else None

        if n_inputs == 1:
            def compute1(a):
                r = semiring_contract(eq, [a], sr, backend)
                return nl_fn(r) if nl_fn else r
            return prim1(prim_name, compute1, [], coder, coder)

        elif n_inputs == 2:
            def compute2(a, b):
                r = semiring_contract(eq, [a, b], sr, backend)
                return nl_fn(r) if nl_fn else r
            return prim2(prim_name, compute2, [], coder, coder, coder)

        elif n_inputs == 3:
            def compute3(a, b, c):
                r = semiring_contract(eq, [a, b, c], sr, backend)
                return nl_fn(r) if nl_fn else r
            return prim3(prim_name, compute3, [], coder, coder, coder, coder)

        else:
            raise ValueError(
                f"Equation '{name}': einsum has {n_inputs} inputs, max supported is 3"
            )

    elif has_nl:
        nl_fn = backend.unary(nl_str)
        return prim1(prim_name, lambda x: nl_fn(x), [], coder, coder)

    else:
        raise ValueError(f"Equation '{name}' has neither einsum nor nonlinearity")
