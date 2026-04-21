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

import hydra.core as core
import hydra.dsl.terms as Terms
from hydra.dsl.prims import prim1, prim2, prim3, float32 as float32_coder, list_ as list_coder

from .semiring import resolve_semiring
from .sort import tensor_coder, sort_coder, is_batched
from .contraction import compile_equation, semiring_contract
from .views import EquationView

if TYPE_CHECKING:
    from .backend import Backend


# ---------------------------------------------------------------------------
# Batch-dimension helpers
# ---------------------------------------------------------------------------

def _prepend_batch_dim(einsum_str: str) -> str:
    """Prepend a fresh batch dimension to every operand of an einsum string.

    Picks the first lowercase letter not already used as an index as the
    batch character, so the result is unambiguous regardless of the original
    equation.

    Examples::

        "ij,j->i"   →  "bij,bj->bi"    (batch char = 'b')
        "ij,jk->ik" →  "aij,ajk->aik"  (if 'a' is unused and 'b' is used)
    """
    if not einsum_str:
        return einsum_str
    used = set(einsum_str) - {",", "-", ">"}
    batch_char = next(c for c in "bcdefghmnopqrstuvwxyz" if c not in used)
    lhs, rhs = einsum_str.split("->")
    inputs = lhs.split(",")
    batched_inputs = ",".join(batch_char + inp for inp in inputs)
    batched_output = batch_char + rhs
    return f"{batched_inputs}->{batched_output}"


# ---------------------------------------------------------------------------
# Equation as Hydra record term
# ---------------------------------------------------------------------------

EQUATION_TYPE_NAME = core.Name("ua.equation.Equation")


def equation(name: str, einsum: str | None,
             domain_sort: core.Term, codomain_sort: core.Term,
             semiring_term: core.Term | None = None,
             nonlinearity: str | None = None,
             inputs: tuple[str, ...] = (),
             param_slots: tuple[str, ...] = ()) -> core.Term:
    """Create a tensor equation as a Hydra record term.

    Args:
        name:           identifier (e.g. "linear1", "attention")
        einsum:         einsum equation string, or None for pure pointwise
        domain_sort:    sort term for input (from sort())
        codomain_sort:  sort term for output (from sort())
        semiring_term:  semiring term (from semiring()), required if einsum is set
        nonlinearity:   optional pointwise op name (e.g. "relu"), applied after contraction
        inputs:         names of upstream tensors this equation reads from
        param_slots:    names of scalar hyperparameters this equation expects
                        before its tensor input(s) (e.g. ("temperature",))
    """
    return Terms.record(EQUATION_TYPE_NAME, [
        Terms.field("name", Terms.string(name)),
        Terms.field("einsum", Terms.string(einsum or "")),
        Terms.field("domainSort", domain_sort),
        Terms.field("codomainSort", codomain_sort),
        Terms.field("semiring", semiring_term if semiring_term is not None else Terms.unit()),
        Terms.field("nonlinearity", Terms.string(nonlinearity or "")),
        Terms.field("inputs", Terms.list_([Terms.string(n) for n in inputs])),
        Terms.field("paramSlots", Terms.list_([Terms.string(p) for p in param_slots])),
    ])


# ---------------------------------------------------------------------------
# Shared field-extraction helper
# ---------------------------------------------------------------------------

def _resolve_fields(eq_term: core.Term, backend: "Backend"):
    """Extract and resolve common equation fields from a Hydra record term.

    Returns a tuple of:
        (fields, name, einsum_str, has_einsum, has_nl, nl_fn,
         in_coder, out_coder, prim_name, sr, eq)

    ``sr`` and ``eq`` are ``None`` when ``has_einsum`` is ``False``.
    ``nl_fn`` is ``None`` when ``has_nl`` is ``False``.

    The returned ``in_coder`` is the bare sort coder for the domain sort.
    Callers that need a wrapped form (e.g. ``list_coder(in_coder)``) must
    apply that wrapping themselves.
    """
    v = EquationView(eq_term)
    name = v.name
    einsum_str = v.einsum
    nl_str = v.nonlinearity

    # Auto-prepend batch dimension when the domain sort is batched.
    # The declaration stores the logical (unbatched) einsum; resolution
    # produces the physical (batched) einsum for the backend.
    has_einsum = bool(einsum_str)
    if has_einsum and is_batched(v.domain_sort):
        einsum_str = _prepend_batch_dim(einsum_str)

    has_nl = bool(nl_str)
    nl_fn = backend.unary(nl_str) if has_nl else None

    in_coder = sort_coder(v.domain_sort, backend)
    out_coder = sort_coder(v.codomain_sort, backend)
    prim_name = core.Name(f"ua.equation.{name}")

    sr = None
    eq = None
    if has_einsum:
        sr = resolve_semiring(v.semiring, backend)
        eq = compile_equation(einsum_str)

    return v, name, einsum_str, has_einsum, has_nl, nl_fn, in_coder, out_coder, prim_name, sr, eq


def _make_prim(prim_name, compute, coders, out_coder):
    """Dispatch a compute closure + coder list to prim1/prim2/prim3."""
    n = len(coders)
    if n == 1:
        return prim1(prim_name, lambda a: compute(a), [], coders[0], out_coder)
    elif n == 2:
        return prim2(prim_name, lambda a, b: compute(a, b), [], coders[0], coders[1], out_coder)
    elif n == 3:
        return prim3(prim_name, lambda a, b, c: compute(a, b, c), [], coders[0], coders[1], coders[2], out_coder)
    else:
        raise ValueError(f"Primitive '{prim_name.value}': arity {n} exceeds max 3")


# ---------------------------------------------------------------------------
# Resolve equation → Hydra Primitive
# ---------------------------------------------------------------------------

def resolve_equation(eq_term: core.Term, backend: Backend) -> hydra.graph.Primitive:
    """Resolve an equation term into a Hydra Primitive.

    Reads the equation's fields, resolves its semiring against the backend,
    compiles the einsum, and produces a Primitive with the correct arity.
    Uses sort-aware TermCoders so the Primitive's type scheme reflects
    the actual domain/codomain sorts.
    """
    v, name, einsum_str, has_einsum, has_nl, nl_fn, in_coder, out_coder, prim_name, sr, eq = \
        _resolve_fields(eq_term, backend)

    # Determine param_slots (scalar hyperparameters before tensor inputs).
    # This is specific to resolve_equation and not shared with resolve_list_merge.
    param_slots = v.param_slots
    n_params = len(param_slots)

    if has_einsum:
        n_inputs = len(eq.input_vars)
    elif has_nl:
        n_inputs = 1
    else:
        raise ValueError(f"Equation '{name}' has neither einsum nor nonlinearity")

    total_arity = n_params + n_inputs
    if total_arity > 3:
        raise ValueError(
            f"Equation '{name}': total arity {total_arity} "
            f"({n_params} params + {n_inputs} tensor inputs) exceeds max 3"
        )

    # Build compute closure: params come first, then tensor inputs
    def _compute(*args):
        params_args = args[:n_params]
        tensor_args = list(args[n_params:])
        if has_einsum:
            r = semiring_contract(eq, tensor_args, sr, backend)
        else:
            r = tensor_args[0]
        if nl_fn:
            return nl_fn(r, *params_args) if params_args else nl_fn(r)
        return r

    coders = [float32_coder()] * n_params + [in_coder] * n_inputs
    return _make_prim(prim_name, _compute, coders, out_coder)


# ---------------------------------------------------------------------------
# List-merge resolution (for fan compositions)
# ---------------------------------------------------------------------------

def resolve_list_merge(eq_term: core.Term, backend: Backend) -> hydra.graph.Primitive:
    """Resolve a binary-einsum equation as a list-consuming merge Primitive.

    Used by fan compositions: the merge receives a list of branch outputs
    and folds the binary combiner pairwise over them.

    The equation must have a 2-input einsum (the binary combiner, e.g. "i,i->i").
    Produces a prim1: list<tensor> → tensor.
    """
    _, name, _einsum_str, has_einsum, has_nl, nl_fn, in_coder, out_coder, prim_name, sr, eq = \
        _resolve_fields(eq_term, backend)

    if has_einsum:
        n_inputs = len(eq.input_vars)

        if n_inputs == 2:
            def compute_list_merge(tensors):
                result = tensors[0]
                for t in tensors[1:]:
                    result = semiring_contract(eq, [result, t], sr, backend)
                if nl_fn:
                    result = nl_fn(result)
                return result
        elif n_inputs == 1:
            def compute_list_merge(tensors):
                if len(tensors) != 1:
                    raise ValueError(f"Unary merge '{name}' expects 1-element list, got {len(tensors)}")
                result = semiring_contract(eq, [tensors[0]], sr, backend)
                if nl_fn:
                    result = nl_fn(result)
                return result
        else:
            raise ValueError(f"List-merge equation '{name}': einsum must have 1 or 2 inputs, got {n_inputs}")

        return _make_prim(prim_name, compute_list_merge, [list_coder(in_coder)], out_coder)

    elif has_nl:
        def compute_list_nl(tensors):
            result = tensors[0]
            for t in tensors[1:]:
                result = result + t
            return nl_fn(result)

        return _make_prim(prim_name, compute_list_nl, [list_coder(in_coder)], out_coder)

    else:
        raise ValueError(f"List-merge equation '{name}' has neither einsum nor nonlinearity")
