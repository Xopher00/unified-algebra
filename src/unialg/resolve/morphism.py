"""Tensor equations as Hydra Primitives.

An equation is simultaneously a morphism (typed: domain → codomain) and a
tensor equation (einsum + semiring + optional nonlinearity). The Equation
class unifies declaration, field access, and resolution in one object.

    Equation(...)           → create an equation
    Equation.from_term(t)   → wrap an existing Hydra record term
    eq.resolve(backend)     → compile to a Hydra Primitive
    eq.resolve_as_merge(b)  → compile as a list-consuming fan-merge Primitive

Compat aliases (public API unchanged):
    equation(...)           → Equation(...)
    resolve_equation(t, b)  → Equation.from_term(t).resolve(b)
    resolve_list_merge(t,b) → Equation.from_term(t).resolve_as_merge(b)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra.core as core
from hydra.core import Name
from hydra.dsl.meta.phantoms import record, string, list_, unit, TTerm
from hydra.dsl.prims import prim1, prim2, prim3, float32 as float32_coder, list_ as list_coder

from unialg.utils import record_fields, string_value
from unialg.views import _RecordView, _StringField, _TermField
from unialg.algebra.semiring import Semiring
from unialg.algebra.sort import sort_coder, is_batched
from unialg.resolve.contraction import compile_einsum, semiring_contract

if TYPE_CHECKING:
    from unialg.backend import Backend


# ---------------------------------------------------------------------------
# Batch-dimension helper
# ---------------------------------------------------------------------------

def _prepend_batch_dim(einsum_str: str) -> str:
    """Prepend a fresh batch dimension to every operand of an einsum string."""
    if not einsum_str:
        return einsum_str
    used = set(einsum_str) - {",", "-", ">"}
    batch_char = next(c for c in "bcdefghmnopqrstuvwxyz" if c not in used)
    lhs, rhs = einsum_str.split("->")
    inputs = lhs.split(",")
    batched_inputs = ",".join(batch_char + inp for inp in inputs)
    return f"{batched_inputs}->{batch_char + rhs}"


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
# Equation
# ---------------------------------------------------------------------------

EQUATION_TYPE_NAME = core.Name("ua.equation.Equation")


class Equation(_RecordView):
    """A tensor equation: declaration, field access, and resolution in one object.

    The underlying Hydra record term is the source of truth. Use .term to
    retrieve it for Hydra interop. Field access reads directly from the term
    on every call — there is no cached copy.

    Construct:
        eq = Equation("linear", "ij,j->i", hidden, output, real_sr)

    Wrap an existing term (e.g. from the parser):
        eq = Equation.from_term(term)
    """

    name         = _StringField("name")
    einsum       = _StringField("einsum")
    nonlinearity = _StringField("nonlinearity")
    domain_sort  = _TermField("domainSort")
    codomain_sort = _TermField("codomainSort")
    semiring      = _TermField("semiring")

    def __init__(
        self,
        name: str,
        einsum: str | None,
        domain_sort: core.Term,
        codomain_sort: core.Term,
        semiring_term: core.Term | None = None,
        nonlinearity: str | None = None,
        inputs: tuple[str, ...] = (),
        param_slots: tuple[str, ...] = (),
    ):
        super().__init__(record(EQUATION_TYPE_NAME, [
            Name("name") >> string(name),
            Name("einsum") >> string(einsum or ""),
            Name("domainSort") >> TTerm(domain_sort),
            Name("codomainSort") >> TTerm(codomain_sort),
            Name("semiring") >> (TTerm(semiring_term.term if isinstance(semiring_term, _RecordView) else semiring_term) if semiring_term is not None else unit()),
            Name("nonlinearity") >> string(nonlinearity or ""),
            Name("inputs") >> list_([string(n) for n in inputs]),
            Name("paramSlots") >> list_([string(p) for p in param_slots]),
        ]).value)

    @classmethod
    def from_term(cls, term) -> "Equation":
        """Wrap an existing Hydra record term as an Equation.

        Idempotent: if term is already an Equation, returns it unchanged.
        """
        if isinstance(term, cls):
            return term
        obj = cls.__new__(cls)
        obj._term = term
        return obj

    @property
    def inputs(self) -> list[str]:
        return [string_value(t) for t in record_fields(self._term)["inputs"].value]

    @property
    def param_slots(self) -> list[str]:
        ps = record_fields(self._term).get("paramSlots")
        if ps is None:
            return []
        if hasattr(ps, "value") and isinstance(ps.value, (list, tuple)):
            return [string_value(t) for t in ps.value]
        return []

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, backend: "Backend"):
        """Compile to a Hydra Primitive. Standard (non-merge) resolution."""
        einsum_str = self.einsum
        has_einsum = bool(einsum_str)
        if has_einsum and is_batched(self.domain_sort):
            einsum_str = _prepend_batch_dim(einsum_str)

        has_nl = bool(self.nonlinearity)
        nl_fn = backend.unary(self.nonlinearity) if has_nl else None
        in_coder = sort_coder(self.domain_sort, backend)
        out_coder = sort_coder(self.codomain_sort, backend)
        prim_name = core.Name(f"ua.equation.{self.name}")
        param_slots = self.param_slots
        n_params = len(param_slots)

        if has_einsum:
            sr = Semiring.from_term(self.semiring).resolve(backend)
            eq = compile_einsum(einsum_str)
            n_inputs = len(eq.input_vars)
        elif has_nl:
            n_inputs = 1
            sr = eq = None
        else:
            raise ValueError(f"Equation '{self.name}' has neither einsum nor nonlinearity")

        total_arity = n_params + n_inputs
        if total_arity > 3:
            raise ValueError(
                f"Equation '{self.name}': total arity {total_arity} "
                f"({n_params} params + {n_inputs} tensor inputs) exceeds max 3"
            )

        def _compute(*args):
            params_args = args[:n_params]
            tensor_args = list(args[n_params:])
            r = semiring_contract(eq, tensor_args, sr, backend) if has_einsum else tensor_args[0]
            if nl_fn:
                return nl_fn(r, *params_args) if params_args else nl_fn(r)
            return r

        coders = [float32_coder()] * n_params + [in_coder] * n_inputs
        return _make_prim(prim_name, _compute, coders, out_coder)

    def resolve_as_merge(self, backend: "Backend"):
        """Compile as a list-consuming merge Primitive (for fan compositions)."""
        einsum_str = self.einsum
        has_einsum = bool(einsum_str)
        if has_einsum and is_batched(self.domain_sort):
            einsum_str = _prepend_batch_dim(einsum_str)
        has_nl = bool(self.nonlinearity)
        nl_fn = backend.unary(self.nonlinearity) if has_nl else None
        in_coder = sort_coder(self.domain_sort, backend)
        out_coder = sort_coder(self.codomain_sort, backend)
        prim_name = core.Name(f"ua.equation.{self.name}")
        name = self.name

        if has_einsum:
            sr = Semiring.from_term(self.semiring).resolve(backend)
            eq = compile_einsum(einsum_str)
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
                        raise ValueError(
                            f"Unary merge '{name}' expects 1-element list, got {len(tensors)}")
                    result = semiring_contract(eq, [tensors[0]], sr, backend)
                    if nl_fn:
                        result = nl_fn(result)
                    return result
            else:
                raise ValueError(
                    f"List-merge equation '{name}': einsum must have 1 or 2 inputs, got {n_inputs}")

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


def resolve_all_primitives(
    eq_terms: list[core.Term],
    backend: "Backend",
    merge_names: set[str],
) -> tuple[dict, dict[str, core.Term]]:
    """Resolve equations to primitives and build eq_by_name lookup."""
    from hydra.sources.libraries import standard_library
    primitives = dict(standard_library())
    equations = [Equation.from_term(eq) for eq in eq_terms]
    eq_by_name: dict[str, Equation] = {eq.name: eq for eq in equations}
    for eq in equations:
        prim = eq.resolve_as_merge(backend) if eq.name in merge_names else eq.resolve(backend)
        primitives[prim.name] = prim
    return primitives, eq_by_name
