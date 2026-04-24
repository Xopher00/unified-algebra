"""Tensor equation declarations.

An equation is simultaneously a morphism (typed: domain → codomain) and a
tensor equation (einsum + semiring + optional nonlinearity). The Equation
class wraps a Hydra record term for declaration and field access.
Resolution (compilation to primitives) lives in unialg.assembly.pipeline.

    Equation(...)           → create an equation
    Equation.from_term(t)   → wrap an existing Hydra record term
"""

from __future__ import annotations

import hydra.core as core
from hydra.core import Name
from hydra.dsl.meta.phantoms import record, string, list_, unit, TTerm

from unialg.terms import record_fields, literal_value
from unialg.terms import _RecordView, _TermField, _ScalarField
from unialg.algebra.sort import sort_wrap
from unialg.algebra.semiring import Semiring
from unialg.algebra.contraction import compile_einsum


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


# ---------------------------------------------------------------------------
# Equation
# ---------------------------------------------------------------------------

class Equation(_RecordView):
    """A tensor equation declaration.

    The underlying Hydra record term is the source of truth. Use .term to
    retrieve it for Hydra interop. Field access reads directly from the term
    on every call — there is no cached copy.

    Construct:
        eq = Equation("linear", "ij,j->i", hidden, output, real_sr)

    Wrap an existing term (e.g. from the parser):
        eq = Equation.from_term(term)
    """

    _type_name = core.Name("ua.equation.Equation")

    name         = _ScalarField("name", str)
    einsum       = _ScalarField("einsum", str)
    nonlinearity = _ScalarField("nonlinearity", str)
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
        super().__init__(record(self._type_name, [
            Name("name") >> string(name),
            Name("einsum") >> string(einsum or ""),
            Name("domainSort") >> TTerm(self._unwrap(domain_sort)),
            Name("codomainSort") >> TTerm(self._unwrap(codomain_sort)),
            Name("semiring") >> (TTerm(self._unwrap(semiring_term)) if semiring_term is not None else unit()),
            Name("nonlinearity") >> string(nonlinearity or ""),
            Name("inputs") >> list_([string(n) for n in inputs]),
            Name("paramSlots") >> list_([string(p) for p in param_slots]),
        ]).value)

    @property
    def inputs(self) -> list[str]:
        return [literal_value(t) for t in record_fields(self._term)["inputs"].value]

    @property
    def param_slots(self) -> list[str]:
        return [literal_value(t) for t in record_fields(self._term)["paramSlots"].value]

    @property
    def semiring_name(self) -> str | None:
        sr = self.semiring
        if not isinstance(sr, core.TermRecord):
            return None
        return Semiring.from_term(sr).name

    @staticmethod
    def resolve_semiring_term(sr_term, backend):
        return Semiring.from_term(sr_term).resolve(backend)

    @property
    def prim_name(self) -> core.Name:
        return core.Name(f"ua.equation.{self.name}")

    def effective_einsum(self) -> str:
        es = self.einsum
        if es and sort_wrap(self.domain_sort).batched:
            return _prepend_batch_dim(es)
        return es

    def coders(self, backend):
        return (sort_wrap(self.domain_sort).coder(backend),
                sort_wrap(self.codomain_sort).coder(backend))

    def register_sorts(self, schema: dict) -> None:
        sort_wrap(self.domain_sort).register_schema(schema)
        sort_wrap(self.codomain_sort).register_schema(schema)

    def compile(self, backend):
        """Prepare this equation for resolution against a backend."""
        einsum_str = self.effective_einsum()
        has_einsum = bool(einsum_str)
        has_nl = bool(self.nonlinearity)
        if has_einsum:
            sr = Semiring.from_term(self.semiring).resolve(backend)
            compiled = compile_einsum(einsum_str)
            n_inputs = len(compiled.input_vars)
        else:
            sr = compiled = None
            n_inputs = 0
        in_coder, out_coder = self.coders(backend)
        nl_fn = backend.unary(self.nonlinearity) if has_nl else None
        return (has_einsum, has_nl, nl_fn, in_coder, out_coder,
                self.prim_name, sr, compiled, n_inputs, len(self.param_slots))

