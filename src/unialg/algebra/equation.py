"""Tensor equation declarations.

An equation is simultaneously a morphism (typed: domain → codomain) and a
tensor equation (einsum + semiring + optional nonlinearity). The Equation
class wraps a Hydra record term for declaration and field access.
Resolution (compilation to primitives) lives in unialg.assembly._equation_resolution.

    Equation(...)           → create an equation
    Equation.from_term(t)   → wrap an existing Hydra record term
"""

from __future__ import annotations

from hydra.core import Name

from unialg.terms import _RecordView
from .sort import sort_wrap, ProductSort
from .semiring import Semiring
from .contraction import compile_einsum


def _prepend_batch_dim(einsum_str: str) -> str:
    """Prepend a fresh batch dimension via the structured CompiledEinsum path."""
    if not einsum_str:
        return einsum_str
    return compile_einsum(einsum_str).prepend_batch_var().to_string()


# ---------------------------------------------------------------------------
# Equation
# ---------------------------------------------------------------------------

class Equation(_RecordView):
    """A tensor equation declaration — the unified-algebra primitive morphism.

    Equation is a declarative spec (name, einsum, domain/codomain sorts,
    semiring, optional nonlinearity). At resolution time
    (``unialg.assembly._equation_resolution.resolve_equation``) it lowers to
    a Hydra ``Primitive`` via ``hydra.dsl.prims.prim1`` / ``prim2`` /
    ``prim3``, with arity > 3 packed into list-coders.

    Boundary:

    - **unified-algebra owns:** the einsum spec, the semiring resolution,
      the contraction execution (``unialg.algebra.contraction``).
    - **Hydra owns:** the ``Primitive`` wrapper, the ``TypeScheme`` (concrete
      function type — currently monomorphic, ``[]`` variables list), graph
      registration, dispatch via ``lookup_primitive``.

    The underlying Hydra record term is the source of truth. Use ``.term``
    to retrieve it for Hydra interop. Field access reads directly from the
    term on every call — there is no cached copy.

    See ``ARCHITECTURE.md`` § "Hydra ↔ unified-algebra boundary".

    Construct:
        eq = Equation("linear", "ij,j->i", hidden, output, real_sr)

    Wrap an existing term (e.g. from the parser):
        eq = Equation.from_term(term)
    """

    _type_name = Name("ua.equation.Equation")

    name          = _RecordView.Scalar(str)
    einsum        = _RecordView.Scalar(str, default="")
    domain_sort   = _RecordView.Term(key="domainSort", coerce=sort_wrap)
    codomain_sort = _RecordView.Term(key="codomainSort", coerce=sort_wrap)
    semiring      = _RecordView.Term(optional=True, coerce=Semiring.from_term)
    nonlinearity  = _RecordView.Scalar(str, default="")
    inputs        = _RecordView.ScalarList()
    param_slots   = _RecordView.ScalarList(key="paramSlots")
    adjoint       = _RecordView.Scalar(bool, default=False)
    skip          = _RecordView.Scalar(bool, default=False)

    @property
    def semiring_name(self) -> str | None:
        sr = self.semiring
        return sr.name if sr is not None else None

    @property
    def prim_name(self) -> Name:
        return Name(f"ua.equation.{self.name}")

    @property
    def output_rank(self) -> int | None:
        es = self.einsum
        if not es:
            return None
        return len(compile_einsum(es).output_vars)

    def input_rank(self, slot: int) -> int | None:
        es = self.einsum
        if not es:
            return None
        ivars = compile_einsum(es).input_vars
        return len(ivars[slot]) if slot < len(ivars) else None

    def effective_einsum(self) -> str:
        es = self.einsum
        if es and self.domain_sort.batched:
            return _prepend_batch_dim(es)
        return es

    def coders(self, backend):
        return (self.domain_sort.coder(backend),
                self.codomain_sort.coder(backend))

    def register_sorts(self, schema: dict) -> None:
        self.domain_sort.register_schema(schema)
        self.codomain_sort.register_schema(schema)

    def validate_axes(self) -> None:
        if isinstance(self.codomain_sort, ProductSort) or isinstance(self.domain_sort, ProductSort):
            # Axis validation against einsum ranks is not yet implemented for
            # ProductSort.  Raise only if axes are actually declared on any
            # component — if no axes are present there is nothing to validate.
            def _has_axes(s):
                if isinstance(s, ProductSort):
                    return any(_has_axes(e) for e in s.elements)
                return bool(getattr(s, "axes", ()))
            if _has_axes(self.codomain_sort) or _has_axes(self.domain_sort):
                raise NotImplementedError(
                    f"Axis validation for ProductSort not implemented (equation '{self.name}')")
            return
        es = self.einsum
        if not es:
            return
        compiled = compile_einsum(es)
        cod_axes = self.codomain_sort.axes
        if cod_axes:
            out_rank = len(compiled.output_vars)
            if out_rank != len(cod_axes):
                raise TypeError(
                    f"Op '{self.name}': einsum output rank {out_rank} != "
                    f"codomain sort '{self.codomain_sort.name}' axes {list(cod_axes)} (rank {len(cod_axes)})")
        dom_axes = self.domain_sort.axes
        if dom_axes:
            last_input = compiled.input_vars[-1]
            if len(last_input) != len(dom_axes):
                raise TypeError(
                    f"Op '{self.name}': einsum last-input rank {len(last_input)} != "
                    f"domain sort '{self.domain_sort.name}' axes {list(dom_axes)} (rank {len(dom_axes)})")