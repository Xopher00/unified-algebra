"""Tensor equation declarations.

An equation is simultaneously a morphism (typed: domain → codomain) and a
tensor equation (einsum + semiring + optional nonlinearity). The Equation
class wraps a Hydra record term for declaration and field access.
Resolution (compilation to primitives) lives in unialg.assembly.pipeline.

    Equation(...)           → create an equation
    Equation.from_term(t)   → wrap an existing Hydra record term
"""

from __future__ import annotations

from dataclasses import dataclass
import hydra.core as core

from collections.abc import Callable
from typing import TYPE_CHECKING

from hydra.dsl.prims import prim1, prim2, prim3, float32 as float32_coder, list_ as list_coder
from hydra.graph import Primitive

from unialg.terms import _RecordView
from unialg.algebra.sort import sort_wrap
from unialg.algebra.semiring import Semiring
from unialg.algebra.contraction import compile_einsum, contract_and_apply, contract_merge

if TYPE_CHECKING:
    from unialg.backend import Backend


def _prepend_batch_dim(einsum_str: str) -> str:
    """Prepend a fresh batch dimension via the structured CompiledEinsum path."""
    if not einsum_str:
        return einsum_str
    return compile_einsum(einsum_str).prepend_batch_var().to_string()


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

    @dataclass(frozen=True, slots=True)
    class Compiled:
        has_einsum: bool
        has_nl: bool
        nl_fn: object
        in_coder: object
        out_coder: object
        prim_name: object
        sr: object
        compiled: object
        n_inputs: int
        n_params: int

    _type_name = core.Name("ua.equation.Equation")

    name          = _RecordView.Scalar(str)
    einsum        = _RecordView.Scalar(str, default="")
    domain_sort   = _RecordView.Term(key="domainSort", coerce=sort_wrap)
    codomain_sort = _RecordView.Term(key="codomainSort", coerce=sort_wrap)
    semiring      = _RecordView.Term(optional=True, coerce=Semiring.from_term)
    nonlinearity  = _RecordView.Scalar(str, default="")
    inputs        = _RecordView.ScalarList()
    param_slots   = _RecordView.ScalarList(key="paramSlots")

    @property
    def semiring_name(self) -> str | None:
        sr = self.semiring
        return sr.name if sr is not None else None

    @staticmethod
    def resolve_semirings(semirings: dict, backend) -> dict:
        return {name: sr.resolve(backend) for name, sr in semirings.items()}

    @property
    def prim_name(self) -> core.Name:
        return core.Name(f"ua.equation.{self.name}")

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

    def compile(self, backend):
        """Prepare this equation for resolution against a backend."""
        einsum_str = self.effective_einsum()
        has_einsum = bool(einsum_str)
        has_nl = bool(self.nonlinearity)
        if has_einsum:
            sr = self.semiring.resolve(backend)
            compiled = compile_einsum(einsum_str)
            n_inputs = len(compiled.input_vars)
        else:
            sr = compiled = None
            n_inputs = 0
        in_coder, out_coder = self.coders(backend)
        nl_fn = backend.unary(self.nonlinearity) if has_nl else None
        return Equation.Compiled(
            has_einsum=has_einsum, has_nl=has_nl, nl_fn=nl_fn,
            in_coder=in_coder, out_coder=out_coder,
            prim_name=self.prim_name, sr=sr, compiled=compiled,
            n_inputs=n_inputs, n_params=len(self.param_slots))

    _PRIMS = {1: prim1, 2: prim2, 3: prim3}

    @staticmethod
    def _make_prim(prim_name, compute, coders, out_coder) -> Primitive:
        n = len(coders)
        if n not in Equation._PRIMS:
            raise ValueError(f"Primitive '{prim_name.value}': packed arity {n} exceeds max 3")
        return Equation._PRIMS[n](prim_name, compute, [], *coders, out_coder)

    @staticmethod
    def _build_resolved(in_coder, n_params, n_inputs, sr, compiled, backend, nl_fn):
        def _core(params, tensors):
            return contract_and_apply(compiled, list(tensors), sr, backend, nl_fn, tuple(params))

        def native_fn(*args):
            return _core(args[:n_params], args[n_params:])

        if n_params + n_inputs <= 3:
            coders = [float32_coder()] * n_params + [in_coder] * n_inputs
            return coders, native_fn, native_fn

        coders = []
        if n_params > 0:
            coders.append(list_coder(float32_coder()))
        if n_inputs > 0:
            coders.append(list_coder(in_coder))

        def hydra_compute(*args):
            i = 0
            params = tuple(args[i]) if n_params > 0 else ()
            if n_params > 0:
                i += 1
            tensors = list(args[i]) if n_inputs > 0 else []
            return _core(params, tensors)

        return coders, hydra_compute, native_fn

    def resolve(self, backend: "Backend") -> tuple[Primitive, Callable, object, object]:
        ctx = self.compile(backend)
        if not ctx.has_einsum and not ctx.has_nl:
            raise ValueError(f"Equation '{self.name}' has neither einsum nor nonlinearity")
        n_inputs = 1 if not ctx.has_einsum else ctx.n_inputs
        coders, hydra_compute, native_fn = self._build_resolved(
            ctx.in_coder, ctx.n_params, n_inputs, ctx.sr, ctx.compiled, backend, ctx.nl_fn)
        prim = self._make_prim(ctx.prim_name, hydra_compute, coders, ctx.out_coder)
        return prim, native_fn, ctx.sr, ctx.in_coder

    def resolve_as_merge(self, backend: "Backend") -> tuple[Primitive, Callable, object, object]:
        ctx = self.compile(backend)
        if ctx.has_einsum:
            def compute_merge(tensors):
                return contract_merge(ctx.compiled, tensors, ctx.sr, backend, ctx.nl_fn, ctx.n_inputs, self.name)
            compute_merge.n_inputs = ctx.n_inputs
            prim = self._make_prim(ctx.prim_name, compute_merge, [list_coder(ctx.in_coder)], ctx.out_coder)
            return prim, compute_merge, ctx.sr, ctx.in_coder
        elif ctx.has_nl:
            def compute_nl(tensors):
                result = tensors[0]
                for t in tensors[1:]:
                    result = result + t
                return ctx.nl_fn(result)
            compute_nl.n_inputs = 1
            prim = self._make_prim(ctx.prim_name, compute_nl, [list_coder(ctx.in_coder)], ctx.out_coder)
            return prim, compute_nl, ctx.sr, ctx.in_coder
        else:
            raise ValueError(f"List-merge equation '{self.name}' has neither einsum nor nonlinearity")

