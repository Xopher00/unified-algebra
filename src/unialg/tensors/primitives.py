"""Tensor contraction primitive leaves and substrate composition helpers.

This module is tensor-owned structure/runtime-boundary glue. It does not
evaluate contractions directly; it builds ordinary ``Morphism`` trees whose
leaves are backend primitives where native tensor operations are required.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import count

from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import Nothing
from hydra.graph import Graph, Primitive, TermCoder

from unialg.objects import BINARY, ExpType, TypeScheme
from unialg.runtime import BackendOps, coder_for_type
from unialg.runtime.codecs import expect_right
from unialg.semantics import morphisms as ops
from unialg.semantics.morphisms import Morphism, MorphismError
from unialg.syntax import expressions as expr

from .notation import AlignmentPlan
from .semantics import ContractSpec


_GENSYM = count()


@dataclass(frozen=True)
class TensorPrimitiveContext:
    """Backend-native state needed to build tensor primitive leaves."""

    backend: BackendOps

    @property
    def store(self):
        if self.backend.store is None:
            raise MorphismError("tensor contraction requires a BINARY RuntimeStore")
        return self.backend.store

    def structural(self, name: str) -> Callable:
        try:
            return self.backend.structural_op(name)
        except KeyError as e:
            raise MorphismError(f"backend is missing structural op {name!r}") from e

    def backend_primitive_for(self, morphism: Morphism):
        node = morphism.node
        if not isinstance(node, expr.BackendPrim):
            raise MorphismError("tensor contraction expected backend primitive morphism")
        for bp in self.backend.primitives.values():
            if bp.primitive.name == node.primitive.name:
                return bp
        raise MorphismError(f"backend primitive {node.primitive.name!r} is not in active backend")


def context_from(value) -> TensorPrimitiveContext:
    """Normalize the opaque domain context into a tensor primitive context."""
    if isinstance(value, TensorPrimitiveContext):
        return value
    if isinstance(value, BackendOps):
        return TensorPrimitiveContext(value)
    raise MorphismError("tensor contraction requires active backend context")


def _leaf_morphism(name_prefix: str, impl, coder: TermCoder) -> Morphism:
    prim_name = Name(f"unialg.tensor.{name_prefix}.{next(_GENSYM)}")
    prim = Primitive(
        prim_name,
        TypeScheme((), ExpType(BINARY, BINARY), Nothing()),
        impl,
    )
    return Morphism(
        node=expr.BackendPrim(prim, 1, BINARY, BINARY),
        aux_primitives=(prim,),
    )


def align_morphism(plan: AlignmentPlan, ctx: TensorPrimitiveContext) -> Morphism:
    """Return a BINARY -> BINARY morphism applying a baked alignment plan."""
    store = ctx.store
    expand_dims = ctx.structural("expand_dims")
    transpose = ctx.structural("transpose")
    coder = coder_for_type(BINARY)

    def impl(
        hctx: Context,
        graph: Graph,
        args,
        *,
        store=store,
        expand_dims=expand_dims,
        transpose=transpose,
        plan=plan,
        coder=coder,
    ):
        key = expect_right(coder.encode(hctx, graph, args[0]), "tensor align decode")
        arr = store.get(key)
        for axis in plan.unsqueeze_axes:
            arr = expand_dims(arr, axis)
        arr = transpose(arr, plan.perm)
        return coder.decode(hctx, store.put(arr))

    return _leaf_morphism("align", impl, coder)


def _call_reduce(fn: Callable, arr, axes: tuple[int, ...]):
    try:
        return fn(arr, axis=axes)
    except TypeError:
        return fn(arr, dim=axes)


def axis_reduce_morphism(fold: Morphism, axes: tuple[int, ...], ctx: TensorPrimitiveContext) -> Morphism:
    """Return a BINARY -> BINARY morphism reducing over baked axes."""
    store = ctx.store
    fold_bp = ctx.backend_primitive_for(fold)
    coder = coder_for_type(BINARY)

    def impl(
        hctx: Context,
        graph: Graph,
        args,
        *,
        store=store,
        fn=fold_bp.fn,
        axes=axes,
        coder=coder,
    ):
        key = expect_right(coder.encode(hctx, graph, args[0]), "tensor reduce decode")
        arr = store.get(key)
        return coder.decode(hctx, store.put(_call_reduce(fn, arr, axes)))

    return _leaf_morphism("reduce", impl, coder)


def par_all(items: list[Morphism]) -> Morphism:
    """Left-nested parallel product matching repeated_product."""
    if not items:
        raise MorphismError("par_all requires at least one morphism")
    out = items[0]
    for item in items[1:]:
        out = ops.par(out, item)
    return out


def fold_product(product: Morphism, n: int) -> Morphism:
    """Left-fold an n-ary BINARY product with a binary product morphism."""
    if n < 1:
        raise MorphismError("fold_product requires at least one input")
    if n == 1:
        return ops.identity(BINARY)
    out = product
    for _ in range(3, n + 1):
        out = ops.compose(ops.par(out, ops.identity(BINARY)), product)
    return out


def compile_contract_spec(spec: ContractSpec, context) -> Morphism:
    """Compile a tensor ContractSpec into an ordinary substrate Morphism tree."""
    ctx = context_from(context)
    eq = spec.equation
    env = spec.semiring.op_env(adjoint=spec.adjoint)

    alignments = [
        align_morphism(eq.alignment_plan(i), ctx)
        for i in range(len(eq.inputs))
    ]
    aligned = par_all(alignments)
    product = fold_product(env["product"], len(eq.inputs))
    contraction = ops.compose(aligned, product)

    if eq.reduced:
        axes = tuple(range(len(eq.output), len(eq.output) + len(eq.reduced)))
        contraction = ops.compose(contraction, axis_reduce_morphism(env["fold"], axes, ctx))

    return contraction
