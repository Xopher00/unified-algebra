"""Tensor contraction primitive leaves and substrate composition helpers.

This module is tensor-owned structure/runtime-boundary glue. It does not
evaluate contractions directly; it builds ordinary ``Morphism`` trees whose
leaves are backend primitives where native tensor operations are required.

No imports from ``unialg.runtime`` — all runtime state is accessed via the
duck-typed domain context passed through from ``main.py``.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import count

from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import Nothing, Right
import hydra.show.errors as ShowErrors
from hydra.graph import Graph, Primitive

from unialg.objects import BINARY, ExpType, TypeScheme
from unialg.semantics import morphisms as ops
from unialg.semantics.morphisms import Morphism, MorphismError
from unialg.syntax import expressions as expr

from .notation import AlignmentPlan
from .semantics import ContractSpec


_GENSYM = count()


def _get_binary_coder(ctx):
    """Extract the BINARY TermCoder from any registered backend primitive."""
    for bp in ctx.primitives.values():
        return bp.arg_coder
    raise MorphismError("no backend primitives registered")


def _callable_for(morphism: Morphism, ctx) -> Callable:
    """Extract the raw Python callable backing a backend primitive morphism."""
    node = morphism.node
    if not isinstance(node, expr.BackendPrim):
        raise MorphismError("tensor contraction expected backend primitive morphism")
    for bp in ctx.primitives.values():
        if bp.primitive.name == node.primitive.name:
            return bp.fn
    raise MorphismError(f"no backend callable for {node.primitive.name!r}")


def context_from(value):
    """Validate the opaque domain context has the required interface."""
    if hasattr(value, 'structural_op') and hasattr(value, 'store'):
        return value
    raise MorphismError("tensor contraction requires active backend context")


def _leaf_morphism(name_prefix: str, impl) -> Morphism:
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


def align_morphism(plan: AlignmentPlan, ctx) -> Morphism:
    """Return a BINARY -> BINARY morphism applying a baked alignment plan."""
    store = ctx.store
    expand_dims = ctx.structural_op("expand_dims")
    transpose = ctx.structural_op("transpose")
    coder = _get_binary_coder(ctx)

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
        result = coder.encode(hctx, graph, args[0])
        if not isinstance(result, Right):
            raise TypeError(f"tensor align decode: {ShowErrors.error(result.value)}")
        arr = store.get(result.value)
        for axis in plan.unsqueeze_axes:
            arr = expand_dims(arr, axis)
        arr = transpose(arr, plan.perm)
        return coder.decode(hctx, store.put(arr))

    return _leaf_morphism("align", impl)


def _adjust_diagonal_axes(raw_pairs: list[tuple[int, int]], ndim: int) -> list[tuple[int, int]]:
    """Pre-compute adjusted axis pairs for iterative np.diagonal calls.

    After each np.diagonal(arr, axis1=a1, axis2=a2):
    - axes a1 and a2 are removed
    - the diagonal is appended as the last axis
    """
    axis_map = list(range(ndim))
    adjusted = []
    current_ndim = ndim

    for a1_orig, a2_orig in raw_pairs:
        a1_cur = axis_map[a1_orig]
        a2_cur = axis_map[a2_orig]
        if a1_cur is None or a2_cur is None:
            raise MorphismError(f"diagonal axis already consumed: {a1_orig}, {a2_orig}")
        if a1_cur > a2_cur:
            a1_cur, a2_cur = a2_cur, a1_cur
        adjusted.append((a1_cur, a2_cur))

        new_ndim = current_ndim - 1
        for i in range(len(axis_map)):
            p = axis_map[i]
            if p is None:
                continue
            if p == a1_cur or p == a2_cur:
                axis_map[i] = None
            elif p > a2_cur:
                axis_map[i] = p - 2
            elif p > a1_cur:
                axis_map[i] = p - 1
        axis_map[a1_orig] = new_ndim - 1
        current_ndim = new_ndim

    return adjusted


def _call_diagonal(fn: Callable, arr, a1: int, a2: int):
    """Call a diagonal function, adapting for torch (dim1/dim2) vs numpy (axis1/axis2)."""
    try:
        return fn(arr, axis1=a1, axis2=a2)
    except TypeError:
        return fn(arr, dim1=a1, dim2=a2)


def diagonal_extract_morphism(diag_axes: list[tuple[int, int]], ctx) -> Morphism:
    """Return a BINARY -> BINARY morphism extracting diagonals for repeated labels."""
    store = ctx.store
    take_diag = ctx.structural_op("take_diagonal")
    coder = _get_binary_coder(ctx)

    def impl(
        hctx: Context,
        graph: Graph,
        args,
        *,
        store=store,
        take_diag=take_diag,
        axes=diag_axes,
        coder=coder,
    ):
        result = coder.encode(hctx, graph, args[0])
        if not isinstance(result, Right):
            raise TypeError(f"tensor diagonal decode: {ShowErrors.error(result.value)}")
        arr = store.get(result.value)
        for a1, a2 in axes:
            arr = _call_diagonal(take_diag, arr, a1, a2)
        return coder.decode(hctx, store.put(arr))

    return _leaf_morphism("diag", impl)


def _call_reduce(fn: Callable, arr, axes: tuple[int, ...]):
    try:
        return fn(arr, axis=axes)
    except TypeError:
        return fn(arr, dim=axes)


def axis_reduce_morphism(fold: Morphism, axes: tuple[int, ...], ctx) -> Morphism:
    """Return a BINARY -> BINARY morphism reducing over baked axes."""
    store = ctx.store
    fold_fn = _callable_for(fold, ctx)
    coder = _get_binary_coder(ctx)

    def impl(
        hctx: Context,
        graph: Graph,
        args,
        *,
        store=store,
        fn=fold_fn,
        axes=axes,
        coder=coder,
    ):
        result = coder.encode(hctx, graph, args[0])
        if not isinstance(result, Right):
            raise TypeError(f"tensor reduce decode: {ShowErrors.error(result.value)}")
        arr = store.get(result.value)
        return coder.decode(hctx, store.put(_call_reduce(fn, arr, axes)))

    return _leaf_morphism("reduce", impl)


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


def _build_alignment_tree(alignments: list, shape) -> Morphism:
    """Build par-tree of alignments matching the polynomial shape."""
    from unialg.syntax.expressions import Prod, Id
    it = iter(alignments)

    def _walk(s):
        if isinstance(s, Prod):
            left = _walk(s.left)
            right = _walk(s.right)
            return ops.par(left, right)
        if isinstance(s, Id):
            return next(it)
        raise MorphismError(f"unexpected shape node in alignment tree: {type(s).__name__}")

    result = _walk(shape)
    remaining = list(it)
    if remaining:
        raise MorphismError(
            f"alignment/shape mismatch: {len(remaining)} alignments unconsumed"
        )
    return result


def _build_fold_tree(product: Morphism, shape) -> Morphism:
    """Build element-wise product fold matching the polynomial shape."""
    from unialg.syntax.expressions import Prod, Id
    if isinstance(shape, Id):
        return ops.identity(BINARY)
    if isinstance(shape, Prod):
        left = _build_fold_tree(product, shape.left)
        right = _build_fold_tree(product, shape.right)
        return ops.compose(ops.par(left, right), product)
    raise MorphismError(f"unexpected shape node in fold tree: {type(shape).__name__}")


def compile_contract_spec(spec: ContractSpec, context) -> Morphism:
    """Compile a tensor ContractSpec into an ordinary substrate Morphism tree."""
    ctx = context_from(context)
    eq = spec.equation
    env = spec.semiring.op_env(adjoint=spec.adjoint)

    alignments = []
    for i in range(len(eq.inputs)):
        diag_axes = eq.diagonal_axes(i)
        align_m = align_morphism(eq.alignment_plan(i), ctx)
        if diag_axes:
            adjusted = _adjust_diagonal_axes(diag_axes, len(eq.inputs[i]))
            diag_m = diagonal_extract_morphism(adjusted, ctx)
            alignments.append(ops.compose(diag_m, align_m))
        else:
            alignments.append(align_m)

    if spec.shape is not None:
        aligned = _build_alignment_tree(alignments, spec.shape)
        product = _build_fold_tree(env["product"], spec.shape)
    else:
        aligned = par_all(alignments)
        product = fold_product(env["product"], len(eq.inputs))

    contraction = ops.compose(aligned, product)

    if eq.reduced:
        axes = tuple(range(len(eq.output), len(eq.output) + len(eq.reduced)))
        contraction = ops.compose(contraction, axis_reduce_morphism(env["fold"], axes, ctx))

    return contraction
