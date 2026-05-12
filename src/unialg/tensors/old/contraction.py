"""Structure layer: Hydra term IR for semiring tensor contractions.

Responsibilities (mirrors structure/realize.py + structure/terms.py for tensors):
  - Symbolic contract-node constructors and matchers.
  - Fusion rewrite rule: same-semiring adjacent contractions collapse to one.
  - Pure-Python contraction execution kernel.
  - Primitive registration: kernel → Hydra Primitive → BackendPrimitive.

Nothing here knows about typed ``Equation`` invariants.  It consumes
``EinsumExpr`` (syntax) and ``Semiring`` (semantics) values; the typed
coordination is the responsibility of ``tensors.lowering``.
"""

from __future__ import annotations

from functools import reduce
from typing import Callable, Any

import hydra.core as core
import hydra.rewriting as Rewriting
import hydra.dsl.meta.phantoms as P
from hydra.dsl.python import Right


# ---------------------------------------------------------------------------
# Hydra term helpers (local, not exported)
# ---------------------------------------------------------------------------

_CONTRACT = core.Name("unialg.tensor.contract")


def _hname(s: str) -> core.Name:
    return core.Name(s)


def _app(f, x):
    return core.TermApplication(core.Application(f, x))


def _appn(f, *args):
    t = f
    for a in args:
        t = _app(t, a)
    return t


def _wrap_str(s: str):
    return P.string(str(s)).value


def _unwrap_str(t) -> str:
    v = t
    while hasattr(v, "value"):
        v = v.value
    return v


def _get_parts(t):
    """Return (function, argument) if t is a TermApplication, else None."""
    if type(t).__name__ != "TermApplication":
        return None
    v = getattr(t, "value", None)
    if v is None:
        return None
    f = getattr(v, "function", None)
    x = getattr(v, "argument", None)
    if f is None or x is None:
        return None
    return f, x


def _flatten(t) -> tuple[object, list[object]]:
    """Flatten a left-curried application into (head, [arg0, arg1, ...])."""
    args = []
    cur = t
    while True:
        parts = _get_parts(cur)
        if parts is None:
            break
        cur, arg = parts
        args.append(arg)
    return cur, list(reversed(args))


# ---------------------------------------------------------------------------
# Contract term constructors and matchers
# ---------------------------------------------------------------------------

def contract_term(semiring_name: str, expr: EinsumExpr, *operand_terms) -> object:
    """Build a symbolic contraction node in the Hydra term IR.

    Shape: ``CONTRACT semiring_name subscript A B ...``

    ``semiring_name`` is wrapped as a Hydra string for identity comparison
    during fusion.  ``subscript`` is the canonical einsum string for ``expr``.
    """
    return _appn(
        core.TermVariable(_CONTRACT),
        _wrap_str(semiring_name),
        _wrap_str(str(expr)),
        *operand_terms,
    )


def match_contract(t) -> tuple[str, EinsumExpr, list] | None:
    """Match a contract node; return ``(semiring_name, EinsumExpr, operands)`` or None."""
    head, args = _flatten(t)
    if not (
        isinstance(head, core.TermVariable)
        and head.value == _CONTRACT
        and len(args) >= 3
    ):
        return None
    semiring_name = _unwrap_str(args[0])
    subscript = _unwrap_str(args[1])
    operands = args[2:]
    try:
        expr = parse_einsum(subscript)
    except ValueError:
        return None
    return semiring_name, expr, operands


# ---------------------------------------------------------------------------
# Fusion rewrite
# ---------------------------------------------------------------------------

def _fuse_once(outer_sr: str, outer_expr: EinsumExpr, outer_ops: list) -> tuple | None:
    """Try to fuse one inner contract node into the outer.

    Returns ``(new_expr, new_ops)`` if a fusion was found, else ``None``.
    The first fusible inner is consumed; call iteratively to fuse all.
    """
    for pos, op in enumerate(outer_ops):
        inner = match_contract(op)
        if inner is None:
            continue
        inner_sr, inner_expr, inner_ops = inner
        if inner_sr != outer_sr:
            continue

        # inner.output must equal outer.inputs[pos] for valid substitution.
        if inner_expr.output != outer_expr.inputs[pos]:
            continue

        # Build fused subscript: replace operand at pos with inner's inputs.
        fused_inputs = (
            outer_expr.inputs[:pos]
            + inner_expr.inputs
            + outer_expr.inputs[pos + 1:]
        )
        fused_expr = EinsumExpr(inputs=fused_inputs, output=outer_expr.output)
        fused_ops = outer_ops[:pos] + list(inner_ops) + outer_ops[pos + 1:]
        return fused_expr, fused_ops

    return None


def fuse_contract_rule(recurse, t):
    """Bottom-up rewrite rule: fuse same-semiring adjacent contractions.

    Compatible with ``hydra.rewriting.rewrite_term``.
    """
    t = recurse(t)
    matched = match_contract(t)
    if matched is None:
        return t
    sr_name, outer_expr, outer_ops = matched

    while True:
        result = _fuse_once(sr_name, outer_expr, outer_ops)
        if result is None:
            break
        outer_expr, outer_ops = result

    return contract_term(sr_name, outer_expr, *outer_ops)


# ---------------------------------------------------------------------------
# Contraction execution kernel
# ---------------------------------------------------------------------------

def _expand_to_full(arr, sub: str, all_chars: list[str], dim: dict[str, int]):
    """Expand arr (with subscript chars ``sub``) to the full ``all_chars`` shape.

    1. Transpose arr so its existing axes align with the ``all_chars`` ordering.
    2. Insert unit axes for each char not in ``sub``.
    """
    import numpy as np

    sub_list = list(sub)
    # Reorder arr axes to match all_chars order (for chars in sub only)
    ordered = [c for c in all_chars if c in sub_list]
    if ordered != sub_list:
        perm = [sub_list.index(c) for c in ordered]
        arr = np.transpose(arr, perm)

    # Insert new axes for missing chars
    for i, c in enumerate(all_chars):
        if c not in sub_list:
            arr = np.expand_dims(arr, axis=i)

    return arr


def _fold_reduce(plus_fn: Callable, arr, axes: tuple[int, ...]):
    """Reduce arr over axes using binary plus_fn via a Python fold.

    Axes are processed in descending order so earlier reductions don't shift
    the remaining axis indices.
    """
    result = arr
    for ax in sorted(axes, reverse=True):
        slices = [slice(None)] * result.ndim
        acc = None
        for i in range(result.shape[ax]):
            slices[ax] = i
            elem = result[tuple(slices)]
            acc = elem if acc is None else plus_fn(acc, elem)
        result = acc
    return result


def contraction_kernel(
    times_fn: Callable,
    plus_fn: Callable,
    expr: EinsumExpr,
    *arrays: Any,
) -> Any:
    """Execute a semiring-parameterized tensor contraction on Python arrays.

    Uses numpy broadcasting:
    1. Expand each input to the full index space.
    2. Elementwise ⊗ across all operands via left-fold of times_fn.
    3. Reduce ⊕ over contracted indices via fold of plus_fn.
    4. Permute output axes to match ``expr.output`` order.

    ``times_fn`` and ``plus_fn`` must be element-wise binary callables that
    broadcast over numpy-compatible arrays (e.g. ``numpy.multiply``,
    ``numpy.minimum``).
    """
    import numpy as np

    if len(arrays) != len(expr.inputs):
        raise ValueError(
            f"contraction_kernel: expected {len(expr.inputs)} arrays, "
            f"got {len(arrays)}"
        )

    # Collect all index characters, preserving first-appearance order.
    all_chars: list[str] = []
    for sub in expr.inputs:
        for c in sub:
            if c not in all_chars:
                all_chars.append(c)

    # Map each character to its dimension.
    dim: dict[str, int] = {}
    for sub, arr in zip(expr.inputs, arrays):
        for c, size in zip(sub, arr.shape):
            dim[c] = size

    # Contracted indices: in inputs but not in output.
    out_set = set(expr.output)
    contract_chars = [c for c in all_chars if c not in out_set]

    # Expand and multiply.
    expanded = [_expand_to_full(arr, sub, all_chars, dim)
                for sub, arr in zip(expr.inputs, arrays)]
    product = reduce(times_fn, expanded)

    # Reduce over contracted axes.
    if contract_chars:
        contract_axes = tuple(all_chars.index(c) for c in contract_chars)
        product = _fold_reduce(plus_fn, product, contract_axes)

    # Permute to match expr.output (output chars now at leading axes in all_chars order).
    remaining = [c for c in all_chars if c in out_set]
    if remaining != list(expr.output):
        perm = [remaining.index(c) for c in expr.output]
        product = np.transpose(product, perm)

    return product


# ---------------------------------------------------------------------------
# Primitive registration
# ---------------------------------------------------------------------------

def _resolve_morphism_fn(morphism, backend_ops) -> Callable:
    """Extract the raw Python callable from a Morphism built by BackendOps.

    Navigates through ``morphism.aux_primitives`` to find the Hydra primitive
    name, then looks it up in ``backend_ops``.
    """
    if not morphism.aux_primitives:
        raise ValueError("Morphism carries no aux_primitives; cannot resolve backend fn")
    prim_name = morphism.aux_primitives[0].name
    return backend_ops.get_fn_by_prim_name(prim_name)


def register_contraction_primitive(
    name: str,
    expr: EinsumExpr,
    semiring,
    backend_ops,
    *,
    use_adjoint: bool = False,
    nonlinearity_fn: Callable | None = None,
):
    """Build and register a contraction kernel as a Hydra ``BackendPrimitive``.

    Parameters
    ----------
    name:
        Canonical Hydra primitive name (e.g. ``"unialg.tensor.matmul"``).
    expr:
        The fused ``EinsumExpr`` describing the contraction.
    semiring:
        The ``Semiring`` providing ⊕ and ⊗.
    backend_ops:
        The ``BackendOps`` from which the semiring's component morphisms were
        built; used to resolve raw Python callables.
    use_adjoint:
        When True, use ``semiring.adjoint`` instead of ``semiring.times`` for
        the inner product step.
    nonlinearity_fn:
        Optional Python callable applied elementwise to the result array.

    Returns
    -------
    BackendPrimitive
        Ready for inclusion in a Morphism's ``aux_primitives``.
    """
    from unialg.tensors.math import (
        register_raw_primitive, TENSOR_TYPE, unwrap_tensor, wrap_tensor,
    )

    plus_fn = _resolve_morphism_fn(semiring.plus, backend_ops)

    if use_adjoint:
        if semiring.adjoint is None:
            raise ValueError(
                f"Equation uses adjoint but semiring {semiring.name!r} has no adjoint"
            )
        times_fn = _resolve_morphism_fn(semiring.adjoint, backend_ops)
    else:
        times_fn = _resolve_morphism_fn(semiring.times, backend_ops)

    n_ops = len(expr.inputs)
    _times = times_fn
    _plus = plus_fn
    _nl = nonlinearity_fn
    _expr = expr

    def kernel(*arrays):
        result = contraction_kernel(_times, _plus, _expr, *arrays)
        return result if _nl is None else _nl(result)

    return register_raw_primitive(
        name, kernel, n_ops, TENSOR_TYPE,
        unwrap=unwrap_tensor, wrap=wrap_tensor,
        result_type=TENSOR_TYPE,
    )
