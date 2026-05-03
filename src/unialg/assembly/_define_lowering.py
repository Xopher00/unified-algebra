"""Compile `define` declarations into backend-callable functions."""
from __future__ import annotations

from collections.abc import Callable

from unialg.terms import _literal_value
from unialg._define_ast import DefineExpr


def compile_expr(
    expr: DefineExpr,
    params: list[str],
    backend,
    define_name: str = "",
) -> Callable:
    kind = expr.kind

    if kind == 'lit':
        v = _literal_value(expr._payload)
        return lambda *_, _v=v: _v

    if kind == 'var':
        name = _literal_value(expr._payload)
        if name not in params:
            raise ValueError(
                f"define '{define_name}': unknown variable '{name}' "
                f"— declared params: {params}")
        idx = params.index(name)
        return lambda *args, _i=idx: args[_i]

    if kind == 'call':
        fields = expr._payload_record_fields()
        fn_name = _literal_value(fields['fn'])
        arg_exprs = [DefineExpr(t) for t in fields['args'].value]
        compiled_args = [compile_expr(a, params, backend, define_name)
                         for a in arg_exprs]
        n = len(compiled_args)
        if n == 1:
            try:
                fn = backend.unary(fn_name)
            except KeyError:
                raise ValueError(
                    f"define '{define_name}': unknown unary function '{fn_name}'")
            ca = compiled_args[0]
            return lambda *args, _fn=fn, _a=ca: _fn(_a(*args))
        else:
            try:
                fn = backend.elementwise(fn_name)
            except KeyError:
                raise ValueError(
                    f"define '{define_name}': unknown binary function '{fn_name}'")
            ca, cb = compiled_args
            return lambda *args, _fn=fn, _a=ca, _b=cb: _fn(_a(*args), _b(*args))

    raise ValueError(f"define '{define_name}': unknown AST node '{kind}'")


def _make_reduce(ew_fn: Callable) -> Callable:
    # Assumes ew_fn is associative; reduction order (left-fold) is unspecified otherwise.
    def _reduce_single(tensor, ax):
        def _select(t, i, a):
            idx = [slice(None)] * t.ndim
            idx[a] = i
            return t[tuple(idx)]
        n = tensor.shape[ax]
        result = _select(tensor, 0, ax)
        for i in range(1, n):
            result = ew_fn(result, _select(tensor, i, ax))
        return result

    def reduce_fn(tensor, axis):
        if isinstance(axis, (tuple, list)):
            result = tensor
            for ax in sorted(axis, reverse=True):
                result = _reduce_single(result, ax)
            return result
        return _reduce_single(tensor, axis)
    return reduce_fn


class _ScopedBackend:
    """A thin overlay over a Backend that adds scoped unary/binary ops.

    Wraps a base Backend and overrides unary_ops and binary_ops with shallow
    copies extended by the new definitions.  All other attribute access
    forwards to the underlying backend so downstream code sees the full API.
    """

    def __init__(self, base, unary_ops: dict, binary_ops: dict):
        # Store under mangled names to avoid conflicts with __getattr__.
        object.__setattr__(self, '_base', base)
        object.__setattr__(self, 'unary_ops', unary_ops)
        object.__setattr__(self, 'binary_ops', binary_ops)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, '_base'), name)

    def __setattr__(self, name: str, value):
        # Allow direct mutation of the scoped dicts; forward everything else.
        if name in ('unary_ops', 'binary_ops'):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_base'), name, value)

    # Override lookup methods so they read from the scoped dicts, not the base.

    def unary(self, op_name: str) -> Callable:
        return object.__getattribute__(self, 'unary_ops')[op_name]

    def elementwise(self, op_name: str) -> Callable:
        return object.__getattribute__(self, 'binary_ops')[op_name].elementwise

    def reduce(self, op_name: str) -> Callable:
        fn = object.__getattribute__(self, 'binary_ops')[op_name].reduce
        if fn is None:
            base = object.__getattribute__(self, '_base')
            raise ValueError(
                f"Backend '{base.name}': binary op '{op_name}' has no "
                "reduction form — cannot be used as semiring +/* in a contraction."
            )
        return fn


def register_defines(defines: list[tuple], backend):
    """Return a new scoped backend with the given define declarations compiled in.

    The original backend is not mutated.  If *defines* is empty the original
    backend is returned unchanged.
    """
    from unialg.backend import Backend
    if not defines:
        return backend

    scoped = _ScopedBackend(
        backend,
        unary_ops=dict(backend.unary_ops),
        binary_ops=dict(backend.binary_ops),
    )

    for arity, name, params, expr_ast in defines:
        if arity == 'unary' and len(params) != 1:
            raise ValueError(
                f"define unary '{name}' must have exactly 1 parameter, "
                f"got {len(params)}: {params}")
        if arity == 'binary' and len(params) != 2:
            raise ValueError(
                f"define binary '{name}' must have exactly 2 parameters, "
                f"got {len(params)}: {params}")

        fn = compile_expr(expr_ast, params, scoped, define_name=name)

        if arity == 'unary':
            scoped.unary_ops[name] = fn
        elif arity == 'binary':
            scoped.binary_ops[name] = Backend.BinaryOp(
                elementwise=fn, reduce=_make_reduce(fn))

    return scoped
