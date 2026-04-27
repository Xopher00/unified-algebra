"""Expression compiler for `define` declarations.

Compiles expression ASTs (from the parser) into backend-callable functions
and registers them on the backend's op tables.
"""
from __future__ import annotations

from typing import Callable


def compile_expr(
    ast: tuple,
    params: list[str],
    backend,
    define_name: str = "",
) -> Callable:
    kind = ast[0]

    if kind == 'lit':
        v = ast[1]
        return lambda *_args, _v=v: _v

    if kind == 'var':
        name = ast[1]
        if name not in params:
            raise ValueError(
                f"define '{define_name}': unknown variable '{name}' "
                f"— declared params: {params}")
        idx = params.index(name)
        return lambda *args, _i=idx: args[_i]

    if kind == 'call':
        fn_name, arg_asts = ast[1], ast[2]
        compiled_args = [compile_expr(a, params, backend, define_name)
                         for a in arg_asts]
        n = len(compiled_args)
        if n == 1:
            try:
                fn = backend.unary(fn_name)
            except KeyError:
                raise ValueError(
                    f"define '{define_name}': unknown unary function '{fn_name}'")
            ca = compiled_args[0]
            return lambda *args, _fn=fn, _a=ca: _fn(_a(*args))
        elif n == 2:
            try:
                fn = backend.elementwise(fn_name)
            except KeyError:
                raise ValueError(
                    f"define '{define_name}': unknown binary function '{fn_name}'")
            ca, cb = compiled_args
            return lambda *args, _fn=fn, _a=ca, _b=cb: _fn(_a(*args), _b(*args))
        else:
            raise ValueError(
                f"define '{define_name}': function '{fn_name}' called with "
                f"{n} args — only 1 (unary) or 2 (binary) supported")

    raise ValueError(f"define '{define_name}': unknown AST node '{kind}'")


def _make_reduce(ew_fn: Callable) -> Callable:
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


def register_defines(defines: list[tuple], backend) -> None:
    from unialg.backend import Backend
    for arity, name, params, expr_ast in defines:
        if arity == 'unary' and len(params) != 1:
            raise ValueError(
                f"define unary '{name}' must have exactly 1 parameter, "
                f"got {len(params)}: {params}")
        if arity == 'binary' and len(params) != 2:
            raise ValueError(
                f"define binary '{name}' must have exactly 2 parameters, "
                f"got {len(params)}: {params}")

        fn = compile_expr(expr_ast, params, backend, define_name=name)

        if arity == 'unary':
            backend.unary_ops[name] = fn
        elif arity == 'binary':
            backend.binary_ops[name] = Backend.BinaryOp(
                elementwise=fn, reduce=_make_reduce(fn))
