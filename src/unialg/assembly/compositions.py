"""Composition types: record-based declarations that generate Hydra lambda terms."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial, reduce

import hydra.core as core
from hydra.dsl.meta.phantoms import var, int32, list_, TTerm
from hydra.dsl.python import Right

from unialg.terms import _RecordView

_EQ_PREFIX = "ua.equation."
_MERGE_SUFFIX = ".__merge__"


def _eq_var(name: str) -> TTerm:
    return var(f"{_EQ_PREFIX}{name}")


def _lookup(native_fns, name):
    return native_fns.get(core.Name(f"{_EQ_PREFIX}{name}"))


def _merge_eq_var(name: str) -> TTerm:
    return var(f"{_EQ_PREFIX}{name}{_MERGE_SUFFIX}")


def _merge_lookup(native_fns, name):
    return native_fns.get(core.Name(f"{_EQ_PREFIX}{name}{_MERGE_SUFFIX}"))


def _decode_init(coder, init_term):
    try:
        match coder.encode(None, None, init_term):
            case Right(value=v): return v, False
            case _: return None, False
    except Exception:
        match init_term.value:
            case core.LiteralFloat(value=v): return v, True
            case core.LiteralInteger(value=v): return int(v), True
            case _: return None, False


def _bind(kind, name, var_name, body):
    from hydra.dsl.meta.phantoms import lam
    if not isinstance(body, TTerm):
        body = TTerm(body)
    term = lam(var_name, body).value
    return (core.Name(f"ua.{kind}.{name}"), term)


class Composition(_RecordView):

    name = _RecordView.Scalar(str)
    _kind: str
    _var_name: str

    def _body(self) -> TTerm:
        raise NotImplementedError

    def to_lambda(self):
        return _bind(self._kind, self.name, self._var_name, self._body())

    def resolve_and_compile(self, native_fns, coder, backend) -> Callable | None:
        raise NotImplementedError


class StepComposition(Composition):

    step_name = _RecordView.Scalar(str, key="stepName")

    def _prim_var(self): raise NotImplementedError
    def _extra_term(self): raise NotImplementedError

    def _body(self):
        return self._prim_var() @ _eq_var(self.step_name) @ self._extra_term() @ var(self._var_name)

    def resolve_and_compile(self, native_fns, coder, backend) -> Callable | None:
        step_fn = _lookup(native_fns, self.step_name)
        if step_fn is None:
            return None
        return self._compile(step_fn, native_fns, coder, backend)

    def _compile(self, step_fn, native_fns, coder, backend) -> Callable | None:
        raise NotImplementedError


class PathComposition(Composition):
    _type_name = core.Name("ua.composition.Path")
    _kind = "path"
    _var_name = "x"
    _params = None

    eq_names          = _RecordView.ScalarList(key="eqNames")
    residual          = _RecordView.Scalar(bool, default=False)
    residual_semiring = _RecordView.Scalar(str, default="", key="residualSemiring")

    def __init__(self, name, eq_names, params=None, residual=False, residual_semiring=None):
        if not eq_names:
            raise ValueError(f"Path '{name}' must have at least one equation")
        self._params = params
        super().__init__(name=name, eq_names=eq_names,
                         residual=residual, residual_semiring=residual_semiring or "")

    def _body(self):
        body: TTerm = var("x")
        for eq_name in self.eq_names:
            fn: TTerm = _eq_var(eq_name)
            if self._params and eq_name in self._params:
                for p in self._params[eq_name]:
                    fn = fn @ TTerm(p)
            body = fn @ body
        if self.residual:
            sr = self.residual_semiring or "default"
            body = var(f"ua.prim.residual_add.{sr}") @ body @ var("x")
        return body

    def resolve_and_compile(self, native_fns, coder, backend):
        fns = []
        for eq_name in self.eq_names:
            fn = _lookup(native_fns, eq_name)
            if fn is None:
                return None
            if self._params and eq_name in self._params:
                decoded = []
                for lit in self._params[eq_name]:
                    if not isinstance(lit, core.TermLiteral):
                        return None
                    match coder.encode(None, None, lit):
                        case Right(value=arr): decoded.append(arr)
                        case _: return None
                fn = partial(fn, *decoded)
            fns.append(fn)
        plus_fn = None
        if self.residual:
            sr = self.residual_semiring or "default"
            plus_fn = native_fns.get(core.Name(f"ua.prim.residual_add.{sr}"))
            if plus_fn is None:
                return None
        def compiled(x):
            out = x
            for f in fns:
                out = f(out)
            return plus_fn(out, x) if plus_fn else out
        return backend.compile(compiled)

    @staticmethod
    def build_lens(name, fwd_eq_names, bwd_eq_names, params=None, has_residual=False):
        if not fwd_eq_names:
            raise ValueError(f"lens_path '{name}' must have at least one lens")
        if has_residual and len(fwd_eq_names) > 1:
            fwd = _bind("path", f"{name}.fwd", "x",
                         var("ua.prim.lens_fwd") @ list_([_eq_var(n) for n in fwd_eq_names]) @ var("x"))
            bwd = _bind("path", f"{name}.bwd", "p",
                         var("ua.prim.lens_bwd") @ list_([_eq_var(n) for n in bwd_eq_names]) @ var("p"))
            return fwd, bwd
        fwd = PathComposition(f"{name}.fwd", fwd_eq_names, params).to_lambda()
        bwd = PathComposition(f"{name}.bwd", list(reversed(bwd_eq_names))).to_lambda()
        return fwd, bwd


class FanComposition(Composition):
    _type_name = core.Name("ua.composition.Fan")
    _kind = "fan"
    _var_name = "x"

    merge_names = _RecordView.ScalarList(key="mergeNames")
    branches    = _RecordView.ScalarList()

    def __init__(self, name, branches, merge_names):
        if not branches:
            raise ValueError(f"Fan '{name}' must have at least one branch")
        if isinstance(merge_names, str):
            merge_names = [merge_names]
        super().__init__(name=name, branches=branches, merge_names=merge_names)

    def _body(self):
        body = _merge_eq_var(self.merge_names[0]) @ list_([_eq_var(b) @ var("x") for b in self.branches])
        if len(self.merge_names) > 1:
            for mn in self.merge_names[1:]:
                body = _merge_eq_var(mn) @ list_([body])
        return body

    def resolve_and_compile(self, native_fns, coder, backend):
        branch_fns = [_lookup(native_fns, b) for b in self.branches]
        if not all(branch_fns):
            return None
        names = self.merge_names
        if len(names) == 1:
            merge_fn = _merge_lookup(native_fns, names[0])
            if merge_fn is None:
                return None
            return backend.compile(lambda x: merge_fn([fn(x) for fn in branch_fns]))
        steps = []
        for mn in names:
            fn = _merge_lookup(native_fns, mn)
            if fn is not None:
                steps.append((fn, getattr(fn, 'n_inputs', 2)))
            else:
                nl = backend.unary(mn)
                steps.append((lambda t, _nl=nl: _nl(t), 1))
        return backend.compile(lambda x: _stack_execute(branch_fns, steps, x))

    @staticmethod
    def build_lens(name, fwd_branches, bwd_branches, merge_fwd, merge_bwd):
        if not fwd_branches:
            raise ValueError(f"lens_fan '{name}' must have at least one branch lens")
        fwd = FanComposition(f"{name}.fwd", fwd_branches, [merge_fwd]).to_lambda()
        bwd = FanComposition(f"{name}.bwd", bwd_branches, [merge_bwd]).to_lambda()
        return fwd, bwd


def _stack_execute(branch_fns, steps, x):
    stack = [fn(x) for fn in branch_fns]
    for step_fn, n_inputs in steps:
        consumed = stack[:n_inputs]
        remaining = stack[n_inputs:]
        result = step_fn(consumed) if n_inputs > 1 else step_fn(consumed[0])
        stack = [result] + remaining
    if len(stack) != 1:
        raise ValueError(f"Merge chain incomplete: {len(stack)} elements remain on stack")
    return stack[0]


class FoldComposition(StepComposition):
    _type_name = core.Name("ua.composition.Fold")
    _kind = "fold"
    _var_name = "seq"

    init_term = _RecordView.Term(key="initTerm")

    def _prim_var(self): return var("hydra.lib.lists.foldl")
    def _extra_term(self): return TTerm(self.init_term)

    def _compile(self, step_fn, native_fns, coder, backend):
        init, is_scalar = _decode_init(coder, self.init_term)
        if init is None:
            return None
        if is_scalar:
            def _compiled_scalar(seq, _i=init, _s=step_fn):
                seq_list = list(seq)
                if not seq_list:
                    raise ValueError("fold: cannot fold empty sequence with scalar init")
                return reduce(_s, seq_list, seq_list[0] * 0 + _i)
            return backend.compile(_compiled_scalar)
        return backend.compile(lambda seq: reduce(step_fn, seq, init))


class UnfoldComposition(StepComposition):
    _type_name = core.Name("ua.composition.Unfold")
    _kind = "unfold"
    _var_name = "state"

    n = _RecordView.Scalar(int, key="nSteps")

    def _prim_var(self): return var("ua.prim.unfold_n")
    def _extra_term(self): return int32(self.n)

    def _compile(self, step_fn, native_fns, coder, backend):
        n = self.n
        def compiled(state):
            outs = []
            for _ in range(n):
                state = step_fn(state); outs.append(state)
            return tuple(outs)
        return backend.compile(compiled)


class FixpointComposition(StepComposition):
    _type_name = core.Name("ua.composition.Fixpoint")
    _kind = "fixpoint"
    _var_name = "state"

    pred_name = _RecordView.Scalar(str, key="predName")
    epsilon   = _RecordView.Scalar(float)
    max_iter  = _RecordView.Scalar(int, key="maxIter")

    def _prim_var(self): return var(f"ua.prim.fixpoint.{self.epsilon}.{self.max_iter}")
    def _extra_term(self): return _eq_var(self.pred_name)

    def _compile(self, step_fn, native_fns, coder, backend):
        pred_fn = _lookup(native_fns, self.pred_name)
        if pred_fn is None:
            return None
        epsilon, max_iter = self.epsilon, self.max_iter
        def cond_fn(c): return (pred_fn(c[0]) > epsilon) & (c[1] < max_iter)
        def body_fn(c): return step_fn(c[0]), c[1] + 1
        return backend.compile(lambda init: backend.while_loop(cond_fn, body_fn, (init, 0)))
