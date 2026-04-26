"""Composition types: record-based declarations that generate Hydra lambda terms."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial, reduce

import hydra.core as core
from hydra.dsl.meta.phantoms import var, int32, list_, TTerm
from hydra.dsl.python import Right

from unialg.terms import _RecordView

_EQ_PREFIX = "ua.equation."


def _eq_var(name: str) -> TTerm:
    return var(f"{_EQ_PREFIX}{name}")


def _bind(kind, name, var_name, body):
    from hydra.dsl.meta.phantoms import lam
    if not isinstance(body, TTerm):
        body = TTerm(body)
    term = lam(var_name, body).value
    return (core.Name(f"ua.{kind}.{name}"), term)


class Composition(_RecordView):

    name = _RecordView.Scalar(str)

    def to_lambda(self):
        raise NotImplementedError

    def resolve_and_compile(self, native_fns, coder, backend) -> Callable | None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

class PathComposition(Composition):
    _type_name = core.Name("ua.composition.Path")

    name              = _RecordView.Scalar(str)
    eq_names          = _RecordView.ScalarList(key="eqNames")
    residual          = _RecordView.Scalar(bool, default=False)
    residual_semiring = _RecordView.Scalar(str, default="", key="residualSemiring")

    def __init__(self, name, eq_names, params=None, residual=False, residual_semiring=None):
        if not eq_names:
            raise ValueError(f"Path '{name}' must have at least one equation")
        self._params = params
        super().__init__(name=name, eq_names=eq_names,
                         residual=residual, residual_semiring=residual_semiring or "")

    def to_lambda(self):
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
        return _bind("path", self.name, "x", body)

    def resolve_and_compile(self, native_fns, coder, backend):
        fns: list[Callable] = []
        for eq_name in self.eq_names:
            fn = native_fns.get(core.Name(f"{_EQ_PREFIX}{eq_name}"))
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
        if self.residual:
            sr = self.residual_semiring or "default"
            plus_fn = native_fns.get(core.Name(f"ua.prim.residual_add.{sr}"))
            if plus_fn is None:
                return None
            def compiled(x):
                out = x
                for f in fns:
                    out = f(out)
                return plus_fn(out, x)
        else:
            def compiled(x):
                for f in fns:
                    x = f(x)
                return x
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


# ---------------------------------------------------------------------------
# Fan
# ---------------------------------------------------------------------------

class FanComposition(Composition):
    _type_name = core.Name("ua.composition.Fan")

    name       = _RecordView.Scalar(str)
    merge_name = _RecordView.Scalar(str, key="mergeName")
    branches   = _RecordView.ScalarList()

    def __init__(self, name, branches, merge_name):
        if not branches:
            raise ValueError(f"Fan '{name}' must have at least one branch")
        super().__init__(name=name, branches=branches, merge_name=merge_name)

    def to_lambda(self):
        body = _eq_var(self.merge_name) @ list_([_eq_var(b) @ var("x") for b in self.branches])
        return _bind("fan", self.name, "x", body)

    def resolve_and_compile(self, native_fns, coder, backend):
        _eq = lambda name: native_fns.get(core.Name(f"{_EQ_PREFIX}{name}"))
        merge_fn = _eq(self.merge_name)
        branch_fns = [_eq(b) for b in self.branches]
        if merge_fn is None or not all(branch_fns):
            return None
        return backend.compile(lambda x: merge_fn([fn(x) for fn in branch_fns]))

    @staticmethod
    def build_lens(name, fwd_branches, bwd_branches, merge_fwd, merge_bwd):
        if not fwd_branches:
            raise ValueError(f"lens_fan '{name}' must have at least one branch lens")
        fwd = FanComposition(f"{name}.fwd", fwd_branches, merge_fwd).to_lambda()
        bwd = FanComposition(f"{name}.bwd", bwd_branches, merge_bwd).to_lambda()
        return fwd, bwd


# ---------------------------------------------------------------------------
# Fold
# ---------------------------------------------------------------------------

class FoldComposition(Composition):
    _type_name = core.Name("ua.composition.Fold")

    name      = _RecordView.Scalar(str)
    step_name = _RecordView.Scalar(str, key="stepName")
    init_term = _RecordView.Term(key="initTerm")

    def to_lambda(self):
        body = var("hydra.lib.lists.foldl") @ _eq_var(self.step_name) @ TTerm(self.init_term) @ var("seq")
        return _bind("fold", self.name, "seq", body)

    def resolve_and_compile(self, native_fns, coder, backend):
        step_fn = native_fns.get(core.Name(f"{_EQ_PREFIX}{self.step_name}"))
        if step_fn is None:
            return None
        match coder.encode(None, None, self.init_term):
            case Right(value=init): pass
            case _: return None
        return backend.compile(lambda seq: reduce(step_fn, seq, init))


# ---------------------------------------------------------------------------
# Unfold
# ---------------------------------------------------------------------------

class UnfoldComposition(Composition):
    _type_name = core.Name("ua.composition.Unfold")

    name      = _RecordView.Scalar(str)
    step_name = _RecordView.Scalar(str, key="stepName")
    n         = _RecordView.Scalar(int, key="nSteps")

    def to_lambda(self):
        body = var("ua.prim.unfold_n") @ _eq_var(self.step_name) @ int32(self.n) @ var("state")
        return _bind("unfold", self.name, "state", body)

    def resolve_and_compile(self, native_fns, coder, backend):
        step_fn = native_fns.get(core.Name(f"{_EQ_PREFIX}{self.step_name}"))
        if step_fn is None:
            return None
        n = self.n
        def compiled(state):
            outs = []
            for _ in range(n):
                state = step_fn(state); outs.append(state)
            return tuple(outs)
        return backend.compile(compiled)


# ---------------------------------------------------------------------------
# Fixpoint
# ---------------------------------------------------------------------------

class FixpointComposition(Composition):
    _type_name = core.Name("ua.composition.Fixpoint")

    name      = _RecordView.Scalar(str)
    step_name = _RecordView.Scalar(str, key="stepName")
    pred_name = _RecordView.Scalar(str, key="predName")
    epsilon   = _RecordView.Scalar(float)
    max_iter  = _RecordView.Scalar(int, key="maxIter")

    def to_lambda(self):
        prim = var(f"ua.prim.fixpoint.{self.epsilon}.{self.max_iter}")
        body = prim @ _eq_var(self.step_name) @ _eq_var(self.pred_name) @ var("state")
        return _bind("fixpoint", self.name, "state", body)

    def resolve_and_compile(self, native_fns, coder, backend):
        step_fn = native_fns.get(core.Name(f"{_EQ_PREFIX}{self.step_name}"))
        pred_fn = native_fns.get(core.Name(f"{_EQ_PREFIX}{self.pred_name}"))
        if step_fn is None or pred_fn is None:
            return None
        epsilon, max_iter = self.epsilon, self.max_iter
        def cond_fn(c): return (pred_fn(c[0]) > epsilon) & (c[1] < max_iter)
        def body_fn(c): return step_fn(c[0]), c[1] + 1
        return backend.compile(lambda init: backend.while_loop(cond_fn, body_fn, (init, 0)))
