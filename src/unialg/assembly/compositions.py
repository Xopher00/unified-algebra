"""Composition types: build, extract, and compile Hydra lambda terms."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial, reduce

import hydra.core as core
from hydra.analysis import gather_applications
from hydra.dsl.meta.phantoms import var, int32, list_, TTerm
from hydra.dsl.python import Right
from hydra.strip import deannotate_term

from unialg.terms import bind_composition, literal_value

_EQ_PREFIX = "ua.equation."


def _eq_var(name: str) -> TTerm:
    return var(f"{_EQ_PREFIX}{name}")


def _eq_name(term: core.Term) -> str | None:
    if isinstance(term, core.TermVariable) and term.value.value.startswith(_EQ_PREFIX):
        return term.value.value[len(_EQ_PREFIX):]
    return None


def _is_var(term: core.Term, name: str) -> bool:
    return isinstance(term, core.TermVariable) and term.value.value == name


def _decode_literal(lit, coder):
    match coder.encode(None, None, lit):
        case Right(value=arr): return arr
    return None


class Composition:
    PREFIX: str

    @classmethod
    def compile_entry(cls, term, native_fns, coder, backend) -> Callable | None:
        extracted = cls.extract(term)
        if extracted is None:
            return None
        return cls._resolve_and_compile(extracted, native_fns, coder, backend)


class PathComposition(Composition):
    PREFIX = "ua.path."
    LENS_PREFIX = "ua.lens."
    RESIDUAL_PREFIX = "ua.prim.residual_add."

    @staticmethod
    def _extract_chain(body, lp):
        steps = []
        t = body
        while True:
            if isinstance(t, core.TermVariable) and t.value.value == lp:
                return steps
            args, head = gather_applications(t)
            if not args or _eq_name(head) is None:
                return None
            params = list(args[:-1])
            if not all(isinstance(p, core.TermLiteral) for p in params):
                return None
            steps.insert(0, (head.value, params))
            t = args[-1]

    @staticmethod
    def build(name, eq_names, params=None, residual=False, residual_semiring=None):
        if not eq_names:
            raise ValueError(f"Path '{name}' must have at least one equation")
        body: TTerm = var("x")
        for eq_name in eq_names:
            fn: TTerm = _eq_var(eq_name)
            if params and eq_name in params:
                for p in params[eq_name]:
                    fn = fn @ TTerm(p)
            body = fn @ body
        if residual:
            body = var(f"{PathComposition.RESIDUAL_PREFIX}{residual_semiring or 'default'}") @ body @ var("x")
        return bind_composition("path", name, "x", body)

    @staticmethod
    def build_lens(name, fwd_eq_names, bwd_eq_names, params=None, has_residual=False):
        if not fwd_eq_names:
            raise ValueError(f"lens_path '{name}' must have at least one lens")
        if has_residual and len(fwd_eq_names) > 1:
            fwd = bind_composition("path", f"{name}.fwd", "x",
                                   var("ua.prim.lens_fwd") @ list_([_eq_var(n) for n in fwd_eq_names]) @ var("x"))
            bwd = bind_composition("path", f"{name}.bwd", "p",
                                   var("ua.prim.lens_bwd") @ list_([_eq_var(n) for n in bwd_eq_names]) @ var("p"))
            return fwd, bwd
        fwd = PathComposition.build(f"{name}.fwd", fwd_eq_names, params)
        bwd = PathComposition.build(f"{name}.bwd", list(reversed(bwd_eq_names)))
        return fwd, bwd

    @staticmethod
    def extract(term):
        if not isinstance(term, core.TermLambda):
            return None
        lp = term.value.parameter.value
        args, head = gather_applications(term.value.body)
        if (isinstance(head, core.TermVariable)
                and head.value.value.startswith(PathComposition.RESIDUAL_PREFIX)
                and len(args) == 2 and _is_var(args[1], lp)):
            steps = PathComposition._extract_chain(args[0], lp)
            if steps is not None:
                return steps, head.value
        steps = PathComposition._extract_chain(term.value.body, lp)
        return (steps, None) if steps is not None else None

    @classmethod
    def _resolve_and_compile(cls, extracted, native_fns, coder, backend):
        steps, residual_prim = extracted
        fns: list[Callable] = []
        for eq_name, param_literals in steps:
            fn = native_fns.get(eq_name)
            if fn is None:
                return None
            if param_literals:
                decoded = []
                for lit in param_literals:
                    arr = _decode_literal(lit, coder)
                    if arr is None:
                        return None
                    decoded.append(arr)
                fn = partial(fn, *decoded)
            fns.append(fn)
        if residual_prim is not None:
            plus_fn = native_fns.get(residual_prim)
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


class FanComposition(Composition):
    PREFIX = "ua.fan."

    @staticmethod
    def build(name, branch_names, merge_name):
        if not branch_names:
            raise ValueError(f"Fan '{name}' must have at least one branch")
        body = _eq_var(merge_name) @ list_([_eq_var(b) @ var("x") for b in branch_names])
        return bind_composition("fan", name, "x", body)

    @staticmethod
    def build_lens(name, fwd_branches, bwd_branches, merge_fwd, merge_bwd):
        if not fwd_branches:
            raise ValueError(f"lens_fan '{name}' must have at least one branch lens")
        fwd = FanComposition.build(f"{name}.fwd", fwd_branches, merge_fwd)
        bwd = FanComposition.build(f"{name}.bwd", bwd_branches, merge_bwd)
        return fwd, bwd

    @staticmethod
    def extract(term):
        if not isinstance(term, core.TermLambda):
            return None
        lp = term.value.parameter.value
        body = term.value.body
        if not isinstance(body, core.TermApplication):
            return None
        merge_name = _eq_name(body.value.function)
        if merge_name is None or not isinstance(body.value.argument, core.TermList):
            return None
        branches = []
        for b in body.value.argument.value:
            if not isinstance(b, core.TermApplication):
                return None
            bname = _eq_name(b.value.function)
            if bname is None or not _is_var(b.value.argument, lp):
                return None
            branches.append(bname)
        return merge_name, branches

    @classmethod
    def _resolve_and_compile(cls, extracted, native_fns, coder, backend):
        _eq = lambda name: native_fns.get(core.Name(f"{_EQ_PREFIX}{name}"))
        merge_fn = _eq(extracted[0])
        branch_fns = [_eq(b) for b in extracted[1]]
        if merge_fn is None or not all(branch_fns):
            return None
        return backend.compile(lambda x: merge_fn([fn(x) for fn in branch_fns]))


class PrimComposition(Composition):
    """λvar. prim @ eq_ref(s) @ literal(s) @ var — shared build/extract."""
    PRIM_PREFIX: str
    KIND: str
    VAR_NAME: str

    @classmethod
    def build(cls, name, step_name, *extra):
        body = cls._prim_term(*extra) @ _eq_var(step_name) @ cls._wrap_extra(*extra) @ var(cls.VAR_NAME)
        return bind_composition(cls.KIND, name, cls.VAR_NAME, body)

    @classmethod
    def _prim_term(cls, *extra):
        return var(cls.PRIM_PREFIX)

    @classmethod
    def _wrap_extra(cls, *extra):
        raise NotImplementedError

    @classmethod
    def extract(cls, term):
        term = deannotate_term(term)
        if not isinstance(term, core.TermLambda):
            return None
        args, head = gather_applications(term.value.body)
        if not (isinstance(head, core.TermVariable) and head.value.value.startswith(cls.PRIM_PREFIX)):
            return None
        lp = term.value.parameter.value
        if len(args) != 3 or not _is_var(args[-1], lp):
            return None
        return cls._parse_args(head.value.value, args[:-1])

    @classmethod
    def _parse_args(cls, prim_name, args):
        step = _eq_name(args[0])
        if step is None or not isinstance(args[1], core.TermLiteral):
            return None
        return step, args[1]

    @classmethod
    def _resolve_and_compile(cls, extracted, native_fns, coder, backend):
        step_fn = native_fns.get(core.Name(f"{_EQ_PREFIX}{extracted[0]}"))
        if step_fn is None:
            return None
        return cls._compile(step_fn, extracted, native_fns, coder, backend)

    @classmethod
    def _compile(cls, step_fn, extracted, native_fns, coder, backend):
        raise NotImplementedError


class FoldComposition(PrimComposition):
    PREFIX = "ua.fold."
    PRIM_PREFIX = "hydra.lib.lists.foldl"
    KIND = "fold"
    VAR_NAME = "seq"

    @classmethod
    def _wrap_extra(cls, init_term): return TTerm(init_term)

    @classmethod
    def _compile(cls, step_fn, extracted, native_fns, coder, backend):
        init = _decode_literal(extracted[1], coder)
        if init is None:
            return None
        return backend.compile(lambda seq: reduce(step_fn, seq, init))


class UnfoldComposition(PrimComposition):
    PREFIX = "ua.unfold."
    PRIM_PREFIX = "ua.prim.unfold_n"
    KIND = "unfold"
    VAR_NAME = "state"

    @classmethod
    def _wrap_extra(cls, n_steps): return int32(n_steps)

    @classmethod
    def _compile(cls, step_fn, extracted, native_fns, coder, backend):
        n = int(literal_value(extracted[1]))
        def compiled(state):
            outs = []
            for _ in range(n):
                state = step_fn(state); outs.append(state)
            return tuple(outs)
        return backend.compile(compiled)


class FixpointComposition(PrimComposition):
    PREFIX = "ua.fixpoint."
    PRIM_PREFIX = "ua.prim.fixpoint."
    KIND = "fixpoint"
    VAR_NAME = "state"

    @classmethod
    def _prim_term(cls, predicate_name, epsilon, max_iter):
        return var(f"{cls.PRIM_PREFIX}{epsilon}.{max_iter}")

    @classmethod
    def _wrap_extra(cls, predicate_name, epsilon, max_iter):
        return _eq_var(predicate_name)

    @classmethod
    def _parse_args(cls, prim_name, args):
        step, pred = _eq_name(args[0]), _eq_name(args[1])
        if step is None or pred is None:
            return None
        tail = prim_name[len(cls.PRIM_PREFIX):]
        dot = tail.rfind(".")
        try:
            return step, pred, float(tail[:dot]), int(tail[dot + 1:])
        except (ValueError, IndexError):
            return None

    @classmethod
    def _compile(cls, step_fn, extracted, native_fns, coder, backend):
        pred_fn = native_fns.get(core.Name(f"{_EQ_PREFIX}{extracted[1]}"))
        if not pred_fn:
            return None
        epsilon, max_iter = extracted[2], extracted[3]
        def cond_fn(c): return (pred_fn(c[0]) > epsilon) & (c[1] < max_iter)
        def body_fn(c): return step_fn(c[0]), c[1] + 1
        return backend.compile(lambda init: backend.while_loop(cond_fn, body_fn, (init, 0)))
