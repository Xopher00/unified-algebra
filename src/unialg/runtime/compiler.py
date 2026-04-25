"""Compile Hydra bound terms to native Python callables.

Each entry-point bound term is walked once at compile_program time, extracting
equation references and inline literals, and composing them into a plain
Python function. The compiled function takes/returns native arrays directly —
no Hydra encode/decode, no wire format between equations. This lets
backend.compile (torch.compile / jax.jit) fuse across boundaries.

Compilable shapes:
  ua.path.* / ua.lens.*   sequential composition (with optional residual_add)
  ua.fan.*                merge @ list_(branch @ var(x), ...)
  ua.fold.*               foldl @ step @ init @ var(seq)
  ua.unfold.*             unfold_n @ step @ n @ var(state)
  ua.fixpoint.*           backend.while_loop on (state, iter_count)
  ua.equation.*           single equation, native_fn used directly

Falls back to reduce_term for ua.param.* runtime values, non-literal params,
or any unrecognised structure.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial, reduce

import hydra.core as core
from hydra.analysis import gather_applications
from hydra.dsl.python import Right
from hydra.strip import deannotate_term

from unialg.terms import literal_value


# ---------------------------------------------------------------------------
# Generic structural helpers
# ---------------------------------------------------------------------------

def _eq_name(term: core.Term) -> str | None:
    """Short name for a ua.equation.X variable, or None."""
    if isinstance(term, core.TermVariable) and term.value.value.startswith("ua.equation."):
        return term.value.value[len("ua.equation."):]
    return None


def _is_var(term: core.Term, name: str) -> bool:
    return isinstance(term, core.TermVariable) and term.value.value == name


def _peel_lambda_app(term: core.Term, prim_prefix: str):
    """Walk λp. <prim @ a1 @ ... @ an>; return (prim_name, args, lp) or None.

    Uses hydra.analysis.gather_applications, which strips annotations and
    returns args in forward application order.
    """
    term = deannotate_term(term)
    if not isinstance(term, core.TermLambda):
        return None
    args, head = gather_applications(term.value.body)
    if not (isinstance(head, core.TermVariable) and head.value.value.startswith(prim_prefix)):
        return None
    return head.value.value, list(args), term.value.parameter.value


def _decode_tensor_literal(lit, coder):
    """Decode a TermLiteral via coder.encode (Hydra → native array)."""
    match coder.encode(None, None, lit):
        case Right(value=arr):
            return arr
    return None


# ---------------------------------------------------------------------------
# Path / lens-path extraction
# ---------------------------------------------------------------------------

def _peel_eq_step(term: core.Term) -> tuple[core.Name, list[core.Term]] | None:
    """Match (eq_name @ p1 @ ... @ pk) where each pk is a TermLiteral.

    Returns (eq_name, [p1, ..., pk]) in application order, or None.
    """
    params: list[core.Term] = []
    t = term
    while True:
        match t:
            case core.TermVariable(value=name) if name.value.startswith("ua.equation."):
                return name, list(reversed(params))
            case core.TermApplication(value=app):
                match app.argument:
                    case core.TermLiteral():
                        params.append(app.argument)
                        t = app.function
                    case _:
                        return None
            case _:
                return None


def _extract_chain_from_body(body, lambda_param):
    steps = []
    t = body
    while True:
        match t:
            case core.TermVariable(value=name) if name.value == lambda_param:
                return steps
            case core.TermApplication(value=app):
                step = _peel_eq_step(app.function)
                if step is None:
                    return None
                steps.insert(0, step)
                t = app.argument
            case _:
                return None


def _extract_steps(term):
    """Extract chain steps + optional residual prim from a path lambda."""
    match term:
        case core.TermLambda(value=lam):
            lp = lam.parameter.value
            match lam.body:
                case core.TermApplication(value=outer):
                    if (isinstance(outer.argument, core.TermVariable)
                            and outer.argument.value.value == lp):
                        match outer.function:
                            case core.TermApplication(value=inner):
                                match inner.function:
                                    case core.TermVariable(value=pn) if pn.value.startswith("ua.prim.residual_add."):
                                        steps = _extract_chain_from_body(inner.argument, lp)
                                        if steps is not None:
                                            return steps, inner.function.value
            steps = _extract_chain_from_body(lam.body, lp)
            return (steps, None) if steps is not None else None
    return None


# ---------------------------------------------------------------------------
# Fixpoint / fold / unfold / fan extraction
# ---------------------------------------------------------------------------

def _extract_fixpoint(term):
    """λstate. ua.prim.fixpoint.<eps>.<max> @ step @ pred @ var(state)."""
    res = _peel_lambda_app(term, "ua.prim.fixpoint.")
    if res is None:
        return None
    prim_name, args, lp = res
    if (len(args) != 3
            or (step_name := _eq_name(args[0])) is None
            or (pred_name := _eq_name(args[1])) is None
            or not _is_var(args[2], lp)):
        return None
    tail = prim_name[len("ua.prim.fixpoint."):]
    dot = tail.rfind(".")
    try:
        return step_name, pred_name, float(tail[:dot]), int(tail[dot + 1:])
    except (ValueError, IndexError):
        return None


def _extract_unary_iter(term, prim_prefix):
    """λp. <prim @ ua.equation.X @ literal @ var(p)>. Used by fold and unfold."""
    res = _peel_lambda_app(term, prim_prefix)
    if res is None:
        return None
    _, args, lp = res
    if (len(args) != 3
            or (step_name := _eq_name(args[0])) is None
            or not isinstance(args[1], core.TermLiteral)
            or not _is_var(args[2], lp)):
        return None
    return step_name, args[1]


def _extract_fan(term):
    """λx. <ua.equation.merge @ list_([ua.equation.b @ var(x), ...])>."""
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


# ---------------------------------------------------------------------------
# Per-structure compilers
# ---------------------------------------------------------------------------

def _compile_sequential(steps, native_fns, coder, backend, residual_prim=None):
    """Build a closure chain from [(eq_name, [param_literals])]."""
    fns: list[Callable] = []
    for eq_name, param_literals in steps:
        fn = native_fns.get(eq_name)
        if fn is None:
            return None
        if param_literals:
            decoded = []
            for lit in param_literals:
                arr = _decode_tensor_literal(lit, coder)
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
            for fn in fns:
                out = fn(out)
            return plus_fn(out, x)
    else:
        def compiled(x):
            for fn in fns:
                x = fn(x)
            return x

    return backend.compile(compiled)


def _compile_fixpoint(step_fn, pred_fn, epsilon, max_iter, backend) -> Callable:
    """Compile a fixpoint to backend.while_loop. Carry = (state, iter_count)."""
    def cond_fn(c): return (pred_fn(c[0]) > epsilon) & (c[1] < max_iter)
    def body_fn(c): return step_fn(c[0]), c[1] + 1
    return backend.compile(lambda init: backend.while_loop(cond_fn, body_fn, (init, 0)))


def _compile_fold(step_fn, init, backend) -> Callable:
    """foldl(step, init, seq). Step is binary (acc, x) -> acc."""
    return backend.compile(lambda seq: reduce(step_fn, seq, init))


def _compile_unfold(step_fn, n, backend) -> Callable:
    """Iterate step n times from initial state, return outputs as tuple."""
    def compiled(state):
        outs = []
        for _ in range(n):
            state = step_fn(state); outs.append(state)
        return tuple(outs)
    return backend.compile(compiled)


def _compile_fan(merge_fn, branch_fns, backend) -> Callable:
    """Branches in parallel, then merge over the list."""
    return backend.compile(lambda x: merge_fn([fn(x) for fn in branch_fns]))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compile_graph(graph, native_fns: dict[core.Name, Callable], coder, backend) -> dict[str, Callable]:
    """Walk every entry point once and return short_name -> native Callable."""
    compiled: dict[str, Callable] = {}

    def _eq(short):
        return native_fns.get(core.Name(f"ua.equation.{short}"))

    def _put(prefix, full, fn):
        if fn is not None:
            compiled[full.split(".", 2)[-1] if prefix in ("ua.path.", "ua.lens.") else full[len(prefix):]] = fn

    for full_name, term in graph.bound_terms.items():
        n = full_name.value

        if n.startswith("ua.path.") or n.startswith("ua.lens."):
            res = _extract_steps(term)
            if res is None: continue
            steps, residual_prim = res
            _put("ua.path." if n.startswith("ua.path.") else "ua.lens.", n,
                 _compile_sequential(steps, native_fns, coder, backend, residual_prim))

        elif n.startswith("ua.fixpoint.") and (fp := _extract_fixpoint(term)):
            step_fn, pred_fn = _eq(fp[0]), _eq(fp[1])
            if step_fn and pred_fn:
                _put("ua.fixpoint.", n, _compile_fixpoint(step_fn, pred_fn, fp[2], fp[3], backend))

        elif n.startswith("ua.fold.") and (res := _extract_unary_iter(term, "hydra.lib.lists.foldl")):
            step_fn = _eq(res[0])
            init = _decode_tensor_literal(res[1], coder)
            if step_fn is not None and init is not None:
                _put("ua.fold.", n, _compile_fold(step_fn, init, backend))

        elif n.startswith("ua.unfold.") and (res := _extract_unary_iter(term, "ua.prim.unfold_n")):
            step_fn = _eq(res[0])
            if step_fn is not None:
                _put("ua.unfold.", n, _compile_unfold(step_fn, int(literal_value(res[1])), backend))

        elif n.startswith("ua.fan.") and (res := _extract_fan(term)):
            merge_fn = _eq(res[0])
            branch_fns = [_eq(b) for b in res[1]]
            if merge_fn is not None and all(f is not None for f in branch_fns):
                _put("ua.fan.", n, _compile_fan(merge_fn, branch_fns, backend))

    for full_name in graph.primitives:
        n = full_name.value
        if n.startswith("ua.equation."):
            fn = native_fns.get(full_name)
            if fn is not None:
                compiled[n[len("ua.equation."):]] = fn

    return compiled
