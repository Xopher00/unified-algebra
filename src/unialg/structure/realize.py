"""Hydra realization for validated DSL syntax.

This module is the backend interpretation layer.  It assumes morphism
composition and visible types have already been checked by ``morphisms.py`` and
translates expression trees to raw Hydra terms.  Derived polynomial actions
live in ``actions.py``; optic recursion helpers live in ``recursion.py``.
"""

from __future__ import annotations

import hydra.dsl.meta.phantoms as P
from hydra.core import Term
from hydra.phantoms import TTerm

from unialg.syntax import expressions as expr
from unialg.objects import Type, TypeUnit, TypePair, TypeEither
from . import terms as H


def realize_term(node: expr.MorphismExpr) -> TTerm:
    """Translate a morphism expression into a typed Hydra term handle."""
    return TTerm(realize(node))


def split_input(x: TTerm, param: Type) -> tuple[TTerm | None, TTerm]:
    """Split a raw argument into optional parameter and visible value."""
    if param == TypeUnit():
        return None, x
    return P.first(x), P.second(x)


def _child_param(param_term: TTerm | None, param: Type, child_param: Type, side: str) -> TTerm | None:
    """Select the parameter fragment required by a contextual child."""
    if child_param == TypeUnit():
        return None
    if param_term is None or child_param == param:
        return param_term
    return P.first(param_term) if side == "left" else P.second(param_term)


def _contextual_term(node: expr.ContextualBinary, build) -> Term:
    """Realize a contextual node by routing child calls and wrapping ``ctx_x``."""
    def wrapped(x: TTerm):
        param_term, value = split_input(x, node.param)
        def call(child, child_param, side, v):
            child_param_term = _child_param(param_term, node.param, child_param, side)
            arg = (v if child_param_term is None else P.pair(child_param_term, v))
            return P.apply(realize_term(child), arg)
        return build(value,
            lambda v: call(node.f, node.f_param, "right", v),
            lambda v: call(node.g, node.g_param, "left", v),
        )
    return H.term_lambda("ctx_x", wrapped).value


def realize(node: expr.MorphismExpr) -> Term:
    """Translate a morphism expression into a raw Hydra term.

    The returned term expects the node's raw domain, not necessarily the visible
    ``Morphism.dom()``.  For parametric/lax contextual nodes, the raw domain and
    codomain are already stored on the expression by ``morphisms.py``.
    """
    match node:
        case expr.Identity():
            return P.identity().value
        case expr.Copy():
            return H.term_lambda("x", lambda x: P.pair(x, x)).value
        case expr.Delete():
            return P.constant(P.unit()).value
        case expr.First():
            return H.pair_first().value
        case expr.Second():
            return H.pair_second().value
        case expr.Left():
            return H.left_injection().value
        case expr.Right():
            return H.right_injection().value
        case expr.Absurd():
            return H.absurd().value
        case expr.Assoc():
            return H.term_lambda(
                "p", 
                lambda p: P.pair(P.first(P.first(p)), P.pair(P.second(P.first(p)), P.second(p))),
            ).value
        case expr.Symmetry(dom=dom, cod=cod):
            if isinstance(dom, TypePair) and isinstance(cod, TypePair):
                return H.pair_swap().value
            if isinstance(dom, TypeEither) and isinstance(cod, TypeEither):
                return H.either_swap().value
        case expr.MonadicEmbed():
            return H.term_lambda("x", lambda x: H.pure(node.monad, P.apply(realize_term(node.f), x))).value
        case expr.Compose():
            # plain-only: P.compose(_t(node.g), _t(node.f))
            # para: shared param must be split and routed into both f and g via _child_arg
            # lax:  f yields T(B); must _bind before feeding into g
            return _contextual_term(
                node,
                lambda value, call_f, call_g: (
                    call_g(call_f(value)) if node.monad is None
                    else H.bind(node.monad, call_f(value), "ctx_b", call_g)
                ),
            )
        case expr.Parallel() | expr.Pair():
            # Parallel splits a product input; Pair shares one input.
            # Lax component results are sequenced into one product effect.
            return _contextual_term(
                node,
                lambda value, call_f, call_g: H.pair_effects(
                    node.monad,
                    call_f(P.first(value) if isinstance(node, expr.Parallel) else value),
                    call_g(P.second(value) if isinstance(node, expr.Parallel) else value),
                ),
            )
        case expr.Case():
            # For lax branches, the selected branch already returns T(C), so
            # case elimination should return that effect directly.
            return _contextual_term(
                node,
                lambda value, call_f, call_g: P.apply(H.eithers_either(
                    P.lam("ctx_left", call_f(P.var("ctx_left"))),
                    P.lam("ctx_right", call_g(P.var("ctx_right"))),
                ), value),
            )
        case expr.Prim(raw, _, _):
            return raw
        case _:
            raise TypeError(f"realize: unknown MorphismExpr {type(node).__name__!r}")
