"""Hydra realization for validated DSL syntax.

This module is the backend interpretation layer.  It assumes morphism
composition and visible types have already been checked by ``morphisms.py`` and
translates expression trees to raw Hydra terms.  Polynomial functor actions
live in ``functors.py``; optic recursion helpers live in ``recursion.py``.
"""

from __future__ import annotations

from hydra.core import Term
from hydra.phantoms import TTerm
import hydra.dsl.meta.phantoms as P
from hydra.graph import Primitive
from hydra.dsl.python import Right
import hydra.dsl.terms as Terms
import hydra.reduction as R


from . import terms as T
from unialg.semantics.morphisms import Morphism
from unialg.syntax import expressions as expr
from unialg.objects import Name, Monad, Type, TypeUnit, TypePair, TypeEither, ExpType, TypeScheme


def realize_term(node: expr.MorphismExpr, _prims=None) -> TTerm:
    """Translate a morphism expression into a typed Hydra term handle."""
    return TTerm(realize(node, _prims))


def primitive_from_raw(raw: TTerm, dom: Type, cod: Type, like: Morphism, extra: tuple = ()) -> tuple[TTerm, Morphism]:
    """Assemble a raw term as a context-preserving primitive morphism."""
    return raw, Morphism(
        node=expr.Prim(raw.value, dom, cod),
        param=like.param,
        monad=like.monad,
        aux_primitives=extra + like.aux_primitives,
    )


def split_input(x: TTerm, param: Type) -> tuple[TTerm | None, TTerm]:
    """Split a raw argument into optional parameter and visible value."""
    if param == TypeUnit():
        return None, x
    return P.first(x), P.second(x)


def poly_action_term(body: expr.PolyExpr, h: TTerm, monad: Monad | None = None) -> TTerm:
    """Build the Hydra term for mapping or traversing a polynomial body."""
    match body:
        case expr.Id():
            return h
        case expr.Zero():
            return T.absurd()
        case expr.One():
            if monad is None:
                return P.constant(P.unit())
            else:
                return T.term_lambda("unit_x", lambda _: T.pure(monad, P.unit()))
        case expr.Const(_):
            if monad is None:
                return P.identity()
            else:
                return T.term_lambda("const_x", lambda x: T.pure(monad, x))
        case expr.Prod(left, right) | expr.Sum(left, right):
            left_action = poly_action_term(left, h, monad)
            right_action = poly_action_term(right, h, monad)
            if isinstance(body, expr.Sum):
                return T.case_effects(monad, left_action, right_action)
            if monad is None:
                return T.pairs_bimap(left_action, right_action)
            return T.term_lambda(
                "fm_x",
                lambda x: T.pair_effects(
                    monad,
                    P.apply(left_action, P.first(x)),
                    P.apply(right_action, P.second(x)),
                ),
            )
        case expr.Exp(_, body_inner):
            if monad is None:
                fh = poly_action_term(body_inner, h)
                return T.term_lambda(
                    "lp_g",
                    lambda g: T.term_lambda("lp_s", lambda s: P.apply(fh, P.apply(g, s))),
                )
            else:
                raise TypeError("_poly_action_term: Exp polynomials are not traversable for arbitrary monads")
        case expr.Maybe(body_inner) | expr.List(body_inner):
            item_action = poly_action_term(body_inner, h, monad)
            if isinstance(body, expr.List):
                return T.list_effects(monad, item_action)
            return T.maybe_effects(monad, item_action)
        case _:
            raise TypeError(f"_poly_action_term: unknown PolyExpr {type(body).__name__!r}")
        

def _apply_parametric_poly_fmap(
    node: expr.PolyFmap, h_term: TTerm, ctx_x: TTerm) -> TTerm:
    param_term, visible_x = split_input(ctx_x, node.param)
    section = T.term_lambda(
        "section_a",
        lambda a: P.apply(h_term, P.pair(param_term, a)),
    )
    lifted = poly_action_term(
        node.body,
        section,
        node.monad,
    )
    return P.apply(lifted, visible_x)


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
    return T.term_lambda("ctx_x", wrapped).value


def realize(node: expr.MorphismExpr, _prims: list | None = None) -> Term:
    """Translate a morphism expression into a raw Hydra term.

    The returned term expects the node's raw domain, not necessarily the visible
    ``Morphism.dom()``.  For parametric/lax contextual nodes, the raw domain and
    codomain are already stored on the expression by ``morphisms.py``.
    """
    match node:
        case expr.Identity():
            return P.identity().value
        case expr.Copy():
            return T.term_lambda("x", lambda x: P.pair(x, x)).value
        case expr.Delete():
            return P.constant(P.unit()).value
        case expr.First():
            return T.pair_first().value
        case expr.Second():
            return T.pair_second().value
        case expr.Left():
            return T.left_injection().value
        case expr.Right():
            return T.right_injection().value
        case expr.Absurd():
            return T.absurd().value
        case expr.Assoc():
            return T.term_lambda(
                "p", 
                lambda p: P.pair(P.first(P.first(p)), P.pair(P.second(P.first(p)), P.second(p))),
            ).value
        case expr.Symmetry(dom=dom, cod=cod):
            if isinstance(dom, TypePair) and isinstance(cod, TypePair):
                return T.pair_swap().value
            if isinstance(dom, TypeEither) and isinstance(cod, TypeEither):
                return T.either_swap().value
        case expr.MonadicEmbed():
            return T.term_lambda("x", lambda x: T.pure(node.monad, P.apply(realize_term(node.f), x))).value
        case expr.Compose():
            # plain-only: P.compose(_t(node.g), _t(node.f))
            # para: shared param must be split and routed into both f and g via _child_arg
            # lax:  f yields T(B); must _bind before feeding into g
            return _contextual_term(
                node,
                lambda value, call_f, call_g: (
                    call_g(call_f(value)) if node.monad is None
                    else T.bind(node.monad, call_f(value), "ctx_b", call_g)
                ),
            )
        case expr.Parallel() | expr.Pair():
            # Parallel splits a product input; Pair shares one input.
            # Lax component results are sequenced into one product effect.
            return _contextual_term(
                node,
                lambda value, call_f, call_g: T.pair_effects(
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
                lambda value, call_f, call_g: P.apply(T.eithers_either(
                    P.lam("ctx_left", call_f(P.var("ctx_left"))),
                    P.lam("ctx_right", call_g(P.var("ctx_right"))),
                ), value),
            )
        case expr.PolyFmap():
            h_term = realize_term(node.f)
            if node.param == TypeUnit():
                return poly_action_term(node.body, h_term, node.monad).value
            return T.term_lambda(
                "ctx_x",
                lambda ctx_x: _apply_parametric_poly_fmap(node, h_term, ctx_x),
            ).value
        case expr.SelfRef(name, _, _):
            return P.primitive(Name(name)).value
        case expr.AlgExpr(name, body, dom, cod) | expr.Cata(name, body, dom, cod) | expr.Ana(name, body, dom, cod):
                prim_name = Name(name)
                raw_body_ref = [None]
                def impl(ctx, graph, args):
                    term = Terms.apply(raw_body_ref[0].value, args[0])
                    result = R.reduce_term(ctx, graph, True, term)
                    if isinstance(result, Right):
                        return result
                    raise RuntimeError(f"{name} reduction failed: {result!r}")
                prim = Primitive(
                    prim_name,
                    TypeScheme(variables=frozenset(), body=ExpType(dom, cod), constraints=None),
                    impl,
                )
                if _prims is not None:
                    _prims.append(prim)
                raw_body_ref[0] = TTerm(realize(body, _prims))
                return P.primitive(prim_name).value
        case expr.Prim(raw, _, _):
            return raw
        case _:
            raise TypeError(f"realize: unknown MorphismExpr {type(node).__name__!r}")
