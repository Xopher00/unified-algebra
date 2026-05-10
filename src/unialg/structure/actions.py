"""Polynomial functor actions for the unialg DSL.

This module lifts already-typed morphisms through polynomial functors.  It
delegates child morphism realization to ``realize.py`` and keeps backend
primitive choices behind ``hydra_primitives.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import hydra.dsl.meta.phantoms as P
from hydra.core import Type
from hydra.phantoms import TTerm

from unialg.objects import Monad, TypeUnit
from unialg.syntax import expressions as expr
from unialg.semantics.morphisms import Morphism, raw_signature
from unialg.semantics.functors import Functor
from . import terms as H
from . import realize


def primitive_from_raw(raw: TTerm, dom: Type, cod: Type, like: Morphism, extra: tuple = ()) -> tuple[TTerm, Morphism]:
    """Assemble a raw term as a context-preserving primitive morphism."""
    return raw, Morphism(
        node=expr.Prim(raw.value, dom, cod),
        param=like.param,
        monad=like.monad,
        aux_primitives=extra + like.aux_primitives,
    )


def _poly_action_term(body: expr.PolyExpr, h: TTerm, monad: Monad | None = None) -> TTerm:
    """Build the Hydra term for mapping or traversing a polynomial body."""
    match body:
        case expr.Id():
            return h
        case expr.One():
            if monad is None:
                return P.constant(P.unit())
            else:
                return H.term_lambda("unit_x", lambda _: H.pure(monad, P.unit()))
        case expr.Const(_):
            if monad is None:
                return P.identity()
            else:
                return H.term_lambda("const_x", lambda x: H.pure(monad, x))
        case expr.Prod(left, right) | expr.Sum(left, right):
            left_action = _poly_action_term(left, h, monad)
            right_action = _poly_action_term(right, h, monad)
            if isinstance(body, expr.Sum):
                return H.case_effects(monad, left_action, right_action)
            if monad is None:
                return H.pairs_bimap(left_action, right_action)
            return H.term_lambda(
                "fm_x",
                lambda x: H.pair_effects(
                    monad,
                    P.apply(left_action, P.first(x)),
                    P.apply(right_action, P.second(x)),
                ),
            )
        case expr.Zero():
            return H.absurd()
        case expr.Exp(_, body_inner):
            if monad is None:
                fh = _poly_action_term(body_inner, h)
                return H.term_lambda(
                    "lp_g",
                    lambda g: H.term_lambda("lp_s", lambda s: P.apply(fh, P.apply(g, s))),
                )
            else:
                raise TypeError("_poly_action_term: Exp polynomials are not traversable for arbitrary monads")
        case expr.Maybe(body_inner) | expr.List(body_inner):
            item_action = _poly_action_term(body_inner, h, monad)
            if isinstance(body, expr.List):
                return H.list_effects(monad, item_action)
            return H.maybe_effects(monad, item_action)
        case _:
            raise TypeError(f"_poly_action_term: unknown PolyExpr {type(body).__name__!r}")


def _lift_source_action(
    h: Morphism,
    fa_type: Type,
) -> tuple[Type, Callable[[expr.PolyExpr], TTerm]]:
    """Adapt a morphism so a polynomial action can use its visible input."""
    h_term = realize.realize_term(h.node)
    if h.param == TypeUnit():
        return fa_type, lambda body: _poly_action_term(body, h_term, h.monad)

    ctx_x = P.var("ctx_x")
    param_term, _ = realize.split_input(ctx_x, h.param)
    section = H.term_lambda("section_a", lambda a: P.apply(h_term, P.pair(param_term, a)))

    def build_action(body: expr.PolyExpr) -> TTerm:
        lifted = _poly_action_term(body, section, h.monad)
        return H.term_lambda("ctx_x", lambda x: P.apply(lifted, realize.split_input(x, h.param)[1]))

    raw_dom, _ = raw_signature(h.param, h.monad, fa_type, fa_type)
    return raw_dom, build_action


def poly_fmap(functor: Functor, h: Morphism) -> Morphism:
    """Apply a polynomial functor to a typed morphism.

    Preserves ``h.param``, ``h.monad``, and ``h.aux_primitives``.  The visible
    type law is ``poly_fmap(F, h) : F(h.dom()) -> F(h.cod())``.  For lax
    morphisms, the raw codomain is wrapped in ``h.monad``.
    """
    fa_type = functor.apply(h.dom())
    fb_type = functor.apply(h.cod())
    raw_dom, build_action = _lift_source_action(h, fa_type)
    raw = build_action(functor.body)
    _, prim_cod = raw_signature(h.param, h.monad, fa_type, fb_type)
    _, morphism = primitive_from_raw(raw, raw_dom, prim_cod, h)
    return morphism
