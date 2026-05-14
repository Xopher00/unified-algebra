"""Hydra realization for validated DSL syntax.

This module is the backend interpretation layer.  It assumes morphism
composition and visible types have already been checked by ``morphisms.py`` and
translates expression trees to raw Hydra terms.  Polynomial functor actions
live in ``functors.py``; optic recursion helpers live in ``recursion.py``.
"""

from __future__ import annotations

from collections.abc import Callable

from hydra.core import Term
from hydra.phantoms import TTerm
import hydra.dsl.meta.phantoms as P
from hydra.graph import Primitive
from hydra.dsl.python import Right
import hydra.dsl.terms as Terms
import hydra.reduction as R
import hydra.analysis as Analysis
from hydra.context import Context
from hydra.graph import Graph

from . import terms as T
from unialg.semantics.morphisms import Morphism
from unialg.syntax import expressions as expr
from unialg.objects import Name, Monad, Type, TypeUnit, TypePair, TypeEither, ExpType, TypeScheme


def realize_term(node: expr.MorphismExpr, _prims=None) -> TTerm:
    """Translate a morphism expression into a typed Hydra term handle."""
    return TTerm(realize(node, _prims))


def realize_term_normalized(node: expr.MorphismExpr, graph=None, _prims=None) -> TTerm:
    """Translate and structurally normalize a morphism expression."""
    return T.normalize_term(realize(node, _prims), graph)


def realize_normalized(node: expr.MorphismExpr, graph=None, _prims: list | None = None) -> Term:
    """Translate a morphism expression into a normalized raw Hydra term."""
    return T.normalize_term(realize(node, _prims), graph).value


def analyze_realized_function(node: expr.MorphismExpr, _prims=None):
    """Analyze a realized Hydra function term."""

    term = realize(node, _prims)
    return Analysis.analyze_function_term(
        Context(),
        lambda g: g,
        lambda g, _env: g,
        Graph(),
        term,
    )


def realized_is_self_tail_recursive(name: Name, node: expr.MorphismExpr, _prims=None) -> bool:
    """Check whether a realized recursive body is self-tail-recursive."""
    return Analysis.is_self_tail_recursive(name, realize(node, _prims))


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


def _exponential_action(body_inner: expr.PolyExpr, h: TTerm) -> TTerm:
    """Realize the functor action for an exponential position ``S → G(X)``.

    Pre-composes ``h`` under the function: ``g ↦ λs. G(h)(g(s))``.
    """
    fh = poly_action_term(body_inner, h)
    return T.lam2("lp_g", "lp_s", lambda g, s: P.apply(fh, P.apply(g, s)))


def poly_action_term(body: expr.PolyExpr, h: TTerm, monad: Monad | None = None) -> TTerm:
    """Build the Hydra term for mapping or traversing a polynomial body."""
    match body:
        case expr.Id():
            return h
        case expr.Zero():
            return T.absurd()
        case expr.One():
            return T.pure_unit(monad)
        case expr.Const(_):
            return T.pure_identity(monad)
        case expr.Prod(left, right):
            left_action = poly_action_term(left, h, monad)
            right_action = poly_action_term(right, h, monad)
            return T.product_action(monad, left_action, right_action)
        case expr.Sum(left, right):
            left_action = poly_action_term(left, h, monad)
            right_action = poly_action_term(right, h, monad)
            return T.case_effects(monad, left_action, right_action)
        case expr.Exp(_, body_inner):
            if monad is not None:
                raise TypeError("_poly_action_term: Exp polynomials are not traversable for arbitrary monads")
            return _exponential_action(body_inner, h)
        case expr.Maybe(body_inner):
            return T.maybe_effects(monad, poly_action_term(body_inner, h, monad))
        case expr.List(body_inner):
            return T.list_effects(monad, poly_action_term(body_inner, h, monad))
        case _:
            raise TypeError(f"_poly_action_term: unknown PolyExpr {type(body).__name__!r}")
        

def _apply_parametric_poly_fmap(
    node: expr.PolyFmap, h_term: TTerm, ctx_x: TTerm) -> TTerm:
    """Realize a parametric ``PolyFmap`` node against a combined ``param × value`` input.

    Splits ``ctx_x`` into its parameter prefix and visible value, builds a
    section that closes over the parameter, then applies the functor action.
    """
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


def _mk_child_call(
    child_term: TTerm,
    child_param: Type,
    side: str,
    param: Type,
    param_term: TTerm | None,
) -> Callable[[TTerm], TTerm]:
    """Return a closure that calls ``child_term`` with the correct parameter fragment prepended.

    ``side`` selects whether to take the ``"left"`` or ``"right"`` fragment of
    a combined ``g_param × f_param`` parameter when the child has a different param
    than the parent.  If the child needs no parameter (``child_param == TypeUnit``),
    the argument is passed through unchanged.
    """
    def call(v: TTerm) -> TTerm:
        """Apply the child term, prepending its contextual parameter if needed."""
        child_param_term = _child_param(param_term, param, child_param, side)
        arg = v if child_param_term is None else P.pair(child_param_term, v)
        return P.apply(child_term, arg)

    return call


def _contextual_term(node: expr.ContextualBinary, build, _prims=None) -> Term:
    """Realize a contextual node by routing child calls and wrapping ``ctx_x``."""
    f_term = realize_term(node.f, _prims)
    g_term = realize_term(node.g, _prims)

    def wrapped(x: TTerm) -> TTerm:
        """Split contextual input and delegate to the node-specific builder."""
        param_term, value = split_input(x, node.param)
        call_f = _mk_child_call(f_term, node.f_param, "right", node.param, param_term)
        call_g = _mk_child_call(g_term, node.g_param, "left", node.param, param_term)
        return build(value, call_f, call_g)

    return T.term_lambda("ctx_x", wrapped).value


def realize(node: expr.MorphismExpr, _prims: list | None = None) -> Term:
    """Translate a morphism expression into a raw Hydra term."""
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
                lambda p: P.pair(
                    P.first(P.first(p)),
                    P.pair(P.second(P.first(p)), P.second(p)),
                ),
            ).value

        case expr.Symmetry(dom=dom, cod=cod):
            if isinstance(dom, TypePair) and isinstance(cod, TypePair):
                return T.pair_swap().value
            if isinstance(dom, TypeEither) and isinstance(cod, TypeEither):
                return T.either_swap().value

        case expr.MonadicEmbed():
            f_term = realize_term(node.f, _prims)
            return T.term_lambda("x", lambda x: T.pure(node.monad, P.apply(f_term, x))).value

        case expr.Compose():
            return _contextual_term(
                node,
                lambda value, call_f, call_g:
                    call_g(call_f(value))
                    if node.monad is None
                    else T.bind(node.monad, call_f(value), "ctx_b", call_g),
                _prims,
            )

        case expr.Parallel():
            return _contextual_term(
                node,
                lambda value, call_f, call_g: T.pair_effects(
                    node.monad,
                    call_f(P.first(value)),
                    call_g(P.second(value)),
                ),
                _prims,
            )

        case expr.Pair():
            return _contextual_term(
                node,
                lambda value, call_f, call_g: T.pair_effects(
                    node.monad,
                    call_f(value),
                    call_g(value),
                ),
                _prims,
            )

        case expr.Case():
            return _contextual_term(
                node,
                lambda value, call_f, call_g: P.apply(
                    T.eithers_either(
                        T.term_lambda("ctx_left", lambda v: call_f(v)),
                        T.term_lambda("ctx_right", lambda v: call_g(v)),
                    ),
                    value,
                ),
                _prims,
            )

        case expr.PolyFmap():
            h_term = realize_term(node.f, _prims)
            if node.param == TypeUnit():
                return poly_action_term(node.body, h_term, node.monad).value
            return T.term_lambda("ctx_x", lambda ctx_x: _apply_parametric_poly_fmap(node, h_term, ctx_x)).value

        case expr.SelfRef(name, _, _):
            return P.primitive(Name(name)).value

        case expr.AlgExpr(name, body, dom, cod) | expr.Cata(name, body, dom, cod) | expr.Ana(name, body, dom, cod):
            prim_name = Name(name)
            raw_body_ref = [None]

            def impl(ctx, graph, args):
                """Hydra primitive implementation for a deferred recursive equation."""
                term = Terms.apply(raw_body_ref[0].value, args[0])
                result = R.reduce_term(ctx, graph, True, term)
                if isinstance(result, Right):
                    return result
                raise RuntimeError(f"{name} reduction failed: {result!r}")

            prim = Primitive(
                prim_name,
                TypeScheme((), ExpType(dom, cod), None),
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
