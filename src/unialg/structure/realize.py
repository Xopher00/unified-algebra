"""Hydra realization for validated DSL syntax.

This module is the backend interpretation layer.  It assumes morphism
composition and visible types have already been checked by ``morphisms.py`` and
translates expression trees to raw Hydra terms.  Polynomial functor actions
live in ``functors.py``; recursive scheme nodes are emitted here as Hydra
primitives.
"""

from __future__ import annotations

from collections.abc import Callable

from hydra.context import Context
from hydra.core import Term
from hydra.graph import Graph, Primitive
from hydra.phantoms import TTerm
import hydra.dsl.meta.phantoms as P
from hydra.dsl.python import Right
import hydra.dsl.terms as Terms
import hydra.reduction as R
import hydra.show.errors as ShowErrors
import hydra.analysis as Analysis

from . import terms as T
from unialg.semantics.morphisms import Morphism
from unialg.semantics.functors import Functor
from unialg.semantics.optics import Optic
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
        Context(), lambda g: g,
        lambda g, _env: g,
        Graph(), term,
    )


def realized_is_self_tail_recursive(name: Name, node: expr.MorphismExpr, _prims=None) -> bool:
    """Check whether a realized recursive body is self-tail-recursive."""
    return Analysis.is_self_tail_recursive(name, realize(node, _prims))


def primitive_from_raw(
    raw: TTerm, dom: Type, cod: Type, like: Morphism, extra: tuple = ()
) -> tuple[TTerm, Morphism]:
    """Assemble a raw term as a context-preserving primitive morphism."""
    return raw, Morphism(
        node=expr.Prim(raw.value, dom, cod),
        param=like.param, monad=like.monad,
        aux_primitives=extra + like.aux_primitives,
    )


def _exp_action(body, h, monad):
    if monad is not None:
        raise TypeError(
            "poly_action_term: Exp polynomials are not traversable for arbitrary monads"
        )
    fh = poly_action_term(body.body, h)  # type: ignore[union-attr]
    return T.lam2("lp_g", "lp_s", lambda g, s: P.apply(fh, P.apply(g, s)))


def _poly_compose_action(body, h, monad):
    mapped = poly_action_term(body.left, h, monad)  # type: ignore[union-attr]
    return poly_action_term(body.right, mapped, monad)  # type: ignore[union-attr]

def _prod_action(b, h, m):
    return T.product_effects(m, poly_action_term(b.left, h, m), poly_action_term(b.right, h, m))


def _sum_action(b, h, m):
    return T.case_effects(m, poly_action_term(b.left, h, m), poly_action_term(b.right, h, m))


def _list_action(b, h, m):
    return T.list_effects(m, poly_action_term(b.body, h, m))


def _maybe_action(b, h, m):
    return T.maybe_effects(m, poly_action_term(b.body, h, m))


def _rose_action(b, h, m):
    return T.product_effects(m, poly_action_term(b.body, h, m), T.list_effects(m, h))


def _tree_action(b, h, m):
    return T.maybe_effects(m, _rose_action(b, h, m))


_POLY_ACTION_DISPATCH: dict = {
    expr.Id:          lambda b, h, m: h,
    expr.Zero:        lambda b, h, m: T.absurd(),
    expr.One:         lambda b, h, m: T.pure_unit(m),
    expr.Const:       lambda b, h, m: T.pure_identity(m),
    expr.Exp:         _exp_action,
    expr.PolyCompose: _poly_compose_action,
    expr.Prod:        _prod_action,
    expr.Sum:         _sum_action,
    expr.List:        _list_action,
    expr.Maybe:       _maybe_action,
    expr.Rose:        _rose_action,
    expr.Tree:        _tree_action,
}


def poly_action_term(body: expr.PolyExpr, h: TTerm, monad: Monad | None = None) -> TTerm:
    """Build the Hydra term for mapping or traversing a polynomial body."""
    handler = _POLY_ACTION_DISPATCH.get(type(body))
    if handler is not None:
        return handler(body, h, monad)
    raise TypeError(f"poly_action_term: unknown PolyExpr {type(body).__name__!r}")


def split_input(x: TTerm, param: Type) -> tuple[TTerm | None, TTerm]:
    """Split a raw argument into optional parameter and visible value."""
    if param == TypeUnit():
        return None, x
    return P.first(x), P.second(x)


def _apply_parametric_poly_fmap(node: expr.PolyFmap, h_term: TTerm, ctx_x: TTerm) -> TTerm:
    """Realize a parametric ``PolyFmap`` node against a combined ``param * value`` input.

    Splits ``ctx_x`` into its parameter prefix and visible value, builds a
    section that closes over the parameter, then applies the functor action.
    """
    param_term, visible_x = split_input(ctx_x, node.param)
    section = T.term_lambda(
        "section_a",
        lambda a: P.apply(h_term, P.pair(param_term, a)),
    )
    lifted = poly_action_term(
        node.body, section, node.monad,
    )
    return P.apply(lifted, visible_x)


def _make_child_caller(child_term: TTerm, parent_param: Type, child_param: Type,
    parent_param_term: TTerm | None, *, take_first: bool) -> Callable[[TTerm], TTerm]:
    """Build a caller that prepends the child's parameter fragment when needed."""
    if child_param == TypeUnit():
        return lambda value: P.apply(child_term, value)
    if parent_param_term is None or child_param == parent_param:
        child_param_term = parent_param_term
    else:
        child_param_term = (
            P.first(parent_param_term) if take_first else P.second(parent_param_term)
        )
    return lambda value: P.apply(child_term, P.pair(child_param_term, value))


def _contextual_term(node: expr.ContextualBinary, _prims=None) -> Term:
    f_term = realize_term(node.f, _prims)
    g_term = realize_term(node.g, _prims)
    op = _CONTEXTUAL_MORPHISMS[type(node)]
    def wrapped(x: TTerm) -> TTerm:
        parent_param_term, value = split_input(x, node.param)
        call_f = _make_child_caller(
            f_term, node.param, node.f_param, parent_param_term, take_first=False
        )
        call_g = _make_child_caller(
            g_term, node.param, node.g_param, parent_param_term, take_first=True
        )
        return op(node, value, call_f, call_g)
    return T.term_lambda("ctx_x", wrapped).value


def _realize_symmetry(node: expr.Symmetry) -> Term:
    dom, cod = node.dom, node.cod
    if isinstance(dom, TypePair) and isinstance(cod, TypePair):
        return T.pair_swap().value
    if isinstance(dom, TypeEither) and isinstance(cod, TypeEither):
        return T.either_swap().value
    raise TypeError(
        "realize: Symmetry requires product or sum operands, "
        f"got dom={type(dom).__name__!r}, cod={type(cod).__name__!r}"
    )


def _realize_assoc() -> Term:
    return T.term_lambda(
        "p",
        lambda p: P.pair(
            P.first(P.first(p)),
            P.pair(P.second(P.first(p)), P.second(p)),
        ),
    ).value


def _realize_distribute(fixed, sumpart, mk_pair) -> Term:
    return T.term_lambda(
        "p",
        lambda p: P.apply(
            T.eithers_bimap(
                T.term_lambda("a", lambda a: mk_pair(fixed(p), a)),
                T.term_lambda("b", lambda b: mk_pair(fixed(p), b)),
            ),
            sumpart(p),
        ),
    ).value


def _realize_distribute_left() -> Term:
    return _realize_distribute(P.first, P.second, P.pair)


def _realize_distribute_right() -> Term:
    return _realize_distribute(P.second, P.first, lambda fixed, x: P.pair(x, fixed))


def _realize_monadic_embed(node: expr.MonadicEmbed, _prims) -> Term:
    f_term = realize_term(node.f, _prims)
    return T.term_lambda("x", lambda x: T.pure(node.monad, P.apply(f_term, x))).value


def _compose_op(n, v, f, g):
    if n.monad is None:
        return g(f(v))
    return T.bind(n.monad, f(v), "ctx_b", g)


def _pair_effects_op(left_of, right_of):
    return lambda n, v, f, g: T.pair_effects(n.monad, f(left_of(v)), g(right_of(v)))


def _case_op(_n, v, f, g):
    branches = T.eithers_either(T.term_lambda("ctx_left", f), T.term_lambda("ctx_right", g))
    return P.apply(branches, v)


def _realize_poly_fmap(node: expr.PolyFmap, _prims) -> Term:
    h_term = realize_term(node.f, _prims)
    if node.param == TypeUnit():
        return poly_action_term(node.body, h_term, node.monad).value
    return T.term_lambda(
        "ctx_x", lambda ctx_x: _apply_parametric_poly_fmap(node, h_term, ctx_x)
    ).value


def _realize_alg_expr(node: expr.AlgExpr, _prims) -> Term:
    prim_name = Name(node.name)
    raw_body_term = None

    def impl(ctx, graph, args):
        term = Terms.apply(raw_body_term.value, args[0])  # type: ignore[union-attr]
        result = R.reduce_term(ctx, graph, True, term)
        if isinstance(result, Right):
            return result
        raise RuntimeError(f"{node.name} reduction failed: {ShowErrors.error(result.value)}")

    prim = Primitive(
        prim_name,
        TypeScheme(variables=frozenset(), body=ExpType(node.dom, node.cod), constraints=None),
        impl,
    )
    if _prims is not None:
        _prims.append(prim)
    raw_body_term = TTerm(realize(node.body, _prims))
    return P.primitive(prim_name).value


def _realize_backend_prim(node: expr.BackendPrim, _prims) -> Term:
    if not node.args:
        return T.primitive_wrapper_term(node.primitive.name, node.arity)
    arg_terms = [realize_term(a, _prims) for a in node.args]
    def _build_applied(x):
        return T.apply_curried_primitive(node.primitive.name, [P.apply(at, x) for at in arg_terms])
    return T.term_lambda("x", _build_applied).value


_FIXED_MORPHISMS: dict = {
    expr.Identity:       lambda n, _p: P.identity().value,
    expr.Copy:           lambda n, _p: T.term_lambda("x", lambda x: P.pair(x, x)).value,
    expr.Delete:         lambda n, _p: P.constant(P.unit()).value,
    expr.First:          lambda n, _p: T.pair_first().value,
    expr.Second:         lambda n, _p: T.pair_second().value,
    expr.Left:           lambda n, _p: T.left_injection().value,
    expr.Right:          lambda n, _p: T.right_injection().value,
    expr.Absurd:         lambda n, _p: T.absurd().value,
    expr.Symmetry:       lambda n, _p: _realize_symmetry(n),
    expr.Assoc:          lambda _n, _p: _realize_assoc(),
    expr.DistributeLeft: lambda _n, _p: _realize_distribute_left(),
    expr.DistributeRight: lambda _n, _p: _realize_distribute_right(),
    expr.Coerce:         lambda n, _p: P.identity().value,
}

_CONTEXTUAL_MORPHISMS: dict = {
    expr.Compose:  _compose_op,
    expr.Parallel: _pair_effects_op(P.first, P.second),
    expr.Pair:     _pair_effects_op(lambda v: v, lambda v: v),
    expr.Case:     _case_op,
}

_SPECIAL_MORPHISMS: dict = {
    expr.MonadicEmbed: _realize_monadic_embed,
    expr.PolyFmap:     _realize_poly_fmap,
    expr.AlgExpr:      _realize_alg_expr,
    expr.BackendPrim:  _realize_backend_prim,
    expr.SelfRef:      lambda n, _p: P.primitive(Name(n.name)).value,
    expr.Prim:         lambda n, _p: n.raw,
}


def realize(node: expr.MorphismExpr, _prims: list | None = None) -> Term:
    """Translate a morphism expression into a raw Hydra term."""
    for handlers in (_FIXED_MORPHISMS, _SPECIAL_MORPHISMS):
        handler = handlers.get(type(node))
        if handler is not None:
            return handler(node, _prims)
    if type(node) in _CONTEXTUAL_MORPHISMS:
        return _contextual_term(node, _prims)
    if isinstance(node, expr.DomainPrim):
        raise NotImplementedError(
            f"DomainPrim({node.tag!r}) must be rewritten before realize — "
            f"ensure the domain's finalize hook ran"
        )
    raise TypeError(f"realize: unknown MorphismExpr {type(node).__name__!r}")


def recursive_carrier(*, functor: Functor, carrier: Type, unroll, roll) -> Optic:
    """Build a carrier ``Optic`` from Python unroll/roll callables.

    ``unroll`` and ``roll`` are called with a single ``TTerm`` argument and must
    return a ``TTerm``.  They are wrapped as ``Prim`` morphisms and assembled
    into an ``Optic`` directly.
    """
    layer = functor.apply(carrier)
    unroll_term = T.normalize_term(T.term_lambda("x", unroll)).value
    roll_term = T.normalize_term(T.term_lambda("layer", roll)).value
    return Optic(
        functor=functor,
        forward=Morphism(expr.Prim(unroll_term, carrier, layer)),
        backward=Morphism(expr.Prim(roll_term, layer, carrier)),
        carrier=carrier,
    )
