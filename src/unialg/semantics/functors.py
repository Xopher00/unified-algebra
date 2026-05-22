"""Polynomial functor semantics for the unialg DSL.

This module sits between pure syntax and structure realization.  It gives
``PolyExpr`` values their object-level meaning by computing ``F(A)``.  The
inverse question, ``given F(A), recover A``, is delegated to Hydra's type
unifier rather than reimplementing type matching here.
"""

from __future__ import annotations
from dataclasses import dataclass

from . import morphisms
from . import typeops as Ty
from unialg.syntax import expressions as expr
from unialg.objects import (
    Type, TypeUnit, TypeList, TypeMaybe,
    ProductType, SumType, ExpType, VoidType,
)

@dataclass(frozen=True)
class Functor:
    """Named polynomial endofunctor descriptor.

    ``body`` is a ``PolyExpr`` describing the object action.  For now,
    ``category="poset"`` is restricted to ``Id`` because nontrivial poset
    functors are not implemented.
    """
    name: str
    body: expr.PolyExpr
    category: str = "set"

    def __post_init__(self) -> None:
        if self.category == "poset" and not isinstance(self.body, expr.Id):
            raise ValueError(
                f"Functor {self.name!r}: category='poset' requires body=Id, "
                f"got {type(self.body).__name__}"
            )

    def summands(self) -> tuple[expr.PolyExpr, ...]:
        """Flatten this functor's top-level coproduct summands left-to-right."""
        return _flatten_sum(self.body)

    def x_arity(self) -> int:
        """Count occurrences of the recursion variable ``Id`` in this functor."""
        return _count_id(self.body)

    def consts(self) -> list[Type]:
        """Collect constant spaces and exponential bases depth-first."""
        return _collect_consts(self.body)

    def apply(self, space: Type) -> Type:
        """Object action: compute ``F(space)`` for this functor."""
        return apply_poly(self.body, space)

    def compose(self, inner: "Functor", name: str | None = None) -> "Functor":
        """Return functor composition ``self ∘ inner``.

        The resulting functor first applies ``inner`` and then ``self``:
        ``self.compose(inner).apply(A) == self.apply(inner.apply(A))``.
        """
        if self.category != inner.category:
            raise ValueError(
                f"Cannot compose functors with different categories: "
                f"{self.category!r} != {inner.category!r}"
            )
        return Functor(
            name or f"{self.name}∘{inner.name}",
            compose_poly(self.body, inner.body),
            category=self.category,
        )

    def unapply(self, fa: Type) -> Type:
        """Solve ``F(A) = fa`` and return the recovered element type ``A``."""
        if self.x_arity() == 0:
            raise TypeError(
                f"Functor {self.name!r}.unapply: body contains no Id; no element type exists"
            )

        a_var = Ty.fresh_type_var()
        pattern = self.apply(a_var)
        match = Ty.unify(pattern, fa, f"{self.name}.unapply")
        recovered = Ty.apply_subst(match.substitution, a_var)
        Ty.roundtrip_equal(None, self.apply, recovered, fa, f"{self.name}.unapply round-trip")
        return recovered
    
    def map(self, h: morphisms.Morphism) -> morphisms.Morphism:
        """Semantic polynomial functor action on a typed morphism."""
        return poly_fmap(self, h)
    

def _has_exp(body: expr.PolyExpr) -> bool:
    if isinstance(body, expr.Exp):
        return True
    if isinstance(body, (expr.Sum, expr.Prod)):
        return _has_exp(body.left) or _has_exp(body.right)
    if isinstance(body, (expr.List, expr.Maybe, expr.Rose, expr.Tree)):
        return _has_exp(body.body)
    return False


def poly_fmap(functor: Functor, h: morphisms.Morphism) -> morphisms.Morphism:
    """Semantic polynomial functor action on a typed morphism.

    Builds syntax for F(h). Does not realize to Hydra and does not construct
    Prim.
    """
    if h.monad is not None and _has_exp(functor.body):
        raise TypeError("poly_fmap: Exp polynomials are not traversable for arbitrary monads")
    fa_type = functor.apply(h.dom())
    fb_type = functor.apply(h.cod())
    raw_dom, raw_cod = morphisms.raw_signature(h.param, h.monad, fa_type, fb_type)

    return morphisms.Morphism(
        node=expr.PolyFmap(
            body=functor.body,
            f=h.node,
            param=h.param,
            monad=h.monad,
            dom=raw_dom,
            cod=raw_cod,
        ),
        param=h.param,
        monad=h.monad,
        aux_primitives=h.aux_primitives,
    )


def zero() -> expr.Zero:
    """Constructor for the zero polynomial functor."""
    return expr.Zero()


def one() -> expr.One:
    """Constructor for the constant-one polynomial functor."""
    return expr.One()


def id_() -> expr.Id:
    """Constructor for the identity polynomial functor."""
    return expr.Id()


def const(space: Type) -> expr.Const:
    """Constructor for the constant functor at ``space``."""
    return expr.Const(space)


def sum_(f: expr.PolyExpr, g: expr.PolyExpr) -> expr.Sum:
    """Constructor for the coproduct of polynomial functors."""
    return expr.Sum(f, g)


def prod(f: expr.PolyExpr, g: expr.PolyExpr) -> expr.Prod:
    """Constructor for the product of polynomial functors."""
    return expr.Prod(f, g)


def exp(base: expr.PolyExpr, body: expr.PolyExpr) -> expr.Exp:
    """Constructor for the exponential polynomial ``base -> body``."""
    return expr.Exp(base, body)


def list_(body: expr.PolyExpr) -> expr.List:
    """Constructor for the list polynomial functor ``F(X) = List[body(X)]``."""
    return expr.List(body)


def maybe(body: expr.PolyExpr) -> expr.Maybe:
    """Constructor for lifted Maybe polynomial."""
    return expr.Maybe(body)


def rose(body: expr.PolyExpr) -> expr.Rose:
    """Constructor for rose-tree functor ``F(Y) = body(Y) × List[Y]``."""
    return expr.Rose(body)


def tree(body: expr.PolyExpr) -> expr.Tree:
    """Constructor for n-ary tree functor ``F(Y) = 1 + Rose(body)(Y)``."""
    return expr.Tree(body)


_COMPOSE_POLY: dict = {
    expr.Id:          lambda _n,  s: s,
    expr.Zero:        lambda  n, _s: n,
    expr.One:         lambda  n, _s: n,
    expr.Const:       lambda  n, _s: n,
    expr.Prod:        lambda  n,  s: expr.Prod(compose_poly(n.left, s), compose_poly(n.right, s)),
    expr.Sum:         lambda  n,  s: expr.Sum(compose_poly(n.left, s), compose_poly(n.right, s)),
    expr.PolyCompose: lambda  n,  s: expr.PolyCompose(compose_poly(n.left, s), compose_poly(n.right, s)),
    expr.Exp:         lambda  n,  s: expr.Exp(n.base, compose_poly(n.body, s)),  # base is fixed exponent domain
    expr.List:        lambda  n,  s: expr.List(compose_poly(n.body, s)),
    expr.Maybe:       lambda  n,  s: expr.Maybe(compose_poly(n.body, s)),
    expr.Rose:        lambda  n,  s: expr.Rose(compose_poly(n.body, s)),
    expr.Tree:        lambda  n,  s: expr.Tree(compose_poly(n.body, s)),
}

_APPLY_POLY: dict = {
    expr.Id:          lambda _n,  s: s,
    expr.Zero:        lambda _n, _s: VoidType(),
    expr.One:         lambda _n, _s: TypeUnit(),
    expr.Const:       lambda  n, _s: n.space,
    expr.Prod:        lambda  n,  s: ProductType(apply_poly(n.left, s), apply_poly(n.right, s)),
    expr.Sum:         lambda  n,  s: SumType(apply_poly(n.left, s), apply_poly(n.right, s)),
    expr.PolyCompose: lambda  n,  s: apply_poly(n.right, apply_poly(n.left, s)),
    expr.Exp:         lambda  n,  s: ExpType(apply_poly(n.base, TypeUnit()), apply_poly(n.body, s)),
    expr.List:        lambda  n,  s: TypeList(apply_poly(n.body, s)),
    expr.Maybe:       lambda  n,  s: TypeMaybe(apply_poly(n.body, s)),
    expr.Rose:        lambda  n,  s: ProductType(apply_poly(n.body, s), TypeList(s)),
    expr.Tree:        lambda  n,  s: SumType(TypeUnit(), ProductType(apply_poly(n.body, s), TypeList(s))),
}


def compose_poly(outer: expr.PolyExpr, inner: expr.PolyExpr) -> expr.PolyExpr:
    """Functor composition: substitute ``inner`` for every ``Id`` in ``outer``.

    ``compose_poly(F, G)`` produces the body of F∘G, so that
    ``apply_poly(compose_poly(F, G), A) == apply_poly(F, apply_poly(G, A))``.
    """
    handler = _COMPOSE_POLY.get(type(outer))
    if handler is None:
        raise TypeError(f"compose_poly: unknown PolyExpr {type(outer).__name__!r}")
    return handler(outer, inner)


def apply_poly(body: expr.PolyExpr, space: Type) -> Type:
    """Compute the object action ``F(space)`` for a polynomial functor body."""
    handler = _APPLY_POLY.get(type(body))
    if handler is None:
        raise TypeError(f"apply_poly: unknown PolyExpr {type(body).__name__!r}")
    return handler(body, space)


def _flatten_sum(node: expr.PolyExpr) -> tuple[expr.PolyExpr, ...]:
    if isinstance(node, expr.Sum):
        return _flatten_sum(node.left) + _flatten_sum(node.right)
    return (node,)


def _count_id(node: expr.PolyExpr) -> int:
    if isinstance(node, (expr.Zero, expr.One, expr.Const)):
        return 0
    if isinstance(node, expr.Id):
        return 1
    if isinstance(node, (expr.Sum, expr.Prod)):
        return _count_id(node.left) + _count_id(node.right)
    if isinstance(node, expr.Exp):
        return _count_id(node.body)
    if isinstance(node, (expr.List, expr.Maybe)):
        return _count_id(node.body)
    if isinstance(node, (expr.Rose, expr.Tree)):
        return _count_id(node.body) + 1  # +1 for the implicit List[Y] Id slot
    raise ValueError(f"_count_id: unknown PolyExpr {type(node).__name__!r}")


def _collect_consts(node: expr.PolyExpr) -> list[Type]:
    if isinstance(node, (expr.Zero, expr.One, expr.Id)):
        return []
    if isinstance(node, expr.Const):
        return [node.space]
    if isinstance(node, (expr.Sum, expr.Prod)):
        return _collect_consts(node.left) + _collect_consts(node.right)
    if isinstance(node, expr.Exp):
        return _collect_consts(node.base) + _collect_consts(node.body)
    if isinstance(node, (expr.List, expr.Maybe, expr.Rose, expr.Tree)):
        return _collect_consts(node.body)
    raise ValueError(f"_collect_consts: unknown PolyExpr {type(node).__name__!r}")
