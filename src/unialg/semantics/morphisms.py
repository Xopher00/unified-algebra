"""Typed semantic morphisms for the unialg DSL.

``expressions.py`` defines syntax.  This module gives that syntax typed meaning:

* derive domains and codomains for ``MorphismExpr`` nodes
* build typed ``Morphism`` handles
* check composition/product/sum compatibility

No Hydra reduction happens here.  Term realization belongs to
``structure/realize.py`` and execution orchestration belongs to ``main.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import typeops as Ty
from unialg.syntax import expressions as expr
from hydra.core import Name, TypeVariable
from unialg.objects import (
    Monad, Type, TypeUnit, TypePair, TypeEither,
    ProductType, SumType, VoidType, show_type,
)


class MorphismError(TypeError):
    @classmethod
    def check(cls, graph, a: Type, b: Type, label: str) -> None:
        try:
            Ty.require_equal(graph, a, b, label)
        except TypeError as e:
            raise cls(str(e)) from e


def _collect_aux_primitives(*morphisms: Morphism) -> tuple:
    return tuple(p for m in morphisms for p in m.aux_primitives)


def _share_param(
    f_param: Type, g_param: Type, *, graph=None, allow_unification: bool = False,
) -> Type:
    try:
        return Ty.share_param(
            graph, f_param, g_param,
            "Cannot share contextual parameter",
            allow_unification=allow_unification,
        )
    except TypeError as e:
        raise MorphismError(str(e)) from e


@dataclass(frozen=True)
class Morphism:
    """Typed handle for a morphism expression.

    A ``Morphism`` wraps syntax with execution context:

    * plain: ``param == Unit`` and ``monad is None``, raw node is ``A -> B``
    * parametric: ``param == P``, raw node is ``P × A -> B``
    * lax/effectful: ``monad == T``, raw node is ``A -> T(B)``
    * lax-parametric: both, raw node is ``P × A -> T(B)``

    Public ``dom()`` and ``cod()`` hide the parameter prefix and monad wrapper.
    They validate that the raw node shape agrees with ``param`` and ``monad``.
    """
    node: expr.MorphismExpr
    param: Type = TypeUnit()
    monad: Monad | None = None
    aux_primitives: tuple = ()

    def dom(self) -> Type:
        """Return the visible domain, stripping a parameter prefix if present."""
        raw = dom_of(self.node)
        try:
            return Ty.visible_domain(raw, self.param, "parametric morphism domain")
        except TypeError as e:
            raise MorphismError(str(e)) from e

    def cod(self) -> Type:
        """Return the visible codomain, stripping the monad wrapper if present."""
        raw = cod_of(self.node)
        if self.monad is None:
            return raw
        inner = self.monad.unwrap(raw)
        if inner is None:
            raise MorphismError(
                f"codomain {show_type(raw)} is not wrapped by {self.monad!r}"
            )
        return inner

    def node_in(self, monad: Monad | None) -> expr.MorphismExpr:
        """Return this node adapted to ``monad``.

        Plain nodes can be embedded into a requested monad with
        ``MonadicEmbed``.  Already-lax nodes may only be used in the same monad.
        """
        if monad is None or self.monad == monad:
            return self.node
        if self.monad is not None:
            raise MorphismError(f"monad mismatch: {self.monad!r} != {monad!r}")
        return expr.MonadicEmbed(self.node, monad)

    def to_lax(self, monad: Monad | None) -> 'Morphism':
        """Return this morphism in the requested lax context if needed."""
        if monad is None or self.monad == monad:
            return self
        return Morphism(self.node_in(monad), self.param, monad, self.aux_primitives)


def _lr(t: TypePair | TypeEither) -> tuple[Type, Type]:
    """Extract the left and right components from a product or sum type."""
    if isinstance(t, TypePair):
        return t.value.first, t.value.second
    return t.value.left, t.value.right


def _assoc_cod(dom: Type) -> Type | None:
    """Build ``A⋆(B⋆C)`` from ``(A⋆B)⋆C``, or None if dom is wrong shape."""
    if not isinstance(dom, (TypePair, TypeEither)):
        return None
    dl, c = _lr(dom)
    if not isinstance(dl, type(dom)):
        return None
    a, b = _lr(dl)
    make = ProductType if isinstance(dom, TypePair) else SumType
    return make(a, make(b, c))


def _symmetry_cod(dom: Type) -> Type | None:
    """Build ``B⋆A`` from ``A⋆B``, or None if dom is wrong shape."""
    if not isinstance(dom, (TypePair, TypeEither)):
        return None
    left, right = _lr(dom)
    make = ProductType if isinstance(dom, TypePair) else SumType
    return make(right, left)


def _distl_cod(dom: Type) -> Type | None:
    """Build ``(A×B)+(A×C)`` from ``A×(B+C)``, or None if dom is wrong shape."""
    if not isinstance(dom, TypePair):
        return None
    a, bc = _lr(dom)
    if not isinstance(bc, TypeEither):
        return None
    b, c = _lr(bc)
    return SumType(ProductType(a, b), ProductType(a, c))


def _distr_cod(dom: Type) -> Type | None:
    """Build ``(A×C)+(B×C)`` from ``(A+B)×C``, or None if dom is wrong shape."""
    if not isinstance(dom, TypePair):
        return None
    ab, c = _lr(dom)
    if not isinstance(ab, TypeEither):
        return None
    a, b = _lr(ab)
    return SumType(ProductType(a, c), ProductType(b, c))


# ---------------------------------------------------------------------------
# Type derivation
# ---------------------------------------------------------------------------

_U  = TypeUnit()
_PU = ProductType(_U, _U)
_SU = SumType(_U, _U)

_SELF_DESCRIBING = (
    expr.ContextualBinary, expr.Prim, expr.DomainPrim,
    expr.BackendPrim, expr.PolyFmap, expr.SelfRef, expr.AlgExpr,
)

_SIG_LEAF: dict = {
    expr.Identity:     lambda n, pn: (n.space, n.space),
    expr.Copy:         lambda n, pn: (n.space, ProductType(n.space, n.space)),
    expr.Delete:       lambda n, pn: (n.space, TypeUnit()),
    expr.First:        lambda n, pn: (n.ab, n.ab.value.first),
    expr.Second:       lambda n, pn: (n.ab, n.ab.value.second),
    expr.Left:         lambda n, pn: (n.ab.value.left, n.ab),
    expr.Right:        lambda n, pn: (n.ab.value.right, n.ab),
    expr.Absurd:       lambda n, pn: (VoidType(), n.cod),
    expr.MonadicEmbed: lambda n, pn: (dom_of(n.f, pn), n.monad.wrap(cod_of(n.f, pn))),
}

_SIG_BINARY: dict = {
    expr.Compose:  (_U, _U, lambda n, pn: (
        dom_of(n.f, pn), cod_of(n.g, pn))),
    expr.Parallel: (_PU, _PU, lambda n, pn: (
        ProductType(dom_of(n.f, pn), dom_of(n.g, pn)),
        ProductType(cod_of(n.f, pn), cod_of(n.g, pn)))),
    expr.Pair:     (_U, _PU, lambda n, pn: (
        dom_of(n.f, pn), ProductType(cod_of(n.f, pn), cod_of(n.g, pn)))),
    expr.Case:     (_SU, _U, lambda n, pn: (
        SumType(dom_of(n.f, pn), dom_of(n.g, pn)), cod_of(n.f, pn))),
}

_SIG_VALIDATED: dict = {
    expr.Assoc:            (_assoc_cod, "Assoc expects (A⋆B)⋆C -> A⋆(B⋆C)"),
    expr.Symmetry:         (_symmetry_cod, "Symmetry expects A⋆B -> B⋆A"),
    expr.DistributeLeft:   (_distl_cod, "DistributeLeft expects A×(B+C) → (A×B)+(A×C)"),
    expr.DistributeRight:  (_distr_cod, "DistributeRight expects (A+B)×C → (A×C)+(B×C)"),
}


def _resolve_binary(node, param_names, exp_dom, exp_cod, compute):
    if node.dom == exp_dom and node.cod == exp_cod:
        return compute(node, param_names)
    return node.dom, node.cod


def _resolve_validated(node, build_cod, message):
    expected = build_cod(node.dom)
    if expected is None or expected != node.cod:
        raise MorphismError(message)
    return node.dom, node.cod


def _resolve_ref(node, param_names):
    if node.name in param_names:
        tv = TypeVariable(Name(node.name))
        return tv, tv
    raise MorphismError(f"signature: unresolved reference {node.name!r}")


def signature(
    node: expr.MorphismExpr,
    param_names: frozenset[str] = frozenset(),
) -> tuple[Type, Type]:
    """Derive the object-level arrow signature for a morphism expression."""
    t = type(node)
    leaf = _SIG_LEAF.get(t)
    if leaf is not None:
        return leaf(node, param_names)
    binary = _SIG_BINARY.get(t)
    if binary is not None:
        return _resolve_binary(node, param_names, *binary)
    if isinstance(node, _SELF_DESCRIBING):
        return node.dom, node.cod
    validated = _SIG_VALIDATED.get(t)
    if validated is not None:
        return _resolve_validated(node, *validated)
    if isinstance(node, expr.Ref):
        return _resolve_ref(node, param_names)
    raise TypeError(f"signature: unknown MorphismExpr {t.__name__!r}")


def dom_of(node: expr.MorphismExpr, param_names: frozenset[str] = frozenset()) -> Type:
    """Return the raw domain of a morphism expression."""
    return signature(node, param_names)[0]


def cod_of(node: expr.MorphismExpr, param_names: frozenset[str] = frozenset()) -> Type:
    """Return the raw codomain of a morphism expression."""
    return signature(node, param_names)[1]


# ---------------------------------------------------------------------------
# Plain constructors
# ---------------------------------------------------------------------------

def identity(space: Type) -> Morphism:
    """Identity morphism ``space -> space``."""
    return Morphism(node=expr.Identity(space))


def _copy(space: Type) -> Morphism:
    """Diagonal morphism ``space -> space × space``."""
    return Morphism(node=expr.Copy(space))


def _delete(space: Type) -> Morphism:
    """Terminal morphism ``space -> Unit``."""
    return Morphism(node=expr.Delete(space))


def _fst(ab: TypePair) -> Morphism:
    """Left projection ``A × B -> A``."""
    return Morphism(node=expr.First(ab))


def _snd(ab: TypePair) -> Morphism:
    """Right projection ``A × B -> B``."""
    return Morphism(node=expr.Second(ab))


def _inl(ab: TypeEither) -> Morphism:
    """Left injection ``A -> A + B``."""
    return Morphism(node=expr.Left(ab))


def _inr(ab: TypeEither) -> Morphism:
    """Right injection ``B -> A + B``."""
    return Morphism(node=expr.Right(ab))


def absurd(cod: Type) -> Morphism:
    """Unique morphism from the initial object, ``Void -> cod``."""
    return Morphism(node=expr.Absurd(cod))


def _assoc(dom: TypePair | TypeEither) -> Morphism:
    return Morphism(node=expr.Assoc(dom, _assoc_cod(dom)))


def _symmetry(dom: TypePair | TypeEither) -> Morphism:
    return Morphism(node=expr.Symmetry(dom, _symmetry_cod(dom)))


def distribute_left(a: Type, b: Type, c: Type) -> Morphism:
    """A × (B + C) → (A × B) + (A × C)."""
    dom = ProductType(a, SumType(b, c))
    return Morphism(node=expr.DistributeLeft(dom, _distl_cod(dom)))


def distribute_right(a: Type, b: Type, c: Type) -> Morphism:
    """(A + B) × C → (A × C) + (B × C)."""
    dom = ProductType(SumType(a, b), c)
    return Morphism(node=expr.DistributeRight(dom, _distr_cod(dom)))



# ---------------------------------------------------------------------------
# Monad resolution
# ---------------------------------------------------------------------------

def _resolve_monad(*morphisms: Morphism) -> Monad | None:
    """Return the unique non-None monad among morphisms, or reject conflicts."""
    monads = {m.monad for m in morphisms if m.monad is not None}
    if len(monads) > 1:
        raise MorphismError(f"conflicting monads: {monads!r}")
    return monads.pop() if monads else None


def raw_signature(param: Type, monad: Monad | None, dom: Type, cod: Type) -> tuple[Type, Type]:
    """Return the raw term signature for a visible morphism signature."""
    raw_dom = dom if param == TypeUnit() else ProductType(param, dom)
    raw_cod = cod if monad is None else monad.wrap(cod)
    return raw_dom, raw_cod


_BINARY_SIG: dict = {
    expr.Compose:  lambda f, g: (f.dom(), g.cod()),
    expr.Parallel: lambda f, g: (ProductType(f.dom(), g.dom()), ProductType(f.cod(), g.cod())),
    expr.Pair:     lambda f, g: (f.dom(), ProductType(f.cod(), g.cod())),
    expr.Case:     lambda f, g: (SumType(f.dom(), g.dom()), f.cod()),
}


def _contextual_binary(
    cls, f: Morphism, g: Morphism, *,
    dom_override: Type | None = None,
    shared_context: bool = False,
    graph=None,
    allow_unification: bool = False,
) -> Morphism:
    dom, cod = _BINARY_SIG[cls](f, g)
    if dom_override is not None:
        dom = dom_override
    monad = _resolve_monad(f, g)
    param = (
        _share_param(
            f.param, g.param,
            graph=graph,
            allow_unification=allow_unification,
        )
        if shared_context else
        Ty.combine_params(f.param, g.param)
    )
    raw_dom, raw_cod = raw_signature(param, monad, dom, cod)
    return Morphism(
        node=cls(
            f.node_in(monad), g.node_in(monad),
            f.param, g.param, param, monad, raw_dom, raw_cod,
        ),
        param=param,
        monad=monad,
        aux_primitives=_collect_aux_primitives(f, g),
    )


# ---------------------------------------------------------------------------
# Plain combinators
# ---------------------------------------------------------------------------

def compose(f: Morphism, g: Morphism, *, shared_context: bool = False,
            graph=None, allow_unification: bool = False) -> Morphism:
    """Compose ``f`` then ``g`` in diagrammatic order.

    Requires ``f.cod() == g.dom()``.  Plain morphisms are automatically embedded
    into a shared lax context when composed with an effectful morphism.

    By default, two non-unit params are independent and combine as
    ``g.param × f.param``.  With ``shared_context=True``, matching non-unit
    params are shared instead; distinct non-unit params are rejected.
    """
    dom_override = None
    try:
        if allow_unification and f.cod() != g.dom():
            match = Ty.unify(f.cod(), g.dom(), "Cannot compose morphisms")
            dom_override = Ty.apply_subst(match.substitution, f.dom())
        else:
            Ty.require_equal(graph, f.cod(), g.dom(), "Cannot compose morphisms")
    except TypeError as e:
        raise MorphismError(str(e)) from e
    return _contextual_binary(
        expr.Compose, f, g,
        dom_override=dom_override,
        shared_context=shared_context,
        graph=graph,
        allow_unification=allow_unification,
    )


def par(f: Morphism, g: Morphism, *, shared_context: bool = False) -> Morphism:
    """Parallel product ``f × g : A × C -> B × D``.

    Works uniformly for plain, parametric, and lax morphisms after resolving a
    parameter and monad context.
    """
    return _contextual_binary(
        expr.Parallel, f, g,
        shared_context=shared_context,
    )


def _validated_binary(
    cls, f: Morphism, g: Morphism,
    left: Type, right: Type, message: str, **kw,
) -> Morphism:
    try:
        Ty.unify_or_equal(
            kw.get("graph"), left, right, message,
            allow_unification=kw.get("allow_unification", False),
        )
    except TypeError as e:
        raise MorphismError(str(e)) from e
    return _contextual_binary(cls, f, g, **kw)


def pair(f: Morphism, g: Morphism, **kw) -> Morphism:
    return _validated_binary(expr.Pair, f, g, f.dom(), g.dom(), "Cannot build pair", **kw)


def merge(a: Type) -> Morphism:
    return case(identity(a), identity(a))


def case(f: Morphism, g: Morphism, **kw) -> Morphism:
    return _validated_binary(expr.Case, f, g, f.cod(), g.cod(), "Cannot build case", **kw)
