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


def _share_param(f_param: Type, g_param: Type, *, graph=None, allow_unification: bool = False) -> Type:
    try:
        return Ty.share_param(
            graph, f_param, g_param,
            "Cannot share contextual parameter",
            allow_unification=allow_unification,
        )
    except TypeError as e:
        raise MorphismError(str(e)) from e


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
    

def _contextual_binary(
    cls,
    f: Morphism,
    g: Morphism,
    dom: Type,
    cod: Type,
    *,
    shared_context: bool = False,
    graph=None,
    allow_unification: bool = False,
) -> Morphism:
    """Construct a binary contextual morphism node, resolving monad and parameter context.

    Wraps ``f`` and ``g`` into a ``cls`` node (one of ``Compose``, ``Parallel``,
    ``Pair``, ``Case``) with the resolved combined domain and codomain.  Plain
    morphisms are automatically embedded into a shared lax context.
    """
    monad = _resolve_monad(f, g)
    param = (
        _share_param(
            f.param,
            g.param,
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
    return (t.value.first, t.value.second) if isinstance(t, TypePair) else (t.value.left, t.value.right)


def _is_assoc(dom: Type, cod: Type, cls: type) -> bool:
    """Return True if (dom, cod) has the shape ``(A⋆B)⋆C → A⋆(B⋆C)`` for ``cls``."""
    if not (isinstance(dom, cls) and isinstance(cod, cls)):
        return False
    dl, dr = _lr(dom)
    cl, cr = _lr(cod)
    if not (isinstance(dl, cls) and isinstance(cr, cls)):
        return False
    dll, dlr = _lr(dl)
    crl, crr = _lr(cr)
    return dll == cl and dlr == crl and dr == crr


def _is_symmetry(dom: Type, cod: Type, cls: type) -> bool:
    """Return True if (dom, cod) has the shape ``A⋆B → B⋆A`` for ``cls``."""
    if not (isinstance(dom, cls) and isinstance(cod, cls)):
        return False
    dl, dr = _lr(dom)
    cl, cr = _lr(cod)
    return dl == cr and dr == cl


# ---------------------------------------------------------------------------
# Type derivation
# ---------------------------------------------------------------------------

def signature(
    node: expr.MorphismExpr,
    param_names: frozenset[str] = frozenset(),
) -> tuple[Type, Type]:
    """Derive the object-level arrow signature for a morphism expression.

    Contextual nodes and primitives are self-describing because their type
    checks occurred at construction time.  Placeholder-typed nodes (from the
    parser) derive types from children.  Ref nodes for declared parameters
    become type variables.
    """
    match node:
        case expr.Identity(space):
            return space, space
        case expr.Copy(space):
            return space, ProductType(space, space)
        case expr.Delete(space):
            return space, TypeUnit()
        case expr.First(ab):
            return ab, ab.value.first
        case expr.Second(ab):
            return ab, ab.value.second
        case expr.Left(ab):
            return ab.value.left, ab
        case expr.Right(ab):
            return ab.value.right, ab
        case expr.Absurd(cod):
            return VoidType(), cod
        case expr.Assoc(dom=dom, cod=cod):
            if not (_is_assoc(dom, cod, TypePair) or _is_assoc(dom, cod, TypeEither)):
                raise MorphismError("Assoc expects (A⋆B)⋆C -> A⋆(B⋆C)")
            return dom, cod
        case expr.Symmetry(dom=dom, cod=cod):
            if not (_is_symmetry(dom, cod, TypePair) or _is_symmetry(dom, cod, TypeEither)):
                raise MorphismError("Symmetry expects A⋆B -> B⋆A")
            return dom, cod
        case expr.MonadicEmbed(f=f, monad=monad):
            return dom_of(f, param_names), monad.wrap(cod_of(f, param_names))
        case expr.Compose(f=f, g=g, dom=dom, cod=cod) if dom == TypeUnit() and cod == TypeUnit():
            return dom_of(f, param_names), cod_of(g, param_names)
        case expr.Parallel(f=f, g=g, dom=dom, cod=cod) if dom == ProductType(TypeUnit(), TypeUnit()) and cod == ProductType(TypeUnit(), TypeUnit()):
            return ProductType(dom_of(f, param_names), dom_of(g, param_names)), ProductType(cod_of(f, param_names), cod_of(g, param_names))
        case expr.Pair(f=f, g=g, dom=dom, cod=cod) if dom == TypeUnit() and cod == ProductType(TypeUnit(), TypeUnit()):
            return dom_of(f, param_names), ProductType(cod_of(f, param_names), cod_of(g, param_names))
        case expr.Case(f=f, g=g, dom=dom, cod=cod) if dom == SumType(TypeUnit(), TypeUnit()) and cod == TypeUnit():
            return SumType(dom_of(f, param_names), dom_of(g, param_names)), cod_of(f, param_names)
        case expr.ContextualBinary(dom=dom, cod=cod):
            return dom, cod
        case expr.Prim(_, dom, cod):
            return dom, cod
        case expr.BackendPrim(_, _, dom, cod):
            return dom, cod
        case expr.PolyFmap(dom=dom, cod=cod):
            return dom, cod
        case expr.SelfRef(dom=dom, cod=cod):
            return dom, cod
        case expr.AlgExpr(dom=dom, cod=cod):
            return dom, cod
        case expr.Ref(name=name):
            if name in param_names:
                tv = TypeVariable(Name(name))
                return tv, tv
            raise MorphismError(f"signature: unresolved reference {name!r}")
        case _:
            raise TypeError(f"signature: unknown MorphismExpr {type(node).__name__!r}")


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
    """Reassociation ``(A ⋆ B) ⋆ C → A ⋆ (B ⋆ C)``, for product or sum."""
    dl, c = _lr(dom)
    a, b = _lr(dl)
    make = ProductType if isinstance(dom, TypePair) else SumType
    return Morphism(node=expr.Assoc(dom, make(a, make(b, c))))


def _symmetry(dom: TypePair | TypeEither) -> Morphism:
    """Swap ``A ⋆ B → B ⋆ A``, for product or sum."""
    l, r = _lr(dom)
    make = ProductType if isinstance(dom, TypePair) else SumType
    return Morphism(node=expr.Symmetry(dom, make(r, l)))


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
    try:
        Ty.unify_or_equal(
            graph, f.cod(), g.dom(),
            "Cannot compose morphisms",
            allow_unification=allow_unification,
        )
    except TypeError as e:
        raise MorphismError(str(e)) from e
    return _contextual_binary(
        expr.Compose,
        f,
        g,
        f.dom(),
        g.cod(),
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
        ProductType(f.dom(), g.dom()),
        ProductType(f.cod(), g.cod()),
        shared_context=shared_context,
    )


def pair(f: Morphism, g: Morphism, *, shared_context: bool = False, graph=None, allow_unification: bool = False) -> Morphism:
    """Product introduction ``<f, g> : A -> B × C``.

    Requires both morphisms to have the same visible domain.
    """
    try:
        Ty.unify_or_equal(
            graph, f.dom(), g.dom(),
            "Cannot build pair",
            allow_unification=allow_unification,
        )
    except TypeError as e:
        raise MorphismError(str(e)) from e
    return _contextual_binary(
        expr.Pair,
        f,
        g,
        f.dom(),
        ProductType(f.cod(), g.cod()),
        shared_context=shared_context,
        graph=graph,
        allow_unification=allow_unification,
    )


def case(f: Morphism, g: Morphism, *, shared_context: bool = False, graph=None, allow_unification: bool = False) -> Morphism:
    """Coproduct elimination ``[f, g] : A + B -> C``.

    Requires both branches to have the same visible codomain.
    """
    try:
        Ty.unify_or_equal(
            graph, f.cod(), g.cod(),
            "Cannot build case",
            allow_unification=allow_unification,
        )
    except TypeError as e:
        raise MorphismError(str(e)) from e

    return _contextual_binary(
        expr.Case,
        f,
        g,
        SumType(f.dom(), g.dom()),
        f.cod(),
        shared_context=shared_context,
        graph=graph,
        allow_unification=allow_unification,
    )

