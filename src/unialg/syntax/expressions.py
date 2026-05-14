"""Pure syntax nodes for the unialg DSL.

This module owns expression structure only.  It does not type-check morphism
composition, lower terms, or execute Hydra code.  The two syntax families are:

* ``MorphismExpr`` — arrow expressions such as identity, product/sum structure,
  contextual combinators, and raw primitive escape hatches.
* ``PolyExpr`` — polynomial functor expressions used as shape descriptors.

Semantic interpretation lives in ``morphisms.py`` and ``functors.py``; backend
realization lives in ``realize.py``.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import singledispatch

from unialg.objects import Monad, Type, TypeEither, TypePair, show_type


@dataclass(frozen=True)
class BackendExor:
    """"User defined backend syntax"""
    space: Type


# ---------------------------------------------------------------------------
# MorphismExpr ADT
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MorphismExpr:
    """Base class for morphism syntax.

    Concrete subclasses are the only valid semantic nodes.  The base class is
    useful for typing and dispatch, but morphisms such as ``pretty`` and
    ``signature`` reject a bare ``MorphismExpr``.
    """


@dataclass(frozen=True)
class Identity(MorphismExpr):
    """id_A : A → A."""
    space: Type


@dataclass(frozen=True)
class Copy(MorphismExpr):
    """Δ_A : A → A × A — diagonal / comonoid copy."""
    space: Type


@dataclass(frozen=True)
class Delete(MorphismExpr):
    """!_A : A → 1 — comonoid delete."""
    space: Type


@dataclass(frozen=True)
class First(MorphismExpr):
    """π₁ : A × B → A — left projection."""
    ab: TypePair


@dataclass(frozen=True)
class Second(MorphismExpr):
    """π₂ : A × B → B — right projection."""
    ab: TypePair


@dataclass(frozen=True)
class Left(MorphismExpr):
    """ι₁ : A → A + B — left injection."""
    ab: TypeEither


@dataclass(frozen=True)
class Right(MorphismExpr):
    """ι₂ : B → A + B — right injection."""
    ab: TypeEither


@dataclass(frozen=True)
class Absurd(MorphismExpr):
    """absurd : 0 → A — unique morphism from the initial object."""
    cod: Type


@dataclass(frozen=True)
class Assoc(MorphismExpr):
    """assoc : (A ⋆ B) ⋆ C → A ⋆ (B ⋆ C), for product or sum."""
    dom: Type
    cod: Type


@dataclass(frozen=True)
class Symmetry(MorphismExpr):
    """sym : A ⋆ B → B ⋆ A, for product or sum."""
    dom: Type
    cod: Type


@dataclass(frozen=True)
class MonadicEmbed(MorphismExpr):
    """pure ∘ f — lift a plain morphism into a monad."""
    f: MorphismExpr
    monad: Monad


@dataclass(frozen=True)
class ContextualBinary(MorphismExpr):
    """Shared payload for binary combinators after type checking.

    ``f`` and ``g`` are already adapted to the resolved monad if needed.
    ``f_param`` and ``g_param`` record each child parameter.  ``param`` is the
    combined parameter expected by this node's raw domain.  ``dom`` and ``cod``
    are stored because contextual composition has already resolved the visible
    source and target in ``morphisms.py``.
    """
    f: MorphismExpr
    g: MorphismExpr
    f_param: Type
    g_param: Type
    param: Type
    monad: Monad | None
    dom: Type
    cod: Type


@dataclass(frozen=True)
class Compose(ContextualBinary):
    """f ; g — sequential composition (diagrammatic order)."""


@dataclass(frozen=True)
class Parallel(ContextualBinary):
    """f × g — parallel composition (A×C → B×D)."""


@dataclass(frozen=True)
class Pair(ContextualBinary):
    """⟨f, g⟩ : A → B × C — product introduction."""


@dataclass(frozen=True)
class Case(ContextualBinary):
    """[f, g] : A + B → C — coproduct elimination."""


@dataclass(frozen=True)
class Ref(MorphismExpr):
    """Unresolved name reference — resolved from env during parsing."""
    name: str


@dataclass(frozen=True)
class Prim(MorphismExpr):
    """Already-built Hydra term with explicit domain and codomain.

    Raw Hydra terms do not carry enough type information for this DSL, so a
    primitive node must provide ``dom`` and ``cod``.  Use this as the backend
    boundary for hand-written Hydra primitives; prefer structured nodes when a
    morphism can be expressed by the DSL itself.
    """
    raw: object
    dom: Type
    cod: Type


@singledispatch
def pretty(expr) -> str:
    """Render a DSL expression for humans.

    This is intentionally lightweight display text, not a parser-stable
    serialization format.  It raises ``ValueError`` for base classes or unknown
    expression objects.
    """
    raise ValueError(f"pretty: unknown type {type(expr).__name__!r}")


@pretty.register(MorphismExpr)
def _pretty_morphism(expr: MorphismExpr) -> str:
    """Render a morphism expression node for diagnostics and notebooks."""
    match expr:
        case Identity():
            return "id"
        case Copy():
            return "copy"
        case Delete():
            return "!"
        case First():
            return "π₁"
        case Second():
            return "π₂"
        case Left():
            return "ι₁"
        case Right():
            return "ι₂"
        case Absurd():
            return "absurd"
        case Assoc():
            return "assoc"
        case Symmetry():
            return "sym"
        case MonadicEmbed(f=f):
            return f"η({pretty(f)})"
        case Compose(f=f, g=g):
            return f"({pretty(f)} ; {pretty(g)})"
        case Parallel(f=f, g=g):
            return f"({pretty(f)} × {pretty(g)})"
        case Pair(f=f, g=g):
            return f"⟨{pretty(f)}, {pretty(g)}⟩"
        case Case(f=f, g=g):
            return f"[{pretty(f)}, {pretty(g)}]"
        case PolyFmap(body=body):
            return f"F({pretty(body)})"
        case SelfRef(node=node):
            return f"self({node.name[-6:]})"
        case Cata(node=node):
            return f"cata({pretty(node.body)})"
        case Ana(node=node):
            return f"ana({pretty(node.body)})"
        case Ref(name=name):
            return name
        case Prim():
            return "prim"
        case _:
            raise ValueError(f"pretty: unknown MorphismExpr {type(expr).__name__!r}")



# ---------------------------------------------------------------------------
# PolyExpr ADT
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolyExpr:
    """Base class for polynomial functor syntax.

    Concrete subclasses describe one layer of shape.  They are not concrete
    spaces by themselves; ``functors.apply_poly`` substitutes a Hydra
    ``Type`` for ``Id`` to compute an actual object.
    """


@dataclass(frozen=True)
class Zero(PolyExpr):
    """F(X) = 0 — initial functor."""


@dataclass(frozen=True)
class One(PolyExpr):
    """F(X) = 1 — constant terminal functor."""


@dataclass(frozen=True)
class Id(PolyExpr):
    """F(X) = X — the recursion variable. Not a concrete space."""


@dataclass(frozen=True)
class Const(PolyExpr):
    """F(X) = S — constant functor at space S."""
    space: Type


@dataclass(frozen=True)
class Sum(PolyExpr):
    """F(X) = G(X) + H(X) — coproduct of functors."""
    left: PolyExpr
    right: PolyExpr


@dataclass(frozen=True)
class Prod(PolyExpr):
    """F(X) = G(X) × H(X) — product of functors."""
    left: PolyExpr
    right: PolyExpr


@dataclass(frozen=True)
class Exp(PolyExpr):
    """F(X) = S → G(X) — exponential with constant base S."""
    base: Type
    body: PolyExpr


@dataclass(frozen=True)
class List(PolyExpr):
    """F(X) = List[G(X)] — list type constructor lifted over a polynomial."""
    body: PolyExpr


@dataclass(frozen=True)
class Maybe(PolyExpr):
    """F(X) = Maybe[G(X)] — maybe type constructor lifted over a polynomial."""
    body: PolyExpr


@dataclass(frozen=True)
class PolyRef(PolyExpr):
    """Unresolved name reference in functor position — resolved from env during parsing."""
    name: str


@dataclass(frozen=True)
class PolyFmap(MorphismExpr):
    """Semantic polynomial functor action F(f).

    This is syntax, not lowering. It records that a morphism expression should
    be mapped through a polynomial body. Backend Hydra construction happens in
    realize.py.
    """

    body: PolyExpr
    f: MorphismExpr
    param: Type
    monad: Monad | None
    dom: Type
    cod: Type


@dataclass(frozen=True)
class SelfRef(MorphismExpr):
    """Abstract self-reference in a fixpoint equation (no Hydra import needed)."""
    name: str
    dom: Type
    cod: Type


@dataclass(frozen=True)
class AlgExpr(MorphismExpr):
    """Base node for deferred recursive scheme expressions.

    ``name`` is a globally unique identifier used as the Hydra primitive name.
    ``body`` contains the algebra/coalgebra equation with embedded ``SelfRef``
    nodes.  ``dom`` and ``cod`` are the raw term types (including any parameter
    prefix and monad wrapper).  Subclasses ``Cata`` and ``Ana`` are realized by
    ``structure/realize.py`` as mutually recursive Hydra ``Primitive`` objects.
    """
    name: str
    body: MorphismExpr
    dom: Type
    cod: Type


@dataclass(frozen=True)
class Cata(AlgExpr):
    """Deferred catamorphism node — realized by structure/realize.py."""


@dataclass(frozen=True)
class Ana(AlgExpr):
    """Deferred anamorphism node — realized by structure/realize.py."""


@pretty.register(PolyExpr)
def _pretty_poly(expr: PolyExpr) -> str:
    """Render a polynomial functor expression for diagnostics and notebooks."""
    if isinstance(expr, Zero):
        return "0"
    if isinstance(expr, One):
        return "1"
    if isinstance(expr, Id):
        return "X"
    if isinstance(expr, Const):
        return show_type(expr.space)
    if isinstance(expr, Sum):
        return f"{pretty(expr.left)} + {pretty(expr.right)}"
    if isinstance(expr, Prod):
        ls = pretty(expr.left)
        rs = pretty(expr.right)
        if isinstance(expr.left, Sum):
            ls = f"({ls})"
        if isinstance(expr.right, Sum):
            rs = f"({rs})"
        return f"{ls} * {rs}"
    if isinstance(expr, Exp):
        bs = pretty(expr.body)
        if isinstance(expr.body, (Sum, Prod)):
            bs = f"({bs})"
        return f"{show_type(expr.base)} -> {bs}"
    if isinstance(expr, List):
        return f"List[{pretty(expr.body)}]"
    if isinstance(expr, Maybe):
        return f"Maybe[{pretty(expr.body)}]"
    if isinstance(expr, PolyRef):
        return expr.name
    raise ValueError(f"pretty: unknown PolyExpr {type(expr).__name__!r}")
