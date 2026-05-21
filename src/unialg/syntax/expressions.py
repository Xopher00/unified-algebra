"""Pure syntax nodes for the unialg DSL.

This module owns expression structure only.  It does not type-check morphism
composition, lower terms, or execute Hydra code.  The two syntax families are:

* ``MorphismExpr`` — arrow expressions such as identity, product/sum structure,
  contextual combinators, and raw primitive escape hatches.
* ``PolyExpr`` — polynomial functor expressions used as shape descriptors.

Semantic interpretation lives in ``semantics/morphisms.py`` and
``semantics/functors.py``; term realization lives in ``structure/realize.py``.
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
class DistributeLeft(MorphismExpr):
    """distl : A × (B + C) → (A × B) + (A × C)."""
    dom: Type
    cod: Type


@dataclass(frozen=True)
class DistributeRight(MorphismExpr):
    """distr : (A + B) × C → (A × C) + (B × C)."""
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
class SharedCompose(Compose):
    """f >>>> g — shared-context sequential composition."""


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


@dataclass(frozen=True)
class DomainPrim(MorphismExpr):
    """Opaque domain-owned primitive. Must be rewritten before realize.

    Carries a domain-specific payload (e.g. ContractSpec for the tensors domain).
    The domain's ``finalize`` hook replaces every DomainPrim with a substrate
    Morphism tree before ``realize`` is called.
    """
    tag: str    # which domain owns this (e.g. "tensors")
    raw: object # domain-specific payload
    dom: Type
    cod: Type


@dataclass(frozen=True)
class BackendPrim(MorphismExpr):
    """Backend primitive: type info + Hydra Primitive, term built at realization.

    When ``args`` is empty: arity-1 leaf, realize builds primitive_wrapper_term.
    When ``args`` is populated: realize builds term from resolved arg morphisms.
    """
    primitive: object
    arity: int
    dom: Type
    cod: Type
    args: tuple[MorphismExpr, ...] = ()


@dataclass(frozen=True)
class Ref(MorphismExpr):
    """Unresolved morphism name. Resolved by the semantic construction pass."""
    name: str


@dataclass(frozen=True)
class MorphismApp(MorphismExpr):
    """Parametric application: fun(arg1, arg2, ...)."""
    fun: MorphismExpr
    args: tuple[MorphismExpr, ...]


@dataclass(frozen=True)
class RecursionApp(MorphismExpr):
    """Recursive scheme application: cata[focus](...), ana[focus](...), hylo[focus](...)."""
    kind: str
    focus: str
    args: tuple[MorphismExpr, ...]


@dataclass(frozen=True)
class CarrierBoundary(MorphismExpr):
    """Recursive carrier boundary: ``roll[focus]`` or ``unroll[focus]``."""
    kind: str
    focus: str


@dataclass(frozen=True)
class MonadicLift(MorphismExpr):
    """Lift a morphism into a built-in monad: ``pure[Maybe](f)``."""
    monad: str
    body: MorphismExpr


@dataclass(frozen=True)
class FocusDecl:
    """Surface optic declaration.

    A focus can either reference a recursive carrier, or explicitly provide a
    functor with forward/backward boundary morphisms.
    """
    carrier: str | None = None
    functor: str | None = None
    forward: MorphismExpr | None = None
    backward: MorphismExpr | None = None
    expr: FocusExpr | None = None


@dataclass(frozen=True)
class CarrierDecl:
    """Surface recursive carrier declaration: ``shape Nat = fix F``."""
    functor: PolyExpr


@dataclass(frozen=True)
class FocusExpr:
    """Base class for focus/optic expressions."""


@dataclass(frozen=True)
class FocusRef(FocusExpr):
    """Unresolved focus name. Resolved by semantic construction."""
    name: str


@dataclass(frozen=True)
class FocusCompose(FocusExpr):
    """Focus composition: first ``left``, then ``right``."""
    left: FocusExpr
    right: FocusExpr


@singledispatch
def pretty(expr) -> str:
    """Render a DSL expression for humans.

    This is intentionally lightweight display text, not a parser-stable
    serialization format.  It raises ``ValueError`` for base classes or unknown
    expression objects.
    """
    raise ValueError(f"pretty: unknown type {type(expr).__name__!r}")


_MORPHISM_LEAVES: dict[type, str] = {
    Identity: "id",
    Copy: "copy",
    Delete: "!",
    First: "π₁",
    Second: "π₂",
    Left: "ι₁",
    Right: "ι₂",
    Absurd: "absurd",
    Assoc: "assoc",
    Symmetry: "sym",
    DistributeLeft: "distl",
    DistributeRight: "distr",
    Prim: "prim",
}

_MORPHISM_BINARY = (Compose, Parallel, Pair, Case)


def _pretty_binary(expr: MorphismExpr) -> str:
    match expr:
        case Compose(f=f, g=g):
            return f"({pretty(f)} ; {pretty(g)})"
        case Parallel(f=f, g=g):
            return f"({pretty(f)} × {pretty(g)})"
        case Pair(f=f, g=g):
            return f"⟨{pretty(f)}, {pretty(g)}⟩"
        case Case(f=f, g=g):
            return f"[{pretty(f)}, {pretty(g)}]"
        case _:
            raise ValueError(f"pretty: unexpected binary {type(expr).__name__!r}")


def _pretty_wrapped(expr: MorphismExpr) -> str | None:
    match expr:
        case MonadicEmbed(f=f):
            return f"η({pretty(f)})"
        case PolyFmap(body=body):
            return f"F({pretty(body)})"
        case SelfRef(node=node):
            return f"self({node.name[-6:]})"
        case Cata(node=node):
            return f"cata({pretty(node.body)})"
        case Ana(node=node):
            return f"ana({pretty(node.body)})"
        case MonadicLift(monad=monad, body=body):
            return f"pure[{monad}]({pretty(body)})"
        case _:
            return None


def _pretty_named(expr: MorphismExpr) -> str:
    match expr:
        case DomainPrim(tag=tag):
            return f"domain_prim[{tag}]"
        case MorphismApp(fun=fun, args=args):
            return f"{pretty(fun)}({', '.join(pretty(a) for a in args)})"
        case RecursionApp(kind=kind, focus=focus, args=args):
            return f"{kind}[{focus}]({', '.join(pretty(a) for a in args)})"
        case CarrierBoundary(kind=kind, focus=focus):
            return f"{kind}[{focus}]"
        case _:
            raise ValueError(f"pretty: unknown MorphismExpr {type(expr).__name__!r}")


@pretty.register(MorphismExpr)
def _pretty_morphism(expr: MorphismExpr) -> str:
    token = _MORPHISM_LEAVES.get(type(expr))
    if token is not None:
        return token
    if isinstance(expr, _MORPHISM_BINARY):
        return _pretty_binary(expr)
    result = _pretty_wrapped(expr)
    if result is not None:
        return result
    return _pretty_named(expr)



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
class PolyCompose(PolyExpr):
    """F ∘ G in diagrammatic syntax: first ``left``, then ``right``."""
    left: PolyExpr
    right: PolyExpr


@dataclass(frozen=True)
class Exp(PolyExpr):
    """F(X) = S → G(X) — exponential functor; base is a PolyExpr."""
    base: PolyExpr
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
    """Unresolved functor name. Resolved by the semantic construction pass."""
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


def _pretty_poly_atom(expr: PolyExpr) -> str | None:
    match expr:
        case Zero():
            return "0"
        case One():
            return "1"
        case Id():
            return "X"
        case Const(space=space):
            return show_type(space)
        case _:
            return None


def _pretty_poly_compound(expr: PolyExpr) -> str:
    match expr:
        case Sum(left=left, right=right):
            return f"{pretty(left)} + {pretty(right)}"
        case Prod(left=left, right=right):
            ls = pretty(left)
            rs = pretty(right)
            if isinstance(left, Sum):
                ls = f"({ls})"
            if isinstance(right, Sum):
                rs = f"({rs})"
            return f"{ls} * {rs}"
        case PolyCompose(left=left, right=right):
            return f"{pretty(left)} >> {pretty(right)}"
        case Exp(base=base, body=body):
            bs = pretty(body)
            if isinstance(body, (Sum, Prod)):
                bs = f"({bs})"
            return f"{pretty(base)} -> {bs}"
        case List(body=body):
            return f"List[{pretty(body)}]"
        case Maybe(body=body):
            return f"Maybe[{pretty(body)}]"
        case _:
            raise ValueError(f"pretty: unknown PolyExpr {type(expr).__name__!r}")


@pretty.register(PolyExpr)
def _pretty_poly(expr: PolyExpr) -> str:
    result = _pretty_poly_atom(expr)
    if result is not None:
        return result
    return _pretty_poly_compound(expr)
