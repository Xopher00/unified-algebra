"""Optics — polynomial functor optics built from Morphism.

Every optic is a triple ``(functor, forward, backward)`` where:

    forward  : S → F(A)     decompose source into F-shaped container
    backward : F(B) → T     reconstruct target from F-shaped container

The action of an optic on a morphism ``h : A → B`` is always:

    act(optic, h) = compose(forward, poly_fmap(F, h), backward)  :  S → T

Lens, Prism, and Traversal are specific functor choices:

    Lens   — F = Prod(Id, Const(residue))   product focus
    Prism  — F = Sum(Id, Const(residue))    sum focus
    Traversal — arbitrary polynomial F      multi-element focus

No backend encoding logic.  The action methods return semantic ``Morphism``
values; ``structure/realize.py`` lowers deferred recursion nodes.  Recursive
carriers use ``Coerce`` expression nodes for their roll/unroll boundary
morphisms.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from hydra.core import Name, TypeVariable

from . import typeops as Ty
from .functors import Functor
from .morphisms import (
    Morphism, MorphismError, compose, identity, raw_signature,
    par as _par_morphisms, copar as _copar_morphisms,
)
from unialg.syntax import expressions as expr
from unialg.objects import Type, ProductType, SumType


def _infer_layer_arg(functor: Functor, fa: Type, label: str) -> Type:
    """Recover the layer argument ``X`` from ``F(X)``, asserting round-trip equality."""
    try:
        arg = functor.unapply(fa)
        Ty.roundtrip_equal(None, functor.apply, arg, fa, label)
    except TypeError as e:
        raise MorphismError(f"invalid optic {label}: {e}") from e
    return arg


@dataclass(frozen=True)
class Optic:
    """Polynomial functor optic over ``(S, A, B, T)``.

    ``functor`` describes the container shape F.
    ``forward : S → F(A)`` decomposes the source.
    ``backward : F(B) → T`` reconstructs the target.

    Focus ``A`` and replacement ``B`` are derived via ``functor.unapply``
    on the forward codomain and backward domain respectively.
    """
    functor: Functor
    forward: Morphism
    backward: Morphism
    carrier: Type | None = None
    kind: str = "optic"
    _focus: Type = field(init=False, repr=False, compare=False)
    _replacement: Type = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_focus",
            _infer_layer_arg(self.functor, self.forward.cod(), "forward codomain"))
        object.__setattr__(self, "_replacement",
            _infer_layer_arg(self.functor, self.backward.dom(), "backward domain"))

    @property
    def source(self) -> Type:
        """S — the type being focused into."""
        return self.forward.dom()

    @property
    def target(self) -> Type:
        """T — the type produced after replacement."""
        return self.backward.cod()

    @property
    def focus(self) -> Type:
        """A — extracted from forward.cod() via functor.unapply."""
        return self._focus

    @property
    def replacement(self) -> Type:
        """B — extracted from backward.dom() via functor.unapply."""
        return self._replacement

    def act_forward(self, h: Morphism) -> Morphism:
        """Decompose through an optic, then lift ``h`` through the optic functor."""
        return compose(self.forward, self.functor.map(h))

    def act_backward(self, h: Morphism) -> Morphism:
        """Lift ``h`` through the optic functor, then reconstruct through the optic."""
        return compose(self.functor.map(h), self.backward)

    def act(self, h: Morphism) -> Morphism:
        """Apply an optic action to ``h``.

        Composition: ``S --forward--> F(A) --functor.map(F,h)--> F(B) --backward--> T``.
        If ``h`` is lax, plain optic boundaries are lifted into the same monad by the
        morphism composition rules.
        """
        return compose(self.act_forward(h), self.backward)

    def compose(self, inner: "Optic") -> "Optic":
        """Compose two optics: focus through ``outer`` then ``inner``."""
        return _compose_optic(self, inner)

    def product(self, other: "Optic") -> "Optic":
        """Same-focus product: functor becomes F(X) × G(X) for shared focus X.

        Both optics must have the same focus and replacement type.  Returns a
        unary ``Optic`` whose single ``act(h)`` applies ``h`` uniformly to the
        shared focus.
        """
        return _combine_optic(self, other, "product")

    def coproduct(self, other: "Optic") -> "Optic":
        """Same-focus coproduct: functor becomes F(X) + G(X) for shared focus X.

        Both optics must have the same focus and replacement type.  Returns a
        unary ``Optic`` whose single ``act(h)`` applies ``h`` uniformly to the
        shared focus.
        """
        return _combine_optic(self, other, "coproduct")

    def par(self, other: "Optic") -> "BinaryOptic":
        """Independent-focus product: F(A) × G(C), focuses may differ.

        Returns a ``ProductOptic`` (not a unary ``Optic``) because the combined
        functor ``(A, C) ↦ F(A) × G(C)`` is binary in its focus variables and
        cannot be encoded as a single polynomial container.  ``act(f, g)`` takes
        two separate morphisms, one per focus.
        """
        return ProductOptic(self, other)

    def choice(self, other: "Optic") -> "BinaryOptic":
        """Independent-focus coproduct: F(A) + G(C), focuses may differ.

        Dual of ``par``.  Returns a ``CoproductOptic`` whose ``act(f, g)`` acts
        on the left branch with ``f`` and the right branch with ``g``.
        """
        return CoproductOptic(self, other)


_COMBINE_KINDS: dict = {
    # kind        → (body_ctor,  morphism_combiner, name_sep)
    "product":    (expr.Prod, _par_morphisms,   "×"),
    "coproduct":  (expr.Sum,  _copar_morphisms, "+"),
    "par":        (expr.Prod, _par_morphisms,   "×"),
    "choice":     (expr.Sum,  _copar_morphisms, "+"),
}


def _combine_optic(left: Optic, right: Optic, kind: str) -> Optic:
    """Combine two same-focus optics by product or coproduct.

    Validates that both optics share the same focus and replacement type, then
    builds a new unary ``Optic`` with functor body ``F(X) ○ G(X)`` where ○ is
    × for "product" and + for "coproduct".
    """
    if left.focus != right.focus:
        raise MorphismError(
            f"optic combine focus: {left.focus!r} != {right.focus!r}"
        )
    if left.replacement != right.replacement:
        raise MorphismError(
            f"optic combine replacement: {left.replacement!r} != {right.replacement!r}"
        )
    mk_body, combine, sep = _COMBINE_KINDS[kind]
    functor = Functor(
        name=f"{left.functor.name}{sep}{right.functor.name}",
        body=mk_body(left.functor.body, right.functor.body),
    )
    carrier = left.carrier if left.carrier == right.carrier else None
    return Optic(
        functor=functor,
        forward=combine(left.forward, right.forward),
        backward=combine(left.backward, right.backward),
        carrier=carrier,
        kind=kind,
    )


@dataclass(frozen=True)
class BinaryOptic:
    """Independent-focus binary optic: product (``par``) or coproduct (``choice``).

    Given ``left : S1 → F(A1)`` and ``right : S2 → G(A2)``, the focuses
    ``A1`` and ``A2`` are independent — this is NOT a unary ``Optic``.
    ``act`` takes two separate morphisms, one per focus.
    """
    left: Optic
    right: Optic
    kind: str

    @property
    def forward(self) -> Morphism:
        _, combine, _ = _COMBINE_KINDS[self.kind]
        return combine(self.left.forward, self.right.forward)

    @property
    def backward(self) -> Morphism:
        _, combine, _ = _COMBINE_KINDS[self.kind]
        return combine(self.left.backward, self.right.backward)

    def act(self, left_h: Morphism, right_h: Morphism) -> Morphism:
        """Apply left optic to ``left_h`` and right optic to ``right_h`` independently."""
        _, combine, _ = _COMBINE_KINDS[self.kind]
        return combine(self.left.act(left_h), self.right.act(right_h))


def ProductOptic(left: Optic, right: Optic) -> BinaryOptic:
    """Independent-focus product of two optics: forward/backward via ``par``."""
    return BinaryOptic(left, right, "par")


def CoproductOptic(left: Optic, right: Optic) -> BinaryOptic:
    """Independent-focus coproduct of two optics: forward/backward via ``copar``."""
    return BinaryOptic(left, right, "choice")


@dataclass(frozen=True)
class RecursiveCarrier:
    """Semantic fixed-point carrier for a polynomial functor.

    ``typ`` is an opaque carrier type standing for ``μ functor``. ``roll`` and
    ``unroll`` are the structural boundary morphisms between the carrier and one
    functor layer. The current runtime representation is layer-shaped, so these
    boundaries are identity terms with distinct semantic types.
    """

    name: str
    functor: Functor
    typ: Type
    roll: Morphism
    unroll: Morphism

    @property
    def layer(self) -> Type:
        """One unrolled layer, ``F(μF)``."""
        return self.functor.apply(self.typ)

    def optic(self) -> Optic:
        """Return the recursive optic induced by this carrier."""
        return carrier_optic(self.name, "optic", self.functor, self.unroll, self.roll)


def recursive_carrier(name: str, functor: Functor) -> RecursiveCarrier:
    """Build a nominal fixed-point carrier and its roll/unroll boundaries."""
    typ = TypeVariable(Name(f"unialg.carrier.{name}"))
    layer = functor.apply(typ)
    return RecursiveCarrier(
        name=name,
        functor=functor,
        typ=typ,
        roll=Morphism(expr.Coerce(layer, typ)),
        unroll=Morphism(expr.Coerce(typ, layer)),
    )


def _compose_optic(outer: Optic, inner: Optic) -> Optic:
    """Compose two optics, threading focus through ``outer`` and then ``inner``.

    The composed functor is ``outer.functor ∘ inner.functor``.  Forward and
    backward morphisms are built using the respective optic actions so that
    the combined optic correctly decomposes and reconstructs both layers.
    """
    composed_functor = outer.functor.compose(inner.functor)
    fwd = outer.act_forward(inner.forward)
    bwd = outer.act_backward(inner.backward)
    return Optic(functor=composed_functor, forward=fwd, backward=bwd, kind=outer.kind)


def _require_carrier(fp: Optic) -> Type:
    """Return ``fp.carrier``, raising ``MorphismError`` if it is not set."""
    if fp.carrier is None:
        raise MorphismError("recursive optic must define carrier")
    return fp.carrier


def algebra(fp: Optic, alg: Morphism, i: int) -> Morphism:
    """Build a deferred catamorphism (``i=0``) or anamorphism (``i=1``) morphism.

    Validates the algebra/coalgebra shape against the carrier, constructs a
    ``SelfRef`` for the recursive call, builds the body equation using optic
    actions, and returns a ``Morphism(AlgExpr(...))`` deferred node.  No Hydra
    primitives are created here; ``structure/realize.py`` materializes the node.
    """
    carrier = _require_carrier(fp)
    kind = ("cata", "ana")[i]
    actual = (alg.dom(), alg.cod())[i]
    expected = fp.functor.apply(carrier)
    try:
        Ty.require_equal(None, actual, expected, f"{kind} shape")
    except TypeError as e:
        raise MorphismError(str(e)) from e
    name = f"unialg.{kind}.{id(fp):x}.{id(alg):x}"
    rec_dom = (carrier, alg.dom())[i]
    rec_cod = (alg.cod(), carrier)[i]
    raw_dom, raw_cod = raw_signature(alg.param, alg.monad, rec_dom, rec_cod)
    self_ref = Morphism(
        expr.SelfRef(name, raw_dom, raw_cod),
        param=alg.param,
        monad=alg.monad,
    )
    left = (fp.act_forward(self_ref), alg)[i]
    right = (alg, fp.act_backward(self_ref))[i]
    body = compose(left, right, shared_context=True)
    return Morphism(
        expr.AlgExpr(name=name, body=body.node, dom=raw_dom, cod=raw_cod),
        param=alg.param,
        monad=alg.monad,
        aux_primitives=alg.aux_primitives,
    )


def cata(fp: Optic, alg: Morphism) -> Morphism:
    """Catamorphism: fold the carrier type ``μF`` using algebra ``alg : F(B) → B``.

    Returns a deferred ``Morphism(Cata(...))`` node.  ``alg.dom()`` must equal
    ``fp.functor.apply(alg.cod())``.
    """
    return algebra(fp, alg, 0)


def ana(fp: Optic, coalg: Morphism) -> Morphism:
    """Anamorphism: unfold into the carrier type ``μF`` using coalgebra ``coalg : A → F(A)``.

    Returns a deferred ``Morphism(Ana(...))`` node.  ``coalg.cod()`` must equal
    ``fp.functor.apply(coalg.dom())``.
    """
    return algebra(fp, coalg, 1)


def hylo(fp: Optic, coalg: Morphism, alg: Morphism) -> Morphism:
    """Hylomorphism: unfold with ``coalg`` then fold with ``alg``, sharing parameter context."""
    return compose(ana(fp, coalg), cata(fp, alg), shared_context=True)


_OPTIC_BODIES: dict = {
    # kind       → body_fn(*kind_specific_args) → PolyExpr
    "lens":      lambda R:        expr.Prod(expr.Id(), expr.Const(R)),
    "prism":     lambda R:        expr.Sum(expr.Id(), expr.Const(R)),
    "affine":    lambda miss, R:  expr.Sum(expr.Const(miss), expr.Prod(expr.Id(), expr.Const(R))),
    "grate":     lambda K:        expr.Exp(expr.Const(K), expr.Id()),
    "identity":  lambda functor:  functor.body,
    "traversal": lambda functor:  functor.body,
}

_RESIDUE_CTORS: dict = {
    "lens":  ProductType,
    "prism": SumType,
}


_CARRIER_KINDS: frozenset = frozenset({"lens", "prism", "traversal"})


def _make_optic(name: str, kind: str, body_args: tuple, forward: Morphism, backward: Morphism) -> Optic:
    """Build a kind-tagged ``Optic`` by dispatching body construction through ``_OPTIC_BODIES``.

    Carrier kinds (lens, prism, traversal, identity) are routed through
    ``carrier_optic`` for boundary validation; free-shape kinds (affine, grate)
    build ``Optic`` directly and rely on ``__post_init__`` for focus/replacement checks.
    """
    body = _OPTIC_BODIES[kind](*body_args)
    functor = Functor(name=f"{kind}<{name}>", body=body)
    if kind in _CARRIER_KINDS:
        return carrier_optic(name, kind, functor, forward, backward)
    return Optic(functor=functor, forward=forward, backward=backward, kind=kind)


def _infer_residue(kind: str, forward: Morphism) -> Type:
    """Solve ``forward.cod() = ctor(forward.dom(), R)`` for R via Hydra unification."""
    A = forward.dom()
    r_var = Ty.fresh_type_var()
    expected_cod = _RESIDUE_CTORS[kind](A, r_var)
    try:
        match = Ty.unify(expected_cod, forward.cod(), f"{kind}: infer residue")  # type: ignore[arg-type]
        return Ty.apply_subst(match.substitution, r_var)
    except TypeError as e:
        raise MorphismError(f"{kind}: forward.cod() is not A ⋆ R: {e}") from e


def carrier_optic(
    name: str, kind: str, functor: Functor,
    forward: Morphism, backward: Morphism,
) -> Optic:
    """Build a carrier-shaped (endo) optic, validating that ``forward.dom()`` is the carrier.

    Specifically enforces:

        carrier = forward.dom()
        forward.cod()  == functor.apply(carrier)
        backward.dom() == functor.apply(carrier)
        backward.cod() == carrier

    Use for recursive fixed-point optics where the carrier type must be pinned
    explicitly.  Named constructors (lens, prism, traversal, identity, affine, grate)
    use ``_make_optic`` instead.
    """
    carrier = forward.dom()
    layer = functor.apply(carrier)
    try:
        Ty.require_equal(None, forward.cod(), layer, f"{kind} {name}.forward")
        Ty.require_equal(None, backward.dom(), layer, f"{kind} {name}.backward")
        Ty.require_equal(None, backward.cod(), carrier, f"{kind} {name}.carrier")
    except TypeError as e:
        raise MorphismError(str(e)) from e
    return Optic(functor=functor, forward=forward, backward=backward,
                 carrier=carrier, kind=kind)


def lens_optic(name: str, forward: Morphism, backward: Morphism) -> Optic:
    """Build a lens optic with functor F = Id × Const(R), inferring R from ``forward.cod()``."""
    return _make_optic(name, "lens", (_infer_residue("lens", forward),), forward, backward)


def prism_optic(name: str, forward: Morphism, backward: Morphism) -> Optic:
    """Build a prism optic with functor F = Id + Const(R), inferring R from ``forward.cod()``."""
    return _make_optic(name, "prism", (_infer_residue("prism", forward),), forward, backward)


def traversal_optic(name: str, functor: Functor, forward: Morphism, backward: Morphism) -> Optic:
    """Build a traversal optic from an arbitrary polynomial functor."""
    return _make_optic(name, "traversal", (functor,), forward, backward)


def identity_optic(*, name: str, functor: Functor, focus: Type) -> Optic:
    """Build an optic where S = T = F(focus), so both boundaries are identity."""
    carrier = functor.apply(focus)
    return _make_optic(name, "identity", (functor,), identity(carrier), identity(carrier))


def affine_optic(name: str, miss: Type, residue: Type, forward: Morphism, backward: Morphism) -> Optic:
    """Build an affine optic with functor F(X) = M + (X × R).

    ``miss`` M is the miss-case type; ``residue`` R is the hit-residue type.
    Unlike ``lens_optic`` and ``prism_optic``, both parameters are explicit
    because the shape carries two independent type variables.
    """
    return _make_optic(name, "affine", (miss, residue), forward, backward)


def grate_optic(name: str, index: Type, forward: Morphism, backward: Morphism) -> Optic:
    """Build a grate (closed/exponential) optic with functor F(X) = X^index.

    Not traversable: ``_exp_action`` in ``structure/realize.py`` rejects monadic
    context, so ``act`` is restricted to plain morphisms.
    """
    return _make_optic(name, "grate", (index,), forward, backward)
