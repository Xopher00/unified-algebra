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
carriers use small Hydra type/term constructors for their roll/unroll boundary
morphisms.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from hydra.core import Name, TypeVariable
import hydra.dsl.meta.phantoms as P

from . import typeops as Ty
from .functors import Functor
from .morphisms import Morphism, MorphismError, compose, identity, par as _par_morphisms, raw_signature
from unialg.syntax import expressions as expr
from unialg.objects import Type


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
    _focus: Type = field(init=False, repr=False, compare=False)
    _replacement: Type = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            focus = self.functor.unapply(self.forward.cod())
            Ty.roundtrip_equal(
                None,
                self.functor.apply,
                focus,
                self.forward.cod(),
                "optic focus shape",
            )
        except TypeError as e:
            raise MorphismError(f"invalid optic forward codomain: {e}") from e

        try:
            replacement = self.functor.unapply(self.backward.dom())
            Ty.roundtrip_equal(
                None,
                self.functor.apply,
                replacement,
                self.backward.dom(),
                "optic replacement shape",
            )
        except TypeError as e:
            raise MorphismError(f"invalid optic backward domain: {e}") from e

        object.__setattr__(self, "_focus", focus)
        object.__setattr__(self, "_replacement", replacement)

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

    def par(self, other: "Optic") -> "Optic":
        """Parallel composition of two optics (product of optics).

        Combines two optics that observe disjoint positions of a product
        (the categorical product on optics, analogous to ``ops.par`` on morphisms):

            functor.body = Prod(self.functor.body, other.functor.body)
            forward      = par(self.forward, other.forward)   : S1×S2 → F(A1)×G(A2)
            backward     = par(self.backward, other.backward) : F(B1)×G(B2) → T1×T2
            carrier      = Prod(self.carrier, other.carrier)  (when both set)

        Both optics must have the same carrier type (A1 = A2), enforced by
        ``Optic.__post_init__`` via ``functor.unapply``.
        """
        body = expr.Prod(self.functor.body, other.functor.body)
        functor = Functor(
            name=f"{self.functor.name}|{other.functor.name}",
            body=body,
        )
        fwd = _par_morphisms(self.forward, other.forward)
        bwd = _par_morphisms(self.backward, other.backward)
        # Both optics must observe the same carrier type X so that
        # apply_poly(Prod(L.body, R.body), X) == fwd.cod().  Keeping carrier
        # as the shared X (not Prod(X, X)) ensures _compose_optic's focus
        # check passes when this optic is composed further.
        if self.carrier is not None and other.carrier is not None:
            if self.carrier != other.carrier:
                raise MorphismError(
                    f"Optic.par: carrier mismatch {self.carrier!r} vs {other.carrier!r}"
                )
            carrier = self.carrier
        else:
            carrier = self.carrier or other.carrier
        return Optic(functor=functor, forward=fwd, backward=bwd, carrier=carrier)


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
        return Optic(
            functor=self.functor,
            forward=self.unroll,
            backward=self.roll,
            carrier=self.typ,
        )


def recursive_carrier(name: str, functor: Functor) -> RecursiveCarrier:
    """Build a nominal fixed-point carrier and its roll/unroll boundaries."""
    typ = TypeVariable(Name(f"unialg.carrier.{name}"))
    layer = functor.apply(typ)
    raw_identity = P.lam("x", P.var("x")).value
    return RecursiveCarrier(
        name=name,
        functor=functor,
        typ=typ,
        roll=Morphism(expr.Prim(raw_identity, layer, typ)),
        unroll=Morphism(expr.Prim(raw_identity, typ, layer)),
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
    return Optic(functor=composed_functor, forward=fwd, backward=bwd)
    

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


def identity_optic(*, name: str, functor: Functor, focus: Type) -> Optic:
    """Build an optic where S = T = F(focus), so both boundaries are identity."""
    carrier = functor.apply(focus)
    return Optic(
        functor=functor,
        forward=identity(carrier),
        backward=identity(carrier),
    )
