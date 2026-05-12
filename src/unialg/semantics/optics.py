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

No Hydra imports.  No encoding logic.  The action lives in ``recursion.act``.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from functors import Functor
from morphisms import Morphism, MorphismError, compose, identity, raw_signature
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
        except TypeError as e:
            raise MorphismError(
                f"invalid optic forward codomain: {e}"
            ) from e
        try:
            replacement = self.functor.unapply(self.backward.dom())
        except TypeError as e:
            raise MorphismError(
                f"invalid optic backward domain: {e}"
            ) from e
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
    
    def compose(self, inner: Optic) -> Optic:
        """Compose two optics: focus through ``outer`` then ``inner``."""
        return _compose_optic(self, inner)
    

def _compose_optic(outer: Optic, inner: Optic) -> Optic:
        """Compose two optics: focus through ``outer`` then ``inner``."""
        composed_functor = outer.functor.compose(inner.functor)
        fwd = outer.act_forward(inner.forward)
        bwd = outer.act_backward(inner.backward)
        return Optic(functor=composed_functor, forward=fwd, backward=bwd)
    

def _require_carrier(fp: Optic) -> Type:
    if fp.carrier is None:
        raise MorphismError("recursive optic must define carrier")
    return fp.carrier


def algebra(fp: Optic, alg: Morphism, i: int) -> Morphism:
    carrier = _require_carrier(fp)
    kind = ("cata", "ana")[i]
    MorphismError.check(
        (alg.dom(), alg.cod())[i],
        fp.functor.apply((alg.cod(), alg.dom())[i]),
        f"{kind} shape",
    )
    name = f"unialg.{kind}.{id(fp):x}.{id(alg):x}"
    raw_dom, raw_cod = raw_signature(
        alg.param, alg.monad,
        (carrier, alg.dom())[i], (alg.cod(), carrier)[i],
    )    
    self_ref = Morphism(
        expr.SelfRef(name, raw_dom, raw_cod),
        param=alg.param, monad=alg.monad,
    )
    body = compose(
        (fp.act_forward(self_ref), alg)[i],
        (alg, fp.act_backward(self_ref))[i],
        shared_context=True,
    )
    return Morphism(
        expr.AlgExpr(name=name, body=body.node, dom=raw_dom, cod=raw_cod),
        param=alg.param, monad=alg.monad, aux_primitives=alg.aux_primitives,
    )


def cata(fp: Optic, alg: Morphism) -> Morphism:
    return algebra(fp, alg, 0)


def ana(fp: Optic, coalg: Morphism) -> Morphism:
    return algebra(fp, coalg, 1)


def hylo(fp: Optic, coalg: Morphism, alg: Morphism) -> Morphism:
    return compose(ana(fp, coalg), cata(fp, alg), shared_context=True)


def identity_optic(*, name: str, functor: Functor, focus: Type) -> Optic:
    """Build an optic where S = T = F(focus), so both boundaries are identity."""
    carrier = functor.apply(focus)
    return Optic(
        functor=functor,
        forward=identity(carrier),
        backward=identity(carrier),
    )


def fixed_point_optic(*, functor: Functor, carrier: Type, unroll, roll, ) -> Optic:
    layer = functor.apply(carrier)
    return Optic(
        functor=functor,
        forward=Morphism(expr.Prim(unroll, carrier, layer)),
        backward=Morphism(expr.Prim(roll, layer, carrier)),
        carrier=carrier,
    )