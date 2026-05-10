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

from unialg.syntax import expressions as expr
from unialg.objects import Type, ProductType, SumType
from . import functors
from . import morphisms


@dataclass(frozen=True)
class Optic:
    """Polynomial functor optic over ``(S, A, B, T)``.

    ``functor`` describes the container shape F.
    ``forward : S → F(A)`` decomposes the source.
    ``backward : F(B) → T`` reconstructs the target.

    Focus ``A`` and replacement ``B`` are derived via ``functor.unapply``
    on the forward codomain and backward domain respectively.
    """
    functor: functors.Functor
    forward: morphisms.Morphism
    backward: morphisms.Morphism
    carrier: Type | None = None
    _focus: Type = field(init=False, repr=False, compare=False)
    _replacement: Type = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            focus = self.functor.unapply(self.forward.cod())
        except TypeError as e:
            raise morphisms.MorphismError(
                f"invalid optic forward codomain: {e}"
            ) from e
        try:
            replacement = self.functor.unapply(self.backward.dom())
        except TypeError as e:
            raise morphisms.MorphismError(
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
    



def identity_optic(*, name: str, functor: functors.Functor, focus: Type) -> Optic:
    """Build an optic where S = T = F(focus), so both boundaries are identity."""
    carrier = functor.apply(focus)
    return Optic(
        functor=functor,
        forward=morphisms._identity(carrier),
        backward=morphisms._identity(carrier),
    )


def fixed_point_optic(*, functor: functors.Functor, carrier: Type, unroll, roll, ) -> Optic:
    layer = functor.apply(carrier)
    return Optic(
        functor=functor,
        forward=morphisms.Morphism(expr.Prim(unroll, carrier, layer)),
        backward=morphisms.Morphism(expr.Prim(roll, layer, carrier)),
        carrier=carrier,
    )