from __future__ import annotations

from unialg.objects import Type
from unialg.syntax import expressions as expr
from . import terms as T
from unialg.semantics.functors import Functor
from unialg.semantics.morphisms import Morphism
from unialg.semantics.optics import Optic


def fixed_point_optic(*, functor: Functor, carrier: Type, unroll, roll) -> Optic:
    """Shim: structural fixed-point optic constructor. Retained for review."""
    layer = functor.apply(carrier)
    return Optic(
        functor=functor,
        forward=Morphism(expr.Prim(unroll, carrier, layer)),
        backward=Morphism(expr.Prim(roll, layer, carrier)),
        carrier=carrier,
    )


def recursive_carrier(*, functor: Functor, carrier: Type, unroll, roll) -> Optic:
    """Build a carrier ``Optic`` from Python unroll/roll callables.

    ``unroll`` and ``roll`` are called with a single ``TTerm`` argument and must
    return a ``TTerm``.  They are wrapped as ``Prim`` morphisms via
    ``T.term_lambda`` and forwarded to ``fixed_point_optic``.
    """
    return fixed_point_optic(
        functor=functor, carrier=carrier,
        unroll=T.term_lambda("x", unroll).value,
        roll=T.term_lambda("layer", roll).value,
    )
