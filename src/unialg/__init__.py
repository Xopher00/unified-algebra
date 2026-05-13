from .objects import Monad, MAYBE, LIST, ProductType, SumType
from .syntax.expressions import PolyExpr, Zero, One, Id, Const, Sum, Prod, Exp
from .semantics.functors import Functor, zero, one, id_, const, sum_, prod, exp, apply_poly, poly_fmap
from .semantics.morphisms import Morphism, MorphismError
from .semantics.morphisms import (
    identity, _copy as copy, _delete as delete,
    _fst as fst, _snd as snd, _inl as inl, _inr as inr,
    _assoc as assoc, _symmetry as symmetry,
    compose, par, pair, case, absurd,
)
from .tensors.semirings import Semiring
from .semantics.optics import Optic, _compose_optic


def act(optic: Optic, h) -> "Morphism":
    return optic.act(h)


def act_forward(optic: Optic, h) -> "Morphism":
    return optic.act_forward(h)


def act_backward(optic: Optic, h) -> "Morphism":
    return optic.act_backward(h)


compose_optic = _compose_optic

from .structure.recursion import recursive_carrier
from .semantics.optics import cata, ana, hylo
from .lowering import lower, run

__all__ = [
    "Morphism",
    "ProductType",
    "SumType",
    "MorphismError",
    "identity",
    "compose",
    "fst",
    "snd",
    "par",
    "pair",
    "copy",
    "delete",
    "inl",
    "inr",
    "case",
    "absurd",
    "assoc",
    "symmetry",
    "Semiring",
    "lower",
    "run",
    "Monad",
    "MAYBE",
    "LIST",
    "PolyExpr",
    "Zero",
    "One",
    "Id",
    "Const",
    "Sum",
    "Prod",
    "Exp",
    "Functor",
    "zero",
    "one",
    "id_",
    "const",
    "sum_",
    "prod",
    "exp",
    "Optic",
    "poly_fmap",
    "apply_poly",
    "recursive_carrier",
    "cata",
    "ana",
    "hylo",
    "act",
    "act_forward",
    "act_backward",
    "compose_optic",
]
