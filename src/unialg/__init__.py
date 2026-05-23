from .objects import Monad, MAYBE, LIST, ProductType, SumType
from .syntax.expressions import PolyExpr, Zero, One, Id, Const, Sum, Prod, Exp
from .semantics.functors import Functor, zero, one, id_, const, sum_, prod, exp, apply_poly, poly_fmap
from .semantics.morphisms import Morphism, MorphismError
from .semantics.morphisms import (
    identity, _copy as copy, _delete as delete,
    _first as fst, _second as snd, _inject_left as inl, _inject_right as inr,
    _assoc as assoc, _symmetry as symmetry,
    compose, par, copar, pair, case, absurd,
)
from .tensors.semirings import Semiring
from .semantics.optics import (
    Optic, BinaryOptic, ProductOptic, CoproductOptic, _compose_optic,
    carrier_optic, affine_optic, grate_optic,
)


def act(optic: Optic, h) -> "Morphism":
    return optic.act(h)


def act_forward(optic: Optic, h) -> "Morphism":
    return optic.act_forward(h)


def act_backward(optic: Optic, h) -> "Morphism":
    return optic.act_backward(h)


compose_optic = _compose_optic

from .semantics.optics import cata, ana, hylo, recursive_carrier
from .main import lower, run, compile_morphism, compile_program, CompiledProgram

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
    "copar",
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
    "compile_morphism",
    "compile_program",
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
    "BinaryOptic",
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
    "ProductOptic",
    "CoproductOptic",
    "carrier_optic",
    "affine_optic",
    "grate_optic",
]
