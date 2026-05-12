"""Small morphisms.py surface for Pynguin.

Pynguin currently fails when it instruments ``unialg.morphisms`` directly on
Python 3.12. This wrapper lets it generate tests around selected behavior while
keeping the real module as an uninstrumented dependency.
"""

from hydra.core import IntegerType, LiteralTypeInteger, TypeLiteral, TypeUnit

from unialg import expressions as expr
from unialg import functors as sem
from unialg import morphisms as ops


INT = TypeLiteral(LiteralTypeInteger(IntegerType.INT32))
UNIT = TypeUnit()


def product_of_int_and_unit():
    return ops.ProductType(INT, UNIT)


def sum_of_int_and_unit():
    return ops.SumType(INT, UNIT)


def apply_id_to_int():
    return sem.apply_poly(expr.Id(), INT)


def apply_one_to_int():
    return sem.apply_poly(expr.One(), INT)


def apply_const_unit_to_int():
    return sem.apply_poly(expr.Const(UNIT), INT)


def apply_product_shape_to_int():
    return sem.apply_poly(expr.Prod(expr.Id(), expr.Const(UNIT)), INT)


def apply_sum_shape_to_int():
    return sem.apply_poly(expr.Sum(expr.One(), expr.Id()), INT)


def identity_int_dom():
    return ops.identity(INT).dom()


def identity_int_cod():
    return ops.identity(INT).cod()


def copy_int_cod():
    return ops._copy(INT).cod()


def compose_identity_int_dom():
    return ops.compose(ops.identity(INT), ops.identity(INT)).dom()


def compose_identity_int_cod():
    return ops.compose(ops.identity(INT), ops.identity(INT)).cod()
