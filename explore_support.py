"""Reader-friendly fixtures for explore.ipynb.

This module keeps Hydra term construction and notebook packing helpers out of the
main tutorial path. The notebook can focus on the unialg API while this file owns
the concrete sample morphisms used for examples.
"""

from __future__ import annotations

import hydra.dsl.meta.phantoms as P
import hydra.dsl.terms as Terms
import hydra.lexical as L
import hydra.sources.libraries as Libs
from hydra.core import (
    IntegerType,
    LiteralTypeInteger,
    Name,
    Type,
    TypeEither,
    TypeFunction,
    TypeList,
    TypeLiteral,
    TypeMaybe,
    TypePair,
    TypeUnit,
    TypeVoid,
)
from hydra.dsl.python import Just, Nothing

from unialg.main import lower as _lower_node, run
from unialg.objects import LIST, MAYBE, ProductType, SumType
from unialg.semantics.functors import Functor
from unialg.semantics.morphisms import (
    Morphism,
    _assoc as assoc,
    _delete as delete,
    _inl as inl,
    _inr as inr,
    _symmetry as symmetry,
    case,
    compose,
    identity,
)
from unialg.semantics.optics import Optic, ana, cata, hylo
from unialg.tensors.semirings import Semiring
from unialg.syntax import expressions as expr


def lower(morphism: Morphism, graph=None):
    """Lower a notebook Morphism through the current source API."""
    return _lower_node(morphism.node, graph)


def make_graph():
    primitives = []
    for attr in dir(Libs):
        if attr.startswith("register_") and attr.endswith("_primitives"):
            primitives.extend(getattr(Libs, attr)().values())
    return L.graph_with_primitives(primitives, ())


graph = make_graph()
ctx = L.empty_context()


INT = TypeLiteral(LiteralTypeInteger(IntegerType.INT32))
INT_PAIR = ProductType(INT, INT)
INT_SUM = SumType(INT, INT)
UNIT = TypeUnit()
MAYBE_INT = TypeMaybe(INT)
LIST_INT = TypeList(INT)


HYDRA_PRIMS = {
    "add": Name("hydra.lib.math.add"),
    "mul": Name("hydra.lib.math.mul"),
}


def _unary_int_op(op: str, rhs):
    x = P.var("x")
    return P.lam("x", P.primitive2(HYDRA_PRIMS[op], x, rhs)).value


def _binary_pair_op(op: str):
    p = P.var("p")
    return P.lam("p", P.primitive2(HYDRA_PRIMS[op], P.first(p), P.second(p))).value


def _maybe_unary_int_op(op: str, rhs):
    x = P.var("x")
    value = P.primitive2(HYDRA_PRIMS[op], x, rhs)
    return P.lam("x", P.apply(P.primitive(MAYBE.pure_name), value)).value


def _list_step_raw():
    x = P.var("x")
    return P.lam("x", P.TTerm(Terms.list_([
        x.value,
        P.primitive2(HYDRA_PRIMS["add"], x, P.int32(1)).value,
    ]))).value


def _list_double_raw():
    x = P.var("x")
    return P.lam("x", P.TTerm(Terms.list_([
        P.primitive2(HYDRA_PRIMS["mul"], x, P.int32(2)).value,
    ]))).value


add1 = Morphism(expr.Prim(_unary_int_op("add", P.int32(1)), INT, INT))
double = Morphism(expr.Prim(_unary_int_op("mul", P.int32(2)), INT, INT))
add3 = Morphism(expr.Prim(_unary_int_op("add", P.int32(3)), INT, INT))
add_pair = Morphism(expr.Prim(_binary_pair_op("add"), INT_PAIR, INT))

add_param = Morphism(expr.Prim(_binary_pair_op("add"), ProductType(INT, INT), INT), param=INT)
add_param_again = Morphism(expr.Prim(_binary_pair_op("add"), ProductType(INT, INT), INT), param=INT)

maybe_add1 = Morphism(expr.Prim(_maybe_unary_int_op("add", P.int32(1)), INT, MAYBE_INT), monad=MAYBE)
maybe_double = Morphism(expr.Prim(_maybe_unary_int_op("mul", P.int32(2)), INT, MAYBE_INT), monad=MAYBE)
list_step = Morphism(expr.Prim(_list_step_raw(), INT, LIST_INT), monad=LIST)
list_double = Morphism(expr.Prim(_list_double_raw(), INT, LIST_INT), monad=LIST)


def show_type(t: Type) -> str:
    if isinstance(t, TypeLiteral):
        lit = t.value
        if isinstance(lit, LiteralTypeInteger):
            return "Int"
        return repr(t)
    if isinstance(t, TypePair):
        return f"{show_type(t.value.first)} x {show_type(t.value.second)}"
    if isinstance(t, TypeEither):
        return f"{show_type(t.value.left)} + {show_type(t.value.right)}"
    if isinstance(t, TypeUnit):
        return "Unit"
    if isinstance(t, TypeVoid):
        return "Void"
    if isinstance(t, TypeMaybe):
        return f"Maybe {show_type(t.value)}"
    if isinstance(t, TypeList):
        return f"List {show_type(t.value)}"
    if isinstance(t, TypeFunction):
        return f"{show_type(t.value.domain)} -> {show_type(t.value.codomain)}"
    return repr(t)


def show_morphism(m: Morphism) -> str:
    cod = show_type(m.cod())
    if m.monad is MAYBE:
        cod = f"Maybe ({cod})" if " " in cod else f"Maybe {cod}"
    elif m.monad is LIST:
        cod = f"List ({cod})" if " " in cod else f"List {cod}"
    if m.param == TypeUnit():
        return f"{show_type(m.dom())} -> {cod}"
    return f"param {show_type(m.param)}, input {show_type(m.dom())} -> {cod}"


def int_term(n: int):
    return P.int32(n)


def int_arg(n: int):
    return int_term(n).value


def pair_arg(left, right):
    return P.pair(left, right).value


def int_pair_arg(pair: tuple[int, int]):
    left, right = pair
    return pair_arg(int_term(left), int_term(right))


def para_int_arg(param: int, value: int):
    return pair_arg(int_term(param), int_term(value))


def composed_para_int_arg(f_param: int, g_param: int, value: int):
    params = P.pair(int_term(g_param), int_term(f_param))
    return pair_arg(params, int_term(value))


def para_int_pair_arg(param: int, value: tuple[int, int]):
    left, right = value
    return pair_arg(int_term(param), P.pair(int_term(left), int_term(right)))


def left_int_arg(sum_type, value: int):
    return Terms.apply(lower(inl(sum_type), graph=None), int_arg(value))


def right_int_arg(sum_type, value: int):
    return Terms.apply(lower(inr(sum_type), graph=None), int_arg(value))


def int_val(result) -> int:
    return result.value


def pair_val(result) -> tuple[int, int]:
    return (int_val(result.value[0]), int_val(result.value[1]))


def maybe_payload(result):
    value = result.value
    if isinstance(value, Nothing):
        return None
    if isinstance(value, Just):
        return value.value
    raise TypeError(f"unknown Maybe value: {value!r}")


def run_int(m: Morphism, value: int) -> int:
    return int_val(run(m, int_arg(value), ctx, graph))


def run_pair(m: Morphism, value: int | tuple[int, int]) -> tuple[int, int]:
    arg = int_pair_arg(value) if isinstance(value, tuple) else int_arg(value)
    return pair_val(run(m, arg, ctx, graph))


def run_int_from_pair(m: Morphism, value: tuple[int, int]) -> int:
    return int_val(run(m, int_pair_arg(value), ctx, graph))


def run_left_int(m: Morphism, value: int) -> int:
    return int_val(run(m, left_int_arg(m.dom(), value), ctx, graph))


def run_right_int(m: Morphism, value: int) -> int:
    return int_val(run(m, right_int_arg(m.dom(), value), ctx, graph))


def run_para_int(m: Morphism, *, param: int, value: int) -> int:
    return int_val(run(m, para_int_arg(param, value), ctx, graph))


def run_composed_para_int(m: Morphism, *, f_param: int, g_param: int, value: int) -> int:
    return int_val(run(m, composed_para_int_arg(f_param, g_param, value), ctx, graph))


def run_para_pair(m: Morphism, *, param: int, value: tuple[int, int]) -> tuple[int, int]:
    return pair_val(run(m, para_int_pair_arg(param, value), ctx, graph))


def run_maybe_int(m: Morphism, value: int) -> int | None:
    payload = maybe_payload(run(m, int_arg(value), ctx, graph))
    return None if payload is None else int_val(payload)


def run_maybe_pair(m: Morphism, value: tuple[int, int]) -> tuple[int, int] | None:
    payload = maybe_payload(run(m, int_pair_arg(value), ctx, graph))
    return None if payload is None else pair_val(payload)


def run_list_int(m: Morphism, value: int) -> list[int]:
    result = run(m, int_arg(value), ctx, graph)
    return [int_val(item) for item in result.value]


def run_sum_int(m: Morphism, *, side: str, value: int) -> str:
    arg = left_int_arg(m.dom(), value) if side == "left" else right_int_arg(m.dom(), value)
    result = run(m, arg, ctx, graph)
    branch = result.value
    label = "Left" if side == "left" else "Right"
    return f"{label} {int_val(branch.value)}"


# --- assoc / symmetry ---------------------------------------------------------

INT_TRIPLE_L = ProductType(ProductType(INT, INT), INT)
mul_pair = Morphism(expr.Prim(_binary_pair_op("mul"), INT_PAIR, INT))
const_zero = Morphism(expr.Prim(P.lam("u", P.int32(0)).value, UNIT, INT))
const_one = Morphism(expr.Prim(P.lam("u", P.int32(1)).value, UNIT, INT))


def nested_left_arg(a: int, b: int, c: int):
    """Pack ((a, b), c) as a Hydra term for assoc input."""
    return P.pair(P.pair(int_term(a), int_term(b)), int_term(c)).value


def nested_right_val(result) -> tuple[int, tuple[int, int]]:
    """Unpack a (a, (b, c)) Hydra result."""
    outer = result.value
    return (int_val(outer[0]), (int_val(outer[1].value[0]), int_val(outer[1].value[1])))


def run_assoc(m: Morphism, a: int, b: int, c: int) -> tuple[int, tuple[int, int]]:
    return nested_right_val(run(m, nested_left_arg(a, b, c), ctx, graph))


# --- optic / cata / ana fixtures ----------------------------------------------

def const_int(value: int) -> Morphism:
    """Constant morphism Unit → Int."""
    raw = P.lam("u", P.int32(value)).value
    return Morphism(expr.Prim(raw, UNIT, INT))


def one_or_self_optic(rolled_value: int = 42) -> Optic:
    """Trivially terminating recursive optic. Carrier = Int, F(X) = 1 + X.

    unroll: Int → 1 + Int  always goes left (stop immediately)
    roll:   1 + Int → Int  returns rolled_value from Unit branch, identity from Int
    """
    shape = SumType(UNIT, INT)
    return Optic(
        functor=Functor("OneOrSelf", expr.Sum(expr.One(), expr.Id())),
        forward=compose(delete(INT), inl(shape)),
        backward=case(const_int(rolled_value), identity(INT)),
        carrier=INT,
    )
