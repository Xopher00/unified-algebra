import pytest

import hydra.lexical as L
import hydra.sources.libraries as Libs
import hydra.dsl.meta.phantoms as P
import hydra.dsl.terms as Terms
from hydra.core import Name, TypeScheme
from hydra.dsl.python import Right
from hydra.graph import Primitive

from unialg.syntax import expressions as expr
from unialg.semantics import morphisms as ops
from unialg.main import run
from unialg.structure.recursion import recursive_carrier
from unialg.semantics.optics import ana, cata, hylo
from unialg.objects import ExpType
from support.strategies import INT


UNIT = ops.TypeUnit()
ADD = Name("hydra.lib.math.add")


@pytest.fixture(scope="module")
def graph():
    primitives = []
    for attr in dir(Libs):
        if attr.startswith("register_") and attr.endswith("_primitives"):
            primitives.extend(getattr(Libs, attr)().values())
    return L.graph_with_primitives(primitives, ())


@pytest.fixture(scope="module")
def ctx():
    return L.empty_context()


def _int(term) -> int:
    return term.value.value.value


def _list_ints(term) -> list[int]:
    return [_int(item) for item in term.value]


def _const_int(value: int) -> ops.Morphism:
    raw = P.lam("u", P.int32(value)).value
    return ops.Morphism(expr.Prim(raw, UNIT, INT))


def _add_pair() -> ops.Morphism:
    cell = P.var("cell")
    raw = P.lam("cell", P.primitive2(ADD, P.first(cell), P.second(cell))).value
    return ops.Morphism(expr.Prim(raw, ops.ProductType(INT, INT), INT))


def _sum_alg() -> ops.Morphism:
    return ops.case(_const_int(0), _add_pair())


def _countdown_coalg() -> ops.Morphism:
    layer = ops.SumType(UNIT, ops.ProductType(INT, INT))
    name = Name("unialg.test.countdownCoalg")

    def impl(_ctx, _graph, args):
        value = _int(args[0])
        if value <= 0:
            return Right(Terms.left(Terms.unit()))
        return Right(Terms.right(Terms.pair(Terms.int32(value), Terms.int32(value - 1))))

    prim = Primitive(
        name,
        TypeScheme(variables=(), body=ExpType(INT, layer), constraints=None),
        impl,
    )
    return ops.Morphism(expr.Prim(P.primitive(name).value, INT, layer), aux_primitives=(prim,))


def test_list_carrier_boundary_types():
    fp = list_carrier(INT)
    expected_layer = fp.functor.apply(fp.carrier)

    assert fp.forward.dom() == fp.carrier
    assert fp.forward.cod() == expected_layer
    assert fp.backward.dom() == expected_layer
    assert fp.backward.cod() == fp.carrier


def test_list_carrier_cata_sums_list(ctx, graph):
    folded = cata(list_carrier(INT), _sum_alg())
    values = P.list_([P.int32(1), P.int32(2), P.int32(3)]).value

    assert _int(run(folded, values, ctx, graph)) == 6


def test_list_carrier_ana_builds_countdown(ctx, graph):
    unfolded = ana(list_carrier(INT), _countdown_coalg())

    assert _list_ints(run(unfolded, P.int32(3).value, ctx, graph)) == [3, 2, 1]


def test_list_carrier_hylo_sums_countdown(ctx, graph):
    transformed = hylo(list_carrier(INT), _countdown_coalg(), _sum_alg())

    assert _int(run(transformed, P.int32(4).value, ctx, graph)) == 10
