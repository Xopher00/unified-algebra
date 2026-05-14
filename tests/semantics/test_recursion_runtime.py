import pytest

import hydra.lexical as L
import hydra.sources.libraries as Libs
import hydra.dsl.meta.phantoms as P
from hydra.core import Name

from unialg.syntax import expressions as expr
from unialg.structure import terms as H
from unialg.semantics import morphisms as ops
from unialg.semantics.functors import Functor
from unialg.main import run
from unialg.semantics.optics import Optic
from unialg.semantics.optics import ana, cata, hylo
from unialg.objects import MAYBE
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


def _maybe_int(term) -> int:
    return _int(term.value.value)


def _one_or_self_carrier(rolled_value: int = 42) -> Optic:
    shape = ops.SumType(UNIT, INT)
    return Optic(
        functor=Functor("OneOrSelf", expr.Sum(expr.One(), expr.Id())),
        forward=ops.compose(ops._delete(INT), ops._inl(shape)),
        backward=ops.case(_const_int(rolled_value), ops.identity(INT)),
        carrier=INT,
    )


def _const_int(value: int) -> ops.Morphism:
    raw = P.lam("u", P.int32(value)).value
    return ops.Morphism(expr.Prim(raw, UNIT, INT))


def _left_unit() -> ops.Morphism:
    shape = ops.SumType(UNIT, INT)
    return ops.compose(ops._delete(INT), ops._inl(shape))


def _lax_para_const_from_param(dom=UNIT, offset: int = 1) -> ops.Morphism:
    x = P.var("x")
    value = P.primitive2(ADD, P.first(x), P.int32(offset))
    raw = P.lam("x", P.apply(P.primitive(MAYBE.pure_name), value)).value
    return ops.Morphism(
        expr.Prim(raw, ops.ProductType(INT, dom), MAYBE.wrap(INT)),
        param=INT,
        monad=MAYBE,
    )


def _lax_para_sum_alg() -> ops.Morphism:
    return ops.case(
        _lax_para_const_from_param(UNIT),
        _lax_para_const_from_param(INT),
        shared_context=True,
    )


def _lax_para_to_left_unit() -> ops.Morphism:
    shape = ops.SumType(UNIT, INT)
    x = P.var("x")
    raw = P.lam(
        "x",
        P.apply(
            P.primitive(MAYBE.pure_name),
            P.apply(H.left_injection(), P.unit()),
        ),
    ).value
    return ops.Morphism(
        expr.Prim(raw, ops.ProductType(INT, INT), MAYBE.wrap(shape)),
        param=INT,
        monad=MAYBE,
    )


def test_cata_runtime_smoke(ctx, graph):
    folded = cata(
        _one_or_self_carrier(),
        ops.case(_const_int(7), ops.identity(INT)),
    )

    assert _int(run(folded, P.int32(999).value, ctx, graph)) == 7


def test_ana_runtime_smoke(ctx, graph):
    unfolded = ana(_one_or_self_carrier(42), _left_unit())

    assert _int(run(unfolded, P.int32(5).value, ctx, graph)) == 42


def test_hylo_runtime_smoke(ctx, graph):
    transformed = hylo(
        _one_or_self_carrier(42),
        _left_unit(),
        ops.case(_const_int(7), ops.identity(INT)),
    )

    assert _int(run(transformed, P.int32(5).value, ctx, graph)) == 7


def test_lax_para_cata_runtime_uses_one_shared_param(ctx, graph):
    folded = cata(_one_or_self_carrier(), _lax_para_sum_alg())
    arg = P.pair(P.int32(10), P.int32(999)).value

    assert folded.param == INT
    assert _maybe_int(run(folded, arg, ctx, graph)) == 11


def test_lax_para_ana_runtime_uses_one_shared_param(ctx, graph):
    unfolded = ana(_one_or_self_carrier(42), _lax_para_to_left_unit())
    arg = P.pair(P.int32(10), P.int32(5)).value

    assert unfolded.param == INT
    assert _maybe_int(run(unfolded, arg, ctx, graph)) == 42


def test_lax_para_hylo_runtime_uses_one_shared_param(ctx, graph):
    transformed = hylo(
        _one_or_self_carrier(42),
        _lax_para_to_left_unit(),
        _lax_para_sum_alg(),
    )
    arg = P.pair(P.int32(10), P.int32(5)).value

    assert transformed.param == INT
    assert _maybe_int(run(transformed, arg, ctx, graph)) == 11
