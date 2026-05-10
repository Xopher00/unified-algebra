import pytest

import hydra.lexical as L
import hydra.sources.libraries as Libs
import hydra.dsl.meta.phantoms as P
from hydra.core import (
    IntegerType,
    LiteralTypeInteger,
    Name,
    TypeList,
    TypeLiteral,
    TypeMaybe,
)

from unialg import expressions as expr
from unialg import morphisms as ops
from unialg.lowering import run
from unialg.space import LIST, MAYBE


INT = TypeLiteral(LiteralTypeInteger(IntegerType.INT32))
MAYBE_INT = TypeMaybe(INT)
LIST_INT = TypeList(INT)
ADD = Name("hydra.lib.math.add")
MUL = Name("hydra.lib.math.mul")


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


def _int(term):
    return term.value.value.value


def _maybe_int(term):
    return term.value.value.value.value.value


def _list_ints(term):
    return [_int(x) for x in term.value]


def _add1():
    x = P.var("x")
    return P.lam("x", P.primitive2(ADD, x, P.int32(1))).value


def _mul2():
    x = P.var("x")
    return P.lam("x", P.primitive2(MUL, x, P.int32(2))).value


def _param_add_maybe():
    x = P.var("x")
    total = P.primitive2(ADD, P.first(x), P.second(x))
    return P.lam("x", P.apply(P.primitive(MAYBE.pure_name), total)).value


def _list_step():
    x = P.var("x")
    return P.lam("x", P.list_([x, P.primitive2(ADD, x, P.int32(1))])).value


def _list_double():
    x = P.var("x")
    return P.lam("x", P.list_([P.primitive2(MUL, x, P.int32(2))])).value


def _plain(term, dom=INT, cod=INT, aux=()):
    return ops.Morphism(expr.Prim(term, dom, cod), aux_primitives=aux)


def _para(term, param=INT, dom=INT, cod=INT, monad=None, aux=()):
    raw_cod = cod if monad is None else monad.wrap(cod)
    return ops.Morphism(
        expr.Prim(term, ops.ProductType(param, dom), raw_cod),
        param=param,
        monad=monad,
        aux_primitives=aux,
    )


def test_param_combination_and_contextual_nodes():
    plain = _plain(_add1())
    para = _para(_param_add_maybe(), monad=MAYBE)

    assert isinstance(ops.compose(plain, plain).node, expr.ContextualBinary)
    assert ops.compose(plain, plain).param == ops.TypeUnit()
    assert ops.compose(para, plain.to_lax(MAYBE)).param == INT
    assert ops.compose(plain.to_lax(MAYBE), para).param == INT
    assert ops.compose(para, para).param == ops.ProductType(INT, INT)


def test_contextual_combinators_preserve_types_and_aux():
    f = _plain(_add1(), aux=("f_aux",))
    g = _plain(_mul2(), aux=("g_aux",))

    for morphism in (ops.compose(f, g), ops.par(f, g), ops.pair(f, g)):
        assert isinstance(morphism.node, expr.ContextualBinary)
        assert morphism.aux_primitives == ("f_aux", "g_aux")

    c = ops.case(f, g)
    assert isinstance(c.node, expr.ContextualBinary)
    assert c.dom() == ops.SumType(INT, INT)
    assert c.cod() == INT


def test_lax_parametric_compose_runs_with_maybe(ctx, graph):
    f = _para(_param_add_maybe(), monad=MAYBE)
    g = _para(_param_add_maybe(), monad=MAYBE)
    h = ops.compose(f, g)

    arg = P.pair(P.pair(P.int32(10), P.int32(2)), P.int32(3)).value
    assert _maybe_int(run(h, arg, ctx, graph)) == 15


def test_lax_parametric_shared_context_compose_reuses_one_param(ctx, graph):
    f = _para(_param_add_maybe(), monad=MAYBE)
    g = _para(_param_add_maybe(), monad=MAYBE)
    h = ops.compose(f, g, shared_context=True)

    arg = P.pair(P.int32(10), P.int32(3)).value
    assert h.param == INT
    assert _maybe_int(run(h, arg, ctx, graph)) == 23


def test_pure_pair_and_par_run(ctx, graph):
    f = _plain(_add1())
    g = _plain(_mul2())

    pair_result = run(ops.pair(f, g), P.int32(5).value, ctx, graph)
    assert (_int(pair_result.value[0]), _int(pair_result.value[1])) == (6, 10)

    par_arg = P.pair(P.int32(5), P.int32(6)).value
    par_result = run(ops.par(f, g), par_arg, ctx, graph)
    assert (_int(par_result.value[0]), _int(par_result.value[1])) == (6, 12)


def test_list_lax_compose_uses_bind_semantics(ctx, graph):
    f = ops.Morphism(expr.Prim(_list_step(), INT, LIST_INT), monad=LIST)
    g = ops.Morphism(expr.Prim(_list_double(), INT, LIST_INT), monad=LIST)

    assert _list_ints(run(ops.compose(f, g), P.int32(3).value, ctx, graph)) == [6, 8]
