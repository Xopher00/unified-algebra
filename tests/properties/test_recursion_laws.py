import hydra.dsl.meta.phantoms as P

from unialg.syntax import expressions as expr
from unialg.semantics import morphisms as ops
from unialg.semantics.functors import Functor
from unialg.semantics.optics import Optic
from unialg.structure.recursion import ana, cata, hylo
from unialg.objects import MAYBE
from support.strategies import INT


def _identity_optic() -> Optic:
    return Optic(
        functor=Functor("Id", expr.Id()),
        forward=ops.identity(INT),
        backward=ops.identity(INT),
        carrier=INT,
    )


def _plain() -> ops.Morphism:
    return ops.Morphism(expr.Prim(P.identity().value, INT, INT))


def _para() -> ops.Morphism:
    x = P.var("x")
    raw = P.lam("x", P.second(x)).value
    return ops.Morphism(expr.Prim(raw, ops.ProductType(INT, INT), INT), param=INT)


def _lax() -> ops.Morphism:
    x = P.var("x")
    raw = P.lam("x", P.apply(P.primitive(MAYBE.pure_name), x)).value
    return ops.Morphism(expr.Prim(raw, INT, MAYBE.wrap(INT)), monad=MAYBE)


def _lax_para() -> ops.Morphism:
    x = P.var("x")
    raw = P.lam("x", P.apply(P.primitive(MAYBE.pure_name), P.second(x))).value
    return ops.Morphism(
        expr.Prim(raw, ops.ProductType(INT, INT), MAYBE.wrap(INT)),
        param=INT,
        monad=MAYBE,
    )


def _assert_context(morphism, source):
    assert morphism.param == source.param
    assert morphism.monad == source.monad
    assert morphism.dom() == INT
    assert morphism.cod() == INT


def test_cata_type_laws_for_plain_para_lax_and_lax_para():
    fp = _identity_optic()
    for alg in (_plain(), _para(), _lax(), _lax_para()):
        _assert_context(cata(fp, alg), alg)


def test_ana_type_laws_for_plain_para_lax_and_lax_para():
    fp = _identity_optic()
    for coalg in (_plain(), _para(), _lax(), _lax_para()):
        _assert_context(ana(fp, coalg), coalg)


def test_hylo_preserves_shared_plain_context():
    fp = _identity_optic()
    result = hylo(fp, _plain(), _plain())

    assert result.param == ops.TypeUnit()
    assert result.monad is None
    assert result.dom() == INT
    assert result.cod() == INT


def test_hylo_preserves_shared_para_context():
    fp = _identity_optic()
    result = hylo(fp, _para(), _para())

    assert result.param == INT
    assert result.monad is None
    assert result.dom() == INT
    assert result.cod() == INT


def test_hylo_preserves_shared_lax_context():
    fp = _identity_optic()
    result = hylo(fp, _lax(), _lax())

    assert result.param == ops.TypeUnit()
    assert result.monad is MAYBE
    assert result.dom() == INT
    assert result.cod() == INT


def test_hylo_preserves_shared_lax_para_context():
    fp = _identity_optic()
    result = hylo(fp, _lax_para(), _lax_para())

    assert result.param == INT
    assert result.monad is MAYBE
    assert result.dom() == INT
    assert result.cod() == INT


def test_recursion_rejects_missing_carrier():
    optic = Optic(
        functor=Functor("Id", expr.Id()),
        forward=ops.identity(INT),
        backward=ops.identity(INT),
    )

    try:
        cata(optic, _plain())
    except ops.MorphismError:
        pass
    else:
        raise AssertionError("cata accepted optic without carrier")
