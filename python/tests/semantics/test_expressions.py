import pytest

from hydra.core import (
    EitherType,
    IntegerType,
    LiteralTypeInteger,
    PairType,
    TypeEither,
    TypeLiteral,
    TypePair,
    TypeUnit,
)

from unialg.syntax import expressions as expr


pytestmark = pytest.mark.semantics


def test_pretty_rejects_base_expression_nodes():
    with pytest.raises(ValueError):
        expr.pretty(expr.PolyExpr())

    with pytest.raises(ValueError):
        expr.pretty(expr.MorphismExpr())


def test_pretty_polynomial_constants_and_variable():
    assert expr.pretty(expr.Zero()) == "0"
    assert expr.pretty(expr.One()) == "1"
    assert expr.pretty(expr.Id()) == "X"


def test_pretty_polynomial_composites():
    int32 = TypeLiteral(LiteralTypeInteger(IntegerType.INT32))
    body = expr.Prod(
        expr.Sum(expr.One(), expr.Id()),
        expr.Sum(expr.Const(int32), expr.Id()),
    )

    assert "int32" in expr.pretty(body)
    assert expr.pretty(body).startswith("(1 + X) * (int32")
    assert expr.pretty(body).endswith(" + X)")


def test_pretty_basic_morphism_nodes():
    unit = TypeUnit()
    product = TypePair(PairType(unit, unit))
    either = TypeEither(EitherType(unit, unit))

    assert expr.pretty(expr.Identity(unit)) == "id"
    assert expr.pretty(expr.Copy(unit)) == "copy"
    assert expr.pretty(expr.Delete(unit)) == "!"
    assert expr.pretty(expr.First(product)) == "π₁"
    assert expr.pretty(expr.Second(product)) == "π₂"
    assert expr.pretty(expr.Left(either)) == "ι₁"
    assert expr.pretty(expr.Right(either)) == "ι₂"


def test_pretty_contextual_morphism_nodes():
    unit = TypeUnit()
    f = expr.Identity(unit)
    g = expr.Copy(unit)

    composed = expr.Compose(f, g, unit, unit, unit, None, unit, unit)
    paired = expr.Pair(f, g, unit, unit, unit, None, unit, unit)

    assert expr.pretty(composed) == "(id ; copy)"
    assert expr.pretty(paired) == "⟨id, copy⟩"
