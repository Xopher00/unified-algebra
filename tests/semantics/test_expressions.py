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


# ---------------------------------------------------------------------------
# TypeVariable in signature — param-aware type derivation
# ---------------------------------------------------------------------------

def test_signature_declared_param_returns_type_variable():
    from unialg.semantics.morphisms import signature
    from unialg.objects import TypeVariable, Name
    for name in ("theta", "bias", "scale", "p", "x", "id"):
        dom, cod = signature(expr.Ref(name), frozenset({name}))
        assert dom == TypeVariable(Name(name))
        assert cod == TypeVariable(Name(name))


def test_signature_undeclared_ref_raises():
    from unialg.semantics.morphisms import signature, MorphismError
    import pytest
    with pytest.raises(MorphismError, match="unresolved reference"):
        signature(expr.Ref("typo"), frozenset({"theta"}))


def test_signature_ref_without_param_names_raises():
    from unialg.semantics.morphisms import signature, MorphismError
    import pytest
    with pytest.raises(MorphismError, match="unresolved reference"):
        signature(expr.Ref("theta"))


def test_type_variable_tracked_by_free_variables_in_type():
    from unialg.objects import TypeVariable, Name
    import hydra.variables as V
    tv = TypeVariable(Name("theta"))
    free = V.free_variables_in_type(tv)
    assert Name("theta") in free


def test_morphism_api_with_param_ref():
    """Build a Morphism from a Ref node with TypeVariable signature."""
    from unialg.semantics.morphisms import Morphism, signature
    from unialg.objects import TypeVariable, Name
    ref = expr.Ref("theta")
    dom, cod = signature(ref, frozenset({"theta"}))
    m = Morphism(node=ref, param=TypeVariable(Name("theta")))
    assert m.param == TypeVariable(Name("theta"))
