"""End-to-end tests for lens, prism, and traversal optic construction."""
import pytest
from hydra.core import LiteralTypeInteger, IntegerType, TypeLiteral
import hydra.dsl.meta.phantoms as P

from unialg.syntax import expressions as expr
from unialg.semantics.morphisms import Morphism, MorphismError
from unialg.semantics.optics import lens_optic, prism_optic
from unialg.objects import ProductType, SumType

INT = TypeLiteral(LiteralTypeInteger(IntegerType.INT32))
PAIR = ProductType(INT, INT)  # type: ignore[arg-type]
EITHER = SumType(INT, INT)  # type: ignore[arg-type]

# Use identity-term placeholders — tests check structure, not execution.
_id = P.lam("x", P.var("x")).value

FWD_LENS = Morphism(expr.Prim(_id, INT, PAIR))  # type: ignore[arg-type]
BWD_LENS = Morphism(expr.Prim(_id, PAIR, INT))  # type: ignore[arg-type]
FWD_PRISM = Morphism(expr.Prim(_id, INT, EITHER))  # type: ignore[arg-type]
BWD_PRISM = Morphism(expr.Prim(_id, EITHER, INT))  # type: ignore[arg-type]


class TestLensOptic:
    def test_lens_kind_recorded(self):
        optic = lens_optic("test", FWD_LENS, BWD_LENS)
        assert optic.kind == "lens"

    def test_lens_carrier(self):
        optic = lens_optic("test", FWD_LENS, BWD_LENS)
        assert optic.carrier == INT

    def test_lens_focus(self):
        optic = lens_optic("test", FWD_LENS, BWD_LENS)
        assert optic.focus == INT

    def test_lens_functor_body_shape(self):
        optic = lens_optic("test", FWD_LENS, BWD_LENS)
        assert isinstance(optic.functor.body, expr.Prod)
        assert isinstance(optic.functor.body.left, expr.Id)
        assert isinstance(optic.functor.body.right, expr.Const)

    def test_lens_wrong_forward_cod_raises(self):
        bad_fwd = Morphism(expr.Prim(_id, INT, INT))
        with pytest.raises(MorphismError):
            lens_optic("bad", bad_fwd, BWD_LENS)

    def test_lens_backward_mismatch_raises(self):
        bad_bwd = Morphism(expr.Prim(_id, PAIR, PAIR))
        with pytest.raises(MorphismError):
            lens_optic("bad", FWD_LENS, bad_bwd)


class TestPrismOptic:
    def test_prism_kind_recorded(self):
        optic = prism_optic("test", FWD_PRISM, BWD_PRISM)
        assert optic.kind == "prism"

    def test_prism_carrier(self):
        optic = prism_optic("test", FWD_PRISM, BWD_PRISM)
        assert optic.carrier == INT

    def test_prism_functor_body_shape(self):
        optic = prism_optic("test", FWD_PRISM, BWD_PRISM)
        assert isinstance(optic.functor.body, expr.Sum)
        assert isinstance(optic.functor.body.left, expr.Id)
        assert isinstance(optic.functor.body.right, expr.Const)

    def test_prism_wrong_forward_cod_raises(self):
        bad_fwd = Morphism(expr.Prim(_id, INT, INT))
        with pytest.raises(MorphismError):
            prism_optic("bad", bad_fwd, BWD_PRISM)


class TestDefaultKind:
    def test_recursive_carrier_optic_has_optic_kind(self):
        from unialg.semantics.optics import recursive_carrier
        from unialg.semantics.functors import Functor
        f = Functor("NatF", expr.Sum(expr.One(), expr.Id()))
        carrier = recursive_carrier("Nat", f)
        assert carrier.optic().kind == "optic"

    def test_composed_optic_inherits_kind(self):
        outer = lens_optic("outer", FWD_LENS, BWD_LENS)
        inner = lens_optic("inner", FWD_LENS, BWD_LENS)
        composed = outer.compose(inner)
        assert composed.kind == "lens"
