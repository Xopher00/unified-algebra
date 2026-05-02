"""Test algebra_hom smart constructor: shapes, validation, summand_domain."""
import pytest

import hydra.core as core
import hydra.dsl.terms as Terms

from unialg import Semiring, Sort
from unialg.morphism._typed_morphism import TypedMorphism as T
from unialg.morphism.functor import Functor, sum_, prod, one, id_, const
from unialg.morphism.algebra_hom import summand_domain
import unialg.morphism as morphism


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)

@pytest.fixture
def carrier(real_sr):
    return Sort("carrier", real_sr)

@pytest.fixture
def base(real_sr):
    return Sort("base", real_sr)

@pytest.fixture
def list_functor(base):
    return Functor("F_list", sum_(one(), prod(const(base), id_())))

@pytest.fixture
def maybe_functor():
    return Functor("F_maybe", sum_(one(), id_()))


def _lit(value, sort):
    v = core.TermLiteral(core.LiteralFloat(core.FloatValueFloat32(value)))
    return morphism.lit(v, sort)


class TestListAlgebra:

    def test_returns_typed_morphism(self, list_functor, carrier, base):
        init = _lit(0.0, carrier)
        cons = T(Terms.identity(), T.product(base, carrier), carrier)
        m = morphism.algebra_hom(list_functor, "algebra", [init, cons])
        assert isinstance(m, T)

    def test_domain_is_list_type(self, list_functor, carrier, base):
        init = _lit(0.0, carrier)
        cons = T(Terms.identity(), T.product(base, carrier), carrier)
        m = morphism.algebra_hom(list_functor, "algebra", [init, cons])
        assert isinstance(m.domain, core.TypeList)
        assert m.codomain is carrier

    def test_term_is_lambda(self, list_functor, carrier, base):
        init = _lit(0.0, carrier)
        cons = T(Terms.identity(), T.product(base, carrier), carrier)
        m = morphism.algebra_hom(list_functor, "algebra", [init, cons])
        assert isinstance(m.term, core.TermLambda)


class TestMaybeAlgebra:

    def test_domain_is_maybe_type(self, maybe_functor, carrier):
        nothing = _lit(0.0, carrier)
        just = T(Terms.identity(), carrier, carrier)
        m = morphism.algebra_hom(maybe_functor, "algebra", [nothing, just])
        assert isinstance(m.domain, core.TypeMaybe)
        assert m.codomain is carrier

    def test_term_is_lambda(self, maybe_functor, carrier):
        nothing = _lit(0.0, carrier)
        just = T(Terms.identity(), carrier, carrier)
        m = morphism.algebra_hom(maybe_functor, "algebra", [nothing, just])
        assert isinstance(m.term, core.TermLambda)


class TestCoalgebra:

    def test_returns_step_unchanged(self, list_functor, carrier):
        step = T(Terms.identity(), carrier, carrier)
        result = morphism.algebra_hom(list_functor, "coalgebra", [step])
        assert result is step

    def test_multiple_morphisms_rejected(self, list_functor, carrier):
        step = T(Terms.identity(), carrier, carrier)
        with pytest.raises(ValueError, match="expected 1 morphism"):
            morphism.algebra_hom(list_functor, "coalgebra", [step, step])


class TestValidation:

    def test_wrong_summand_count(self, list_functor, carrier):
        init = _lit(0.0, carrier)
        with pytest.raises(ValueError, match="summand count"):
            morphism.algebra_hom(list_functor, "algebra", [init])

    def test_mismatched_codomains(self, list_functor, carrier, base):
        init = _lit(0.0, carrier)
        cons = T(Terms.identity(), T.product(base, carrier), base)
        with pytest.raises(TypeError, match="codomain"):
            morphism.algebra_hom(list_functor, "algebra", [init, cons])

    def test_init_must_be_lit(self, list_functor, carrier, base):
        not_lit = T(Terms.identity(), T.unit(), carrier)
        cons = T(Terms.identity(), T.product(base, carrier), carrier)
        with pytest.raises(ValueError, match="lit"):
            morphism.algebra_hom(list_functor, "algebra", [not_lit, cons])

    def test_cons_domain_mismatch(self, list_functor, carrier, base):
        init = _lit(0.0, carrier)
        bad_cons = T(Terms.identity(), carrier, carrier)
        with pytest.raises(TypeError, match="cons.domain"):
            morphism.algebra_hom(list_functor, "algebra", [init, bad_cons])

    def test_invalid_direction(self, list_functor, carrier):
        with pytest.raises(ValueError, match="direction"):
            morphism.algebra_hom(list_functor, "invalid", [_lit(0.0, carrier)])

    def test_empty_morphisms(self, list_functor):
        with pytest.raises(ValueError, match="at least one"):
            morphism.algebra_hom(list_functor, "algebra", [])


class TestSummandDomain:

    def test_one_gives_unit(self):
        from unialg.morphism.functor import one as poly_one
        assert isinstance(summand_domain(poly_one(), "carrier"), core.TypeUnit)

    def test_id_gives_carrier(self, carrier):
        from unialg.morphism.functor import id_ as poly_id
        assert summand_domain(poly_id(), carrier) is carrier

    def test_const_gives_sort(self, base):
        from unialg.morphism.functor import const as poly_const
        result = summand_domain(poly_const(base), "unused")
        assert result.name == base.name

    def test_prod_gives_product(self, carrier, base):
        result = summand_domain(prod(const(base), id_()), carrier)
        assert isinstance(result, core.TypePair)
