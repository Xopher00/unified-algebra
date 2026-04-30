"""Test morphism smart constructors: term shapes and boundary types."""
import pytest
import hydra.core as core

from unialg import NumpyBackend, Semiring, Sort
from unialg.assembly._typed_morphism import TypedMorphism
import unialg.assembly.morphism as morphism


@pytest.fixture
def backend():
    return NumpyBackend()

@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)

@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)

@pytest.fixture
def base_sort(real_sr):
    return Sort("base", real_sr)


class TestIden:
    def test_returns_typed_morphism(self, hidden):
        m = morphism.iden(hidden)
        assert isinstance(m, TypedMorphism)

    def test_domain_equals_codomain(self, hidden):
        m = morphism.iden(hidden)
        assert m.domain is hidden
        assert m.codomain is hidden

    def test_term_is_identity_lambda(self, hidden):
        m = morphism.iden(hidden)
        assert isinstance(m.term, core.TermLambda)
        param = m.term.value.parameter.value
        body = m.term.value.body
        assert isinstance(body, core.TermVariable)
        assert body.value.value == param

    def test_rejects_non_sort(self):
        with pytest.raises(TypeError, match="expected Type"):
            morphism.iden("not a sort")


class TestCopy:
    def test_domain_and_codomain(self, hidden):
        m = morphism.copy(hidden)
        assert m.domain is hidden
        assert isinstance(m.codomain_type, core.TypePair)

    def test_term_is_pair_lambda(self, hidden):
        m = morphism.copy(hidden)
        assert isinstance(m.term, core.TermLambda)
        body = m.term.value.body
        assert isinstance(body, core.TermPair)


class TestDelete:
    def test_codomain_is_unit(self, hidden):
        m = morphism.delete(hidden)
        assert m.domain is hidden
        assert isinstance(m.codomain_type, core.TypeUnit)

    def test_term_is_constant_unit(self, hidden):
        m = morphism.delete(hidden)
        assert isinstance(m.term, core.TermLambda)
        assert m.term.value.parameter.value == "_"
        assert isinstance(m.term.value.body, core.TermUnit)


class TestLit:
    def test_domain_is_unit(self, hidden):
        v = core.TermLiteral(core.LiteralFloat(core.FloatValueFloat32(1.0)))
        m = morphism.lit(v, hidden)
        assert isinstance(m.domain_type, core.TypeUnit)
        assert m.codomain is hidden

    def test_term_is_constant_lambda(self, hidden):
        v = core.TermLiteral(core.LiteralFloat(core.FloatValueFloat32(1.0)))
        m = morphism.lit(v, hidden)
        assert isinstance(m.term, core.TermLambda)
        assert m.term.value.parameter.value == "_"


class TestEq:
    def test_returns_typed_morphism(self, hidden, base_sort):
        m = morphism.eq("foo", domain=hidden, codomain=base_sort)
        assert isinstance(m, TypedMorphism)
        assert m.domain is hidden
        assert m.codomain is base_sort

    def test_term_is_variable(self, hidden):
        m = morphism.eq("foo", domain=hidden, codomain=hidden)
        assert isinstance(m.term, core.TermVariable)
        assert m.term.value.value == "ua.equation.foo"

    def test_rejects_empty_name(self, hidden):
        with pytest.raises(TypeError, match="non-empty str"):
            morphism.eq("", domain=hidden, codomain=hidden)


class TestSeq:
    def test_domain_and_codomain(self, hidden, base_sort):
        f = morphism.eq("f", domain=hidden, codomain=hidden)
        g = morphism.eq("g", domain=hidden, codomain=base_sort)
        m = morphism.seq(f, g)
        assert m.domain is hidden
        assert m.codomain is base_sort

    def test_term_is_compose_lambda(self, hidden):
        f = morphism.eq("f", domain=hidden, codomain=hidden)
        g = morphism.eq("g", domain=hidden, codomain=hidden)
        m = morphism.seq(f, g)
        assert isinstance(m.term, core.TermLambda)
        assert m.term.value.parameter.value == "arg_"

    def test_mismatched_sorts_rejected(self, hidden, base_sort):
        f = morphism.eq("f", domain=hidden, codomain=hidden)
        g = morphism.eq("g", domain=base_sort, codomain=base_sort)
        with pytest.raises(TypeError):
            morphism.seq(f, g)

    def test_non_morphism_rejected(self, hidden):
        f = morphism.iden(hidden)
        with pytest.raises(TypeError):
            morphism.seq(f, "not a morphism")


class TestPar:
    def test_domain_and_codomain_are_products(self, hidden, base_sort):
        f = morphism.eq("f", domain=hidden, codomain=hidden)
        g = morphism.eq("g", domain=base_sort, codomain=base_sort)
        m = morphism.par(f, g)
        assert isinstance(m.domain_type, core.TypePair)
        assert isinstance(m.codomain_type, core.TypePair)

    def test_term_is_application(self, hidden):
        f = morphism.iden(hidden)
        g = morphism.iden(hidden)
        m = morphism.par(f, g)
        assert isinstance(m.term, core.TermApplication)
