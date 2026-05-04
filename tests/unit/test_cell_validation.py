"""Tests for cell typing and validation.

Covers:
- TypedMorphism.infer_type() returns correct (domain_sort, codomain_sort)
- TypedMorphism.validate() passes for smart-constructed morphisms
- TypedMorphism.validate() fails for kind=None (direct construction)
- Construction-time invariant checks for each morphism kind
- lens residual_sort parameter validation
- lens_seq operand kind validation
- kind attribute is set on every smart constructor output
"""
import pytest
import numpy as np

import hydra.core as core
import hydra.dsl.terms as Terms

from unialg import NumpyBackend, Semiring, Sort
from unialg.morphism import (
    TypedMorphism,
    eq, lit, iden, copy, delete, seq, par,
    lens, lens_seq,
    algebra_hom, Functor, one, id_, sum_, const,
)
from unialg.morphism._typed_morphism import _boundary_type
from unialg.assembly.graph import build_graph


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


@pytest.fixture
def output(real_sr):
    return Sort("output", real_sr)


# ---------------------------------------------------------------------------
# infer_type()
# ---------------------------------------------------------------------------

class TestInferType:

    def test_eq_returns_domain_codomain(self, hidden, output):
        m = eq("my_op", domain=hidden, codomain=output)
        dom, cod = m.infer_type()
        assert dom is hidden
        assert cod is output

    def test_iden_domain_equals_codomain(self, hidden):
        m = iden(hidden)
        dom, cod = m.infer_type()
        assert dom is hidden
        assert cod is hidden

    def test_lit_domain_is_unit(self, hidden):
        val = core.TermLiteral(core.LiteralFloat(core.FloatValueFloat32(1.0)))
        m = lit(val, hidden)
        dom, cod = m.infer_type()
        assert dom == TypedMorphism.unit()
        assert cod is hidden

    def test_copy_codomain_is_product(self, hidden):
        m = copy(hidden)
        dom, cod = m.infer_type()
        assert dom is hidden
        assert cod == TypedMorphism.product(hidden, hidden)

    def test_delete_codomain_is_unit(self, hidden):
        m = delete(hidden)
        dom, cod = m.infer_type()
        assert dom is hidden
        assert cod == TypedMorphism.unit()

    def test_seq_domain_from_left_codomain_from_right(self, hidden, output):
        f = eq("f", domain=hidden, codomain=output)
        g = iden(output)
        m = seq(f, g)
        dom, cod = m.infer_type()
        assert dom is hidden
        assert cod is output

    def test_par_product_boundaries(self, hidden, output):
        f = iden(hidden)
        g = iden(output)
        m = par(f, g)
        dom, cod = m.infer_type()
        assert dom == TypedMorphism.product(hidden, output)
        assert cod == TypedMorphism.product(hidden, output)


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

class TestValidate:

    def test_smart_constructed_morphisms_pass(self, hidden, output):
        val = core.TermLiteral(core.LiteralFloat(core.FloatValueFloat32(0.0)))
        for m in [
            eq("op", domain=hidden, codomain=output),
            iden(hidden),
            copy(hidden),
            delete(hidden),
            lit(val, hidden),
            seq(iden(hidden), iden(hidden)),
            par(iden(hidden), iden(output)),
        ]:
            assert m.validate() is m  # returns self

    def test_validate_fails_without_kind(self, hidden):
        m = TypedMorphism(Terms.identity(), hidden, hidden)
        assert m.kind is None
        with pytest.raises(ValueError, match="no kind set"):
            m.validate()


# ---------------------------------------------------------------------------
# kind attribute
# ---------------------------------------------------------------------------

class TestKindAttribute:

    def test_eq_kind(self, hidden, output):
        assert eq("op", domain=hidden, codomain=output).kind == "eq"

    def test_iden_kind(self, hidden):
        assert iden(hidden).kind == "iden"

    def test_copy_kind(self, hidden):
        assert copy(hidden).kind == "copy"

    def test_delete_kind(self, hidden):
        assert delete(hidden).kind == "delete"

    def test_lit_kind(self, hidden):
        val = core.TermLiteral(core.LiteralFloat(core.FloatValueFloat32(0.0)))
        assert lit(val, hidden).kind == "lit"

    def test_seq_kind(self, hidden):
        assert seq(iden(hidden), iden(hidden)).kind == "seq"

    def test_par_kind(self, hidden, output):
        assert par(iden(hidden), iden(output)).kind == "par"

    def test_lens_kind(self, hidden, output, real_sr):
        residual = Sort("residual", real_sr)
        prod_fwd = Sort("prod_fwd", real_sr)
        from unialg.algebra.sort import ProductSort
        fwd = eq("fwd", domain=hidden, codomain=ProductSort([residual, prod_fwd]))
        bwd = eq("bwd", domain=ProductSort([residual, prod_fwd]), codomain=output)
        l = lens(fwd, bwd)
        assert l.kind == "lens"

    def test_cata_kind(self, hidden):
        f_maybe = Functor("F_maybe", sum_(one(), id_()))
        nothing = lit(
            core.TermLiteral(core.LiteralFloat(core.FloatValueFloat32(0.0))),
            hidden,
        )
        just = iden(hidden)
        m = algebra_hom(f_maybe, "algebra", [nothing, just])
        assert m.kind == "cata"

    def test_ana_kind(self, hidden):
        f_maybe = Functor("F_maybe", sum_(one(), id_()))
        step = iden(hidden)
        m = algebra_hom(f_maybe, "coalgebra", [step])
        assert m.kind == "ana"


# ---------------------------------------------------------------------------
# Construction-time invariant checks
# ---------------------------------------------------------------------------

class TestSeqValidation:

    def test_seq_type_mismatch_raises(self, hidden, output):
        f = eq("f", domain=hidden, codomain=hidden)
        g = eq("g", domain=output, codomain=output)
        with pytest.raises(TypeError, match="seq.left.codomain"):
            seq(f, g)

    def test_seq_requires_typed_morphism(self, hidden):
        with pytest.raises(TypeError, match="seq.left"):
            seq("not_a_morphism", iden(hidden))


class TestLensValidation:

    def test_lens_residual_sort_mismatch_raises(self, hidden, output, real_sr):
        residual = Sort("residual", real_sr)
        wrong_residual = Sort("wrong", real_sr)
        from unialg.algebra.sort import ProductSort
        fwd = eq("fwd", domain=hidden, codomain=ProductSort([residual, output]))
        bwd = eq("bwd", domain=ProductSort([residual, output]), codomain=hidden)
        with pytest.raises(TypeError, match="lens.residual_sort"):
            lens(fwd, bwd, residual_sort=wrong_residual)

    def test_lens_forward_not_product_raises(self, hidden, output):
        fwd = eq("fwd", domain=hidden, codomain=output)
        bwd = eq("bwd", domain=output, codomain=hidden)
        with pytest.raises(TypeError, match="lens.forward.codomain"):
            lens(fwd, bwd)


class TestLensSeqValidation:

    def test_lens_seq_rejects_non_lens_l1(self, hidden, output, real_sr):
        residual = Sort("residual", real_sr)
        from unialg.algebra.sort import ProductSort
        fwd = eq("fwd", domain=hidden, codomain=ProductSort([residual, output]))
        bwd = eq("bwd", domain=ProductSort([residual, output]), codomain=hidden)
        good_lens = lens(fwd, bwd)
        bad = eq("op", domain=hidden, codomain=output)
        with pytest.raises(TypeError, match="lens_seq.l1.*kind='lens'"):
            lens_seq(bad, good_lens)

    def test_lens_seq_rejects_non_lens_l2(self, hidden, output, real_sr):
        residual = Sort("residual", real_sr)
        from unialg.algebra.sort import ProductSort
        fwd = eq("fwd", domain=hidden, codomain=ProductSort([residual, output]))
        bwd = eq("bwd", domain=ProductSort([residual, output]), codomain=hidden)
        good_lens = lens(fwd, bwd)
        bad = eq("op", domain=hidden, codomain=output)
        with pytest.raises(TypeError, match="lens_seq.l2.*kind='lens'"):
            lens_seq(good_lens, bad)
