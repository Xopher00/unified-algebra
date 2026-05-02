"""Parser coverage for the current cell and functor expression syntax."""
import pytest

import hydra.core as core

from unialg.morphism._typed_morphism import TypedMorphism
from unialg.parser import NamedCell
from unialg.parser import parse_ua, parse_ua_spec


def _assert_var(term, name: str):
    assert isinstance(term, core.TermVariable)
    assert term.value.value == name


def _seq_parts(term):
    """Return (f, g, arg) from lambda x. g(f(x))."""
    if isinstance(term, TypedMorphism):
        term = term.term
    assert isinstance(term, core.TermLambda)
    body = term.value.body
    assert isinstance(body, core.TermApplication)
    inner = body.value.argument
    assert isinstance(inner, core.TermApplication)
    _assert_var(inner.value.argument, term.value.parameter.value)
    return inner.value.function, body.value.function, inner.value.argument


def _par_parts(term):
    """Return (f, g) from hydra.lib.pairs.bimap f g."""
    if isinstance(term, TypedMorphism):
        term = term.term
    assert isinstance(term, core.TermApplication)
    inner = term.value.function
    assert isinstance(inner, core.TermApplication)
    _assert_var(inner.value.function, "hydra.lib.pairs.bimap")
    return inner.value.argument, term.value.argument


def _field_names(record_term):
    assert isinstance(record_term, core.TermRecord)
    return [field.name.value for field in record_term.value.fields]


class TestCellDSLParse:
    _BASE = """
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op f : hidden -> hidden
  einsum = "i,i->i"
  algebra = real
op g : hidden -> hidden
  einsum = "i,i->i"
  algebra = real
op plus : hidden -> hidden
  einsum = "i,i->i"
  algebra = real
"""

    def test_seq(self):
        spec = parse_ua_spec(self._BASE + "cell foo : hidden -> hidden = f > g\n")
        assert len(spec.cells) == 1
        nc = spec.cells[0]
        assert isinstance(nc, NamedCell)
        assert nc.name == "foo"
        assert isinstance(nc.cell, TypedMorphism)
        f, g, _ = _seq_parts(nc.cell)
        _assert_var(f, "ua.equation.f")
        _assert_var(g, "ua.equation.g")

    def test_par(self):
        spec = parse_ua_spec(self._BASE + "cell bar : hidden -> hidden = f & g\n")
        f, g = _par_parts(spec.cells[0].cell)
        _assert_var(f, "ua.equation.f")
        _assert_var(g, "ua.equation.g")

    def test_copy_iden_delete_boundary_error(self):
        src = self._BASE + (
            "cell c : hidden -> hidden = copy[hidden] > (f & id[hidden]) > plus\n"
        )
        with pytest.raises(TypeError, match="seq.left.codomain"):
            parse_ua_spec(src)

    def test_symbolic_copy_iden_delete_boundary_error(self):
        src = self._BASE + (
            "cell c : hidden -> hidden = ^[hidden] > (f & _[hidden]) > plus\n"
        )
        with pytest.raises(TypeError, match="seq.left.codomain"):
            parse_ua_spec(src)

    def test_lens_height1_syntax_reaches_typed_lens(self):
        src = self._BASE + "cell o : hidden -> hidden = f ~ g\n"
        with pytest.raises(TypeError, match="TypePair"):
            parse_ua_spec(src)

    def test_lens_height2_with_residual_annotation(self):
        src = self._BASE + (
            "cell o : hidden -> (hidden, hidden) = "
            "copy[hidden] ~ (id[hidden] & id[hidden]) *[hidden]\n"
        )
        cell = parse_ua_spec(src).cells[0].cell
        assert isinstance(cell, TypedMorphism)
        assert "forward" in _field_names(cell.term)
        assert "backward" in _field_names(cell.term)
        assert "residualSort" in _field_names(cell.term)
        assert isinstance(cell.codomain_type, core.TypePair)

    def test_left_assoc_seq(self):
        src = self._BASE + "cell foo : hidden -> hidden = f > g > plus\n"
        cell = parse_ua_spec(src).cells[0].cell
        left, plus, _ = _seq_parts(cell)
        _assert_var(plus, "ua.equation.plus")
        f, g, _ = _seq_parts(left)
        _assert_var(f, "ua.equation.f")
        _assert_var(g, "ua.equation.g")

    def test_paren_grouping(self):
        src = self._BASE + "cell foo : hidden -> hidden = (f > g) & plus\n"
        cell = parse_ua_spec(src).cells[0].cell
        seq, plus = _par_parts(cell)
        _assert_var(plus, "ua.equation.plus")
        f, g, _ = _seq_parts(seq)
        _assert_var(f, "ua.equation.f")
        _assert_var(g, "ua.equation.g")


class TestCellDSLPrecedence:
    _BASE = TestCellDSLParse._BASE + """
op h : hidden -> hidden
  einsum = "i,i->i"
  algebra = real
"""

    def test_product_binds_tighter_than_sequence(self):
        with pytest.raises(TypeError, match="seq.left.codomain"):
            parse_ua_spec(
                self._BASE + "cell foo : hidden -> hidden = f > g & h\n"
            )

    def test_grouped_sequence_can_be_product_operand(self):
        spec = parse_ua_spec(
            self._BASE + "cell foo : hidden -> hidden = (f > g) & h\n"
        )
        left, h = _par_parts(spec.cells[0].cell)
        _assert_var(h, "ua.equation.h")
        f, g, _ = _seq_parts(left)
        _assert_var(f, "ua.equation.f")
        _assert_var(g, "ua.equation.g")

    def test_lens_binds_looser_than_sequence(self):
        src = self._BASE + "cell o : hidden -> hidden = f > g ~ h\n"
        with pytest.raises(TypeError, match="TypePair"):
            parse_ua_spec(src)


class TestNamedCellConstructors:
    _BASE = TestCellDSLParse._BASE

    def test_named_seq(self):
        spec = parse_ua_spec(self._BASE + "cell foo : hidden -> hidden = seq(f, g)\n")
        f, g, _ = _seq_parts(spec.cells[0].cell)
        _assert_var(f, "ua.equation.f")
        _assert_var(g, "ua.equation.g")

    def test_named_par(self):
        spec = parse_ua_spec(self._BASE + "cell foo : hidden -> hidden = par(f, g)\n")
        f, g = _par_parts(spec.cells[0].cell)
        _assert_var(f, "ua.equation.f")
        _assert_var(g, "ua.equation.g")

    def test_named_id_copy_drop(self):
        spec = parse_ua_spec(
            self._BASE
            + "cell i : hidden -> hidden = id[hidden]\n"
            + "cell c : hidden -> (hidden, hidden) = copy[hidden]\n"
            + "cell d : hidden -> hidden = drop[hidden]\n"
        )
        identity, copy, drop = [nc.cell for nc in spec.cells]
        assert isinstance(identity.term, core.TermLambda)
        assert isinstance(copy.codomain_type, core.TypePair)
        assert isinstance(drop.codomain_type, core.TypeUnit)

    def test_named_lens_with_residual_annotation(self):
        src = self._BASE + (
            "cell o : hidden -> (hidden, hidden) = "
            "lens(copy[hidden], id[hidden] & id[hidden]) *[hidden]\n"
        )
        cell = parse_ua_spec(src).cells[0].cell
        assert "forward" in _field_names(cell.term)
        assert "backward" in _field_names(cell.term)
        assert "residualSort" in _field_names(cell.term)

    def test_named_constructor_arity_errors(self):
        with pytest.raises(ValueError, match=r"seq\(\) takes 2 args"):
            parse_ua_spec(self._BASE + "cell bad : hidden -> hidden = seq(f)\n")
        with pytest.raises(ValueError, match=r"lens\(\) takes 2 args"):
            parse_ua_spec(self._BASE + "cell bad : hidden -> hidden = lens(f)\n")


class TestCellDSLEndToEnd:
    def test_seq_compiles_and_runs(self, backend):
        src = """
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op f : hidden -> hidden
  einsum = "i,i->i"
  algebra = real
op g : hidden -> hidden
  einsum = "i,i->i"
  algebra = real
cell foo : hidden -> hidden = f > g
"""
        import numpy as np
        prog = parse_ua(src, backend=backend)
        assert prog is not None
        assert "foo" in prog.entry_points()

    def test_residual_decomposition_boundary_error(self, backend):
        src = """
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
define unary halve(x) = x / 2.0
op f : hidden -> hidden
  nonlinearity = halve
op plus : hidden -> hidden
  einsum = "i,i->i"
  algebra = real
cell residual_layer : hidden -> hidden = copy[hidden] > (f & id[hidden]) > plus
"""
        with pytest.raises(TypeError, match="seq.left.codomain"):
            parse_ua(src, backend=backend)


class TestFunctorDecl:
    _BASE = """
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec base(real)
spec hidden(real)
spec output(real)
"""

    def test_simple_functor(self):
        parse_ua_spec(self._BASE + "functor F_iter : X\n")

    def test_functor_product_uses_ampersand(self):
        src = self._BASE + "functor F_pair : base & X\n"
        parse_ua_spec(src)

    def test_functor_composition_stub(self):
        src = self._BASE + "functor F_comp : base @ X\n"
        with pytest.raises(NotImplementedError, match=r"composition \(@\)"):
            parse_ua_spec(src)

    def test_list_functor_with_symbolic_cata(self):
        src = self._BASE + (
            "op step : hidden -> hidden\n"
            "  einsum = \"i,i->i\"\n"
            "  algebra = real\n"
            "functor F_list : 1 + base & X\n"
            "cell my_fold : hidden -> hidden = >[F_list](0, step)\n"
        )
        nc = parse_ua_spec(src).cells[0]
        assert nc.name == "my_fold"
        assert isinstance(nc.cell, TypedMorphism)
        assert isinstance(nc.cell.term, core.TermLambda)

    def test_list_functor_with_named_fold(self):
        src = self._BASE + (
            "op step : hidden -> hidden\n"
            "  einsum = \"i,i->i\"\n"
            "  algebra = real\n"
            "functor F_list : 1 + base & X\n"
            "cell my_fold : hidden -> hidden = fold[F_list](0, step)\n"
        )
        cell = parse_ua_spec(src).cells[0].cell
        assert isinstance(cell, TypedMorphism)
        assert isinstance(cell.domain_type, core.TypeList)

    def test_unfold_named_constructor_uses_coalgebra_step(self):
        src = self._BASE + (
            "op step : hidden -> hidden\n"
            "  einsum = \"i,i->i\"\n"
            "  algebra = real\n"
            "functor F_iter : X\n"
            "cell my_unfold : hidden -> hidden = unfold[F_iter](step)\n"
        )
        cell = parse_ua_spec(src).cells[0].cell
        assert isinstance(cell, TypedMorphism)
        _assert_var(cell.term, "ua.equation.step")

    def test_tree_functor_not_supported_by_algebra_hom_yet(self):
        src = self._BASE + (
            "op leaf : hidden -> hidden\n"
            "  einsum = \"i,i->i\"\n"
            "  algebra = real\n"
            "op node : hidden -> hidden\n"
            "  einsum = \"i,i->i\"\n"
            "  algebra = real\n"
            "functor F_tree : base + X & X\n"
            "cell tree_fold : hidden -> hidden = >[F_tree](leaf, node)\n"
        )
        with pytest.raises(NotImplementedError, match="not yet supported"):
            parse_ua_spec(src)

    def test_poset_category_not_supported_by_algebra_hom_yet(self):
        src = self._BASE + (
            "op step : hidden -> hidden\n"
            "  einsum = \"i,i->i\"\n"
            "  algebra = real\n"
            "functor F_poset : X\n"
            "  category = poset\n"
            "cell tarski : hidden -> hidden = >[F_poset](step)\n"
        )
        with pytest.raises(NotImplementedError, match="not yet supported"):
            parse_ua_spec(src)

    def test_poset_with_non_id_rejected(self):
        src = self._BASE + (
            "functor F_bad : 1 + base & X\n"
            "  category = poset\n"
        )
        with pytest.raises(ValueError, match="poset requires body=X"):
            parse_ua_spec(src)

    def test_unknown_category_rejected(self):
        src = self._BASE + (
            "functor F_q : X\n"
            "  category = quantale\n"
        )
        with pytest.raises(ValueError, match="must be 'set' or 'poset'"):
            parse_ua_spec(src)

    def test_unknown_functor_in_cata_rejected(self):
        src = self._BASE + (
            "op step : hidden -> hidden\n"
            "  einsum = \"i,i->i\"\n"
            "  algebra = real\n"
            "cell oops : hidden -> hidden = >[F_ghost](0, step)\n"
        )
        with pytest.raises(ValueError, match="unknown functor 'F_ghost'"):
            parse_ua_spec(src)


class TestCellDSLErrors:
    def test_old_sequence_operator_rejected(self):
        src = TestCellDSLParse._BASE + "cell foo : hidden -> hidden = f ; g\n"
        with pytest.raises(ValueError, match="use '>' not ';'"):
            parse_ua_spec(src)

    def test_old_product_operator_rejected(self):
        src = TestCellDSLParse._BASE + "cell foo : hidden -> hidden = f * g\n"
        with pytest.raises(ValueError, match="use '&' not '\\*'"):
            parse_ua_spec(src)

    def test_old_functor_product_operator_rejected(self):
        src = TestFunctorDecl._BASE + "functor F_bad : 1 + base * X\n"
        with pytest.raises(ValueError, match="use '&' for functor product"):
            parse_ua_spec(src)

    def test_unknown_eq_in_cell_rejected(self, backend):
        src = TestCellDSLParse._BASE + "cell foo : hidden -> hidden = f > ghost\n"
        with pytest.raises(ValueError, match="unknown equation 'ghost'"):
            parse_ua(src, backend=backend)

    def test_unknown_sort_in_copy_rejected(self, backend):
        src = TestCellDSLParse._BASE + "cell foo : hidden -> hidden = ^[ghost_sort] > f\n"
        with pytest.raises(ValueError, match="(?i)unknown spec|sort"):
            parse_ua(src, backend=backend)
