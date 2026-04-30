"""Named constructor aliases and local equation-reference modifiers."""
import pytest

import hydra.core as core

from unialg.parser import parse_ua_spec


BASE = """
import numpy
algebra residuated(plus=maximum, times=minimum, zero=0.0, one=1.0, residual=maximum)
spec hidden(residuated)
op f : hidden -> hidden
  einsum = "i,i->i"
  algebra = residuated
op g : hidden -> hidden
  einsum = "i,i->i"
  algebra = residuated
"""


def _assert_var(term, name: str):
    assert isinstance(term, core.TermVariable)
    assert term.value.value == name


def _seq_parts(term):
    term = term.term
    assert isinstance(term, core.TermLambda)
    body = term.value.body
    assert isinstance(body, core.TermApplication)
    inner = body.value.argument
    assert isinstance(inner, core.TermApplication)
    return inner.value.function, body.value.function


def test_named_seq_alias_lowers_like_sequence():
    spec = parse_ua_spec(BASE + "cell c : hidden -> hidden = seq(f, g)\n")
    f, g = _seq_parts(spec.cells[0].cell)
    _assert_var(f, "ua.equation.f")
    _assert_var(g, "ua.equation.g")


def test_adjoint_suffix_synthesizes_internal_equation_reference():
    spec = parse_ua_spec(BASE + "cell c : hidden -> hidden = f'\n")
    assert [eq.name for eq in spec.equations] == ["f", "g", "f__adjoint"]
    assert spec.equations[-1].adjoint is True
    _assert_var(spec.cells[0].cell.term, "ua.equation.f__adjoint")


def test_adjoint_suffix_composes_without_source_duplicate():
    spec = parse_ua_spec(BASE + "cell c : hidden -> hidden = f' > g\n")
    f_adj, g = _seq_parts(spec.cells[0].cell)
    _assert_var(f_adj, "ua.equation.f__adjoint")
    _assert_var(g, "ua.equation.g")


def test_adjoint_suffix_requires_einsum_backed_op():
    src = BASE + """
define unary relu(x) = x
op h : hidden -> hidden
  nonlinearity = relu
cell c : hidden -> hidden = h'
"""
    with pytest.raises(NotImplementedError, match="einsum-backed op"):
        parse_ua_spec(src)


def test_masked_suffix_is_reserved_but_not_implemented():
    with pytest.raises(NotImplementedError, match="masked references"):
        parse_ua_spec(BASE + "cell c : hidden -> hidden = f?\n")
