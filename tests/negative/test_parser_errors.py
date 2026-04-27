"""Parser negative tests: error cases that must raise ValueError or SyntaxError."""

import pytest

from unialg import (
    NumpyBackend, parse_ua, parse_ua_spec,
)


# ---------------------------------------------------------------------------
# Unknown sort name raises ValueError
# ---------------------------------------------------------------------------

class TestUnknownSortError:

    def test_unknown_sort_in_equation(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op bad : unknown -> hidden
  nonlinearity = relu
"""
        with pytest.raises(ValueError, match="Unknown spec 'unknown'"):
            parse_ua_spec(text)

    def test_error_message_lists_known_sorts(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec visible(real)

op bad : missing_sort -> visible
  nonlinearity = relu
"""
        with pytest.raises(ValueError, match="visible"):
            parse_ua_spec(text)

    def test_unknown_equation_in_path(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

seq bad_path : hidden -> hidden = nonexistent_eq
"""
        with pytest.raises(ValueError, match="op 'nonexistent_eq' not found"):
            parse_ua(text, NumpyBackend())

    def test_unknown_semiring_in_sort(self):
        text = """
spec hidden(nonexistent_semiring)
"""
        with pytest.raises(ValueError, match="Unknown algebra 'nonexistent_semiring'"):
            parse_ua_spec(text)


# ---------------------------------------------------------------------------
# Bad syntax raises SyntaxError
# ---------------------------------------------------------------------------

class TestSyntaxErrors:

    def test_missing_closing_paren(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0
"""
        with pytest.raises(SyntaxError):
            parse_ua_spec(text)

    def test_unrecognised_keyword_at_top_level(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
unknown_keyword foo
"""
        with pytest.raises(SyntaxError):
            parse_ua_spec(text)

    def test_missing_arrow_in_equation(self):
        """op without -> in signature should raise SyntaxError."""
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op bad : hidden hidden
  nonlinearity = relu
"""
        with pytest.raises(SyntaxError):
            parse_ua_spec(text)


# ---------------------------------------------------------------------------
# Lens-fan error cases
# ---------------------------------------------------------------------------

_LENS_FAN_BASE = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op fwd1 : hidden -> hidden
  nonlinearity = relu

op bwd1 : hidden -> hidden
  nonlinearity = relu

op fwd2 : hidden -> hidden
  nonlinearity = tanh

op bwd2 : hidden -> hidden
  nonlinearity = tanh

op merge_fwd : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

op merge_bwd : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

lens backprop1 : hidden <-> hidden
  fwd = fwd1
  bwd = bwd1

lens backprop2 : hidden <-> hidden
  fwd = fwd2
  bwd = bwd2

lens merge_lens : hidden <-> hidden
  fwd = merge_fwd
  bwd = merge_bwd
"""


class TestLensFanErrors:

    def test_lens_fan_unknown_lens_error(self):
        text = _LENS_FAN_BASE + """
lens_branch attention : hidden <-> hidden = backprop1 | nonexistent_lens
  merge = merge_lens
"""
        with pytest.raises(ValueError, match="Unknown lens 'nonexistent_lens'"):
            parse_ua_spec(text)

    def test_lens_fan_unknown_merge_error(self):
        text = _LENS_FAN_BASE + """
lens_branch attention : hidden <-> hidden = backprop1 | backprop2
  merge = nonexistent_merge
"""
        with pytest.raises(ValueError, match="Unknown lens 'nonexistent_merge'"):
            parse_ua_spec(text)
