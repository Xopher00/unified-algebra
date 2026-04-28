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

    def test_unknown_equation_in_cell(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

cell bad_cell : hidden -> hidden = nonexistent_eq
"""
        with pytest.raises(ValueError, match="unknown equation 'nonexistent_eq'"):
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


class TestUnknownBackend:

    def test_unknown_backend_raises(self):
        text = """
import fakebackend
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""
        with pytest.raises(ValueError, match="Unknown backend 'fakebackend'"):
            parse_ua(text)
