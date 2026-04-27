"""Negative tests for `define` declarations: error cases."""

import pytest
from unialg import parse_ua, parse_ua_spec, NumpyBackend


class TestDefineErrors:

    def test_unknown_function_in_define(self):
        with pytest.raises(ValueError, match="unknown unary function 'nonexistent_fn'"):
            parse_ua('''
define unary f(x) = nonexistent_fn(x)
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op act : hidden -> hidden
  nonlinearity = f
''', NumpyBackend())

    def test_wrong_param_count_unary(self):
        with pytest.raises(ValueError, match="must have exactly 1 parameter"):
            parse_ua('''
define unary f(a, b) = add(a, b)
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
''', NumpyBackend())

    def test_wrong_param_count_binary(self):
        with pytest.raises(ValueError, match="must have exactly 2 parameters"):
            parse_ua('''
define binary f(x) = exp(x)
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
''', NumpyBackend())

    def test_undeclared_variable(self):
        with pytest.raises(ValueError, match="unknown variable 'y'"):
            parse_ua('''
define unary f(x) = add(x, y)
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op act : hidden -> hidden
  nonlinearity = f
''', NumpyBackend())
