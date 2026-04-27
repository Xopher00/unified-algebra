"""End-to-end tests for DSL `define` declarations."""

import numpy as np
import pytest
from unialg import parse_ua, NumpyBackend


class TestDefineUnary:

    def test_clamp_nonlinearity(self):
        prog = parse_ua('''
define unary clamp(x) = minimum(1.0, maximum(0.0, x))
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op activate : hidden -> hidden
  nonlinearity = clamp
''', NumpyBackend())
        x = np.array([-0.5, 0.3, 1.5])
        result = prog('activate', x)
        np.testing.assert_allclose(result, np.minimum(1.0, np.maximum(0.0, x)))

    def test_infix_leaky_relu(self):
        prog = parse_ua('''
define unary leaky(x) = maximum(0.0, x) + x * 0.01 - maximum(0.0, x) * 0.01
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op act : hidden -> hidden
  nonlinearity = leaky
''', NumpyBackend())
        x = np.array([-2.0, 0.0, 3.0])
        result = prog('act', x)
        expected = np.where(x > 0, x, 0.01 * x)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_nested_calls(self):
        prog = parse_ua('''
define unary softclamp(x) = log(1.0 + exp(minimum(5.0, maximum(-5.0, x))))
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op act : hidden -> hidden
  nonlinearity = softclamp
''', NumpyBackend())
        x = np.array([-10.0, 0.0, 10.0])
        result = prog('act', x)
        clipped = np.minimum(5.0, np.maximum(-5.0, x))
        expected = np.log(1.0 + np.exp(clipped))
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestDefineBinary:

    def test_smooth_max_as_semiring_plus(self):
        prog = parse_ua('''
define binary smooth_max(a, b) = log(exp(a) + exp(b))
algebra lse(plus=smooth_max, times=add, zero=-inf, one=0.0)
spec hidden(lse)
op step : hidden -> hidden
  einsum = "ij,j->i"
  algebra = lse
''', NumpyBackend())
        W = np.array([[0.0, -1.0], [1.0, 0.0]])
        x = np.array([2.0, 3.0])
        result = prog('step', W, x)
        expected = np.array([
            np.log(np.exp(0.0 + 2.0) + np.exp(-1.0 + 3.0)),
            np.log(np.exp(1.0 + 2.0) + np.exp(0.0 + 3.0)),
        ])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_custom_tropical_alias(self):
        # max-plus (tropical) semiring via define aliases: plus=my_max, times=my_add
        # Laws: max(max(a,-inf),b)=max(a,b), add(add(a,0),b)=add(a,b), etc. — all hold.
        prog = parse_ua('''
define binary my_max(a, b) = maximum(a, b)
define binary my_add(a, b) = a + b
algebra tropical2(plus=my_max, times=my_add, zero=-inf, one=0.0)
spec hidden(tropical2)
op step : hidden -> hidden
  einsum = "ij,j->i"
  algebra = tropical2
''', NumpyBackend())
        W = np.array([[1.0, 2.0], [3.0, 0.0]])
        x = np.array([1.0, 2.0])
        result = prog('step', W, x)
        # max-plus matvec: max_j(W_ij + x_j)
        expected = np.array([
            max(1.0 + 1.0, 2.0 + 2.0),  # max(2, 4) = 4
            max(3.0 + 1.0, 0.0 + 2.0),  # max(4, 2) = 4
        ])
        np.testing.assert_allclose(result, expected)


class TestSmoothTropicalSemiring:
    """Temperature-controlled smooth max as a semiring — the core use case from
    ua_tensors/core/activations.py.

    SmoothMax_T(a, b) = T * logaddexp(a/T, b/T)

    At T→0 this is the hard max (tropical semiring). At T=1 it's logaddexp.
    At any T>0 it's a smooth, differentiable upper bound on max with gap ≤ T*ln(2).
    """

    def test_smooth_max_t1_equals_logaddexp(self):
        prog = parse_ua('''
define binary smooth_max(a, b) = logaddexp(a, b)
algebra smooth_trop(plus=smooth_max, times=add, zero=-inf, one=0.0)
spec hidden(smooth_trop)
op step : hidden -> hidden
  einsum = "ij,j->i"
  algebra = smooth_trop
''', NumpyBackend())
        W = np.array([[0.0, -1.0], [1.0, 0.0]])
        x = np.array([2.0, 3.0])
        result = prog('step', W, x)
        expected = np.array([
            np.logaddexp(0.0 + 2.0, -1.0 + 3.0),
            np.logaddexp(1.0 + 2.0, 0.0 + 3.0),
        ])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_smooth_max_with_temperature(self):
        T = 0.5
        prog = parse_ua(f'''
define binary smooth_max_t(a, b) = {T} * logaddexp(a / {T}, b / {T})
algebra smooth_trop(plus=smooth_max_t, times=add, zero=-inf, one=0.0)
spec hidden(smooth_trop)
op step : hidden -> hidden
  einsum = "ij,j->i"
  algebra = smooth_trop
''', NumpyBackend())
        W = np.array([[0.0, -1.0], [1.0, 0.0]])
        x = np.array([2.0, 3.0])
        result = prog('step', W, x)
        expected = np.array([
            T * np.logaddexp((0.0+2.0)/T, (-1.0+3.0)/T),
            T * np.logaddexp((1.0+2.0)/T, (0.0+3.0)/T),
        ])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_low_temperature_approaches_hard_max(self):
        T = 0.001
        prog = parse_ua(f'''
define binary smooth_max_t(a, b) = {T} * logaddexp(a / {T}, b / {T})
algebra smooth_trop(plus=smooth_max_t, times=add, zero=-inf, one=0.0)
spec hidden(smooth_trop)
op step : hidden -> hidden
  einsum = "ij,j->i"
  algebra = smooth_trop
''', NumpyBackend())
        W = np.array([[1.0, 2.0], [3.0, 0.0]])
        x = np.array([1.0, 2.0])
        result = prog('step', W, x)
        hard_max = np.array([
            max(1.0 + 1.0, 2.0 + 2.0),
            max(3.0 + 1.0, 0.0 + 2.0),
        ])
        np.testing.assert_allclose(result, hard_max, atol=T * np.log(2) + 1e-6)

    def test_smooth_min_via_de_morgan(self):
        T = 1.0
        prog = parse_ua(f'''
define binary smooth_min(a, b) = -1.0 * {T} * logaddexp(a / -{T}, b / -{T})
algebra smooth_min_sr(plus=smooth_min, times=add, zero=inf, one=0.0)
spec hidden(smooth_min_sr)
op step : hidden -> hidden
  einsum = "ij,j->i"
  algebra = smooth_min_sr
''', NumpyBackend())
        W = np.array([[1.0, 2.0], [3.0, 0.0]])
        x = np.array([1.0, 2.0])
        result = prog('step', W, x)
        expected = np.array([
            -np.logaddexp(-(1.0+1.0), -(2.0+2.0)),
            -np.logaddexp(-(3.0+1.0), -(0.0+2.0)),
        ])
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestDefineBuiltinUnchanged:

    def test_builtin_ops_survive(self):
        backend = NumpyBackend()
        relu_before = backend.unary('relu')
        add_before = backend.elementwise('add')
        parse_ua('''
define unary custom_act(x) = tanh(x)
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
''', backend)
        assert backend.unary('relu') is relu_before
        assert backend.elementwise('add') is add_before
        assert 'custom_act' in backend.unary_ops
