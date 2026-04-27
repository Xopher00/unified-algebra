"""Fuzzy relational core stack expressed in UA.

Implements the core computation from ua_tensors/core/:
  - algebra.py  → fuzzy semiring (max / min, with smooth contraction hook)
  - tensor.py   → Join as semiring contraction "ij,jk->ik"
  - activations.py → temperature-controlled smooth ops via define

Under the fuzzy semiring (⊕ = max, ⊗ = min), the contraction "ij,jk->ik" computes:
    result[i,k] = max_j min(A[i,j], B[j,k])

This is exactly relational composition (Join).

At temperature T>0, the contraction hook replaces the hard operations with
smooth approximations (T * logaddexp). The algebra declaration stays exact
(hard max/min passes law checks); the contraction hook controls smoothness.
This is the "second order semiring" pattern: declare exact laws, override
the contraction strategy.
"""

import numpy as np
import pytest
from unialg import parse_ua, NumpyBackend
from unialg.algebra.contraction import CONTRACTION_REGISTRY


# ---------------------------------------------------------------------------
# Contraction hook factory
# ---------------------------------------------------------------------------

def _make_smooth_fuzzy_hook(T):
    """Build a contraction hook that uses smooth min/max at temperature T.

    The hook receives compute_sum(times_fn, reduce_fn) from the contraction
    engine. It substitutes smooth_min for times and a fold-based smooth_max
    for reduce, so the contraction computes:
        result[i,k] = smooth_max_j smooth_min(A[i,j], B[j,k])
    """
    def hook(compute_sum, _backend):
        def smooth_min(a, b):
            return -T * np.logaddexp(-a / T, -b / T)

        def smooth_max_reduce(tensor, axis):
            if isinstance(axis, (tuple, list)):
                result = tensor
                for ax in sorted(axis, reverse=True):
                    result = smooth_max_reduce(result, ax)
                return result
            n = tensor.shape[axis]
            idx = [slice(None)] * tensor.ndim
            idx[axis] = 0
            result = tensor[tuple(idx)]
            for i in range(1, n):
                idx[axis] = i
                result = T * np.logaddexp(result / T, tensor[tuple(idx)] / T)
            return result

        return compute_sum(smooth_min, smooth_max_reduce)
    return hook


_FUZZY_DSL = '''\
define binary hard_max(a, b) = maximum(a, b)
define binary hard_min(a, b) = minimum(a, b)
algebra fuzzy(plus=hard_max, times=hard_min, zero=-inf, one=inf{contraction})
spec relation(fuzzy)

op join : relation -> relation
  einsum = "ij,jk->ik"
  algebra = fuzzy
'''


@pytest.fixture
def fuzzy_prog():
    """Hard fuzzy program (no contraction hook)."""
    return parse_ua(_FUZZY_DSL.format(contraction=''), NumpyBackend())


def _smooth_fuzzy_prog(T):
    """Register a contraction hook at temperature T, parse, then clean up."""
    key = f'_test_smooth_fuzzy_{str(T).replace(".", "_")}'
    CONTRACTION_REGISTRY[key] = _make_smooth_fuzzy_hook(T)
    try:
        return parse_ua(
            _FUZZY_DSL.format(contraction=f', contraction={key}'),
            NumpyBackend(),
        )
    finally:
        del CONTRACTION_REGISTRY[key]


# ---------------------------------------------------------------------------
# Tests: hard fuzzy contraction
# ---------------------------------------------------------------------------

class TestHardFuzzyContraction:
    """Hard fuzzy semiring (max/min) via pure DSL syntax."""

    def test_hard_fuzzy_join(self, fuzzy_prog):
        A = np.array([[0.8, 0.2],
                      [0.1, 0.9]])
        B = np.array([[0.7, 0.3],
                      [0.4, 0.6]])

        result = fuzzy_prog('join', A, B)

        expected = np.array([
            [max(min(0.8, 0.7), min(0.2, 0.4)),
             max(min(0.8, 0.3), min(0.2, 0.6))],
            [max(min(0.1, 0.7), min(0.9, 0.4)),
             max(min(0.1, 0.3), min(0.9, 0.6))],
        ])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_two_hop_reachability(self, fuzzy_prog):
        E = np.array([[0.0, 0.9, 0.3],
                      [0.0, 0.0, 0.8],
                      [0.0, 0.0, 0.0]])

        one_hop = fuzzy_prog('join', E, E)
        assert one_hop[0, 2] == pytest.approx(0.8, abs=1e-6)
        assert one_hop[0, 0] == pytest.approx(0.0, abs=1e-6)

        two_hop = fuzzy_prog('join', one_hop, E)
        assert two_hop.shape == (3, 3)

    def test_join_is_semiring_polymorphic(self, fuzzy_prog):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])

        real_prog = parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec mat(real)
op matmul : mat -> mat
  einsum = "ij,jk->ik"
  algebra = real
''', NumpyBackend())

        fuzzy_result = fuzzy_prog('join', A, B)
        real_result = real_prog('matmul', A, B)

        np.testing.assert_allclose(real_result, A @ B, rtol=1e-6)
        assert not np.allclose(fuzzy_result, real_result), \
            "Fuzzy and real contractions must differ — semiring changes semantics"


# ---------------------------------------------------------------------------
# Tests: smooth fuzzy contraction (temperature controls the binary ops)
# ---------------------------------------------------------------------------

class TestSmoothFuzzyContraction:
    """Temperature-controlled smooth contraction via the contraction hook.

    The algebra declaration uses hard max/min (exact semiring, passes law checks).
    The contraction= field overrides HOW the contraction is computed: smooth_min
    replaces min as ⊗, smooth_max replaces max as ⊕. Temperature T controls the
    binary operations that ARE the contraction.
    """

    def test_smooth_join_matches_manual(self):
        """At T=1, the contraction hook uses smooth binary ops."""
        prog = _smooth_fuzzy_prog(T=1.0)

        A = np.array([[0.8, 0.2],
                      [0.1, 0.9]])
        B = np.array([[0.7, 0.3],
                      [0.4, 0.6]])

        result = prog('join', A, B)

        T = 1.0
        def sm(a, b):
            return -T * np.logaddexp(-a/T, -b/T)
        def sx(a, b):
            return T * np.logaddexp(a/T, b/T)

        expected = np.array([
            [sx(sm(0.8, 0.7), sm(0.2, 0.4)),
             sx(sm(0.8, 0.3), sm(0.2, 0.6))],
            [sx(sm(0.1, 0.7), sm(0.9, 0.4)),
             sx(sm(0.1, 0.3), sm(0.9, 0.6))],
        ])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_temperature_controls_contraction(self, fuzzy_prog):
        """Varying T interpolates between hard and soft binary ops."""
        A = np.array([[0.9, 0.1], [0.3, 0.7]])
        B = np.array([[0.6, 0.4], [0.2, 0.8]])

        hard = fuzzy_prog('join', A, B)
        warm = _smooth_fuzzy_prog(T=0.1)('join', A, B)
        hot  = _smooth_fuzzy_prog(T=1.0)('join', A, B)

        warm_error = np.abs(warm - hard).max()
        hot_error = np.abs(hot - hard).max()
        assert hot_error > warm_error, \
            "Higher T should produce larger deviation from hard result"
        assert warm_error > 0, "T=0.1 should differ from hard"

    def test_low_temperature_approaches_hard(self, fuzzy_prog):
        """At low T, smooth contraction converges to hard max-min."""
        A = np.array([[0.8, 0.2, 0.5],
                      [0.1, 0.9, 0.3],
                      [0.6, 0.4, 0.7]])
        B = np.array([[0.7, 0.3],
                      [0.4, 0.6],
                      [0.5, 0.8]])

        hard = fuzzy_prog('join', A, B)
        for T in [0.1, 0.01, 0.001]:
            smooth = _smooth_fuzzy_prog(T=T)('join', A, B)
            np.testing.assert_allclose(smooth, hard, atol=T * np.log(2) * 2 + 1e-6)


# ---------------------------------------------------------------------------
# Tests: smooth activations via define
# ---------------------------------------------------------------------------

class TestSmoothActivations:
    """Temperature-controlled unary activations from the define expression language."""

    def test_softplus_via_smooth_max(self):
        """Softplus(x) = SmoothMax(0, x) — max(x, 0) smoothed."""
        T = 1.0
        prog = parse_ua(f'''
define unary softplus(x) = {T} * logaddexp(0.0 / {T}, x / {T})
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op activate : hidden -> hidden
  nonlinearity = softplus
''', NumpyBackend())

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = prog('activate', x)
        expected = T * np.logaddexp(0.0, x / T)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_softplus_low_temp_approaches_relu(self):
        """At low T, softplus approaches max(x, 0) = relu."""
        T = 0.001
        prog = parse_ua(f'''
define unary softplus(x) = {T} * logaddexp(0.0 / {T}, x / {T})
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op activate : hidden -> hidden
  nonlinearity = softplus
''', NumpyBackend())

        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = prog('activate', x)
        relu = np.maximum(0, x)
        np.testing.assert_allclose(result, relu, atol=T * np.log(2) + 1e-6)
