"""Fixpoint unit tests: semiring residual field and backend axis-aware operations."""

import numpy as np
import pytest

from unialg import NumpyBackend, Semiring, Sort


# ===========================================================================
# Semiring residual field
# ===========================================================================

class TestSemiringResidualField:
    """Semiring() residual kwarg and resolve_semiring residual_elementwise extraction."""

    def test_semiring_with_residual_creates_record_with_residual_field(self):
        """Semiring(..., residual='divide') exposes residual via the .residual property."""
        sr = Semiring("real_res", plus="add", times="multiply",
                      residual="divide", zero=0.0, one=1.0)
        assert sr.residual == "divide"

    def test_resolve_semiring_extracts_residual_elementwise(self, backend):
        """Semiring.resolve extracts residual_elementwise as the divide callable."""
        sr = Semiring("real_res2", plus="add", times="multiply",
                      residual="divide", zero=0.0, one=1.0)
        rsr = sr.resolve(backend)
        assert rsr.residual_name == "divide"
        assert rsr.residual_elementwise is not None
        result = rsr.residual_elementwise(
            np.array([6.0]), np.array([3.0])
        )
        np.testing.assert_allclose(result, [2.0])

    def test_resolve_semiring_without_residual_gives_none(self, backend):
        """Semiring.resolve with no residual gives residual_elementwise=None."""
        sr = Semiring("real_nores", plus="add", times="multiply", zero=0.0, one=1.0)
        rsr = sr.resolve(backend)
        assert rsr.residual_name is None
        assert rsr.residual_elementwise is None

    def test_residual_operation_real_semiring_divide(self, backend):
        """Real semiring residual 'divide': residual_elementwise(a, c) = a / c."""
        sr = Semiring("real_div", plus="add", times="multiply",
                      residual="divide", zero=0.0, one=1.0)
        rsr = sr.resolve(backend)
        a = np.array([8.0, 12.0, 25.0])
        c = np.array([2.0, 4.0, 5.0])
        result = rsr.residual_elementwise(a, c)
        np.testing.assert_allclose(result, a / c)

    def test_residual_operation_tropical_semiring_subtract(self, backend):
        """Tropical semiring residual 'subtract': residual_elementwise(a, c) = a - c."""
        sr = Semiring("tropical_res", plus="minimum", times="add",
                      residual="subtract", zero=float("inf"), one=0.0)
        rsr = sr.resolve(backend)
        a = np.array([10.0, 5.0, 9.0])
        c = np.array([3.0, 1.0, 7.0])
        result = rsr.residual_elementwise(a, c)
        np.testing.assert_allclose(result, a - c)


# ===========================================================================
# Backend axis-aware softmax and where
# ===========================================================================

class TestBackendAxisAwareOps:
    """NumpyBackend softmax over last axis, custom axis-0 softmax, and where."""

    def test_softmax_normalizes_over_last_axis(self, backend):
        """NumpyBackend 'softmax' normalizes each row of a 2D input (rows sum to 1)."""
        x = np.array([[1.0, 2.0, 3.0],
                       [0.0, 0.0, 0.0],
                       [-1.0, 0.0, 1.0]])
        softmax_fn = backend.unary("softmax")
        result = softmax_fn(x)
        assert result.shape == x.shape
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-6)
        np.testing.assert_array_less(result[0, 0], result[0, 1])
        np.testing.assert_array_less(result[0, 1], result[0, 2])

    def test_custom_axis0_softmax_registered_and_works(self, backend):
        """A custom axis-0 softmax via functools.partial works when registered."""
        import functools
        from scipy.special import softmax as scipy_softmax
        backend.unary_ops["softmax_axis0"] = functools.partial(scipy_softmax, axis=0)
        x = np.array([[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0]])
        softmax_ax0 = backend.unary("softmax_axis0")
        result = softmax_ax0(x)
        assert result.shape == x.shape
        col_sums = result.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(2), atol=1e-6)

    def test_backend_where_exists_and_fills_masked_values(self, backend):
        """backend.where(mask, x, fill) returns filled values where mask is False."""
        assert backend.where is not None, "NumpyBackend must expose 'where'"
        x = np.array([1.0, 2.0, 3.0, 4.0])
        fill = np.array([-999.0, -999.0, -999.0, -999.0])
        mask = np.array([True, False, True, False])
        result = backend.where(mask, x, fill)
        np.testing.assert_allclose(result, [1.0, -999.0, 3.0, -999.0])
