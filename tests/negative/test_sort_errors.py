"""Negative tests for ProductSort construction errors and semiring law counterexamples."""

import pytest

from unialg import NumpyBackend, Semiring, Sort, ProductSort


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


# Triplets of unbounded reals — work for real, tropical, max-plus, log-prob.
_REAL_SAMPLES = [
    (1.0, 2.0, 3.0), (-1.5, 0.5, 2.5),
    (0.0, 1.0, -1.0), (5.0, -2.0, 4.0),
]


# ===========================================================================
# ProductSort construction errors
# ===========================================================================

class TestProductSortErrors:
    """ProductSort([single]) and ProductSort([]) must raise ValueError."""

    def test_product_sort_requires_at_least_two_elements(self, hidden):
        """ProductSort([single]) raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            ProductSort([hidden])

    def test_product_sort_empty_raises(self):
        """ProductSort([]) raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            ProductSort([])


# ===========================================================================
# Semiring law counterexamples
# ===========================================================================

class TestCounterexamples:

    def test_swapped_zero_one(self, backend):
        """Real semiring with zero=1, one=0 violates ⊕ identity."""
        with pytest.raises(ValueError, match="⊕ identity"):
            Semiring("broken", "add", "multiply", 1.0, 0.0).check_laws(backend, _REAL_SAMPLES)

    def test_nan_producing_semiring_rejected(self, backend):
        """Semiring whose operations produce NaN must be caught by check_laws."""
        sr = Semiring("broken", "add", "multiply", 0.0, 1.0)
        with pytest.raises(ValueError, match="NaN"):
            sr.check_laws(backend, [(float('nan'), 1.0, 1.0)])
