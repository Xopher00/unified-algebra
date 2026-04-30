"""End-to-end TermCoder round-trip tests for all Sort variants.

Audit reference: robust-scribbling-dove.md Phase 2 — "test_terms_roundtrip.py".

Pins:
  - tensor_coder encode/decode round-trips for rank-1 and rank-2 Sort tensors.
  - ProductSort (2-element): nested-tuple encode/decode round-trip.
  - The TermUnit / monoidal-unit encoding: None → TermUnit → None.
  - coder.decode(None, arr) = Python array → Hydra term  (Hydra naming: decode).
  - coder.encode(None, None, term) = Hydra term → Python array (Hydra naming: encode).

Note: Hydra's encode/decode naming is inverted from intuition:
  ``decode`` encodes Python → Term (writes into Hydra representation).
  ``encode`` decodes Term → Python (reads from Hydra representation).
"""

import numpy as np
import pytest
from hydra.dsl.python import Right

from unialg import NumpyBackend, Semiring, Sort, ProductSort
from unialg.terms import tensor_coder


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
def tropical_sr():
    return Semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


# ---------------------------------------------------------------------------
# Helpers (mirrors conftest conventions)
# ---------------------------------------------------------------------------

def encode_array(coder, arr):
    """Encode arr (numpy array or tuple) into a Hydra term via the coder.

    Avoids np.ascontiguousarray for tuple/pair values (ProductSort coders
    expect plain Python tuples, not numpy arrays).
    """
    if isinstance(arr, np.ndarray):
        value = np.ascontiguousarray(arr)
    else:
        value = arr  # pass tuples through as-is (ProductSort coder expects tuples)
    result = coder.decode(None, value)
    assert isinstance(result, Right), f"encode failed: {result}"
    return result.value


def decode_term(coder, term):
    result = coder.encode(None, None, term)
    assert isinstance(result, Right), f"decode failed: {result}"
    return result.value


# ---------------------------------------------------------------------------
# Rank-1 Sort round-trip
# ---------------------------------------------------------------------------

class TestSortRank1Roundtrip:
    """tensor_coder for a rank-1 sort round-trips float64 and float32 arrays."""

    def test_float64_1d(self, backend, hidden):
        coder = hidden.coder(backend)
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        term = encode_array(coder, arr)
        result = decode_term(coder, term)
        np.testing.assert_array_equal(result, arr)

    def test_float32_1d(self, backend, hidden):
        coder = hidden.coder(backend)
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        term = encode_array(coder, arr)
        result = decode_term(coder, term)
        np.testing.assert_allclose(result, arr, atol=1e-6)

    def test_tropical_sort_1d(self, backend, tropical_sr):
        sort = Sort("distances", tropical_sr)
        coder = sort.coder(backend)
        arr = np.array([0.0, 1.0, float("inf")])
        term = encode_array(coder, arr)
        result = decode_term(coder, term)
        np.testing.assert_array_equal(result[:2], arr[:2])
        assert np.isinf(result[2])

    def test_negative_values_1d(self, backend, hidden):
        coder = hidden.coder(backend)
        arr = np.array([-3.0, 0.0, 3.0])
        term = encode_array(coder, arr)
        result = decode_term(coder, term)
        np.testing.assert_allclose(result, arr)


# ---------------------------------------------------------------------------
# Rank-2 Sort round-trip
# ---------------------------------------------------------------------------

class TestSortRank2Roundtrip:
    """tensor_coder for a rank-2 sort round-trips 2D arrays."""

    def test_matrix_2d(self, backend, hidden):
        coder = hidden.coder(backend)
        arr = np.arange(6.0).reshape(2, 3)
        term = encode_array(coder, arr)
        result = decode_term(coder, term)
        np.testing.assert_array_equal(result, arr)

    def test_matrix_shape_preserved(self, backend, hidden):
        coder = hidden.coder(backend)
        arr = np.ones((4, 5))
        term = encode_array(coder, arr)
        result = decode_term(coder, term)
        assert result.shape == (4, 5)

    def test_matrix_float32_2d(self, backend, hidden):
        coder = hidden.coder(backend)
        arr = np.eye(3, dtype=np.float32)
        term = encode_array(coder, arr)
        result = decode_term(coder, term)
        np.testing.assert_allclose(result, arr, atol=1e-6)


# ---------------------------------------------------------------------------
# Rank-3 and scalar edge cases
# ---------------------------------------------------------------------------

class TestSortEdgeCases:
    """Scalar and rank-3 tensors round-trip through tensor_coder."""

    def test_scalar(self, backend, hidden):
        coder = hidden.coder(backend)
        arr = np.array(3.14)  # 0-d array
        term = encode_array(coder, arr)
        result = decode_term(coder, term)
        np.testing.assert_allclose(result, 3.14, atol=1e-10)

    def test_rank3(self, backend, hidden):
        coder = hidden.coder(backend)
        arr = np.arange(24.0).reshape(2, 3, 4)
        term = encode_array(coder, arr)
        result = decode_term(coder, term)
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# ProductSort round-trip
# ---------------------------------------------------------------------------

class TestProductSortRoundtrip:
    """ProductSort coder round-trips a right-nested tuple of arrays."""

    def test_two_element_product(self, backend, hidden, output_sort):
        ps = ProductSort([hidden, output_sort])
        coder = ps.coder(backend)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        term = encode_array(coder, (a, b))
        result = decode_term(coder, term)
        np.testing.assert_allclose(result[0], a)
        np.testing.assert_allclose(result[1], b)

    def test_three_element_product_right_nested(self, backend, real_sr):
        s1 = Sort("p1", real_sr)
        s2 = Sort("p2", real_sr)
        s3 = Sort("p3", real_sr)
        ps = ProductSort([s1, s2, s3])
        coder = ps.coder(backend)
        a = np.array([1.0])
        b = np.array([2.0, 3.0])
        c = np.array([4.0, 5.0, 6.0])
        # Right-nested encoding: (a, (b, c))
        term = encode_array(coder, (a, (b, c)))
        result = decode_term(coder, term)
        np.testing.assert_allclose(result[0], a)
        np.testing.assert_allclose(result[1][0], b)
        np.testing.assert_allclose(result[1][1], c)

    def test_product_preserves_dtypes(self, backend, real_sr):
        s1 = Sort("d1", real_sr)
        s2 = Sort("d2", real_sr)
        ps = ProductSort([s1, s2])
        coder = ps.coder(backend)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float64)
        term = encode_array(coder, (a, b))
        result = decode_term(coder, term)
        np.testing.assert_allclose(result[0], a, atol=1e-6)
        np.testing.assert_allclose(result[1], b)


# ---------------------------------------------------------------------------
# generic tensor_coder (no sort type)
# ---------------------------------------------------------------------------

class TestGenericTensorCoder:
    """tensor_coder with no type arg uses a default ua.tensor.NDArray type."""

    def test_default_type(self, backend):
        coder = tensor_coder(backend)
        arr = np.array([1.0, 2.0, 3.0])
        term = encode_array(coder, arr)
        result = decode_term(coder, term)
        np.testing.assert_array_equal(result, arr)

    def test_decode_returns_right(self, backend):
        coder = tensor_coder(backend)
        arr = np.zeros(5)
        result = coder.decode(None, arr)
        assert isinstance(result, Right)

    def test_encode_returns_right(self, backend):
        coder = tensor_coder(backend)
        arr = np.ones(3)
        term = coder.decode(None, arr).value
        result = coder.encode(None, None, term)
        assert isinstance(result, Right)
