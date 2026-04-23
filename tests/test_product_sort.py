"""Product sort tests: typed tuples of sorts with round-trip TermCoder.

Covers product_sort construction, element extraction, type identity,
is_product_sort predicate, and sort_coder dispatch to product_sort_coder.
"""

import numpy as np
import pytest

import hydra.core as core
from hydra.dsl.python import Right

from unialg import (
    numpy_backend, Semiring, sort, sort_coder,
    product_sort, is_product_sort,
)
from unialg.algebra import sort_type_from_term
from unialg.algebra.sort import product_sort_elements, PRODUCT_SORT_TYPE_NAME


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return numpy_backend()


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return sort("hidden", real_sr)


@pytest.fixture
def output_sort(real_sr):
    return sort("output", real_sr)


@pytest.fixture
def residual_sort(real_sr):
    return sort("residual", real_sr)


# ===========================================================================
# Product sorts
# ===========================================================================

class TestProductSorts:
    """product_sort, is_product_sort, product_sort_elements, sort_type_from_term."""

    def test_product_sort_creates_record_with_correct_type_name(self, hidden, output_sort):
        """product_sort([a, b]) is a TermRecord with type_name PRODUCT_SORT_TYPE_NAME."""
        ps = product_sort([hidden, output_sort])
        assert isinstance(ps, core.TermRecord)
        assert ps.value.type_name == PRODUCT_SORT_TYPE_NAME

    def test_is_product_sort_true_for_product(self, hidden, output_sort):
        """is_product_sort returns True for a product_sort term."""
        ps = product_sort([hidden, output_sort])
        assert is_product_sort(ps) is True

    def test_is_product_sort_false_for_plain_sort(self, hidden):
        """is_product_sort returns False for an ordinary sort term."""
        assert is_product_sort(hidden) is False

    def test_product_sort_elements_round_trips(self, hidden, output_sort, residual_sort):
        """product_sort_elements recovers the same sorts in declaration order."""
        ps = product_sort([hidden, output_sort, residual_sort])
        elements = product_sort_elements(ps)
        assert len(elements) == 3
        expected_types = [
            sort_type_from_term(hidden),
            sort_type_from_term(output_sort),
            sort_type_from_term(residual_sort),
        ]
        actual_types = [sort_type_from_term(e) for e in elements]
        assert actual_types == expected_types

    def test_product_sort_requires_at_least_two_elements(self, hidden):
        """product_sort([single]) raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            product_sort([hidden])

    def test_product_sort_empty_raises(self):
        """product_sort([]) raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            product_sort([])

    def test_sort_type_from_term_distinct_for_product_vs_components(self, hidden, output_sort):
        """sort_type_from_term produces a nested TypePair for product sorts, distinct from components."""
        ps = product_sort([hidden, output_sort])
        ps_type = sort_type_from_term(ps)
        hidden_type = sort_type_from_term(hidden)
        output_type = sort_type_from_term(output_sort)
        assert ps_type != hidden_type
        assert ps_type != output_type
        assert isinstance(ps_type, core.TypePair)

    def test_sort_coder_on_product_sort_encodes_decodes_tuple(self, hidden, output_sort, backend):
        """sort_coder on a product sort encodes/decodes a tuple of arrays correctly."""
        ps = product_sort([hidden, output_sort])
        prod_coder = sort_coder(ps, backend)

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])

        encoded = prod_coder.decode(None, (a, b))
        assert isinstance(encoded, Right)
        term = encoded.value

        decoded = prod_coder.encode(None, None, term)
        assert isinstance(decoded, Right)
        pair = decoded.value

        np.testing.assert_allclose(pair[0], a)
        np.testing.assert_allclose(pair[1], b)
