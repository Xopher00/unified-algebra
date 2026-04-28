"""Unit tests for ProductSort: construction, predicate, element round-trips, coder.

Covers test_product_sort_creates_record_with_correct_type_name,
test_is_product_sort_true_for_product, test_is_product_sort_false_for_plain_sort,
test_product_sort_elements_round_trips, test_sort_type_from_term_distinct_for_product_vs_components,
test_sort_coder_on_product_sort_encodes_decodes_tuple.
"""

import numpy as np
import pytest

import hydra.core as core
from hydra.dsl.python import Right

from unialg import (
    Sort,
    ProductSort,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


@pytest.fixture
def residual_sort(real_sr):
    return Sort("residual", real_sr)


# ===========================================================================
# Product sorts — structural and round-trip tests
# ===========================================================================

class TestProductSorts:
    """product_sort, is_product_sort, product_sort_elements, sort_type_from_term."""

    def test_product_sort_creates_record_with_correct_type_name(self, hidden, output_sort):
        """ProductSort([a, b]) wraps a TermRecord with type_name PRODUCT_SORT_TYPE_NAME."""
        ps = ProductSort([hidden, output_sort])
        assert isinstance(ps.term, core.TermRecord)
        assert ps.term.value.type_name == ProductSort._type_name

    def test_is_product_sort_true_for_product(self, hidden, output_sort):
        """is_product_sort returns True for a product_sort term."""
        ps = ProductSort([hidden, output_sort])
        assert isinstance(ps, ProductSort) is True

    def test_is_product_sort_false_for_plain_sort(self, hidden):
        """is_product_sort returns False for an ordinary sort term."""
        assert isinstance(Sort.from_term(hidden), ProductSort) is False

    def test_product_sort_elements_round_trips(self, hidden, output_sort, residual_sort):
        """product_sort_elements recovers the same sorts in declaration order."""
        ps = ProductSort([hidden, output_sort, residual_sort])
        elements = ProductSort.from_term(ps).elements
        assert len(elements) == 3
        expected_types = [
            Sort.from_term(hidden).type_,
            Sort.from_term(output_sort).type_,
            Sort.from_term(residual_sort).type_,
        ]
        actual_types = [Sort.from_term(e).type_ for e in elements]
        assert actual_types == expected_types

    def test_sort_type_from_term_distinct_for_product_vs_components(self, hidden, output_sort):
        """sort_type_from_term produces a nested TypePair for product sorts, distinct from components."""
        ps = ProductSort([hidden, output_sort])
        ps_type = ps.type_
        hidden_type = Sort.from_term(hidden).type_
        output_type = Sort.from_term(output_sort).type_
        assert ps_type != hidden_type
        assert ps_type != output_type
        assert isinstance(ps_type, core.TypePair)

    def test_sort_coder_on_product_sort_encodes_decodes_tuple(self, hidden, output_sort, backend):
        """sort_coder on a product sort encodes/decodes a tuple of arrays correctly."""
        ps = ProductSort([hidden, output_sort])
        prod_coder = ps.coder(backend)

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
