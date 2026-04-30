"""Unit tests for ProductSort: construction, predicate, element round-trips, coder.

Behavior-preservation tests per audit Phase 2 (robust-scribbling-dove.md).
Pins:
  - ProductSort is monoidal-only (ARCHITECTURE.md § "ProductSort is monoidal-only")
  - Single-element rejection (len >= 2 required)
  - .elements accessor round-trip
  - .type_ produces right-nested core.TypePair chain (3-element verified structurally)
  - .coder() round-trips a nested tuple through encode/decode
  - No projection methods present (fst, snd, pi_i, first, second, __getitem__)
  - ProductSort.from_term() round-trips back to equivalent ProductSort
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


# ===========================================================================
# Behavior-preservation additions (audit Phase 2 — robust-scribbling-dove.md)
# ===========================================================================

class TestProductSortMonoidalContract:
    """Pin the monoidal-only contract; assert no cartesian structure exists."""

    def test_single_element_raises(self, hidden):
        """ProductSort([single_sort]) must raise ValueError (len >= 2 required)."""
        with pytest.raises(ValueError, match="at least 2"):
            ProductSort([hidden])

    def test_empty_raises(self):
        """ProductSort([]) must raise ValueError."""
        with pytest.raises((ValueError, IndexError)):
            ProductSort([])

    def test_three_element_type_pair_structure(self, hidden, output_sort, residual_sort):
        """3-element ProductSort.type_ is TypePair(A, TypePair(B, C)) — right-nested."""
        ps = ProductSort([hidden, output_sort, residual_sort])
        t = ps.type_
        # Outermost is TypePair
        assert isinstance(t, core.TypePair)
        # Second field of outer pair is itself a TypePair (right-nesting)
        assert isinstance(t.value.second, core.TypePair)
        # Inner pair's components are the leaf sort types
        h_type = Sort.from_term(hidden).type_
        o_type = Sort.from_term(output_sort).type_
        r_type = Sort.from_term(residual_sort).type_
        assert t.value.first == h_type
        assert t.value.second.value.first == o_type
        assert t.value.second.value.second == r_type

    def test_no_projection_methods(self, hidden, output_sort):
        """ProductSort has no fst, snd, pi_i, first, second, or __getitem__."""
        ps = ProductSort([hidden, output_sort])
        forbidden = ("fst", "snd", "pi_i", "first", "second", "__getitem__")
        for attr in forbidden:
            assert not hasattr(ps, attr), (
                f"ProductSort must not have '{attr}' — it is monoidal-only, not cartesian. "
                f"See ARCHITECTURE.md § ProductSort is monoidal-only."
            )

    def test_from_term_round_trips(self, hidden, output_sort):
        """ProductSort.from_term(ps.term) recovers an equivalent ProductSort."""
        ps = ProductSort([hidden, output_sort])
        ps2 = ProductSort.from_term(ps.term)
        assert isinstance(ps2, ProductSort)
        # Same type_ — structural equality on the Hydra type
        assert ps2.type_ == ps.type_
        # Same number of elements
        assert len(ps2.elements) == len(ps.elements)
        # Same component types
        orig_types = [e.type_ for e in ps.elements]
        rt_types = [e.type_ for e in ps2.elements]
        assert orig_types == rt_types

    def test_coder_right_nested_pair(self, hidden, output_sort, residual_sort, backend):
        """3-element coder encodes a right-nested tuple and decodes back correctly."""
        ps = ProductSort([hidden, output_sort, residual_sort])
        coder = ps.coder(backend)

        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0])
        c = np.array([6.0])

        # Encoding a 3-tuple as right-nested: (a, (b, c))
        encoded = coder.decode(None, (a, (b, c)))
        assert isinstance(encoded, Right)
        decoded = coder.encode(None, None, encoded.value)
        assert isinstance(decoded, Right)
        result = decoded.value
        np.testing.assert_allclose(result[0], a)
        np.testing.assert_allclose(result[1][0], b)
        np.testing.assert_allclose(result[1][1], c)
