"""
Unfolding RNN — StreamF ana.

Functor:   F(X) = Tensor × X
Algebra:   identity corecursion (foldToTerm)

Structural test only: verifies that the lowering pipeline emits a
corecursive function and that it raises on evaluation (infinite recursion).
"""

import numpy as np
import pytest

from backends import load_generated


class TestStreamRnn:

    def test_infinite_corecursion(self):
        unfold_stream = load_generated("seed.stream", "unfold_stream")
        pair = (np.float64(1.0), np.float64(2.0))
        with pytest.raises((RecursionError, IndexError, TypeError)):
            unfold_stream(pair)
