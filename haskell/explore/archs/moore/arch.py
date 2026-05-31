"""
Moore machine — MooreF cata.

Functor:   F(X) = Tensor × (Tensor → X)
Algebra:   identity corecursion (foldToTerm)

Structural test only: verifies that the output is a 2-tuple (tensor, callable).
"""

import numpy as np

from backends import load_generated


class TestMoore:

    def test_output_structure(self):
        moore_step = load_generated("seed.moore", "moore_step")
        pair = (np.float64(1.0), lambda inp: np.float64(inp + 1.0))
        result = moore_step(pair)
        assert isinstance(result, tuple) and len(result) == 2, \
            f"expected 2-tuple, got {type(result)}"
