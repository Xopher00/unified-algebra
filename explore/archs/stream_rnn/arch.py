"""
Unfolding RNN — StreamF ana.

Functor:          F(X) = Tensor × X
Layer rebuilding: anaModule/buildLayer

Structural test only: verifies that the lowering pipeline emits a
corecursive function and that it raises on evaluation (infinite recursion).
"""

import numpy as np
import pytest

from backends import BackendSpec, NumpyBackend, arch_generated_root

GENERATED_ROOT = arch_generated_root(__file__)

BACKENDS = [
    BackendSpec(
        NumpyBackend(), module="seed.stream", fn="unfold_stream", reference=None
    )
]


@pytest.fixture(params=BACKENDS, ids=lambda s: s.backend.name)
def spec(request):
    return request.param


class TestStreamRnn:

    def test_infinite_corecursion(self, spec):
        unfold_stream = spec.load(GENERATED_ROOT)
        pair = (np.float64(1.0), np.float64(2.0))
        with pytest.raises((RecursionError, IndexError, TypeError)):
            unfold_stream(pair)
