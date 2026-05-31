"""
Moore machine — MooreF cata.

Functor:          F(X) = Tensor × (Tensor → X)
Layer rebuilding: anaModule/buildLayer

Structural test only: verifies that the output is a 2-tuple (tensor, callable).
"""

import numpy as np
import pytest

from backends import BackendSpec, NumpyBackend, arch_generated_root

GENERATED_ROOT = arch_generated_root(__file__)

BACKENDS = [
    BackendSpec(NumpyBackend(), module="seed.moore", fn="moore_step", reference=None)
]


@pytest.fixture(params=BACKENDS, ids=lambda s: s.backend.name)
def spec(request):
    return request.param


class TestMoore:

    def test_output_structure(self, spec):
        moore_step = spec.load(GENERATED_ROOT)
        pair = (np.float64(1.0), lambda inp: np.float64(inp + 1.0))
        result = moore_step(pair)
        assert (
            isinstance(result, tuple) and len(result) == 2
        ), f"expected 2-tuple, got {type(result)}"
