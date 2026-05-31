"""
Unfolding RNN — StreamLazyF ana.

Functor:   F(X) = Tensor × (() → X)
Coalgebra: unfold_stream_linear(w, s) = (W·s, λ_. unfold_stream_linear(w, W·s))

The continuation is a thunk so Python does not recurse immediately.
Taking N steps yields [W·s, W²·s, …, Wⁿ·s], which is the differential
reference: successive applications of the same linear map.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from backends import (
    BackendSpec,
    NumpyBackend,
    TFBackend,
    TorchBackend,
    HYPO,
    arch_generated_root,
)

GENERATED_ROOT = arch_generated_root(__file__)

HIDDEN_DIMS = [1, 2, 4]
MAX_STEPS = 5


def _take_n(fn, w, s0, n):
    """Unroll the lazy stream for n steps without full recursion."""
    outputs = []
    step = fn(w, s0)
    for _ in range(n):
        output, cont = step
        outputs.append(np.array(output))
        step = cont(None)
    return outputs


def _numpy_reference(backend, w, s0, n):
    lib = backend.framework
    outputs = []
    s = s0
    for _ in range(n):
        s = lib.einsum("hi,i->h", w, s)
        outputs.append(np.array(s))
    return outputs


def _tf_reference(backend, w, s0, n):
    lib = backend.framework
    outputs = []
    s = s0
    for _ in range(n):
        s = lib.linalg.matvec(w, s)
        outputs.append(np.array(s))
    return outputs


def _torch_reference(backend, w, s0, n):
    lib = backend.framework
    outputs = []
    s = s0
    for _ in range(n):
        s = lib.mv(w, s)
        outputs.append(np.array(s.detach()))
    return outputs


BACKENDS = [
    BackendSpec(
        NumpyBackend(),
        module="seed.stream_linear",
        fn="unfold_stream_linear",
        reference=_numpy_reference,
    ),
    BackendSpec(
        TFBackend(),
        module="seed.stream_linear",
        fn="unfold_stream_linear",
        reference=_tf_reference,
    ),
    BackendSpec(
        TorchBackend(),
        module="seed.stream_linear",
        fn="unfold_stream_linear",
        reference=_torch_reference,
    ),
]


@st.composite
def stream_inputs(draw, backend):
    """Draw (w, s0, n): square weight matrix, initial state, step count."""
    dim = draw(st.sampled_from(HIDDEN_DIMS))
    w = backend.random_matrix(draw, dim, dim)
    s0 = backend.random_vector(draw, dim)
    n = draw(st.integers(min_value=1, max_value=MAX_STEPS))
    return w, s0, n


@pytest.fixture(params=BACKENDS, ids=lambda s: s.backend.name)
def spec(request):
    return request.param


class TestStreamRnn:

    @given(data=st.data())
    @settings(max_examples=50, **HYPO)
    def test_stream_linear(self, spec, data):
        backend = spec.backend
        w, s0, n = data.draw(stream_inputs(backend))

        unfold = spec.load(GENERATED_ROOT)
        gen = _take_n(unfold, w, s0, n)
        ref = spec.reference(backend, w, s0, n)

        assert len(gen) == n, f"[{backend.name}] expected {n} steps, got {len(gen)}"
        for k, (g, r) in enumerate(zip(gen, ref)):
            assert backend.allclose(
                g, r, atol=1e-5
            ), f"[{backend.name}] mismatch at step {k + 1} of {n}"
