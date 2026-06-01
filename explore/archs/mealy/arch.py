"""
Mealy machine — MealyF ana.

Functor:   F(X) = (Input → Output) × (Input → X)
Coalgebra: mealy_step(v, c, w, u, s) = (λinp. V·s + C·inp,
                                         λinp. mealy_step(v, c, w, u, W·s + U·inp))

Output depends on both state and input; state transitions are input-driven.
Reference: unroll N steps over an input sequence, compare (V·s_k + C·inp_k) at each step.
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

HIDDEN_DIMS = [2, 3]
OUTPUT_DIMS = [1, 2]
INPUT_DIMS = [2, 3]
MAX_STEPS = 4


def _take_n(fn, v, c, w, u, s0, inputs):
    """Unroll the Mealy machine for len(inputs) steps."""
    outputs = []
    step = fn(v, c, w, u, s0)
    for inp in inputs:
        out_fn, trans_fn = step
        outputs.append(np.array(out_fn(inp)))
        step = trans_fn(inp)
    return outputs


def _numpy_reference(backend, v, c, w, u, s0, inputs):
    lib = backend.framework
    outputs = []
    s = s0
    for inp in inputs:
        o = lib.einsum("oh,h->o", v, s) + lib.einsum("od,d->o", c, inp)
        outputs.append(np.array(o))
        s = lib.einsum("hi,i->h", w, s) + lib.einsum("hi,i->h", u, inp)
    return outputs


def _tf_reference(backend, v, c, w, u, s0, inputs):
    lib = backend.framework
    outputs = []
    s = s0
    for inp in inputs:
        o = lib.linalg.matvec(v, s) + lib.linalg.matvec(c, inp)
        outputs.append(np.array(o))
        s = lib.linalg.matvec(w, s) + lib.linalg.matvec(u, inp)
    return outputs


def _torch_reference(backend, v, c, w, u, s0, inputs):
    lib = backend.framework
    outputs = []
    s = s0
    for inp in inputs:
        o = lib.mv(v, s) + lib.mv(c, inp)
        outputs.append(np.array(o.detach()))
        s = lib.mv(w, s) + lib.mv(u, inp)
    return outputs


BACKENDS = [
    BackendSpec(
        NumpyBackend(), module="seed.mealy", fn="mealy_step", reference=_numpy_reference
    ),
    BackendSpec(
        TFBackend(), module="seed.mealy", fn="mealy_step", reference=_tf_reference
    ),
    BackendSpec(
        TorchBackend(), module="seed.mealy", fn="mealy_step", reference=_torch_reference
    ),
]


@st.composite
def mealy_inputs(draw, backend):
    """Draw (v, c, w, u, s0, inputs, n): weight matrices, initial state, input sequence."""
    h = draw(st.sampled_from(HIDDEN_DIMS))
    o = draw(st.sampled_from(OUTPUT_DIMS))
    d = draw(st.sampled_from(INPUT_DIMS))
    n = draw(st.integers(min_value=1, max_value=MAX_STEPS))
    v = backend.random_matrix(draw, o, h)
    c = backend.random_matrix(draw, o, d)
    w = backend.random_matrix(draw, h, h)
    u = backend.random_matrix(draw, h, d)
    s0 = backend.random_vector(draw, h)
    inputs = [backend.random_vector(draw, d) for _ in range(n)]
    return v, c, w, u, s0, inputs, n


@pytest.fixture(params=BACKENDS, ids=lambda s: s.backend.name)
def spec(request):
    return request.param


class TestMealy:

    @given(data=st.data())
    @settings(max_examples=50, **HYPO)
    def test_mealy_step(self, spec, data):
        backend = spec.backend
        v, c, w, u, s0, inputs, n = data.draw(mealy_inputs(backend))

        mealy = spec.load(GENERATED_ROOT)
        gen = _take_n(mealy, v, c, w, u, s0, inputs)
        ref = spec.reference(backend, v, c, w, u, s0, inputs)

        assert len(gen) == n, f"[{backend.name}] expected {n} steps, got {len(gen)}"
        for k, (g, r) in enumerate(zip(gen, ref)):
            assert backend.allclose(
                g, r, atol=1e-5
            ), f"[{backend.name}] mismatch at step {k + 1} of {n}"
