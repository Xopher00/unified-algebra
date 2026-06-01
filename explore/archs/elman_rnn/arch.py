"""
ElmanRnn — Ana.

Functor: {'tag': 'Product', 'left': {'tag': 'KConst'}, 'right': {'tag': 'Exp', 'arg': {'tag': 'Hole'}}}
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

SR_REAL_ZERO = 0.0
SR_REAL_ONE = 1.0


def _take_n(fn, v, bv, w, u, bh, s0, inputs):
    """Unroll the coalgebra for len(inputs) steps."""
    outputs = []
    step = fn(v, bv, w, u, bh, s0)
    for inp in inputs:
        output, cont = step
        outputs.append(np.array(output))
        step = cont(inp)
    return outputs


def _numpy_reference(backend, v, bv, w, u, bh, s0, inputs):
    lib = backend.framework
    outputs = []
    _h = s0
    for inp in inputs:
        lin_vs = lib.einsum("oi,i->o", v, _h)
        out = lin_vs + bv
        outputs.append(np.array(out))
        lin_ws = lib.einsum("hi,i->h", w, _h)
        lin_ux = lib.einsum("hi,i->h", u, inp)
        sum1 = lin_ws + lin_ux
        sum2 = sum1 + bh
        h_next = lib.tanh(sum2)
        _h = h_next
    return outputs


def _tf_reference(backend, v, bv, w, u, bh, s0, inputs):
    tf = backend.framework
    hidden = int(s0.shape[0])
    inp_size = int(inputs[0].shape[0])
    cell = tf.keras.layers.SimpleRNNCell(
        hidden, activation="tanh", use_bias=True, dtype=tf.float64
    )
    cell.build((None, inp_size))
    # TF row-vector convention: inp @ kernel + h @ recurrent_kernel + bias
    # kernel: (input_dim, hidden)  ← u.T   (our u is (hidden, input))
    # recurrent_kernel: (hidden, hidden) ← w.T  (our w is (hidden, hidden))
    cell.set_weights([u.numpy().T, w.numpy().T, bh.numpy()])
    h = s0[None]  # (1, hidden)
    outputs = []
    for inp in inputs:
        out = tf.matmul(h, tf.transpose(v)) + bv  # (1, output_dim)
        outputs.append(np.array(out.numpy().squeeze()))
        h, _ = cell(inp[None], [h])
    return outputs


def _torch_reference(backend, v, bv, w, u, bh, s0, inputs):
    torch = backend.framework
    hidden = s0.shape[0]
    inp_size = inputs[0].shape[0]
    cell = torch.nn.RNNCell(inp_size, hidden, nonlinearity="tanh").double()
    with torch.no_grad():
        # Torch convention: weight_ih (hidden, input), weight_hh (hidden, hidden)
        cell.weight_ih.copy_(u)
        cell.weight_hh.copy_(w)
        cell.bias_ih.copy_(bh)
        cell.bias_hh.zero_()
    h = s0.unsqueeze(0)  # (1, hidden)
    outputs = []
    for inp in inputs:
        out = torch.mv(v, h.squeeze()) + bv  # (output_dim,)
        outputs.append(np.array(out.detach()))
        h = cell(inp.unsqueeze(0), h)
    return outputs


BACKENDS = [
    BackendSpec(
        NumpyBackend(),
        module="seed.elman_rnn",
        fn="elman_rnn_step",
        reference=_numpy_reference,
    ),
    BackendSpec(
        TFBackend(),
        module="seed.elman_rnn",
        fn="elman_rnn_step",
        reference=_tf_reference,
    ),
    BackendSpec(
        TorchBackend(),
        module="seed.elman_rnn",
        fn="elman_rnn_step",
        reference=_torch_reference,
    ),
]


@st.composite
def elmanRnn_inputs(draw, backend):
    o = draw(st.sampled_from(OUTPUT_DIMS))
    h = draw(st.sampled_from(HIDDEN_DIMS))
    i = draw(st.sampled_from(INPUT_DIMS))
    n = draw(st.integers(min_value=1, max_value=MAX_STEPS))
    v = backend.random_matrix(draw, o, h)
    bv = backend.random_vector(draw, o)
    w = backend.random_matrix(draw, h, h)
    u = backend.random_matrix(draw, h, i)
    bh = backend.random_vector(draw, h)
    s0 = backend.random_vector(draw, h)
    inputs = [backend.random_vector(draw, i) for _ in range(n)]
    return v, bv, w, u, bh, s0, inputs, n


@pytest.fixture(params=BACKENDS, ids=lambda s: s.backend.name)
def spec(request):
    return request.param


class TestElmanRnn:

    @given(data=st.data())
    @settings(max_examples=50, **HYPO)
    def test_elman_rnn(self, spec, data):
        backend = spec.backend
        v, bv, w, u, bh, s0, inputs, _n = data.draw(elmanRnn_inputs(backend))

        fn = spec.load(GENERATED_ROOT)
        gen = _take_n(fn, v, bv, w, u, bh, s0, inputs)
        ref = spec.reference(backend, v, bv, w, u, bh, s0, inputs)

        assert len(gen) == _n
        for k, (g, r) in enumerate(zip(gen, ref)):
            assert backend.allclose(
                g, r, atol=1e-5
            ), f"[{backend.name}] mismatch at step {k + 1} of {_n}"
