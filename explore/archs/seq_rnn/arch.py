"""
Folding RNN — SeqF cata.

Functor:   F(X) = 1 + (Tensor × X)
Semiring:  real  (add_id=0.0, mul_id=1.0)
Algebra:   h_t = W_in · x_t + W_rec · h_{t-1} + b   (linear)
           h_t = tanh(...)                             (torch variant)

Backends tested:
  tf    — seqCata     vs SimpleRNN(activation='linear')
  torch — seqCataTanh vs torch.nn.RNN(nonlinearity='tanh')
"""

import pytest
from hypothesis import given, settings, strategies as st
from hydra.dsl.python import Left, Right

from backends import BackendSpec, TFBackend, TorchBackend, HYPO, arch_generated_root

GENERATED_ROOT = arch_generated_root(__file__)


ADD_ID = 0.0
MUL_ID = 1.0

INPUT_DIMS = [1, 2, 3]
HIDDEN_DIMS = [1, 2, 4]


def _tf_reference(backend, wIn, wRec, b, s0, elements, idim, hdim):
    tf = backend.framework
    rev = list(reversed(elements))
    rnn = tf.keras.layers.SimpleRNN(
        units=hdim,
        activation="linear",
        use_bias=True,
        return_sequences=False,
        dtype="float64",
    )
    x_tf = tf.expand_dims(tf.stack(rev), axis=0)
    rnn(x_tf)
    rnn.set_weights(
        [
            tf.transpose(wIn).numpy(),  # kernel: (input, hidden)
            tf.transpose(wRec).numpy(),  # recurrent_kernel: (hidden, hidden)
            b.numpy(),
        ]
    )
    init = tf.reshape(s0, (1, hdim))
    return rnn(x_tf, initial_state=init)[0]


def _torch_reference(backend, wIn, wRec, b, s0, elements, idim, hdim):
    torch = backend.framework
    rev = list(reversed(elements))
    x_t = torch.stack(rev).unsqueeze(1)  # (steps, batch=1, input)
    h0 = s0.reshape(1, 1, hdim)  # (layers=1, batch=1, hidden)
    rnn = torch.nn.RNN(
        input_size=idim,
        hidden_size=hdim,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        batch_first=False,
        dtype=torch.float64,
    )
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(wIn)
        rnn.weight_hh_l0.copy_(wRec)
        rnn.bias_ih_l0.copy_(b)
        rnn.bias_hh_l0.zero_()
        _, hn = rnn(x_t, h0)
    return hn.squeeze()


BACKENDS = [
    BackendSpec(TFBackend(), module="seed.seq", fn="fold_seq", reference=_tf_reference),
    BackendSpec(
        TorchBackend(),
        module="seed.seq_tanh",
        fn="fold_seq_tanh",
        reference=_torch_reference,
    ),
]


def _make_seq(elements):
    node = Left(())
    for x in reversed(elements):
        node = Right((x, node))
    return node


@st.composite
def seq_inputs(draw, backend, max_len=5):
    """Draw (wIn, wRec, b, s0, elements, seq, input_dim, hidden_dim)."""
    input_dim = draw(st.sampled_from(INPUT_DIMS))
    hidden_dim = draw(st.sampled_from(HIDDEN_DIMS))

    wIn = backend.random_matrix(draw, hidden_dim, input_dim)
    wRec = backend.random_matrix(draw, hidden_dim, hidden_dim)
    b = backend.random_vector(draw, hidden_dim)
    s0 = backend.fill_vector(hidden_dim, ADD_ID)

    n = draw(st.integers(1, max_len))
    elements = [backend.random_vector(draw, input_dim) for _ in range(n)]
    seq = _make_seq(elements)

    return wIn, wRec, b, s0, elements, seq, input_dim, hidden_dim


@pytest.fixture(params=BACKENDS, ids=lambda s: s.backend.name)
def spec(request):
    return request.param


class TestSeqRnn:

    @given(data=st.data())
    @settings(max_examples=50, **HYPO)
    def test_rnn_cell(self, spec, data):
        backend = spec.backend
        wIn, wRec, b, s0, elements, seq, idim, hdim = data.draw(seq_inputs(backend))

        fold = spec.load(GENERATED_ROOT)
        gen = fold(wIn, wRec, b, s0, seq)
        ref = spec.reference(backend, wIn, wRec, b, s0, elements, idim, hdim)

        assert backend.allclose(
            gen, ref, atol=1e-4
        ), f"[{backend}] mismatch: input_dim={idim}, hidden_dim={hdim}"
