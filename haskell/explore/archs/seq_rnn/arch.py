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
from hypothesis import given, settings, HealthCheck, strategies as st
from hydra.dsl.python import Left, Right

from backends import BackendSpec, TFBackend, TorchBackend, load_generated


ADD_ID = 0.0
MUL_ID = 1.0

INPUT_DIMS  = [1, 2, 3]
HIDDEN_DIMS = [1, 2, 4]

HYPO = dict(deadline=None,
            suppress_health_check=[HealthCheck.function_scoped_fixture])


def _tf_reference(backend, wIn, wRec, b, s0, elements, idim, hdim):
    return backend.run_reference_rnn(wIn, wRec, b, s0, elements, idim, hdim)

def _torch_reference(backend, wIn, wRec, b, s0, elements, idim, hdim):
    return backend.run_reference_rnn(wIn, wRec, b, s0, elements, idim, hdim)


BACKENDS = [
    BackendSpec(TFBackend(),    module="seed.seq",      fn="fold_seq",      reference=_tf_reference),
    BackendSpec(TorchBackend(), module="seed.seq_tanh", fn="fold_seq_tanh", reference=_torch_reference),
]


def _make_seq(elements):
    node = Left(())
    for x in reversed(elements):
        node = Right((x, node))
    return node


@st.composite
def seq_inputs(draw, backend, max_len=5):
    """Draw (wIn, wRec, b, s0, elements, seq, input_dim, hidden_dim)."""
    input_dim  = draw(st.sampled_from(INPUT_DIMS))
    hidden_dim = draw(st.sampled_from(HIDDEN_DIMS))

    wIn  = backend.random_matrix(draw, hidden_dim, input_dim)
    wRec = backend.random_matrix(draw, hidden_dim, hidden_dim)
    b    = backend.random_vector(draw, hidden_dim)
    s0   = backend.fill_vector(hidden_dim, ADD_ID)

    n        = draw(st.integers(1, max_len))
    elements = [backend.random_vector(draw, input_dim) for _ in range(n)]
    seq      = _make_seq(elements)

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

        fold = load_generated(spec.module, spec.fn)
        gen  = fold(wIn, wRec, b, s0, seq)
        ref  = spec.reference(backend, wIn, wRec, b, s0, elements, idim, hdim)

        assert backend.allclose(gen, ref, atol=1e-4), \
            f"[{backend}] mismatch: input_dim={idim}, hidden_dim={hdim}"
