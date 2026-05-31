"""
Recursive NN — RTreeF cata.

Functor:   F(X) = Tensor + (X × X)
Semiring:  real  (add_id=0.0, mul_id=1.0)
Algebra:   leaf = W · a (contraction hi,i->h), node = left + right (elementwise)

Differential test (TF): fold_tree vs SimpleRNN(activation='linear') with
W_rec=I, b=0, h_0=0. By linearity of W, fold(tree) = W @ sum(leaves) for
any tree shape — the same value SimpleRNN accumulates over the leaf sequence.

Invariant test (torch, numpy): fold_tree vs the closed form W @ sum(leaves).
This avoids writing a second recursive tree fold as the oracle.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hydra.dsl.python import Left, Right

from backends import (
    BackendSpec,
    NumpyBackend,
    TFBackend,
    TorchBackend,
    HYPO,
    arch_generated_root,
)

GENERATED_ROOT = arch_generated_root(__file__)

INPUT_DIMS = [1, 2, 3]
HIDDEN_DIMS = [1, 2, 4]


def _tf_reference(backend, w, leaves, idim, hdim):
    tf = backend.framework
    rnn = tf.keras.layers.SimpleRNN(
        units=hdim,
        activation="linear",
        use_bias=False,
        return_sequences=False,
        dtype="float64",
    )
    x_tf = tf.expand_dims(tf.stack(leaves), axis=0)  # (1, n_leaves, idim)
    rnn(x_tf)  # build weights
    rnn.set_weights(
        [
            tf.transpose(w).numpy(),  # kernel: (idim, hdim)
            np.eye(hdim, dtype=np.float64),  # recurrent_kernel: (hdim, hdim) = I
        ]
    )
    h0 = tf.zeros((1, hdim), dtype=tf.float64)
    return rnn(x_tf, initial_state=h0)[0]


def _closed_form_reference(backend, w, leaves, idim, hdim):
    del idim, hdim
    lib = backend.framework
    if backend.name == "numpy":
        return w @ lib.sum(lib.stack(leaves), axis=0)
    if backend.name == "tensorflow":
        return lib.linalg.matvec(w, lib.reduce_sum(lib.stack(leaves), axis=0))
    if backend.name == "torch":
        return lib.mv(w, lib.sum(lib.stack(leaves), dim=0))
    raise AssertionError(f"unsupported backend: {backend.name}")


BACKENDS = [
    BackendSpec(
        TFBackend(), module="seed.tree", fn="fold_tree", reference=_tf_reference
    ),
    BackendSpec(
        TorchBackend(),
        module="seed.tree",
        fn="fold_tree",
        reference=_closed_form_reference,
    ),
    BackendSpec(
        NumpyBackend(),
        module="seed.tree",
        fn="fold_tree",
        reference=_closed_form_reference,
    ),
]


@st.composite
def _tree_recursive(draw, backend, dim, depth):
    if depth <= 0 or draw(st.booleans()):
        val = backend.random_vector(draw, dim)
        return [val], Left(val)
    else:
        left_vals, left_tree = draw(_tree_recursive(backend, dim, depth - 1))
        right_vals, right_tree = draw(_tree_recursive(backend, dim, depth - 1))
        return left_vals + right_vals, Right((left_tree, right_tree))


@st.composite
def tree_inputs(draw, backend, max_depth=3):
    """Draw (w, leaves, tree, input_dim, hidden_dim)."""
    input_dim = draw(st.sampled_from(INPUT_DIMS))
    hidden_dim = draw(st.sampled_from(HIDDEN_DIMS))

    w = backend.random_matrix(draw, hidden_dim, input_dim)
    leaves, tree = draw(_tree_recursive(backend, input_dim, max_depth))

    return w, leaves, tree, input_dim, hidden_dim


@pytest.fixture(params=BACKENDS, ids=lambda s: s.backend.name)
def spec(request):
    return request.param


class TestTreeRnn:

    @given(data=st.data())
    @settings(max_examples=50, **HYPO)
    def test_tree_cata(self, spec, data):
        backend = spec.backend
        w, leaves, tree, idim, hdim = data.draw(tree_inputs(backend))

        fold = spec.load(GENERATED_ROOT)
        gen = fold(w, tree)
        assert tuple(gen.shape) == (
            hdim,
        ), f"[{backend.name}] wrong shape: expected {(hdim,)}, got {gen.shape}"
        assert backend.is_finite(gen), f"[{backend.name}] non-finite output"

        ref = spec.reference(backend, w, leaves, idim, hdim)
        assert backend.allclose(
            gen, ref, atol=1e-5
        ), f"[{backend.name}] mismatch: input_dim={idim}, hidden_dim={hdim}"
