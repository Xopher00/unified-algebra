"""
Arm B: Differential testing against library-native modules.

Backend-polymorphic. Tensors are generated in the target runtime and stay
there. Each test runs once per backend via pytest parameterization.

Generated code uses real contractions (applyEquation "hi,i->h") — no
elementwise multiply hack. Weight copy to library modules is direct,
no diagonal adapter needed.

Epistemic status: a pass means "no counterexample on sampled inputs."
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck, strategies as st

from backends import TFBackend, TorchBackend
from strategies import seq_inputs, tree_inputs


HYPO = dict(deadline=None,
            suppress_health_check=[HealthCheck.function_scoped_fixture])


@pytest.fixture(params=[TFBackend(), TorchBackend()], ids=lambda b: b.name)
def backend(request):
    return request.param


# ══════════════════════════════════════════════════════════════════════════════
# seqCata — Folding RNN vs library-native RNN
#
# Generated: fold_seq(wIn, wRec, b, s0, x)
#   h_t = W_in · x_t + W_rec · h_{t-1} + b           (TF, linear)
#   h_t = tanh(W_in · x_t + W_rec · h_{t-1} + b)     (torch, tanh)
#
# Direct weight copy to SimpleRNN / torch.nn.RNN — no diag adapter.
# Cata is right-fold; torch.nn.RNN is left-fold → reversed input.
# ══════════════════════════════════════════════════════════════════════════════

class TestSeqCata:

    @given(data=st.data())
    @settings(max_examples=50, **HYPO)
    def test_rnn_cell(self, backend, data):
        wIn, wRec, b, s0, elements, seq, input_dim, hidden_dim = \
            data.draw(seq_inputs(backend, add_id=0.0))
        fold = backend.load_fold_seq()
        gen = fold(wIn, wRec, b, s0, seq)
        ref = backend.run_reference_rnn(wIn, wRec, b, s0, elements,
                                        input_dim, hidden_dim)
        assert backend.allclose(gen, ref, atol=1e-4), \
            f"[{backend}] mismatch: input_dim={input_dim}, hidden_dim={hidden_dim}"


# ══════════════════════════════════════════════════════════════════════════════
# treeCata — Recursive NN
# Bucket (iii): no library native. Smoke-test finite outputs.
# ══════════════════════════════════════════════════════════════════════════════

class TestTreeCata:

    @given(data=st.data())
    @settings(max_examples=50, **HYPO)
    def test_tree_runs(self, backend, data):
        w, _, tree, _, _ = data.draw(tree_inputs(backend))
        fold = backend.load_fold_tree()
        result = fold(w, tree)
        assert np.all(np.isfinite(np.asarray(result))), \
            f"[{backend}] non-finite tree output"


# ══════════════════════════════════════════════════════════════════════════════
# streamAna / mooreCata — novel, structural
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamAna:
    def test_infinite_corecursion(self):
        from seed.stream import unfold_stream
        pair = (np.float64(1.0), np.float64(2.0))
        with pytest.raises((RecursionError, IndexError, TypeError)):
            unfold_stream(pair)


class TestMooreCata:
    def test_output_structure(self):
        from seed.moore import moore_step
        pair = (np.float64(1.0), lambda inp: np.float64(inp + 1.0))
        result = moore_step(pair)
        assert isinstance(result, tuple) and len(result) == 2
