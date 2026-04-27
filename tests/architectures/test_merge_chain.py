"""Merge-chain pattern tests.

Tests scaled dot-product attention pattern expressed as a 3-branch merge chain,
and cross-composition references (seq referencing branch, seq referencing seq,
full transformer block, unrolled transformer stack).

Uses NumpyBackend directly — these tests exercise the architecture pattern,
not backend parity (see tests/backend/test_backend_parity.py for that).
"""

import numpy as np
import pytest

from unialg import NumpyBackend, parse_ua


def _assert_close(actual, expected, rtol=1e-6):
    np.testing.assert_allclose(actual, expected, rtol=rtol)


_REAL = """\
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""


class TestScaledDotProductAttention:
    """Attention pattern as a 3-branch merge chain.

    Branch ops are nonlinearities (standing in for projections, which need
    parameterized weights). The merge chain is the key part: two contractions
    with softmax between them.

    Stack: [Q,K,V] -> score(Q,K) -> [S,V] -> softmax(S) -> [P,V] -> mix(P,V) -> [out]
    """

    _ATTN_PROG = _REAL + """\
op ~act : hidden -> hidden
  nonlinearity = abs

op score : hidden -> hidden
  einsum = "ik,jk->ij"
  algebra = real

op mix : hidden -> hidden
  einsum = "ij,jk->ik"
  algebra = real

branch head : hidden -> hidden = act[q] | act[k] | act[v]
  merge = score ~> softmax ~> mix
"""

    def test_attention_parses(self):
        prog = parse_ua(self._ATTN_PROG, NumpyBackend())
        assert 'head' in prog.entry_points()

    def test_attention_matches_numpy(self):
        prog = parse_ua(self._ATTN_PROG, NumpyBackend())
        x = np.array([[1.0, -1.0], [0.5, 2.0], [0.0, -0.5]], dtype=np.float32)
        result = prog('head', x)

        q = k = v = np.abs(x)
        scores = np.einsum("ik,jk->ij", q, k)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        expected = np.einsum("ij,jk->ik", probs, v)

        _assert_close(result, expected, rtol=1e-5)

    def test_attention_with_extra_nonlinearity(self):
        """Extended chain: score ~> softmax ~> abs ~> mix."""
        text = _REAL + """\
op ~act : hidden -> hidden
  nonlinearity = abs

op score : hidden -> hidden
  einsum = "ik,jk->ij"
  algebra = real

op mix : hidden -> hidden
  einsum = "ij,jk->ik"
  algebra = real

branch head : hidden -> hidden = act[q] | act[k] | act[v]
  merge = score ~> softmax ~> abs ~> mix
"""
        prog = parse_ua(text, NumpyBackend())
        x = np.array([[1.0, -1.0], [0.5, 2.0], [0.0, -0.5]], dtype=np.float32)

        q = k = v = np.abs(x)
        scores = np.einsum("ik,jk->ij", q, k)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        after_abs = np.abs(probs)
        expected = np.einsum("ij,jk->ik", after_abs, v)

        _assert_close(prog('head', x), expected, rtol=1e-5)


class TestCrossCompositionReferences:

    def test_seq_referencing_branch(self):
        """A seq can reference a branch by name."""
        text = _REAL + """\
op relu : hidden -> hidden
  nonlinearity = relu
op tanh_act : hidden -> hidden
  nonlinearity = tanh

op hadamard : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

branch fan : hidden -> hidden = relu | tanh_act
  merge = hadamard

op abs_act : hidden -> hidden
  nonlinearity = abs

seq pipe : hidden -> hidden = fan >> abs_act
"""
        prog = parse_ua(text, NumpyBackend())
        x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        fan_out = np.maximum(0, x) * np.tanh(x)
        expected = np.abs(fan_out)
        _assert_close(prog('pipe', x), expected)

    def test_seq_referencing_seq(self):
        """A seq can reference another seq by name."""
        text = _REAL + """\
op relu : hidden -> hidden
  nonlinearity = relu
op tanh_act : hidden -> hidden
  nonlinearity = tanh
op abs_act : hidden -> hidden
  nonlinearity = abs

seq inner : hidden -> hidden = relu >> tanh_act
seq outer : hidden -> hidden = inner >> abs_act
"""
        prog = parse_ua(text, NumpyBackend())
        x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        expected = np.abs(np.tanh(np.maximum(0.0, x)))
        _assert_close(prog('outer', x), expected)

    def test_transformer_block(self):
        """Full transformer block: attention head + FFN, both with residuals."""
        text = _REAL + """\
op ~act : hidden -> hidden
  nonlinearity = abs

op score : hidden -> hidden
  einsum = "ik,jk->ij"
  algebra = real

op mix : hidden -> hidden
  einsum = "ij,jk->ik"
  algebra = real

branch attn : hidden -> hidden = act[q] | act[k] | act[v]
  merge = score ~> softmax ~> mix

seq attn_block+ : hidden -> hidden = attn
  algebra = real

op relu : hidden -> hidden
  nonlinearity = relu
op tanh_act : hidden -> hidden
  nonlinearity = tanh

seq ffn : hidden -> hidden = relu >> tanh_act

seq ffn_block+ : hidden -> hidden = ffn
  algebra = real

seq transformer : hidden -> hidden = attn_block >> ffn_block
"""
        prog = parse_ua(text, NumpyBackend())
        x = np.array([[1.0, -1.0], [0.5, 2.0], [0.0, -0.5]], dtype=np.float32)

        q = k = v = np.abs(x)
        scores = np.einsum("ik,jk->ij", q, k)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        attn_out = np.einsum("ij,jk->ik", probs, v)
        attn_residual = attn_out + x

        ffn_out = np.tanh(np.maximum(0, attn_residual))
        ffn_residual = ffn_out + attn_residual

        _assert_close(prog('transformer', x), ffn_residual, rtol=1e-5)

    def test_unrolled_transformer_stack(self):
        """Unroll a transformer block N times: a stacked transformer."""
        text = _REAL + """\
op ~act : hidden -> hidden
  nonlinearity = abs

op score : hidden -> hidden
  einsum = "ik,jk->ij"
  algebra = real

op mix : hidden -> hidden
  einsum = "ij,jk->ik"
  algebra = real

branch attn : hidden -> hidden = act[q] | act[k] | act[v]
  merge = score ~> softmax ~> mix

seq attn_block+ : hidden -> hidden = attn
  algebra = real

op relu : hidden -> hidden
  nonlinearity = relu
op tanh_act : hidden -> hidden
  nonlinearity = tanh

seq ffn : hidden -> hidden = relu >> tanh_act

seq ffn_block+ : hidden -> hidden = ffn
  algebra = real

seq layer : hidden -> hidden = attn_block >> ffn_block

unroll stack : hidden -> hidden
  step = layer
  steps = 3
"""
        prog = parse_ua(text, NumpyBackend())
        x = np.array([[1.0, -1.0], [0.5, 2.0], [0.0, -0.5]], dtype=np.float32)

        def one_layer(inp):
            q = k = v = np.abs(inp)
            scores = np.einsum("ik,jk->ij", q, k)
            probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            attn_out = np.einsum("ij,jk->ik", probs, v) + inp
            return np.tanh(np.maximum(0, attn_out)) + attn_out

        state = x
        for _ in range(3):
            state = one_layer(state)

        result = prog('stack', x)
        _assert_close(result[-1], state, rtol=1e-4)
