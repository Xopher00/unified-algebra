"""N-ary merge and composition integration tests.

Exercises the lifted N-ary merge restriction: branch merge equations with
N einsum operands matching N branches apply all operands in a single
contraction rather than binary-folding. Also covers seq, residual, and
semiring polymorphism as integration tests.
"""

import numpy as np
import pytest
from scipy.special import expit

from unialg import NumpyBackend, JaxBackend, PytorchBackend, parse_ua


def _available_backends():
    pairs = [("numpy", NumpyBackend)]
    for name, cls in [("jax", JaxBackend), ("pytorch", PytorchBackend)]:
        try:
            cls()
            pairs.append((name, cls))
        except (ImportError, ModuleNotFoundError):
            pass
    return pairs


def _assert_close(actual, expected, rtol=1e-6):
    try:
        import torch
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(
                actual, torch.tensor(expected, dtype=actual.dtype), rtol=rtol, atol=1e-7)
            return
    except ImportError:
        pass
    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.fixture(params=_available_backends(), ids=lambda b: b[0])
def backend(request):
    return request.param[1]()


_REAL = """\
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""

_TROPICAL = """\
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
spec node(tropical)
"""


class TestNaryMerge:

    def test_ternary_merge(self, backend):
        """3-branch fan with 3-operand einsum: single contraction, not binary fold."""
        text = _REAL + """\
op relu : hidden -> hidden
  nonlinearity = relu
op tanh_act : hidden -> hidden
  nonlinearity = tanh
op sigmoid_act : hidden -> hidden
  nonlinearity = sigmoid

op trilinear : hidden -> hidden
  einsum = "ik,jk,jl->il"
  algebra = real

branch tri : hidden -> hidden = relu | tanh_act | sigmoid_act
  merge = trilinear
"""
        prog = parse_ua(text, backend)
        x = np.array([[1.0, -1.0], [0.5, 2.0], [0.0, -0.5]])
        a = np.maximum(0, x)
        b = np.tanh(x)
        c = expit(x)
        expected = np.einsum("ik,jk,jl->il", a, b, c)
        _assert_close(prog('tri', x), expected)

    def test_ternary_merge_tropical(self, backend):
        """Same 3-branch structure, tropical algebra: contraction uses min-plus."""
        text = _TROPICAL + """\
op relu : node -> node
  nonlinearity = relu
op tanh_act : node -> node
  nonlinearity = tanh
op abs_act : node -> node
  nonlinearity = abs

op trilinear : node -> node
  einsum = "i,i,i->i"
  algebra = tropical

branch tri : node -> node = relu | tanh_act | abs_act
  merge = trilinear
"""
        prog = parse_ua(text, backend)
        x = np.array([1.0, -1.0, 0.5, 2.0])
        a = np.maximum(0, x)
        b = np.tanh(x)
        c = np.abs(x)
        expected = a + b + c
        _assert_close(prog('tri', x), expected)

    def test_template_ternary_merge(self, backend):
        """Template ops feeding a 3-operand merge: templates + N-ary compose."""
        text = _REAL + """\
op ~act : hidden -> hidden
  nonlinearity = relu

op trilinear : hidden -> hidden
  einsum = "ik,jk,jl->il"
  algebra = real

branch head : hidden -> hidden = act[q] | act[k] | act[v]
  merge = trilinear
"""
        prog = parse_ua(text, backend)
        x = np.array([[1.0, -1.0], [0.5, 2.0], [0.0, -0.5]])
        r = np.maximum(0, x)
        expected = np.einsum("ik,jk,jl->il", r, r, r)
        _assert_close(prog('head', x), expected)

    def test_binary_fold_unchanged(self, backend):
        """2-branch fan with binary merge still folds correctly (regression)."""
        text = _REAL + """\
op relu : hidden -> hidden
  nonlinearity = relu
op tanh_act : hidden -> hidden
  nonlinearity = tanh

op hadamard : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

branch pair : hidden -> hidden = relu | tanh_act
  merge = hadamard
"""
        prog = parse_ua(text, backend)
        x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        expected = np.maximum(0, x) * np.tanh(x)
        _assert_close(prog('pair', x), expected)


class TestSeqAndResidual:

    def test_seq_composition(self, backend):
        text = _REAL + """\
op relu : hidden -> hidden
  nonlinearity = relu
op tanh_act : hidden -> hidden
  nonlinearity = tanh

seq chain : hidden -> hidden = relu >> tanh_act
"""
        prog = parse_ua(text, backend)
        x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        expected = np.tanh(np.maximum(0.0, x))
        _assert_close(prog('chain', x), expected)

    def test_residual_seq(self, backend):
        text = _REAL + """\
op relu : hidden -> hidden
  nonlinearity = relu
op tanh_act : hidden -> hidden
  nonlinearity = tanh

seq skip+ : hidden -> hidden = relu >> tanh_act
  algebra = real
"""
        prog = parse_ua(text, backend)
        x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        expected = np.tanh(np.maximum(0.0, x)) + x
        _assert_close(prog('skip', x), expected)
