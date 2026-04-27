"""N-ary merge and composition integration tests across backends.

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


class TestMergeChain:

    def test_merge_chain_two_contractions(self, backend):
        """3-branch with score ~> softmax ~> mix: bare nonlinearity between contractions."""
        text = _REAL + """\
op relu : hidden -> hidden
  nonlinearity = relu
op tanh_act : hidden -> hidden
  nonlinearity = tanh
op abs_act : hidden -> hidden
  nonlinearity = abs

op score : hidden -> hidden
  einsum = "ik,jk->ij"
  algebra = real

op mix : hidden -> hidden
  einsum = "ij,jk->ik"
  algebra = real

branch head : hidden -> hidden = relu | tanh_act | abs_act
  merge = score ~> softmax ~> mix
"""
        prog = parse_ua(text, backend)
        x = np.array([[1.0, -1.0], [0.5, 2.0], [0.0, -0.5]])
        a = np.maximum(0, x)
        b = np.tanh(x)
        c = np.abs(x)
        scores = np.einsum("ik,jk->ij", a, b)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        expected = np.einsum("ij,jk->ik", probs, c)
        _assert_close(prog('head', x), expected)

    def test_merge_chain_single_step_unchanged(self, backend):
        """Single-op merge (no ~>) still uses fold semantics — regression test."""
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

    def test_merge_chain_consecutive_nonlinearities(self, backend):
        """Two bare nonlinearities in a row: score ~> relu ~> abs ~> mix."""
        text = _REAL + """\
op id1 : hidden -> hidden
  nonlinearity = abs
op id2 : hidden -> hidden
  nonlinearity = abs
op id3 : hidden -> hidden
  nonlinearity = abs

op score : hidden -> hidden
  einsum = "ik,jk->ij"
  algebra = real

op mix : hidden -> hidden
  einsum = "ij,jk->ik"
  algebra = real

branch head : hidden -> hidden = id1 | id2 | id3
  merge = score ~> relu ~> abs ~> mix
"""
        prog = parse_ua(text, backend)
        x = np.array([[1.0, -1.0], [0.5, 2.0], [0.0, -0.5]])
        a = np.abs(x)
        b = np.abs(x)
        c = np.abs(x)
        scores = np.einsum("ik,jk->ij", a, b)
        after_relu = np.maximum(0, scores)
        after_abs = np.abs(after_relu)
        expected = np.einsum("ij,jk->ik", after_abs, c)
        _assert_close(prog('head', x), expected)

    def test_merge_chain_tropical(self, backend):
        """Merge chain with tropical algebra: min-plus contractions."""
        text = _TROPICAL + """\
op id1 : node -> node
  nonlinearity = abs
op id2 : node -> node
  nonlinearity = abs
op id3 : node -> node
  nonlinearity = abs

op score : node -> node
  einsum = "i,i->i"
  algebra = tropical

op mix : node -> node
  einsum = "i,i->i"
  algebra = tropical

branch head : node -> node = id1 | id2 | id3
  merge = score ~> abs ~> mix
"""
        prog = parse_ua(text, backend)
        x = np.array([1.0, -1.0, 0.5, 2.0])
        a = np.abs(x)
        b = np.abs(x)
        c = np.abs(x)
        scores = a + b  # tropical times = add
        after_abs = np.abs(scores)
        expected = after_abs + c  # tropical times = add
        _assert_close(prog('head', x), expected)


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


class TestImportBackend:

    def test_import_numpy(self):
        """import numpy in .ua source sets the backend automatically."""
        text = """\
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op relu : hidden -> hidden
  nonlinearity = relu

op tanh_act : hidden -> hidden
  nonlinearity = tanh

seq chain : hidden -> hidden = relu >> tanh_act
"""
        prog = parse_ua(text)
        x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        expected = np.tanh(np.maximum(0.0, x))
        _assert_close(prog('chain', x), expected)

    def test_backend_kwarg_overrides_import(self):
        """Explicit backend kwarg takes precedence over import."""
        text = """\
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op relu : hidden -> hidden
  nonlinearity = relu

seq chain : hidden -> hidden = relu
"""
        prog = parse_ua(text, NumpyBackend())
        x = np.array([-1.0, 0.0, 1.0])
        _assert_close(prog('chain', x), np.maximum(0, x))

    def test_no_backend_raises(self):
        """parse_ua with no backend and no import raises ValueError."""
        text = """\
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op relu : hidden -> hidden
  nonlinearity = relu

seq chain : hidden -> hidden = relu
"""
        with pytest.raises(ValueError, match="No backend specified"):
            parse_ua(text)


class TestMergeAndPathCoexistence:
    """Regression: equation used in both fan merge and seq path must work.

    Before the fix, equations in merge_names were globally resolved with
    the merge calling convention (expects list), breaking their use in paths
    (expects positional args).
    """

    _PROG = """\
import numpy

algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op proj : hidden -> hidden
  nonlinearity = abs

op hadamard : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

branch head : hidden -> hidden = proj | proj
  merge = hadamard

seq chain : hidden -> hidden = proj >> proj
"""

    def test_fan_merge_works(self):
        """hadamard in merge: merge([abs(x), abs(x)]) = abs(x) * abs(x)."""
        prog = parse_ua(self._PROG, NumpyBackend())
        x = np.array([1.0, -2.0, 3.0])
        result = prog('head', x)
        ax = np.abs(x)
        np.testing.assert_allclose(result, ax * ax)

    def test_path_not_using_merge_equation_works(self):
        """Path with non-merge equations still compiles after merge registration."""
        prog = parse_ua(self._PROG, NumpyBackend())
        x = np.array([1.0, -2.0, 3.0])
        result = prog('chain', x)
        np.testing.assert_allclose(result, np.abs(np.abs(x)))

    def test_merge_equation_callable_standalone(self):
        """hadamard as a standalone equation still works via standard convention."""
        prog = parse_ua(self._PROG, NumpyBackend())
        a = np.array([2.0, 3.0])
        b = np.array([4.0, 5.0])
        result = prog('hadamard', a, b)
        np.testing.assert_allclose(result, a * b)
