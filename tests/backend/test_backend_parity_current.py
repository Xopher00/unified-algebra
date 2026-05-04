"""Backend parity tests using current cell-DSL syntax.

Migrated from test_backend_parity.py which uses legacy seq/branch grammar.
Covers: seq composition via cell >, import-backend mechanism, residual skip.
"""

import numpy as np
import pytest

from unialg import NumpyBackend, parse_ua

try:
    import torch
    from unialg import PytorchBackend
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


_NUMPY_BASE = """\
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op relu : hidden -> hidden
  nonlinearity = relu

op tanh_act : hidden -> hidden
  nonlinearity = tanh
"""

_TORCH_BASE = """\
import pytorch
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op relu : hidden -> hidden
  nonlinearity = relu

op tanh_act : hidden -> hidden
  nonlinearity = tanh
"""


class TestSeqCompositionNumpy:

    def test_seq_cell(self):
        prog = parse_ua(_NUMPY_BASE + "\ncell chain : hidden -> hidden = relu > tanh_act\n")
        x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        expected = np.tanh(np.maximum(0.0, x))
        np.testing.assert_allclose(prog("chain", x), expected, rtol=1e-6)

    def test_residual_via_python_addition(self):
        prog = parse_ua(_NUMPY_BASE + "\ncell chain : hidden -> hidden = relu > tanh_act\n")
        x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        result = prog("chain", x) + x
        expected = np.tanh(np.maximum(0.0, x)) + x
        np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestSeqCompositionTorch:

    def test_seq_cell(self):
        torch.set_default_dtype(torch.float64)
        prog = parse_ua(_TORCH_BASE + "\ncell chain : hidden -> hidden = relu > tanh_act\n")
        x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        expected = torch.tanh(torch.relu(x))
        torch.testing.assert_close(prog("chain", x), expected)

    def test_residual_via_python_addition(self):
        torch.set_default_dtype(torch.float64)
        prog = parse_ua(_TORCH_BASE + "\ncell chain : hidden -> hidden = relu > tanh_act\n")
        x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        result = prog("chain", x) + x
        expected = torch.tanh(torch.relu(x)) + x
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestHadamardTorch:

    def test_hadamard_standalone(self):
        torch.set_default_dtype(torch.float64)
        text = _TORCH_BASE.replace("op relu", "op hadamard : hidden -> hidden\n  einsum = \"i,i->i\"\n  algebra = real\n\nop relu")
        text = """\
import pytorch
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op hadamard : hidden -> hidden
  einsum = "i,i->i"
  algebra = real
"""
        prog = parse_ua(text)
        a = torch.tensor([2.0, 3.0, 4.0])
        b = torch.tensor([5.0, 6.0, 7.0])
        torch.testing.assert_close(prog("hadamard", a, b), a * b)


class TestImportBackend:

    def test_import_numpy(self):
        prog = parse_ua(_NUMPY_BASE + "\ncell chain : hidden -> hidden = relu > tanh_act\n")
        x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        expected = np.tanh(np.maximum(0.0, x))
        np.testing.assert_allclose(prog("chain", x), expected, rtol=1e-6)

    def test_backend_kwarg_overrides_import(self):
        prog = parse_ua(_NUMPY_BASE + "\ncell single : hidden -> hidden = relu\n", NumpyBackend())
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(prog("single", x), np.maximum(0, x))

    def test_no_backend_raises(self):
        text = """\
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op relu : hidden -> hidden
  nonlinearity = relu
cell single : hidden -> hidden = relu
"""
        with pytest.raises(ValueError, match="No backend specified"):
            parse_ua(text)


class TestHadamardNumpy:

    def test_hadamard_standalone(self):
        text = """\
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op hadamard : hidden -> hidden
  einsum = "i,i->i"
  algebra = real
"""
        prog = parse_ua(text)
        a = np.array([2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0])
        np.testing.assert_allclose(prog("hadamard", a, b), a * b, rtol=1e-6)
