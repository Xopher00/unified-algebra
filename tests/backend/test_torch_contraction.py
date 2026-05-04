"""Torch backend contraction regression tests.

Covers the fix for empty reduced_dims: torch.sum(x, dim=()) reduces all
dimensions to a scalar, unlike numpy which treats axis=() as a no-op.
The fix is in _make_compute_sum which now skips the reduce call when
reduced_dims is empty.
"""

import numpy as np
import pytest

from unialg import NumpyBackend, Semiring
from unialg.algebra.contraction import compile_einsum, semiring_contract

try:
    import torch
    from unialg import PytorchBackend
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@pytest.fixture
def torch_backend():
    torch.set_default_dtype(torch.float64)
    return PytorchBackend()


@pytest.fixture
def numpy_backend():
    return NumpyBackend()


@pytest.fixture(params=["numpy", "torch"])
def any_backend(request, numpy_backend, torch_backend):
    return numpy_backend if request.param == "numpy" else torch_backend


def _real(backend):
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0).resolve(backend)


def _tropical(backend):
    return Semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0).resolve(backend)


def _to_backend(arr, backend):
    """Convert numpy array to backend tensor."""
    if isinstance(backend, PytorchBackend):
        return torch.from_numpy(np.asarray(arr))
    return np.asarray(arr)


def _to_numpy(result):
    """Convert result to numpy for assertions."""
    if isinstance(result, np.ndarray):
        return result
    return result.numpy()


class TestEmptyReducedDims:
    """Regression: einsum with no reduced vars must preserve shape."""

    def test_elementwise_multiply_preserves_shape(self, any_backend):
        eq = compile_einsum("i,i->i")
        a = _to_backend([1.0, 2.0, 3.0], any_backend)
        b = _to_backend([4.0, 5.0, 6.0], any_backend)
        sr = _real(any_backend)
        result = semiring_contract(eq, [a, b], sr, any_backend)
        assert result.shape == (3,), f"Expected (3,), got {result.shape}"
        np.testing.assert_allclose(_to_numpy(result), [4.0, 10.0, 18.0])

    def test_identity_passthrough(self, any_backend):
        eq = compile_einsum("i->i")
        a = _to_backend([1.0, 2.0, 3.0], any_backend)
        sr = _real(any_backend)
        result = semiring_contract(eq, [a], sr, any_backend)
        assert result.shape == (3,), f"Expected (3,), got {result.shape}"

    def test_elementwise_2d(self, any_backend):
        eq = compile_einsum("ij,ij->ij")
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]])
        a = _to_backend(a_np, any_backend)
        b = _to_backend(b_np, any_backend)
        sr = _real(any_backend)
        result = semiring_contract(eq, [a, b], sr, any_backend)
        assert result.shape == (2, 2), f"Expected (2, 2), got {result.shape}"
        np.testing.assert_allclose(_to_numpy(result), a_np * b_np)

    def test_tropical_elementwise(self, any_backend):
        eq = compile_einsum("i,i->i")
        a = _to_backend([1.0, 2.0, 3.0], any_backend)
        b = _to_backend([4.0, 5.0, 6.0], any_backend)
        sr = _tropical(any_backend)
        result = semiring_contract(eq, [a, b], sr, any_backend)
        assert result.shape == (3,), f"Expected (3,), got {result.shape}"
        np.testing.assert_allclose(_to_numpy(result), [5.0, 7.0, 9.0])  # tropical times = add


class TestTorchNumpyParity:
    """Cross-backend numerical parity for contraction operations."""

    def _parity(self, einsum_str, args_np, numpy_backend, torch_backend):
        eq = compile_einsum(einsum_str)
        sr_np = _real(numpy_backend)
        sr_pt = _real(torch_backend)
        args_pt = [torch.from_numpy(a) for a in args_np]
        r_np = semiring_contract(eq, args_np, sr_np, numpy_backend)
        r_pt = semiring_contract(eq, args_pt, sr_pt, torch_backend)
        np.testing.assert_allclose(r_np, r_pt.numpy(), rtol=1e-12)

    def test_matmul_parity(self, numpy_backend, torch_backend):
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 1.0])
        self._parity("ij,j->i", [W, x], numpy_backend, torch_backend)

    def test_hadamard_parity(self, numpy_backend, torch_backend):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0, 8.0])
        self._parity("i,i->i", [a, b], numpy_backend, torch_backend)

    def test_outer_product_parity(self, numpy_backend, torch_backend):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0])
        self._parity("i,j->ij", [a, b], numpy_backend, torch_backend)


class TestTransformerEndToEnd:
    """End-to-end transformer example on torch backend."""

    def test_transformer_torch(self, torch_backend):
        from unialg import parse_ua
        from pathlib import Path

        with open(Path(__file__).parents[2] / "examples" / "transformer.ua") as f:
            src = f.read().replace("import numpy", "import pytorch")

        prog = parse_ua(src)
        np.random.seed(42)
        d = 8
        W_q = np.random.randn(d, d) * 0.1
        W_k = np.random.randn(d, d) * 0.1
        W_v = np.random.randn(d, d) * 0.1
        W_up = np.random.randn(d, d) * 0.1
        W_down = np.random.randn(d, d) * 0.1
        x_np = np.random.randn(d)

        W_q_t = torch.from_numpy(W_q)
        W_k_t = torch.from_numpy(W_k)
        W_v_t = torch.from_numpy(W_v)
        W_up_t = torch.from_numpy(W_up)
        W_down_t = torch.from_numpy(W_down)
        x_t = torch.from_numpy(x_np)

        q = prog("proj", W_q_t, x_t)
        k = prog("proj", W_k_t, x_t)
        v = prog("proj", W_v_t, x_t)
        assert q.shape == (d,)
        assert k.shape == (d,)

        scores = prog("hadamard", q, k)
        assert scores.shape == (d,), f"hadamard returned {scores.shape}, expected ({d},)"

        attn_weights = prog("attn_gate", scores)
        assert attn_weights.shape == (d,)

        context = prog("hadamard", attn_weights, v)
        h = context + x_t
        h_up = prog("proj", W_up_t, h)
        h_act = prog("ffn_act", h_up)
        h_down = prog("proj", W_down_t, h_act)
        output = h_down + h
        assert output.shape == (d,)

        # Cross-check against numpy oracle
        q_o = W_q @ x_np
        k_o = W_k @ x_np
        sc = q_o * k_o
        sw = np.exp(sc - sc.max()); sw /= sw.sum()
        v_o = W_v @ x_np
        ctx = sw * v_o
        h_o = ctx + x_np

        def gelu_np(x):
            return x * (1 / (1 + np.exp(-1.702 * x)))

        act_o = gelu_np(W_up @ h_o)
        out_o = (W_down @ act_o) + h_o

        np.testing.assert_allclose(output.numpy(), out_o, rtol=1e-6)
