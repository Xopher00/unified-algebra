"""Roundtrip tests for Backend.to_wire / from_wire."""
import numpy as np
import pytest

from unialg import NumpyBackend

try:
    import jax as _jax  # noqa: F401
    from unialg import JaxBackend as _JaxBackend  # noqa: F401
    HAS_JAX = True
except (ImportError, ModuleNotFoundError):
    HAS_JAX = False

try:
    import torch as _torch  # noqa: F401
    from unialg import PytorchBackend as _PytorchBackend  # noqa: F401
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    HAS_TORCH = False


class TestNumpyWire:

    def setup_method(self):
        self.b = NumpyBackend()

    def test_1d_float64_roundtrip(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = self.b.from_wire(self.b.to_wire(arr))
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == arr.dtype
        assert result.shape == arr.shape

    def test_2d_float32_roundtrip(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = self.b.from_wire(self.b.to_wire(arr))
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == arr.dtype
        assert result.shape == arr.shape

    def test_scalar_roundtrip(self):
        # 0-d arrays serialize and deserialize as shape-(1,); shape is not preserved.
        arr = np.array([3.14], dtype=np.float64)
        result = self.b.from_wire(self.b.to_wire(arr))
        np.testing.assert_array_equal(result, arr)
        assert result.shape == (1,)

    def test_to_wire_returns_bytes(self):
        arr = np.array([1.0, 2.0])
        assert isinstance(self.b.to_wire(arr), bytes)

    def test_shape_preserved_3d(self):
        arr = np.zeros((3, 4, 5), dtype=np.float32)
        result = self.b.from_wire(self.b.to_wire(arr))
        assert result.shape == (3, 4, 5)


@pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
class TestJaxWire:

    def test_1d_float64_roundtrip(self):
        from unialg import JaxBackend
        import jax.numpy as jnp
        b = JaxBackend()
        arr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        result = b.from_wire(b.to_wire(arr))
        np.testing.assert_allclose(np.array(result), np.array(arr))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestPytorchWire:

    def test_1d_float64_roundtrip(self):
        import torch
        from unialg import PytorchBackend
        b = PytorchBackend()
        arr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = b.from_wire(b.to_wire(arr))
        torch.testing.assert_close(result, arr)
