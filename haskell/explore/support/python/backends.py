"""
Backend abstraction for differential testing.

Each backend owns the full tensor lifecycle. No runtime crossing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional
import importlib
import importlib.util
import os
import sys

from hypothesis import HealthCheck, strategies as st


SCALAR = "scalar"
VECTOR = "vector"
MATRIX = "matrix"

VECTOR_DIMS = [1, 2, 4]
MATRIX_DIMS = [2, 3]

_floats = st.floats(min_value=-2, max_value=2,
                    allow_nan=False, allow_infinity=False)


HYPO = dict(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


def arch_generated_root(arch_file: str) -> str:
    """Return the generated/ directory co-located with an arch.py file."""
    return os.path.join(os.path.dirname(arch_file), "generated")


def load_generated(backend_name: str, module: str, fn: str, generated_root: str):
    """Load a generated function from a backend-specific generated directory."""
    cache_key = f"_gen_{backend_name}_{module.replace('.', '_')}"
    if cache_key not in sys.modules:
        file_path = os.path.join(generated_root, backend_name,
                                 module.replace(".", os.sep) + ".py")
        spec = importlib.util.spec_from_file_location(cache_key, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[cache_key] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            del sys.modules[cache_key]
            raise
    return getattr(sys.modules[cache_key], fn)


@dataclass
class BackendSpec:
    backend: "Backend"
    module: str           # generated Python module, e.g. "seed.seq"
    fn: str               # function name in that module, e.g. "fold_seq"
    reference: Optional[Callable]  # None = structural test only

    def load(self, generated_root: str):
        return load_generated(self.backend.name, self.module, self.fn, generated_root)


class Backend(ABC):
    name: str

    @abstractmethod
    def random_vector(self, draw, dim):
        """Random vector of shape (dim,)."""

    @abstractmethod
    def random_matrix(self, draw, rows, cols):
        """Random matrix of shape (rows, cols)."""

    @abstractmethod
    def fill_vector(self, dim, value):
        """Constant-fill vector with the given identity value."""

    @abstractmethod
    def allclose(self, a, b, atol=1e-5):
        """Native allclose comparison."""

    @abstractmethod
    def is_finite(self, tensor) -> bool:
        """True iff all elements are finite."""

    def __repr__(self):
        return self.name


# ── NumPy ────────────────────────────────────────────────────────────────────

class NumpyBackend(Backend):
    name = "numpy"

    def random_vector(self, draw, dim):
        import numpy as np
        return np.array([draw(_floats) for _ in range(dim)], dtype=np.float64)

    def random_matrix(self, draw, rows, cols):
        import numpy as np
        return np.array([[draw(_floats) for _ in range(cols)]
                         for _ in range(rows)], dtype=np.float64)

    def fill_vector(self, dim, value):
        import numpy as np
        return np.full(dim, value, dtype=np.float64)

    def allclose(self, a, b, atol=1e-5):
        import numpy as np
        return bool(np.allclose(a, b, atol=atol))

    def is_finite(self, tensor) -> bool:
        import numpy as np
        return bool(np.all(np.isfinite(tensor)))

    @property
    def framework(self):
        import numpy
        return numpy


# ── TensorFlow ───────────────────────────────────────────────────────────────

class TFBackend(Backend):
    name = "tensorflow"

    def random_vector(self, draw, dim):
        import tensorflow as tf
        return tf.constant([draw(_floats) for _ in range(dim)], dtype=tf.float64)

    def random_matrix(self, draw, rows, cols):
        import tensorflow as tf
        return tf.constant([[draw(_floats) for _ in range(cols)]
                            for _ in range(rows)], dtype=tf.float64)

    def fill_vector(self, dim, value):
        import tensorflow as tf
        return tf.constant([value] * dim, dtype=tf.float64)

    def allclose(self, a, b, atol=1e-5):
        import tensorflow as tf
        return bool(tf.reduce_all(tf.abs(a - b) <= atol).numpy())

    def is_finite(self, tensor) -> bool:
        import tensorflow as tf
        return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())

    @property
    def framework(self):
        import tensorflow
        return tensorflow


# ── PyTorch ──────────────────────────────────────────────────────────────────

class TorchBackend(Backend):
    name = "torch"

    def random_vector(self, draw, dim):
        import torch
        return torch.tensor([draw(_floats) for _ in range(dim)],
                            dtype=torch.float64)

    def random_matrix(self, draw, rows, cols):
        import torch
        return torch.tensor([[draw(_floats) for _ in range(cols)]
                             for _ in range(rows)], dtype=torch.float64)

    def fill_vector(self, dim, value):
        import torch
        return torch.full((dim,), value, dtype=torch.float64)

    def allclose(self, a, b, atol=1e-5):
        import torch
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float64)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float64)
        return torch.allclose(a.detach(), b.detach(), atol=atol)

    def is_finite(self, tensor) -> bool:
        import torch
        return bool(torch.all(torch.isfinite(tensor.detach())))

    @property
    def framework(self):
        import torch
        return torch
