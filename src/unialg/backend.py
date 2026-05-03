"""Backend abstraction — maps op names to library implementations.

Backend is an ABC. NumpyApiBackend is an intermediate class for backends with
a numpy-compatible API (numpy, cupy, jax); it builds the shared op tables from
a `lib` module. Concrete subclasses are directly instantiable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Callable
from functools import partial


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Backend(ABC):
    """Abstract base for all tensor backends."""

    # Unary ops where op name == getattr(lib, name) across numpy, torch, and jax.
    DIRECT_UNARY = ("tanh", "exp", "log", "log1p", "sqrt", "abs", "reciprocal", "sign", "square", "sin", "cos")

    @dataclass(frozen=True)
    class BinaryOp:
        """Elementwise and reduction forms of a binary operation."""
        elementwise: Callable
        reduce: Callable | None = None

    def __init__(
        self,
        name: str,
        lib,
        binary_ops: dict[str, BinaryOp],
        unary_ops: dict[str, Callable],
        constants: dict[str, object],
        expand_dims: Callable,
        transpose: Callable,
        broadcast_copy: Callable,
        where: Callable | None = None,
    ):
        self.name = name
        self._lib = lib
        self.binary_ops = binary_ops
        self.unary_ops = unary_ops
        self.constants = constants
        self.expand_dims = expand_dims
        self.transpose = transpose
        self.broadcast_copy = broadcast_copy
        self.where = where

    # ---- wire format: shared header, backend-specific serialization ----

    @staticmethod
    def _parse_wire_header(raw: bytes) -> tuple[str, tuple[int, ...], bytes]:
        i = raw.index(0)
        j = raw.index(0, i + 1)
        dtype = raw[:i].decode()
        shape = tuple(int(x) for x in raw[i + 1:j].decode().split(",") if x)
        return dtype, shape, raw[j + 1:]

    @staticmethod
    def _encode_wire_header(dtype_str: str, shape) -> bytes:
        return dtype_str.encode() + b"\x00" + ",".join(str(s) for s in shape).encode() + b"\x00"

    @abstractmethod
    def from_wire(self, raw: bytes): ...

    @abstractmethod
    def to_wire(self, arr) -> bytes: ...

    # ---- compilation and control flow ----

    @abstractmethod
    def compile(self, fn: Callable) -> Callable:
        """Wrap fn in backend-native JIT. Returns fn unchanged if unsupported."""

    def while_loop(self, cond_fn: Callable, body_fn: Callable, init_val):
        """Python while loop. JAX overrides with jax.lax.while_loop."""
        state = init_val
        while cond_fn(state):
            state = body_fn(state)
        return state

    # ---- dict lookups ----

    def elementwise(self, op_name: str) -> Callable:
        return self.binary_ops[op_name].elementwise

    def reduce(self, op_name: str) -> Callable:
        fn = self.binary_ops[op_name].reduce
        if fn is None:
            raise ValueError(
                f"Backend '{self.name}': binary op '{op_name}' has no "
                "reduction form — cannot be used as semiring +/* in a contraction."
            )
        return fn

    def unary(self, op_name: str) -> Callable:
        return self.unary_ops[op_name]

    def constant(self, name: str) -> object:
        return self.constants[name]

    def available_memory(self) -> int | None:
        """Available memory in bytes. Returns None if unknown."""
        try:
            import psutil
            return psutil.virtual_memory().available
        except ImportError:
            try:
                import os
                return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
            except (AttributeError, ValueError):
                return None


# ---------------------------------------------------------------------------
# Intermediate: backends with a numpy-compatible API (numpy, cupy, jax)
# ---------------------------------------------------------------------------

class NumpyApiBackend(Backend):
    """Base for backends whose lib mirrors numpy's API.

    Subclasses set NAME, LIB_PACKAGE, and (for the default _build_activations)
    SCIPY_PACKAGE as class attributes. Override _build_activations,
    _logaddexp_reduce, or _broadcast_copy where the backend differs.
    """

    NAME: str
    LIB_PACKAGE: str
    SCIPY_PACKAGE: str = ""

    def __init__(self):
        import importlib
        lib = importlib.import_module(self.LIB_PACKAGE)
        np_api = getattr(lib, "numpy", lib)  # jax → jax.numpy; numpy/cupy → lib itself
        Op = Backend.BinaryOp
        binary_ops = {
            "add":       Op(elementwise=np_api.add,       reduce=np_api.sum),
            "subtract":  Op(elementwise=np_api.subtract),
            "multiply":  Op(elementwise=np_api.multiply,  reduce=np_api.prod),
            "divide":    Op(elementwise=np_api.divide),
            "power":     Op(elementwise=np_api.power),
            "minimum":   Op(elementwise=np_api.minimum,   reduce=np_api.min),
            "maximum":   Op(elementwise=np_api.maximum,   reduce=np_api.max),
            "logaddexp": Op(elementwise=np_api.logaddexp, reduce=self._logaddexp_reduce(np_api)),
        }
        unary_ops = {
            **{n: getattr(np_api, n) for n in Backend.DIRECT_UNARY},
            "neg": np_api.negative,
            **self._build_activations(np_api),
        }
        super().__init__(
            name=self.NAME, lib=lib,
            binary_ops=binary_ops, unary_ops=unary_ops,
            constants={"inf": np_api.inf, "ninf": -np_api.inf, "pi": np_api.pi, "e": np_api.e},
            expand_dims=np_api.expand_dims,
            transpose=np_api.transpose,
            broadcast_copy=self._broadcast_copy(np_api),
            where=np_api.where,
        )
        self._argmax = np_api.argmax

    def _build_activations(self, np_api) -> dict[str, Callable]:
        """Default: scipy-style. Override for backends with their own unary op set."""
        import importlib
        scipy = importlib.import_module(self.SCIPY_PACKAGE)
        return {
            "relu": lambda x: np_api.maximum(0, x), "sigmoid": scipy.expit,
            "softmax": partial(scipy.softmax, axis=-1),
            "softplus": lambda x: np_api.log1p(np_api.exp(x)),
        }

    def _logaddexp_reduce(self, np_api) -> Callable:
        return np_api.logaddexp.reduce

    def _broadcast_copy(self, np_api) -> Callable:
        return lambda a, shape: np_api.broadcast_to(a, shape).copy()

    def argmax(self, tensor, axis):
        return self._argmax(tensor, axis=axis)


# ---------------------------------------------------------------------------
# Concrete backends
# ---------------------------------------------------------------------------

class NumpyBackend(NumpyApiBackend):
    NAME = "numpy"
    LIB_PACKAGE = "numpy"
    SCIPY_PACKAGE = "scipy.special"

    def __init__(self, *, jit: Callable | None = None):
        self._jit = jit
        super().__init__()

    def from_wire(self, raw: bytes):
        dtype, shape, data = self._parse_wire_header(raw)
        return self._lib.frombuffer(data, dtype=dtype).reshape(shape)

    def to_wire(self, arr) -> bytes:
        a = self._lib.ascontiguousarray(arr)
        return self._encode_wire_header(a.dtype.str, a.shape) + a.tobytes()

    def compile(self, fn):
        return self._jit(fn) if self._jit is not None else fn


class CupyBackend(NumpyApiBackend):
    NAME = "cupy"
    LIB_PACKAGE = "cupy"
    SCIPY_PACKAGE = "cupyx.scipy.special"

    def from_wire(self, raw: bytes):
        import numpy as np
        dtype, shape, data = self._parse_wire_header(raw)
        return self._lib.asarray(np.frombuffer(data, dtype=dtype).reshape(shape))

    def to_wire(self, arr) -> bytes:
        a = arr.get()
        return self._encode_wire_header(a.dtype.str, a.shape) + a.tobytes()

    def compile(self, fn):
        return fn  # CuPy kernels are JIT-compiled per-op

    def available_memory(self) -> int | None:
        try:
            import cupy
            free, _ = cupy.cuda.Device().mem_info
            return free
        except Exception:
            return super().available_memory()


class JaxBackend(NumpyApiBackend):
    NAME = "jax"
    LIB_PACKAGE = "jax"

    def _build_activations(self, np_api):
        import jax
        return {
            "relu": jax.nn.relu, "sigmoid": jax.nn.sigmoid,
            "softmax": partial(jax.nn.softmax, axis=-1),
            "softplus": jax.nn.softplus,
        }

    def _logaddexp_reduce(self, np_api):
        import jax
        # Take axis as a runtime parameter — same contract as torch's reduce form.
        # Hardcoding axis=-1 here (via partial) collides with callers that pass axis
        # explicitly (semiring_contract.plus_reduce(term, reduced_dims)).
        return lambda a, axis: jax.scipy.special.logsumexp(a, axis=axis)

    def _broadcast_copy(self, np_api):
        return lambda a, shape: np_api.array(np_api.broadcast_to(a, shape))

    def from_wire(self, raw: bytes):
        dtype, shape, data = self._parse_wire_header(raw)
        return self._lib.numpy.frombuffer(data, dtype=dtype).reshape(shape)

    def to_wire(self, arr) -> bytes:
        a = self._lib.device_get(arr)
        return self._encode_wire_header(a.dtype.str, a.shape) + a.tobytes()

    def compile(self, fn):
        return self._lib.jit(fn)

    def while_loop(self, cond_fn, body_fn, init_val):
        return self._lib.lax.while_loop(cond_fn, body_fn, init_val)

    def available_memory(self) -> int | None:
        try:
            import jax
            stats = jax.devices()[0].memory_stats()
            if stats:
                return stats.get('bytes_limit', 0) - stats.get('bytes_in_use', 0)
        except Exception:
            pass
        return super().available_memory()


class PytorchBackend(Backend):
    """Torch uses its own naming conventions — does not inherit NumpyApiBackend."""

    _DTYPE_MAP = {
        "float32": "float32", "<f4": "float32",
        "float64": "float64", "<f8": "float64",
        "int32":   "int32",   "<i4": "int32",
        "int64":   "int64",   "<i8": "int64",
    }

    def __init__(self):
        import torch
        super().__init__(
            name="pytorch", lib=torch,
            expand_dims=lambda a, axis: a.unsqueeze(axis),
            transpose=lambda a, perm: a.permute(perm),
            broadcast_copy=lambda a, shape: a.expand(shape).clone(),
            where=torch.where,
            binary_ops={
                "add":       Backend.BinaryOp(elementwise=torch.add,       reduce=lambda a, axis: torch.sum(a, dim=axis)),
                "subtract":  Backend.BinaryOp(elementwise=torch.sub),
                "multiply":  Backend.BinaryOp(elementwise=torch.mul,       reduce=lambda a, axis: torch.prod(a, dim=axis)),
                "divide":    Backend.BinaryOp(elementwise=torch.div),
                "power":     Backend.BinaryOp(elementwise=torch.pow),
                "minimum":   Backend.BinaryOp(elementwise=torch.minimum,   reduce=lambda a, axis: torch.amin(a, dim=axis)),
                "maximum":   Backend.BinaryOp(elementwise=torch.maximum,   reduce=lambda a, axis: torch.amax(a, dim=axis)),
                "logaddexp": Backend.BinaryOp(elementwise=torch.logaddexp, reduce=lambda a, axis: torch.logsumexp(a, dim=axis)),
            },
            unary_ops={
                **{name: getattr(torch, name) for name in Backend.DIRECT_UNARY},
                "relu":     torch.relu,
                "sigmoid":  torch.sigmoid,
                "softmax":  partial(torch.nn.functional.softmax, dim=-1),
                "softplus": torch.nn.functional.softplus,
                "neg":      torch.neg,
            },
            constants={"inf": float("inf"), "ninf": float("-inf"), "pi": 3.141592653589793, "e": 2.718281828459045},
        )

    def from_wire(self, raw: bytes):
        dtype_str, shape, data = self._parse_wire_header(raw)
        dtype = getattr(self._lib, self._DTYPE_MAP[dtype_str])
        return self._lib.frombuffer(bytearray(data), dtype=dtype).reshape(shape).clone()

    def to_wire(self, arr) -> bytes:
        a = arr.contiguous().detach().cpu()
        dtype_str = str(a.dtype).replace("torch.", "")
        return self._encode_wire_header(dtype_str, a.shape) + a.untyped_storage().tobytes()

    def argmax(self, tensor, axis):
        return self._lib.argmax(tensor, dim=axis)

    def compile(self, fn):
        return self._lib.compile(fn)

    def available_memory(self) -> int | None:
        try:
            if self._lib.cuda.is_available():
                free, _ = self._lib.cuda.mem_get_info()
                return free
        except Exception:
            pass
        return super().available_memory()


# ---------------------------------------------------------------------------
# Backend name resolution
# ---------------------------------------------------------------------------

_BACKEND_MAP = {
    'numpy': 'NumpyBackend',
    'jax': 'JaxBackend',
    'pytorch': 'PytorchBackend',
    'cupy': 'CupyBackend',
}


def resolve_backend(name: str) -> Backend:
    """Instantiate a backend by name. Raises ValueError for unknown names."""
    cls_name = _BACKEND_MAP.get(name)
    if cls_name is None:
        raise ValueError(
            f"Unknown backend {name!r} — available: {list(_BACKEND_MAP)}")
    import sys
    return getattr(sys.modules[__name__], cls_name)()
