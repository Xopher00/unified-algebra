"""Backend abstraction for tensor operations.

A backend maps operation names to their concrete implementations in a
numerical computing library (numpy, pytorch, jax). Operations come in
three categories:

  - binary: two-argument ops with both elementwise and reduction forms
    e.g. "add" -> elementwise: np.add(a, b), reduce: np.sum(arr, axis=k)
  - unary: single-argument pointwise functions
    e.g. "relu" -> np.maximum(0, x), "exp" -> np.exp(x)
  - constants: named scalar values
    e.g. "inf" -> np.inf, "pi" -> np.pi

The backend is represented as a Hydra record type, so it participates in
Hydra's type system and can be inspected, composed, and validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field as datafield
from collections.abc import Callable



# ---------------------------------------------------------------------------
# Hydra types
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Python-side types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinaryOp:
    """A binary operation's two forms: elementwise and reduction.

    ``reduce`` may be ``None`` for operations that have no natural reduction
    form (e.g. masking, conditional ops).  Attempting to use such an op as
    a semiring plus/times in a contraction will raise ``ValueError`` at
    resolve time via ``Backend.reduce()``.
    """
    elementwise: Callable        # (a, b) -> c
    reduce: Callable | None = None  # (arr, axis=k) -> arr; None = not reducible

@dataclass(frozen=True)
class UnaryOp:
    """A unary pointwise operation.

    ``fn`` is a bare ``Callable``.  For axis-aware operations (e.g. softmax
    along a specific axis) use ``functools.partial`` when registering::

        import functools, scipy.special
        backend.unary_ops["softmax_last"] = UnaryOp(
            fn=functools.partial(scipy.special.softmax, axis=-1)
        )

    For parametric operations (e.g. temperature-scaled softmax) leave the
    axis/parameter as a positional argument and declare ``param_slots`` on
    the equation — the parameter value is then injected at call time via a
    Hydra bound term.
    """
    fn: Callable            # (a, *extra_params) -> b


@dataclass
class Backend:
    """A resolved backend mapping operation names to implementations."""
    name: str
    binary_ops: dict[str, BinaryOp] = datafield(default_factory=dict)
    unary_ops: dict[str, UnaryOp] = datafield(default_factory=dict)
    constants: dict[str, object] = datafield(default_factory=dict)
    expand_dims: Callable = None     # (arr, axis) -> arr
    transpose: Callable = None       # (arr, perm) -> arr
    broadcast_copy: Callable = None  # (arr, shape) -> writable arr of given shape
    where: Callable | None = None    # (condition, x, y) -> arr  — optional masking op
    from_wire: Callable = None       # (wire_bytes) -> array — deserialize with header
    to_wire: Callable = None         # (array) -> wire_bytes — serialize with header

    def __post_init__(self):
        for attr in ("expand_dims", "transpose", "broadcast_copy", "from_wire", "to_wire"):
            if getattr(self, attr) is None:
                raise ValueError(f"Backend '{self.name}' missing structural op: {attr}")

    def elementwise(self, op_name: str) -> Callable:
        """Get the elementwise form of a binary operation."""
        return self.binary_ops[op_name].elementwise

    def reduce(self, op_name: str) -> Callable:
        """Get the reduction form of a binary operation.

        Raises ``ValueError`` if the operation was registered without a
        reduce form (e.g. a masking op registered with ``reduce=None``).
        """
        fn = self.binary_ops[op_name].reduce
        if fn is None:
            raise ValueError(
                f"Backend '{self.name}': binary op '{op_name}' has no "
                "reduction form.  It cannot be used as a semiring ⊕ or ⊗ "
                "in a contraction.  Use it as a BinaryOp in a custom "
                "equation nonlinearity or as a structural op instead."
            )
        return fn

    def unary(self, op_name: str) -> Callable:
        """Get a unary pointwise operation."""
        return self.unary_ops[op_name].fn

    def constant(self, name: str) -> object:
        """Get a named constant."""
        return self.constants[name]



# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------

def _np_from_wire(raw: bytes):
    """Deserialize wire format → numpy array."""
    import numpy as np
    i = raw.index(0)
    j = raw.index(0, i + 1)
    dtype = raw[:i].decode()
    shape = tuple(int(x) for x in raw[i + 1:j].decode().split(",") if x)
    return np.frombuffer(raw[j + 1:], dtype=dtype).reshape(shape)


def _np_to_wire(arr) -> bytes:
    """Serialize numpy array → wire format."""
    import numpy as np
    a = np.ascontiguousarray(arr)
    return a.dtype.str.encode() + b"\x00" + ",".join(str(s) for s in a.shape).encode() + b"\x00" + a.tobytes()


def numpy_backend() -> Backend:
    """Build a backend from numpy."""
    import numpy as np
    from functools import partial
    from scipy.special import softmax, expit

    return Backend(
        name="numpy",
        expand_dims=np.expand_dims,
        transpose=np.transpose,
        broadcast_copy=lambda a, shape: np.broadcast_to(a, shape).copy(),
        where=np.where,
        from_wire=_np_from_wire,
        to_wire=_np_to_wire,
        binary_ops={
            # Arithmetic
            "add":      BinaryOp(elementwise=np.add,       reduce=np.sum),
            "subtract": BinaryOp(elementwise=np.subtract,  reduce=None),
            "multiply": BinaryOp(elementwise=np.multiply,  reduce=np.prod),
            "divide":   BinaryOp(elementwise=np.divide,    reduce=None),
            "power":    BinaryOp(elementwise=np.power,     reduce=None),

            # Min / max
            "minimum":  BinaryOp(elementwise=np.minimum,   reduce=np.min),
            "maximum":  BinaryOp(elementwise=np.maximum,   reduce=np.max),

            # Log-space
            "logaddexp": BinaryOp(elementwise=np.logaddexp, reduce=np.logaddexp.reduce),
        },
        unary_ops={
            # Activations
            "relu":     UnaryOp(fn=lambda x: np.maximum(0, x)),
            "sigmoid":  UnaryOp(fn=expit),
            "tanh":     UnaryOp(fn=np.tanh),
            "softmax":  UnaryOp(fn=partial(softmax, axis=-1)),
            "softplus": UnaryOp(fn=lambda x: np.log1p(np.exp(x))),

            # Elementary
            "exp":      UnaryOp(fn=np.exp),
            "log":      UnaryOp(fn=np.log),
            "log1p":    UnaryOp(fn=np.log1p),
            "sqrt":     UnaryOp(fn=np.sqrt),
            "abs":      UnaryOp(fn=np.abs),
            "neg":      UnaryOp(fn=np.negative),
            "reciprocal": UnaryOp(fn=np.reciprocal),
            "sign":     UnaryOp(fn=np.sign),
            "square":   UnaryOp(fn=np.square),

            # Trig
            "sin":      UnaryOp(fn=np.sin),
            "cos":      UnaryOp(fn=np.cos),
        },
        constants={
            "inf":  np.inf,
            "ninf": -np.inf,
            "pi":   np.pi,
            "e":    np.e,
        },
    )


# ---------------------------------------------------------------------------
# pytorch backend
# ---------------------------------------------------------------------------

def pytorch_backend() -> Backend:
    """Build a backend from pytorch."""
    import torch

    from functools import partial

    return Backend(
        name="pytorch",
        expand_dims=lambda a, axis: a.unsqueeze(axis),
        transpose=lambda a, perm: a.permute(perm),
        broadcast_copy=lambda a, shape: a.expand(shape).clone(),
        where=torch.where,
        from_wire=lambda raw: torch.from_numpy(_np_from_wire(raw).copy()),
        to_wire=lambda a: _np_to_wire(a.detach().cpu().numpy()),
        binary_ops={
            # Arithmetic
            "add":      BinaryOp(elementwise=torch.add,  reduce=lambda a, axis: torch.sum(a, dim=axis)),
            "subtract": BinaryOp(elementwise=torch.sub,  reduce=None),
            "multiply": BinaryOp(elementwise=torch.mul,  reduce=lambda a, axis: torch.prod(a, dim=axis)),
            "divide":   BinaryOp(elementwise=torch.div,  reduce=None),
            "power":    BinaryOp(elementwise=torch.pow,  reduce=None),

            # Min / max
            "minimum":  BinaryOp(elementwise=torch.minimum, reduce=lambda a, axis: torch.amin(a, dim=axis)),
            "maximum":  BinaryOp(elementwise=torch.maximum, reduce=lambda a, axis: torch.amax(a, dim=axis)),

            # Log-space
            "logaddexp": BinaryOp(elementwise=torch.logaddexp, reduce=lambda a, axis: torch.logsumexp(a, dim=axis)),
        },
        unary_ops={
            # Activations
            "relu":     UnaryOp(fn=torch.relu),
            "sigmoid":  UnaryOp(fn=torch.sigmoid),
            "tanh":     UnaryOp(fn=torch.tanh),
            "softmax":  UnaryOp(fn=partial(torch.nn.functional.softmax, dim=-1)),
            "softplus": UnaryOp(fn=torch.nn.functional.softplus),

            # Elementary
            "exp":      UnaryOp(fn=torch.exp),
            "log":      UnaryOp(fn=torch.log),
            "log1p":    UnaryOp(fn=torch.log1p),
            "sqrt":     UnaryOp(fn=torch.sqrt),
            "abs":      UnaryOp(fn=torch.abs),
            "neg":      UnaryOp(fn=torch.neg),
            "reciprocal": UnaryOp(fn=torch.reciprocal),
            "sign":     UnaryOp(fn=torch.sign),
            "square":   UnaryOp(fn=torch.square),

            # Trig
            "sin":      UnaryOp(fn=torch.sin),
            "cos":      UnaryOp(fn=torch.cos),
        },
        constants={
            "inf":  float("inf"),
            "ninf": float("-inf"),
            "pi":   3.141592653589793,
            "e":    2.718281828459045,
        },
    )
