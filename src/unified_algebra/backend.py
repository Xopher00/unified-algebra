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
    """A binary operation's two forms: elementwise and reduction."""
    elementwise: Callable   # (a, b) -> c
    reduce: Callable        # (arr, axis=k) -> arr

@dataclass(frozen=True)
class UnaryOp:
    """A unary pointwise operation."""
    fn: Callable            # (a) -> b


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

    def __post_init__(self):
        for attr in ("expand_dims", "transpose", "broadcast_copy"):
            if getattr(self, attr) is None:
                raise ValueError(f"Backend '{self.name}' missing structural op: {attr}")

    def elementwise(self, op_name: str) -> Callable:
        """Get the elementwise form of a binary operation."""
        return self.binary_ops[op_name].elementwise

    def reduce(self, op_name: str) -> Callable:
        """Get the reduction form of a binary operation."""
        return self.binary_ops[op_name].reduce

    def unary(self, op_name: str) -> Callable:
        """Get a unary pointwise operation."""
        return self.unary_ops[op_name].fn

    def constant(self, name: str) -> object:
        """Get a named constant."""
        return self.constants[name]



# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------

def numpy_backend() -> Backend:
    """Build a backend from numpy."""
    import numpy as np
    from scipy.special import softmax, expit

    return Backend(
        name="numpy",
        expand_dims=np.expand_dims,
        transpose=np.transpose,
        broadcast_copy=lambda a, shape: np.broadcast_to(a, shape).copy(),
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
            "softmax":  UnaryOp(fn=softmax),
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

    return Backend(
        name="pytorch",
        expand_dims=lambda a, axis: a.unsqueeze(axis),
        transpose=lambda a, perm: a.permute(perm),
        broadcast_copy=lambda a, shape: a.expand(shape).clone(),
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
            "softplus": UnaryOp(fn=lambda x: torch.nn.functional.softplus(x)),

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
