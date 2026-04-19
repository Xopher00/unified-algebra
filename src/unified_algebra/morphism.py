"""Morphisms: typed tensor operations registered as Hydra Primitives.

``parametric_morphism`` wraps a semiring contraction (learnable weights).
``pointwise_morphism`` wraps a unary backend op (activation function).
Both are returned as Hydra ``Primitive`` objects ready to be inserted into a
``Graph.primitives`` dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import unified_algebra._hydra_setup  # noqa: F401 — must precede hydra imports
import hydra.core as core
from hydra.dsl.prims import prim1, prim2

from .sort import tensor_coder
from .contraction import compile_equation, semiring_contract

if TYPE_CHECKING:
    from hydra.graph import Primitive
    from .semiring import ResolvedSemiring
    from .backend import Backend


def parametric_morphism(
    name: str,
    equation_str: str,
    resolved_semiring: ResolvedSemiring,
    backend: Backend,
) -> Primitive:
    """Return a Hydra Primitive for a parametric (weight-bearing) morphism.

    The primitive takes two tensor arguments ``(x, W)`` and returns a tensor
    computed via semiring contraction.  The equation is compiled once at
    construction time and captured in the closure.

    Args:
        name:              identifier (becomes ``ua.morphism.<name>``).
        equation_str:      einsum-style equation, e.g. ``"ij,j->i"``.
        resolved_semiring: semiring with backend-resolved callables.
        backend:           structural tensor ops (expand_dims, transpose).
    """
    eq = compile_equation(equation_str)
    sr = resolved_semiring
    coder = tensor_coder()

    return prim2(
        name=core.Name(f"ua.morphism.{name}"),
        compute=lambda x, W: semiring_contract(eq, [W, x], sr, backend),
        variables=[],
        input1=coder,
        input2=coder,
        output=coder,
    )


def pointwise_morphism(
    name: str,
    op_name: str,
    backend: Backend,
) -> Primitive:
    """Return a Hydra Primitive for a pointwise (non-parametric) morphism.

    The primitive takes one tensor argument and applies a unary backend op.

    Args:
        name:     identifier (becomes ``ua.pointwise.<name>``).
        op_name:  key in ``backend.unary_ops``, e.g. ``"relu"``.
        backend:  backend providing the unary callable.
    """
    fn = backend.unary(op_name)
    coder = tensor_coder()

    return prim1(
        name=core.Name(f"ua.pointwise.{name}"),
        compute=lambda x: fn(x),
        variables=[],
        input1=coder,
        output=coder,
    )
