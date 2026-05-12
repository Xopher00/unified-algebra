"""Semantics layer: typed equation handle for tensor contractions.

An ``Equation`` is the tensor-layer analog of ``Morphism``: a typed declarative
handle for a single semiring-parameterized contraction.  It refers to a parsed
``EinsumExpr`` (syntax) and a ``Semiring`` (semantics), carries no Hydra terms.

Lowering (``Equation`` → ``Morphism``) is the responsibility of
``tensors.lowering``, not this module.
"""

from __future__ import annotations
from dataclasses import dataclass

from unialg.semantics.morphisms import Morphism
from unialg.tensors.expressions import EinsumExpr
from unialg.tensors.semirings import Semiring


@dataclass(frozen=True)
class Equation:
    """A typed declarative tensor contraction over a semiring.

    Parameters
    ----------
    name:
        Human-readable label, used to derive primitive names during lowering.
    expr:
        The parsed einsum subscript expression (input/output indices).
    semiring:
        The algebraic structure (⊕, ⊗, 0, 1) and carrier type.
    nonlinearity:
        Optional morphism applied elementwise to the output after contraction.
    adjoint:
        When True, uses ``semiring.adjoint`` instead of ``semiring.times`` for
        the inner product step.  Requires ``semiring.adjoint`` to be non-None.
    """
    name: str
    expr: EinsumExpr
    semiring: Semiring
    nonlinearity: Morphism | None = None
    adjoint: bool = False
