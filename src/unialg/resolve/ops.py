"""Semiring resolution: term → callable operations.

Converts a semiring Hydra record term (from algebra.semiring()) into a
ResolvedSemiring with concrete callables looked up from the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import unialg.views as vw

if TYPE_CHECKING:
    import hydra.core as core
    from unialg.backend import Backend


def resolve_semiring(semiring_term: "core.Term", backend: "Backend") -> "ResolvedSemiring":
    """Resolve a semiring term against a backend to get callable operations.

    Extracts the operation names from the Hydra record and looks them up
    in the backend to produce concrete callables for contraction.
    """
    v = vw.SemiringView(semiring_term)
    name = v.name
    plus_name = v.plus
    times_name = v.times
    residual_name = v.residual

    residual_elementwise = None
    if residual_name:
        residual_elementwise = backend.elementwise(residual_name)

    return ResolvedSemiring(
        name=name,
        plus_name=plus_name,
        times_name=times_name,
        plus_elementwise=backend.elementwise(plus_name),
        plus_reduce=backend.reduce(plus_name),
        times_elementwise=backend.elementwise(times_name),
        times_reduce=backend.reduce(times_name),
        zero=v.zero,
        one=v.one,
        residual_name=residual_name or None,
        residual_elementwise=residual_elementwise,
    )


@dataclass(frozen=True, slots=True)
class ResolvedSemiring:
    """A semiring with its operations resolved against a specific backend."""
    name: str
    plus_name: str
    times_name: str
    plus_elementwise: object
    plus_reduce: object
    times_elementwise: object
    times_reduce: object
    zero: float
    one: float
    residual_name: str | None = None
    residual_elementwise: object | None = None
