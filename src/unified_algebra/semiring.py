"""User-defined semiring parameterisation for unified-algebra.

A semiring is defined by naming two binary operations (plus/⊕ and times/⊗)
and their identity elements (zero and one). The operation names reference
entries in a Backend — the user never passes callables directly.

The semiring is represented as a Hydra record term so it participates in
Hydra's type system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from unified_algebra._hydra_setup import record_fields, string_value, float_value  # noqa: F401
import hydra.core as core
import hydra.dsl.terms as Terms

if TYPE_CHECKING:
    from .backend import Backend


# ---------------------------------------------------------------------------
# Hydra type for a semiring
# ---------------------------------------------------------------------------

SEMIRING_TYPE_NAME = core.Name("ua.semiring.Semiring")


# ---------------------------------------------------------------------------
# Semiring construction
# ---------------------------------------------------------------------------

def semiring(name: str, plus: str, times: str, zero: float, one: float) -> core.Term:
    """Create a semiring as a Hydra record term.

    Args:
        name:  identifier for this semiring (e.g. "real", "tropical")
        plus:  name of the ⊕ binary op in the backend (e.g. "add", "minimum")
        times: name of the ⊗ binary op in the backend (e.g. "multiply", "add")
        zero:  additive identity
        one:   multiplicative identity

    Returns:
        A Hydra TermRecord representing the semiring.

    Example:
        real = semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        tropical = semiring("tropical", plus="minimum", times="add", zero=float('inf'), one=0.0)
        fuzzy = semiring("fuzzy", plus="maximum", times="minimum", zero=0.0, one=1.0)
    """
    return Terms.record(SEMIRING_TYPE_NAME, [
        Terms.field("name", Terms.string(name)),
        Terms.field("plus", Terms.string(plus)),
        Terms.field("times", Terms.string(times)),
        Terms.field("zero", Terms.float64(zero)),
        Terms.field("one", Terms.float64(one)),
    ])


def resolve_semiring(semiring_term: core.Term, backend: Backend) -> ResolvedSemiring:
    """Resolve a semiring term against a backend to get callable operations.

    Extracts the operation names from the Hydra record and looks them up
    in the backend to produce concrete callables for contraction.
    """
    fields = record_fields(semiring_term)
    name = string_value(fields["name"])
    plus_name = string_value(fields["plus"])
    times_name = string_value(fields["times"])

    return ResolvedSemiring(
        name=name,
        plus_name=plus_name,
        times_name=times_name,
        plus_elementwise=backend.elementwise(plus_name),
        plus_reduce=backend.reduce(plus_name),
        times_elementwise=backend.elementwise(times_name),
        times_reduce=backend.reduce(times_name),
        zero=float_value(fields["zero"]),
        one=float_value(fields["one"]),
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
