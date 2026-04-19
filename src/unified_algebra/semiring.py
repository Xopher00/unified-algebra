"""User-defined semiring parameterisation for unified-algebra.

A semiring is defined by naming two binary operations (plus/⊕ and times/⊗)
and their identity elements (zero and one). The operation names reference
entries in a Backend — the user never passes callables directly.

The semiring is represented as a Hydra record term so it participates in
Hydra's type system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import unified_algebra._hydra_setup  # noqa: F401 — must run before hydra imports
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


def _record_fields(term: core.Term) -> dict[str, core.Term]:
    """Extract a Hydra record's fields as a name -> term dict."""
    return {f.name.value: f.term for f in term.value.fields}


def _string_value(term: core.Term) -> str:
    """Extract a plain string from a Hydra TermLiteral(LiteralString(...))."""
    return term.value.value


def _float_value(term: core.Term) -> float:
    """Extract a float from a Hydra TermLiteral(LiteralFloat(FloatValueFloat64(...)))."""
    return term.value.value


def resolve_semiring(semiring_term: core.Term, backend: Backend) -> ResolvedSemiring:
    """Resolve a semiring term against a backend to get callable operations.

    Extracts the operation names from the Hydra record and looks them up
    in the backend to produce concrete callables for contraction.
    """
    fields = _record_fields(semiring_term)
    name = _string_value(fields["name"])
    plus_name = _string_value(fields["plus"])
    times_name = _string_value(fields["times"])

    return ResolvedSemiring(
        name=name,
        plus_name=plus_name,
        times_name=times_name,
        plus_elementwise=backend.elementwise(plus_name),
        plus_reduce=backend.reduce(plus_name),
        times_elementwise=backend.elementwise(times_name),
        zero=_float_value(fields["zero"]),
        one=_float_value(fields["one"]),
    )


class ResolvedSemiring:
    """A semiring with its operations resolved against a specific backend.

    This is the runtime object used by the contraction algorithm.
    """
    __slots__ = (
        "name", "plus_name", "times_name",
        "plus_elementwise", "plus_reduce", "times_elementwise",
        "zero", "one",
    )

    def __init__(self, name, plus_name, times_name,
                 plus_elementwise, plus_reduce, times_elementwise,
                 zero, one):
        self.name = name
        self.plus_name = plus_name
        self.times_name = times_name
        self.plus_elementwise = plus_elementwise
        self.plus_reduce = plus_reduce
        self.times_elementwise = times_elementwise
        self.zero = zero
        self.one = one
