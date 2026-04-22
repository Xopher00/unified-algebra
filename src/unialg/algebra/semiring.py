"""User-defined semiring parameterisation for unified-algebra.

A semiring is defined by naming two binary operations (plus/⊕ and times/⊗)
and their identity elements (zero and one). The operation names reference
entries in a Backend — the user never passes callables directly.

The semiring is represented as a Hydra record term so it participates in
Hydra's type system.
"""

from __future__ import annotations

import hydra.core as core
from hydra.core import Name
from hydra.dsl.meta.phantoms import record, string, float64


# ---------------------------------------------------------------------------
# Hydra type for a semiring
# ---------------------------------------------------------------------------

SEMIRING_TYPE_NAME = core.Name("ua.semiring.Semiring")


# ---------------------------------------------------------------------------
# Semiring construction
# ---------------------------------------------------------------------------

def semiring(name: str, plus: str, times: str, zero: float, one: float,
             residual: str | None = None) -> core.Term:
    """Create a semiring as a Hydra record term.

    Args:
        name:     identifier for this semiring (e.g. "real", "tropical")
        plus:     name of the ⊕ binary op in the backend (e.g. "add", "minimum")
        times:    name of the ⊗ binary op in the backend (e.g. "multiply", "add")
        zero:     additive identity
        one:      multiplicative identity
        residual: optional name of the ⊘ binary op (right adjoint of ⊗).
                  Satisfies: a ⊗ b ≤ c ⟺ b ≤ a ⊘ c.
                  Examples: "divide" for real, "subtract" for tropical.

    Returns:
        A Hydra TermRecord representing the semiring.

    Example:
        real = semiring("real", plus="add", times="multiply", residual="divide", zero=0.0, one=1.0)
        tropical = semiring("tropical", plus="minimum", times="add", residual="subtract", zero=float('inf'), one=0.0)
        fuzzy = semiring("fuzzy", plus="maximum", times="minimum", residual="implies", zero=0.0, one=1.0)
    """
    return record(SEMIRING_TYPE_NAME, [
        Name("name") >> string(name),
        Name("plus") >> string(plus),
        Name("times") >> string(times),
        Name("zero") >> float64(zero),
        Name("one") >> float64(one),
        Name("residual") >> string(residual or ""),
    ]).value
