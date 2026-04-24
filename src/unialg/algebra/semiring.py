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

import hydra.core as core
from hydra.core import Name
from hydra.dsl.meta.phantoms import record, string, float64

from unialg.terms import _RecordView, _ScalarField, record_fields, literal_value

if TYPE_CHECKING:
    from unialg.backend import Backend


# ---------------------------------------------------------------------------
# Hydra type for a semiring
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Semiring class
# ---------------------------------------------------------------------------

class Semiring(_RecordView):
    """A semiring: declaration, field access, and resolution in one object.

    The underlying Hydra record term is the source of truth. Use .term to
    retrieve it for Hydra interop.

    Construct:
        sr = Semiring("real", "add", "multiply", 0.0, 1.0)

    Wrap an existing term (e.g. from the parser):
        sr = Semiring.from_term(term)

    Example semirings:
        real     = Semiring("real",     "add",     "multiply", 0.0,         1.0)
        tropical = Semiring("tropical", "minimum", "add",      float('inf'), 0.0)
        fuzzy    = Semiring("fuzzy",    "maximum", "minimum",  0.0,         1.0)
    """

    _type_name = core.Name("ua.semiring.Semiring")
    name  = _ScalarField("name", str)
    plus  = _ScalarField("plus", str)
    times = _ScalarField("times", str)
    zero  = _ScalarField("zero", float)
    one   = _ScalarField("one", float)

    def __init__(self, name: str, plus: str, times: str, zero: float, one: float,
                 residual: str | None = None):
        super().__init__(record(self._type_name, [
            Name("name") >> string(name),
            Name("plus") >> string(plus),
            Name("times") >> string(times),
            Name("zero") >> float64(zero),
            Name("one") >> float64(one),
            Name("residual") >> string(residual or ""),
        ]).value)

    @property
    def residual(self) -> str:
        from hydra.dsl.meta.phantoms import string as phantom_string
        return literal_value(record_fields(self._term).get("residual", phantom_string("").value))

    def resolve(self, backend: "Backend") -> "ResolvedSemiring":
        """Resolve this semiring against a backend to get callable operations."""
        name = self.name
        plus_name = self.plus
        times_name = self.times
        residual_name = self.residual

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
            zero=self.zero,
            one=self.one,
            residual_name=residual_name or None,
            residual_elementwise=residual_elementwise,
        )


# ---------------------------------------------------------------------------
# Resolved semiring (runtime result of Semiring.resolve)
# ---------------------------------------------------------------------------

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
