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

from unialg.terms import _RecordView, _ScalarField

if TYPE_CHECKING:
    from unialg.backend import Backend


class Semiring(_RecordView):
    """A semiring: declaration, field access, validation, and resolution.

    The underlying Hydra record term is the source of truth. Use .term to
    retrieve it for Hydra interop.

    Construct:
        sr = Semiring("real", "add", "multiply", 0.0, 1.0)

    Wrap an existing term (e.g. from the parser):
        sr = Semiring.from_term(term)

    Example semirings:
        real     = Semiring("real",     "add",     "multiply", 0.0,          1.0)
        tropical = Semiring("tropical", "minimum", "add",      float('inf'), 0.0)
        fuzzy    = Semiring("fuzzy",    "maximum", "minimum",  0.0,          1.0)
    """

    _type_name = core.Name("ua.semiring.Semiring")

    name     = _ScalarField("name", str)
    plus     = _ScalarField("plus", str)
    times    = _ScalarField("times", str)
    zero     = _ScalarField("zero", float)
    one      = _ScalarField("one", float)
    residual = _ScalarField("residual", str, default="")
    bottom   = _ScalarField("bottom", float, default=-10.0)
    top      = _ScalarField("top", float, default=10.0)

    @dataclass(frozen=True, slots=True)
    class Resolved:
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

    def __init__(self, name: str, plus: str, times: str, zero: float, one: float,
                 residual: str | None = None,
                 bottom: float = -10.0, top: float = 10.0):
        super().__init__(record(self._type_name, [
            Name("name") >> string(name),
            Name("plus") >> string(plus),
            Name("times") >> string(times),
            Name("zero") >> float64(zero),
            Name("one") >> float64(one),
            Name("residual") >> string(residual or ""),
            Name("bottom") >> float64(bottom),
            Name("top") >> float64(top),
        ]).value)

    def _validation_samples(self, n: int = 5, seed: int = 42) -> list:
        """Draw n triplets uniformly from [bottom, top]."""
        import random
        rng = random.Random(seed)
        return [tuple(rng.uniform(self.bottom, self.top) for _ in range(3)) for _ in range(n)]

    def resolve(self, backend: "Backend", *, samples=None) -> "Semiring.Resolved":
        """Resolve this semiring against a backend, validating laws first.

        Samples for law validation are drawn uniformly from [bottom, top]
        unless overridden via `samples`.
        """
        self.validate_laws(backend, samples or self._validation_samples())
        residual_name = self.residual
        return Semiring.Resolved(
            name=self.name,
            plus_name=self.plus,
            times_name=self.times,
            plus_elementwise=backend.elementwise(self.plus),
            plus_reduce=backend.reduce(self.plus),
            times_elementwise=backend.elementwise(self.times),
            times_reduce=backend.reduce(self.times),
            zero=self.zero,
            one=self.one,
            residual_name=residual_name or None,
            residual_elementwise=backend.elementwise(residual_name) if residual_name else None,
        )

    def validate_laws(self, backend: "Backend", samples, atol: float = 1e-9) -> None:
        """Verify ⊕ and ⊗ form a valid semiring on Python-scalar samples.

        Pulls only the elementwise ops from `backend` and evaluates the 7 axioms:
        ⊕ commutativity, ⊕ associativity, ⊕ identity, ⊗ associativity,
        ⊗ identity, 0 annihilates ⊗, ⊗ distributes over ⊕.

        `samples` is an iterable of `(a, b, c)` triplets. The caller picks a
        domain the user-defined semiring is closed over (e.g. [0, 1] for fuzzy).
        """
        plus = backend.elementwise(self.plus)
        times = backend.elementwise(self.times)
        zero, one = self.zero, self.one

        def _check(lhs, rhs, axiom: str, a, b, c):
            if lhs == rhs:  # exact equality (handles ±inf == ±inf without nan blow-up)
                return
            if abs(lhs - rhs) > atol:
                raise ValueError(
                    f"Semiring '{self.name}': {axiom} fails at (a,b,c)=({a},{b},{c}): "
                    f"lhs={lhs}, rhs={rhs}"
                )

        for a, b, c in samples:
            _check(plus(a, b),             plus(b, a),                              "⊕ commutativity",     a, b, c)
            _check(plus(plus(a, b), c),    plus(a, plus(b, c)),                     "⊕ associativity",     a, b, c)
            _check(plus(a, zero),          a,                                       "⊕ identity",          a, b, c)
            _check(times(times(a, b), c),  times(a, times(b, c)),                   "⊗ associativity",     a, b, c)
            _check(times(a, one),          a,                                       "⊗ identity",          a, b, c)
            _check(times(a, zero),         zero,                                    "0 annihilates ⊗",     a, b, c)
            _check(times(a, plus(b, c)),   plus(times(a, b), times(a, c)),          "⊗ distributes over ⊕", a, b, c)
