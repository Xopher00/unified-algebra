"""User-defined semiring parameterisation for unified-algebra.

A semiring is defined by naming two binary operations (plus/⊕ and times/⊗)
and their identity elements (zero and one). The operation names reference
entries in a Backend — the user never passes callables directly.

The semiring is represented as a Hydra record term so it participates in
Hydra's type system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra.core as core

from unialg.terms import _RecordView

if TYPE_CHECKING:
    from unialg.backend import Backend


class Semiring(_RecordView):
    """A semiring: declaration, field access, law checking, and resolution.

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

    name     = _RecordView.Scalar(str)
    plus     = _RecordView.Scalar(str)
    times    = _RecordView.Scalar(str)
    zero     = _RecordView.Scalar(float)
    one      = _RecordView.Scalar(float)
    residual = _RecordView.Scalar(str, default="")
    bottom      = _RecordView.Scalar(float, default=-10.0)
    top         = _RecordView.Scalar(float, default=10.0)
    contraction = _RecordView.Scalar(str, default="")
    leq         = _RecordView.Scalar(str, default="")

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
        contraction_fn: object | None = None
        leq_name: str | None = None
        leq_elementwise: object | None = None

    def _law_check_samples(self, n: int = 5, seed: int = 42) -> list:
        """Draw n triplets uniformly from [bottom, top]."""
        import random
        rng = random.Random(seed)
        return [tuple(rng.uniform(self.bottom, self.top) for _ in range(3)) for _ in range(n)]

    def resolve(self, backend: "Backend", *, samples=None, check_laws: bool = True) -> "Semiring.Resolved":
        """Resolve this semiring against a backend.

        Law checking runs by default. Pass check_laws=False for semirings that
        intentionally violate exact semiring laws (e.g. smooth/approximate ops).
        Samples for law checking are drawn uniformly from [bottom, top]
        unless overridden via `samples`.
        """
        if check_laws:
            self.check_laws(backend, samples or self._law_check_samples())
        residual_name = self.residual
        contraction_fn = None
        if self.contraction:
            from unialg.algebra.contraction import CONTRACTION_REGISTRY
            contraction_fn = CONTRACTION_REGISTRY.get(self.contraction)
            if contraction_fn is None:
                raise ValueError(f"Unknown contraction strategy: {self.contraction!r}")
        leq_name = self.leq
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
            contraction_fn=contraction_fn,
            leq_name=leq_name or None,
            leq_elementwise=backend.elementwise(leq_name) if leq_name else None,
        )

    def check_laws(self, backend: "Backend", samples, atol: float = 1e-9) -> None:
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
            if math.isnan(lhs) or math.isnan(rhs):
                raise ValueError(
                    f"Semiring '{self.name}': {axiom} produces NaN at (a,b,c)=({a},{b},{c}): "
                    f"lhs={lhs}, rhs={rhs}"
                )
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

        if self.leq:
            # The named op is the meet of the induced order:
            #   a ≤ b  iff  meet(a, b) ≈ a  (within atol).
            # backend.elementwise(name) returns a value, not a bool — so we
            # derive the boolean relation from the meet.
            meet = backend.elementwise(self.leq)

            def _le(x, y) -> bool:
                m = meet(x, y)
                if math.isnan(m):
                    return False
                return abs(m - x) <= atol

            for a, b, c in samples:
                if not _le(a, a):
                    raise ValueError(
                        f"Semiring '{self.name}': leq reflexivity fails at a={a} "
                        f"(meet(a,a)={meet(a, a)})"
                    )
                if _le(a, b) and _le(b, c) and not _le(a, c):
                    raise ValueError(
                        f"Semiring '{self.name}': leq transitivity fails at "
                        f"(a,b,c)=({a},{b},{c})"
                    )
