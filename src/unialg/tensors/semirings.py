"""Semiring semantics for the unialg DSL.

A semiring is a carrier type with two binary operations (⊕ and ⊗) and their
identity elements.  All four components are ``Morphism`` objects so they
participate in the typed composition machinery.

Invariants (not checked at construction, checked at use sites):
    plus  : carrier × carrier → carrier
    times : carrier × carrier → carrier
    zero  : 1 → carrier          (additive identity for ⊕)
    one   : 1 → carrier          (multiplicative identity for ⊗)

Adjoint (right residual of ⊗, optional):
    adjoint : carrier × carrier → carrier

    The adjoint is the right residual of ⊗.  When the adjoint flag is set at
    call time, the contraction algorithm runs in adjoint mode.  The operation
    roles are swapped relative to standard mode:

        standard:  ⊕_j  A[i,j] ⊗ B[j,k]   — plus accumulates, times products
        adjoint:   ⊗_j  A[i,j] ⊘ B[j,k]   — times accumulates, adjoint products

    Examples:
        real semiring   — adjoint = division        ⊗=×, adjoint=÷
                          adjoint mode: ×_j (A[i,j] / B[j,k])
        fuzzy semiring  — adjoint = Gödel implication  ⊕=max, ⊗=min, adjoint=→
                          adjoint mode: min_j (A[i,j] → B[j,k])
        tropical        — adjoint = subtraction      ⊕=min, ⊗=+, adjoint=−
                          adjoint mode: +_j (A[i,j] − B[j,k])

    The user does not define a separate operation for adjoint mode.  They
    declare the adjoint on the semiring once, then set the flag at call time.
"""

from __future__ import annotations

from dataclasses import dataclass

from unialg.objects import Type
from unialg.semantics import morphisms


Morphism = morphisms.Morphism


@dataclass(frozen=True)
class Semiring:
    """A semiring: carrier type plus two binary operations and their identities."""
    name: str
    carrier: Type
    plus: Morphism
    times: Morphism
    zero: Morphism
    one: Morphism
    adjoint: Morphism | None = None  # right residual of ⊗; absent means no adjoint mode
    plus_reduce: Morphism | None = None    # axis-fold ⊕, seed = zero
    times_reduce: Morphism | None = None   # axis-fold ⊗, seed = one
    adjoint_reduce: Morphism | None = None # axis-fold ⊘ (if adjoint present)

    # -----------------------------------------------------------------------
    # No from_backend classmethod on Semiring
    # -----------------------------------------------------------------------
    #
    # Semiring takes Morphisms directly.  It does not import BackendOps
    # (structure layer) — that would be a layer violation.
    #
    # A user who already holds Morphisms constructs Semiring directly:
    #   sr = Semiring("real", FLOAT, add_m, mul_m, zero_m, one_m,
    #                 adjoint=div_m)
    #
    # The convenience factory that resolves op names against a BackendOps
    # instance lives in structure/semiring_factory.py (see that file).
    # It is allowed to import from semantics/ and back, and returns a Semiring.
    #
    # Lowering: each Morphism carries its BackendPrimitive in aux_primitives.
    # run() → realize.py traces those automatically.  A Semiring assembled
    # from Morphisms lowers through the standard path with no special handling.

    def op_env(self, *, adjoint: bool = False) -> dict[str, "Morphism"]:
        """Select the operation pair for a contraction.

        adjoint=False → {"product": times,   "fold": plus_reduce,   "seed": zero}
        adjoint=True  → {"product": adjoint, "fold": times_reduce,  "seed": one}

        Raises ValueError if the required reduce fields are None (i.e. semiring
        was not constructed via semiring_from_backend or reduce fields were not
        supplied explicitly).
        """
        if adjoint:
            if self.adjoint is None:
                raise ValueError(f"Semiring '{self.name}' has no adjoint operation")
            if self.times_reduce is None:
                raise ValueError(f"Semiring '{self.name}' is missing times_reduce (needed for adjoint contraction)")
            return {"product": self.adjoint, "fold": self.times_reduce, "seed": self.one}
        else:
            if self.plus_reduce is None:
                raise ValueError(f"Semiring '{self.name}' is missing plus_reduce (needed for contraction)")
            return {"product": self.times, "fold": self.plus_reduce, "seed": self.zero}


# ---------------------------------------------------------------------------
# Custom ops and factory — see structure/semiring_factory.py
# ---------------------------------------------------------------------------
#
# Registration of custom callables and construction of Semiring from backend
# op names both require BackendOps (structure layer).  Those concerns live in
# structure/semiring_factory.py, which is allowed to import from semantics/.
# This file stays free of any structure-layer imports.
