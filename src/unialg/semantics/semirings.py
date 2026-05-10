"""Semiring semantics for the unialg DSL.

A semiring is a carrier type with two binary operations (⊕ and ⊗) and their
identity elements.  All four components are ``Morphism`` objects so they
participate in the typed composition machinery.

Invariants (not checked at construction, checked at use sites):
    plus  : carrier × carrier → carrier
    times : carrier × carrier → carrier
    zero  : 1 → carrier          (additive identity for ⊕)
    one   : 1 → carrier          (multiplicative identity for ⊗)
"""

from __future__ import annotations

from dataclasses import dataclass

from unialg.objects import Type
from . import morphisms


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
