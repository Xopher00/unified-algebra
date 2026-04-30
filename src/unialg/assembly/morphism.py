"""Smart constructors for typed algebraic morphisms.

Each function constructs a ``TypedMorphism(term, domain, codomain)`` that
directly expresses a categorical morphism as a Hydra term.
"""
from __future__ import annotations

from ._typed_morphism import TypedMorphism, Name, Terms, SortLike
from ._algebra_hom import algebra_hom
from .lens import lens

_EQUATION_PREFIX = "ua.equation."
_BIMAP_NAME = Name("hydra.lib.pairs.bimap")

__all__ = [
    "eq", "lit", "iden", "copy", "delete",
    "seq", "par",
    "algebra_hom", "lens",
]

T = TypedMorphism

# ---------------------------------------------------------------------------
# Trivial leaf morphisms — wrap Hydra terms (Step 3)
# ---------------------------------------------------------------------------

def eq(name: str, *, domain: SortLike, codomain: SortLike) -> TypedMorphism:
    """Reference to a declared Equation by name."""
    if not isinstance(name, str) or not name:
        raise TypeError(f"morphism.eq.name: expected non-empty str, got {name!r}")
    return T(Terms.var(f"{_EQUATION_PREFIX}{name}"), domain, codomain)


def iden(sort: SortLike) -> TypedMorphism:
    """Identity morphism ``id_A : A → A``."""
    return T(Terms.identity(), sort, sort)


def copy(sort: SortLike) -> TypedMorphism:
    """Comonoid copy ``Δ_A : A → A × A`` as ``λx. (x, x)``."""
    return T(
        Terms.lambda_("x_", Terms.pair(Terms.var("x_"), Terms.var("x_"))),
        sort,
        T.product(sort, sort),
    )


def delete(sort: SortLike) -> TypedMorphism:
    """Comonoid delete ``!_A : A → 1`` as ``λ_. ()``."""
    return T(Terms.constant(Terms.unit()), sort, T.unit())


def lit(value_term, sort: SortLike) -> TypedMorphism:
    """0-ary constant morphism ``1 → A``.

    Wraps ``value_term`` in ``Terms.constant`` so the resulting term is an
    honest function ``λ_. value_term`` (i.e. ``1 → A``), not a bare value
    masquerading as a morphism.
    """
    return T(Terms.constant(value_term), T.unit(), sort)


# ---------------------------------------------------------------------------
# Composition variants
# ---------------------------------------------------------------------------

def seq(f, g) -> TypedMorphism:
    """Sequential composition ``f ; g``."""
    f = T.require(f, "seq.left")
    g = T.require(g, "seq.right")
    T.same_sort(f.codomain_type, g.domain_type, "seq.left.codomain")
    return T(Terms.compose(g.term, f.term), f.domain, g.codomain)


def par(f, g) -> TypedMorphism:
    """Monoidal product ``f ⊗ g``."""
    f = T.require(f, "par.left")
    g = T.require(g, "par.right")
    bimap = Terms.primitive(_BIMAP_NAME)
    return T(
        Terms.apply_all(bimap, [f.term, g.term]),
        T.product(f.domain, g.domain),
        T.product(f.codomain, g.codomain),
    )
