"""Smart constructors for typed algebraic morphisms.

Each function constructs a ``TypedMorphism(term, domain, codomain)`` that
directly expresses a categorical morphism as a Hydra term.
"""
from __future__ import annotations

import hydra.dsl.terms as Terms

from ._typed_morphism import TypedMorphism, Name, SortLike
from .algebra_hom import algebra_hom
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
    return T(Terms.var(f"{_EQUATION_PREFIX}{name}"), domain, codomain, kind="eq")


def iden(sort: SortLike) -> TypedMorphism:
    """Identity morphism ``id_A : A → A``."""
    m = T(Terms.identity(), sort, sort, kind="iden")
    if m.domain_type != m.codomain_type:
        raise ValueError(
            f"iden: domain must equal codomain, got {m.domain_type} vs {m.codomain_type}"
        )
    return m


def copy(sort: SortLike) -> TypedMorphism:
    """Comonoid copy ``Δ_A : A → A × A`` as ``λx. (x, x)``."""
    m = T(
        Terms.lambda_("x_", Terms.pair(Terms.var("x_"), Terms.var("x_"))),
        sort,
        T.product(sort, sort),
        kind="copy",
    )
    fst, snd = T.split_product2(m.codomain_type, "copy.codomain")
    if fst != m.domain_type or snd != m.domain_type:
        raise ValueError("copy: codomain must equal domain × domain")
    return m


def delete(sort: SortLike) -> TypedMorphism:
    """Comonoid delete ``!_A : A → 1`` as ``λ_. ()``."""
    m = T(Terms.constant(Terms.unit()), sort, T.unit(), kind="delete")
    if m.codomain_type != T.unit():
        raise ValueError(
            f"delete: codomain must be unit, got {m.codomain_type}"
        )
    return m


def lit(value_term, sort: SortLike) -> TypedMorphism:
    """0-ary constant morphism ``1 → A``.

    Wraps ``value_term`` in ``Terms.constant`` so the resulting term is an
    honest function ``λ_. value_term`` (i.e. ``1 → A``), not a bare value
    masquerading as a morphism.
    """
    m = T(Terms.constant(value_term), T.unit(), sort, kind="lit")
    if m.domain_type != T.unit():
        raise ValueError(
            f"lit: domain must be unit, got {m.domain_type}"
        )
    return m


# ---------------------------------------------------------------------------
# Composition variants
# ---------------------------------------------------------------------------

def seq(f, g) -> TypedMorphism:
    """Sequential composition ``f ; g``."""
    f = T.require(f, "seq.left")
    g = T.require(g, "seq.right")
    T.same_sort(f.codomain_type, g.domain_type, "seq.left.codomain")
    return T(Terms.compose(g.term, f.term), f.domain, g.codomain, kind="seq")


def par(f, g) -> TypedMorphism:
    """Monoidal product ``f ⊗ g``."""
    f = T.require(f, "par.left")
    g = T.require(g, "par.right")
    bimap = Terms.primitive(_BIMAP_NAME)
    return T(
        Terms.apply_all(bimap, [f.term, g.term]),
        T.product(f.domain, g.domain),
        T.product(f.codomain, g.codomain),
        kind="par",
    )
