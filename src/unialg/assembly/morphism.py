"""Smart constructors for typed algebraic morphisms.

This module is the long-term replacement for the ``ua.cell.Cell`` union and
its constructor surface in ``_para.py``. Each function here is a smart
constructor that produces a typed morphism wrapper whose ``.term`` is a Hydra
term (or, transitionally, a Cell) and performs argument validation.

**Migration status (Steps 4-5):**

Trivial leaf variants (``eq``, ``lit``, ``iden``, ``copy``, ``delete``)
wrap standard Hydra terms — ``TermVariable`` or constant lambdas — *not*
``ua.cell.Cell`` injections. The closure-emitting compiler in
``_morphism_compile`` recognizes the wrapped term shapes by name / shape and
emits the corresponding Python closure that ``prim1`` registration consumes.

``seq`` and ``par`` now return standard Hydra application/lambda shapes.
``algebra_hom`` still delegates to the ``_para`` Cell builder; its migration
starts at Step 6 of the Cell collapse plan. ``lens`` is represented as a
Hydra record term and lowers to ``CompiledLens`` at compile time.

The trivial-variant migration is genuine: the Sort argument is validated at
construction time (replacing ``_cell_types.py::_eq_cell_domain`` and
neighbours), and the wrapped term carries no Cell-injection wrapper.
Polymorphism over the carrier sort lives in the dispatcher (one closure per
named variant, regardless of Sort), not the term shape — Sort instantiation
is currently a Python-side validation; Hydra type-checker integration is
deferred.
"""
from __future__ import annotations

from hydra.core import Name
import hydra.dsl.terms as Terms

from ._typed_morphism import TypedMorphism, SortLike
from ._algebra_hom import algebra_hom
from .lens import lens

# Names referenced by the closure-emitting dispatcher.
# Equations are resolved by the ua.equation.* prefix; bimap is used for par.
# _IDEN_NAME      = "hydra.lib.equality.identity"
# _COPY_NAME      = "ua.morphism.copy"
# _DELETE_NAME    = "ua.morphism.delete"
_EQUATION_PREFIX = "ua.equation."
_BIMAP_NAME = "hydra.lib.pairs.bimap"

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
    # Terms.primitive(Name(_IDEN_NAME)) 
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
    bimap = Terms.primitive(Name(_BIMAP_NAME))
    return T(
        Terms.apply_all(bimap, [f.term, g.term]),
        T.product(f.domain, g.domain),
        T.product(f.codomain, g.codomain),
    )
