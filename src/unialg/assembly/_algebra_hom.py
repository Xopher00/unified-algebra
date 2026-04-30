"""Smart constructor for ``algebra_hom`` typed morphisms.

The constructor emits a Hydra term that wires the per-summand morphisms
together via registered Hydra stdlib primitive references. The result IS a
Hydra term — no intermediate ``ua.morphism.AlgHom`` record. Compilation falls
back to Hydra reduction for stdlib application terms, so Hydra remains the
owner of stdlib primitive execution.

Phase 1 supports two polynomial F shapes plus the trivial coalgebra case:

- **List functor** ``F = 1 + Const × Id`` + ``algebra`` direction
  → ``hydra.lib.lists.foldr``. Per-summand morphisms are
  ``[init: 1 → carrier, cons: (base × carrier) → carrier]``. ``init`` must
  be a ``lit`` morphism so its value can be unwrapped at construction.

- **Maybe functor** ``F = 1 + Id`` + ``algebra`` direction
  → ``hydra.lib.maybes.cases``. Per-summand morphisms are
  ``[nothing: 1 → carrier, just: carrier → carrier]``.

- **Coalgebra direction** with any body shape → return the step morphism
  unchanged. The step IS the coalgebra; nothing to wire.

Custom recursive F shapes (binary trees, etc.) raise ``NotImplementedError``
in Phase 1. Phase 2 adds a fallback that registers a Python walker primitive.
"""
from __future__ import annotations

import hydra.core as core
from hydra.core import Name
import hydra.dsl.terms as Terms

from unialg.assembly.functor import Functor, PolyExpr
from ._typed_morphism import TypedMorphism

T = TypedMorphism


def algebra_hom(functor: Functor, direction: str, morphisms: list) -> TypedMorphism:
    """Induced (co)algebra hom over a polynomial Functor.

    Parameters
    ----------
    functor:
        Polynomial endofunctor declaring the F-shape.
    direction:
        ``"algebra"`` for an inductive walker (cata-like); ``"coalgebra"`` for
        a coinductive driver (the step morphism itself).
    morphisms:
        For ``algebra`` direction: one ``TypedMorphism`` per summand of
        ``functor.summands()``, each ``F_summand_args → carrier``. The shared
        codomain is the carrier.
        For ``coalgebra`` direction: a single step morphism
        ``carrier → F(carrier)``; its domain is the carrier.

    Returns a ``TypedMorphism`` whose ``.term`` is a Hydra term referencing
    registered stdlib primitives by name. Compilation is handled by the
    dispatcher in ``_morphism_compile.py``.
    """
    if direction not in ("algebra", "coalgebra"):
        raise ValueError(
            f"algebra_hom.direction: expected 'algebra' or 'coalgebra', "
            f"got {direction!r}"
        )

    morphisms = [
        T.require(m, f"algebra_hom.morphisms[{i}]")
        for i, m in enumerate(morphisms)
    ]
    if not morphisms:
        raise ValueError("algebra_hom: at least one morphism is required")

    if direction == "coalgebra":
        if len(morphisms) != 1:
            raise ValueError(
                f"coalgebra_hom over {functor.name!r}: expected 1 morphism, "
                f"got {len(morphisms)}"
            )
        step = morphisms[0]
        return T(step.term, step.domain, step.codomain)

    # algebra direction
    n_summands = len(functor.summands())
    if len(morphisms) != n_summands:
        raise ValueError(
            f"algebra_hom: morphisms length {len(morphisms)} != functor "
            f"{functor.name!r} summand count {n_summands}"
        )

    carrier = morphisms[0].codomain
    for i, m in enumerate(morphisms[1:], 1):
        T.same_sort(m.codomain, carrier, f"algebra_hom.morphisms[{i}].codomain")

    body = functor.body
    if _matches_list_functor(body):
        term = _build_list_fold_term(body, morphisms)
    elif _matches_maybe_functor(body):
        term = _build_maybe_cata_term(body, morphisms)
    else:
        raise NotImplementedError(
            f"algebra_hom over functor {functor.name!r}: polynomial shape "
            f"not yet supported (Phase 1 supports List = 1 + Const × Id and "
            f"Maybe = 1 + Id). Body kind={body.kind!r}."
        )
    return T(term, carrier, carrier)


# ---------------------------------------------------------------------------
# Polynomial-shape recognisers
# ---------------------------------------------------------------------------

def _matches_list_functor(body: PolyExpr) -> bool:
    """``F = 1 + Const × Id`` (or its mirror)."""
    if body.kind != "sum":
        return False
    branches = (body.left, body.right)
    if {b.kind for b in branches} != {"one", "prod"}:
        return False
    prod = branches[0] if branches[0].kind == "prod" else branches[1]
    return {prod.left.kind, prod.right.kind} == {"const", "id"}


def _matches_maybe_functor(body: PolyExpr) -> bool:
    """``F = 1 + Id`` (or its mirror)."""
    if body.kind != "sum":
        return False
    return {body.left.kind, body.right.kind} == {"one", "id"}


# ---------------------------------------------------------------------------
# Term builders — wire morphisms via Hydra stdlib references
# ---------------------------------------------------------------------------

def _build_list_fold_term(body: PolyExpr, morphisms: list) -> object:
    """Emit ``λxs. foldr((λel. λacc. cons((el, acc))), init, xs)``."""
    one_idx = 0 if body.left.kind == "one" else 1
    init_morphism = morphisms[one_idx]
    cons_morphism = morphisms[1 - one_idx]
    init_value_term = _unwrap_lit(init_morphism, "list functor")
    return Terms.lambda_(
        "xs",
        Terms.apply_all(
            Terms.primitive(Name("hydra.lib.lists.foldr")),
            [_curried_pair_step(cons_morphism.term), init_value_term, Terms.var("xs")],
        ),
    )


def _build_maybe_cata_term(body: PolyExpr, morphisms: list) -> object:
    """Emit ``λm. apply_all(hydra.lib.maybes.cases, [m, nothing_value, just])``."""
    one_idx = 0 if body.left.kind == "one" else 1
    nothing_morphism = morphisms[one_idx]
    just_morphism = morphisms[1 - one_idx]
    nothing_value_term = _unwrap_lit(nothing_morphism, "Maybe functor")
    return Terms.lambda_(
        "m",
        Terms.apply_all(
            Terms.primitive(Name("hydra.lib.maybes.cases")),
            [Terms.var("m"), nothing_value_term, just_morphism.term],
        ),
    )


def _curried_pair_step(term) -> object:
    """Adapt a tuple-arg morphism ``(a, b) -> c`` to Hydra's ``a -> b -> c``."""
    return Terms.lambda_(
        "el",
        Terms.lambda_(
            "acc",
            Terms.apply(term, Terms.pair(Terms.var("el"), Terms.var("acc"))),
        ),
    )


def _unwrap_lit(morphism: TypedMorphism, context: str) -> object:
    """Extract ``value_term`` from a ``lit`` morphism whose term is ``λ_. value_term``.

    The One-summand morphism in a list / Maybe catamorphism must be a ``lit``
    so we can pass its underlying value (not a 0-ary function) to the stdlib
    primitive's initial-value argument.
    """
    term = morphism.term
    if not (
        isinstance(term, core.TermLambda)
        and term.value.parameter.value == "_"
    ):
        raise ValueError(
            f"algebra_hom over {context}: One-summand morphism must be "
            f"constructed via lit (term shape ``λ_. value``); got "
            f"{type(term).__name__}"
        )
    return term.value.body
