"""Smart constructor for ``algebra_hom`` typed morphisms.

Builds the induced (co)algebra morphism over a polynomial functor:

- **List** ``F = 1 + B × X``, algebra direction
  ``cata(init, cons) : List B → C  =  λxs. foldr(curry(cons), init, xs)``

- **Maybe** ``F = 1 + X``, algebra direction
  ``cata(nothing, just) : Maybe C → C  =  λm. cases(m, nothing, just)``

- **Coalgebra** direction: returns the step morphism unchanged.
"""
from __future__ import annotations

from hydra.core import TermLambda

import hydra.dsl.terms as Terms

from .functor import Functor, PolyExpr
from ._typed_morphism import TypedMorphism, Name

T = TypedMorphism

_FOLDR = Name("hydra.lib.lists.foldr")
_CASES = Name("hydra.lib.maybes.cases")


def algebra_hom(functor: Functor, direction: str, morphisms: list) -> TypedMorphism:
    """Induced (co)algebra hom over a polynomial Functor.

    ``algebra``:  one TypedMorphism per summand, all sharing a codomain (the
    carrier).  Returns ``cata_F(α) : μF → carrier``.

    ``coalgebra``:  a single step morphism.  Returned unchanged.
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

    # coalgebra: step IS the coalgebra
    if direction == "coalgebra":
        if len(morphisms) != 1:
            raise ValueError(
                f"coalgebra_hom over {functor.name!r}: expected 1 morphism, "
                f"got {len(morphisms)}"
            )
        step = morphisms[0]
        return T(step.term, step.domain_sort, step.codomain_sort, kind="ana")

    # algebra: one morphism per summand, shared codomain = carrier
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

    # List:  F = 1 + B × X  →  cata(init, cons) : List B → C
    parts = _list_parts(body)
    if parts is not None:
        one_idx, base_sort = parts
        init, cons = morphisms[one_idx], morphisms[1 - one_idx]
        T.same_sort(init.domain_type, T.unit(), "algebra_hom.init.domain")
        T.same_sort(cons.domain_type, T.product(base_sort, carrier),
                    "algebra_hom.cons.domain")
        return T(
            Terms.lambda_("xs", Terms.apply_all(
                Terms.primitive(_FOLDR),
                [_curry_pair(cons.term), _lit_value(init, "list"), Terms.var("xs")],
            )),
            T.list_type(base_sort),
            carrier,
            kind="cata",
        )

    # Maybe:  F = 1 + X  →  cata(nothing, just) : Maybe C → C
    one_idx = _maybe_parts(body)
    if one_idx is not None:
        nothing, just = morphisms[one_idx], morphisms[1 - one_idx]
        T.same_sort(nothing.domain_type, T.unit(), "algebra_hom.nothing.domain")
        T.same_sort(just.domain_type, carrier, "algebra_hom.just.domain")
        return T(
            Terms.lambda_("m", Terms.apply_all(
                Terms.primitive(_CASES),
                [Terms.var("m"), _lit_value(nothing, "maybe"), just.term],
            )),
            T.maybe_type(carrier),
            carrier,
            kind="cata",
        )

    raise NotImplementedError(
        f"algebra_hom over functor {functor.name!r}: polynomial shape "
        f"not yet supported (Phase 1 supports List = 1 + Const × Id and "
        f"Maybe = 1 + Id). Body kind={body.kind!r}."
    )


# ---------------------------------------------------------------------------
# Summand-domain computation (used by the resolver to set correct domains)
# ---------------------------------------------------------------------------

def summand_domain(summand: PolyExpr, carrier) -> object:
    """Expected domain of the case morphism for a functor summand.

    Given a summand shape and carrier sort, return the domain type:
    ``one`` → unit, ``id`` → carrier, ``const(S)`` → S,
    ``prod(A, B)`` → product of sub-domains.
    """
    k = summand.kind
    if k == "one":
        return T.unit()
    if k == "id":
        return carrier
    if k == "const":
        return summand.sort
    if k == "prod":
        return T.product(
            summand_domain(summand.left, carrier),
            summand_domain(summand.right, carrier),
        )
    raise NotImplementedError(f"summand_domain: unsupported kind {k!r}")


# ---------------------------------------------------------------------------
# Polynomial-shape recognisers
# ---------------------------------------------------------------------------

def _list_parts(body: PolyExpr) -> tuple[int, object] | None:
    """Recognise ``F = 1 + Const × Id``.  Returns ``(one_idx, base_sort)``."""
    if body.kind != "sum":
        return None
    branches = (body.left, body.right)
    kinds = (branches[0].kind, branches[1].kind)
    if "one" not in kinds or "prod" not in kinds:
        return None
    one_idx = 0 if kinds[0] == "one" else 1
    prod = branches[1 - one_idx]
    if prod.left.kind == "const" and prod.right.kind == "id":
        return one_idx, prod.left.sort
    if prod.left.kind == "id" and prod.right.kind == "const":
        raise NotImplementedError(
            "algebra_hom: Id × Const detected; only Const × Id is supported. "
            "Reorder the product in the functor declaration."
        )
    return None


def _maybe_parts(body: PolyExpr) -> int | None:
    """Recognise ``F = 1 + Id``.  Returns ``one_idx``."""
    if body.kind != "sum":
        return None
    kinds = (body.left.kind, body.right.kind)
    if set(kinds) != {"one", "id"}:
        return None
    return 0 if kinds[0] == "one" else 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _curry_pair(term) -> object:
    """Adapt a pair-domain morphism ``(A × B) → C`` to curried ``A → B → C``."""
    return Terms.lambdas(
        ["el", "acc"],
        Terms.apply(term, Terms.pair(Terms.var("el"), Terms.var("acc"))),
    )


def _lit_value(morphism: TypedMorphism, context: str) -> object:
    """Extract the value term from a ``lit`` morphism (shape ``λ_. value``)."""
    term = morphism.term
    if not (isinstance(term, TermLambda)
            and term.value.parameter.value == "_"):
        raise ValueError(
            f"algebra_hom ({context}): One-summand morphism must be a lit "
            f"(term shape λ_. value); got {type(term).__name__}"
        )
    return term.value.body
