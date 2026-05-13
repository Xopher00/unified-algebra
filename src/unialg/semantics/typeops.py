# path: src/unialg/semantics/typeops.py

from __future__ import annotations

from dataclasses import dataclass

from hydra.context import Context
from hydra.dsl.python import Left, Right
import hydra.checking as Checking
import hydra.inference as Inference
import hydra.lib.maps as HMaps
import hydra.substitution as Substitution
import hydra.unification as Unification

from unialg.objects import (
    Type,
    TypeEither,
    TypePair,
    TypeUnit,
    ProductType,
    SumType,
    show_type,
)


@dataclass(frozen=True)
class TypeMatch:
    pattern: Type
    actual: Type
    substitution: object


def normalized(typ: Type) -> Type:
    """Normalize free type variables for stable comparison."""
    return Checking.normalize_type_free_vars(typ)


def effectively_equal(graph, left: Type, right: Type) -> bool:
    """Hydra-backed semantic type equality."""
    return Checking.types_effectively_equal(graph, left, right)


def require_equal(graph, left: Type, right: Type, context: str) -> None:
    """Raise if two types are not effectively equal."""
    if not effectively_equal(graph, left, right):
        raise TypeError(
            f"{context}: {show_type(normalized(left))} != {show_type(normalized(right))}"
        )


def require_all_equal(graph, types: tuple[Type, ...], context: str) -> Type:
    """Require a list of types to be effectively equal."""
    if not Checking.types_all_effectively_equal(graph, types):
        shown = ", ".join(show_type(normalized(t)) for t in types)
        raise TypeError(f"{context}: types do not agree: {shown}")
    return normalized(types[0])


def unify(pattern: Type, actual: Type, context: str) -> TypeMatch:
    """Solve pattern = actual using Hydra unification."""
    result = Unification.unify_types(None, HMaps.empty(), pattern, actual, context)
    if not isinstance(result, Right):
        raise TypeError(
            f"{context}: cannot unify {show_type(pattern)} with {show_type(actual)}: {result!r}"
        )

    subst = result.value.value
    checked = Checking.check_type_subst(None, None, subst)
    if isinstance(checked, Left):
        raise TypeError(f"{context}: invalid substitution: {checked.value!r}")

    return TypeMatch(pattern=pattern, actual=actual, substitution=subst)


def unify_lists(patterns: tuple[Type, ...], actuals: tuple[Type, ...], context: str):
    """Unify two type lists pairwise."""
    result = Unification.unify_type_lists(None, HMaps.empty(), patterns, actuals, context)
    if not isinstance(result, Right):
        raise TypeError(f"{context}: cannot unify type lists: {result!r}")

    subst = result.value.value
    checked = Checking.check_type_subst(None, None, subst)
    if isinstance(checked, Left):
        raise TypeError(f"{context}: invalid substitution: {checked.value!r}")

    return subst


def apply_subst(subst, typ: Type) -> Type:
    """Apply a Hydra type substitution to a type."""
    return Substitution.subst_in_type(subst, typ)


def compose_subst(left, right):
    """Compose two Hydra type substitutions."""
    return Substitution.compose_type_subst(left, right)


def unify_or_equal(
    graph,
    pattern: Type,
    actual: Type,
    context: str,
    *,
    allow_unification: bool,
) -> Type:
    """
    Controlled boundary policy.

    Returns a representative type:
    - normalized type when equality succeeds
    - substituted pattern when unification succeeds
    """
    if allow_unification:
        match = unify(pattern, actual, context)
        return normalized(apply_subst(match.substitution, pattern))

    require_equal(graph, pattern, actual, context)
    return normalized(pattern)


def require_product(typ: Type, context: str) -> tuple[Type, Type]:
    """Require a product type and return its components."""
    if not isinstance(typ, TypePair):
        raise TypeError(f"{context}: expected product type, got {show_type(typ)}")
    return typ.value.first, typ.value.second


def require_sum(typ: Type, context: str) -> tuple[Type, Type]:
    """Require a sum type and return its components."""
    if not isinstance(typ, TypeEither):
        raise TypeError(f"{context}: expected sum type, got {show_type(typ)}")
    return typ.value.left, typ.value.right


def visible_domain(raw_domain: Type, param: Type, context: str, *, graph=None) -> Type:
    """
    Strip a contextual parameter prefix from a raw domain.
    raw_domain must be param × visible when param is non-unit.
    """
    if param == TypeUnit():
        return raw_domain

    left, right = require_product(raw_domain, context)
    require_equal(graph, left, param, f"{context}: parameter prefix")
    return right


def product(left: Type, right: Type) -> Type:
    return ProductType(left, right)


def sum_(left: Type, right: Type) -> Type:
    return SumType(left, right)


def combine_params(left: Type, right: Type) -> Type:
    """
    Independent contextual parameters.
    Matches current greenfield behavior: g_param × f_param.
    """
    if left == TypeUnit():
        return right
    if right == TypeUnit():
        return left
    return ProductType(right, left)


def share_param(
    graph,
    left: Type,
    right: Type,
    context: str,
    *,
    allow_unification: bool = False,
) -> Type:
    """Resolve a shared contextual parameter."""
    if left == TypeUnit():
        return right
    if right == TypeUnit():
        return left
    return unify_or_equal(
        graph,
        left,
        right,
        context,
        allow_unification=allow_unification,
    )


def fresh_variable_type(cx: Context) -> tuple[Type, Context]:
    """Generate a truly fresh Hydra type variable."""
    return Inference.fresh_variable_type(cx)


def roundtrip_equal(graph, builder, recovered: Type, original: Type, context: str) -> None:
    """Validate a recovered type by rebuilding through a caller-supplied function."""
    rebuilt = builder(recovered)
    require_equal(graph, rebuilt, original, context)