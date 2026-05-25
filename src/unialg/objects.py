"""Object-space constructors and effect descriptors.

The core DSL talks about Hydra ``Type`` values as objects.  This module owns
small constructors for common object shapes, plus the ``Monad`` descriptors
used by lax morphisms.  Actual effect sequencing is implemented by
``structure/realize.py`` and its term helpers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import cast

import hydra.dsl.types as T
import hydra.show.core as ShowCore
from hydra.core import (
    Name, Type, TypeEither, TypeFunction, TypePair, TypeList,
    TypeLiteral, TypeMaybe, TypeVoid, TypeScheme, TypeUnit, TypeVariable,
)
from hydra.lib import names as Names


def ProductType(left: Type, right: Type) -> TypePair:
    """Build the structural product type ``left × right``."""
    return cast(TypePair, T.pair(left, right))


def SumType(left: Type, right: Type) -> TypeEither:
    """Build the structural coproduct type ``left + right``."""
    return cast(TypeEither, T.either(left, right))


def ExpType(left: Type, right: Type) -> TypeFunction:
    """Build the exponentiated function type, used in functors"""
    return cast(TypeFunction, T.function(left, right))


BINARY: TypeLiteral = T.binary()


def ListType(inner: Type) -> TypeList:
    """Build the list type constructor ``List[inner]``."""
    return cast(TypeList, T.list_(inner))


def MaybeType(inner: Type) -> TypeMaybe:
    """Build the maybe type constructor ``Maybe[inner]``."""
    return cast(TypeMaybe, T.maybe(inner))


_SCALAR_TYPE_NAMES = {
    "STRING":  T.string,
    "INT":     T.int32,
    "FLOAT":   T.float64,
    "BOOL":    T.boolean,
    "BOOLEAN": T.boolean,
    "BINARY":  T.binary,
}


def type_from_name(name: str) -> Type:
    """Resolve a backend type name to a Hydra Type.

    Supports scalar names (STRING, INT, FLOAT, BOOL, BINARY) and
    compound forms List[NAME] and Maybe[NAME].
    """
    ctor = _SCALAR_TYPE_NAMES.get(name)
    if ctor is not None:
        return ctor()
    if name.startswith("List[") and name.endswith("]"):
        return ListType(type_from_name(name[5:-1]))
    if name.startswith("Maybe[") and name.endswith("]"):
        return MaybeType(type_from_name(name[6:-1]))
    raise ValueError(f"type_from_name: unrecognized type {name!r}")


def VoidType() -> TypeVoid:
    """Build the structural Hydra void type."""
    return TypeVoid() # T.Nothing?


def show_type(typ: Type) -> str:
    """Return Hydra's compact display form for a type."""
    try:
        return ShowCore.type(typ)
    except (AssertionError, AttributeError):
        return repr(typ)


@dataclass(frozen=True)
class Monad:
    """Hydra-backed monad descriptor for lax morphisms.

    ``type_ctor`` wraps visible codomains.  ``bind_name`` and ``pure_name`` are
    Hydra primitive names used by realization/action code to sequence effects.
    """
    type_ctor: type
    bind_name: Name
    pure_name: Name

    def wrap(self, typ: Type) -> Type:
        """Wrap a visible codomain in this monad's Hydra type constructor."""
        return cast(Type, self.type_ctor(typ))

    def unwrap(self, typ: Type) -> Type | None:
        """Return the inner type when ``typ`` is wrapped by this monad."""
        if isinstance(typ, self.type_ctor):
            return typ.value
        return None


MAYBE: Monad = Monad(TypeMaybe, Names.maybes_bind, Names.maybes_pure)
LIST: Monad = Monad(TypeList, Names.lists_bind, Names.lists_pure)
MONADS: dict[str, Monad] = {
    "Maybe": MAYBE,
    "List": LIST,
}


def monad_by_name(name: str) -> Monad:
    """Resolve a built-in monad descriptor by DSL name."""
    try:
        return MONADS[name]
    except KeyError as e:
        raise ValueError(f"unknown monad {name!r}") from e


def repeated_product(t: Type, n: int) -> Type:
    """Build the left-nested product type ``t × t × ... × t`` (n copies)."""
    if n == 1:
        return t
    out = t
    for _ in range(n - 1):
        out = ProductType(out, t)
    return out


from functools import lru_cache
import hydra.sources.libraries as _Libs
import hydra.lexical as _Lexical


@lru_cache(maxsize=1)
def standard_graph():
    """Hydra graph with all standard library primitives. Cached."""
    prims = []
    for name in dir(_Libs):
        if name.startswith('register_') and name.endswith('_primitives'):
            prims.extend(getattr(_Libs, name)().values())
    return _Lexical.graph_with_primitives(prims, ())
