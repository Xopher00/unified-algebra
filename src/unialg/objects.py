"""Object-space constructors and effect descriptors.

The core DSL talks about Hydra ``Type`` values as objects.  This module owns
small constructors for common object shapes, plus the ``Monad`` descriptors
used by lax morphisms.  Actual effect sequencing is implemented by
``realize.py`` and ``actions.py``.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import cast

import hydra.dsl.types as T
import hydra.show.core as ShowCore
from hydra.core import (
    Name, Type, TypeEither, TypeFunction, TypePair, TypeList, 
    TypeMaybe, TypeVoid, TypeScheme, TypeUnit, TypeVariable,
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


def ListType(inner: Type) -> TypeList:
    return cast(TypeList, T.list_(inner))


def MaybeType(inner: Type) -> TypeMaybe:
    return cast(TypeMaybe, T.maybe(inner))


def VoidType() -> TypeVoid:
    """Build the structural Hydra void type."""
    return TypeVoid() # T.Nothing?


def show_type(typ: Type) -> str:
    """Return Hydra's compact display form for a type."""
    return ShowCore.type(typ)


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
