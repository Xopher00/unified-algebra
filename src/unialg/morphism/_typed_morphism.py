from __future__ import annotations

import hydra.core as core
import hydra.dsl.types as Types

from unialg.algebra.sort import Sort, ProductSort
from unialg.terms import _RecordView

Name = core.Name

SortLike = Sort | ProductSort | core.Type
SORT_TYPES = (Sort, ProductSort)
HYDRA_TYPE_TYPES = tuple(
    obj for name in dir(core)
    if name.startswith("Type") and isinstance(obj := getattr(core, name), type)
    and name not in ("TypeVariableMetadata", "TypeApplicationTerm", "TypeScheme", "TypeAlias")
)

def _expected_name(expected) -> str:
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    if isinstance(expected, type):
        return expected.__name__
    return str(expected)


def _require(value, expected, label: str, *, predicate=None):
    ok = isinstance(value, expected) if predicate is None else predicate(value)
    if not ok:
        raise TypeError(
            f"morphism.{label}: expected {_expected_name(expected)}, got {value!r}"
        )
    return value


def _boundary_type(value: SortLike, label: str = "boundary") -> core.Type:
    if isinstance(value, SORT_TYPES):
        return value.type_
    return _require(value, HYDRA_TYPE_TYPES, label)


class TypedMorphism(core.TypeFunction):
    """A Hydra function type paired with the term that inhabits it.

    ``value`` is the inherited Hydra ``FunctionType``:
        ``domain.type_ -> codomain.type_``.

    Boundary values may be UA Sort/ProductSort records or native Hydra Type
    variants. ``value`` always stores the normalized Hydra function type.
    """

    term: object
    domain_sort: SortLike
    codomain_sort: SortLike

    def __init__(self, term, domain: SortLike, codomain: SortLike):
        super().__init__(
            core.FunctionType(
                _boundary_type(domain, "TypedMorphism.domain"),
                _boundary_type(codomain, "TypedMorphism.codomain"),
            )
        )
        object.__setattr__(self, "term", self.unwrap(term))
        object.__setattr__(self, "domain_sort", domain)
        object.__setattr__(self, "codomain_sort", codomain)

    @property
    def domain(self) -> SortLike:
        return self.domain_sort

    @property
    def codomain(self) -> SortLike:
        return self.codomain_sort

    @property
    def domain_type(self) -> core.Type:
        return self.value.domain

    @property
    def codomain_type(self) -> core.Type:
        return self.value.codomain

    @property
    def type_(self) -> core.Type:
        return self
    
    def require_boundary(
        self, domain: SortLike, codomain: SortLike,
        label: str,) -> "TypedMorphism":
        self.same_sort(self.domain_type, domain, f"{label}.domain")
        self.same_sort(self.codomain_type, codomain, f"{label}.codomain")
        return self
    
    @staticmethod
    def unwrap(value):
        return _RecordView._unwrap(value)

    @staticmethod
    def require(value, label: str) -> "TypedMorphism":
        return _require(value, TypedMorphism, label)

    @staticmethod
    def same_sort(actual: SortLike, expected: SortLike, label: str) -> SortLike:
        actual_type = _boundary_type(actual, label)
        expected_type = _boundary_type(expected, f"{label}.expected")
        _require(
            actual_type, expected_type,
            label, predicate=lambda value: value == expected_type,
        )
        return actual
    
    @classmethod
    def expect(
        cls, value,
        domain: SortLike, codomain: SortLike,
        label: str,) -> "TypedMorphism":
        return cls.require(value, label).require_boundary(domain, codomain, label)

    @staticmethod
    def boundary_type(value: SortLike) -> core.Type:
        return _boundary_type(value)

    @classmethod
    def unit(cls) -> core.Type:
        return Types.unit()

    @classmethod
    def product(cls, *sorts: SortLike) -> core.Type:
        types = [_boundary_type(s) for s in sorts]
        return cls.unit() if not types else Types.product(types)

    @staticmethod
    def list_type(element: SortLike) -> core.Type:
        return Types.list_(_boundary_type(element))

    @staticmethod
    def maybe_type(element: SortLike) -> core.Type:
        return Types.maybe(_boundary_type(element))

    @staticmethod
    def split_product2(value: SortLike, label: str) -> tuple[core.Type, core.Type]:
        typ = _require(
            _boundary_type(value, label),
            core.TypePair, label,
        )
        pair = typ.value
        return pair.first, pair.second
