"""Sorts: named tensor types carrying a semiring.

A sort is a named space of tensors (e.g. "hidden", "output") bound to a
semiring. Two sorts are compatible iff they share the same semiring.

Sorts map to Hydra TypeVariables for type checking, and sort declarations
are registered in the Graph's schema_types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra.core as core
import hydra.graph
from hydra.dsl.meta.phantoms import record, string, boolean, list_, TTerm
from hydra.dsl.python import Nothing

from unialg.terms import _RecordView, _TermField, _ScalarField, record_fields, literal_value
from unialg.algebra.semiring import Semiring
from unialg.terms import tensor_coder

if TYPE_CHECKING:
    import hydra.errors


class Sort(_RecordView):
    """A named tensor sort bound to a semiring.

    Construct:
        s = Sort("hidden", real_sr)
        s = Sort("batch_hidden", real_sr, batched=True)

    Wrap an existing term:
        s = Sort.from_term(term)
    """

    _type_name = core.Name("ua.sort.Sort")
    name     = _ScalarField("name", str)
    semiring = _TermField("semiring")

    def __init__(self, name: str, semiring_term, batched: bool = False):
        semiring_term = self._unwrap(semiring_term)
        super().__init__(record(self._type_name, [
            core.Name("name") >> string(name),
            core.Name("semiring") >> TTerm(semiring_term),
            core.Name("batched") >> boolean(batched),
        ]).value)

    @property
    def semiring_name(self) -> str:
        return Semiring.from_term(self.semiring).name

    @property
    def batched(self) -> bool:
        b = record_fields(self._term).get("batched")
        return False if b is None else bool(literal_value(b))

    @property
    def type_(self) -> core.Type:
        base = core.TypeApplication(core.ApplicationType(
            core.TypeVariable(core.Name(f"ua.sort.{self.name}")),
            core.TypeVariable(core.Name(f"ua.semiring.{self.semiring_name}"))))
        if self.batched:
            return core.TypeApplication(core.ApplicationType(
                core.TypeVariable(core.Name("ua.batched")), base))
        return base

    def coder(self, backend) -> hydra.graph.TermCoder:
        return tensor_coder(backend, type_=self.type_)

    def register_schema(self, schema: dict) -> None:
        for name_str in (f"ua.sort.{self.name}", f"ua.semiring.{self.semiring_name}", "ua.batched"):
            n = core.Name(name_str)
            schema[n] = core.TypeScheme((), core.TypeVariable(n), Nothing())


# ---------------------------------------------------------------------------
# ProductSort class
# ---------------------------------------------------------------------------

class ProductSort(_RecordView):
    """A product sort — a typed pair/tuple of sorts.

    Construct:
        ps = ProductSort([hidden_sort, output_sort])

    Wrap an existing term:
        ps = ProductSort.from_term(term)
    """

    _type_name = core.Name("ua.sort.Product")

    def __init__(self, sorts: list):
        if len(sorts) < 2:
            raise ValueError("Product sort requires at least 2 component sorts")
        raw_sorts = [self._unwrap(s) for s in sorts]
        super().__init__(record(self._type_name, [
            core.Name("sorts") >> list_([TTerm(s) for s in raw_sorts]),
        ]).value)

    @property
    def elements(self) -> list[core.Term]:
        return list(record_fields(self._term)["sorts"].value)

    @property
    def type_(self) -> core.Type:
        types = [Sort.from_term(e).type_ for e in self.elements]
        result = types[-1]
        for t in reversed(types[:-1]):
            result = core.TypePair(core.PairType(first=t, second=result))
        return result

    def coder(self, backend):
        from hydra.dsl.prims import pair as pair_coder
        coders = [Sort.from_term(e).coder(backend) for e in self.elements]
        result = coders[-1]
        for c in reversed(coders[:-1]):
            result = pair_coder(c, result)
        return result

    def register_schema(self, schema: dict) -> None:
        for elem in self.elements:
            sort_wrap(elem).register_schema(schema)


def sort_wrap(term) -> "Sort | ProductSort":
    """Return Sort or ProductSort wrapper for a raw sort term."""
    if isinstance(term, _RecordView):
        term = term._term
    if term.value.type_name == ProductSort._type_name:
        return ProductSort.from_term(term)
    return Sort.from_term(term)


# ---------------------------------------------------------------------------
# Semiring compatibility
# ---------------------------------------------------------------------------


def check_sort_compatibility(sort_a, sort_b) -> bool:
    """Return True iff two sorts share the same semiring."""
    def _semiring(t):
        app = t.value
        if app.function == core.TypeVariable(core.Name("ua.batched")):
            app = app.argument.value
        return app.argument
    return _semiring(Sort.from_term(sort_a).type_) == _semiring(Sort.from_term(sort_b).type_)


