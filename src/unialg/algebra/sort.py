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
from hydra.dsl.python import Right, Left
from hydra.dsl.meta.phantoms import record, string, boolean, list_, unit, TTerm, binary

from unialg.views import _RecordView, _StringField
from unialg.utils import record_fields, string_value

if TYPE_CHECKING:
    import hydra.errors


# ---------------------------------------------------------------------------
# Type name constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sort class
# ---------------------------------------------------------------------------

class Sort(_RecordView):
    """A named tensor sort bound to a semiring.

    Construct:
        s = Sort("hidden", real_sr)
        s = Sort("batch_hidden", real_sr, batched=True)

    Wrap an existing term:
        s = Sort.from_term(term)

    Sort.from_term is polymorphic: returns a ProductSort for product sort records.
    """

    _type_name = core.Name("ua.sort.Sort")
    name = _StringField("name")

    def __init__(self, name: str, semiring_term, batched: bool = False):
        semiring_term = self._unwrap(semiring_term)
        super().__init__(record(self._type_name, [
            core.Name("name") >> string(name),
            core.Name("semiring") >> TTerm(semiring_term),
            core.Name("batched") >> boolean(batched),
        ]).value)

    @classmethod
    def from_term(cls, term) -> "Sort":
        """Wrap an existing Hydra record term as a Sort or ProductSort.

        Polymorphic: returns a ProductSort if the term has type_name
        ProductSort._type_name. Idempotent for Sort/ProductSort instances.
        """
        if isinstance(term, ProductSort):
            return term
        try:
            if (isinstance(term, core.TermRecord)
                    and term.value.type_name == ProductSort._type_name):
                return ProductSort.from_term(term)
        except (AttributeError, TypeError):
            pass
        if isinstance(term, Sort):
            return term
        obj = cls.__new__(cls)
        obj._term = term
        return obj

    @property
    def semiring_name(self) -> str:
        return string_value(record_fields(record_fields(self._term)["semiring"])["name"])

    @property
    def batched(self) -> bool:
        b = record_fields(self._term).get("batched")
        if b is None:
            return False
        return hasattr(b, "value") and hasattr(b.value, "value") and b.value.value is True

    @property
    def type_(self) -> core.Type:
        base = core.TypeApplication(core.ApplicationType(
            core.TypeVariable(core.Name(f"ua.sort.{self.name}")),
            core.TypeVariable(core.Name(f"ua.semiring.{self.semiring_name}"))))
        if self.batched:
            return core.TypeApplication(core.ApplicationType(
                core.TypeVariable(core.Name("ua.batched")), base))
        return base


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


# ---------------------------------------------------------------------------
# Semiring compatibility
# ---------------------------------------------------------------------------

def check_rank_junction(upstream_eq: core.Term, downstream_eq: core.Term,
                        input_slot: int) -> None:
    """Raise TypeError if upstream output rank != downstream input rank at slot."""
    from unialg.resolve.morphism import Equation
    up = Equation.from_term(upstream_eq)
    down = Equation.from_term(downstream_eq)
    up_einsum = up.einsum
    down_einsum = down.einsum

    out_rank = len(up_einsum.split("->")[1].strip()) if up_einsum else None
    parts = down_einsum.split("->")[0].split(",") if down_einsum else []
    in_rank = len(parts[input_slot]) if input_slot < len(parts) else None

    if out_rank is not None and in_rank is not None and out_rank != in_rank:
        raise TypeError(
            f"Rank mismatch: '{up.name}' output rank {out_rank} != "
            f"'{down.name}' input rank {in_rank} at slot {input_slot}"
        )


def check_sort_compatibility(sort_a, sort_b) -> bool:
    """Return True iff two sorts share the same semiring."""
    def _semiring(t):
        app = t.value
        if app.function == core.TypeVariable(core.Name("ua.batched")):
            app = app.argument.value
        return app.argument
    return _semiring(Sort.from_term(sort_a).type_) == _semiring(Sort.from_term(sort_b).type_)


# ---------------------------------------------------------------------------
# Tensor TermCoder
# ---------------------------------------------------------------------------

from unialg.algebra.coders import tensor_coder  # noqa: F401
from hydra.extract.core import binary as _extract_binary


def sort_coder(sort_term, backend) -> hydra.graph.TermCoder:
    """Create a TermCoder with the sort's TypeVariable."""
    s = Sort.from_term(sort_term)
    if isinstance(s, ProductSort):
        return product_sort_coder(sort_term, backend)

    fw = backend.from_wire
    tw = backend.to_wire

    def encode(cx, graph, term):
        result = _extract_binary(graph, term)
        match result:
            case Right(value=raw): pass
            case _: raw = term.value.value
        return Right(fw(raw))

    def decode(cx, arr):
        return Right(binary(tw(arr)).value)

    return hydra.graph.TermCoder(
        type=s.type_,
        encode=encode,
        decode=decode,
    )


def product_sort_coder(sort_term, backend):
    """Create a composite TermCoder for a product sort."""
    from hydra.dsl.prims import pair as pair_coder
    elements = ProductSort.from_term(sort_term).elements
    coders = [sort_coder(s, backend) for s in elements]
    result = coders[-1]
    for c in reversed(coders[:-1]):
        result = pair_coder(c, result)
    return result
