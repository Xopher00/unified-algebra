"""Sorts: named tensor types carrying a semiring.

A sort is a named space of tensors (e.g. "hidden", "output") bound to a
semiring. Two sorts are compatible iff they share the same semiring.

Sorts map to Hydra TypeVariables for type checking, and sort declarations
are registered in the Graph's schema_types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..views import EquationView, SortView, ProductSortView
import hydra.core as core
import hydra.graph
from hydra.dsl.python import Right, Left
from hydra.dsl.meta.phantoms import record, string, boolean, list_, unit, TTerm, binary

if TYPE_CHECKING:
    import hydra.errors


# ---------------------------------------------------------------------------
# Sort construction
# ---------------------------------------------------------------------------

SORT_TYPE_NAME = core.Name("ua.sort.Sort")


def sort(name: str, semiring_term: core.Term, batched: bool = False) -> core.Term:
    """Create a sort as a Hydra record term.

    Args:
        name:          sort identifier (e.g. "hidden", "output")
        semiring_term: the semiring this sort belongs to (from semiring())
        batched:       if True, equations on this sort automatically prepend a
                       batch dimension to their einsum at resolution time.

    Returns:
        A Hydra TermRecord representing the sort.
    """
    return record(SORT_TYPE_NAME, [
        core.Name("name") >> string(name),
        core.Name("semiring") >> TTerm(semiring_term),
        core.Name("batched") >> boolean(batched),
    ]).value




# ---------------------------------------------------------------------------
# Semiring compatibility
# ---------------------------------------------------------------------------


def is_batched(sort_term: core.Term) -> bool:
    """Return True if the sort has the batched flag set."""
    return SortView(sort_term).batched


def sort_type_from_term(sort_term: core.Term) -> core.Type:
    """Extract the Hydra Type for a sort from its record term."""
    if is_product_sort(sort_term):
        elements = product_sort_elements(sort_term)
        types = [sort_type_from_term(e) for e in elements]
        result = types[-1]
        for t in reversed(types[:-1]):
            result = core.TypePair(core.PairType(first=t, second=result))
        return result
    v = SortView(sort_term)
    base = core.TypeApplication(core.ApplicationType(
        core.TypeVariable(core.Name(f"ua.sort.{v.name}")),
        core.TypeVariable(core.Name(f"ua.semiring.{v.semiring_name}"))))
    if v.batched:
        return core.TypeApplication(core.ApplicationType(
            core.TypeVariable(core.Name("ua.batched")), base))
    return base


def check_rank_junction(upstream_eq: core.Term, downstream_eq: core.Term,
                        input_slot: int) -> None:
    """Raise TypeError if upstream output rank != downstream input rank at slot.

    Only checks when both equations have non-empty einsum strings.
    """
    up = EquationView(upstream_eq)
    down = EquationView(downstream_eq)
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


def check_sort_compatibility(sort_a: core.Term, sort_b: core.Term) -> bool:
    """Return True iff two sorts share the same semiring."""
    def _semiring(t):
        app = t.value
        if app.function == core.TypeVariable(core.Name("ua.batched")):
            app = app.argument.value
        return app.argument
    return _semiring(sort_type_from_term(sort_a)) == _semiring(sort_type_from_term(sort_b))


# ---------------------------------------------------------------------------
# Tensor TermCoder
# ---------------------------------------------------------------------------

from ._coders import tensor_coder  # noqa: F401
from hydra.extract.core import binary as _extract_binary


def sort_coder(sort_term: core.Term, backend) -> hydra.graph.TermCoder:
    """Create a TermCoder with the sort's TypeVariable.

    Backend provides from_wire/to_wire for array construction.
    For product sorts, dispatches to product_sort_coder().
    """
    if is_product_sort(sort_term):
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
        type=sort_type_from_term(sort_term),
        encode=encode,
        decode=decode,
    )


# ---------------------------------------------------------------------------
# Product sorts
# ---------------------------------------------------------------------------

PRODUCT_SORT_TYPE_NAME = core.Name("ua.sort.Product")


def product_sort(sorts: list[core.Term]) -> core.Term:
    """Create a product sort — a typed pair/tuple of sorts.

    Product sorts represent tensor tuples: morphisms that produce or consume
    multiple tensors simultaneously, typed by a right-nested pair of component
    sorts.

    Args:
        sorts: list of at least 2 component sort terms (from sort()).

    Returns:
        A Hydra TermRecord with type name ua.sort.Product and a 'sorts' field
        containing the list of component sort terms.
    """
    if len(sorts) < 2:
        raise ValueError("Product sort requires at least 2 component sorts")
    return record(PRODUCT_SORT_TYPE_NAME, [
        core.Name("sorts") >> list_([TTerm(s) for s in sorts]),
    ]).value


def is_product_sort(sort_term: core.Term) -> bool:
    """Check if a sort term is a product sort.

    Uses the record's type_name for a precise check: product sorts are
    TermRecord(Record(type_name=PRODUCT_SORT_TYPE_NAME, ...)).
    """
    try:
        return (
            isinstance(sort_term, core.TermRecord)
            and sort_term.value.type_name == PRODUCT_SORT_TYPE_NAME
        )
    except (AttributeError, TypeError):
        return False


def product_sort_elements(sort_term: core.Term) -> list[core.Term]:
    """Extract component sorts from a product sort.

    Args:
        sort_term: a product sort term (produced by product_sort()).

    Returns:
        List of component sort terms in declaration order.

    Raises:
        KeyError: if sort_term is not a product sort.
    """
    return ProductSortView(sort_term).elements


def product_sort_coder(sort_term: core.Term, backend):
    """Create a composite TermCoder for a product sort."""
    from hydra.dsl.prims import pair as pair_coder
    elements = product_sort_elements(sort_term)
    coders = [sort_coder(s, backend) for s in elements]
    # Right-nest: [a, b, c] -> pair(a, pair(b, c))
    result = coders[-1]
    for c in reversed(coders[:-1]):
        result = pair_coder(c, result)
    return result

