"""Sorts: named tensor types carrying a semiring.

A sort is a named space of tensors (e.g. "hidden", "output") bound to a
semiring. Two sorts are compatible iff they share the same semiring.

Sorts map to Hydra TypeVariables for type checking, and sort declarations
are registered in the Graph's schema_types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .views import EquationView, SortView, ProductSortView
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


def sort_to_type(name: str, semiring_name: str, batched: bool = False) -> core.Type:
    """Map a sort to a Hydra Type encoding both sort and semiring identity.

    The semiring is part of the type name so Hydra's type checker
    naturally distinguishes sorts over different semirings.  Batched sorts
    append a ':B' suffix so they are a distinct type from their unbatched
    counterparts.
    """
    suffix = ":B" if batched else ""
    return core.TypeVariable(core.Name(f"ua.sort.{name}:{semiring_name}{suffix}"))


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
        names = [sort_type_from_term(e).value.value for e in elements]
        return core.TypeVariable(core.Name(f"ua.sort.product:({'*'.join(names)})"))
    v = SortView(sort_term)
    return sort_to_type(v.name, v.semiring_name, v.batched)


def _check_sort(
    eq_terms_by_name: dict[str, core.Term],
    eq_name: str,
    field: str,
    expected_sort: core.Term,
    label: str,
) -> None:
    """Assert that an equation's sort field matches an expected sort."""
    v = EquationView(eq_terms_by_name[eq_name])
    sort_term = v.domain_sort if field == "domainSort" else v.codomain_sort
    actual = sort_type_from_term(sort_term)
    expected = sort_type_from_term(expected_sort)
    if actual != expected:
        raise TypeError(f"{label}: {actual.value.value!r} != {expected.value.value!r}")


def check_sort_junction(upstream_eq: core.Term, downstream_eq: core.Term) -> None:
    """Raise TypeError if upstream's codomain sort != downstream's domain sort.

    Compares full Hydra TypeVariable identity — both sort name and semiring
    must match.
    """
    up = EquationView(upstream_eq)
    down = EquationView(downstream_eq)
    codomain_type = sort_type_from_term(up.codomain_sort)
    domain_type = sort_type_from_term(down.domain_sort)
    if codomain_type != domain_type:
        raise TypeError(
            f"Sort junction error: '{up.name}' codomain "
            f"{codomain_type.value.value!r} != '{down.name}' domain "
            f"{domain_type.value.value!r}"
        )


def _output_rank(einsum_str: str) -> int | None:
    """Count output indices from einsum RHS. None if no einsum."""
    if not einsum_str:
        return None
    return len(einsum_str.split("->")[1].strip())


def _input_rank(einsum_str: str, slot: int) -> int | None:
    """Count input indices for a given slot in einsum LHS. None if no einsum."""
    if not einsum_str:
        return None
    parts = einsum_str.split("->")[0].split(",")
    if slot >= len(parts):
        return None
    return len(parts[slot])


def check_rank_junction(upstream_eq: core.Term, downstream_eq: core.Term,
                        input_slot: int) -> None:
    """Raise TypeError if upstream output rank != downstream input rank at slot.

    Only checks when both equations have non-empty einsum strings.
    """
    up = EquationView(upstream_eq)
    down = EquationView(downstream_eq)
    up_einsum = up.einsum
    down_einsum = down.einsum

    out_rank = _output_rank(up_einsum)
    in_rank = _input_rank(down_einsum, input_slot)

    if out_rank is not None and in_rank is not None and out_rank != in_rank:
        raise TypeError(
            f"Rank mismatch: '{up.name}' output rank {out_rank} != "
            f"'{down.name}' input rank {in_rank} at slot {input_slot}"
        )


def _semiring_from_type_name(type_name: str) -> str:
    """Extract the semiring portion from a sort type name.

    Type name formats:
      ua.sort.<sort>:<semiring>       (unbatched)
      ua.sort.<sort>:<semiring>:B     (batched)

    Returns the semiring string (e.g. "real").
    """
    # Strip the optional ':B' suffix first, then take the part after the first ':'
    name = type_name.removesuffix(":B")
    return name.split(":", 1)[1]


def check_sort_compatibility(sort_a: core.Term, sort_b: core.Term) -> bool:
    """Return True iff two sorts share the same semiring.

    Uses Hydra type identity: sorts over different semirings have
    different TypeVariable names and are therefore incompatible.
    The batched flag does not affect semiring compatibility.
    """
    type_a = sort_type_from_term(sort_a)
    type_b = sort_type_from_term(sort_b)
    sr_a = _semiring_from_type_name(type_a.value.value)
    sr_b = _semiring_from_type_name(type_b.value.value)
    return sr_a == sr_b


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

