"""Sorts: named tensor types carrying a semiring.

A sort is a named space of tensors (e.g. "hidden", "output") bound to a
semiring. Two sorts are compatible iff they share the same semiring.

Sorts map to Hydra TypeVariables for type checking, and sort declarations
are registered in the Graph's schema_types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unified_algebra.utils import record_fields, string_value
import hydra.core as core
import hydra.dsl.terms as Terms
import hydra.graph
from hydra.dsl.python import Right, Left

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
    return Terms.record(SORT_TYPE_NAME, [
        Terms.field("name", Terms.string(name)),
        Terms.field("semiring", semiring_term),
        Terms.field("batched", Terms.boolean(batched)),
    ])


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

def _is_batched_field(fields: dict) -> bool:
    """Read the 'batched' flag from a pre-extracted fields dict.

    Returns False when the field is absent (backwards-compat with old records).
    """
    if "batched" not in fields:
        return False
    b_term = fields["batched"]
    # TermLiteral(LiteralBoolean(bool))
    return (
        hasattr(b_term, "value")
        and hasattr(b_term.value, "value")
        and b_term.value.value is True
    )


def is_batched(sort_term: core.Term) -> bool:
    """Return True if the sort has the batched flag set."""
    return _is_batched_field(record_fields(sort_term))


def sort_type_from_term(sort_term: core.Term) -> core.Type:
    """Extract the Hydra Type for a sort from its record term."""
    fields = record_fields(sort_term)
    name = string_value(fields["name"])
    sr_name = string_value(record_fields(fields["semiring"])["name"])
    batched = _is_batched_field(fields)
    return sort_to_type(name, sr_name, batched)


def check_sort_junction(upstream_eq: core.Term, downstream_eq: core.Term) -> None:
    """Raise TypeError if upstream's codomain sort != downstream's domain sort.

    Compares full Hydra TypeVariable identity — both sort name and semiring
    must match.
    """
    up_fields = record_fields(upstream_eq)
    down_fields = record_fields(downstream_eq)
    codomain_type = sort_type_from_term(up_fields["codomainSort"])
    domain_type = sort_type_from_term(down_fields["domainSort"])
    if codomain_type != domain_type:
        up_name = string_value(up_fields["name"])
        down_name = string_value(down_fields["name"])
        raise TypeError(
            f"Sort junction error: '{up_name}' codomain "
            f"{codomain_type.value.value!r} != '{down_name}' domain "
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
    up_fields = record_fields(upstream_eq)
    down_fields = record_fields(downstream_eq)
    up_einsum = string_value(up_fields["einsum"])
    down_einsum = string_value(down_fields["einsum"])

    out_rank = _output_rank(up_einsum)
    in_rank = _input_rank(down_einsum, input_slot)

    if out_rank is not None and in_rank is not None and out_rank != in_rank:
        up_name = string_value(up_fields["name"])
        down_name = string_value(down_fields["name"])
        raise TypeError(
            f"Rank mismatch: '{up_name}' output rank {out_rank} != "
            f"'{down_name}' input rank {in_rank} at slot {input_slot}"
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

def tensor_coder() -> hydra.graph.TermCoder:
    """Create a TermCoder that bridges numpy arrays and Hydra Terms.

    Wire format: <dtype>\\x00<dim0>,<dim1>,...\\x00<raw bytes>
    Uses a generic NDArray type — for sort-specific types, use sort_coder().
    """
    return hydra.graph.TermCoder(
        type=core.TypeVariable(core.Name("ua.tensor.NDArray")),
        encode=_tensor_encode,
        decode=_tensor_decode,
    )


def sort_coder(sort_term: core.Term) -> hydra.graph.TermCoder:
    """Create a TermCoder with a sort-specific Hydra type.

    Same encode/decode as tensor_coder(), but the type is the sort's
    TypeVariable (e.g. ua.sort.hidden:real) rather than the generic NDArray.
    This lets Hydra's type checker distinguish primitives operating on
    different sorts.
    """
    return hydra.graph.TermCoder(
        type=sort_type_from_term(sort_term),
        encode=_tensor_encode,
        decode=_tensor_decode,
    )


def _tensor_encode(cx, graph, term):
    """Term -> ndarray."""
    import numpy as np
    raw = term.value.value  # TermLiteral(LiteralBinary(bytes))
    i = raw.index(0)
    j = raw.index(0, i + 1)
    dtype = raw[:i].decode()
    shape_str = raw[i + 1:j].decode()
    shape = tuple(int(x) for x in shape_str.split(",") if x)
    data = raw[j + 1:]
    return Right(np.frombuffer(data, dtype=dtype).reshape(shape))


def _tensor_decode(cx, arr):
    """ndarray -> Term."""
    import numpy as np
    a = np.ascontiguousarray(arr)
    hdr = (
        a.dtype.str.encode()
        + b"\x00"
        + ",".join(str(s) for s in a.shape).encode()
        + b"\x00"
    )
    return Right(Terms.binary(hdr + a.tobytes()))

