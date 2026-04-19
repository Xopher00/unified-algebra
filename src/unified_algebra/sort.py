"""Sorts: named tensor types carrying a semiring.

A sort is a named space of tensors (e.g. "hidden", "output") bound to a
semiring. Two sorts are compatible iff they share the same semiring.

Sorts map to Hydra TypeVariables for type checking, and sort declarations
are registered in the Graph's schema_types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unified_algebra._hydra_setup import record_fields, string_value
import hydra.core as core
import hydra.dsl.terms as Terms
import hydra.graph
from hydra.dsl.python import FrozenDict, Right, Left, Nothing

if TYPE_CHECKING:
    import hydra.errors


# ---------------------------------------------------------------------------
# Sort construction
# ---------------------------------------------------------------------------

SORT_TYPE_NAME = core.Name("ua.sort.Sort")


def sort(name: str, semiring_term: core.Term) -> core.Term:
    """Create a sort as a Hydra record term.

    Args:
        name: sort identifier (e.g. "hidden", "output")
        semiring_term: the semiring this sort belongs to (from semiring())

    Returns:
        A Hydra TermRecord representing the sort.
    """
    return Terms.record(SORT_TYPE_NAME, [
        Terms.field("name", Terms.string(name)),
        Terms.field("semiring", semiring_term),
    ])


def sort_to_type(name: str, semiring_name: str) -> core.Type:
    """Map a sort to a Hydra Type encoding both sort and semiring identity.

    The semiring is part of the type name so Hydra's type checker
    naturally distinguishes sorts over different semirings.
    """
    return core.TypeVariable(core.Name(f"ua.sort.{name}:{semiring_name}"))


# ---------------------------------------------------------------------------
# Semiring compatibility
# ---------------------------------------------------------------------------

def sort_type_from_term(sort_term: core.Term) -> core.Type:
    """Extract the Hydra Type for a sort from its record term."""
    fields = record_fields(sort_term)
    name = string_value(fields["name"])
    sr_name = string_value(record_fields(fields["semiring"])["name"])
    return sort_to_type(name, sr_name)


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


def check_sort_compatibility(sort_a: core.Term, sort_b: core.Term) -> bool:
    """Return True iff two sorts share the same semiring.

    Uses Hydra type identity: sorts over different semirings have
    different TypeVariable names and are therefore incompatible.
    """
    type_a = sort_type_from_term(sort_a)
    type_b = sort_type_from_term(sort_b)
    # Same semiring iff the semiring portion of the type name matches
    sr_a = type_a.value.value.rsplit(":", 1)[1]
    sr_b = type_b.value.value.rsplit(":", 1)[1]
    return sr_a == sr_b


# ---------------------------------------------------------------------------
# Tensor TermCoder
# ---------------------------------------------------------------------------

def tensor_coder() -> hydra.graph.TermCoder:
    """Create a TermCoder that bridges numpy arrays and Hydra Terms.

    Wire format: <dtype>\\x00<dim0>,<dim1>,...\\x00<raw bytes>
    """
    return hydra.graph.TermCoder(
        type=core.TypeVariable(core.Name("ua.tensor.NDArray")),
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


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(
    sort_terms: list[core.Term],
    primitives: dict | None = None,
    bound_terms: dict | None = None,
) -> hydra.graph.Graph:
    """Assemble a Hydra Graph with sorts registered as schema_types.

    Args:
        sort_terms: list of sort record terms (from sort())
        primitives: optional dict of Name -> Primitive (for Phase 3+)
        bound_terms: optional dict of Name -> Term
    """
    schema = {}
    terms = dict(bound_terms or {})

    # Register the tensor type
    tensor_name = core.Name("ua.tensor.NDArray")
    schema[tensor_name] = core.TypeScheme(
        (), core.TypeVariable(tensor_name), Nothing()
    )

    # Register each sort with semiring identity in the type
    for st in sort_terms:
        fields = record_fields(st)
        name = string_value(fields["name"])
        sr_fields = record_fields(fields["semiring"])
        sr_name = string_value(sr_fields["name"])
        sort_type_name = core.Name(f"ua.sort.{name}:{sr_name}")
        schema[sort_type_name] = core.TypeScheme(
            (), core.TypeVariable(sort_type_name), Nothing()
        )
        terms[sort_type_name] = st

    return hydra.graph.Graph(
        bound_terms=FrozenDict(terms),
        bound_types=FrozenDict({}),
        class_constraints=FrozenDict({}),
        lambda_variables=frozenset(),
        metadata=FrozenDict({}),
        primitives=FrozenDict(primitives or {}),
        schema_types=FrozenDict(schema),
        type_variables=frozenset(),
    )
