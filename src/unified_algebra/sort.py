"""Sorts: named tensor types carrying a semiring.

A sort is a named space of tensors (e.g. "hidden", "output") bound to a
semiring. Two sorts are compatible iff they share the same semiring.

Sorts map to Hydra TypeVariables for type checking, and sort declarations
are registered in the Graph's schema_types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import unified_algebra._hydra_setup  # noqa: F401
import hydra.core as core
import hydra.dsl.terms as Terms
import hydra.graph
from hydra.dsl.python import FrozenDict, Right, Left, Nothing

from .semiring import _record_fields, _string_value

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


def sort_to_type(name: str) -> core.Type:
    """Map a sort name to a Hydra Type (nominal TypeVariable)."""
    return core.TypeVariable(core.Name(f"ua.sort.{name}"))


# ---------------------------------------------------------------------------
# Semiring compatibility
# ---------------------------------------------------------------------------

def check_sort_compatibility(sort_a: core.Term, sort_b: core.Term) -> bool:
    """Return True iff two sorts share the same semiring."""
    sr_a = _record_fields(_record_fields(sort_a)["semiring"])
    sr_b = _record_fields(_record_fields(sort_b)["semiring"])
    return _string_value(sr_a["name"]) == _string_value(sr_b["name"])


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

    # Register each sort
    for st in sort_terms:
        fields = _record_fields(st)
        name = _string_value(fields["name"])
        sort_name = core.Name(f"ua.sort.{name}")
        schema[sort_name] = core.TypeScheme(
            (), core.TypeVariable(sort_name), Nothing()
        )
        terms[sort_name] = st

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
