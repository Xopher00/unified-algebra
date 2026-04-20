"""Private: default tensor TermCoder for user-level encoding.

Tests and user code use tensor_coder() to encode arrays as Hydra terms
before passing them to reduce_term. Reuses numpy wire format from backend.
"""

from __future__ import annotations

import hydra.core as core
import hydra.dsl.terms as Terms
import hydra.graph
from hydra.dsl.python import Right
from hydra.extract.core import binary as extract_binary
from .backend import _np_from_wire, _np_to_wire


def tensor_coder() -> hydra.graph.TermCoder:
    """Create a TermCoder that bridges arrays and Hydra Terms."""

    def encode(cx, graph, term):
        result = extract_binary(graph, term)
        match result:
            case Right(value=raw): pass
            case _: raw = term.value.value
        return Right(_np_from_wire(raw))

    def decode(cx, arr):
        return Right(Terms.binary(_np_to_wire(arr)))

    return hydra.graph.TermCoder(
        type=core.TypeVariable(core.Name("ua.tensor.NDArray")),
        encode=encode,
        decode=decode,
    )
