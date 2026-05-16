"""Program I/O boundary: native values ↔ Hydra terms for CompiledProgram.run().

Infers boundary types from the compiled term, packs multi-arg inputs,
and delegates encode/decode to existing BackendOps methods.
"""
from __future__ import annotations

from hydra.core import Type, TypeFunction
from hydra.dsl.python import Right
import hydra.inference as HI

from .codecs import coder_for_type, expect_right


def infer_boundary_types(term, ctx, graph) -> tuple[Type, Type]:
    """Infer (domain, codomain) from a compiled Hydra term."""
    result = HI.infer_type_of_term(ctx, graph, term, "program boundary")
    if not isinstance(result, Right):
        raise RuntimeError(f"program_io: could not infer boundary types: {result}")
    fn_type = result.value.type
    if not isinstance(fn_type, TypeFunction):
        raise RuntimeError(
            f"program_io: expected function type, got {type(fn_type).__name__}"
        )
    return fn_type.value.domain, fn_type.value.codomain


def pack_args(args: tuple):
    """Pack multiple native args into a left-nested pair structure."""
    if len(args) == 1:
        return args[0]
    out = (args[0], args[1])
    for arg in args[2:]:
        out = (out, arg)
    return out


def encode_input(backend, domain: Type, ctx, value):
    """Encode a native input value into a Hydra term via the backend store."""
    encoded = backend.encode_boundary_input(domain, value)
    coder = coder_for_type(domain)
    return expect_right(coder.decode(ctx, encoded), "program_io encode_input")


def decode_output(backend, codomain: Type, ctx, graph, result_term):
    """Decode a Hydra result term into a native output value via the backend store."""
    coder = coder_for_type(codomain)
    decoded = expect_right(coder.encode(ctx, graph, result_term), "program_io decode_output")
    return backend.decode_boundary_output(codomain, decoded)
