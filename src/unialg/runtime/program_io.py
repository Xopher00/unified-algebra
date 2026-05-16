"""Program I/O boundary: native values ↔ Hydra terms for CompiledProgram.run().

Packs multi-arg inputs and delegates encode/decode to existing BackendOps methods.
"""
from __future__ import annotations

from hydra.core import Type
from hydra.dsl.python import Right

from .codecs import coder_for_type, expect_right


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
