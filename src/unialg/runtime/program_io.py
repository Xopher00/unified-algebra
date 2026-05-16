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


def _term_value(term):
    """Structurally decode a reduced Hydra term into a Python value."""
    from hydra.core import (
        TermLiteral, TermPair, TermEither, TermUnit, TermList, TermMaybe,
    )
    from hydra.dsl.python import Just, Nothing, Left, Right

    if isinstance(term, TermUnit):
        return None
    if isinstance(term, TermLiteral):
        lit = term.value
        if hasattr(lit, 'value') and hasattr(lit.value, 'value'):
            return lit.value.value
        return lit.value
    if isinstance(term, TermPair):
        return (_term_value(term.value[0]), _term_value(term.value[1]))
    if isinstance(term, TermEither):
        inner = term.value
        if isinstance(inner, Left):
            return ("left", _term_value(inner.value))
        if isinstance(inner, Right):
            return ("right", _term_value(inner.value))
    if isinstance(term, TermList):
        return [_term_value(item) for item in term.value]
    if isinstance(term, TermMaybe):
        inner = term.value
        if isinstance(inner, Nothing):
            return None
        if isinstance(inner, Just):
            return _term_value(inner.value)
    return term


def decode_output(backend, codomain: Type, ctx, graph, result_term):
    """Decode a reduced Hydra term into a Python value.

    Always structurally decodes via _term_value.
    If a backend exists, additionally resolves native store handles.
    """
    value = _term_value(result_term)
    if backend is None:
        return value
    return backend.decode_boundary_output(codomain, value)
