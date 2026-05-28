"""Program and native-value boundary helpers for runtime execution."""

from __future__ import annotations

import io
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hydra.core import (
    LiteralTypeBinary,
    Type, TypeEither, TypeList,
    TypeMaybe, TypePair, TypeLiteral,
)
from hydra.dsl.python import Left, Right

from .codecs import coder_for_type, expect_right, term_value


class RuntimeStore:
    """UUID-keyed store for native tensor values during one CompiledProgram.run() call."""

    def __init__(self):
        self._data: dict[bytes, Any] = {}

    def reset(self) -> None:
        self._data.clear()

    def put(self, native) -> bytes:
        """Store a native value and return its 16-byte UUID handle."""
        key = uuid.uuid4().bytes
        self._data[key] = native
        return key

    def get(self, key: bytes):
        """Retrieve a native value by its handle. Raises KeyError on miss."""
        return self._data[key]


def is_binary_type(typ: Type) -> bool:
    """Return True when a Hydra type is the BINARY handle type."""
    return isinstance(typ, TypeLiteral) and isinstance(typ.value, LiteralTypeBinary)


@dataclass(frozen=True)
class BinaryAdapter:
    """Generic bytes-to-native tensor adapter using ``BytesIO`` framing."""

    dump_fn: Callable
    load_fn: Callable
    dump_style: str
    load_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.dump_style not in ("file_first", "value_first"):
            raise ValueError(
                f"BinaryAdapter: unknown dump_style {self.dump_style!r}. "
                "Expected 'file_first' or 'value_first'."
            )

    def dump(self, native) -> bytes:
        buf = io.BytesIO()
        if self.dump_style == "file_first":
            self.dump_fn(buf, native)
        elif self.dump_style == "value_first":
            self.dump_fn(native, buf)
        return buf.getvalue()

    def load(self, b: bytes):
        return self.load_fn(io.BytesIO(b), **self.load_kwargs)


def encode_boundary_input(typ: Type, value, put_binary):
    """Encode native BINARY leaves inside a whole-program input value."""
    if not is_binary_type(typ):
        if isinstance(typ, TypePair):
            left, right = value
            return (
                encode_boundary_input(typ.value.first, left, put_binary),
                encode_boundary_input(typ.value.second, right, put_binary),
            )
        if isinstance(typ, TypeList):
            return [encode_boundary_input(typ.value, item, put_binary) for item in value]
        if isinstance(typ, TypeMaybe):
            if value is None:
                return None
            return encode_boundary_input(typ.value, value, put_binary)
        if isinstance(typ, TypeEither):
            if isinstance(value, Left):
                return Left(encode_boundary_input(typ.value.left, value.value, put_binary))
            if isinstance(value, Right):
                return Right(encode_boundary_input(typ.value.right, value.value, put_binary))
        return value
    return put_binary(value)


def decode_boundary_output(typ: Type, value, get_binary):
    """Decode native BINARY leaves inside a whole-program output value."""
    if not is_binary_type(typ):
        if isinstance(typ, TypePair):
            left, right = value
            return (
                decode_boundary_output(typ.value.first, left, get_binary),
                decode_boundary_output(typ.value.second, right, get_binary),
            )
        if isinstance(typ, TypeList):
            return [decode_boundary_output(typ.value, item, get_binary) for item in value]
        if isinstance(typ, TypeMaybe):
            if value is None:
                return None
            return decode_boundary_output(typ.value, value, get_binary)
        if isinstance(typ, TypeEither):
            tag, branch = value
            if tag == "left":
                return ("left", decode_boundary_output(typ.value.left, branch, get_binary))
            if tag == "right":
                return ("right", decode_boundary_output(typ.value.right, branch, get_binary))
        return value
    return get_binary(value)


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
    return expect_right(coder.decode(ctx, encoded), "runtime encode_input")


def decode_output(backend, codomain: Type, ctx, graph, result_term):
    """Decode a reduced Hydra term into a Python value.

    Always structurally decodes via ``codecs.term_value``. If a backend exists,
    additionally resolves native store handles.
    """
    value = term_value(result_term, "runtime decode_output")
    if backend is None:
        return value
    return backend.decode_boundary_output(codomain, value)
