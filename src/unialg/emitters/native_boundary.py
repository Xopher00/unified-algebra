"""Native tensor boundary helpers for backend execution."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass, field

from hydra.core import LiteralType, Type, TypeEither, TypeList, TypeMaybe, TypePair, TypeLiteral
from hydra.dsl.python import Left, Right


def is_binary_type(typ: Type) -> bool:
    """Return True when a Hydra type is the BINARY handle type."""
    return isinstance(typ, TypeLiteral) and typ.value == LiteralType.BINARY


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
