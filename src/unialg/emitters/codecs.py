"""Hydra TermCoder and Type codec layer for backend primitive encoding.

Type-directed architecture: `type_from_spec` parses JSON type declarations into Hydra
Types; `coder_for_type` derives a TermCoder recursively from any supported Type.

Boundary: this module knows Hydra types and plain Python values (int, float, str, bool,
bytes, list, tuple, None, Left/Right). It does NOT import numpy, torch, cupy, jax, or
any framework-specific serialization.

Public API:
  type_from_spec(spec) -> Type       -- parse JSON type declaration
  coder_for_type(typ)  -> TermCoder  -- derive coder from Hydra Type
  encode_python(value) -> Term       -- convenience helper; prefer coder_for_type when type is known

Legacy (backward compat, not used by load_spec):
  TYPE_REGISTRY         -- str -> Type
  TERM_CODER_REGISTRY   -- str -> TermCoder
"""

from __future__ import annotations

from typing import Callable

from hydra.core import (
    EitherType,
    LiteralBinary,
    LiteralBoolean,
    LiteralString,
    LiteralType,
    PairType,
    Term,
    TermEither,
    TermList,
    TermLiteral,
    TermMaybe,
    TermPair,
    TermUnit,
    Type,
    TypeEither,
    TypeList,
    TypeLiteral,
    TypeMaybe,
    TypePair,
    TypeUnit,
)
from hydra.dsl.python import Just, Left, Nothing, Right
from hydra.graph import TermCoder
import hydra.dsl.meta.phantoms as P


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expect_right(result, context: str):
    """Unwrap a Hydra Either result or raise a readable error."""
    if isinstance(result, Left):
        raise TypeError(f"{context}: {result.value!r}")
    return result.value


def _literal_value(term: Term, context: str):
    """Extract the Python literal payload from a Hydra literal term.

    Depth from ``lit = term.value``:
      LiteralBinary / LiteralBoolean / LiteralString: lit.value      (1 level)
      LiteralFloat / LiteralInteger:                  lit.value.value (2 levels, via FloatValueFloat64 / IntegerValueInt64)
    """
    try:
        lit = term.value
        if isinstance(lit, (LiteralBinary, LiteralBoolean, LiteralString)):
            return lit.value
        return lit.value.value  # FloatValueFloat64.value or IntegerValueInt64.value
    except Exception as e:
        raise TypeError(f"{context}: expected literal term, got {term!r}") from e


def _term_value(term: Term, context: str):
    """Decode ordinary Hydra values into small Python structures."""
    match term:
        case TermLiteral():
            return _literal_value(term, context)
        case TermUnit():
            return None
        case TermPair(value=pair):
            left, right = pair
            return (_term_value(left, context), _term_value(right, context))
        case TermList(value=items):
            return [_term_value(item, context) for item in items]
        case TermMaybe(value=m):
            if isinstance(m, Nothing):
                return None
            return _term_value(m.value, context)
        case TermEither(value=branch):
            if isinstance(branch, Left):
                return ("left", _term_value(branch.value, context))
            if isinstance(branch, Right):
                return ("right", _term_value(branch.value, context))
    raise TypeError(f"{context}: expected value term, got {term!r}")


def _mk_term_coder(
    typ: Type,
    decode_term: Callable[[Term], object],
    encode_value: Callable[[object], Term],
) -> TermCoder:
    """Construct a Hydra TermCoder from native decode/encode callables.

    Hydra's naming convention (intentionally inverted from intuition):
      encode(ctx, graph, term) -> Either[str, Python]   -- Term to Python
      decode(ctx, value)       -> Either[str, Term]      -- Python to Term
    """
    return TermCoder(
        type=typ,
        encode=lambda _cx, _graph, term: Right(decode_term(term)),
        decode=lambda _cx, value: Right(encode_value(value)),
    )


# ---------------------------------------------------------------------------
# Type-directed API
# ---------------------------------------------------------------------------

def type_from_spec(spec: str | dict) -> Type:
    """Parse a JSON type declaration into a Hydra Type.

    Shorthands: ``"FLOAT"``, ``"INT"``, ``"STRING"``, ``"BOOL"``, ``"BINARY"``, ``"UNIT"``
    Structured:
      ``{"list": T}``
      ``{"pair": [A, B]}``
      ``{"either": [L, R]}``
      ``{"maybe": T}``
    """
    if isinstance(spec, str):
        match spec:
            case "FLOAT":  return TypeLiteral(LiteralType.FLOAT)
            case "INT":    return TypeLiteral(LiteralType.INTEGER)
            case "STRING": return TypeLiteral(LiteralType.STRING)
            case "BOOL":   return TypeLiteral(LiteralType.BOOLEAN)
            case "BINARY": return TypeLiteral(LiteralType.BINARY)
            case "UNIT":   return TypeUnit()
            case _:        raise ValueError(f"type_from_spec: unknown shorthand {spec!r}")
    if "list" in spec:
        return TypeList(type_from_spec(spec["list"]))
    if "pair" in spec:
        a, b = spec["pair"]
        return TypePair(PairType(type_from_spec(a), type_from_spec(b)))
    if "either" in spec:
        l, r = spec["either"]
        return TypeEither(EitherType(type_from_spec(l), type_from_spec(r)))
    if "maybe" in spec:
        return TypeMaybe(type_from_spec(spec["maybe"]))
    raise ValueError(f"type_from_spec: unknown spec {spec!r}")


def _list_coder(elem_coder: TermCoder) -> TermCoder:
    typ = TypeList(elem_coder.type)

    def decode_term(term):
        if not isinstance(term, TermList):
            raise TypeError(f"list coder: expected TermList, got {type(term).__name__}")
        return [_expect_right(elem_coder.encode(None, None, item), "list coder") for item in term.value]

    def encode_value(value):
        return TermList(value=[_expect_right(elem_coder.decode(None, item), "list coder") for item in value])

    return _mk_term_coder(typ, decode_term, encode_value)


def _pair_coder(first_coder: TermCoder, second_coder: TermCoder) -> TermCoder:
    typ = TypePair(PairType(first_coder.type, second_coder.type))

    def decode_term(term):
        if not isinstance(term, TermPair):
            raise TypeError(f"pair coder: expected TermPair, got {type(term).__name__}")
        a, b = term.value
        return (
            _expect_right(first_coder.encode(None, None, a), "pair coder"),
            _expect_right(second_coder.encode(None, None, b), "pair coder"),
        )

    def encode_value(value):
        a, b = value
        return TermPair(value=(
            _expect_right(first_coder.decode(None, a), "pair coder"),
            _expect_right(second_coder.decode(None, b), "pair coder"),
        ))

    return _mk_term_coder(typ, decode_term, encode_value)


def _maybe_coder(elem_coder: TermCoder) -> TermCoder:
    typ = TypeMaybe(elem_coder.type)

    def decode_term(term):
        if not isinstance(term, TermMaybe):
            raise TypeError(f"maybe coder: expected TermMaybe, got {type(term).__name__}")
        m = term.value
        if isinstance(m, Nothing):
            return None
        return _expect_right(elem_coder.encode(None, None, m.value), "maybe coder")

    def encode_value(value):
        if value is None:
            return TermMaybe(value=Nothing())
        return TermMaybe(value=Just(_expect_right(elem_coder.decode(None, value), "maybe coder")))

    return _mk_term_coder(typ, decode_term, encode_value)


def _either_coder(left_coder: TermCoder, right_coder: TermCoder) -> TermCoder:
    typ = TypeEither(EitherType(left_coder.type, right_coder.type))

    def decode_term(term):
        if not isinstance(term, TermEither):
            raise TypeError(f"either coder: expected TermEither, got {type(term).__name__}")
        branch = term.value
        if isinstance(branch, Left):
            return Left(_expect_right(left_coder.encode(None, None, branch.value), "either coder"))
        if isinstance(branch, Right):
            return Right(_expect_right(right_coder.encode(None, None, branch.value), "either coder"))
        raise TypeError(f"either coder: unexpected branch: {branch!r}")

    def encode_value(value):
        if isinstance(value, Left):
            return TermEither(value=Left(_expect_right(left_coder.decode(None, value.value), "either coder")))
        if isinstance(value, Right):
            return TermEither(value=Right(_expect_right(right_coder.decode(None, value.value), "either coder")))
        raise TypeError(f"either coder: expected Left or Right, got {type(value).__name__}")

    return _mk_term_coder(typ, decode_term, encode_value)


def _unit_encode(x):
    if x is not None:
        raise TypeError(f"UNIT coder: expected None, got {type(x).__name__!r}")
    return TermUnit()


def _unit_decode(term):
    if not isinstance(term, TermUnit):
        raise TypeError(f"UNIT coder: expected TermUnit, got {type(term).__name__!r}")
    return None


_unit_coder: TermCoder = _mk_term_coder(TypeUnit(), _unit_decode, _unit_encode)

def _float_encode(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"FLOAT coder: expected int or float, got {type(x).__name__!r}")
    return P.float64(float(x)).value  # explicit float() ensures FloatValueFloat64 stores float, not int


def _int_encode(x):
    if isinstance(x, bool) or not isinstance(x, int):
        raise TypeError(f"INT coder: expected int (not bool), got {type(x).__name__!r}")
    return P.int64(x).value


def _string_encode(x):
    if not isinstance(x, str):
        raise TypeError(f"STRING coder: expected str, got {type(x).__name__!r}")
    return P.string(x).value


def _bool_encode(x):
    if not isinstance(x, bool):
        raise TypeError(f"BOOL coder: expected bool, got {type(x).__name__!r}")
    return P.boolean(x).value


def _binary_encode(x):
    if not isinstance(x, (bytes, bytearray)):
        raise TypeError(f"BINARY coder: expected bytes or bytearray, got {type(x).__name__!r}")
    return P.binary(bytes(x)).value


_LITERAL_CODERS: dict = {
    LiteralType.FLOAT:   _mk_term_coder(TypeLiteral(LiteralType.FLOAT),   lambda t: _literal_value(t, "FLOAT coder"),  _float_encode),
    LiteralType.INTEGER: _mk_term_coder(TypeLiteral(LiteralType.INTEGER), lambda t: _literal_value(t, "INT coder"),    _int_encode),
    LiteralType.STRING:  _mk_term_coder(TypeLiteral(LiteralType.STRING),  lambda t: _literal_value(t, "STRING coder"), _string_encode),
    LiteralType.BOOLEAN: _mk_term_coder(TypeLiteral(LiteralType.BOOLEAN), lambda t: _literal_value(t, "BOOL coder"),   _bool_encode),
    LiteralType.BINARY:  _mk_term_coder(TypeLiteral(LiteralType.BINARY),  lambda t: _literal_value(t, "BINARY coder"), _binary_encode),
}


def coder_for_type(typ: Type) -> TermCoder:
    """Derive a TermCoder from a Hydra Type. Recursive for compound types."""
    if isinstance(typ, TypeUnit):
        return _unit_coder
    if isinstance(typ, TypeLiteral):
        coder = _LITERAL_CODERS.get(typ.value)
        if coder is None:
            raise TypeError(f"coder_for_type: no coder for literal type {typ.value!r}")
        return coder
    if isinstance(typ, TypeList):
        return _list_coder(coder_for_type(typ.value))
    if isinstance(typ, TypePair):
        return _pair_coder(coder_for_type(typ.value.first), coder_for_type(typ.value.second))
    if isinstance(typ, TypeMaybe):
        return _maybe_coder(coder_for_type(typ.value))
    if isinstance(typ, TypeEither):
        return _either_coder(coder_for_type(typ.value.left), coder_for_type(typ.value.right))
    raise TypeError(f"coder_for_type: unsupported type {type(typ).__name__!r}")


def encode_python(value) -> Term:
    """Encode a plain Python value to a Hydra Term via type detection.

    Convenience helper for ad-hoc use. Prefer ``coder_for_type(known_type).decode(ctx, v)``
    when the Hydra type is known at the call site.
    """
    if value is None:            return TermUnit()
    if isinstance(value, bool):  return P.boolean(value).value
    if isinstance(value, int):   return P.int64(int(value)).value
    if isinstance(value, float): return P.float64(float(value)).value
    if isinstance(value, str):   return P.string(value).value
    if isinstance(value, bytes): return P.binary(value).value
    try:
        return TermList(value=[encode_python(item) for item in value])
    except TypeError:
        pass
    raise TypeError(f"encode_python: cannot encode {type(value).__name__!r}")


# ---------------------------------------------------------------------------
# Legacy registries (backward compat — not used by load_spec)
# ---------------------------------------------------------------------------

TYPE_REGISTRY: dict[str, Type] = {
    "INT":    TypeLiteral(LiteralType.INTEGER),
    "FLOAT":  TypeLiteral(LiteralType.FLOAT),
    "STRING": TypeLiteral(LiteralType.STRING),
    "BOOL":   TypeLiteral(LiteralType.BOOLEAN),
    "BINARY": TypeLiteral(LiteralType.BINARY),
    "UNIT":   TypeUnit(),
}

TERM_CODER_REGISTRY: dict[str, TermCoder] = {
    "int32": _mk_term_coder(
        TypeLiteral(LiteralType.INTEGER),
        lambda t: int(_literal_value(t, "int32 coder")),
        lambda x: P.int32(int(x)).value,
    ),
    "int64": _mk_term_coder(
        TypeLiteral(LiteralType.INTEGER),
        lambda t: int(_literal_value(t, "int64 coder")),
        lambda x: P.int64(int(x)).value,
    ),
    "float32": _mk_term_coder(
        TypeLiteral(LiteralType.FLOAT),
        lambda t: float(_literal_value(t, "float32 coder")),
        lambda x: P.float32(float(x)).value,
    ),
    "float64": _mk_term_coder(
        TypeLiteral(LiteralType.FLOAT),
        lambda t: float(_literal_value(t, "float64 coder")),
        lambda x: P.float64(float(x)).value,
    ),
}
