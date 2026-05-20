"""Hydra TermCoder and Type codec layer for backend primitive encoding.

Type-directed architecture: `type_from_spec` parses JSON type declarations into Hydra
Types; `coder_for_type` derives a TermCoder recursively from any supported Type.

Boundary: this module knows Hydra types and plain Python values (int, float, str, bool,
bytes, list, tuple, None, Left/Right). It does NOT import numpy, torch, cupy, jax, or
any framework-specific serialization.

Public API:
  type_from_spec(spec) -> Type       -- parse JSON type declaration
  coder_for_type(typ)  -> TermCoder  -- derive coder from Hydra Type
"""

from __future__ import annotations

from typing import Callable

from hydra.core import (
    EitherType, PairType, LiteralType,
    LiteralBinary, LiteralBoolean, LiteralString, 
    Term, TermEither, TermList, TermLiteral, TermMaybe, TermPair, TermUnit,
    Type, TypeEither, TypeList, TypeLiteral, TypeMaybe, TypePair, TypeUnit,
)
from hydra.dsl.python import Just, Left, Nothing, Right
from hydra.graph import TermCoder
import hydra.dsl.meta.phantoms as P
import hydra.show.core as ShowCore
import hydra.show.errors as ShowErrors


def _show_term(term) -> str:
    try:
        return ShowCore.term(term)
    except (AssertionError, AttributeError):
        return repr(term)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def expect_right(result, context: str):
    """Unwrap a Hydra Either result or raise a readable error."""
    if isinstance(result, Left):
        raise TypeError(f"{context}: {ShowErrors.error(result.value)}")
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
        raise TypeError(f"{context}: expected literal term, got {_show_term(term)}") from e


def _term_value_maybe(m, context: str):
    if isinstance(m, Nothing):
        return None
    return term_value(m.value, context)


def _term_value_either(branch, context: str):
    if isinstance(branch, Left):
        return ("left", term_value(branch.value, context))
    if isinstance(branch, Right):
        return ("right", term_value(branch.value, context))
    raise TypeError(f"{context}: unexpected Either branch {type(branch).__name__!r}")


def term_value(term: Term, context: str = "term_value"):
    """Decode ordinary Hydra values into small Python structures.

    Binary literals are returned as raw bytes. RuntimeStore dereferencing is a
    separate program-boundary concern handled by ``boundary.decode_output`` and
    ``BackendOps.decode_boundary_output``.
    """
    match term:
        case TermLiteral():
            return _literal_value(term, context)
        case TermUnit():
            return None
        case TermPair(value=pair):
            left, right = pair
            return (term_value(left, context), term_value(right, context))
        case TermList(value=items):
            return [term_value(item, context) for item in items]
        case TermMaybe(value=m):
            return _term_value_maybe(m, context)
        case TermEither(value=branch):
            return _term_value_either(branch, context)
    raise TypeError(f"{context}: expected value term, got {_show_term(term)}")


def _mk_term_coder(typ: Type, 
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

_TYPE_SHORTHANDS: dict[str, Type] = {
    "FLOAT":  TypeLiteral(LiteralType.FLOAT),
    "INT":    TypeLiteral(LiteralType.INTEGER),
    "STRING": TypeLiteral(LiteralType.STRING),
    "BOOL":   TypeLiteral(LiteralType.BOOLEAN),
    "BINARY": TypeLiteral(LiteralType.BINARY),
    "UNIT":   TypeUnit(),
}


def _type_from_dict(spec: dict) -> Type:
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
        t = _TYPE_SHORTHANDS.get(spec)
        if t is None:
            raise ValueError(f"type_from_spec: unknown shorthand {spec!r}")
        return t
    return _type_from_dict(spec)


def _list_coder(elem_coder: TermCoder) -> TermCoder:
    typ = TypeList(elem_coder.type)

    def decode_term(term):
        if not isinstance(term, TermList):
            raise TypeError(f"list coder: expected TermList, got {type(term).__name__}")
        return [expect_right(elem_coder.encode(None, None, item), "list coder") for item in term.value]

    def encode_value(value):
        return TermList(value=[expect_right(elem_coder.decode(None, item), "list coder") for item in value])

    return _mk_term_coder(typ, decode_term, encode_value)


def _pair_coder(first_coder: TermCoder, second_coder: TermCoder) -> TermCoder:
    typ = TypePair(PairType(first_coder.type, second_coder.type))

    def decode_term(term):
        if not isinstance(term, TermPair):
            raise TypeError(f"pair coder: expected TermPair, got {type(term).__name__}")
        a, b = term.value
        return (
            expect_right(first_coder.encode(None, None, a), "pair coder"),
            expect_right(second_coder.encode(None, None, b), "pair coder"),
        )

    def encode_value(value):
        a, b = value
        return TermPair(value=(
            expect_right(first_coder.decode(None, a), "pair coder"),
            expect_right(second_coder.decode(None, b), "pair coder"),
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
        return expect_right(elem_coder.encode(None, None, m.value), "maybe coder")

    def encode_value(value):
        if value is None:
            return TermMaybe(value=Nothing())
        return TermMaybe(value=Just(expect_right(elem_coder.decode(None, value), "maybe coder")))

    return _mk_term_coder(typ, decode_term, encode_value)


def _either_coder(left_coder: TermCoder, right_coder: TermCoder) -> TermCoder:
    typ = TypeEither(EitherType(left_coder.type, right_coder.type))

    def decode_term(term):
        if not isinstance(term, TermEither):
            raise TypeError(f"either coder: expected TermEither, got {type(term).__name__}")
        branch = term.value
        if isinstance(branch, Left):
            return Left(expect_right(left_coder.encode(None, None, branch.value), "either coder"))
        if isinstance(branch, Right):
            return Right(expect_right(right_coder.encode(None, None, branch.value), "either coder"))
        raise TypeError(f"either coder: unexpected branch: {branch!r}")

    def encode_value(value):
        if isinstance(value, Left):
            return TermEither(value=Left(expect_right(left_coder.decode(None, value.value), "either coder")))
        if isinstance(value, Right):
            return TermEither(value=Right(expect_right(right_coder.decode(None, value.value), "either coder")))
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
