"""Hydra TermCoder and Type registries for backend primitive encoding.

Maps string keys from JSON specs (``"FLOAT"``, ``"float64"``, etc.) to Hydra
``Type`` objects and ``TermCoder`` instances.  No morphism or expression
dependencies — this layer only bridges Python scalars to Hydra literal terms.
"""

from __future__ import annotations

from typing import Callable

from hydra.core import (
    LiteralType,
    Term,
    TermEither,
    TermList,
    TermLiteral,
    TermPair,
    TermUnit,
    Type,
    TypeLiteral,
)
from hydra.dsl.python import Left, Right
from hydra.graph import TermCoder
import hydra.dsl.meta.phantoms as P


def _expect_right(result, context: str):
    """Unwrap a Hydra Either result or raise a readable error."""
    if isinstance(result, Left):
        raise TypeError(f"{context}: {result.value!r}")
    return result.value


def _literal_value(term: Term, context: str):
    """Extract the Python literal payload from a Hydra literal term."""
    try:
        return term.value.value.value
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
    """Construct a Hydra TermCoder from native decode/encode callables."""
    return TermCoder(
        type=typ,
        encode=lambda _cx, _graph, term: Right(decode_term(term)),
        decode=lambda _cx, value: Right(encode_value(value)),
    )


TYPE_REGISTRY: dict[str, Type] = {
    "INT":   TypeLiteral(LiteralType.INTEGER),
    "FLOAT": TypeLiteral(LiteralType.FLOAT),
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
