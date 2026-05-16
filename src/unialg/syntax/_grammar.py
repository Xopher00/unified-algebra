"""Pratt grammar callbacks for morphism and functor expressions.

No env/fenv resolution. All names become Ref/PolyRef for the semantic
construction pass to resolve.
"""
from __future__ import annotations
from typing import Any

from unialg.syntax.expressions import (
    MorphismExpr, PolyExpr, PolyFmap, MorphismApp,
    Identity, Copy, Delete, First, Second, Left, Right,
    Absurd, Assoc, Symmetry, Ref, PolyRef,
    Id, Zero, One, List as PolyList, Maybe as PolyMaybe,
)
from unialg.syntax._pratt import PrattParser, ParseError
from unialg.syntax._ops import (
    _U, _PU, _SU,
    morphism_bp, functor_bp,
    make_binary, make_compose, make_pair,
    make_prod, make_sum,
)

Token = tuple[str, Any]


def _copy_power(count: int) -> MorphismExpr:
    if count < 2:
        raise ParseError("copy power expects an integer >= 2")
    out: MorphismExpr = Copy(_U)
    for _ in range(2, count):
        out = make_pair(out, Identity(_U))
    return out


def _case_injection(index: int) -> MorphismExpr:
    if index == 0:
        return Left(_SU)
    if index == 1:
        return Right(_SU)
    raise ParseError("case injection index must be 0 or 1")


def _parse_arg_list(p: PrattParser) -> tuple[MorphismExpr, ...]:
    args: list[MorphismExpr] = [p.parse(0)]  # type: ignore[assignment]
    while p.peek()[0] == "COMMA":
        p.advance()
        args.append(p.parse(0))  # type: ignore[arg-type]
    p.expect("RPAREN", "closing )")
    return tuple(args)


# ---------------------------------------------------------------------------
# Morphism grammar
# ---------------------------------------------------------------------------

def _morphism_nud(p: PrattParser, tok: Token) -> MorphismExpr:
    kind, val = tok

    if kind == "LPAREN":
        inner = p.parse(0)
        p.expect("RPAREN", "closing )")
        return inner  # type: ignore[return-value]

    if kind == "LBRACKET":
        _, index = p.expect("INT", "projection index")
        p.expect("RBRACKET", "closing ]")
        if index == 0:
            return First(_PU)
        if index == 1:
            return Second(_PU)
        raise ParseError("projection index must be 0 or 1")

    if kind == "BANG":
        return Delete(_U)

    if kind == "STAR":
        if p.peek()[0] != "INT":
            raise ParseError("copy power expects an integer >= 2")
        _, count = p.expect("INT", "copy power")
        return _copy_power(count)

    if kind == "CASE":
        if p.peek()[0] != "INT":
            raise ParseError("'|' in prefix position expects 0 or 1")
        _, index = p.expect("INT", "injection index")
        return _case_injection(index)

    if kind == "INT":
        if val == 0:
            return Absurd(_U)
        if val == 1:
            return Delete(_U)
        raise ParseError(f"integer {val!r} is not valid in morphism context")

    if kind == "NAME":
        name: str = val

        if name == "x":
            if p.peek()[0] == "LBRACE":
                p.advance()
                f = p.parse(0)
                p.expect("RBRACE", "closing }")
                return PolyFmap(body=Id(), f=f,  # type: ignore[arg-type]
                                param=_U, monad=None, dom=_U, cod=_U)
            return Identity(_U)

        if name in ("delete", "drop", "del"):
            return Delete(_U)
        if name == "copy":
            return Copy(_U)
        if name == "dup":
            p.expect("LPAREN", "(")
            _, count = p.expect("INT", "dup count")
            p.expect("RPAREN", "closing )")
            return _copy_power(count)
        if name in ("id", "identity"):
            return Identity(_U)
        if name == "absurd":
            return Absurd(_U)
        if name == "assoc":
            return Assoc(_U, _U)
        if name in ("sym", "symmetry"):
            return Symmetry(_U, _U)

        # name{f} — functor action
        if p.peek()[0] == "LBRACE":
            p.advance()
            f = p.parse(0)
            p.expect("RBRACE", "closing }")
            return PolyFmap(body=PolyRef(name), f=f,  # type: ignore[arg-type]
                            param=_U, monad=None, dom=_U, cod=_U)

        # name(args) — parametric application
        if p.peek()[0] == "LPAREN":
            p.advance()
            args = _parse_arg_list(p)
            return MorphismApp(Ref(name), args)

        return Ref(name)

    raise ParseError(f"unexpected token {kind!r} ({val!r}) in morphism expression")


def _morphism_led(p: PrattParser, left: MorphismExpr, tok: Token, rbp: int) -> MorphismExpr:
    kind = tok[0]

    if kind == "LBRACKET":
        _, index = p.expect("INT", "projection index")
        p.expect("RBRACKET", "closing ]")
        if index == 0:
            return make_compose(left, First(_PU))
        if index == 1:
            return make_compose(left, Second(_PU))
        raise ParseError("projection index must be 0 or 1")

    if kind == "STAR":
        if p.peek()[0] != "INT":
            raise ParseError("copy power expects an integer >= 2")
        _, count = p.expect("INT", "copy power")
        return make_compose(left, _copy_power(count))

    if kind == "CASE":
        if p.peek()[0] == "INT":
            _, index = p.expect("INT", "injection index")
            return make_compose(left, _case_injection(index))
        right: MorphismExpr = p.parse(rbp)  # type: ignore[assignment]
        return make_binary("CASE", left, right)

    right = p.parse(rbp)  # type: ignore[assignment]
    return make_binary(kind, left, right)


def make_morphism_grammar() -> tuple[Any, Any, dict]:
    return _morphism_nud, _morphism_led, morphism_bp()


# ---------------------------------------------------------------------------
# Functor grammar
# ---------------------------------------------------------------------------

def _functor_nud(p: PrattParser, tok: Token) -> PolyExpr:
    kind, val = tok

    if kind == "LPAREN":
        inner = p.parse(0)
        p.expect("RPAREN", "closing )")
        return inner  # type: ignore[return-value]

    if kind == "INT":
        if val == 0:
            return Zero()
        if val == 1:
            return One()
        raise ParseError(f"unexpected integer {val!r} in functor expression")

    if kind == "NAME":
        if val in ("x", "id"):
            return Id()
        if val == "List":
            p.expect("LBRACKET", "[")
            inner = p.parse(0)
            p.expect("RBRACKET", "]")
            return PolyList(inner)  # type: ignore[arg-type]
        if val == "Maybe":
            p.expect("LBRACKET", "[")
            inner = p.parse(0)
            p.expect("RBRACKET", "]")
            return PolyMaybe(inner)  # type: ignore[arg-type]
        return PolyRef(val)

    raise ParseError(f"unexpected token {kind!r} ({val!r}) in functor expression")


def _functor_led(p: PrattParser, left: PolyExpr, tok: Token, rbp: int) -> PolyExpr:
    kind = tok[0]
    right: PolyExpr = p.parse(rbp)  # type: ignore[assignment]
    if kind == "PAIR":
        return make_prod(left, right)
    if kind == "CASE":
        return make_sum(left, right)
    raise ParseError(f"unknown functor operator {kind!r}")


def make_functor_grammar() -> tuple[Any, Any, dict]:
    return _functor_nud, _functor_led, functor_bp()
