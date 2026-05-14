"""Pratt grammar for polynomial functor expressions.

Operator binding powers (left-assoc: lbp=p, rbp=p+1):

  *  postfix  List(body)   BP 80
  &  infix    Prod(l, r)   BP 70
  |  infix    Sum(l, r)    BP 60

Atoms: 0 → Zero, 1 → One, x → Id, <name> → PolyRef or env lookup, (F) grouping.

Imports: objects, syntax/expressions, syntax/_pratt only. No semantics imports.
"""
from __future__ import annotations
from typing import Any

from unialg.syntax.expressions import PolyExpr, Zero, One, Id, PolyRef
from unialg.syntax._pratt import PrattParser, ParseError
from unialg.syntax._ops import functor_bp, make_list, make_prod, make_sum

Token = tuple[str, Any]
Env = dict[str, PolyExpr]


def _nud(env: Env, p: PrattParser, tok: Token) -> PolyExpr:
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
        if val == "x":
            return Id()
        if val in env:
            return env[val]
        return PolyRef(val)
    raise ParseError(f"unexpected token {kind!r} ({val!r}) in functor expression")


def _led(p: PrattParser, left: PolyExpr, tok: Token, rbp: int) -> PolyExpr:
    kind = tok[0]
    if kind == "STAR":
        return make_list(left)
    right: PolyExpr = p.parse(rbp)  # type: ignore[assignment]
    if kind == "PAIR":
        return make_prod(left, right)
    if kind == "CASE":
        return make_sum(left, right)
    raise ParseError(f"unknown functor operator {kind!r}")


def make_functor_grammar(env: Env | None = None) -> tuple[Any, Any, dict]:
    _env: Env = env or {}

    def nud(p: PrattParser, tok: Token) -> PolyExpr:
        return _nud(_env, p, tok)

    def led(p: PrattParser, left: object, tok: Token, rbp: int) -> PolyExpr:
        return _led(p, left, tok, rbp)  # type: ignore[arg-type]

    return nud, led, functor_bp()
