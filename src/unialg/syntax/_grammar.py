"""Pratt grammar callbacks for morphism and functor expressions.

No env/fenv resolution. All names become Ref/PolyRef for the semantic
construction pass to resolve.
"""
from __future__ import annotations
from typing import Any

from unialg.syntax.expressions import (
    MorphismExpr, PolyExpr, PolyFmap, MorphismApp, RecursionApp,
    CarrierBoundary, MonadicLift, FocusExpr, FocusRef, FocusCompose,
    Identity, Copy, Delete, Literal, First, Second, Left, Right,
    Absurd, Assoc, Symmetry, DistributeLeft, DistributeRight, Ref, PolyRef,
    Id, Zero, One, Exp as PolyExp, List as PolyList, Maybe as PolyMaybe,
    Rose as PolyRose, Tree as PolyTree,
)
from unialg.syntax._pratt import PrattParser, ParseError
from unialg.syntax._ops import (
    _U, _PU, _SU,
    morphism_bp, functor_bp,
    make_binary, make_compose, make_pair,
    make_prod, make_sum, make_poly_compose,
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


def _parse_recursion_app(p: PrattParser, kind: str) -> RecursionApp:
    p.expect("LBRACKET", "[")
    focus = str(p.expect("NAME", "focus name")[1])
    p.expect("RBRACKET", "closing ]")
    p.expect("LPAREN", "(")
    args = _parse_arg_list(p)
    return RecursionApp(kind=kind, focus=focus, args=args)


def _parse_carrier_boundary(p: PrattParser, kind: str) -> CarrierBoundary:
    p.expect("LBRACKET", "[")
    focus = str(p.expect("NAME", "focus name")[1])
    p.expect("RBRACKET", "closing ]")
    return CarrierBoundary(kind=kind, focus=focus)


def _parse_monadic_lift(p: PrattParser) -> MonadicLift:
    p.expect("LBRACKET", "[")
    monad = str(p.expect("NAME", "monad name")[1])
    p.expect("RBRACKET", "closing ]")
    p.expect("LPAREN", "(")
    body = p.parse(0)
    p.expect("RPAREN", "closing )")
    return MonadicLift(monad=monad, body=body)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Morphism grammar
# ---------------------------------------------------------------------------

_STRUCTURAL_NAME_MAP: dict[str, MorphismExpr] = {
    "delete": Delete(_U), "drop": Delete(_U), "del": Delete(_U),
    "copy": Copy(_U),
    "id": Identity(_U), "identity": Identity(_U),
    "absurd": Absurd(_U),
    "assoc": Assoc(_U, _U),
    "sym": Symmetry(_U, _U), "symmetry": Symmetry(_U, _U),
    "merge": make_binary("CASE", Identity(_U), Identity(_U)),
    "distl": DistributeLeft(_U, _U),
    "distr": DistributeRight(_U, _U),
}


def _nud_lbracket(p: PrattParser) -> MorphismExpr:
    _, index = p.expect("INT", "projection index")
    p.expect("RBRACKET", "closing ]")
    if index == 0:
        return First(_PU)
    if index == 1:
        return Second(_PU)
    raise ParseError("projection index must be 0 or 1")


def _nud_int(val: int) -> MorphismExpr:
    if val == 0:
        return Absurd(_U)
    if val == 1:
        return Delete(_U)
    raise ParseError(f"integer {val!r} is not valid in morphism context")


def _nud_lookahead_forms(p: PrattParser, name: str) -> MorphismExpr | None:
    if name == "pure" and p.peek()[0] == "LBRACKET":
        return _parse_monadic_lift(p)
    if name in ("cata", "ana", "hylo") and p.peek()[0] == "LBRACKET":
        return _parse_recursion_app(p, name)
    if name in ("roll", "unroll") and p.peek()[0] == "LBRACKET":
        return _parse_carrier_boundary(p, name)
    return None


def _nud_name_token(p: PrattParser, name: str) -> MorphismExpr:
    if name == "x":
        if p.peek()[0] == "LBRACE":
            p.advance()
            f = p.parse(0)
            p.expect("RBRACE", "closing }")
            return PolyFmap(body=Id(), f=f,  # type: ignore[arg-type]
                            param=_U, monad=None, dom=_U, cod=_U)
        return Identity(_U)
    structural = _STRUCTURAL_NAME_MAP.get(name)
    if structural is not None:
        return structural
    if name == "dup":
        p.expect("LPAREN", "(")
        _, count = p.expect("INT", "dup count")
        p.expect("RPAREN", "closing )")
        return _copy_power(count)
    lookahead = _nud_lookahead_forms(p, name)
    if lookahead is not None:
        return lookahead
    if p.peek()[0] == "LBRACE":
        p.advance()
        f = p.parse(0)
        p.expect("RBRACE", "closing }")
        return PolyFmap(body=PolyRef(name), f=f,  # type: ignore[arg-type]
                        param=_U, monad=None, dom=_U, cod=_U)
    if p.peek()[0] == "LPAREN":
        p.advance()
        args = _parse_arg_list(p)
        return MorphismApp(Ref(name), args)
    from unialg.extensions import get_expr_handler
    ext_handler = get_expr_handler(name)
    if ext_handler is not None:
        return ext_handler(p)
    return Ref(name)


def _nud_paren(p: PrattParser, _val) -> MorphismExpr:
    inner = p.parse(0)
    p.expect("RPAREN", "closing )")
    return inner  # type: ignore[return-value]


def _nud_star(p: PrattParser, _val) -> MorphismExpr:
    if p.peek()[0] != "INT":
        raise ParseError("copy power expects an integer >= 2")
    _, count = p.expect("INT", "copy power")
    return _copy_power(count)


def _nud_case(p: PrattParser, _val) -> MorphismExpr:
    if p.peek()[0] != "INT":
        raise ParseError("'|' in prefix position expects 0 or 1")
    _, index = p.expect("INT", "injection index")
    return _case_injection(index)


_MORPHISM_NUD_DISPATCH: dict[str, Any] = {
    "LPAREN": _nud_paren,
    "LBRACKET": lambda p, _v: _nud_lbracket(p),
    "BANG": lambda _p, _v: Delete(_U),
    "STAR": _nud_star,
    "CASE": _nud_case,
    "INT": lambda _p, v: _nud_int(v),
    "QUOTED": lambda _p, v: Literal(v, None, _U),
    "NAME": lambda p, v: _nud_name_token(p, v),
}


def _morphism_nud(p: PrattParser, tok: Token) -> MorphismExpr:
    kind, val = tok
    handler = _MORPHISM_NUD_DISPATCH.get(kind)
    if handler is not None:
        return handler(p, val)
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

def _functor_nud_name(p: PrattParser, val: str) -> PolyExpr:
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
    if val == "Exp":
        p.expect("LBRACKET", "[")
        base = p.parse(0)
        p.expect("COMMA", ",")
        body = p.parse(0)
        p.expect("RBRACKET", "]")
        return PolyExp(base, body)  # type: ignore[arg-type]
    if val == "Rose":
        p.expect("LBRACKET", "[")
        inner = p.parse(0)
        p.expect("RBRACKET", "]")
        return PolyRose(inner)  # type: ignore[arg-type]
    if val == "Tree":
        p.expect("LBRACKET", "[")
        inner = p.parse(0)
        p.expect("RBRACKET", "]")
        return PolyTree(inner)  # type: ignore[arg-type]
    return PolyRef(val)


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
        return _functor_nud_name(p, val)

    raise ParseError(f"unexpected token {kind!r} ({val!r}) in functor expression")


def _functor_led(p: PrattParser, left: PolyExpr, tok: Token, rbp: int) -> PolyExpr:
    kind = tok[0]
    right: PolyExpr = p.parse(rbp)  # type: ignore[assignment]
    if kind == "COMPOSE":
        return make_poly_compose(left, right)
    if kind == "PAIR":
        return make_prod(left, right)
    if kind == "CASE":
        return make_sum(left, right)
    raise ParseError(f"unknown functor operator {kind!r}")


def make_functor_grammar() -> tuple[Any, Any, dict]:
    return _functor_nud, _functor_led, functor_bp()


# ---------------------------------------------------------------------------
# Focus grammar
# ---------------------------------------------------------------------------

def _focus_nud(p: PrattParser, tok: Token) -> FocusExpr:
    kind, val = tok
    if kind == "LPAREN":
        inner = p.parse(0)
        p.expect("RPAREN", "closing )")
        return inner  # type: ignore[return-value]
    if kind == "NAME":
        return FocusRef(str(val))
    raise ParseError(f"unexpected token {kind!r} ({val!r}) in focus expression")


def _focus_led(p: PrattParser, left: FocusExpr, tok: Token, rbp: int) -> FocusExpr:
    kind = tok[0]
    right: FocusExpr = p.parse(rbp)  # type: ignore[assignment]
    if kind == "COMPOSE":
        return FocusCompose(left, right)
    raise ParseError(f"unknown focus operator {kind!r}")


def make_focus_grammar() -> tuple[Any, Any, dict]:
    return _focus_nud, _focus_led, {"COMPOSE": (50, 51)}
