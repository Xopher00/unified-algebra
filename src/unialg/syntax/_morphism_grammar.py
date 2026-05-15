"""Pratt grammar for morphism expressions.

Operator binding powers (left-assoc: lbp=p, rbp=p+1):

  &   pair(f,g)      BP 70  — same input forks into both branches
  ||  par(f,g)       BP 65  — separate parallel lanes
  >>  compose(f,g)   BP 60  — sequential composition
  |   case(f,g)      BP 50  — coproduct branch

dom/cod fields on ContextualBinary nodes are TypeUnit() placeholders.
The parser builds structurally correct expression trees; the semantics
layer (morphisms.py, functors.py) owns type derivation.

Imports: objects, syntax/expressions, syntax/_pratt only. No semantics imports.
"""
from __future__ import annotations
from typing import Any

from unialg.syntax.expressions import (
    MorphismExpr, PolyExpr, PolyFmap, MorphismApp,
    Identity, Copy, Delete, First, Second, Left, Right,
    Absurd, Assoc, Symmetry, Ref, PolyRef,
    Id, List as PolyList,
)
from unialg.syntax._pratt import PrattParser, ParseError
from unialg.syntax._ops import _U, _PU, _SU, morphism_bp, make_binary, make_compose, make_pair

Token = tuple[str, Any]
Env = dict[str, MorphismExpr]
FunctorEnv = dict[str, PolyExpr]
MorphismParams = dict[str, tuple[str, ...]]


def _poly_prefix(p: PrattParser, base: PolyExpr) -> PolyExpr:
    """Consume trailing STAR tokens and wrap base in List(...)."""
    body = base
    while p.peek()[0] == "STAR":
        p.advance()
        body = PolyList(body)
    return body


def _copy_power(count: int) -> MorphismExpr:
    """Expand ``*n`` to a left-nested copy tree."""
    if count < 2:
        raise ParseError("copy power expects an integer >= 2")
    out: MorphismExpr = Copy(_U)
    for _ in range(2, count):
        out = make_pair(out, Identity(_U))
    return out


def _case_injection(index: int) -> MorphismExpr:
    """Expand ``|0``/``|1`` to the corresponding sum injection."""
    if index == 0:
        return Left(_SU)
    if index == 1:
        return Right(_SU)
    raise ParseError("case injection index must be 0 or 1")


def _parse_arg_list(p: PrattParser) -> tuple[MorphismExpr, ...]:
    """Parse a parenthesized, comma-separated morphism argument list."""
    p.expect("LPAREN", "(")
    args: list[MorphismExpr] = [p.parse(0)]  # type: ignore[assignment]
    while p.peek()[0] == "COMMA":
        p.advance()
        args.append(p.parse(0))  # type: ignore[arg-type]
    p.expect("RPAREN", "closing )")
    return tuple(args)


def _nud(env: Env, fenv: FunctorEnv, mparams: MorphismParams,
         lexical_params: frozenset[str], p: PrattParser, tok: Token) -> MorphismExpr:
    """Parse a morphism atom, prefix form, grouping, or functor action head."""
    kind, val = tok

    if kind == "LPAREN":
        inner = p.parse(0)
        p.expect("RPAREN", "closing )")
        return inner  # type: ignore[return-value]

    if kind == "BANG":
        return Delete(_U)

    if kind == "STAR":
        if p.peek()[0] != "INT":
            raise ParseError("copy power expects an integer >= 2")
        _, count = p.expect("INT", "copy power")
        return _copy_power(count)

    if kind == "CASE":
        # Prefix |0 / |1 — sum injection.
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

        if name in lexical_params:
            if p.peek()[0] == "LPAREN":
                args = _parse_arg_list(p)
                return MorphismApp(Ref(name), args, ())
            return Ref(name)

        if name == "x":
            body = _poly_prefix(p, Id())
            if p.peek()[0] == "LBRACE":
                p.advance()
                f = p.parse(0)
                p.expect("RBRACE", "closing }")
                return PolyFmap(body=body, f=f,  # type: ignore[arg-type]
                                param=_U, monad=None, dom=_U, cod=_U)
            if isinstance(body, Id):
                return Identity(_U)
            raise ParseError("functor expression x* without {f}")

        if name in ("delete", "drop", "del"):
            if p.peek()[0] == "LPAREN":
                p.advance()
                p.expect("INT", "del count")
                p.expect("RPAREN", "closing )")
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
        if name == "fst":
            return First(_PU)
        if name == "snd":
            return Second(_PU)
        if name == "inl":
            return Left(_SU)
        if name == "inr":
            return Right(_SU)
        if name == "absurd":
            return Absurd(_U)
        if name == "assoc":
            return Assoc(_U, _U)
        if name in ("sym", "symmetry"):
            return Symmetry(_U, _U)

        # E[name] or E[name]{f} — einsum functor reference or map
        if name == "E":
            p.expect("LBRACKET", "[")
            _, aname = p.expect("NAME", "einsum name")
            p.expect("RBRACKET", "]")
            if p.peek()[0] == "LBRACE":
                p.advance()
                f = p.parse(0)
                p.expect("RBRACE", "closing }")
                body = fenv.get(aname, PolyRef(aname))
                return PolyFmap(body=body, f=f,
                                param=_U, monad=None, dom=_U, cod=_U)
            if aname in env:
                return env[aname]
            return Ref(aname)

        # name{f} — named functor applied to morphism
        if p.peek()[0] == "LBRACE":
            p.advance()
            f = p.parse(0)
            p.expect("RBRACE", "closing }")
            return PolyFmap(body=fenv.get(name, PolyRef(name)), f=f,
                            param=_U, monad=None, dom=_U, cod=_U)

        # Body-position application: name(arg1, arg2, ...)
        if name in env and p.peek()[0] == "LPAREN":
            resolved = env[name]
            args = _parse_arg_list(p)
            params = mparams.get(name, ())
            if params and len(args) != len(params):
                raise ParseError(
                    f"{name!r} expects {len(params)} arguments, got {len(args)}"
                )
            return MorphismApp(resolved, args, params)

        # Generic name → env lookup or unresolved Ref
        if name in env:
            return env[name]
        return Ref(name)

    raise ParseError(f"unexpected token {kind!r} ({val!r}) in morphism expression")


def _led(p: PrattParser, left: MorphismExpr, tok: Token, rbp: int) -> MorphismExpr:
    """Parse a postfix or infix morphism operator after ``left``."""
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
        # Postfix |0 / |1 — inject then compose; otherwise infix case elimination.
        if p.peek()[0] == "INT":
            _, index = p.expect("INT", "injection index")
            return make_compose(left, _case_injection(index))
        right: MorphismExpr = p.parse(rbp)  # type: ignore[assignment]
        return make_binary("CASE", left, right)
    right = p.parse(rbp)  # type: ignore[assignment]
    return make_binary(kind, left, right)


def make_morphism_grammar(
    env: Env | None = None,
    functor_env: FunctorEnv | None = None,
    morphism_params: MorphismParams | None = None,
    lexical_params: frozenset[str] = frozenset(),
) -> tuple[Any, Any, dict]:
    """Return Pratt callbacks and binding powers for morphism parsing."""
    _env: Env = env or {}
    _fenv: FunctorEnv = functor_env or {}
    _mparams: MorphismParams = morphism_params or {}
    _lexical_params: frozenset[str] = lexical_params

    def nud(p: PrattParser, tok: Token) -> MorphismExpr:
        """Parse a null-denotation morphism token using captured environments."""
        return _nud(_env, _fenv, _mparams, _lexical_params, p, tok)

    def led(p: PrattParser, left: object, tok: Token, rbp: int) -> MorphismExpr:
        """Parse a left-denotation morphism token using captured environments."""
        return _led(p, left, tok, rbp)  # type: ignore[arg-type]

    return nud, led, morphism_bp()
