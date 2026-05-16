"""Public parser API for unialg surface syntax.

    parse_morphism(src) -> MorphismExpr
    parse_functor(src)  -> PolyExpr
    parse_program(src)  -> Program
"""
from __future__ import annotations

from dataclasses import dataclass, field

from unialg.syntax.expressions import MorphismExpr, PolyExpr
from unialg.syntax._lex import tokenize_morphism, tokenize_functor, tokenize
from unialg.syntax._pratt import PrattParser, ParseError, TokenCursor
from unialg.syntax._grammar import make_morphism_grammar, make_functor_grammar


__all__ = ["parse_morphism", "parse_functor", "parse_program", "Program", "ParseError"]


@dataclass
class Program:
    """Parsed program: named routes and functor definitions."""
    loads: tuple[str, ...] = ()
    routes: dict[str, MorphismExpr] = field(default_factory=dict)
    functors: dict[str, PolyExpr] = field(default_factory=dict)
    route_params: dict[str, tuple[str, ...]] = field(default_factory=dict)


_DECL_KINDS = frozenset({"ROUTE", "MAP", "LOAD", "EOF"})


def parse_morphism(src: str) -> MorphismExpr:
    """Parse one morphism expression."""
    tokens = tokenize_morphism(src)
    nud, led, bp = make_morphism_grammar()
    p = PrattParser(tokens, label="morphism", binding_powers=bp, nud=nud, led=led)
    return p.parse_all()  # type: ignore[return-value]


def parse_functor(src: str) -> PolyExpr:
    """Parse one polynomial functor expression."""
    tokens = tokenize_functor(src)
    nud, led, bp = make_functor_grammar()
    p = PrattParser(tokens, label="functor", binding_powers=bp, nud=nud, led=led)
    return p.parse_all()  # type: ignore[return-value]


def parse_program(src: str) -> Program:
    """Parse a sequence of route/map definitions."""
    tokens = tokenize(src)
    cursor = TokenCursor(tokens, label="program")

    prog = Program()
    nud_m, led_m, bp_m = make_morphism_grammar()
    nud_f, led_f, bp_f = make_functor_grammar()

    loads: list[str] = []

    while cursor.peek()[0] != "EOF":
        kw_tok = cursor.advance()

        if kw_tok[0] == "LOAD":
            loads.append(str(cursor.expect("NAME", "backend name")[1]))
            continue

        if kw_tok[0] not in ("ROUTE", "MAP"):
            raise ParseError(
                f"program: expected 'load', 'route', or 'map', got {kw_tok[0]!r} ({kw_tok[1]!r})"
            )
        name = str(cursor.expect("NAME", "definition name")[1])

        # Optional parameter list: route f(x, y) = ...
        params: tuple[str, ...] = ()
        if cursor.peek()[0] == "LPAREN":
            cursor.advance()
            param_names: list[str] = []
            param_names.append(str(cursor.expect("NAME", "parameter name")[1]))
            while cursor.peek()[0] == "COMMA":
                cursor.advance()
                param_names.append(str(cursor.expect("NAME", "parameter name")[1]))
            cursor.expect("RPAREN", "closing )")
            params = tuple(param_names)

        cursor.expect("EQ", "'='")

        # Slice RHS tokens up to next declaration or EOF
        start = cursor.pos
        end = start
        while end < len(cursor._tokens) and cursor._tokens[end][0] not in _DECL_KINDS:
            end += 1
        rhs_tokens = cursor._tokens[start:end] + [("EOF", None)]
        cursor.seek(end)

        if kw_tok[0] == "ROUTE":
            if params:
                prog.route_params[name] = params
            p = PrattParser(rhs_tokens, label=f"route {name}", binding_powers=bp_m, nud=nud_m, led=led_m)
            prog.routes[name] = p.parse_all()  # type: ignore[assignment]
        elif kw_tok[0] == "MAP":
            p = PrattParser(rhs_tokens, label=f"map {name}", binding_powers=bp_f, nud=nud_f, led=led_f)
            prog.functors[name] = p.parse_all()  # type: ignore[assignment]

    prog.loads = tuple(loads)
    return prog
