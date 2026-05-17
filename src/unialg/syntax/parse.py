"""Public parser API for unialg surface syntax.

    parse_morphism(src) -> MorphismExpr
    parse_functor(src)  -> PolyExpr
    parse_program(src)  -> Program
"""
from __future__ import annotations

from dataclasses import dataclass, field

from unialg.syntax.expressions import (
    CarrierDecl, FocusDecl, FocusCompose, FocusRef, MorphismExpr, PolyCompose,
    PolyExpr, PolyRef,
)
from unialg.syntax._lex import tokenize_morphism, tokenize_functor, tokenize
from unialg.syntax._pratt import PrattParser, ParseError, TokenCursor
from unialg.syntax._grammar import make_morphism_grammar, make_functor_grammar


__all__ = ["parse_morphism", "parse_functor", "parse_program", "Program", "ParseError"]


@dataclass
class Program:
    """Parsed program: named morphisms and structural shape declarations."""
    loads: tuple[str, ...] = ()
    morphisms: dict[str, MorphismExpr] = field(default_factory=dict)
    functors: dict[str, PolyExpr] = field(default_factory=dict)
    carriers: dict[str, CarrierDecl] = field(default_factory=dict)
    focuses: dict[str, FocusDecl] = field(default_factory=dict)
    morphism_params: dict[str, tuple[str, ...]] = field(default_factory=dict)


_DECL_KINDS = frozenset({"LET", "SHAPE", "LOAD", "EOF"})


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
    """Parse a sequence of ``load``, ``let``, and ``shape`` declarations."""
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

        if kw_tok[0] not in ("LET", "SHAPE"):
            raise ParseError(
                "program: expected 'load', 'let', or 'shape', "
                f"got {kw_tok[0]!r} ({kw_tok[1]!r})"
            )
        name = str(cursor.expect("NAME", "definition name")[1])

        # Optional parameter list: let f(x, y) = ... / shape F(a) = ...
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

        if kw_tok[0] == "SHAPE" and cursor.peek()[0] == "COLON":
            cursor.advance()
            prog.focuses[name] = _parse_explicit_optic_shape(cursor, name, bp_m, nud_m, led_m)
            continue

        cursor.expect("EQ", "'='")

        # Slice RHS tokens up to next declaration or EOF
        start = cursor.pos
        end = start
        while end < len(cursor._tokens):
            tok = cursor._tokens[end]
            if tok[0] in _DECL_KINDS:
                break
            end += 1
        rhs_tokens = cursor._tokens[start:end] + [("EOF", None)]
        cursor.seek(end)

        if kw_tok[0] == "LET":
            if params:
                prog.morphism_params[name] = params
            p = PrattParser(rhs_tokens, label=f"let {name}", binding_powers=bp_m, nud=nud_m, led=led_m)
            prog.morphisms[name] = p.parse_all()  # type: ignore[assignment]
        elif kw_tok[0] == "SHAPE":
            if params:
                raise ParseError("shape parameters are reserved but not implemented yet")
            if rhs_tokens[0][0] == "FIX":
                carrier_cursor = TokenCursor(rhs_tokens, label=f"shape {name}")
                carrier_cursor.expect("FIX", "'fix'")
                functor_tokens = carrier_cursor._tokens[carrier_cursor.pos:]
                p = PrattParser(
                    functor_tokens,
                    label=f"shape {name} carrier functor",
                    binding_powers=bp_f,
                    nud=nud_f,
                    led=led_f,
                )
                functor = p.parse_all()
                assert isinstance(functor, PolyExpr)
                prog.carriers[name] = CarrierDecl(functor=functor)
                continue

            p = PrattParser(rhs_tokens, label=f"shape {name}", binding_powers=bp_f, nud=nud_f, led=led_f)
            prog.functors[name] = p.parse_all()  # type: ignore[assignment]

    prog.loads = tuple(loads)
    return prog


def _parse_explicit_optic_shape(cursor: TokenCursor, name: str, bp_m, nud_m, led_m) -> FocusDecl:
    """Parse ``shape name : F <-> T by forward / backward``.

    The current semantic optic constructor still needs a polynomial functor.
    Until full type-to-functor inference exists, the first name on the left
    side of ``<->`` is treated as that functor name. The rest of the boundary
    annotation is accepted as syntax but not interpreted semantically yet.
    """
    source_start = cursor.pos
    while cursor.peek()[0] not in ("BIDIR", "EOF"):
        cursor.advance()
    source_tokens = cursor._tokens[source_start:cursor.pos]
    cursor.expect("BIDIR", "'<->'")
    while cursor.peek()[0] not in ("BY", "EOF"):
        cursor.advance()
    if cursor.peek()[0] == "EOF":
        raise ParseError(f"shape {name}: optic declaration requires 'by <forward> / <backward>'")
    cursor.expect("BY", "'by'")

    source_names = [str(tok[1]) for tok in source_tokens if tok[0] == "NAME"]
    if not source_names:
        raise ParseError(
            f"shape {name}: explicit optic source must include a functor name"
        )
    functor = source_names[0]

    f_start = cursor.pos
    while cursor.peek()[0] not in ("SLASH", "EOF"):
        cursor.advance()
    forward_tokens = cursor._tokens[f_start:cursor.pos] + [("EOF", None)]
    cursor.expect("SLASH", "'/'")

    b_start = cursor.pos
    while cursor.peek()[0] not in _DECL_KINDS:
        cursor.advance()
    backward_tokens = cursor._tokens[b_start:cursor.pos] + [("EOF", None)]

    forward = PrattParser(
        forward_tokens,
        label=f"shape {name}.forward",
        binding_powers=bp_m,
        nud=nud_m,
        led=led_m,
    ).parse_all()
    backward = PrattParser(
        backward_tokens,
        label=f"shape {name}.backward",
        binding_powers=bp_m,
        nud=nud_m,
        led=led_m,
    ).parse_all()

    return FocusDecl(
        functor=functor,
        forward=forward,  # type: ignore[arg-type]
        backward=backward,  # type: ignore[arg-type]
    )


def poly_to_focus_expr(node: PolyExpr):
    """Translate a name/composition-only shape expression into a focus expression."""
    match node:
        case PolyRef(name=name):
            return FocusRef(name)
        case PolyCompose(left=left, right=right):
            return FocusCompose(poly_to_focus_expr(left), poly_to_focus_expr(right))
        case _:
            raise ParseError(f"shape expression cannot be used as an optic alias: {type(node).__name__}")
