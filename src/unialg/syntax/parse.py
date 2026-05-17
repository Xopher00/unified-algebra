"""Public parser API for unialg surface syntax.

    parse_morphism(src) -> MorphismExpr
    parse_functor(src)  -> PolyExpr
    parse_program(src)  -> Program
"""
from __future__ import annotations

from dataclasses import dataclass, field

from unialg.syntax.expressions import CarrierDecl, FocusDecl, MorphismExpr, PolyExpr
from unialg.syntax._lex import tokenize_morphism, tokenize_functor, tokenize
from unialg.syntax._pratt import PrattParser, ParseError, TokenCursor
from unialg.syntax._grammar import make_focus_grammar, make_morphism_grammar, make_functor_grammar


__all__ = ["parse_morphism", "parse_functor", "parse_program", "Program", "ParseError"]


@dataclass
class Program:
    """Parsed program: named routes, functors, carriers, and focus declarations."""
    loads: tuple[str, ...] = ()
    routes: dict[str, MorphismExpr] = field(default_factory=dict)
    functors: dict[str, PolyExpr] = field(default_factory=dict)
    carriers: dict[str, CarrierDecl] = field(default_factory=dict)
    focuses: dict[str, FocusDecl] = field(default_factory=dict)
    route_params: dict[str, tuple[str, ...]] = field(default_factory=dict)


_DECL_KINDS = frozenset({"ROUTE", "MAP", "LOAD", "FOCUS", "CARRIER", "EOF"})
_FOCUS_FIELDS = frozenset({"carrier", "functor", "forward", "backward"})


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
    """Parse a sequence of load, route, map, carrier, and focus declarations."""
    tokens = tokenize(src)
    cursor = TokenCursor(tokens, label="program")

    prog = Program()
    nud_m, led_m, bp_m = make_morphism_grammar()
    nud_f, led_f, bp_f = make_functor_grammar()
    nud_focus, led_focus, bp_focus = make_focus_grammar()

    loads: list[str] = []

    while cursor.peek()[0] != "EOF":
        kw_tok = cursor.advance()

        if kw_tok[0] == "LOAD":
            loads.append(str(cursor.expect("NAME", "backend name")[1]))
            continue

        if kw_tok[0] not in ("ROUTE", "MAP", "FOCUS", "CARRIER"):
            raise ParseError(
                "program: expected 'load', 'route', 'map', 'carrier', or "
                f"'focus', got {kw_tok[0]!r} ({kw_tok[1]!r})"
            )
        name = str(cursor.expect("NAME", "definition name")[1])

        # Optional parameter list: route f(x, y) = ...
        params: tuple[str, ...] = ()
        if kw_tok[0] == "ROUTE" and cursor.peek()[0] == "LPAREN":
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
        while end < len(cursor._tokens):
            tok = cursor._tokens[end]
            nxt = cursor._tokens[end + 1] if end + 1 < len(cursor._tokens) else ("EOF", None)
            if tok[0] in _DECL_KINDS and not (
                kw_tok[0] == "FOCUS" and tok[0] == "CARRIER" and nxt[0] == "EQ"
            ):
                break
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
        elif kw_tok[0] == "CARRIER":
            carrier_cursor = TokenCursor(rhs_tokens, label=f"carrier {name}")
            marker = carrier_cursor.expect("NAME", "'fix'")
            if marker[1] != "fix":
                raise ParseError(f"carrier {name}: expected 'fix', got {marker[1]!r}")
            prog.carriers[name] = CarrierDecl(
                functor=str(carrier_cursor.expect("NAME", "functor name")[1])
            )
            carrier_cursor.expect("EOF", "end of carrier declaration")
        elif kw_tok[0] == "FOCUS":
            if not _looks_like_focus_fields(rhs_tokens):
                p = PrattParser(
                    rhs_tokens,
                    label=f"focus {name}",
                    binding_powers=bp_focus,
                    nud=nud_focus,
                    led=led_focus,
                )
                prog.focuses[name] = FocusDecl(expr=p.parse_all())  # type: ignore[arg-type]
                continue

            fields: dict[str, object] = {}
            field_cursor = TokenCursor(rhs_tokens, label=f"focus {name}")
            while field_cursor.peek()[0] != "EOF":
                key_tok = field_cursor.advance()
                if key_tok[0] == "CARRIER":
                    key = "carrier"
                elif key_tok[0] == "NAME":
                    key = str(key_tok[1])
                else:
                    raise ParseError(
                        f"focus {name}: expected focus field name, got {key_tok[0]!r}"
                    )
                if key not in _FOCUS_FIELDS:
                    raise ParseError(f"focus {name}: unknown field {key!r}")
                field_cursor.expect("EQ", "'='")
                if key in ("carrier", "functor"):
                    fields[key] = str(field_cursor.expect("NAME", f"{key} name")[1])
                    continue

                start = field_cursor.pos
                end = start
                while end < len(field_cursor._tokens):
                    tok = field_cursor._tokens[end]
                    nxt = field_cursor._tokens[end + 1] if end + 1 < len(field_cursor._tokens) else ("EOF", None)
                    if tok[0] == "EOF" or (
                        tok[0] == "NAME" and tok[1] in _FOCUS_FIELDS and nxt[0] == "EQ"
                    ) or (
                        tok[0] == "CARRIER" and nxt[0] == "EQ"
                    ):
                        break
                    end += 1
                value_tokens = field_cursor._tokens[start:end] + [("EOF", None)]
                field_cursor.seek(end)
                p = PrattParser(
                    value_tokens,
                    label=f"focus {name}.{key}",
                    binding_powers=bp_m,
                    nud=nud_m,
                    led=led_m,
                )
                fields[key] = p.parse_all()

            has_carrier = "carrier" in fields
            manual_fields = {"functor", "forward", "backward"}
            has_manual = bool(manual_fields & fields.keys())
            if has_carrier and has_manual:
                raise ParseError(f"focus {name}: carrier focus cannot also define functor/forward/backward")
            if not has_carrier:
                missing = manual_fields - fields.keys()
                if missing:
                    raise ParseError(f"focus {name}: missing field(s): {', '.join(sorted(missing))}")
            prog.focuses[name] = FocusDecl(
                carrier=fields.get("carrier"),  # type: ignore[arg-type]
                functor=fields.get("functor"),  # type: ignore[arg-type]
                forward=fields.get("forward"),  # type: ignore[arg-type]
                backward=fields.get("backward"),  # type: ignore[arg-type]
            )

    prog.loads = tuple(loads)
    return prog


def _looks_like_focus_fields(tokens: list[tuple[str, object]]) -> bool:
    """Return True when a focus RHS starts with ``field = value`` syntax."""
    if len(tokens) < 3:
        return False
    first, second = tokens[0], tokens[1]
    if second[0] != "EQ":
        return False
    if first[0] == "CARRIER":
        return True
    return first[0] == "NAME" and first[1] in _FOCUS_FIELDS
