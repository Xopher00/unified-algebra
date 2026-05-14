"""Public parser API for the unialg surface syntax.

  parse_program(src, ...)         -> Program   parse a sequence of definitions
  parse_morphism(src, env=None)   -> MorphismExpr
  parse_functor(src, env=None)    -> PolyExpr

Top-level definition syntax:
  load  BACKEND                         load backend primitives into env
  route NAME = <morphism-expr>
  map   NAME = <functor-expr>

env maps previously-defined names to their expression nodes so the
parser resolves references inline. Unresolved names become Ref / PolyRef
placeholder nodes.

dom/cod fields on parsed ContextualBinary nodes are TypeUnit() placeholders.
Semantics (morphisms.py) owns type derivation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from unialg.syntax.expressions import MorphismExpr, PolyExpr
from unialg.syntax._lex import tokenize, tokenize_morphism, tokenize_functor, Token
from unialg.syntax._morphism_grammar import make_morphism_grammar, Env as MEnv
from unialg.syntax._functor_grammar import make_functor_grammar, Env as FEnv
from unialg.syntax._pratt import TokenCursor, PrattParser, ParseError  # re-export

__all__ = ["parse_program", "parse_morphism", "parse_functor", "Program", "ParseError"]

# Callback signature: given a backend name returns alias → MorphismExpr bindings.
LoadHandler = Callable[[str], dict[str, MorphismExpr]]

# Keywords that mark a new top-level declaration — used as RHS slice boundaries.
_DECL_KINDS = {"ROUTE", "MAP", "LOAD", "EOF"}


@dataclass(frozen=True)
class Program:
    """Parsed top-level source with loads, routes, and named functors."""

    loads:     tuple[str, ...]         = ()
    morphisms: dict[str, MorphismExpr] = field(default_factory=dict)
    functors:  dict[str, PolyExpr]     = field(default_factory=dict)


def parse_program(
    src: str,
    morph_env: MEnv | None = None,
    functor_env: FEnv | None = None,
    *,
    load_handler: LoadHandler | None = None,
) -> Program:
    """Parse a sequence of `load`/`route`/`map` definitions into a Program."""
    tokens: list[Token] = tokenize(src)
    cursor = TokenCursor(tokens, label="program")

    menv: MEnv = dict(morph_env or {})
    fenv: FEnv = dict(functor_env or {})
    morphisms: dict[str, MorphismExpr] = {}
    functors:  dict[str, PolyExpr]     = {}
    loads: list[str] = []

    while cursor.peek()[0] != "EOF":
        kw_tok = cursor.advance()

        if kw_tok[0] == "LOAD":
            backend = str(cursor.expect("NAME", "backend name")[1])
            loads.append(backend)
            if load_handler is not None:
                menv.update(load_handler(backend))
            continue

        if kw_tok[0] not in ("ROUTE", "MAP"):
            raise ParseError(
                f"program: expected 'load', 'route', or 'map', got {kw_tok[0]!r} ({kw_tok[1]!r})"
            )
        name = str(cursor.expect("NAME", "definition name")[1])
        cursor.expect("EQ", "'='")

        # Slice RHS tokens up to the next declaration keyword or EOF.
        start = cursor.pos
        end = start
        while end < len(tokens) and tokens[end][0] not in _DECL_KINDS:
            end += 1
        rhs_tokens = tokens[start:end] + [("EOF", None)]
        cursor.seek(end)

        if kw_tok[0] == "ROUTE":
            nud, led, bp = make_morphism_grammar(menv, fenv)
            p = PrattParser(rhs_tokens, label=f"route {name}", binding_powers=bp, nud=nud, led=led)
            expr: MorphismExpr = p.parse_all()  # type: ignore[assignment]
            morphisms[name] = expr
            menv[name] = expr
        elif kw_tok[0] == "MAP":
            nud, led, bp = make_functor_grammar(fenv)
            p = PrattParser(rhs_tokens, label=f"map {name}", binding_powers=bp, nud=nud, led=led)
            fexpr: PolyExpr = p.parse_all()  # type: ignore[assignment]
            functors[name] = fexpr
            fenv[name] = fexpr

    return Program(loads=tuple(loads), morphisms=morphisms, functors=functors)


def parse_morphism(
    src: str,
    env: MEnv | None = None,
    functor_env: FEnv | None = None,
) -> MorphismExpr:
    """Parse one morphism expression with optional route and functor environments."""
    tokens = tokenize_morphism(src)
    nud, led, bp = make_morphism_grammar(env, functor_env)
    p = PrattParser(tokens, label="morphism", binding_powers=bp, nud=nud, led=led)
    return p.parse_all()  # type: ignore[return-value]


def parse_functor(src: str, env: FEnv | None = None) -> PolyExpr:
    """Parse one polynomial functor expression with an optional functor environment."""
    tokens = tokenize_functor(src)
    nud, led, bp = make_functor_grammar(env)
    p = PrattParser(tokens, label="functor", binding_powers=bp, nud=nud, led=led)
    return p.parse_all()  # type: ignore[return-value]
