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

from unialg.syntax.expressions import MorphismExpr, PolyExpr, First, Second, Identity
from unialg.syntax._lex import tokenize, tokenize_morphism, tokenize_functor, Token
from unialg.syntax._morphism_grammar import make_morphism_grammar, Env as MEnv
from unialg.syntax._functor_grammar import make_functor_grammar, Env as FEnv
from unialg.syntax._pratt import TokenCursor, PrattParser, ParseError  # re-export
from unialg.syntax._ops import _PU, _U, make_compose

__all__ = ["parse_program", "parse_morphism", "parse_functor", "Program", "ParseError"]

# Callback signature: given a backend name returns alias → MorphismExpr bindings.
LoadHandler = Callable[[str], dict[str, MorphismExpr]]


def _param_navigate(index: int, total: int) -> MorphismExpr:
    """Morphism from the combined Para parameter P to the i-th named parameter.

    P is left-nested: ((P_0 × P_1) × P_2) × …
    For total=1: P itself is P_0, so return Identity.
    For total>1: P_{total-1} = snd(P); P_i for i<total-1 = navigate deeper via fst.
    """
    if total == 1:
        return Identity(_U)
    if index == total - 1:
        return Second(_PU)
    return make_compose(First(_PU), _param_navigate(index, total - 1))


def _param_projections(params: tuple[str, ...]) -> dict[str, MorphismExpr]:
    """Return name → projection-from-(P×X) for each named Para parameter."""
    n = len(params)
    out: dict[str, MorphismExpr] = {}
    for i, name in enumerate(params):
        nav = _param_navigate(i, n)
        if isinstance(nav, Identity):
            out[name] = First(_PU)
        else:
            out[name] = make_compose(First(_PU), nav)
    return out

# Keywords that mark a new top-level declaration — used as RHS slice boundaries.
_DECL_KINDS = {"ROUTE", "MAP", "LOAD", "EOF"}


@dataclass(frozen=True)
class Program:
    """Parsed top-level source with loads, routes, and named functors."""

    loads:            tuple[str, ...]              = ()
    morphisms:        dict[str, MorphismExpr]      = field(default_factory=dict)
    functors:         dict[str, PolyExpr]          = field(default_factory=dict)
    morphism_params:  dict[str, tuple[str, ...]]   = field(default_factory=dict)


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
    morphisms:       dict[str, MorphismExpr]    = {}
    functors:        dict[str, PolyExpr]        = {}
    morphism_params: dict[str, tuple[str, ...]] = {}
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

        # Optional parameter list: route f(W) = … or route f(W, B) = …
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

        # Slice RHS tokens up to the next declaration keyword or EOF.
        start = cursor.pos
        end = start
        while end < len(tokens) and tokens[end][0] not in _DECL_KINDS:
            end += 1
        rhs_tokens = tokens[start:end] + [("EOF", None)]
        cursor.seek(end)

        if kw_tok[0] == "ROUTE":
            route_menv = dict(menv)
            if params:
                route_menv.update(_param_projections(params))
                morphism_params[name] = params
            nud, led, bp = make_morphism_grammar(route_menv, fenv)
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

    return Program(
        loads=tuple(loads),
        morphisms=morphisms,
        functors=functors,
        morphism_params=morphism_params,
    )


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
